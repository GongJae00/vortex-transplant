# Implementation Specification — Per-File Migration Plan

**Principle**: V1 source files are immutable. V2 is a parallel implementation, not a V1 modification.

**V2 Package Location**: `topological/v2/` (separate namespace, no V1 file modification)

---

## File Migration Matrix

### 1. `topological/model.py` → `topological/v2/model.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `ModelSpec` | Unchanged | Reuse; no subclass needed | `ModelSpec(channels=1)` for C=1 gateway; `ModelSpec(channels=8)` for multichannel |
| `U1ConvRNN` | Unchanged | Reuse | Same interface |
| `PlainConvRNN` | Unchanged | Reuse | Same interface |
| — | No C=1 gateway model | Add | `make_model(spec=ModelSpec(channels=1))` — no separate class; construction path is identical |
| — | No factorial baseline variants | Add | `make_model(layout='U1CommutingLinear_RadialNonlinear')` etc. — implement as model factory switch |

**C=1 design**: `U1ConvRNN(ModelSpec(channels=1))` is a fully valid model. The recurrent linear map `recurrent_linear` computes complex convolution on a single-channel (2-real-component) field; `radial_tanh` is unchanged. A distinct `ScalarU1ConvRNN` class is not justified — the architecture has zero structural differences between C=1 and C>1. The constructor path, parameter shapes, forward logic, and `radial_tanh` are identical modulo channel count. No subclass, no separate module, no `if channels == 1` branch in the model class itself.

**Factorial baseline naming (2×2 design)**:

| Label | Linear Map | Nonlinearity | Role |
|-------|------------|-------------|------|
| `U1CommutingLinear_RadialNonlinear` | Complex convolution, cross-coupled real/imag | `radial_tanh` | Full U(1)-equivariant (primary model) |
| `U1CommutingLinear_ElementwiseNonlinear` | Complex convolution, cross-coupled real/imag | `tanh(real) + 1j * tanh(imag)` | Breaks equivariance at activation only |
| `UnrestrictedLinear_RadialNonlinear` | Real Conv2d, uncoupled weights | `radial_tanh` | Breaks equivariance at linear map only |
| `UnrestrictedLinear_ElementwiseNonlinear` | Real Conv2d, uncoupled weights | `tanh(real) + 1j * tanh(imag)` | Fully broken = PlainConvRNN |

Names `ComplexNoEquiv` and `RealWithEquiv` are retired. The factorial names are descriptive and self-documenting: each encodes both factors.

**Factory API**:
```python
def make_model(
    spec: ModelSpec,
    layout: str = "U1CommutingLinear_RadialNonlinear",
    *,
    generator: torch.Generator | None = None,
) -> nn.Module:
    """Create a model variant by factorial cell label.

    Valid layouts:
      - "U1CommutingLinear_RadialNonlinear"       → U1ConvRNN(spec)
      - "U1CommutingLinear_ElementwiseNonlinear"  → U1ConvRNN with elementwise activation
      - "UnrestrictedLinear_RadialNonlinear"      → RealLinearConvRNN with radial_tanh
      - "UnrestrictedLinear_ElementwiseNonlinear" → PlainConvRNN(spec)
    """
```

`UnrestrictedLinear_RadialNonlinear` requires a new thin module `_RealLinearConvRNN`
that replaces `recurrent_linear` with a standard `nn.Conv2d` but retains `radial_tanh`.
It exists purely to express the 2×2 combinatorial space — not as a independent
architectural innovation. Its implementation is: `Conv2d(2C, 2C, kernel_size=3,
padding_mode='circular')` followed by `radial_tanh`, analogous to `PlainConvRNN`
but with U(1)-equivariant activation.

---

### 2. `topological/task.py` → `topological/v2/task.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `CopyBatch`, `CopyTrace` | Unchanged | Reuse | Same |
| `generate_copy_batch` | Unchanged | Reuse | Add split namespacing |
| — | No explicit split namespace | Add | `split: Literal["dev", "cal", "confirm"]` parameter |
| `run_copy`, `write_copy`, `continue_copy` | Unchanged | Reuse | Same |

---

### 3. `topological/topology.py` → `topological/v2/topology.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `extract_charge` | No branch margin output | Extend | Add `branch_margins: BranchStability` to return value |
| `canonical_vortex_field` | Unchanged | Reuse | Same |
| — | No charge-flip radius measurement | Add | `charge_flip_radius(field, position, eps_grid)` |
| — | No signed-pair tracking | Add | `defect_tracking(field_t, field_t1)` — signed periodic optimal matching with births/deaths |

**Defect tracking specification**: The signed Jaccard index (V1) only captures
presence/absence overlap. V2 replaces it with signed periodic optimal matching:

```python
@dataclass(frozen=True)
class DefectEvent:
    position: tuple[int, int]       # (x, y) lattice coordinate
    charge: int                     # ±1
    channel: int
    event_type: str                 # "birth", "death", "persist", "flip"
    displacement: float | None      # L2 distance to matched partner (None for birth/death)

@dataclass(frozen=True)
class DefectTracking:
    matched_pairs: list[tuple[DefectEvent, DefectEvent]]
    births: list[DefectEvent]
    deaths: list[DefectEvent]
    total_births: int
    total_deaths: int
    total_persistent: int
    bounding_distance: float        # maximum match radius (toroidal L∞)
```

Matching algorithm: Hungarian assignment in toroidal L∞ metric on (x ± H, y ± W)
for each charge sign independently, bounded by `bounding_distance`. Unmatched
defects are classified as births (in destination) or deaths (in source). Charge
flips (± → ∓) are tracked as separate matched events with flipped sign.

**Branch margin**: Return `BranchStability` (typed frozen dataclass, not tuple):

```python
@dataclass(frozen=True)
class BranchStability:
    min_margin: float               # global minimum pi - |link| across all edges
    q01_margin: float               # 1% quantile
    q05_margin: float               # 5% quantile
    median_margin: float
    charge_flip_radius_median: float  # median epsilon to flip any charge
```

The `extract_charge` function is extended to compute per-edge margins `π − |Δθ_e|`
and aggregate them into a `BranchStability` record attached to `ChargeExtraction`.

---

### 4. `topological/decomposition.py` → `topological/v2/decomposition.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `decompose` | No branch stability validation | Extend | Add `branch_stability: BranchStability` to `CompactDecomposition` |
| `transplant_*` | Unchanged | Reuse | Same |
| — | No minimal-surgery transplant | Add | `transplant_vortex_minimal(recipient, target_charge_map)` |
| — | No harmonic sector swap | Add | `transplant_harmonic(donor_harmonic, recipient)` |
| — | No single-pair annihilation | Add | `annihilate_pair(field, pair_position)` |

**Minimal-surgery design**: See § "New Module: `topological/v2/surgery.py`" below.

---

### 5. `topological/interventions.py` → `topological/v2/interventions.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `HiddenComponents` | Unused `minimum_magnitude` field | Keep + use in gate | Add `branch_stability` field |
| `component_intervention` | Hardcoded to 6 arms | Extend | Add `vortex_minimal`, `harmonic`, `vortex_remove_pair`, `vortex_sham`, `vortex_sign_flip` |
| `matched_global_phase` | Unchanged | Reuse | Same |
| `matched_zero_charge_phase` | Single control index | Extend | Multiple control indices for bootstrap |
| — | No same-charge representative sampling | Add | `sample_representatives(field, n=10, seed)` |
| — | No natural neighbor search | Add | `find_natural_neighbor(field, pool, target_topology)` |
| — | No relaxation post-intervention | Add | `relax(model, field, steps=5)` |
| — | No manifold distance metric | Add | `manifold_distance(field, pca_model, kNN)` |

---

### 6. `topological/learned_evaluation.py` → `topological/v2/evaluation.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `StateTopology` | No density or branch margin | Extend | Add `defect_density`, `branch_margin`, `per_channel_prevalence` |
| `analyze_topology` | `nonzero_defect` too coarse | Redesign | Return `TopologyStatsV2` with density and stability |
| `select_donor_pair` | Selection collider | Keep for primary; add sensitivity | Add `random_donor_pair` for sensitivity |
| `evaluate_selected_pairs` | Nuisance pooled | Separate families | Per-family margin reporting |
| `_per_model_decision` | `defect_learned_not_innate` broken | Redesign gate | Use density + branch_margin |
| `decide_learned_pilot` | Cross-model comparison | Extend | Paired bootstrap CI |
| — | No manifold validity check | Add | `check_manifold_validity(intervened_field, manifold_model)` |

---

### 7. `topological/training.py` → `topological/v2/training.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `TrainingSpec` | Unchanged | Reuse | Same |
| `make_model` | No C=1 or factorial variants | Extend | Add `layout` parameter (see model factory above) |
| `train_seed` | Unchanged | Reuse | Same |
| — | No training convergence audit across seeds | Add | `aggregate_training_curves(artifact_dir)` |

---

### 8. `topological/learned_smoke.py` → `topological/v2/smoke.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `run_smoke` | Unchanged architecture | Extend | Add `branch_stability_smoke`, `representative_smoke` |
| `_exercise_device` | Shallow equivariance check | Extend | Separate blank/input/readout checks, phase grid, error reporting |

---

### 9. `topological/learned_pilot.py` → `topological/v2/pilot.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `run_canonical_pilot` | Unchanged architecture | Extend | Split-aware (cal/confirm), manifest V2 schema |
| — | No split gating | Add | `require_split(split_name)`, `assert_not_confirmatory()` |
| — | No training curve aggregation | Add | `write_aggregated_training_curves()` |

---

### 10. `topological/pilot.py` Stale Dependency

| Action |
|--------|
| Document `replay` subprocess call as feasibility-only infrastructure hotfix (see below) |
| V2 MUST NOT replicate the `aligned_mask_transplant.pm1_pilot` stale module import |

**Stale dependency**: The `replay` mechanism at `topological/pilot.py:131–137` calls
`sys.executable -m aligned_mask_transplant.pm1_pilot --replay-hash` as a subprocess.
This is a **non-scientific infrastructure hotfix** specific to V1 feasibility
pilot. It was introduced to verify clean-process replay exactness for the
`PM1-FEASIBILITY-V1` synthetic-field contract only. The scientific payload
computation does not depend on it; the subprocess validates that re-running the
same scientific computation in an independent interpreter instance produces
byte-identical results. This is a process isolation assertion, not a scientific
control. V2 replaces this with the proper subprocess replay described in
`12_REPRODUCIBILITY_AND_ARTIFACTS.md`.

---

## New Module Specifications

### `topological/v2/hodge.py` — Compact Hodge Decomposition

Full Hodge decomposition of a compact U(1) field on T² into four components:

```python
@dataclass(frozen=True)
class HodgeDecomposition:
    exact: np.ndarray           # curl-free (gradient of scalar potential)
    coexact: np.ndarray         # divergence-free (carries topological charge Q)
    harmonic: np.ndarray        # both curl-free and divergence-free (2D on T²)
    branch_integer: np.ndarray  # integer cochain (⌊Δθ/(2π)⌉, the Q-generating part)
    reconstruction_error: float
    charge: ChargeExtraction
    branch_stability: BranchStability

def hodge_decompose(field: np.ndarray) -> HodgeDecomposition:
    """Full Helmholtz-Hodge decomposition on T².

    Algorithm:
    1. Extract charge Q via extract_charge (curled link variables).
    2. coexact = canonical_vortex_field(Q)  (Poisson inversion of Q).
    3. Subtract coexact phase from unit-compact field → residual phase.
    4. exact = gradient of scalar potential φ solved via Poisson on residual curl-free part.
    5. harmonic = residual after removing exact (both curl-free and div-free on T²).
    6. branch_integer = floor(Δθ_original / 2π) * 2π — the integer-valued 1-cochain
       that captures branch choices without invoking a potential.
    7. Reconstruction error: max |field − recompose(exact, coexact, harmonic, branch_integer)|.

    Implementation: FFT-based Poisson solver (same spectral method as V1
    _poisson_stream) extended to extract all four components. The harmonic
    component uses the 2D subspace of harmonic forms (constant-vector holonomy
    modes). The branch_integer cochain uses the integer-rounded link phases
    before the wrapping step, capturing the Z-valued part of the link that the
    Poisson stream function recovers from its Laplacian.
    """
```

The Hodge decomposition is the mathematical foundation for:
- **Harmonic swap intervention** (C13): replace donor harmonic sector into recipient.
- **Charge-arrangement-shuffle null family**: permute co-exact charge positions
  while preserving exact + harmonic + branch-integer components.
- **Zero-charge representative sampling**: sample alternate smooth fields within
  the Q=0-everywhere topological sector (same harmonic, same branch-integer,
  different exact gauge).

### `topological/v2/representatives.py` — Same-Charge Representative Sampling

```python
@dataclass(frozen=True)
class RepresentativeSpec:
    n_representatives: int = 10
    seed_offset: int = 0
    method: str = "harmonic_random"  # or "displacement_minimal"

@dataclass(frozen=True)
class RepresentativeSample:
    fields: list[np.ndarray]        # n representative fields, all Q=0 everywhere
    harmonic_sectors: list[tuple[float, float]]  # (wx, wy) for each
    pairwise_distances: np.ndarray   # (n, n) L2 distance matrix
    coverage_fraction: float         # fraction of harmonic space covered
    method: str

def sample_representatives(
    field: np.ndarray,
    spec: RepresentativeSpec = RepresentativeSpec(),
) -> RepresentativeSample:
    """Generate same-charge-class representative fields.

    Algorithm (harmonic_random method):
    1. Decompose field into full Hodge components.
    2. Verify Q_p = 0 for ALL plaquettes (not merely sum(Q)=0).
    3. Generate n random harmonic sectors (wx, wy) uniform on T²:
       harmonic_phase(x, y) = wx * x / H + wy * y / W.
    4. For each harmonic sector, recompose: magnitude * exp(i * (exact_phase + harmonic_phase)).
    5. Verify each representative has Q=0 everywhere.
    6. Broadcast harmonic sectors uniformly to maximize coverage.

    Algorithm (displacement_minimal method):
    1. Start from original field.
    2. Apply small random spatial perturbations to the exact (smooth) phase component
       while preserving Q=0 constraint.
    3. Select n perturbations with minimum total displacement from original.
    """
```

### `topological/v2/surgery.py` — Minimal Topological Surgery Solver

```python
@dataclass(frozen=True)
class SurgeryTarget:
    target_charge_map: np.ndarray          # (H, W) integer charge map, sum=0
    preserve_harmonic: bool = True
    preserve_magnitude: bool = True
    max_iterations: int = 200
    convergence_tol: float = 1e-8

@dataclass(frozen=True)
class SurgeryResult:
    success: bool                          # surgery completed successfully
    converged: bool                        # optimization converged to target
    target_charge_exact: bool              # Q(final_field) == target_charge_map exactly
    post_relax_charge_exact: bool          # after relaxation, charge still matches
    harmonic_preserved: bool               # harmonic sector unchanged from recipient
    magnitude_error: float                 # max |final_magnitude - recipient_magnitude|
    energy_error: float                    # gradient_energy difference from recipient
    spectrum_error: float                  # radial_link_spectrum error from recipient
    manifold_distance: float               # manifold distance post-intervention
    iterations: int                        # optimization steps taken
    failure_reason: str | None             # None if success; otherwise descriptive

def transplant_vortex_minimal(
    recipient: np.ndarray,
    target_charge_map: np.ndarray,
    target: SurgeryTarget = SurgeryTarget(target_charge_map=np.zeros((16, 16), dtype=np.int64)),
) -> SurgeryResult:
    """Minimal-displacement same-charge vortex transplant.

    Given a recipient field (C, H, W) complex and a target charge map Q_target
    (H, W) integers with sum=0, find a field H' such that:
      - Q(H') = Q_target (exact charge match)
      - ||H' - recipient|| is minimized (minimum displacement)
      - harmonic sector of H' matches recipient (if preserve_harmonic=True)
      - magnitude of H' matches recipient (if preserve_magnitude=True)

    Algorithm (iterative constrained optimization):
    1. Hodge-decompose recipient into exact, coexact, harmonic, branch_integer.
    2. Compute target coexact component: canonical_vortex_field(Q_target).
    3. Set initial candidate: recompose(recipient_magnitude, target_coexact,
       recipient_exact, recipient_harmonic, receive_branch_integer).
    4. Iteratively adjust the exact (smooth) phase component:
       a. Compute gradient of ||candidate - recipient|| w.r.t. exact phase.
       b. Project gradient onto Q=0 subspace (via Poisson kernel).
       c. Line search to reduce displacement while maintaining Q constraint.
    5. Stop when displacement change < convergence_tol or max_iterations reached.
    6. Verify final charge map, harmonic sector, and magnitude.
    7. Return SurgeryResult with all diagnostic fields.

    This is more complex than the simple canonical vortex swap because it searches
    for the gauge-equivalent field (same charge class) closest to the original recipient.
    The canonical vortex field alone is not minimal — different exact-phase choices
    produce different displacements.
    """

def annihilate_pair(
    field: np.ndarray,
    pair_position: tuple[int, int],
) -> SurgeryResult:
    """Remove a single ± pair from the field.

    Locates the nearest +1/-1 charge pair to pair_position, sets both to 0
    in the target charge map, then runs transplant_vortex_minimal to find
    the minimum-displacement charge-free field.
    """
```

### `topological/v2/manifold.py` — Manifold Diagnostics

```python
@dataclass(frozen=True)
class ManifoldDiagnostics:
    reconstruction_error: float            # reconstruction error under trained PCA
    kNN_density_ratio: float               # density(intervened) / density(natural pool)
    nearest_natural_distance: float        # L2 distance to nearest natural hidden state
    relaxation_drift: float                # ||relax(field) − field|| after n blank steps
    on_manifold: bool                      # composite gate: all three < threshold
    pca_components: int                    # number of PCA components used
    kNN_k: int                             # k for kNN density estimation

def fit_manifold_model(
    natural_states: list[np.ndarray],
    pca_components: int = 32,
    kNN_k: int = 10,
) -> tuple[PCADecomposition, Any]:         # (pca_model, kNN index)

def check_manifold_validity(
    intervened_field: np.ndarray,
    pca_model: PCADecomposition,
    kNN_index: Any,                        # sklearn NearestNeighbors-like
    natural_pool: list[np.ndarray],
    *,
    reconstruction_threshold: float | None = None,
    density_ratio_threshold: float = 0.1,
    neighbor_threshold: float | None = None,
) -> ManifoldDiagnostics:
    """Check whether an intervened state lies on the natural hidden-state manifold.

    Three diagnosticians, all must pass for on_manifold = True:
    1. PCA reconstruction error ≤ threshold (calibrated from natural distribution).
    2. kNN density ratio > density_ratio_threshold (intervened density is at least
       10% of median natural density).
    3. Nearest natural neighbor distance ≤ neighbor_threshold (calibrated from
       natural distribution's 95th percentile).

    Thresholds for reconstruction_error and nearest_natural_distance are set from
    the calibration natural-state distribution (e.g., 95th percentile). These are
    NOT hardcoded but frozen at calibration freeze time.
    """
```

### `topological/v2/statistics.py` — Hierarchical Bootstrap, IUT, Per-Family Tests

```python
@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    confidence: float         # e.g., 0.95
    resamples: int
    method: str               # "percentile" or "studentized"

@dataclass(frozen=True)
class PerFamilyResult:
    family: str
    delta_hat: float           # estimated vortex-minus-family margin
    p_value: float             # one-sided bootstrap p-value
    ci_95: BootstrapCI
    effective_samples: int     # seeds contributing analyzable pairs

@dataclass(frozen=True)
class IUTDecision:
    rejected: bool
    p_max: float               # max(p_f) over all families
    alpha: float
    per_family: dict[str, PerFamilyResult]
    descriptive_A_worst: float  # min_f(delta_hat_f) — weakest link point estimate

def hierarchical_bootstrap_seed_level(
    per_seed_statistics: dict[int, list[float]],  # seed → list of within-seed contributions
    *,
    statistic_fn: Callable = np.mean,
    resamples: int = 9999,
    seed: int = 20260722,
) -> BootstrapCI:
    """Hierarchical bootstrap with seed as resampling unit.

    1. Aggregate within each seed: μ_s = statistic_fn(contributions_s).
    2. Resample seeds with replacement B times.
    3. Compute bootstrap distribution of mean(μ_s*) over resampled seeds.
    4. Return percentile CI.
    """

def iut_per_family_test(
    per_seed_family_margins: dict[str, dict[int, list[float]]],
    *,
    resamples: int = 9999,
    alpha: float = 0.05,
) -> IUTDecision:
    """Intersection-Union Test over all null families.

    For each family f:
      1. Compute seed-level mean margins.
      2. Bootstrap p_f = P(delta_f*(b) ≤ 0).
    IUT rejection: max_f(p_f) ≤ alpha.

    Per-family results are returned regardless of global decision.
    """
```

### `topological/v2/protocol.py` — Split Enforcement, Contract Loading, Freeze Verification

```python
@dataclass(frozen=True)
class SplitConfig:
    name: str                  # "dev", "cal", "confirm"
    namespace_prefix: str
    seed_count: int
    allowed_actions: list[str]
    forbidden_actions: list[str]

@dataclass(frozen=True)
class FrozenContract:
    version: str
    contract_hash: str
    splits: dict[str, SplitConfig]
    config_digest: str
    split_registry_digest: str
    frozen_at: str             # ISO 8601 timestamp

def require_split(
    split_name: str,
    contract: FrozenContract,
) -> SplitConfig:
    """Validate that the current split is allowed and return its config."""

def assert_not_confirmatory(contract: FrozenContract) -> None:
    """Raise RuntimeError if current split is confirmatory.

    Used to enforce the no-peeking discipline: calibration code must not
    be called with the confirmatory split loaded.
    """

def load_contract(path: Path) -> FrozenContract:
    """Load and hash-verify a frozen contract file."""

def verify_freeze(
    contract: FrozenContract,
    working_tree: Path,
    *,
    require_clean_tree: bool = True,
) -> bool:
    """Verify that the working tree matches the frozen contract.

    Checks:
    - Git tree is clean (no uncommitted changes) if require_clean_tree.
    - contract.config_digest matches current canonical config.
    - contract.split_registry_digest matches current split registry.
    """
```

---

## New Dataclass Specifications

### `TopologyStatsV2`
```python
@dataclass(frozen=True)
class TopologyStatsV2:
    n_channels: int
    H: int
    W: int
    defect_density: float                # mean defects per plaquette per channel
    per_channel_prevalence: tuple[bool, ...]     # per-channel any(defect)
    state_prevalence: bool                # any channel has both + and −
    branch_margin_min: float              # global minimum pi − |link|
    branch_margin_q01: float              # 1% quantile
    branch_margin_median: float
    signed_tuples: frozenset[tuple[int, int, int, int]]  # (ch, sign, x, y)
    valid_channels: tuple[bool, ...]
    net_charge_valid: bool
    maximum_integer_residual: float
```

### `BranchStability`
```python
@dataclass(frozen=True)
class BranchStability:
    min_margin: float
    q01_margin: float
    q05_margin: float
    median_margin: float
    charge_flip_radius_median: float      # median epsilon to flip any charge
```

### `RepresentativeSpec`
```python
@dataclass(frozen=True)
class RepresentativeSpec:
    n_representatives: int = 10
    seed_offset: int = 0
    method: str = "harmonic_random"       # or "displacement_minimal"
```

### `SurgeryResult`
```python
@dataclass(frozen=True)
class SurgeryResult:
    success: bool
    converged: bool
    target_charge_exact: bool
    post_relax_charge_exact: bool
    harmonic_preserved: bool
    magnitude_error: float
    energy_error: float
    spectrum_error: float
    manifold_distance: float
    iterations: int
    failure_reason: str | None
```

### `ManifoldDiagnostics`
```python
@dataclass(frozen=True)
class ManifoldDiagnostics:
    reconstruction_error: float            # under trained PCA
    kNN_density_ratio: float               # density(intervened) / density(natural)
    nearest_natural_distance: float
    relaxation_drift: float                # ||relax(field) − field||
    on_manifold: bool                      # composite gate
```

### `BehavioralOutcomeV2`
```python
@dataclass(frozen=True)
class BehavioralOutcomeV2:
    arm: str
    donor_ll: np.ndarray                   # (output_positions,)
    recipient_ll: np.ndarray
    margin: float                          # mean(donor_ll − recipient_ll)
    normalized_recovery: float             # Ra formula
    recovery_valid: bool                   # denominator sanity
    commutation_residuals: dict[str, float]
    component_guards: dict[str, bool]
    manifold_diagnostics: ManifoldDiagnostics
```

---

## V2 Package Structure

```
topological/v2/
├── __init__.py
├── model.py              # Reuses topological.model, adds factory + factorial variants
├── task.py               # Reuses topological.task, adds split namespacing
├── topology.py           # Extends topological.topology
├── decomposition.py      # Extends topological.decomposition
├── hodge.py              # Full Hodge decomposition (exact, coexact, harmonic, branch integer)
├── representatives.py    # Same-charge representative sampling
├── surgery.py            # Minimal topological surgery solver + pair annihilation
├── interventions.py      # Extends topological.interventions
├── evaluation.py         # Rewrite of learned_evaluation with corrected gates
├── training.py           # Extends topological.training
├── manifold.py           # Manifold diagnostics (PCA, kNN, relaxation)
├── statistics.py         # Hierarchical bootstrap, IUT, per-family tests
├── protocol.py           # Split enforcement, contract loading, freeze verification
├── smoke.py              # Extends topological.learned_smoke
├── pilot.py              # Extends topological.learned_pilot
├── _artifacts.py         # Reuses topological._artifacts
├── _types.py             # All V2 dataclasses
└── _contract.py          # Contract loading, validation, freeze/verify
```

**V1 imports are safe**: V2 modules can `from topological.X import Y` but never modify V1 files.

---

## Source Hash for V2 Artifacts

Every V2 artifact bundle MUST include a `source.sha256` manifest containing:

```json
{
  "commit_sha": "<git rev-parse HEAD>",
  "tree_sha": "<git rev-parse HEAD^{tree}>",
  "working_tree_clean": true,
  "scientific_files": {
    "topological/topology.py": "<sha256>",
    "topological/decomposition.py": "<sha256>",
    "topological/interventions.py": "<sha256>",
    "topological/model.py": "<sha256>",
    "topological/task.py": "<sha256>",
    "topological/v2/hodge.py": "<sha256>",
    "topological/v2/surgery.py": "<sha256>",
    "topological/v2/representatives.py": "<sha256>",
    "topological/v2/manifold.py": "<sha256>",
    "topological/v2/statistics.py": "<sha256>",
    "topological/v2/protocol.py": "<sha256>"
  },
  "imported_v1_sources": {
    "topological/topology.py": "<sha256>",
    "topological/decomposition.py": "<sha256>",
    "topological/interventions.py": "<sha256>",
    "topological/model.py": "<sha256>",
    "topological/task.py": "<sha256>",
    "topological/training.py": "<sha256>"
  },
  "config_digest": "<sha256 of frozen config.json>",
  "split_registry_digest": "<sha256 of split registry>",
  "environment_digest": "<sha256 of pip freeze sorted + nvidia-smi>"
}
```

The `scientific_files` map covers all V2-original scientific source files.
The `imported_v1_sources` map covers all V1 files that V2 imports from —
these must be captured because V2 depends on their byte-level identity.
A change to any V1 scientific source invalidates all V2 artifact bundles.
This is enforced via `working_tree_clean = true` — no dirty working tree is permitted.

---

## V2 Writing Rules

1. **V1 files are read-only**. Never modify `topological/*.py` (V1 sources).
2. **V2 imports V1 safely**: `from topological.topology import extract_charge` is permitted; V2 wraps/extends, never patches.
3. **No subclassing V1 dataclasses**: Copy fields explicitly into V2 dataclasses.
4. **Branch margin**: Always use `BranchStability` typed frozen dataclass, never raw tuples.
5. **Defect tracking**: Always use signed periodic optimal matching with births/deaths, never signed Jaccard alone.
6. **Factorial baselines**: Always use the 2×2 descriptive names; never `ComplexNoEquiv` or `RealWithEquiv`.
7. **Source hash**: Every artifact bundle records SHA-256 of all imported scientific V1 sources.
8. **V1 feasibility replay**: The `aligned_mask_transplant.pm1_pilot` call is V1-specific legacy hotfix; V2 uses proper subprocess replay as specified in `12_REPRODUCIBILITY_AND_ARTIFACTS.md`.
