# Implementation Specification — Per-File Migration Plan

**Principle**: V1 source files are immutable. V2 is a parallel implementation, not a V1 modification.

**V2 Package Location**: `topological/v2/` (separate namespace, no V1 file modification)

---

## File Migration Matrix

### 1. `topological/model.py` → `topological/v2/model.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `ModelSpec` | Unchanged | Reuse | Keep C=8 for calibration; add C=1 variant |
| `U1ConvRNN` | Unchanged | Reuse | Same interface |
| `PlainConvRNN` | Unchanged | Reuse | Same interface |
| — | No C=1 gateway model | Add | `ScalarU1ConvRNN(channels=1)` |
| — | No factorial baseline variants | Add | `ComplexNoEquivConvRNN`, `RealWithEquivConvRNN` |

### 2. `topological/task.py` → `topological/v2/task.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `CopyBatch`, `CopyTrace` | Unchanged | Reuse | Same |
| `generate_copy_batch` | Unchanged | Reuse | Add split namespacing |
| — | No explicit split namespace | Add | `split='cal'|'confirm'` parameter |
| `run_copy`, `continue_copy` | Unchanged | Reuse | Same |

### 3. `topological/topology.py` → `topological/v2/topology.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `extract_charge` | No branch margin output | Extend | Add `branch_margins: tuple[min, q1, q99]` |
| `canonical_vortex_field` | Unchanged | Reuse | Same |
| — | No charge-flip radius measurement | Add | `charge_flip_radius(field, position, eps_grid)` |
| — | No signed-pair tracking | Add | `defect_tracking(field_t, field_t1)` — pairs matched by proximity |

### 4. `topological/decomposition.py` → `topological/v2/decomposition.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `decompose` | No branch stability validation | Extend | Add `branch_stability_valid: bool` to output |
| `transplant_*` | Unchanged | Reuse | Same |
| — | No minimal-surgery transplant | Add | `transplant_vortex_minimal(recipient, target_charge_map)` |
| — | No harmonic sector swap | Add | `transplant_harmonic(donor_harmonic, recipient)` |
| — | No single-pair annihilation | Add | `annihilate_pair(field, pair_position)` |

### 5. `topological/interventions.py` → `topological/v2/interventions.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `HiddenComponents` | Unused `minimum_magnitude` | Keep + use in gate | Same fields |
| `component_intervention` | Hardcoded to 6 arms | Extend | Add `vortex_minimal`, `harmonic`, `vortex_remove_pair` |
| `matched_global_phase` | Unchanged | Reuse | Same |
| `matched_zero_charge_phase` | Single control index | Extend | Multiple control indices for bootstrap |
| — | No same-charge representative sampling | Add | `sample_representatives(field, n=10, seed)` |
| — | No natural neighbor search | Add | `find_natural_neighbor(field, pool, target_topology)` |
| — | No relaxation post-intervention | Add | `relax(model, field, steps=5)` |
| — | No manifold distance metric | Add | `manifold_distance(field, pca_model, kNN)` |

### 6. `topological/learned_evaluation.py` → `topological/v2/evaluation.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `StateTopology` | No density or branch margin | Extend | Add `defect_density`, `branch_margin`, `per_channel_prevalence` |
| `analyze_topology` | `nonzero_defect` too coarse | Redesign | Return `TopologyStatsV2` with density and stability |
| `select_donor_pair` | Selection collider | Keep for primary, add sensitivity | Add `random_donor_pair` for sensitivity |
| `evaluate_selected_pairs` | Nuisance pooled | Separate families | Per-family margin reporting |
| `_per_model_decision` | `defect_learned_not_innate` broken | Redesign gate | Use density + branch_margin |
| `decide_learned_pilot` | Cross-model comparison | Extend | Paired bootstrap CI |
| — | No manifold validity check | Add | `check_manifold_validity(intervened_field, manifold_model)` |

### 7. `topological/training.py` → `topological/v2/training.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `TrainingSpec` | Unchanged | Reuse | Same |
| `make_model` | No C=1 variant | Extend | Add `model_type='scalar_u1'` |
| `train_seed` | Unchanged | Reuse | Same |
| — | No training convergence audit across seeds | Add | `aggregate_training_curves(artifact_dir)` |

### 8. `topological/learned_smoke.py` → `topological/v2/smoke.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `run_smoke` | Unchanged architecture | Extend | Add `branch_stability_smoke`, `representative_smoke` |
| `_exercise_device` | Shallow equivariance check | Extend | Separate blank/input/readout checks, phase grid, error reporting |

### 9. `topological/learned_pilot.py` → `topological/v2/pilot.py`

| Current Symbol | Problem | Action | V2 API |
|---------------|---------|--------|--------|
| `run_canonical_pilot` | Unchanged architecture | Extend | Split-aware (calibration vs confirmatory), manifest V2 schema |
| — | No split gating | Add | `require_split(split_name)`, `assert_not_confirmatory()` |
| — | No training curve aggregation | Add | `write_aggregated_training_curves()` |

### 10. `topological/pilot.py` and `topological/smoke.py`

| Action |
|--------|
| Remove stale `aligned_mask_transplant.pm1_pilot` dependency |
| Or document with explicit version and provenance |

---

## New Dataclass Specifications

### `TopologyStatsV2`
```python
@dataclass(frozen=True)
class TopologyStatsV2:
    n_channels: int
    H: int
    W: int
    defect_density: float          # mean defects per plaquette per channel
    per_channel_prevalence: tuple[bool, ...]  # per-channel any(defect)
    state_prevalence: bool          # any channel has both + and -
    branch_margin_min: float        # global minimum pi - |link|
    branch_margin_q01: float        # 1% quantile
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
    charge_flip_radius_median: float  # median epsilon to flip any charge
```

### `RepresentativeSpec`
```python
@dataclass(frozen=True)
class RepresentativeSpec:
    n_representatives: int = 10
    seed_offset: int = 0
    method: str = "harmonic_random"  # or "displacement_minimal"
```

### `SurgeryResult`
```python
@dataclass(frozen=True)
class SurgeryResult:
    target_charge_map: np.ndarray        # (H, W)
    intervened_field: np.ndarray         # (C, H, W) complex
    displacement: float                   # L2 distance from recipient
    energy_ratio: float                   # gradient_energy ratio
    harmonic_shift: tuple[float, float]   # (Δwx, Δwy)
    manifold_distance: float
    charge_verification: bool             # does Q(field) == target?
```

### `ManifoldDiagnostics`
```python
@dataclass(frozen=True)
class ManifoldDiagnostics:
    reconstruction_error: float           # under trained PCA
    kNN_density_ratio: float              # density(intervened) / density(natural)
    nearest_natural_distance: float
    relaxation_drift: float               # ||relax(field) - field||
    on_manifold: bool                     # composite gate
```

### `BehavioralOutcomeV2`
```python
@dataclass(frozen=True)
class BehavioralOutcomeV2:
    arm: str
    donor_ll: np.ndarray        # (output_positions,)
    recipient_ll: np.ndarray
    margin: float                # mean(donor_ll - recipient_ll)
    normalized_recovery: float   # Ra formula
    recovery_valid: bool         # denominator sanity
    commutation_residuals: dict[str, float]
    component_guards: dict[str, bool]
    manifold_diagnostics: ManifoldDiagnostics
```

---

## V2 Package Structure

```
topological/v2/
├── __init__.py
├── model.py         # Reuses topological.model, adds C=1 + factorial variants
├── task.py           # Reuses topological.task, adds split namespacing
├── topology.py      # Extends topological.topology
├── decomposition.py # Extends topological.decomposition
├── interventions.py # Extends topological.interventions
├── evaluation.py    # Rewrite of learned_evaluation with corrected gates
├── training.py      # Extends topological.training
├── smoke.py         # Extends topological.learned_smoke
├── pilot.py         # Extends topological.learned_pilot
├── _artifacts.py    # Reuses topological._artifacts
├── _types.py        # All V2 dataclasses
└── _contract.py     # Contract loading, validation, freeze/verify
```

**V1 imports are safe**: V2 modules can `from topological.X import Y` but never modify V1 files.
