# Reproducibility and Artifact Plan

---

## Source Identity

### Required Identity Payload

Every V2 artifact bundle records a `source.json` with the following fields:

```json
{
  "commit_sha": "7a3b2c...",
  "tree_sha": "9e1f4d...",
  "working_tree_clean": true,
  "scientific_files": {
    "topological/topology.py": "sha256:e4a3...",
    "topological/decomposition.py": "sha256:1b9c...",
    "topological/interventions.py": "sha256:7f2d...",
    "topological/model.py": "sha256:a8e1...",
    "topological/task.py": "sha256:3c5b...",
    "topological/training.py": "sha256:f6a0...",
    "topological/learned_evaluation.py": "sha256:d4b2...",
    "topological/v2/hodge.py": "sha256:...",
    "topological/v2/surgery.py": "sha256:...",
    "topological/v2/representatives.py": "sha256:...",
    "topological/v2/manifold.py": "sha256:...",
    "topological/v2/statistics.py": "sha256:...",
    "topological/v2/protocol.py": "sha256:..."
  },
  "imported_v1_module_sha256": {
    "topological/topology.py": "sha256:e4a3...",
    "topological/decomposition.py": "sha256:1b9c...",
    "topological/interventions.py": "sha256:7f2d...",
    "topological/model.py": "sha256:a8e1...",
    "topological/task.py": "sha256:3c5b...",
    "topological/training.py": "sha256:f6a0..."
  },
  "config_digest": "sha256:6d11...",
  "split_registry_digest": "sha256:9f32...",
  "environment_digest": "sha256:0a1b..."
}
```

**Fields**:

| Field | Source | Enforcement |
|-------|--------|-------------|
| `commit_sha` | `git rev-parse HEAD` | Captured at experiment execution time; recorded immutably |
| `tree_sha` | `git rev-parse HEAD^{tree}` | Must match `commit_sha` — if tree differs from commit, working tree was amended |
| `working_tree_clean` | `git diff --stat --exit-code` (exit 0) | MUST be `true`; dirty working tree = abort. No exception. |
| `scientific_files` | `sha256sum` of each V2 scientific `.py` file listed above | Byte-identical check; any modification invalidates the experiment |
| `imported_v1_module_sha256` | `sha256sum` of each V1 `.py` file that V2 imports from | Captures the exact V1 dependency state; needed because V2 builds atop immutable V1 sources |
| `config_digest` | SHA-256 of `json.dumps(config, sort_keys=True)` | Tamper detection for frozen config |
| `split_registry_digest` | SHA-256 of split namespace registry | Ensures cal/confirm separation was not breached |
| `environment_digest` | SHA-256 of `pip freeze \| sort` + `nvidia-smi` (or CPU equivalent) | Full software environment identity |

**Rationale**: `git rev-parse HEAD + git diff --stat` is insufficient. A single
commit SHA and a diff stat line do not verify that the scientific source files
actually executed match the planning contract. Byte-level SHA-256 of all
imported scientific files, together with `working_tree_clean = true`, provides
the cryptographic guarantee that the code running is exactly the code reviewed.

---

## Artifact Schema V2

### Per-Seed Directory
```
{root}/{split}/{model_type}/{seed}/
├── source.json              # Source identity (commit, tree, file hashes, config digest, env digest)
├── environment.json         # pip freeze + CUDA version + hardware info
├── training.json            # TrainingRecord with history, log, selected checkpoint
├── model.pt                 # Best checkpoint state_dict
├── data_hashes.json         # SHA-256 of generated data artifacts (see § Generated Data Hashes)
├── evaluation/
│   ├── topology.json        # TopologyStatsV2 per test example (100 examples)
│   ├── pairs.json           # Selected donor-recipient pairs with diagnostics
│   ├── outcomes.json        # BehavioralOutcomeV2 per arm per pair
│   └── summary.json         # SeedEvaluationV2 aggregate
└── manifold/
    ├── pca_basis.npz        # Trained PCA basis (from calibration split)
    └── diagnostics.json     # ManifoldDiagnostics per pair
```

### Per-Split Decision
```
{root}/{split}/
├── source.json              # Split-level source identity (commit, tree, file hashes)
├── decision.json            # DecideV2 with all gate clauses (per §8.7)
├── iut_result.json          # IUTDecision with per-family PerFamilyResult
└── bootstrap_ci.json        # BootstrapCI per family
```

### Manifest
```
{root}/
├── source.json              # Root-level source identity
├── config.json              # Frozen config used (hash-verified via config_digest)
├── environment.json         # pip freeze + CUDA version + hardware info
├── manifest.sha256           # SHA-256 of all artifact files (written last, after finalization)
└── receipt.json             # Contract version, config digest, status, interpretation boundary
```

---

## Generated Data Hashes

All deterministically generated data artifacts are hashed at creation time.
The `data_hashes.json` file records content SHA-256 for each generated artifact.
**No entry may read `"data.sha256": "N/A"`.**

```json
{
  "train_batch_registry": "sha256:a1b2...",
  "validation_registry": "sha256:c3d4...",
  "calibration_registry": "sha256:e5f6...",
  "confirmatory_registry": "sha256:07a8...",
  "donor_catalog": "sha256:9b0c...",
  "topology_diagnostic_inputs": "sha256:d1e2..."
}
```

| Artifact | Description | Hash Method |
|----------|-------------|-------------|
| `train_batch_registry` | Hash-separated RNG state + deterministic batch indices for all training steps | SHA-256 of sorted JSON serialization |
| `validation_registry` | Validation batch indices for all checkpoints | SHA-256 of sorted JSON serialization |
| `calibration_registry` | Calibration split seed registry (seed → RNG state map) | SHA-256 of sorted JSON serialization |
| `confirmatory_registry` | Confirmatory split seed registry | SHA-256 of sorted JSON serialization |
| `donor_catalog` | Pre-computed donor field catalog per seed (8 donors × 100 recipients) | SHA-256 of `.npz` bytes |
| `topology_diagnostic_inputs` | Input fields fed to Hodge decomposition during topology analysis | SHA-256 of `.npz` bytes |

**Generation**: Each registry is generated once per split, at experiment start,
and stored immutably. The hash is recorded in `data_hashes.json` before any
scientific computation begins. This ensures that downstream analysis artifacts
can be traced to specific input data.

**Verification**: The `verify_manifest` function checks that every `data_hashes.json`
entry matches the on-disk content of the corresponding registry. A mismatch
indicates either corruption or post-hoc data modification.

---

## Smoke Gates

### CPU Integrity Smoke

Runs on CPU without CUDA. Verifies:

1. All model variants instantiate and forward-pass (U1ConvRNN C=1/8, PlainConvRNN, all 4 factorial layouts)
2. All intervention arms execute on synthetic data (vortex, vortex_minimal, harmonic, vortex_remove_pair, vortex_sham, vortex_sign_flip, smooth, magnitude, whole_phase, global_phase, zero_charge_phase, fourier_low, fourier_high, pca, random_direction, natural_recipient, whole_state)
3. Full Hodge decomposition roundtrips within tolerance (exact + coexact + harmonic + branch_integer)
4. Minimal surgery solver converges on 10 random synthetic charge maps
5. Same-charge representative sampling produces 10 Q=0-everywhere fields
6. Branch margin and density statistics are computed
7. Signed periodic optimal matching (defect tracking) produces valid birth/death/persist counts
8. Artifact schema compliance (all required fields present in JSON outputs)
9. Hash-chain verification of smoke outputs (manifest.sha256 self-check)
10. **No scientific result inspection** (FORBIDDEN_RESULT_KEYS enforcement identical to V1 smoke.py:82–90)

### CUDA Resource Smoke

Requires CUDA. Verifies:

1. Training throughput measurement (steps/sec on 16×16 torus, C=1 and C=8)
2. Evaluation throughput measurement (pairs/sec, 100 test examples)
3. Peak VRAM measurement (training: forward + backward; evaluation: forward only)
4. Projected total runtime ≤ budget (wall clock estimate from throughput × experiment registry)
5. No silent CPU fallback (assert `torch.cuda.is_available()` and `model.parameters().device.type == 'cuda'`)

### Smoke Gate States

```
CPU_INTEGRITY_PASS   → All 10 CPU smoke checks pass (no device-specific hardware required)
CUDA_RESOURCE_PASS   → GPU pipeline fits resource budget (no feature-limited pipeline substitution)
SCIENTIFIC_RUN_AUTHORIZED = CPU_INTEGRITY_PASS AND CUDA_RESOURCE_PASS
```

These three states are **distinct and independently recorded**. V1's
`require_promotion` (smoke.py:163–170) conflates the smoke authorization state
with a single binary gate. V2 separates them:

```python
@dataclass(frozen=True)
class SmokeStates:
    cpu_integrity_pass: bool
    cuda_resource_pass: bool
    smoke_completed_at: str         # ISO 8601 timestamp
    smoke_manifest_sha256: str
    cuda_device_info: dict[str, str] | None   # None if CPU-only

    @property
    def scientific_run_authorized(self) -> bool:
        return self.cpu_integrity_pass and self.cuda_resource_pass
```

The pilot executor reads `SmokeStates` and checks `scientific_run_authorized`
before proceeding. It does NOT call `require_promotion` or any function that
returns a single opaque gate string.

---

## Clean-Room Replay Instructions

1. Clone repository at frozen commit SHA (verified against `source.json.commit_sha`)
2. Verify `working_tree_clean = true` (must be clean; abort if dirty)
3. Create fresh virtual environment from frozen lockfile (`uv sync --frozen` or `pip install -r requirements.frozen.txt`; verify `pip freeze` matches `environment_digest`)
4. Run `CPU_INTEGRITY_SMOKE`: `python -m topological.v2.smoke --cpu` → must produce `CPU_INTEGRITY_PASS`
5. Run `CUDA_RESOURCE_SMOKE`: `python -m topological.v2.smoke --cuda` → must produce `CUDA_RESOURCE_PASS`
6. Run `SCIENTIFIC_PILOT` with frozen config: `python -m topological.pilot --config <frozen_config.json> --split confirm` → produces artifact dir
7. Verify `manifest.sha256` matches expected (if pre-computed)
8. Compare `decision.json` with expected decision state

### Clean-Process Replay

The clean-process replay is **not** a same-process internal function. It MUST
remain a subprocess call to guarantee process isolation:

```bash
python -m topological.pilot --replay-hash
```

This spawns an independent Python interpreter instance that:
1. Loads the frozen config
2. Re-runs the scientific payload computation (identical logic, fresh interpreter)
3. Computes SHA-256 of the result
4. Outputs the hash to stdout

The parent process captures stdout and compares the replay hash with the
original run's `scientific_sha256`. A match confirms that the computation
is deterministic and independent of interpreter state.

**Why subprocess, not internal function**: A same-process function call shares
the Python interpreter state (module cache, global state, RNG state, file
descriptors). A subprocess provides genuine process isolation: no shared memory,
no leaked state, no accidental dependence on import order or cached results.
This is a stronger guarantee of clean replay than any internal function can provide.

**Stale dependency note**: The V1 pilot (`topological/pilot.py:131–137`) uses
`aligned_mask_transplant.pm1_pilot` as the subprocess module. This is a
V1-feasibility-specific hotfix (non-scientific infrastructure patch). V2 MUST
NOT replicate this stale module import; the V2 subprocess command is always
`python -m topological.pilot --replay-hash` (or the V2 equivalent in
`topological/v2/pilot.py`).

---

## Planning Diagnostic Receipt

The `receipt.json` file includes SHA-256 hashes of all planning diagnostic
scripts and their raw outputs, providing traceability from experiment artifacts
back to the planning phase that authorized them:

```json
{
  "contract": "PM1-LEARNED-V2",
  "contract_version": "2.0.0",
  "contract_frozen_at": "2026-07-28T00:00:00Z",
  "config_digest": "sha256:6d11...",
  "planning_commit_sha": "34d0adb...",
  "planning_digest": "sha256:...",
  "diagnostic_scripts": {
    "diagnostics/run_untrained_topology.py": "sha256:a12b...",
    "diagnostics/metric_definitions.py": "sha256:c34d...",
    "diagnostics/run_rng_audit.py": "sha256:e56f...",
    "diagnostics/run_canonical_order_audit.py": "sha256:078a...",
    "tools/run_statistical_method_simulation.py": "sha256:9b0c..."
  },
  "diagnostic_raw_outputs": {
    "diagnostics/raw/untrained_gate_replication.json": "sha256:d1e2...",
    "diagnostics/raw/rng_audit.json": "sha256:f3a4..."
  },
  "status": "GO_CONFIRMATORY",
  "interpretation_boundary": "V2 confirmatory evidence only; V1 and calibration excluded from primary claims."
}
```

| Field | Purpose |
|-------|---------|
| `diagnostic_scripts` | SHA-256 of each diagnostic script that produced pre-planning evidence. Ensures the exact script version that generated a finding is recorded. |
| `diagnostic_raw_outputs` | SHA-256 of raw diagnostic output files. Ensures the data supporting each planning claim is immutable and traceable. |
| `planning_commit_sha` | The planning-phase commit at which these diagnostics were executed. |
| `planning_digest` | SHA-256 of `sorted(planning/PM1-LEARNED-V2/**/*)` at planning freeze. |

---

## CI Plan

| Stage | Command | Expected |
|-------|---------|----------|
| Lint | `ruff check topological/v2/ tests/` | 0 errors |
| Type | `mypy topological/v2/` | 0 errors |
| Fast tests | `pytest tests/ -q -m "not slow and not cuda"` | All pass |
| Smoke | `python -m topological.v2.smoke --cpu` | CPU_INTEGRITY_PASS |
| Coverage | `pytest --cov=topological/v2 --cov-report=term` | ≥ 90% |

---

## Write-Once Policy

V1's `WriteOnceArtifact` pattern is preserved:

1. Directory created via `mkdir(exist_ok=False)` → fails if exists
2. Files written with `open(path, 'x')` → fails if exists
3. `data_hashes.json` written after all generated data registries are created
4. Manifest finalized after all writes (`verify_manifest` → `finalize`)
5. `receipt.json` written last (after manifest verification passes)
6. No overwrites, no appends, no deletes after finalization

### Hash Chain Integrity

The verification order during manifest check:

1. Verify `source.json.commit_sha` matches current `git rev-parse HEAD`
2. Verify `source.json.working_tree_clean == true`
3. Verify `source.json.scientific_files` SHA-256 map matches on-disk files
4. Verify `source.json.imported_v1_module_sha256` matches on-disk V1 files
5. Verify `source.json.config_digest` matches `config.json` content hash
6. Verify `data_hashes.json` matches on-disk generated data registries
7. Verify `manifest.sha256` covers all files in the artifact root
8. Verify `receipt.json` is present and references the correct contract version

Any mismatch → abort. The hash chain is integrity-checked, not warning-checked.
