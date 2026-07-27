# Reproducibility and Artifact Plan

## Artifact Schema V2

### Per-Seed Directory
```
{root}/{split}/{model_type}/{seed}/
├── training.json          # TrainingRecord with history, log, selected checkpoint
├── model.pt               # Best checkpoint state_dict
├── evaluation/
│   ├── topology.json      # TopologyStatsV2 per test example
│   ├── pairs.json         # Selected donor-recipient pairs with diagnostics
│   ├── outcomes.json      # BehavioralOutcomeV2 per arm per pair
│   └── summary.json       # SeedEvaluationV2 aggregate
└── manifold/
    ├── pca_basis.npz      # Trained PCA basis (from calibration)
    └── diagnostics.json   # ManifoldDiagnostics per pair
```

### Per-Split Decision
```
{root}/{split}/decision.json   # DecideV2 with all gate clauses
```

### Manifest
```
{root}/manifest.sha256          # SHA-256 of all artifact files
{root}/source.sha256             # SHA-256 of git tree at experiment start
{root}/config.json               # Frozen config used (hash-verified)
{root}/environment.json          # pip freeze + CUDA version + hardware info
```

## Smoke Gates

### CPU Integrity Smoke
Runs on CPU without CUDA. Verifies:
1. All model variants instantiate and forward-pass
2. All intervention arms execute on synthetic data
3. Decomposition roundtrips within tolerance
4. Branch margin and density statistics are computed
5. Artifact schema compliance (all required fields present)
6. No scientific result inspection
7. Hash-chain verification of smoke outputs

### CUDA Resource Smoke
Requires CUDA. Verifies:
1. Training throughput measurement (steps/sec)
2. Evaluation throughput measurement (pairs/sec)
3. Peak VRAM measurement
4. Projected total runtime ≤ budget
5. No silent CPU fallback

### Smoke Gate States
```
CPU_INTEGRITY_PASS   → CPU-only pipeline is functional
CUDA_RESOURCE_PASS   → GPU pipeline fits resource budget
SCIENTIFIC_RUN_AUTHORIZED = CPU_INTEGRITY_PASS AND CUDA_RESOURCE_PASS
```

Current V1 promotion logic (`require_promotion`) conflates these. V2 separates them.

## Clean-Room Replay Instructions

1. Clone repository at frozen commit hash
2. Create fresh virtual environment from `uv.lock` or `requirements.frozen.txt`
3. Run `CPU_INTEGRITY_SMOKE` → must pass
4. Run `CUDA_RESOURCE_SMOKE` → must pass
5. Run `SCIENTIFIC_PILOT` with frozen config → produces artifact dir
6. Verify `manifest.sha256` matches expected (if pre-computed)
7. Compare `decision.json` with expected decision state

## Stale Dependency Removal

`topological/pilot.py:132`: Remove `aligned_mask_transplant.pm1_pilot` call. If the clean-process replay is genuinely needed, implement it as an internal function that:
1. Re-runs the scientific payload computation
2. Compares SHA-256 with original run
3. Reports match/mismatch

This should be a simple internal verification, not a subprocess call to an external module.

## CI Plan

| Stage | Command | Expected |
|-------|---------|----------|
| Lint | `ruff check topological/v2/ tests/` | 0 errors |
| Type | `mypy topological/v2/` | 0 errors |
| Fast tests | `pytest tests/ -q -m "not slow and not cuda"` | All pass |
| Smoke | `python -m topological.v2.smoke --cpu` | CPU_INTEGRITY_PASS |
| Coverage | `pytest --cov=topological/v2 --cov-report=term` | ≥ 90% |

---

## Source Integrity

| Artifact | Hash | Purpose |
|----------|------|---------|
| `source.sha256` | `git rev-parse HEAD` + `git diff --stat` | Exact code version |
| `config.sha256` | SHA-256 of canonical config JSON | Config tamper detection |
| `environment.sha256` | `pip freeze` sorted + `nvidia-smi` | Environment reproduction |
| `data.sha256` | Not applicable (data generated deterministically) | — |

---

## Write-Once Policy

V1's `WriteOnceArtifact` pattern is preserved:
1. Directory created via `mkdir(exist_ok=False)` → fails if exists
2. Files written with `open(path, 'x')` → fails if exists
3. Manifest finalized after all writes
4. No overwrites, no appends, no deletes after finalization
