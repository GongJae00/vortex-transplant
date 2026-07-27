# Repository Manifest — `67b216b` Full Census

Generated from `git ls-tree -r --long HEAD`.

## Tracked Files (42 total)

### Configs (2 files)
| Path | Blob SHA | Size | Scientific Role |
|------|----------|-----:|-----------------|
| `configs/topological_feasibility_v1.json` | `cb3c992` | 459 | Frozen feasibility config (5 seeds, 32x32) |
| `configs/topological_learned_v1.json` | `3dd7a41` | 708 | Frozen learned config (10 seeds, 30k updates) |

### Scripts (3 files, all executable)
| Path | Blob SHA | Size | Role | Entry Point |
|------|----------|-----:|------|-------------|
| `scripts/run_topological_feasibility.sh` | `fd3da76` | 191 | Feasibility smoke + pilot | `topological.smoke` → `topological.pilot` |
| `scripts/run_topological_smoke.sh` | `44e50c8` | 75 | Learned smoke gate only | `topological.learned_smoke` |
| `scripts/run_topological_pilot.sh` | `c877f29` | 75 | Learned full pilot | `topological.learned_pilot` |

### Source — topological/ (13 files)
| Path | Blob SHA | Size | Domain | Depends On |
|------|----------|-----:|--------|------------|
| `__init__.py` | `e69de29` | 0 | Empty init | — |
| `_artifacts.py` | `b922622` | 2962 | WriteOnceArtifact, manifest verification | — |
| `baseline.py` | `afde62f` | 6365 | LSTM baseline model, training | `model.py`, `task.py` |
| `decomposition.py` | `f78c4a1` | 5422 | Compact field decomposition (vortex/smooth/mag) | `topology.py`, `fixture.py` |
| `evaluation.py` | `8420735` | 8616 | Feasibility evaluator (synthetic fields) | `fixture.py`, `decomposition.py`, `_artifacts.py` |
| `fixture.py` | `0125f25` | 4827 | Synthetic compact field generators | `topology.py` |
| `interventions.py` | `6cfc632` | 12978 | 13-arm intervention suite | `decomposition.py`, `topology.py` |
| `learned_evaluation.py` | `34f82fa` | 32570 | Causal evaluation on trained nets (largest file) | `interventions.py`, `decomposition.py`, `task.py`, `topology.py`, `model.py` |
| `learned_pilot.py` | `ce35623` | 12263 | Canonical 10-seed x 2-model pilot executor | `training.py`, `learned_evaluation.py`, `_artifacts.py`, `task.py` |
| `learned_smoke.py` | `716d2bc` | 15351 | Learned smoke gate (result-blind) | `model.py`, `training.py`, `interventions.py`, `task.py`, `_artifacts.py` |
| `model.py` | `b34df84` | 9639 | U1ConvRNN, PlainConvRNN, ModelSpec | — |
| `pilot.py` | `41f70f7` | 6854 | Feasibility pilot executor | `evaluation.py`, `smoke.py`, `_artifacts.py`, `aligned_mask_transplant.pm1_pilot` (stale) |
| `smoke.py` | `27a259f` | 5650 | Feasibility smoke gate | `fixture.py`, `decomposition.py`, `_artifacts.py` |
| `task.py` | `a117f2f` | 6382 | Copy task batch/trace/run | — |
| `topology.py` | `de097dd` | 6391 | Charge extraction, canonical vortex fields | — |
| `training.py` | `94b98ed` | 11120 | Training loop, validation, checkpointing | `model.py`, `task.py` |

### Tests — tests/ (14 files)
| Path | Blob SHA | Size | Coverage |
|------|----------|-----:|----------|
| `test_topological_smoke.py` | `e2fee80` | 597 | Feasibility smoke (write-once, result-blind) |
| `test_topological_task.py` | `e2d7246` | 2564 | Copy batch/trace determinism, donor catalog |
| `test_topological_model.py` | `ba31d72` | 5204 | U1 equivariance, blank embedding, param counts |
| `test_topological_topology.py` | `a7930b3` | 2307 | Charge extraction roundtrip, gauce invariance |
| `test_topological_fixture.py` | `5976ac8` | 1093 | Synthetic fixture determinism, template invariance |
| `test_topological_baseline.py` | `04038e5` | 2584 | LSTM interface, training step |
| `test_topological_training.py` | `e99bfe0` | 4260 | Training determinism, checkpoint, logs |
| `test_topological_evaluation.py` | `b331780` | 1455 | Feasibility evaluator coverage |
| `test_topological_pilot.py` | `add61f1` | 808 | Pilot config contract |
| `test_topological_decomposition.py` | `e644c9e` | 4027 | Decomp roundtrip, transplants |
| `test_topological_interventions.py` | `163426d` | 4757 | Hidden conversion, matched controls |
| `test_topological_learned_smoke.py` | `25380d4` | 1508 | Learned smoke gate contract |
| `test_topological_learned_pilot.py` | `c586013` | 1317 | Learned pilot hash chain, checkpoint |
| `test_topological_learned_evaluation.py` | `058d373` | 11118 | Signed topology, donor selection, 13-arm |
| **Total tests** | | | **71 tests** (all pass at HEAD) |

### Metadata (4 files)
| Path | Blob SHA | Size | Role |
|------|----------|-----:|------|
| `.gitignore` | `330776f` | 638 | Git ignore rules |
| `CITATION.cff` | `49292a7` | 286 | Citation metadata |
| `LICENSE` | `14fac91` | 1056 | MIT License |
| `README.md` | `60f09e8` | 5325 | Primary documentation, package map |

### Build (2 files)
| Path | Blob SHA | Size | Role |
|------|----------|-----:|------|
| `pyproject.toml` | `6a97b52` | 1192 | Build config (setuptools), deps |
| `uv.lock` | `e904db1` | 151831 | Dependency lock file |

### Paper Directory (not in main package; placeholder only)
| Path | Role |
|------|------|
| `paper/main.tex` | Placeholder LaTeX scaffold |
| `paper/main.pdf` | Compiled placeholder |
| `paper/references.bib` | One unrelated entry |
| `paper/AGENTS.md` | Paper workspace instructions |
| `paper/README.md` | Paper workspace README |
| `paper/figures/.gitkeep` | Empty |
| `paper/tables/.gitkeep` | Empty |
| `paper/sections/.gitkeep` | Empty |

## Stale External Dependency

**`aligned_mask_transplant.pm1_pilot`** — Referenced at `topological/pilot.py:132`:
```python
replay = subprocess.run(
    [sys.executable, "-m", "aligned_mask_transplant.pm1_pilot", "--replay-hash"],
    ...
)
```

This is a stale module path. The feasibility pipeline's clean-process replay (`clean_process_replay_exact` clause) depends on an external package that does not exist in this repository. **This finding (#40) is CONFIRMED**, but scope is limited to the feasibility path, not the learned path.

## Dependency Graph (Simplified)

```
task.py ─────────────────────────────────────────────────────────┐
model.py ────────────────────────────────────────────────────────┤
topology.py ─────► decomposition.py ──► interventions.py ────────┤
fixture.py ───────┘                    │                          │
                                       ▼                          │
training.py ◄─────────────────────────────────────────────────────┤
   │                                                              │
   ▼                                                              │
learned_smoke.py ────► _artifacts.py                              │
learned_pilot.py ─────┘    │                                      │
   │                       ▼                                      │
   ▼                 WriteOnceArtifact                            │
learned_evaluation.py ◄── interventions.py ◄── decomposition.py  │
   │                                                              │
   └─► select_donor_pair ──► evaluate_selected_pairs              │
        │                       │                                 │
        ▼                       ▼                                 │
    13-arm intervention   decide_learned_pilot ◄── _per_model_decision
```
