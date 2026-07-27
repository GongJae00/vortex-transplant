# Test and Traceability Matrix

Each finding must be traceable to a code change and a test.

## Required New Tests

| Test ID | Finding | Test Type | What It Verifies |
|---------|---------|-----------|-----------------|
| T-V2-01 | F-NEW-A | Unit + Integration | `defect_density > 0` and `> untrained` gate pass/fail |
| T-V2-02 | F-NEW-A | Property | StatePrevalence ≤ 1.0 (trivial but validates type) |
| T-V2-03 | F-NEW-D | Property | `branch_margin_min ≥ threshold` for valid decompositions |
| T-V2-04 | F-NEW-F | Unit | DefectDensity sparsity: known vortex fields have expected density |
| T-V2-05 | F-NEW-G | Integration | `CleanProcessReplay` raises clear error for missing external module |
| T-V2-06 | C-05 (correction) | Unit | `generate_copy_batch` with split namespace produces correct split prefix |
| T-V2-07 | C-15 | Unit | `minimal_annihilation` removes target pair, preserves others, minimizes displacement |
| T-V2-08 | C-16 | Unit | Per-family null statistics computed independently, not pooled |
| T-V2-09 | C-08 | Property | Charge-flip radius measurement consistent with known thresholds |
| T-V2-10 | C-03 | Integration | Equivariance check on blank transition, input transition, readout separately |
| T-V2-11 | C-14 | Unit | C=1 model forward pass, training step, validation |
| T-V2-12 | C-16 | Integration | Factorial baseline models: ComplexNoEquiv, RealWithEquiv have distinct behavior |
| T-V2-13 | SAP | Unit | Hierarchical bootstrap CI contains expected coverage (~95%) on simulated data |
| T-V2-14 | SAP | Unit | Split enforcement: confirmatory namespace blocks calibration config; vice versa |
| T-V2-15 | SAP | Integration | Missingness reporting: each invalid-pair reason counted, no silent drops |
| T-V2-16 | C-11 | Integration | Manifold diagnostics: PCA reconstruction, kNN density ratio all finite |
| T-V2-17 | C-12 | Integration | Natural neighbor search finds states with similar topology |
| T-V2-18 | C-13 | Integration | Harmonic sector swap: donor harmonic applied, vortex unchanged |
| T-V2-19 | Reproducibility | Smoke | CPU integrity smoke exercises all 13 arms without CUDA; CUDA smoke adds resource check |
| T-V2-20 | Contract | Smoke | Frozen config hash matches expected; deviation → SMOKE_FAIL |

## Existing Tests — Mapping to V2

| Existing Test | V1 File | V2 Relevance | Action |
|--------------|---------|-------------|--------|
| `test_topological_model.py` | model.py | Core equivariance | Extend with C=1 test |
| `test_topological_task.py` | task.py | Task determinism | Extend with split namespace test |
| `test_topological_topology.py` | topology.py | Charge extraction | Extend with branch margin output |
| `test_topological_decomposition.py` | decomposition.py | Decomposition | Extend with minimal surgery |
| `test_topological_interventions.py` | interventions.py | Intervention arms | Extend with V2 arms |
| `test_topological_training.py` | training.py | Training loop | Reuse, add C=1 check |
| `test_topological_learned_smoke.py` | learned_smoke.py | Smoke gate | Extend with branch stability smoke |
| `test_topological_learned_evaluation.py` | learned_evaluation.py | Evaluation | Rewrite with corrected gates |

## Finding-Test Traceability

| Finding | Code | Unit Test | Integration Test | Scientific Diagnostic |
|---------|------|-----------|-----------------|----------------------|
| F-NEW-A (defect gate) | `evaluation.py:defect_learned_not_innate` | T-V2-01 | T-V2-14 | D-03 diagnostic |
| F-NEW-B (embedding topology) | `evaluation.py:untrained_baseline` | T-V2-02 | — | D-03 diagnostic |
| F-NEW-C (Plain also has defects) | `evaluation.py:cross_model` | — | T-V2-12 | D-02 diagnostic |
| F-NEW-D (branch margins) | `decomposition.py:branch_check` | T-V2-03 | T-V2-19 | D-02 diagnostic |
| F-NEW-G (stale replay) | `pilot.py:clean_replay` | T-V2-05 | — | — |
| C-01 (stale module) | `pilot.py:132` | T-V2-05 | — | `git grep` |
| C-02 (min magnitude unused) | `interventions.py:99` | — | T-V2-03 | — |
| C-03 (equivariance coverage) | `smoke.py:_exercise_device` | T-V2-10 | T-V2-19 | — |
| C-04 (training curves) | `pilot.py:_training_record` | — | T-V2-14 | — |
| C-05 (task support) | `task.py:generate_copy_batch` | T-V2-06 | — | Math calculation |
| C-08 (branch margin threshold) | `topology.py:branch_margins` | T-V2-03 | T-V2-19 | C03 calibration |
| C-09 (null ensemble size) | `evaluation.py:null_draws` | T-V2-08 | — | C08 calibration |
| C-10 (sample size) | `pilot.py:n_seeds` | T-V2-13 | — | C05-C08 calibration |
