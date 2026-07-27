# PM1-LEARNED-V2 MASTER PLAN — Artifact Index

**Status**: DRAFT-2 — NOT FROZEN
**Target commit**: `67b216bf05382c344425ea493119828412c5b3c5`
**Planning commit**: `34d0adb` (DRAFT-1, superseded)

## Deliverable Map

| # | File | Status | Content |
|---|------|--------|---------|
| D0 | `00_INDEX.md` | COMPLETE | This index |
| D1 | `01_REPOSITORY_MANIFEST.md` | COMPLETE | Full tracked-file census (42 files) |
| D2 | `02_DRAFT0_CORRECTION_LEDGER.md` | COMPLETE | 17 corrections with verdicts |
| D3a | `03_PRIOR_AUDIT_43_FINDINGS.md` | COMPLETE | Individual adjudication of all 43 findings |
| D3b | `03_DIAGNOSTIC_RESULTS.md` | COMPLETE | CPU diagnostic results (exact 128-example replication) |
| D4 | `04_ADDITIONAL_FINDINGS.md` | COMPLETE | 9 new findings with severity grading |
| D5 | `05_LITERATURE_AND_NOVELTY.md` | COMPLETE | Literature matrix; systematic search executed |
| D6 | `06_MATHEMATICAL_FOUNDATIONS.md` | COMPLETE | Equivariance, Hodge, minimal surgery, C>1 |
| D7 | `07_CAUSAL_IDENTIFICATION.md` | COMPLETE | IUT logic, donor metrics, missingness |
| D8 | `08_STATISTICAL_ANALYSIS_PLAN.md` | COMPLETE | IUT primary test, SESOI, missingness policy |
| D9 | `09_EXPERIMENT_REGISTRY.md` | PARTIAL | Full experiment cards needed |
| D10 | `10_IMPLEMENTATION_SPEC.md` | PARTIAL | Design corrections needed |
| D11 | `11_TEST_AND_TRACEABILITY.md` | COMPLETE | 20 new tests + existing mapping |
| D12 | `12_REPRODUCIBILITY_AND_ARTIFACTS.md` | PARTIAL | Subprocess isolation, source hash pending |
| D13 | `13_RESOURCE_SCHEDULE_AND_VOI.md` | PARTIAL | Training step count, stop conditions pending |
| D14 | `14_MANUSCRIPT_REVIEWER_AND_VENUE.md` | COMPLETE | Distill removed, conditional venue matrix |
| D15 | `15_PM1_LEARNED_V2_CONTRACT.yaml` | COMPLETE | Status tags, IUT, SESOI, all families |
| D16 | `16_OPEN_QUESTIONS_AND_DECISIONS.md` | COMPLETE | 10 PI decisions with recommendations |
| D17 | `99_COMPLETENESS_CHECKLIST.md` | COMPLETE | Cross-verification checklist |
| S1 | `PLAN_VALIDATION.json` | COMPLETE | Automated plan integrity check |
| S2 | `diagnostics/README.md` | COMPLETE | Diagnostic package README |
| S3 | `diagnostics/environment.json` | COMPLETE | Diagnostic environment info |
| S4 | `diagnostics/metric_definitions.py` | COMPLETE | Canonical metric definitions |
| S5 | `diagnostics/run_untrained_topology.py` | COMPLETE | Exact gate replication script |
| S6 | `diagnostics/commands.txt` | COMPLETE | Reproduction commands |
| S7 | `diagnostics/raw/untrained_gate_replication.json` | COMPLETE | Raw diagnostic output |
| S8 | `diagnostics/sha256.json` | COMPLETE | SHA-256 of all planning docs |
| S9 | `tools/validate_plan.py` | COMPLETE | Plan integrity linter (5-layer) |
| S10 | `tools/verify_manifest.py` | COMPLETE | SHA-256 manifest verifier |
| S11 | `tools/run_statistical_method_simulation.py` | COMPLETE | Statistical method simulation |
| S12 | `tools/simulation_results.json` | COMPLETE | Simulation output |
| S13 | `diagnostics/run_rng_audit.py` | COMPLETE | Multi-process RNG audit |
| S14 | `diagnostics/run_canonical_order_audit.py` | COMPLETE | Canonical-order audit |
| S15 | `diagnostics/raw/rng_audit.json` | COMPLETE | RNG audit raw output |

## Status Totals

| Status | Count |
|--------|------:|
| COMPLETE | 30 |
| PARTIAL | 4 |
| BLOCKED | 0 |
| NOT_STARTED | 0 |

## Quick Nav

1. **[02_DRAFT0_CORRECTION_LEDGER.md](./02_DRAFT0_CORRECTION_LEDGER.md)** — DRAFT-0 errors and corrections
2. **[03_PRIOR_AUDIT_43_FINDINGS.md](./03_PRIOR_AUDIT_43_FINDINGS.md)** — Full 43-finding adjudication
3. **[03_DIAGNOSTIC_RESULTS.md](./03_DIAGNOSTIC_RESULTS.md)** — CPU diagnostic results
4. **[15_PM1_LEARNED_V2_CONTRACT.yaml](./15_PM1_LEARNED_V2_CONTRACT.yaml)** — Draft V2 contract
5. **[16_OPEN_QUESTIONS_AND_DECISIONS.md](./16_OPEN_QUESTIONS_AND_DECISIONS.md)** — PI decisions needed
6. **[diagnostics/](./diagnostics/)** — Scripts and raw data

## Boundaries

- Source code: READ ONLY
- V1 configs/artifacts: READ ONLY
- Full training: FORBIDDEN until V2 frozen
- GPU training: FORBIDDEN
- Low-cost CPU diagnostics: ALLOWED (<30 min wall)
- Planning documents: WRITE ALLOWED
