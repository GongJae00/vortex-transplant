# PM1-LEARNED-V2 MASTER PLAN — Artifact Index

**Status**: IN PROGRESS
**Target commit**: `67b216bf05382c344425ea493119828412c5b3c5`
**Branch**: `main` (HEAD == origin/main)
**Working tree**: clean (only untracked `PM1_LEARNED_V2_MASTER_PLAN.md`)

## Deliverable Map

| # | File | Status | Content |
|---|------|--------|---------|
| D0 | `00_INDEX.md` | DONE | This index |
| D1 | `01_REPOSITORY_MANIFEST.md` | DONE | Full tracked-file census with blob SHA, size, role, dependencies |
| D2 | `02_DRAFT0_CORRECTION_LEDGER.md` | DONE | 17 correction items with verdicts, evidence, reasoning |
| D3 | `03_PRIOR_AUDIT_43_FINDINGS.md` | DONE | Individual adjudication of all 43 prior audit findings |
| D4 | `04_ADDITIONAL_FINDINGS.md` | DONE | New findings discovered during this audit |
| D5 | `05_LITERATURE_AND_NOVELTY.md` | DONE | Systematic literature matrix and novelty adjudication |
| D6 | `06_MATHEMATICAL_FOUNDATIONS.md` | IN PROGRESS | Equivariance, homotopy, multichannel, Hodge theory |
| D7 | `07_CAUSAL_IDENTIFICATION.md` | IN PROGRESS | SCM, estimands, sufficiency/necessity/specificity |
| D8 | `08_STATISTICAL_ANALYSIS_PLAN.md` | IN PROGRESS | Hierarchy, splits, primary test, power, multiplicity |
| D9 | `09_EXPERIMENT_REGISTRY.md` | IN PROGRESS | 20 experiment cards with full specifications |
| D10 | `10_IMPLEMENTATION_SPEC.md` | IN PROGRESS | Per-file migration plan, API, dataclass specs |
| D11 | `11_TEST_AND_TRACEABILITY.md` | IN PROGRESS | Finding-test-code mapping, required new tests |
| D12 | `12_REPRODUCIBILITY_AND_ARTIFACTS.md` | IN PROGRESS | Smoke gates, artifact schema, clean-room instructions |
| D13 | `13_RESOURCE_SCHEDULE_AND_VOI.md` | IN PROGRESS | Throughput model, phase estimates, VOI ordering |
| D14 | `14_MANUSCRIPT_REVIEWER_AND_VENUE.md` | IN PROGRESS | Claim tiers, reviewer simulation, venue analysis |
| D15 | `15_PM1_LEARNED_V2_CONTRACT.yaml` | IN PROGRESS | Frozen/freezable YAML contract for V2 |
| D16 | `16_OPEN_QUESTIONS_AND_DECISIONS.md` | IN PROGRESS | Unresolved decisions pending PI input |
| D17 | `99_COMPLETENESS_CHECKLIST.md` | PENDING | Cross-verification of all deliverables |

## Quick Nav: Most Critical Items

1. **[02_DRAFT0_CORRECTION_LEDGER.md](./02_DRAFT0_CORRECTION_LEDGER.md)** — Start here. Four factual errors found in DRAFT-0.
2. **[03_PRIOR_AUDIT_43_FINDINGS.md](./03_PRIOR_AUDIT_43_FINDINGS.md)** — Adjudication of all 43 prior findings.
3. **[16_OPEN_QUESTIONS_AND_DECISIONS.md](./16_OPEN_QUESTIONS_AND_DECISIONS.md)** — Questions needing PI input before freezing V2.
4. **[15_PM1_LEARNED_V2_CONTRACT.yaml](./15_PM1_LEARNED_V2_CONTRACT.yaml)** — The target frozen contract (when ready).

## Plan-Only / Compute Boundaries

- Source code: READ ONLY
- V1 configs: READ ONLY
- V1 artifacts: READ ONLY
- Full training: FORBIDDEN until V2 frozen
- GPU training: FORBIDDEN
- Low-cost CPU diagnostics: ALLOWED (≤30 min wall, ≤100 training updates)
- Planning documents: WRITE ALLOWED under `planning/PM1-LEARNED-V2/`
