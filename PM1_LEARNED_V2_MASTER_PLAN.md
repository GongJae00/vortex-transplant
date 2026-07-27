# PM1-LEARNED-V2 Master Plan

**Status**: DRAFT-2 — NOT FROZEN
**Audit base commit**: `67b216bf05382c344425ea493119828412c5b3c5`
**Planning commit**: `34d0adb969887d21ee16d6c4307f3ebb281c1cc7` (DRAFT-1, superseded)

## Primary Scientific Question

Do U(1) vortex defects in the hidden state of a U(1)-equivariant ConvRNN causally control output behavior on a variable-delay copy task, with effects that are (a) stronger than matched nuisance perturbations, (b) representative-invariant, and (c) on the natural hidden-state manifold?

## Artifact Package

Complete planning package at: [planning/PM1-LEARNED-V2/00_INDEX.md](planning/PM1-LEARNED-V2/00_INDEX.md)

### Quick Links
- [Open decisions (PI input needed)](planning/PM1-LEARNED-V2/16_OPEN_QUESTIONS_AND_DECISIONS.md)
- [Draft contract](planning/PM1-LEARNED-V2/15_PM1_LEARNED_V2_CONTRACT.yaml)
- [DRAFT-0 correction ledger](planning/PM1-LEARNED-V2/02_DRAFT0_CORRECTION_LEDGER.md)
- [Prior 43 findings adjudication](planning/PM1-LEARNED-V2/03_PRIOR_AUDIT_43_FINDINGS.md)
- [Diagnostic results & raw data](planning/PM1-LEARNED-V2/diagnostics/)
- [Plan validation report](planning/PM1-LEARNED-V2/PLAN_VALIDATION.json)

## Current P0 Blockers

1. `defect_learned_not_innate` gate is structurally impassable (untrained prevalence → 1.0 under the `nonzero_defect` metric)
2. V2 contract primary estimand excludes Fourier/PCA/harmonic/null families
3. All-null statistical test uses invalid `min(p_f)` instead of IUT `max(p_f)`
4. C>1 channelwise charge is not a genuine topological invariant (\(\pi_1(S^{2C-1})=0\))
5. Existing `transplant_vortex` identified incorrectly as minimal surgery

## Implementation Status

- **Scientific V2 source code**: FORBIDDEN (contract not frozen)
- **V1 source code**: IMMUTABLE
- **GPU training**: FORBIDDEN
- **CPU planning diagnostics only**: ALLOWED

## Freeze Checklist

Before V2 implementation can begin:

- [ ] 43 prior findings individually adjudicated
- [ ] All diagnostic results reproducible from committed scripts
- [ ] Exact 128-example untrained gate replication completed
- [ ] Mathematical equivariance/readout analysis corrected
- [ ] Primary IUT statistical test frozen
- [ ] All comparable null families in primary estimand
- [ ] Contract status tags resolved (UNFROZEN_CALIBRATION vs FROZEN)
- [ ] Literature search actually executed
- [ ] Plan validator passes all checks
- [ ] PI approval of contract
