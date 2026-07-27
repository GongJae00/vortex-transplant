# PM1-LEARNED-V2 Master Plan

**Status**: PLANNING-DRAFT-3 — FREEZE-CANDIDATE
**Audit base commit**: `67b216bf05382c344425ea493119828412c5b3c5`
**Planning commit**: `a9c21ab` (DRAFT-3)

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

- ~~`defect_learned_not_innate` gate~~ → **REMOVED** (replaced with analyzable-stable-topology gate)
- ~~Contract primary estimand excludes Fourier/PCA/harmonic~~ → **FIXED** (all 10 families included)
- ~~All-null statistical test uses `min(p_f)`~~ → **FIXED** (IUT with `max(p_f)`)
- ~~C>1 channelwise charge is not a genuine topological invariant~~ → **DOCUMENTED** (C=1 gateway strategy)

## Implementation Status

- **Planning**: FREEZE-CANDIDATE (all 34 artifacts COMPLETE, all 5 validator layers PASS)
- **Scientific V2 source code**: AWAITING PI APPROVAL
- **V1 source code**: IMMUTABLE
- **GPU training**: AWAITING CONTRACT FREEZE

## Freeze Checklist

- [x] 43 prior findings individually adjudicated
- [x] All diagnostic results reproducible from committed scripts
- [x] Exact 128-example untrained gate replication completed
- [x] Mathematical equivariance/readout analysis corrected
- [x] Primary IUT statistical test frozen (`max(p_f)`)
- [x] All comparable null families in primary estimand (10 families)
- [x] Contract status tags resolved
- [x] Literature search executed
- [x] Plan validator passes all 5 layers
- [x] All 34 artifacts COMPLETE (0 PARTIAL)
- [ ] PI approval of contract
