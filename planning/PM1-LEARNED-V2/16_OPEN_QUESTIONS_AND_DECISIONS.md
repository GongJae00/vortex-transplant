# Open Questions and Decisions

These require explicit PI input before the V2 contract can be frozen.

## P0: Defect Prevalence Gate Redesign [PI DECISION REQUIRED]

**Problem**: `defect_learned_not_innate` cannot pass because untrained prevalence = 1.0.

**Options**:
1. **Defect density**: `trained_density > untrained_density` — does training create more defects?
   - Risk: training might *reduce* defect density (smoothing effect). Gate would fail in opposite direction.
2. **Branch margin improvement**: `trained_branch_margin > untrained_branch_margin` — does training stabilize defects?
   - Risk: unclear if U1 specifically stabilizes defects or if any trained model does.
3. **Signed Jaccard stability**: Cross-seed topology convergence.
   - Risk: requires multiple trained seeds to evaluate before gate is decided.
4. **Per-channel prevalence**: `trained_per_channel_prev > untrained_per_channel_prev` in ≥ 6/8 channels.
   - Risk: similar to state-level but with more granularity.
5. **Remove the gate**: Accept that defects "exist at init" and focus on causal role, not emergence.
   - Simplest but changes the claim: "trained dynamics reorganize vortex structure" not "create it."

**Recommendation**: (2) + (3) as the primary gate. Require trained models to have more stable (higher branch margin) AND more reproducible (higher cross-seed Jaccard) topology than untrained.

---

## P1: C=1 Gateway vs C>1 with Phase Locking [PI DECISION REQUIRED]

**Problem**: Channelwise vortices in C>1 are not topologically protected. The multichannel field can unwind through other channel dimensions.

**Options**:
1. **Go all-in on C=1**: Run the entire study (calibration + confirmatory) on C=1. Simplest topologically.
   - Risk: C=1 may not learn the task well; C=1 model uses fewer parameters.
2. **C=1 as calibration gateway**: Verify C=1 phenomenon exists, then scale to C=8 with phase-locking analysis.
   - Risk: Two separate code paths; C=8 results may not scale.
3. **C=8 with empirical phase-locking**: Measure inter-channel phase correlation; claim topological protection if correlation > threshold.
   - Risk: Reviewer may reject "empirical phase locking" as insufficient for topological claim.

**Recommendation**: (2). Run C=1 micro-pilot first. If C=1 produces clean, causally relevant defects, that's a stronger finding than C=8 anyway. If C=1 fails to learn, fall back to C=8 with phase-locking analysis.

---

## P2: Confirmatory Seed Count [PI DECISION REQUIRED]

**Problem**: Cannot determine seed count without calibration variance data.

**Decision criteria**:
- Calibration N = 5-10 seeds (adequate for variance estimation)
- Confirmatory N = determined by simulation-based power analysis after calibration
- Minimum confirmatory N = 20 (arbitrary, adjust)
- Budget constraint: ceiling at 100 seeds

---

## P3: Null Family Draws per Pair [PI DECISION REQUIRED]

**Options**:
- \(B=99\): Order statistic for q95 = 94th. CI width ≈ ±3%. p_min = 0.01.
- \(B=199\): Order statistic for q95 = 189th. CI width ≈ ±2%. p_min = 0.005.
- \(B=499\): Order statistic for q95 = 474th. CI width ≈ ±1%. p_min = 0.002.

**Trade-off**: More draws = more compute per pair = fewer pairs or seeds at same budget.

**Recommendation**: \(B=199\) as minimum; \(B=499\) if compute budget allows.

---

## P4: Primary Null Comparison Method [PI DECISION REQUIRED]

**Options**:
1. **Pool all null families → q95**: Simple, single threshold. But mixes families with different hypotheses.
2. **Per-family q95 then max**: Vortex must beat EACH family's q95. Conservative (Bonferroni-like).
3. **Per-family q95 then min p-value with correction**: Vortex tested against each family; reject if adjusted p < 0.05.

**Recommendation**: (2) — conservative, interpretable, each family is a distinct counterfactual.

---

## P5: Manifold Model Selection [PI DECISION REQUIRED]

**Problem**: Multiple ways to define "on-manifold." Calibration should compare them.

**Candidates**:
1. PCA reconstruction error (fit on calibration training data)
2. kNN density ratio (intervened vs natural)
3. Nearest natural neighbor distance
4. Relaxation drift (||relax(intervened) - intervened||)

**Recommendation**: Run all four in calibration. Select the one with best separation between natural states and known-off-manifold perturbations (random perturbation, large phase rotation).

---

## P6: Factorial Baseline Scope [PI DECISION REQUIRED]

**Problem**: PlainConvRNN differs from U1ConvRNN in multiple ways. Which do we isolate?

**Candidates**:
1. `ComplexNoEquiv`: complex cross-coupled conv + tanh + LayerNorm (removes radial_tanh)
2. `RealWithEquiv`: real conv + radial_tanh (allows magnitude-varying fields with equivariant nonlinearity)
3. U1ConvRNN without blank embedding enforcement
4. U1ConvRNN with LayerNorm added
5. PlainConvRNN with complex convolution

**Recommendation**: Start with (1) and (2). If U1 > ComplexNoEquiv AND U1 > RealWithEquiv, the effect is specific to the U(1)-equivariant combination, not either component alone. This is a strong claim. Add more variants only if needed to disambiguate.

---

## P7: V1 Pilot Execution Policy [PI DECISION REQUIRED]

**Problem**: Should we run V1's learned pilot or skip directly to V2?

**Options**:
1. **Skip V1**: Gate is structurally impassable. Don't waste GPU hours.
2. **Run V1 in exploratory mode**: Execute with `EXPLORATORY_NONCONFIRMATORY` namespace to diagnose failure modes.
3. **Run V1 with modified gate**: Hotfix the `defect_learned_not_innate` clause and run.

**Recommendation**: (2) — valuable diagnostic. But results must NOT be presented as confirmatory evidence. Run with reduced seeds (3-5) and reduced arm set to save compute.

---

## P8: Stale Dependency Resolution [PI DECISION REQUIRED]

**Problem**: `topological/pilot.py:132` calls `aligned_mask_transplant.pm1_pilot`.

**Options**:
1. **Remove**: Delete the subprocess call. Replace with internal replay verification.
2. **Document**: Keep the call, add explicit version + provenance documentation.
3. **Ignore**: Leave as-is. Feasibility pipeline is V1; V2 is learned pipeline.

**Recommendation**: (1). The stale dependency is a reproducibility defect. Fix it in V2; V1 feasibility pipeline is frozen.

---

## P9: Venue Target [PI DECISION REQUIRED]

**Options**: ICLR 2027 (deadline ~Sep 2026, unconfirmed), TMLR (rolling), workshop first.

**Recommendation**: TMLR for no-deadline pressure. ICLR if timeline allows. Workshop if the result is preliminary.

---

## P10: Negative Result Publication Threshold [PI DECISION REQUIRED]

**Problem**: At what point is a null result sufficiently rigorous to publish?

**Threshold**: If calibration establishes that the methodology *can* detect a causal vortex effect (e.g., whole_state has large effect, vortex doesn't), a null result is informative and publishable. If the methodology itself fails (can't train, can't decompose, can't run interventions), it's not publishable as a null result.

---

## Decision Log

| ID | Question | Status | Deadline |
|----|----------|--------|----------|
| D-P0 | Defect prevalence gate redesign | **Needs PI input** | Before calibration |
| D-P1 | C=1 vs C>1 strategy | **Needs PI input** | Before C05 |
| D-P2 | Confirmatory seed count | Deferred to calibration | After C07-C10 |
| D-P3 | Null family draws | **PI input recommended** | Before C07 |
| D-P4 | Primary null comparison method | **PI input recommended** | Before C07 |
| D-P5 | Manifold model | Deferred to calibration | After C11 |
| D-P6 | Factorial baseline scope | **PI input recommended** | Before C16 |
| D-P7 | V1 pilot execution | **Needs PI input** | Immediate |
| D-P8 | Stale dependency fix | **PI input recommended** | Before V2 code |
| D-P9 | Venue target | Informational | Before paper writing |
| D-P10 | Negative result threshold | Philosophical | Before confirmatory |
