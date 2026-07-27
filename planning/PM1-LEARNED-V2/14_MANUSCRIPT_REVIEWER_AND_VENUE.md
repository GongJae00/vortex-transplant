# Manuscript, Reviewers, and Venue Analysis

## Claim Tiers and Required Evidence

Each tier builds on the previous. A paper should claim only what is evidenced.

| Tier | Claim | Required Evidence |
|:----:|-------|-------------------|
| T1 | U(1)-equivariant ConvRNNs develop vortex defects during copy task training | Observational: charge extraction on trained states, prevalence > baseline |
| T2 | Vortex topology differs between U1ConvRNN and PlainConvRNN | Cross-model comparison: density, stability, signed Jaccard, phase locking |
| T3 | Vortex transplant causes behavioral shift (sufficiency) | Intervention: vortex margin > natural_recipient margin |
| T4 | Vortex effect is stronger than nuisance components | Intervention: vortex margin > max(smooth, magnitude, global, zero, random) |
| T5 | Vortex effect is charge-specific (not just any perturbation) | Intervention: vortex > matched zero-charge phase, matched displacement |
| T6 | Vortex effect is representative-invariant | Multiple same-charge representatives with variance decomposition |
| T7 | Vortex effect is on-manifold | Manifold diagnostics: PCA reconstruction, kNN density, relaxation |
| T8 | Vortex removal eliminates behavior (necessity) | Intervention: single-pair annihilation + sham surgery comparison |
| T9 | Vortex effect is specific to output position and sign | Position × sign interaction analysis |
| T10 | Vortex effect transfers to other tasks | Cross-task generalization (reverse copy, etc.) |
| T11 | U(1) equivariance is the causal factor (not complex conv or radial_tanh) | Factorial baseline decomposition |

**V2 aims for T1-T7. T8-T11 are stretch goals.**

---

## Reviewer Simulation

### Reviewer 1: Topological Defects Expert (Physics/Applied Math)
- **Criteria**: Is the topological analysis mathematically sound? Are the claims about U(1) vortices physically correct?
- **Strongest reject reason**: "Channelwise vortices are not genuine topological defects for C>1. The authors must prove the dynamics constrain the state to a phase-locked submanifold."
- **Required experiment**: C=1 gateway, multichannel phase locking analysis (C14)
- **Possible rebuttal**: Show that projected order parameter captures robust winding; show empirical phase locking
- **Non-rebuttable**: If C>1 channels rotate independently, channelwise charge is not topologically protected
- **Score before V2**: 3 (Weak Reject)
- **Score after V2**: 5-6 (Weak Accept) if C=1 or phase locking is demonstrated

### Reviewer 2: Equivariant ML Expert
- **Criteria**: Is the U(1)-equivariant architecture sound? Is the comparison with PlainConvRNN fair?
- **Strongest reject reason**: "The PlainConvRNN differs in multiple ways (linearity, nonlinearity, normalization). This is not a controlled comparison."
- **Required experiment**: Factorial baseline decomposition (C16)
- **Possible rebuttal**: Show that complex convolution alone doesn't produce the effect; radial_tanh alone doesn't
- **Non-rebuttable**: If PlainConvRNN with complex convolution added shows comparable vortex effects
- **Score before V2**: 4 (Borderline)
- **Score after V2**: 6 (Weak Accept) if factorial baseline isolates equivariance

### Reviewer 3: Causal Inference Expert
- **Criteria**: Is the intervention design valid? Are there selection colliders, unmeasured confounders, or identification failures?
- **Strongest reject reason**: "Donor pair selection is post-treatment. The effect estimate is biased. No sensitivity analysis."
- **Required experiment**: Pre-registered selection + random pair sensitivity, all-admissible reporting
- **Possible rebuttal**: Report both selected and all-admissible results; bound selection bias
- **Non-rebuttable**: If selected-pair effect is large but all-admissible effect is zero
- **Score before V2**: 3 (Weak Reject)
- **Score after V2**: 5 (Borderline Accept) with sensitivity analysis

### Reviewer 4: Mechanistic Interpretability Expert
- **Criteria**: Does this advance understanding of how neural networks compute? Is the method generalizable?
- **Strongest reject reason**: "Vortex transplant resembles activation patching. The faithfulness concerns from Makelov et al. apply here too."
- **Required experiment**: Manifold validity check, relaxation post-intervention
- **Possible rebuttal**: Show that vortex transplant stays on-manifold; patching critiques address subspace choice, not topology
- **Non-rebuttable**: If most intervened states are off-manifold
- **Score before V2**: 4 (Borderline)
- **Score after V2**: 6 (Weak Accept) with manifold controls

### Reviewer 5: Statistics Expert
- **Criteria**: Are the statistical methods appropriate? Is the sample size adequate? Are multiplicity corrections applied?
- **Strongest reject reason**: "Seed-level N is too small. Recipient-level analysis inflates Type I error. No pre-registration."
- **Required experiment**: Pre-registered contract, IUT with hierarchical bootstrap, SESOI-based power, pattern-mixture missingness analysis
- **Possible rebuttal**: IUT controls global Type I error without multiplicity penalty; seed-level resampling accounts for within-seed correlation; frozen contract is pre-registration
- **Non-rebuttable**: If calibration shows seed variance >> SESOI (underpowered)
- **Score before V2**: 3 (Weak Reject)
- **Score after V2**: 5-6 (Weak Accept) with proper IUT, SESOI power, and missingness handling

### Reviewer 6: Reproducibility Expert
- **Criteria**: Can the results be reproduced? Are artifacts complete? Is the code clean?
- **Strongest reject reason**: "Stale external dependency breaks clean-room replay. No CI. No environment lock."
- **Required experiment**: Remove stale dependency, add CI, document environment
- **Possible rebuttal**: All artifacts are write-once with SHA-256 manifest; clean-room instructions provided
- **Non-rebuttable**: If V2 retains stale external dependency
- **Score before V2**: 3 (Weak Reject)
- **Score after V2**: 6-7 (Accept) with clean reproducibility

---

## Venue Analysis

### Criteria for Venue Selection

1. Accepts mechanistic interpretability + novel methodology
2. Does not require multiple benchmark datasets (this is a single-task, single-phenomenon study)
3. Peer review quality (not desk-rejected for scope)
4. Timeline feasibility

### Official vs. Heuristic Deadlines

The deadline table below distinguishes **official_deadline** (published by the venue in a confirmed CFP) from **heuristic_projection** (estimated from historical patterns with no confirmed CFP). Heuristic projections are labeled `heuristic_projection` and must not be treated as confirmed deadlines.

| Venue | Official Deadline | Heuristic Projection | Type | Fit | Task Requirement | Risk |
|-------|:---:|:---:|:---:|:---:|:---:|------|
| NeurIPS 2027 | UNKNOWN | May 2027 | Conference | Good | No explicit N-task requirement | High competition; CFP not yet released |
| ICML 2027 | UNKNOWN | Jan 2027 | Conference | Good | No explicit N-task requirement | Moderate fit; CFP not yet released |
| ICLR 2027 | UNKNOWN | Sep 2026 | Conference | Good | No explicit N-task requirement | Good fit; **most recent confirmed CFP was ICLR 2026** (deadline Sep 2025); ICLR 2027 CFP not published as of July 2026 |
| TMLR | Rolling | — | Journal | Good | No explicit N-task requirement | Low time pressure; no deadline risk |
| NeurIPS/ICML Workshop 2026–2027 | Varies by workshop | Late 2026 / Early 2027 (dependent on parent conference CFP) | Workshop | Good for preliminary results | Low bar | Limited archival value; dependent on parent conference timeline |

### Removed Venues

| Venue | Reason for Removal |
|-------|-------------------|
| Distill | Indefinite hiatus. Distill has not published new articles since 2021 and does not accept submissions. The journal's editorial team has publicly acknowledged the hiatus. It is not a viable submission target. |

### Task Requirement Note

No CFP for these venues mandates multiple tasks for mechanistic interpretability papers. The "3-task rule" is a community heuristic for applied ML papers claiming general-purpose improvements. This paper claims a mechanistic discovery about one architecture on one task — additional tasks would strengthen the claim but are not a submission requirement.

---

## Venue Recommendation Matrix

The appropriate venue depends on the **achieved claim tier** at the conclusion of confirmatory experiments. Submitting above the evidenced tier risks desk rejection; submitting below wastes evidential strength.

### Conditional Recommendations

| Achieved Claim Tier | Primary Recommendation | Rationale | Fallback |
|:---:|---------|-----------|----------|
| T1–T4 only | Workshop (ICLR/NeurIPS/ICML) or short-format venue | T4 confirms vortex > nuisances but without representative-invariance and manifold validity gates, the mechanistic specificity claim is incomplete. Workshop submission allows community feedback while additional evidence is gathered. | TMLR (if manifold and representative gates are partially met) |
| T1–T5 | TMLR rolling submission | T5 establishes charge specificity. The rolling format removes deadline pressure, allowing careful revision. The missing representative (T6) and manifold (T7) gates limit reviewer confidence but do not invalidate the core claim. | Workshop |
| T1–T6 | TMLR or ICLR 2027 (if CFP confirmed) | T6 adds representative invariance, closing a major validity concern. At this tier, the paper has sufficiency, specificity, and invariance evidence. ICLR historically accepts focused mechanistic studies. | TMLR |
| T1–T7 (V2 target) | ICLR 2027 (if CFP confirmed) or TMLR | Full V2 scope: sufficiency, specificity, representative invariance, and manifold validity. The paper makes a complete mechanistic case. ICLR or TMLR are both appropriate. | ICML 2027 (if CFP) or NeurIPS 2027 (if CFP) |
| T1–T8 | Any top venue (ICLR, ICML, NeurIPS, TMLR) | T8 adds necessity evidence, elevating the paper from "vortices are sufficient" to "vortices are necessary." This is a significantly stronger claim warranting any top venue. | — |
| T1–T11 | Any top venue; strong candidate for oral/podium | Full factorial decomposition isolates U(1) equivariance as the causal factor. The paper establishes a new causal mechanism in neural computation. | — |

### Recommendation Decision Rule

```
IF achieved_tier ≤ T4:
    RECOMMEND = Workshop or short-format
ELSE IF achieved_tier == T5:
    RECOMMEND = TMLR (rolling)
ELSE IF achieved_tier in {T6, T7}:
    IF ICLR_2027_CFP_confirmed:
        RECOMMEND = ICLR 2027
    ELSE:
        RECOMMEND = TMLR
ELSE:  # T8+
    RECOMMEND = Best-fitting top venue with earliest feasible deadline
```

### Default V2 Strategy

Given V2 aims for T1-T7, the recommended strategy is:

1. **Primary target**: TMLR rolling submission (no deadline risk)
2. **Opportunistic**: ICLR 2027, IF AND ONLY IF the official CFP is published with a feasible deadline and the paper reaches T1-T7 before the deadline
3. **Fallback**: If T1-T4 only, submit to a workshop and continue toward T5-T7 for a subsequent TMLR submission

**Critical constraint**: Do NOT rush confirmatory experiments to meet a heuristic deadline projection. The statistical validity of the experiment (frozen contract, adequate seed count from SESOI-based power analysis, no peeking) takes priority over any submission timeline.

---

## Manuscript Strategy

### Achieved Claim Tier Scenarios

#### Scenario A: Full V2 Success (T1-T7)
**Claim**: "U(1)-equivariant ConvRNNs develop causally relevant vortex defects that carry task-specific information, surpassing all comparable nuisance perturbations in intervention efficacy, with representative-invariant and on-manifold effects."

**Venue**: ICLR 2027 (if CFP confirmed) or TMLR rolling.

#### Scenario B: Partial V2 Success (T1-T5)
**Claim**: "U(1)-equivariant ConvRNNs develop vortex defects that causally influence copy task output. The vortex effect surpasses all matched nuisance controls and is charge-specific, but representative invariance and manifold validity remain to be fully established."

**Venue**: TMLR rolling submission.

#### Scenario C: Limited V2 Success (T1-T4)
**Claim**: "U(1)-equivariant ConvRNNs develop vortex defects that causally influence copy task output, with stronger effects than matched nuisance controls. Representative and manifold gates were not met or not tested."

**Venue**: Workshop submission or short-format venue. Results are valid but the mechanistic claim is preliminary.

#### Scenario D: Negative Result
**Claim**: "No Evidence for Causal Vortex Computation in U(1)-Equivariant ConvRNNs."

This is a valid contribution. The Iqbal et al. paper hypothesizes causal relevance but does not establish it. A rigorous null result with the 10-arm IUT design and comprehensive missingness analysis would be publishable even if negative.

**Venue**: TMLR (rolling, negative results welcome) or workshop.

### If V2 fails at a specific gate

| Failed Gate | Manuscript Strategy |
|-------------|-------------------|
| IUT does not reject (C1 FALSE) | Publish as negative result; report which families the vortex failed to beat; this is scientifically informative |
| Representative gate fails (C2 FALSE) | Claim limited to T1-T5; acknowledge gauge-dependence as a limitation; suggest future work on gauge-invariant interventions |
| Manifold gate fails (C3 FALSE) | Claim limited to T1-T5; acknowledge off-manifold concern; suggest improved manifold modeling |
| Analyzable fraction too low (C4 FALSE) | Report calibration as exploratory; fix pipeline; increase seeds; do not publish |
| PlainConvRNN also passes IUT (C6 FALSE) | Claim limited to T1-T5; acknowledge that vortex effects are not U(1)-specific; the paper becomes a comparative study of vortex phenomena |
