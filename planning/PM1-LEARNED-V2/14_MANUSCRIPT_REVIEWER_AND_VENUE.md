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

**V1 aims for T1-T4. V2 aims for T1-T7. T8-T11 are stretch goals.**

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
- **Required experiment**: Pre-registered contract, hierarchical bootstrap, calibration-based power
- **Possible rebuttal**: Hierarchical bootstrap accounts for within-seed correlation; contract is pre-registration
- **Non-rebuttable**: If calibration shows seed variance >> effect size (underpowered)
- **Score before V2**: 3 (Weak Reject)
- **Score after V2**: 5-6 (Weak Accept) with proper power analysis and pre-registration

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

| Venue | Deadline (Known) | Fit | Task Requirement | Risk |
|-------|:---:|:---:|:---:|------|
| NeurIPS | TBD (May 2027?) | Good | No explicit N-task requirement in CFP | High competition |
| ICML | TBD (Jan 2027?) | Good | No explicit N-task requirement | Moderate fit |
| ICLR | TBD (Sep 2026?) | Good | No explicit N-task requirement | Good fit |
| TMLR | Rolling | Good | No explicit N-task requirement | Low time pressure |
| Distill / similar | Rolling | Excellent (visual, mechanistic) | None | Limited venue prestige |
| Workshop (NeurIPS/ICML) | Varies | Good for preliminary results | Low bar | Limited archival value |

**Recommendation**: Target ICLR 2027 or TMLR. The single-phenomenon, single-task design is appropriate for venues that accept focused mechanistic studies. ICLR has historically accepted such work. TMLR provides rolling submission with no deadline pressure.

**Task requirement note**: No CFP for these venues mandates multiple tasks for mechanistic interpretability papers. The "3-task rule" is community heuristic for applied ML papers claiming general-purpose improvements. This paper claims a mechanistic discovery about one architecture on one task — additional tasks would strengthen the claim but are not a submission requirement.

---

## Manuscript Strategy

### Default Claim Tier for V2: T1-T6
"U(1)-equivariant ConvRNNs develop causally relevant vortex defects that carry task-specific information, surpassing nuisance components in intervention efficacy, with representative-invariant and on-manifold effects."

### If V2 is successful but limited: T1-T4
"U(1)-equivariant ConvRNNs develop vortex defects that causally influence copy task output, with stronger effects than matched nuisance controls."

### If V2 fails (negative result): Publish as "No Evidence for Causal Vortex Computation"
This is a valid contribution — the Iqbal et al. paper hypothesizes causal relevance but doesn't establish it. A rigorous null result with the 13-arm design would be publishable even if negative.
