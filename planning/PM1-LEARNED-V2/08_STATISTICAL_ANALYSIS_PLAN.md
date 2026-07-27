# Statistical Analysis Plan (SAP)

## 8.1 Data Hierarchy

```
Architecture type (U1, Plain) — fixed, 2 levels
  └─ Task (copy) — fixed, 1 level
      └─ Seed (1..N_confirm) — independent randomization and generalization unit
          └─ Recipient example (1..M) — within-seed, correlated via shared model parameters
              └─ Donor (1..K) — within-recipient
                  └─ Null family (1..10) — within-recipient-donor
                      └─ Output position (1..4) — within-arm
```

**The seed is the sole generalization unit.** Each seed is an independent trained model instance. Within-seed recipients are correlated through shared model parameters θ_s. Treating recipients or donors as independent observations inflates the effective sample size and produces anti-conservative confidence intervals.

**Variance decomposition (conceptual):**
```
σ²_total = σ²_seed + σ²_recipient:seed + σ²_residual
```

All inference operations (bootstrap resampling, p-value computation, CI construction) must treat the seed as the atomic resampling unit. Recipient-level resampling is prohibited in the primary analysis.

---

## 8.2 Splits

| Split | Seeds | Purpose | Constraints |
|-------|-------|---------|-------------|
| Development | `dev` namespace (separate from V1 and V2 confirmatory) | Iterative experimentation, debugging | Results NOT used in any statistical test; NOT used for threshold setting |
| Calibration | `cal_0`...`cal_N_cal` | Estimate variances, tune intervention thresholds, power simulation inputs | Results used ONLY to set V2 contract parameters, NOT as evidence for claims |
| Confirmatory | `confirm_0`...`confirm_N_confirm` | Primary hypothesis test | No-peeking; results evaluated by frozen contract only |

**Data namespace rule**: All task generation, model initialization, and training batches use hash-separated namespaces with split-specific prefixes. Examples: `"cal/hash_seed_0"`, `"confirm/hash_seed_0"`.

**Frozen at confirmatory start**: Model architecture, training hyperparameters, PCA basis, manifold model, donor selection function, primary statistic, rejection threshold (α = 0.05), all threshold values (τ_rep, manifold quantile).

---

## 8.3 Primary Statistical Test: Intersection-Union Test (IUT)

### 8.3.1 Hypothesis Structure

The scientific claim is: **vortex effect exceeds every comparable null family effect simultaneously.**

```
H₀ = ∪_{f ∈ F} H₀_f    where H₀_f: δ_f ≤ 0    (vortex fails to beat at least one null family)
H₁ = ∩_{f ∈ F} H₁_f    where H₁_f: δ_f > 0     (vortex beats every null family)
```

where δ_f = E[M_vortex − M_f] is the seed-level expectation of the vortex-minus-null-family-f margin, and F = {smooth, magnitude, global_phase, zero_charge_phase, fourier_low, fourier_high, pca, random_direction, harmonic, charge_arrangement_shuffle}.

This is the Intersection-Union Test (IUT; Berger 1982). H₀ is true if **any** null family performs as well or better than vortex.

### 8.3.2 Rejection Rule

For each null family f, compute a one-sided p-value p_f testing H₀_f: δ_f ≤ 0.

```
IUT rejection rule: Reject H₀  ⇔  max_{f ∈ F} p_f ≤ α
```

Equivalently: reject H₀ if p_f ≤ α for **all** f ∈ F.

**Use max(p_f), never min(p_f).** min(p_f) tests the wrong composite hypothesis (vortex beats at least one null), which corresponds to H₀ for a union-of-intersections, not an intersection-of-unions.

### 8.3.3 Why the IUT Does Not Need Multiplicity Adjustment

The IUT controls the Type I error rate at level α without any correction:

```
P_H₀(reject) = P_H₀(∩_{f} {p_f ≤ α}) ≤ P_H₀(p_f* ≤ α) ≤ α
```

where f* is any family for which δ_f* ≤ 0 under H₀ (at least one such family exists). The IUT's "cost" is the conjunction requirement itself — vortex must beat every family individually at level α, which is a strict standard in its own right. Bonferroni correction would be unnecessarily conservative and is not applied.

### 8.3.4 Per-Family p-value Computation

For each null family f, the per-family test is:

```
H₀_f: δ_f ≤ 0    vs.    H₁_f: δ_f > 0
```

Procedure (hierarchical bootstrap, seed-level resampling):

1. Compute the per-seed statistic: for each seed s, compute the seed-level mean vortex-minus-null-f margin δ̂_f,s across all analyzable recipient-donor pairs.
2. Resample B = 9999 times at the seed level:
   - Draw N seeds with replacement from {s₁, …, s_N}
   - Compute the resampled mean: δ̂_f^*(b) = mean_{s in resample} δ̂_f,s
3. Compute the one-sided bootstrap p-value:
   ```
   p_f = (1 / (B + 1)) * (1 + Σ_{b=1..B} I[δ̂_f^*(b) ≤ 0])
   ```
4. IUT global rejection: max_f p_f ≤ α.

**Alternative: Simultaneous confidence intervals via max-T bootstrap (optional).** For simultaneous 95% one-sided CIs that automatically account for between-family dependence:

1. For each bootstrap replicate b, compute the studentized maximum:
   ```
   t_max^*(b) = max_{f ∈ F} (δ̂_f^*(b) − δ̂_f) / σ̂_f
   ```
2. Take the 95th quantile c_{0.95} of {t_max^*(b)}.
3. Simultaneous one-sided CI for each family: [δ̂_f − c_{0.95} · σ̂_f, ∞).
4. Reject H₀ if all simultaneous CIs exclude 0.

### 8.3.5 Cross-Model Comparison (Secondary)

For the secondary claim that U1ConvRNN vortex effect exceeds PlainConvRNN vortex effect:

```
H₀_cross: δ_U1 − δ_Plain ≤ 0
```

Test: hierarchical bootstrap on the paired per-seed differences (U1_seed_i − Plain_seed_i), one-sided, p_cross ≤ α.

### 8.3.6 Primary Test Summary

| Component | Specification |
|-----------|--------------|
| Global null | H₀ = ∪_f {δ_f ≤ 0} |
| Global test | Intersection-Union Test |
| Rejection rule | max_f p_f ≤ α |
| α | 0.05 |
| Per-family test | One-sided hierarchical bootstrap |
| Bootstrap resamples | B = 9999 |
| Resampling level | Seed (generalization unit) |
| Multiplicity adjustment | None (IUT is self-adjusting) |
| CI method | Bootstrap percentile (two-sided 95% reported; one-sided for decision) |
| Minimum B for p-value stability | 1999 (B = 9999 recommended) |

---

## 8.4 Power Analysis

### 8.4.1 Smallest Effect Size of Scientific Interest (SESOI)

Power is computed for the **SESOI**, not for 0.5 × the calibration effect. The SESOI is the minimum vortex-over-null-family advantage that would constitute a scientifically meaningful result.

The SESOI is a **domain judgment**, not a data-derived quantity. It must be set by the PI before confirmatory experiments begin, informed by (but not determined by) calibration data.

| Parameter | Symbol | Source |
|-----------|--------|--------|
| SESOI (per-family minimum δ_f) | δ_SESOI | PI domain judgment, informed by calibration distributions |
| Target power | 1 − β | 0.80 (conventional minimum) |
| Significance level | α | 0.05 |
| Null families | \|F\| | 10 |

**Anti-pattern**: Do NOT set SESOI = 0.5 × observed calibration effect. This would be post-hoc and circular.

**Acceptable SESOI calibration**: The PI reviews calibration effect-size distributions (mean, variance, quantiles) across null families and declares, before seeing confirmatory results: "A mean vortex-over-f advantage below X nats per position would not constitute an interesting finding."

### 8.4.2 Simulation-Based Power Estimation

Power is estimated by simulation, not by closed-form formula. Formula-based power assumes normality, independence, and balanced design — none of which hold in this setting.

**Simulation algorithm:**

```
for n_seeds in candidate_range:  # e.g., 10, 15, 20, …, 100
    n_rejections = 0
    for sim in 1..N_sim:         # N_sim = 2000
        # Generate synthetic per-seed statistics:
        # For each family f, draw δ̂_f,seed ~ D_f(θ_cal) with true mean δ_SESOI
        #   where D_f captures seed-level variance, within-seed ICC,
        #   and missingness patterns estimated from calibration
        # The generating distribution preserves:
        #   - Seed-level variance σ²_seed,f (from calibration)
        #   - Between-family correlation Σ (from calibration)
        #   - Within-seed sample size distribution (from calibration)
        #   - Missingness rate ψ_analyze,f (from calibration)

        # Run IUT:
        #   Compute per-family bootstrap p-values p_f
        #   if max_f(p_f) ≤ α: n_rejections += 1

    power = n_rejections / N_sim
    report(n_seeds, power, SESOI)
```

**Minimum required n_seeds**: The smallest n_seeds such that power ≥ 0.80 at SESOI.

**Sensitivity**: Re-run power simulation at 0.8 × SESOI and 1.2 × SESOI to characterize the power curve.

**Reporting**: Report the full power curve (n_seeds vs. power at SESOI), not a single number.

### 8.4.3 Calibration Inputs for Power Simulation

| Parameter | Symbol | Estimate Source | Status |
|-----------|--------|-----------------|--------|
| Seed-level per-family variance | σ²_seed,f | C01–C04 calibration | Required |
| Between-family correlation matrix | Σ | Calibration null family runs | Required |
| Per-family analyzable fraction | ψ_analyze,f | Calibration selection funnel | Required |
| Within-seed recipient-donor count distribution | N_pair(s) | Calibration selection funnel | Required |
| Representative variance fraction | σ²_rep / σ²_charge | C07 calibration | Required |
| Manifold distance distribution | D(D_nat) | C11 calibration | Required |

**Do not freeze the confirmatory seed count before all calibration rows above are populated.**

---

## 8.5 Multiplicity

### 8.5.1 Primary Global Test

The IUT is the primary global test and does **not** require multiplicity adjustment across the 10 null families (see §8.3.3). The IUT's conjunction requirement (vortex must beat every family) is the substantive claim; no further correction is needed.

### 8.5.2 Secondary Per-Family Claims

After the global IUT rejects, per-family claims are secondary and descriptive. If per-family p-values are reported as confirmatory secondary claims (not exploratory), apply Bonferroni correction across the k_fam = 10 families within each model type:

```
α_secondary = α / k_fam = 0.05 / 10 = 0.005
```

Per-family p-values ≤ 0.005 are reported as "significant after Bonferroni correction for 10 null families."

### 8.5.3 Cross-Model Comparison

The cross-model test (U1 vs. Plain) is a separate pre-registered secondary hypothesis and is tested at α = 0.05 without additional correction (it is a single test, not a family).

If multiple cross-model comparisons are tested (e.g., per-family U1 vs. Plain), apply Bonferroni across those comparisons.

### 8.5.4 OOD Generalization

The OOD generalization test (delay=64 vs. training delays) is a separate pre-registered secondary hypothesis, tested at α = 0.05 without correction.

### 8.5.5 Summary

| Claim | Test | α level | Correction |
|-------|------|:-------:|------------|
| Vortex > all nulls (global) | IUT: max p_f ≤ α | 0.05 | None (IUT is self-adjusting) |
| Per-family superiority (secondary) | One-sided bootstrap per family | 0.005 | Bonferroni (k=10) |
| U1 > Plain (cross-model) | Paired bootstrap | 0.05 | None (single test) |
| OOD generalization | Paired bootstrap | 0.05 | None (single test) |
| Trained > untrained | Two-sided bootstrap | 0.05 | None (single test) |
| All other exploratory | Bootstrap, nominal p | — | Report as exploratory; no correction claimed |

---

## 8.6 Missingness and Analyzability

### 8.6.1 Invalid Pair Reasons

| Code | Description | Classification |
|------|-------------|:---:|
| `NO_DECOM_PAIR` | One state has near-zero magnitude or charge neutrality failure | Not analyzable |
| `NO_DONOR_SELECTED` | No admissible donor in catalog for this recipient | Not analyzable |
| `NO_MATCHED_NULL` | Cannot construct matched control for a null family | Not analyzable |
| `MANIFOLD_FAILURE` | Intervention produces off-manifold state (d_M(H') > Q_{0.95}(D_nat)) | Not analyzable |
| `NUMERICAL_FAILURE` | Decomposition or intervention throws a numerical error | Not analyzable |
| `TASK_FAILURE` | Model does not achieve minimum accuracy (0.95) | Not analyzable |
| `DEGENERATE_DISTRIBUTION` | Donor/recipient output distribution has near-zero entropy | Not analyzable |

### 8.6.2 Estimands Under Missingness

Complete-case analysis (analyze only pairs that pass all gates) is **not acceptable as primary without qualification**. Three estimands must be reported:

#### A. Selected-Subpopulation Estimand (Primary – but qualified)

```
Δ_mech = E_{r∈R(s), d∈D_S(r)}[M_vortex(r,d) − max_f M_f(r,d) | A=1]
```

where A = 1 denotes an analyzable pair and D_S is the frozen donor selection function. This is the primary estimand but **must be accompanied by the analyzable fraction and the composite estimand**.

#### B. Analyzability Estimand

```
ψ_analyze(s) = E[A | s]    (per-seed analyzable fraction)
ψ_analyze    = E_s[ψ_analyze(s)]   (overall analyzable fraction)
```

Report ψ_analyze per seed and overall. If ψ_analyze < 0.70, the selected-subpopulation estimand is not credible on its own. If ψ_analyze < 0.50, the experiment is underpowered regardless of the selected-subpopulation result.

#### C. Failure-as-Non-Supportive Composite Estimand

```
Δ_mech^composite = (1/N_total) · Σ_{all pairs} {
    M_vortex − max_f M_f     if analyzable (A=1)
    0                         if not analyzable (A=0)
}
```

This imputes a null (zero advantage) effect for every non-analyzable pair. If Δ_mech^composite ≤ 0 while Δ_mech > 0, the result depends critically on the analyzability filter.

### 8.6.3 Pattern-Mixture Sensitivity Analysis

Model the joint distribution of (Δ_mech, A) as a mixture:

```
P(Δ_mech, A) = P(A=1) · P(Δ_mech | A=1) + P(A=0) · P(Δ_mech | A=0)
```

P(Δ_mech | A=1) is observed. P(Δ_mech | A=0) is unidentified. Vary assumptions:

| Assumption | P(Δ_mech \| A=0) | Interpretation |
|------------|-------------------|---------------|
| MAR (complete-case) | Same as observed | Primary analysis (strongest assumption) |
| Worst-case | min(0, Δ_mech^(min)) | Non-analyzable pairs show zero or negative advantage |
| Shift-model | Δ_mech \| A=1 − λ | Assume non-analyzable pairs differ by a sensitivity parameter λ ≥ 0 |
| Tipping-point | Δ_c | Find the smallest Δ_c such that the overall conclusion reverses |

Report the **tipping-point** Δ_c: the minimum assumed mean advantage in non-analyzable pairs that would reduce the composite Δ_mech^composite to ≤ 0. If Δ_c is implausibly large (e.g., beyond the observed range), the conclusion is robust to missingness.

### 8.6.4 Differential Missingness by Arm

Test whether missingness rates differ across arms using within-seed paired comparisons. If missingness is differential and associated with arm type, report arm-specific missingness patterns and evaluate whether differential missingness could explain the observed effect.

### 8.6.5 Reporting Requirements

1. **Table S1**: Per-seed analyzable pair count, total pair count, analyzable fraction ψ_analyze(s)
2. **Table S2**: Failure reason counts (7 codes above), per seed and pooled
3. **Primary result**: IUT rejection decision with Δ_mech and 95% CI
4. **Alongside**: Δ_mech^composite, ψ_analyze (overall)
5. **Sensitivity**: Worst-case bounds, tipping-point Δ_c
6. **Arm differential**: Missingness rate per arm, test for differential missingness
7. **Interpretation guideline**: If ψ_analyze < 0.50, declare INCONCLUSIVE_STATISTICS regardless of IUT result

---

## 8.7 Go/No-Go States

### 8.7.1 Decision Conditions

The decision logic combines the IUT result with the separate validity gates and context conditions.

```
C1  = max_f p_f ≤ α                         (IUT: vortex > all null families)
C2  = Var_rep / Var_charge < τ_rep          (representative invariance gate)
C3  = median(d_M(H_adj)) ≤ Q_{0.95}(D_nat)  (manifold validity gate)
C4  = ψ_analyze ≥ 0.70                      (analyzable fraction sufficient)
C5  = ψ_analyze ≥ 0.50                      (analyzable fraction minimally acceptable)
C6  = max_f p_f^(Plain) > α                 (PlainConvRNN IUT does NOT reject; competitor failure)
C7  = p_cross ≤ α                           (U1 vortex advantage > Plain vortex advantage)
C8  = ρ_sel ≥ 0.8                           (selection amplification ratio adequate)
```

### 8.7.2 State Definitions

#### GO States

| State | Condition | Action |
|-------|-----------|--------|
| `GO_CONFIRMATORY` | C1 ∧ C2 ∧ C3 ∧ C4 ∧ C6 ∧ C7 | Confirmatory experiment succeeds: vortex is causally superior, representative-invariant, manifold-valid, and specific to U(1) equivariance. Proceed to manuscript. |
| `GO_WEAK_SPECIFICITY` | C1 ∧ C2 ∧ C3 ∧ C4 ∧ C7 | Confirmatory succeeds on mechanism, but PlainConvRNN also shows vortex superiority (C6 false). The specificity claim is weakened: vortex effects exist in both architectures, but U1 effect is still larger. Claim tier drops to T5. |
| `GO_PARTIAL` | C1 ∧ C2 ∧ C3 ∧ C4 ∧ ¬C7 ∧ C6 | Mechanism is valid and specific (U1-only), but cross-model margin is not statistically significant. The claim is qualitative ("vortex effects in U1, absent in Plain") rather than quantitative. |

#### INCONCLUSIVE States

| State | Condition | Action |
|-------|-----------|--------|
| `INCONCLUSIVE_BASELINE` | C1 ∧ C6 is FALSE ∧ C7 is FALSE | PlainConvRNN also passes IUT (shows vortex > all nulls) AND U1 is not significantly better. Both architectures show comparable vortex effects. Reassess mechanism specificity claim. This is NOT a success: competitor-failure is necessary but not sufficient. |
| `INCONCLUSIVE_COMPETITOR_SUPERIOR` | C1 ∧ C6 is FALSE ∧ C7 is TRUE | U1 beats Plain, but Plain itself passes IUT. Vortex effects are present in both architectures and stronger in U1. Result is interesting but does not support the claim that U(1) equivariance is uniquely causal. |
| `INCONCLUSIVE_REPRESENTATIVE` | C1 ∧ C2 is FALSE | IUT passes but representative variance exceeds threshold. The effect depends on specific gauge choices, not on the topological charge class. |
| `INCONCLUSIVE_MANIFOLD` | C1 ∧ C3 is FALSE | IUT passes but intervened states are off the natural manifold. The behavioral effect may be an off-manifold artifact. |
| `INCONCLUSIVE_STATISTICS` | C1 ∧ C5 is FALSE | Analyzable fraction is too low (ψ_analyze < 0.50). The selected-subpopulation result is not credible. Increase seeds or fix pipeline failures. |
| `INCONCLUSIVE_POWER` | C1 ∧ C4 is FALSE ∧ C5 is TRUE | Marginal analyzable fraction (0.50 ≤ ψ_analyze < 0.70). Result is suggestive but underpowered. Report with strong caveats. |
| `INCONCLUSIVE_SELECTION` | C1 ∧ C8 is FALSE | Selection amplification ratio is low. Selection function does not meaningfully concentrate the effect. Report both selected and all-admissible results. |

#### NO_GO States

| State | Condition | Action |
|-------|-----------|--------|
| `NO_GO_MECHANISM` | C1 is FALSE | IUT does not reject. Vortex fails to surpass at least one null family. The mechanism claim is not supported. |
| `NO_GO_PHENOMENON` | No analyzable vortex topology in trained models | The model does not develop analyzable vortex defects on the copy task. Publish negative result. |
| `NO_GO_BOTH_C1_FAIL` | Both U1 and Plain IUT fail | No evidence of causal vortex effects in either architecture. |

#### RESOURCE States

| State | Condition | Action |
|-------|-----------|--------|
| `RESOURCE_NO_GO` | Exceeds computational budget before confirmatory completes | Reduce scope, acquire resources, or publish calibration results as exploration. |

### 8.7.3 Key Principle: Competitor Failure ≠ Success

PlainConvRNN failing the IUT (C6 is TRUE) is a **necessary condition** for the specificity claim, not a standalone success criterion. The U1 must independently pass IUT (C1) AND show cross-model superiority (C7) AND pass validity gates (C2, C3) AND have adequate analyzable fraction (C4).

If PlainConvRNN fails the IUT but U1 also fails (C1 FALSE), the result is `NO_GO_MECHANISM`, not `INCONCLUSIVE_BASELINE`. The competitor failure alone does not salvage a failed primary test.

If PlainConvRNN passes the IUT (C6 FALSE), the result is `INCONCLUSIVE_BASELINE` (or variants), regardless of U1 performance. The scientific claim requires U(1)-specificity.

### 8.7.4 Combined Decision Logic (Formal)

```
IF ¬C5:
    VERDICT = INCONCLUSIVE_STATISTICS
    EXIT

IF ¬C1:
    IF ¬C1_U1 AND ¬C1_Plain:
        VERDICT = NO_GO_BOTH_C1_FAIL
    ELSE:
        VERDICT = NO_GO_MECHANISM
    EXIT

# C1 is TRUE: IUT passes for at least U1
IF ¬C2: VERDICT = INCONCLUSIVE_REPRESENTATIVE; EXIT
IF ¬C3: VERDICT = INCONCLUSIVE_MANIFOLD; EXIT
IF ¬C4: VERDICT = INCONCLUSIVE_POWER; EXIT

# All validity gates pass
IF ¬C6:  # PlainConvRNN IUT also rejects
    IF C7: VERDICT = INCONCLUSIVE_COMPETITOR_SUPERIOR
    ELSE:  VERDICT = INCONCLUSIVE_BASELINE
ELSE:  # PlainConvRNN IUT does NOT reject (competitor failure)
    IF C7: VERDICT = GO_CONFIRMATORY
    ELSE:  VERDICT = GO_PARTIAL
```

---

## 8.8 Analysis Pipeline Checklist

At confirmatory analysis time, execute in order:

1. [ ] Verify confirmatory namespace separation (no cal/confirm hash collision)
2. [ ] Verify frozen contract hash matches analysis commit
3. [ ] Load all confirmatory seeds; report per-seed participant flow (accuracy → magnitude → charge → decomposition → donor → manifold)
4. [ ] Compute ψ_analyze per seed and overall; if ψ_analyze < 0.50, STOP: INCONCLUSIVE_STATISTICS
5. [ ] Compute Δ_mech per seed per null family
6. [ ] Run hierarchical bootstrap (B = 9999, seed-level resampling) for each null family
7. [ ] Compute p_f for all f ∈ F; compute IUT decision: max_f p_f ≤ 0.05?
8. [ ] Run representative variance decomposition; check C2
9. [ ] Run manifold distance gate; check C3
10. [ ] Run cross-model comparison (U1 vs. Plain)
11. [ ] Compute Δ_mech^composite and tipping-point Δ_c
12. [ ] Compute pattern-mixture sensitivity bounds
13. [ ] Compute ρ_sel (selection amplification ratio)
14. [ ] Apply Go/No-Go decision logic (§8.7.4)
15. [ ] Report all results, regardless of outcome

---

## 8.9 Summary of Critical Statistical Safeguards

| Safeguard | Mechanism |
|-----------|-----------|
| No peeking at confirmatory results | Frozen contract, separate namespace |
| Seed-level generalization | Bootstrap and test statistics computed per seed |
| IUT for conjunction claim | max(p_f) ≤ α, NOT min(p_f) |
| No Bonferroni on primary test | IUT is self-adjusting; Bonferroni for secondary only |
| Power via SESOI, not 0.5 × calibration | PI-set SESOI, simulation-based power curve |
| Missingness transparency | Three estimands (selected, composite, analyzability) + pattern-mixture + tipping-point |
| Competitor failure gated correctly | C6 is necessary, not sufficient; explicit INCONCLUSIVE states |
| Validity gates separate | Representative invariance and manifold validity are mandatory hurdles, not pooled in IUT |
| All results reported | No cherry-picking; negative results are publishable |
