# Statistical Analysis Plan (SAP)

## 8.1 Data Hierarchy

```
Architecture type (U1, Plain) — fixed, 2 levels
  └─ Task (copy) — fixed, 1 level
      └─ Seed (1..N) — independent randomization unit
          └─ Recipient example (1..M) — within-seed
              └─ Donor (1..K) — within-recipient
                  └─ Arm (13) — within-recipient-donor
                      └─ Output position (1..4) — within-arm
```

**The seed is the generalization unit**. Within-seed examples are correlated via shared model parameters. Treating recipients as independent inflates the effective sample size.

---

## 8.2 Splits

Given that V1 used seeds 0-9, V2 needs fresh namespaces:

| Split | Seeds | Purpose | Constraints |
|-------|-------|---------|-------------|
| Development | dev namespace (separate from V1) | Iterative experimentation | Results NOT used in any statistical test |
| Calibration | `cal_0`...`cal_N_cal` | Estimate variances, tune thresholds | Results used ONLY to set V2 parameters, NOT as evidence |
| Confirmatory | `confirm_0`...`confirm_N_confirm` | Primary hypothesis test | No-peeking; results evaluated by frozen contract only |

**Data namespace rule**: All task generation, model initialization, and training batch uses hash-separated namespaces with split-specific prefixes. Example: `"cal/hash_seed_0"`, `"confirm/hash_seed_0"`.

**Frozen at confirmatory start**: Model architecture, training hyperparameters, PCA basis, manifold model, donor selection function, primary statistic, rejection threshold, all threshold values.

---

## 8.3 Primary Statistical Test

### Candidate Comparison

| Test | Pros | Cons | Recommendation |
|------|------|------|:---:|
| Seed-level paired t-test (U1-Plain) | Simple, interpretable | Assumes normality; few seeds | Sensitivity only |
| Seed-level Wilcoxon signed-rank | Nonparametric | Low power with few seeds | Sensitivity only |
| Hierarchical bootstrap | No distribution assumption; accounts for hierarchy | Computationally heavy | **Recommended primary** |
| Random-effects model (lme4-style) | Explicit variance decomposition | Complex specification | Secondary |
| Sign-flip permutation | Exact under null of zero median | Few seeds → discrete p-values | Sensitivity only |

### Recommended Primary: Hierarchical Bootstrap

1. Resample seeds with replacement
2. Within each resampled seed, resample recipients with replacement
3. Within each recipient, resample donors with replacement
4. Compute the mechanism advantage (seed-level mean)
5. Repeat B = 9999 times
6. Compute bootstrap CI for the mean mechanism advantage

**Rejection criterion**: Bootstrap 2.5% CI lower bound > 0 (one-sided or two-sided depending on pre-registration).

---

## 8.4 Power Analysis (Placeholder — Needs Calibration Data)

Simulation inputs (to be populated from calibration):

| Parameter | Symbol | Estimate | Source |
|-----------|--------|----------|--------|
| Seed-level mean effect | \(\mu\) | ? | Calibration |
| Seed-level SD | \(\sigma_s\) | ? | Calibration |
| Within-seed ICC | \(\rho\) | ? | Calibration |
| Invalid pair rate | \(p_{\text{invalid}}\) | ? | Calibration |
| Representative variance fraction | \(\sigma^2_{\text{rep}} / \sigma^2_{\text{charge}}\) | ? | Calibration |
| Multiplicity correction factor | \(k\) | 8 (null families) | Design |

**Simulation procedure** (pseudocode):
```
for n_seeds in range(5, 100, 5):
    n_sig = 0
    for sim in range(1000):
        # Generate synthetic data: n_seeds draws from N(mu, sigma_s)
        # Add within-seed noise, invalid pairs, rep variance
        # Compute hierarchical bootstrap CI
        if CI excludes 0: n_sig += 1
    power = n_sig / 1000
    # Report (n_seeds, power, MDE)
```

**Do not freeze seed count before calibration variance is available.**

---

## 8.5 Multiplicity

8 null families × 2 model types = 16 comparisons.

**Strategy**: Hierarchical gate:
1. **Gate 1 (per-model)**: Mechanism advantage > 0 (vortex > all nuisance). Tested with hierarchical bootstrap.
2. **Gate 2 (cross-model)**: U1 advantage > Plain advantage. Tested with paired bootstrap.
3. **Gate 3 (per-family)**: Which specific families does vortex beat? Reported as secondary, not used for go/no-go.

Bonferroni correction for Gate 3 only (exploratory). Gate 1 and Gate 2 use single primary test.

---

## 8.6 Missingness

### Invalid Pair Reasons
1. `NO_DECOM_PAIR` — one state has near-zero magnitude or charge neutrality failure
2. `NO_DONOR_SELECTED` — no admissible donor in catalog
3. `NO_MATCHED_NULL` — cannot construct matched control
4. `MANIFOLD_FAILURE` — intervention produces off-manifold state
5. `NUMERICAL_FAILURE` — decomposition or intervention crashes
6. `TASK_FAILURE` — model does not achieve minimum accuracy

### Reporting
- Count each reason per seed
- Complete-case analysis as primary
- Sensitivity: worst-case imputation (missing = null effect)
- Sensitivity: multiple imputation (predict missing effects from observed features)

---

## 8.7 Go/No-Go States

| State | Condition | Action |
|-------|-----------|--------|
| `GO_CONFIRMATORY` | All calibration gates pass, contract frozen | Run confirmatory experiment |
| `GO_REDESIGN` | Phenomenon exists but mechanism insufficient | Redesign interventions, re-calibrate |
| `NO_GO_PHENOMENON` | No U(1)-specific vortex phenomenon | Publish negative result or pivot |
| `NO_GO_MECHANISM` | Vortices exist but causally irrelevant | Publish negative result |
| `INCONCLUSIVE_BASELINE` | PlainConvRNN also shows causal vortex effects | Redesign baseline or accept weaker claim |
| `INCONCLUSIVE_REPRESENTATIVE` | Effect not robust to gauge choice | Improve representative controls |
| `INCONCLUSIVE_MANIFOLD` | Effect depends on off-manifold artifacts | Improve manifold projection |
| `INCONCLUSIVE_STATISTICS` | Underpowered | Increase seeds or accept uncertainty |
| `RESOURCE_NO_GO` | Exceeds computational budget | Reduce scope or acquire resources |

**Crucial**: Competitor model (PlainConvRNN) failing ≠ success. PlainConvRNN failure is a *necessary condition* for the specificity claim, but if PlainConvRNN fails AND the U1 effect is small, the result is `INCONCLUSIVE_BASELINE`, not `GO`.
