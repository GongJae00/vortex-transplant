# Causal Identification Framework

## 7.1 Structural Causal Model (SCM)

```
Input X ──► Trained parameters θ
  │              │
  │              ▼
  │         Natural hidden state H = F(X; θ)
  │              │
  │     ┌────────┼────────┬──────────┐
  │     ▼        ▼        ▼          ▼
  │  Topology  Smooth  Magnitude  Harmonic   Validity(V)
  │   Q(H)     S(H)     M(H)      Hm(H)        │
  │     │        │        │          │          │
  │     ├────────┴────────┴──────────┤          │
  │     │     Representative ρ       │          │
  │     │   (gauge choice for Q)     │          │
  │     └────────────┬───────────────┘          │
  │                  │                          │
  │     ┌────────────┼──────────────┐          │
  │     ▼            ▼              ▼          │
  │  Donor         Donor          Natural      │
  │  Selection(S)  Field(z_d)     Recipient(z_r)│
  │     │            │              │          │
  │     │            ▼              ▼          │
  │     │      Intervention       Natural       │
  │     │    z_int = I(z_d,z_r)   Continuation  │
  │     │            │              │          │
  │     └────────────┼──────────────┘          │
  │                  ▼                          │
  │         Continuation H' = F_cont(z_int)   │
  │                  │                          │
  │                  ▼                          │
  └──────────────► Output Y = R(H')            │
```

### Selection Collider Warning

The donor pair is selected based on hidden-state geometry (signed count difference, energy, spectrum). This selection conditions on a post-treatment variable:
- If donor selection is correlated with the outcome (e.g., more "computationally active" pairs produce larger effects), the estimated intervention effect is biased.
- **Mitigation**: Pre-register the donor selection criteria. Report both "selected pairs" and "all-admissible pairs" effects. The primary analysis uses pre-registered selection; sensitivity uses all-admissible.

---

## 7.2 Estimands

### Primary: Mechanism Advantage
\[
\Delta_{\text{mech}} = M_{\text{vortex}} - \max(M_{\text{smooth}}, M_{\text{magnitude}}, M_{\text{global}}, M_{\text{zero}}, M_{\text{random}})
\]
where \(M_a = \text{mean}(\text{donor\_ll} - \text{recipient\_ll})\) for arm \(a\).

**Interpretation**: How much more does vortex transplantation shift behavior toward the donor, compared to the best nuisance baseline?

### Secondary: Normalized Recovery
\[
R_a = \frac{M_a - M_{\text{NR}}}{M_{\text{WS}} - M_{\text{NR}}}
\]
where NR = natural_recipient (no intervention), WS = whole_state (complete donor state).

**Edge cases requiring specification**:
1. \(M_{\text{WS}} - M_{\text{NR}} \approx 0\) → denominator near zero. Implies either: (a) donor and recipient produce identical outputs (no information difference to recover), or (b) whole-state intervention is ineffective (architecture limitation).
   - **Decision**: Mark pair as `INVALID_NORMALIZATION` and exclude from \(R_a\) analysis.
2. \(M_{\text{WS}} - M_{\text{NR}} < 0\) → whole-state sanity failure. The complete donor state should produce donor-like outputs; if it doesn't, the model or task is broken.
   - **Decision**: Mark seed as `FAILED_DIRECTIONAL_SANITY` if >10% of pairs have this.
3. \(R_a > 1\) → vortex intervention produces stronger effect than whole-state transplant. This is *possible* (vortex alone might be a purer information carrier than noisy full state) but needs justification.
   - **Decision**: Report as-is, flag as "vortex > whole_state" pairs.

---

## 7.3 Intervention Taxonomy

### Sufficiency Arms
Does transplanting the vortex *suffice* to shift behavior?
- `vortex`: canonical vortex field from donor → recipient
- `vortex_alternate`: different same-charge representative
- `vortex_minimal`: minimal-surgery vortex (smallest displacement for same charge)

### Necessity Arms
Does *removing* the vortex eliminate donor-specific behavior?
- `vortex_remove_all`: replace vortex with \(v_0\) (charge-neutral field)
- `vortex_remove_pair`: annihilate a single defect pair
- `vortex_sham`: same-displacement surgery with no charge change

### Specificity Arms
Is the effect specific to *particular* vortex properties?
- Sign flip: \(Q \to -Q\)
- Spatial shift: translate vortex pattern
- Density change: add or remove charges

### Representative Arms
Is the effect invariant to the specific gauge choice?
- Same-charge, different representative (multiple samples)
- Representative variance decomposition

### Mechanistic Baselines
Do other hidden-state features carry similar information?
- Fourier low-pass / high-pass
- PCA projection
- random_direction (norm-matched null)
- smooth component transplant
- magnitude component transplant

### Harmonic / Competing Topology
Could the "vortex" effect actually be a harmonic sector effect?
- Harmonic sector swap (same vortex, different harmonic)
- Mixed charge-count matched random arrangement

### Natural Controls
Is the intervention on-manifold?
- Natural neighbor (nearest natural hidden state with target topology)
- Denoising projection
- Relaxation post-intervention

---

## 7.4 Patch Efficacy vs Mechanism Faithfulness

| Criterion | Measured By |
|-----------|-------------|
| Intervention changes behavior | \(M_a > 0\) |
| Changes toward *target donor* behavior | \(M_a > M_{\text{NR}}\) |
| Corresponds to naturally used variable | Vortex margin > smooth/magnitude margins |
| Stays on-manifold | Manifold distance, relaxation check |
| Is representative-invariant | Variance across same-charge representatives |

A successful claim requires all five, not just the first two.

The core risk: a perturbation that is *large enough* and pushes the state toward the donor's *general neighborhood* will shift outputs. But this doesn't prove that the vortex *per se* is the causal mechanism — it could be any sufficiently large perturbation toward the donor.

---

## 7.5 Null Family Separation

**DO NOT POOL**. Each null family tests a different counterfactual:

| Family | Counterfactual | Expected Under H₀ | p-value computation |
|--------|---------------|-------------------|---------------------|
| random_direction | Any perturbation of this norm shifts behavior | vortex ≤ random | One-sided paired |
| fourier_low | Low-frequency info carries the effect | vortex ≤ fourier_low | One-sided paired |
| fourier_high | High-frequency info carries the effect | vortex ≤ fourier_high | One-sided paired |
| PCA | Top-k variance directions carry the effect | vortex ≤ PCA | One-sided paired |
| smooth | Smooth phase carries the effect | vortex ≤ smooth | One-sided paired |
| magnitude | Amplitude modulation carries the effect | vortex ≤ magnitude | One-sided paired |
| harmonic | Global holonomy carries the effect | vortex ≤ harmonic | One-sided paired |
| same-charge rep | Specific gauge choice, not charge, matters | Rep variance >> charge variance | Variance decomposition |

**Primary p-value**: \(\min_{f \in \text{families}} p_f\) with Bonferroni correction. **Reject H₀** (null: vortex ≤ all families) at α = 0.05/8 ≈ 0.00625.

**Secondary**: Hierarchical test: first test vortex > max(null families), then decompose which families it beats.

---

## 7.6 Selection Collider Mitigation

The donor pair selection uses hidden-state features (charge count, energy, spectrum). If these features correlate with "how much behavior will shift," the estimated effect on selected pairs is biased upward.

**Mitigation strategy**:
1. Pre-register the selection function (already frozen in V1)
2. Report effect on *all admissible pairs* (not just selected)
3. Sensitivity: random pair selection vs geometric selection comparison
4. Sensitivity: worst-case selection via propensity-score stratification
