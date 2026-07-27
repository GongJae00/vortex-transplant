# Causal Identification Framework (인과 식별 체계)

## 7.1 Structural Causal Model (SCM)

### Corrected Graph

D_train/seed (seed, hash, training data)와 input X를 독립적인 exogenous 노드로 분리한다.
θ는 D_train의 함수일 뿐 X의 descendant가 아니다. D_train/seed → θ는 학습 과정,
X → H → Y는 추론 경로이며, 이 둘은 교차하지 않는다.

```
D_train/seed ──► θ (trained parameters)
                    │
                    │  (applies the parameterized function)
                    ▼
  Input X ──────► Hidden state H = F(X; θ)
                    │
       ┌────────────┼────────────┬─────────────┐
       ▼            ▼            ▼             ▼
   Topology     Smooth       Magnitude     Harmonic      Validity(V)
    Q(H)         S(H)          M(H)         Hm(H)           │
       │            │            │             │            │
       ├────────────┴────────────┴─────────────┤            │
       │     Representative ρ                  │            │
       │   (gauge choice for Q)                │            │
       └────────────┬──────────────────────────┘            │
                    │                                       │
       ┌────────────┼──────────────┐                        │
       ▼            ▼              ▼                        │
    Donor         Donor          Natural                    │
    Selection(S)  Field(z_d)     Recipient(z_r)             │
       │            │              │                        │
       │            ▼              ▼                        │
       │      Intervention       Natural                    │
       │    z_int = I(z_d,z_r)   Continuation               │
       │            │              │                        │
       └────────────┼──────────────┘                        │
                    ▼                                       │
           Continuation H' = F_cont(z_int)                  │
                    │                                       │
       ┌────────────┼──────────────┐                        │
       ▼            ▼              ▼                        │
   Manifold      Relaxation    Output Y = R(H')             │
   Distance      Drift                                      │
       │            │                                       │
       └────────────┴──────────► Validity Gates             │
```


### Estimand Definitions (정확한 조건부 estimand)

**Primary (selected-subpopulation estimand)**:
\[
\Delta_{\text{mech}}(D,s) = \mathbb{E}_{r \in \mathcal{R}(s),\, d \in \mathcal{D}_S(r)}\left[
    M_{\text{vortex}}(r,d) - \max_{f \in \mathcal{F}} M_f(r,d)
    \mid D,s
\right]
\]
where:
- \(s\): seed (generalization unit), drawn from \(P(D_{\text{train}})\)
- \(\mathcal{R}(s)\): recipient examples for seed \(s\)
- \(\mathcal{D}_S(r)\): donor candidates selected by frozen selection function \(S\)
- \(\mathcal{F} = \{\text{smooth}, \text{magnitude}, \text{global\_phase}, \text{zero\_charge\_phase}, \text{fourier\_low}, \text{fourier\_high}, \text{PCA}, \text{random\_direction}, \text{harmonic}, \text{charge\_arrangement\_shuffle}\}\)
  <!-- canonical null family set: smooth, magnitude, global_phase, zero_charge_phase, fourier_low, fourier_high, PCA, random_direction, harmonic, charge_arrangement_shuffle -->
  <!-- NULL_FAMILIES: smooth, magnitude, global_phase, zero_charge_phase, fourier_low, fourier_high, pca, random_direction, harmonic, charge_arrangement_shuffle
  -->
- \(M_a(r,d)\): arm \(a\)의 behavioral effect (donor-transfer metric, §7.2)

**All-admissible estimand (sensitivity)**:
\[
\Delta_{\text{mech}}^{\text{all}}(D,s) = \mathbb{E}_{r \in \mathcal{R}(s),\, d \in \mathcal{A}(r)}
\left[
    M_{\text{vortex}}(r,d) - \max_{f \in \mathcal{F}} M_f(r,d)
    \mid D,s
\right]
\]
where \(\mathcal{A}(r)\)는 selection function 없이 decomposition validity gate만 통과한 모든 admissible pair.

**Per-family estimand (secondary)**:
\[
\delta_f(D,s) = \mathbb{E}_{r,d}\left[M_{\text{vortex}}(r,d) - M_f(r,d) \mid D,s\right], \quad \forall f \in \mathcal{F}
\]


### Selection Mechanism: Effect-Modifier Conditioning, Not Collider Bias

Donor selection function \(S\)는 hidden-state geometry (charge count, energy, spectrum)로 pair를 필터링한다.
이 selection은 post-treatment collider가 **아니다**. 그 이유:

1. θ는 D_train/seed → θ 경로에서 결정되고, selection S는 이미 학습된 θ 하에서 H의 함수이다. S → Y 경로는 H → Y를 통해 naturally 흐르며, S가 별도로 Y의 cause가 아니다.
2. Selection은 **effect-modifier conditioning**이다: S는 어떤 pair가 intervention의 효과를 더 크게/작게 만들지 결정한다. 이는 precisely the estimand we want — "selected pairs에서 vortex가 다른 feature보다 더 큰 효과를 내는가?"
3. Collider bias라면 S가 θ의 descendant이면서 Y의 ancestor여야 하지만, 여기서 S는 H → Y 경로상의 conditioning node일 뿐, θ → Y 경로를 confound하지 않는다.

**Proper handling**:
- Primary estimand \(\Delta_{\text{mech}}(D,s)\)는 pre-registered selection function \(S\)로 정의된 selected subpopulation에서 평가한다.
- Sensitivity analysis로 all-admissible estimand \(\Delta_{\text{mech}}^{\text{all}}(D,s)\)를 보고하여, selection이 결과를 얼마나 바꾸는지 정량화한다.
- 두 estimand 간 차이가 크면 selection function이 effect modifier로 작동한다는 증거이며, 이 자체가 scientifically interesting하다.

---

## 7.2 Donor-Transfer Behavioral Metrics (기증자 전이 행동 지표)

단순한 "donor_ll - recipient_ll" margin은 **불충분**하다. Vortex transplant가 진짜 donor-specific behavioral change를 일으키는지 평가하려면 아래 다섯 지표로 분해해야 한다.

### Metric Decomposition

입력 X에 대해 모델이 출력하는 vocabulary distribution \(p(y \mid X, H')\) (또는 per-position log-probability)에 대해:

| Metric | Definition | Interpretation |
|--------|-----------|---------------|
| **Target donor Δ log-likelihood** | \(\Delta \log p(y_d) = \log p(y_d \mid H') - \log p(y_d \mid H_r)\) | Intervened state가 donor output을 얼마나 더 선호하는가 (signed). 양수여야 함. |
| **Recipient Δ log-likelihood** | \(\Delta \log p(y_r) = \log p(y_r \mid H') - \log p(y_r \mid H_r)\) | Recipient output이 얼마나 억제되는가. 음수여야 함 (또는 최소한 donor보다 작아야 함). |
| **Max alternative donor gain** | \(\max_{j \neq d} \Delta \log p(y_j)\) | 다른 donor 중 가장 큰 gain. 이 값이 target donor gain보다 작아야 specificity가 있음. |
| **Entropy change** | \(\Delta H(p) = H(p(\cdot \mid H')) - H(p(\cdot \mid H_r))\) | Distribution의 불확도 변화. 크게 변하면 non-targeted perturbation 의심. |
| **Target donor rank** | \(\text{rank}(y_d \mid p(\cdot \mid H'))\) | Intervention 후 donor token의 순위 (1 = highest probability). 1에 가까울수록 강한 효과. |

### Validity Conditions for a Positive Donor-Transfer Claim

Claim "vortex transplant가 donor-specific behavior를 유도한다"가 성립하려면:

1. \(\Delta \log p(y_d) > 0\) (donor로 향함)
2. \(\Delta \log p(y_d) > \max_{j \neq d} \Delta \log p(y_j)\) (donor-specific, 다른 donor보다 큼)
3. \(\Delta \log p(y_r) < 0\) 또는 최소한 \(\Delta \log p(y_d) > \Delta \log p(y_r)\) (recipient로부터 멀어짐)
4. \(\Delta H(p)\)가 작음 (targeted perturbation; large entropy change는 non-specific perturbation을 시사)
5. \(\text{rank}(y_d)\)가 낮음 (가급적 1)

### Aggregate Arm-Level Metric \(M_a\)

위 per-pair 지표들을 seed-level로 aggregate할 때:

\[
M_a(r,d) = \Delta \log p(y_d) - \max_{j \neq d, j \neq r} \Delta \log p(y_j)
\]

이 metric은 donor-specific gain에서 strongest alternative gain을 뺀 "specificity-adjusted donor margin"이다.
\(M_a > 0\)는 donor가 모든 다른 alternative보다 더 큰 gain을 얻었음을 의미한다.

Primary estimand \(\Delta_{\text{mech}}\)는 이 \(M_a\)를 사용하여 계산한다.

---

## 7.3 Primary Null Hypothesis: Intersection-Union Test (IUT)

### Claim Structure

Vortex의 인과적 우월성 claim은 **"vortex가 모든 null family를 능가한다"** 이다:

\[
H_0 = \bigcup_{f \in \mathcal{F}} \{\delta_f \leq 0\}, \quad
H_1 = \bigcap_{f \in \mathcal{F}} \{\delta_f > 0\}
\]

where \(\delta_f = \mathbb{E}[M_{\text{vortex}} - M_f]\).

이것은 Intersection-Union Test (IUT; Berger 1982)이다. IUT에서:
- \(H_0\)는 **하나라도** \(\delta_f \leq 0\)인 family가 존재하면 참이다 (vortex가 그 family를 능가하지 못함).
- \(H_1\)은 **모든** family에 대해 \(\delta_f > 0\)여야 참이다 (vortex가 모든 family를 능가함).

### Rejection Rule

각 family \(f\)에 대한 one-sided test \(T_f\): \(H_{0,f}: \delta_f \leq 0\) vs \(H_{1,f}: \delta_f > 0\). p-value \(p_f\).

**IUT rejection rule**: \(\text{Reject } H_0 \iff \max_{f \in \mathcal{F}} p_f \leq \alpha\)

**증명**:
- IUT의 size-\(\alpha\) test는: reject \(H_0 \iff \text{reject } H_{0,f} \text{ for all } f\).
- 각 \(H_{0,f}\)를 level \(\alpha\)에서 reject한다: reject each \(H_{0,f} \iff p_f \leq \alpha\).
- 모든 \(f\)에 대해 reject \(\iff \max_f p_f \leq \alpha\).

**절대 min(p_f)를 사용하지 말 것.** max(p_f)가 맞다. min(p_f)은 "하나라도 유의하면 reject"이고, 이는 \(\cup\) null의 complement인 \(H_1\)이 아니라 \(H_0\)에 대응된다.

### Why NOT Bonferroni

IUT는 각 individual test \(T_f\)를 level \(\alpha\)에서 수행해도 overall test가 level \(\alpha\)를 유지한다:
\[
P_{H_0}(\text{reject}) = P_{H_0}\left(\bigcap_f \{p_f \leq \alpha\}\right) \leq P_{H_0}(p_{f^*} \leq \alpha) \leq \alpha
\]
where \(f^*\)는 \(H_0\) 하에서 \(\delta_{f^*} \leq 0\)인 (적어도 하나 존재하는) family.

따라서 **Bonferroni correction은 불필요**하며, 오히려 검정력을 불필요하게 낮춘다.
IUT의 대가(price)는 multiplicity penalty가 아니라 **conjunction requirement** 자체이다
(모든 family를 동시에 이겨야 하므로 각 family가 individually significant해야 함).

### Recommended Primary Test Procedure

1. 각 family \(f \in \mathcal{F}\)에 대해 hierarchical bootstrap으로 \(\delta_f\)의 one-sided 95% CI를 계산한다.
2. IUT reject 조건: 모든 \(f\)에 대해 CI 하한 > 0.
3. Equivalent decision: 각 family에 대해 bootstrap one-sided p-value \(p_f\)를 계산하고, \(\max_f p_f \leq 0.05\).

### Simultaneous Confidence Intervals (선택적)

Individual CI의 동시 커버리지를 보장하고 싶다면 max-T bootstrap을 사용:

각 bootstrap replicate \(b = 1,\dots,B\)에서 모든 \(f\)에 대한 \(\hat{\delta}_f^{(b)}\)를 계산하고,
studentized maximum:
\[
t_{\text{max}}^{(b)} = \max_{f \in \mathcal{F}} \frac{\hat{\delta}_f^{(b)} - \hat{\delta}_f}{\hat{\sigma}_f}
\]
95% quantile \(c_{0.95}\)로 모든 family에 대한 동시 one-sided CI:
\[
\hat{\delta}_f - c_{0.95} \cdot \hat{\sigma}_f
\]

이 CI가 모든 \(f\)에서 > 0이면 \(H_0\)를 reject. 이 접근은 family 간 의존성을 자동으로 반영한다.

### Per-Family p-value Computation

각 family별 p-value는 hierarchical bootstrap으로 계산:
1. B = 9999 resamples of seeds
2. 각 resample에서 \(\hat{\delta}_f\) 계산
3. One-sided p-value: \(p_f = \frac{1}{B+1}\left(1 + \sum_{b=1}^B \mathbb{I}[\hat{\delta}_f^{(b)} \leq 0]\right)\)

최소 B = 1999 (p_min ≈ 0.0005), 권장 B = 9999.

### IUT and the Go/No-Go Decision

| Condition | Verdict |
|-----------|---------|
| \(\max_f p_f \leq \alpha\) AND 대표성 gate 통과 AND 다양체 gate 통과 | `GO_CONFIRMATORY` |
| \(\max_f p_f \leq \alpha\) BUT 대표성 또는 다양체 gate 실패 | `INCONCLUSIVE_REPRESENTATIVE` 또는 `INCONCLUSIVE_MANIFOLD` |
| \(\max_f p_f > \alpha\) | `NO_GO_MECHANISM` |

---

## 7.4 Separate Validity Gates (Mandatory, Not Pooled)

Scalar nuisance maximum에 representative invariance나 manifold validity를 섞어선 안 된다.
이들은 **별도의 mandatory gate**로 취급해야 한다.

### Gate 1: Representative Invariance

**Claim**: Vortex의 인과 효과는 specific gauge choice가 아니라 topological charge class 자체에서 비롯된다.

**Procedure**:
- 각 donor-recipient pair에 대해 \(K \geq 5\)개의 서로 다른 same-charge representative를 생성한다.
- 각 representative \(\rho_k\)에 대해 vortex transplant를 수행하고 donor-transfer metric \(M_{\text{vortex}}(r,d,\rho_k)\)를 측정한다.
- Representative variance 분해:

\[
\text{Var}_{\text{total}} = \text{Var}_{\text{charge}} + \text{Var}_{\text{repr}} + \text{Var}_{\text{noise}}
\]

where \(\text{Var}_{\text{charge}}\)는 서로 다른 charge configuration 간의 variance,
\(\text{Var}_{\text{repr}}\)는 same-charge 내 representative 간의 variance.

**Gate**: \(\text{Var}_{\text{repr}} / \text{Var}_{\text{charge}} < \tau_{\text{rep}}\) (recommended \(\tau_{\text{rep}} = 0.25\)).
즉, representative variance가 charge variance의 25% 미만이어야 한다.

**Failure mode**: Representative variance가 크다면, "vortex charge"가 아니라 "particular smooth component"가 효과를 내는 것이다.

### Gate 2: Manifold Validity

**Claim**: Vortex-transplanted state가 natural hidden state manifold 위에 있어야 한다.
Off-manifold perturbation은 behavioral change를 일으키더라도 "vortex mechanism"의 증거가 아니다.

**Procedure (calibration에서 선택된 최선의 manifold model 사용)**:
1. **Nearest natural neighbor distance** (primary):
   \[
   d_{\mathcal{M}}(H') = \min_{H_n \in \mathcal{N}} \|H' - H_n\|
   \]
   where \(\mathcal{N}\)은 calibration training data에서 수집한 natural hidden state들의 pool.
2. **Reconstruction error**: PCA subspace로 projection 후 residual norm.
3. **Relaxation drift**: \(F\) recurrence 10 step 후 \(H'\)와의 거리.
4. **kNN density ratio**: \(\log \frac{\text{knn\_dist}_{\text{natural}}(H')}{\text{knn\_dist}_{\text{intervened}}(H')}\).

**Gate**: Natural state들의 manifold distance distribution \(D_{\text{nat}}\)에 대해,
intervened state distance가 \(D_{\text{nat}}\)의 95% quantile 이하여야 한다.
\[
d_{\mathcal{M}}(H') \leq Q_{0.95}(D_{\text{nat}})
\]

**Failure mode**: Intervened state가 off-manifold라면, behavioral change가 manifold 바깥으로 밀어낸 artifact일 가능성이 높다.

### Gate Wiring

이 두 gate는 IUT의 일부가 아니다 (IUT는 null family 대비 vortex 우월성만 검증).
Gate는 IUT 통과 후 **separate hurdle**로 적용된다:

| Test | What it checks | Binding? |
|------|---------------|:--------:|
| IUT (max p_f ≤ α) | Vortex > ALL null families | Yes (primary claim) |
| Representative invariance | Var_rep / Var_charge < τ_rep | Yes (mechanism specificity) |
| Manifold validity | d_M(H') ≤ Q_0.95(D_nat) | Yes (internal validity) |

**셋 모두 통과해야** "vortex가 인과적으로, specifically, on-manifold로 작동한다"는 claim이 성립한다.

---

## 7.5 Intervention Taxonomy (재구성)

### Sufficiency Arms
Does transplanting the vortex *suffice* to shift behavior toward the donor?
- `vortex`: canonical vortex field from donor → recipient
- `vortex_alternate`: different same-charge representative (representative invariance gate에 사용)
- `vortex_minimal`: minimal-surgery vortex (smallest displacement for same charge)

### Necessity Arms
Does *removing* the vortex eliminate donor-specific behavior?
- `vortex_remove_all`: replace vortex with \(v_0\) (charge-neutral field)
- `vortex_remove_pair`: annihilate a single defect pair
- `vortex_sham`: same-displacement surgery with no charge change

### Specificity Arms
Is the effect specific to *particular* vortex properties?
- Sign flip: \(Q \to -Q\)
- Spatial shift: translate vortex pattern (\(\pm 2\pi\) to a different plaquette or region)
- Density change: add or remove charges

### Representative Arms (Gate 1 input)
Is the effect invariant to the specific gauge choice?
- Same-charge, different representative (\(K \geq 5\) samples)
- Representative variance decomposition

### Mechanistic Baselines (Null Families for IUT)
Do other hidden-state features carry similar information?
- `smooth`: smooth component transplant
- `magnitude`: magnitude component transplant
- `global_phase`: uniform phase rotation control
- `zero_charge_phase`: spatially structured zero-charge control
- `fourier_low`: Fourier low-pass (retain k ≤ cutoff)
- `fourier_high`: Fourier high-pass (retain k > cutoff)
- `PCA`: PCA projection transplant (top-k components)
- `random_direction`: norm-matched random direction (B ≥ 99 samples per pair)
- `harmonic`: harmonic sector swap (same vortex, different harmonic cycle holonomy)
- `charge_arrangement_shuffle`: charge-count-matched random arrangement (keeps charge density, randomizes positions)

**Note on representative sensitivity**:
Representative sensitivity (`same_charge_rep`) is a separate mandatory gate, NOT part of the comparable null family list. It tests invariance under the zero-charge multiplicative factor, which is a scientific validity condition rather than a competing mechanistic hypothesis.

### Natural Controls
Is the intervention on-manifold?
- Natural neighbor (nearest natural hidden state with target topology)
- Denoising projection
- Relaxation post-intervention (Gate 2 input)

---

## 7.6 Patch Efficacy Checklist (재구성)

A successful causal claim requires all five criteria. The first two alone are insufficient.

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Intervention changes behavior toward donor | \(\Delta \log p(y_d) > 0\) | Per-pair |
| Changes are donor-specific | \(\Delta \log p(y_d) > \max_{j \neq d} \Delta \log p(y_j)\) | Per-pair |
| Vortex beats ALL null families | \(\max_f p_f \leq \alpha\) (IUT) | Seed-aggregate |
| Effect is representative-invariant | \(\text{Var}_{\text{rep}} / \text{Var}_{\text{charge}} < \tau_{\text{rep}}\) | Seed-aggregate |
| Effect is on-manifold | \(d_{\mathcal{M}}(H') \leq Q_{0.95}(D_{\text{nat}})\) | Per-intervention |

The core risk: a perturbation that is *large enough* and pushes the state toward the donor's *general neighborhood* will shift outputs. Without specificity metrics (§7.2), representative-invariance gate (§7.4), and manifold gate (§7.4), a positive donor-recipient margin proves nothing about the vortex mechanism.

---

## 7.7 Missingness and Analyzability (결측 처리)

Complete-case analysis는 다음과 같은 이유로 **unqualified하게 accept할 수 없다**:
missingness가 outcome에 informative할 가능성이 높고, missing pattern이 arm 간 differential할 수 있다.

### Missing/Invalid Pair Reasons (재분류)

| Code | Description | Analyzability |
|------|-------------|:---:|
| `NO_DECOM_PAIR` | 한 state의 magnitude가 near-zero이거나 charge neutrality failure | Not analyzable |
| `NO_DONOR_SELECTED` | Selection function이 admissible donor를 찾지 못함 | Not analyzable |
| `NO_MATCHED_NULL` | Matched control을 구성할 수 없음 | Not analyzable |
| `MANIFOLD_FAILURE` | Intervention이 off-manifold state를 생성 | Not analyzable |
| `NUMERICAL_FAILURE` | Decomposition 또는 intervention이 numerical error | Not analyzable |
| `TASK_FAILURE` | Model이 minimum accuracy를 달성하지 못함 | Not analyzable |
| `DEGENERATE_DISTRIBUTION` | Donor/recipient distribution entropy near-zero (division by zero risk) | Not analyzable |

### Estimands Under Missingness

**1. Analyzability estimand (\(\psi_{\text{analyze}}\))**:
\[
\psi_{\text{analyze}} = \mathbb{E}[A \mid s]
\]
where \(A \in \{0,1\}\)는 pair가 analyzable한지 여부. Seed별 analyzable fraction을 보고한다.

**2. Failure-as-non-supportive composite (\(\Delta_{\text{mech}}^{\text{composite}}\))**:
Missing/invalid pair의 outcome을 vortex 효과가 없는 것으로 impute한다:
\[
\Delta_{\text{mech}}^{\text{composite}} = \frac{1}{N_{\text{total}}} \sum_{\text{all pairs}} 
\begin{cases}
M_{\text{vortex}} - \max_f M_f & \text{if analyzable} \\
0 & \text{if not analyzable}
\end{cases}
\]

이 composite estimand는 "분석 가능한 pair 중에서는 효과가 있지만, 분석 불가능한 pair를 포함하면 효과가 희석된다"는
시나리오를 보수적으로 잡아낸다.

**3. Selected-subpopulation estimand (\(\Delta_{\text{mech}}\))**: analyzable pair만 사용. 이 값이 primary이다.
단, 반드시 analyzable fraction \(\psi_{\text{analyze}}\)와 composite \(\Delta_{\text{mech}}^{\text{composite}}\)를
함께 보고하여 selection transparency를 확보한다.

### Sensitivity Bounds

**Worst-case bounds**: analyzable pair의 효과가 가장 크고, non-analyzable pair는 이 효과의 하한/상한에 있다고 가정한다.
\[
\Delta_{\text{mech}}^{\text{lower}} = \min\left(\Delta_{\text{mech}}^{\text{composite}}, \Delta_{\text{mech}} \cdot \bar{A}\right)
\]
\[
\Delta_{\text{mech}}^{\text{upper}} = \max\left(\Delta_{\text{mech}}, \Delta_{\text{mech}} \cdot \bar{A} + (1 - \bar{A}) \cdot \Delta_{\text{best-case}}\right)
\]
where \(\bar{A}\)는 overall analyzable fraction, \(\Delta_{\text{best-case}}\)는 관측된 최대 효과.

**Tipping-point analysis**: analyzable fraction이 어느 수준 이하로 떨어지면 결론이 뒤집히는지 계산한다:
\[
\bar{A}_{\text{critical}} = \frac{-\Delta_{\text{mech}}}{|\Delta_{\text{wc}}| - \Delta_{\text{mech}}}
\]
where \(\Delta_{\text{wc}}\)는 non-analyzable pair에서 가정하는 worst-case effect (보통 0 또는 negative).

### Reporting Requirements

1. Table: seed별 analyzable pair count, total pair count, analyzable fraction
2. Table: failure reason별 count (위 7가지 코드)
3. Primary result: \(\Delta_{\text{mech}}\) (analyzable only) with IUT
4. Reported alongside: \(\Delta_{\text{mech}}^{\text{composite}}\), \(\psi_{\text{analyze}}\)
5. Sensitivity: worst-case bounds, tipping-point \(\bar{A}_{\text{critical}}\)
6. Arm 간 differential missingness test: 각 arm별 missing rate을 비교한다 (Fisher's exact 또는 bootstrap). Differential missingness가 발견되면 arm-specific report가 필요하다.

---

## 7.8 Normalized Recovery (Edge Cases)

\[
R_a = \frac{M_a - M_{\text{NR}}}{M_{\text{WS}} - M_{\text{NR}}}
\]
where NR = natural_recipient (no intervention), WS = whole_state (complete donor state).

**Edge case 1: \(M_{\text{WS}} - M_{\text{NR}} \approx 0\)** — denominator near zero.
Donor와 recipient가 identical outputs를 생성하거나, whole-state intervention이 효과가 없음.
→ Mark pair as `INVALID_NORMALIZATION`. \(R_a\) 분석에서 제외. Per-seed invalid fraction 보고.

**Edge case 2: \(M_{\text{WS}} - M_{\text{NR}} < 0\)** — whole-state sanity failure.
Complete donor state가 donor-like output을 생성하지 못함. Model 또는 task에 심각한 문제.
→ Mark seed as `FAILED_DIRECTIONAL_SANITY` if >10% of pairs. 해당 seed는 확인적 분석에서 제외 검토.

**Edge case 3: \(R_a > 1\)** — vortex가 whole-state보다 강한 효과.
Possible (vortex가 noisy full state보다 pure한 information carrier일 수 있음)하지만 justification 필요.
→ Report as-is. Flag as "vortex > whole_state" pair. Per-seed fraction 보고.

**Edge case 4: \(M_{\text{NR}} > M_{\text{vortex}}\) but \(M_{\text{vortex}} > M_{\text{NR}}\)가 claim의 전제** — vortex가 recipient baseline보다 못하면 negative margin.
→ IUT에서 제외하지 않음. Negative vortex margin 자체가 \(\delta_f \leq 0\)에 기여하므로 IUT에서 자연히 reject된다.

---

## 7.9 Null Family Separation (IUT 기준 재작성)

**DO NOT POOL**. 각 null family는 distinct counterfactual을 test한다.

| Family | Counterfactual | \(H_{0,f}\) | Test |
|--------|---------------|------------|------|
| `random_direction` | Norm-matched random perturbation이면 충분 | \(\delta_{\text{rand}} \leq 0\) | One-sided paired bootstrap |
| `fourier_low` | Low-frequency info가 효과의 원인 | \(\delta_{\text{low}} \leq 0\) | One-sided paired bootstrap |
| `fourier_high` | High-frequency info가 효과의 원인 | \(\delta_{\text{high}} \leq 0\) | One-sided paired bootstrap |
| `PCA` | Top-k variance direction이 효과의 원인 | \(\delta_{\text{PCA}} \leq 0\) | One-sided paired bootstrap |
| `smooth` | Smooth phase component가 효과의 원인 | \(\delta_{\text{smooth}} \leq 0\) | One-sided paired bootstrap |
| `magnitude` | Amplitude modulation이 효과의 원인 | \(\delta_{\text{mag}} \leq 0\) | One-sided paired bootstrap |
| `harmonic` | Global holonomy가 효과의 원인 | \(\delta_{\text{harm}} \leq 0\) | One-sided paired bootstrap |
| `charge_arrangement_shuffle` | Charge spatial arrangement이 효과의 원인 | \(\delta_{\text{shuffle}} \leq 0\) | One-sided paired bootstrap |

**Primary test (IUT)**: \(\max_{f} p_f \leq \alpha\). Bonferroni 불필요.

**Secondary decomposition**: IUT reject 이후 어떤 family에서 margin이 가장 tight한지 보고 (가장 큰 p_f).
이것이 vortex가 가장 근소하게 이긴 family이며, 잠재적 weakness를 드러낸다.

**Per-family randomization draws**:
- `random_direction`: per-pair B ≥ 199 random draws. Family-level statistic: per-seed q95 of random draws.
- `charge_arrangement_shuffle`: B ≥ 199 shuffles per pair. Family-level statistic: per-seed mean shuffle margin.

**Separate gate — representative sensitivity** (NOT in null family list):
- K ≥ 5 same-charge representatives per pair. Gate: Var_rep / Var_charge ≤ τ_rep (UNFROZEN).

---

## 7.10 Selection Subpopulation vs All-Admissible (구분)

### Selected-Subpopulation Estimand (Primary)

\[
\Delta_{\text{mech}}(D,s) = \mathbb{E}_{r \in \mathcal{R}(s),\, d \in \mathcal{D}_S(r)}[\dots]
\]

\(\mathcal{D}_S(r)\)는 frozen selection function \(S\)로 선택된 donor set. 이 estimand는
"우리의 selection 기준으로 고른 pair들에서 vortex가 인과적으로 우월한가?"를 묻는다.

과학적 질문: selection function이 scientific hypothesis의 일부이다. 만약 vortex가 truly causal mechanism이라면,
적절한 selection criteria로 고른 pair에서 그 효과가 가장 잘 드러날 것이다. Selection function 자체가
"어떤 조건에서 vortex가 중요한가"라는 과학적 질문의 operationalization이다.

### All-Admissible Estimand (Sensitivity)

\[
\Delta_{\text{mech}}^{\text{all}}(D,s) = \mathbb{E}_{r \in \mathcal{R}(s),\, d \in \mathcal{A}(r)}[\dots]
\]

\(\mathcal{A}(r)\)는 validity gate (decomposition 성공, charge 존재 등)만 통과한 모든 pair.
Selection function 없이 admissible한 모든 pair에서의 평균 효과.

### Diagnostic: Selection Amplification Ratio

\[
\rho_{\text{sel}} = \frac{\Delta_{\text{mech}}}{\Delta_{\text{mech}}^{\text{all}}}
\]

- \(\rho_{\text{sel}} > 1\): selection이 효과를 amplify한다. Science가 작동하고 있다.
- \(\rho_{\text{sel}} \approx 1\): selection이 큰 차이를 만들지 않는다. Selection function이 효과적이지 않거나, 효과가 모든 admissible pair에서 균일하다.
- \(\rho_{\text{sel}} < 1\): selection이 오히려 효과를 약화시킨다. Selection 기준을 재검토해야 한다.

### Reporting

1. Primary: \(\Delta_{\text{mech}}\) (selected subpopulation) with IUT
2. Sensitivity: \(\Delta_{\text{mech}}^{\text{all}}\) (all admissible)
3. Diagnostic: \(\rho_{\text{sel}}\), selection amplification ratio
4. 두 estimand의 CI가 겹치는지 확인 (overlap → selection이 큰 차이를 만들지 않음)

---

## 7.11 Combined Decision Logic (종합 판정)

```
CONDITIONS:
  C1 = max_f p_f ≤ α (IUT, §7.3)
  C2 = Var_rep / Var_charge < τ_rep (대표성 gate, §7.4)
  C3 = median(d_M(H')) ≤ Q_0.95(D_nat) (다양체 gate, §7.4)
  C4 = ρ_sel ≥ 0.8 (selection sensitivity, §7.10)
  C5 = ψ_analyze ≥ 0.70 (analyzable fraction, §7.7)

VERDICT:
  C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 → GO_CONFIRMATORY
  C1 ∧ C2 ∧ C3 ∧ ¬C4 → GO_CONFIRMATORY (selection weak but mechanism valid)
  C1 ∧ ¬C2 → INCONCLUSIVE_REPRESENTATIVE
  C1 ∧ ¬C3 → INCONCLUSIVE_MANIFOLD
  ¬C1 → NO_GO_MECHANISM
  ¬C5 → INCONCLUSIVE_STATISTICS (too many missing; increase seeds or fix pipeline)
```

IUT가 primary go/no-go를 결정한다. 대표성과 다양체 gate는 mechanism claim의 internal validity를 보장한다.
Selection과 analyzability는 robustness 조건이다.
