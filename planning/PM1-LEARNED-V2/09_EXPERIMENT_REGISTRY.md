# Experiment Registry (실험 등록대장)

**Status**: COMPLETE
**Phase**: All calibration (C01–C16) + confirmatory (F01–F05) cards frozen
**Language**: 한국어 + English technical terms
**Contract ref**: `15_PM1_LEARNED_V2_CONTRACT.yaml`

각 experiment card는 사전등록(pre-registration)된 specification이다. Calibration phase 실험은 V2 contract의 UNFROZEN_CALIBRATION parameter를 결정한다. Confirmatory phase 실험은 frozen contract 하에서 hypothesis test를 수행한다.

---

## Calibration Phase (C01–C16)

---

### C01: Random Field Null Baseline (랜덤장 영기저선)

```
Experiment ID:       C01
Scientific question: 학습되지 않은 랜덤 위상장 (random phase field) 의 vortex prevalence, defect density, branch margin 분포는 무엇인가?
Formal hypothesis:
  H0: Random 위상장의 defect density와 branch margin 분포가 trained model hidden state와 동일하다.
  H1: Random 위상장이 trained model보다 높은 defect density와 낮은 branch margin (불안정한 topology) 을 보인다.
Alternative explanations:
  - Random field의 high prevalence가 단순히 위상 roughness의 artifact일 수 있으며, trained model도 유사한 roughness를 가질 수 있다 (→ C02가 확인).
  - Branch margin이 낮아도 charge extraction이 수치적으로 유효할 수 있다 (→ C04가 확인).
  - Random field가 오히려 trained보다 "clean"한 topology를 가질 가능성 (branch margin이 더 클 수 있음).
Model:               N/A (순수 랜덤 복소장, untrained 모델 아님)
Task:                 N/A (모방 대상 없음 — random complex field on T²)
Input parameterization: Random spatial: z(x) = r(x) · e^{iθ(x)} where r(x) ~ U(0.1, 1.0), θ(x) ~ U(−π, π), independent per lattice site
Split:                Calibration
Unit of analysis:     Random field instance (128 instances across 10 RNG seeds)
Recipient selection:  N/A (분석 대상은 개별 random field)
Donor selection:      N/A
Intervention arms:    N/A (null baseline 측정만 — 개입 없음)
Sham/controls:        N/A
Null families:        N/A (이 실험 자체가 null baseline 정의)
Primary metric:       defect_density (plaquette 당 mean absolute charge), site_density (nonzero-charge plaquette fraction), branch_margin distribution (min, q01, q05, median)
Validity metrics:     net_charge_zero (T² topological constraint), decomposition_fidelity < 1e-10
Statistical test:     Descriptive only — 분포 요약 (quantile table, histogram). No hypothesis test.
Sample-size rule:     128 instances × 10 RNG seeds = 1,280 total fields. 정밀한 quantile 추정에 충분 (q01 안정화에 ~1000 samples).
Pass:                 branch_margin q01 분포가 안정적이고 (seed 간 variance 작음), defect_density 분포가 symmetric unimodal
Fail:                 branch_margin 분포가 degenerate (all zeros) — charge extraction 자체가 불가
Inconclusive:         seed 간 branch_margin variance가 지나치게 커서 null baseline이 informative하지 않음
Required code:        topological/v2/topology.py: extract_charge, branch_margin
                      diagnostic/run_untrained_topology.py (random field variant)
Required tests:       test_topology_v2.py: test_random_field_branch_margin_positive, test_random_field_net_charge_zero
Raw artifact schema:  per-instance: {field_id, rng_seed, n_channels(1), H(16), W(16),
                      defect_density, site_density, per_site_charge_map: [(x,y,q)], 
                      branch_margin: {min, q01, q05, q50}, net_charge, decomposition_error}
Compute formula:      128 fields × 10 seeds × (field generation + curl computation + charge extraction + branch margin) / CPU throughput
                      ≈ 1280 × ~0.05s ≈ 64s CPU (~1 CPU-minute)
Dependency:            None (standalone null baseline — 최우선 실행)
Claim enabled:        "untrained random fields의 topology 분포가 trained model과 구분된다" (C02와 결합 시)
Claim prohibited:     "Random field가 trained model과 identical한 topology를 가진다" (C02에서 검증 필요)
```

---

### C02: Untrained Model Topology (비학습 모델 위상 분석)

```
Experiment ID:       C02
Scientific question: U1ConvRNN과 PlainConvRNN의 초기화 직후 (untrained) hidden state는 어떤 vortex topology를 가지며, 한 step blank recurrence 후 어떻게 변화하는가?
Formal hypothesis:
  H0: Untrained U1ConvRNN과 PlainConvRNN의 topology (defect_density, branch_margin) 가 동일하다.
  H1: U1ConvRNN의 U(1)-동변적 recurrence가 PlainConvRNN과 구분되는 topology 변화를 일으킨다 (예: branch_margin 증가 또는 defect density 감소).
Alternative explanations:
  - U1과 Plain이 동일한 initialization scheme을 사용하므로 post-write topology가 동일할 수 있다 (진단 결과에서 이미 확인: density ~0.33).
  - Pre-GO topology 차이가 U1 vs Plain이 아니라 단순히 tanh vs radial_tanh의 magnitude scaling 차이일 수 있다.
  - Untrained topology가 trained topology와 질적으로 동일하면 "defect가 학습으로 생성된다"는 주장이 불가 (C05-C06으로 넘어가야 함).
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8) — untrained (random initialization)
Task:                 Copy task, test split (heldout delay=64), 128 examples per seed
Input parameterization: Standard token embedding (random full-spatial, V1 default)
Split:                Calibration
Unit of analysis:     Seed (10 seeds per model type, 총 20 seeds)
Recipient selection:  N/A (모든 test example의 hidden state 분석)
Donor selection:      N/A
Intervention arms:    N/A
Sham/controls:        N/A
Null families:        C01 random field null 분포가 비교 기준 (informal)
Primary metric:       state_prevalence (any channel has both + and − defects), defect_density, site_density, branch_margin distribution (q01, q05, median), per-channel prevalence
                      Source stratification: post_write (token embedding + write operation 직후), pre_go (blank recurrence 1 step 후)
Validity metrics:     net_charge_zero, decomposition_fidelity < 1e-10, min(|field|) > 1e-8
Statistical test:     Descriptive comparison: U1 vs Plain effect size (Cohen's d on defect_density, branch_margin q01). Two-sided bootstrap CI on density difference. No formal hypothesis test — calibration only.
Sample-size rule:     10 seeds × 2 model types = 20 seeds × 128 examples = 2,560 states. V1 진단에서 이미 실행 완료.
Pass:                 U1 pre-GO branch_margin q01이 Plain보다 실질적으로 크고 (d > 0.5), C01 random null 분포와 구분됨
Fail:                 U1과 Plain이 모든 topology metric에서 동일 (d < 0.2) — U(1) recurrence가 topology에 영향을 주지 않음
Inconclusive:         U1과 Plain의 차이가 존재하나 작고 (0.2 < d < 0.5), seed 간 variance가 커서 calibration threshold 설정이 모호함
Required code:        topological/v2/topology.py: analyze_topology (TopologyStatsV2)
                      diagnostics/run_untrained_topology.py (이미 존재, V2로 확장)
Required tests:       test_topology_v2.py: test_untrained_prevalence_bounded, test_untrained_u1_vs_plain_density
Raw artifact schema:  per-seed: {seed, model_type, split,
                      post_write: {state_prevalence, defect_density, site_density, branch_margin, per_channel_prevalence},
                      pre_go:     {state_prevalence, defect_density, site_density, branch_margin, per_channel_prevalence}}
Compute formula:      20 seeds × 128 examples × (1 forward pass + topology_analysis) / CPU throughput
                      ≈ 2,560 × ~0.1s ≈ 256s CPU (~4 CPU-minutes). 이미 V1 진단에서 실행 완료 (139s).
Dependency:            None (C01과 병렬 실행 가능). C01 결과를 비교 기준으로 사용.
Claim enabled:        "Untrained U1ConvRNN은 PlainConvRNN과 구분되는 branch stability를 가진다" — analyzable_stable_topology_gate의 untrained baseline.
Claim prohibited:     "Untrained model에 learned vortex mechanism이 존재한다" — C05-C06에서 trained model 검증 필요.
```

---

### C03: Token Embedding Topology Baseline (토큰 임베딩 위상 기준선)

```
Experiment ID:       C03
Scientific question: 개별 토큰 임베딩 (lookup table row) 자체가 vortex charge를 가지는가?
Formal hypothesis:
  H0: 모든 토큰 임베딩이 모든 plaquette에서 charge-free이다 (Q_p = 0 for all p, all tokens).
  H1: 적어도 하나의 토큰 임베딩이 nonzero plaquette charge를 가진다.
Alternative explanations:
  - 토큰 임베딩이 near-zero magnitude point를 포함하면 phase가 undefined → charge가 artifact일 수 있다.
  - 모든 토큰이 similar한 defect density를 가지면, vortex 구조가 token-specific information을 encode하지 않는다.
  - 임베딩 차원 (embedding_dim = C × H × W) 이 커서 random initialization만으로도 dense vortex plasma가 불가피하다.
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8) — untrained (random initialization, seed 0)
Task:                 N/A (token embedding lookup table 분석)
Input parameterization: Raw token embedding matrix (vocab_size × C × H × W), 정규분포 초기화
Split:                Calibration
Unit of analysis:     Individual token embedding (vocab_size=10: tokens 0–9, 단 blank=0과 go=9 포함)
Recipient selection:  N/A
Donor selection:      N/A
Intervention arms:    N/A
Sham/controls:        C01 random field (동일 차원) 과 비교하여 임베딩의 defect density가 random expectation과 다른지 확인
Null families:        N/A
Primary metric:       Per-token: nonzero_defect (any ± pair 존재), defect_density, site_density, valid_channel_fraction (C개 채널 중 magnitude_threshold 초과 채널 수 / C), min_magnitude
Validity metrics:     net_charge_zero per token, decomposition_fidelity < 1e-10
Statistical test:     Descriptive. Per-token table. U1 vs Plain embedding density equivalence (두 모델이 동일 initialization을 사용하므로 유사해야 함).
Sample-size rule:     10 tokens × 2 model types = 20 embeddings. V1 진단에서 이미 실행 완료.
Pass:                 모든 토큰이 nonzero_defect를 가지며, min_magnitude가 0에 근접 (random init artifact 확인). U1과 Plain 임베딩이 유사.
Fail:                 토큰 임베딩이 모두 charge-free — 이 경우 initialization이 topological charge를 생성하지 않으며, trained model의 charge는 순수히 학습 산물 (강한 evidence).
Inconclusive:         일부 토큰만 charge를 가짐 — magnitude threshold 선택에 민감.
Required code:        topological/v2/topology.py: analyze_topology on raw embedding tensor
Required tests:       test_topology_v2.py: test_token_embedding_prevalence, test_token_embedding_magnitude_nonzero
Raw artifact schema:  per-token: {token_idx, model_type, nonzero_defect, defect_density, site_density,
                      per_channel: [{channel, valid, defect_count}], min_magnitude, max_magnitude}
Compute formula:      20 embeddings × topology_analysis / CPU throughput ≈ 20 × ~0.05s ≈ 1s CPU
Dependency:            None (C01, C02와 병렬 실행 가능)
Claim enabled:        "Token embedding은 initialization 시점에 이미 vortex charge를 포함한다 — trained model의 vortex는 charge 생성이 아니라 재구성(reorganization)이다."
Claim prohibited:     "Token embedding이 task-relevant vortex structure를 encoding한다" (untrained이므로 task 정보 없음).
```

---

### C04: Branch Stability Across Recurrence (순환에 따른 가지안정성)

```
Experiment ID:       C04
Scientific question: Blank recurrence를 거치면서 branch margin이 U1ConvRNN과 PlainConvRNN에서 어떻게 진화하는가?
Formal hypothesis:
  H0: U1과 Plain의 branch margin 시계열이 동일한 분포를 따른다 (Δbranch_margin = 0).
  H1: U1 recurrence는 branch margin을 증가시키고 (안정화), Plain recurrence는 그렇지 않다 (U1 > Plain in terminal branch margin).
Alternative explanations:
  - U1의 radial_tanh가 magnitude를 saturation시켜서 branch margin이 artifact로 증가할 수 있다.
  - Recurrence step 수가 충분하지 않아 transient effect만 관찰될 수 있다.
  - Defect pair annihilation이 branch margin을 증가시키는 진짜 물리적 메커니즘일 수 있으며, 이는 U1의 바람직한 특성이다.
Model:               U1ConvRNN (C=8, untrained), PlainConvRNN (C=8, untrained)
Task:                 Blank recurrence (no input tokens, autonomous dynamics only)
Input parameterization: Post-write hidden state (token embedding write 직후) from test examples, 이후 순수 blank recurrence
Split:                Calibration
Unit of analysis:     Seed (per-model, per-step statistics)
Recipient selection:  N/A (8 test examples per seed)
Donor selection:      N/A
Intervention arms:    N/A
Sham/controls:        C01 random field의 branch margin 분포 (static baseline)
Null families:        N/A
Primary metric:       branch_margin_min(t) 시계열, branch_margin_q01(t) 시계열, defect_density(t) 시계열, signed_Jaccard(t) (t=0 대비 charge map 유사도), charge_lifetime (추적 가능 시)
Validity metrics:     net_charge_zero at each t, decomposition_fidelity < 1e-10 at each t, min(|field|) > 1e-8 at each t
Statistical test:     Repeated-measures comparison: U1 vs Plain terminal (t=100) branch_margin_q01 차이의 bootstrap CI. Signed Jaccard 시계열의 plateau detection.
Sample-size rule:     10 seeds × 2 model types × 8 examples × 100 recurrence steps = 16,000 state 분석. 충분.
Pass:                 U1 terminal branch_margin_q01 > Plain terminal, and signed_Jaccard(t)가 U1에서 더 높거나 안정적
Fail:                 U1과 Plain 모두 branch margin이 개선되지 않거나 (stuck near 0), 둘 다 동일하게 개선됨
Inconclusive:         Early steps에서 차이가 있으나 t=100에서 수렴 (두 model이 동일 terminal state에 도달)
Required code:        topological/v2/topology.py: analyze_topology 시계열, branch_margin 시계열, signed_Jaccard
                      topological/v2/task.py: continue_copy (blank recurrence)
                      diagnostics/ branch stability 확장 script
Required tests:       test_topology_v2.py: test_branch_stability_time_series, test_signed_jaccard_monotonic
Raw artifact schema:  per-seed, per-model: [{step: t, branch_margin: {min, q01, q05, q50}, defect_density,
                      signed_Jaccard(t vs t=0), net_charge, decomposition_error}] for t in 0..100
Compute formula:      20 seeds × 8 examples × 100 steps × (1 recurrence step forward + topology_analysis) / CPU throughput
                      ≈ 20 × 8 × 100 × ~0.02s ≈ 320s CPU (~5 CPU-minutes)
Dependency:            C02 (같은 untrained model 사용, topology baseline 제공)
Claim enabled:        "U1ConvRNN의 U(1)-동변적 recurrence가 branch stability를 향상시킨다" — analyzable_stable_topology_gate의 stability_threshold calibration
Claim prohibited:     "Trained model에서도 동일한 안정화가 발생한다" — C05-C06에서 trained model로 검증 필요.
```

---

### C05: C=1 Trainability Screen (C=1 학습 가능성 선별)

```
Experiment ID:       C05
Scientific question: C=1 (단일 복소 채널) U1ConvRNN이 copy task를 validation accuracy ≥ 90%로 학습할 수 있는가?
Formal hypothesis:
  H0: C=1 U1ConvRNN의 최대 validation accuracy < 0.90 (C=1로는 copy task를 충분히 학습할 수 없다).
  H1: C=1 U1ConvRNN이 validation accuracy ≥ 0.90을 달성한다.
Alternative explanations:
  - C=1이 underparameterized되어 task를 해결하지 못할 수 있으며, 이는 topology의 문제가 아니라 capacity의 문제다.
  - Reduced updates로 인한 undertraining이 원인일 수 있다 (→ 충분한 convergence check 필요).
  - 특정 seed에서만 수렴하고 나머지는 실패할 경우, C=1 학습이 불안정하다는 신호.
  - Charge-free smooth input embedding을 사용하지 않으면, untrained embedding에서 유입된 dense vortex plasma가 학습을 방해할 수 있다.
Model:               ScalarU1ConvRNN (C=1, U1ConvRNN with channels=1)
Task:                 Copy task, delay_range=[16,32], vocab=10, copy_length=4
Input parameterization: TWO variants tested:
                        (a) Random full-spatial embedding (V1 default, dense vortex plasma 유입)
                        (b) Charge-free smooth positive-magnitude embedding (모든 plaquette Q_p=0, everywhere min(|z|)>0.1, smooth spatial profile)
Split:                Calibration
Unit of analysis:     Seed (5 seeds)
Recipient selection:  N/A (training + validation only)
Donor selection:      N/A
Intervention arms:    N/A
Sham/controls:        N/A
Null families:        N/A
Primary metric:       Maximum validation accuracy across training (best checkpoint by lexicographic rule: accuracy > cross_entropy > update)
Validity metrics:     Training curve convergence (last 5000 updates slope ≈ 0), no NaN loss, checkpoint selection validity
Statistical test:     Per-seed accuracy report. Trainability gate: 5/5 seeds ≥ 0.90 → PASS. 3-4/5 → investigate. <3/5 → FAIL.
Sample-size rule:     5 seeds (not 3). Calibration에서 3 seeds는 seed variance 추정에 불충분. 5 seeds이면 median accuracy와 worst-case accuracy를 신뢰성 있게 추정 가능.
Pass:                 5/5 seeds achieve validation accuracy ≥ 0.90. Input variant (b)가 (a)보다 유의미하게 더 높은 accuracy를 보이면 (b)를 V2 default로 채택.
Fail:                 <3 seeds achieve ≥ 0.90. C=1 strategy 실패 → 원인 진단 (capacity vs initialization) 후 redesign 또는 C>1 escalation policy 검토.
Inconclusive:         3-4/5 seeds achieve ≥ 0.90. Seed variance가 큼 → 추가 seeds로 확인하거나 training hyperparameter 조정.
Required code:        topological/v2/model.py: ScalarU1ConvRNN
                      topological/v2/training.py: train_seed with split='cal'
                      topological/v2/task.py: generate_copy_batch with split='cal'
                      Charge-free embedding constructor (신규 구현)
Required tests:       test_model_v2.py: test_scalar_u1_forward, test_scalar_u1_equivariance_blank
                      test_training_v2.py: test_c1_convergence, test_c1_charge_free_embedding
Raw artifact schema:  per-seed: training.json {seed, model_type='scalar_u1', input_variant, updates, batch_size,
                      history: [{update, train_loss, val_accuracy, val_loss}], best_val_accuracy, convergence_slope_last_5k}
Compute formula:      Training: 5 seeds × U_updates × (delay~24 + copy_len 4) × batch 64 / GPU_throughput
                      U_updates: 레거시 교정 (C-12): NOT 30k full training. Reduced updates으로 convergence 확인.
                      Initial setting: U = 5,000 (validation frequency 500 → 10 eval points, sufficient for convergence check).
                      Training time per seed ≈ 5,000 × 28 × 64 / T_train ≈ 140s at ~6400 steps/s → ~2.3 min/seed.
                      Total: 5 × 2.3 min ≈ 12 min GPU (variant a). With variant b: ~24 min GPU total.
Dependency:            C01-C04 (topology baseline 확립 후). C01-C04 결과가 C=1 분석의 필요성/방향에 영향.
Claim enabled:        "C=1 U1ConvRNN은 copy task를 학습할 수 있으며, trained C=1 hidden state에서 vortex topology 분석이 가능하다" — C=1 strategy의 GO 조건.
Claim prohibited:     "C=1이 C=8과 동등한 task performance를 가진다" — capacity 차이로 인해 accuracy ceiling이 다를 수 있음.
```

---

### C06: C=1 Topology Emergence (C=1 위상 출현)

```
Experiment ID:       C06
Scientific question: C=1 U1ConvRNN이 copy task 학습 후 hidden state에 어떤 vortex topology가 출현하는가?
Formal hypothesis:
  H0: Trained C=1 hidden state의 defect density와 branch margin이 untrained (C02)와 동일하다 (topology가 학습으로 변하지 않는다).
  H1: Trained C=1 hidden state가 untrained보다 낮은 defect density와 높은 branch margin을 보이며, seed 간 signed Jaccard similarity가 유의미하게 높다 (structured topology emergence).
Alternative explanations:
  - Input embedding이 charge-free가 아니면, 학습으로 인한 변화와 initialization artifact를 구분할 수 없다.
  - C=1에서도 여전히 dense vortex plasma가 유지된다면, U(1) equivariance가 topology를 organize하지 않는다는 의미.
  - Sparse defect가 나타나더라도, 이것이 task-relevant한지 아니면 단순히 low-capacity model의 sparsity artifact인지 구분 필요.
Model:               ScalarU1ConvRNN (C=1), trained from C05
Task:                 Copy task, test split (heldout delay=64), 100 examples
Input parameterization: C05에서 선택된 variant 사용 (charge-free smooth 권장)
Split:                Calibration
Unit of analysis:     Seed (C05에서 trainability 통과한 모든 seed, 최대 5)
Recipient selection:  N/A (trained model의 test example hidden state 분석)
Donor selection:      N/A
Intervention arms:    N/A
Sham/controls:        C02 untrained C=1 topology (별도 실행 — C05와 동일 initialization의 untrained C=1)
Null families:        N/A
Primary metric:       defect_density (per-plaquette mean |Q|), site_density, branch_margin distribution (q01, q05, q50), signed_Jaccard between seeds (동일 입력에 대한 hidden state charge map 유사도), defect_count_sparsity (total defects / (H×W))
Validity metrics:     net_charge_zero, decomposition_fidelity < 1e-10, min(|field|) > 1e-8, analyzable_stable_topology_gate 통과율
Statistical test:     Trained vs untrained (C=1): effect size on defect_density, branch_margin_q01. Signed Jaccard across seeds: shuffle test (H0: seed 간 유사도 = random chance).
Sample-size rule:     C05 통과 seed 수에 의존. 최소 3 seeds 필요.
Pass:                 Trained C=1이 untrained C=1보다 유의미하게 낮은 defect_density (d > 0.8) AND 높은 branch_margin_q01 (d > 0.5). Sparse defect (site_density < 0.05).
Fail:                 Trained C=1 topology가 untrained와 구분 불가 (d < 0.2) 또는 모든 state에서 charge-free (defect_density = 0 — topology 사라짐).
Inconclusive:         Defect density 감소는 있으나 branch margin이 여전히 낮음 (수치적 불안정). 또는 seed 간 signed Jaccard가 낮아 (seed-specific random topology) structured emergence가 아님.
Required code:        topological/v2/topology.py: analyze_topology on C=1 field
                      topological/v2/evaluation.py: cross-seed signed Jaccard comparison
Required tests:       test_topology_v2.py: test_c1_trained_vs_untrained_density, test_c1_signed_jaccard_cross_seed
Raw artifact schema:  per-seed: {seed, model_type='scalar_u1', split='cal',
                      test_examples: [{example_id, post_write: TopologyStatsV2, pre_go: TopologyStatsV2, ...}]}
                      cross-seed: {seed_pairs: [(i,j)], signed_jaccard_matrix: [[float]]}
Compute formula:      C05 trained model 사용 → training cost 없음. Evaluation only:
                      5 seeds × 100 examples × topology_analysis / CPU ≈ 500 × ~0.02s ≈ 10s CPU
Dependency:            C05 (trained C=1 model 필요). C02 (untrained C=1 baseline 비교).
Claim enabled:        "C=1 U1ConvRNN 학습이 sparse하고 stable한 vortex topology를 출현시킨다" — C=1 causal analysis의 전제조건.
Claim prohibited:     "C=1 vortex가 causally relevant하다" — C07-C10 causal intervention 실험 필요.
```

---

### C07: Representative Sensitivity (대표 민감도)

```
Experiment ID:       C07
Scientific question: 동일 전하 지도(same charge map) 내에서 서로 다른 대표(representative) 선택이 vortex transplant 결과에 얼마나 큰 분산을 유발하는가?
Formal hypothesis:
  H0: Var_rep / Var_charge ≥ τ_rep (0.25) — 대표 선택이 전하 부류보다 결과를 더 크게 좌우한다.
  H1: Var_rep / Var_charge < τ_rep — 전하 부류가 대표 선택보다 dominant한 결정 요인이다.
Alternative explanations:
  - 대표 선택이 평활 질감(smooth texture)과 조화 섹터(harmonic sector)를 동시에 변화시키면, 분산이 단일 요인으로 귀속될 수 없다 (→ 3-way variance decomposition 필요).
  - 대표 생성 방법(method="harmonic_random")이 대표 공간을 충분히 cover하지 못하면 variance가 과소추정된다.
  - K=10 representatives가 부족하면 variance 추정치가 불안정하다 (bootstrap CI로 확인).
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair (within-seed nested)
Recipient selection:  Per seed, accuracy ≥ 0.95, magnitude > threshold, charge exists, decomposition valid → ~50 pairs
Donor selection:      Geometric selection function (signed count, energy, spectrum, displacement), 8 donors per recipient
Intervention arms:
  - vortex (canonical): canonical vortex field from donor → recipient (baseline)
  - vortex_rep_k for k=1..K: K=10 different same-charge representatives per donor-recipient pair
  - vortex_alternate_charge: different charge configuration (for Var_charge estimation)
Sham/controls:        N/A (representative variance가 control 대비 metric이 아니라 decomposition metric이므로)
Null families:        N/A (representative sensitivity는 IUT null family가 아닌 별도 gate)
Primary metric:       Var_rep = within-charge-class variance of donor-transfer metric M across K representatives. Var_charge = between-charge-class variance. Ratio: Var_rep / Var_charge.
                      Three-way decomposition: Var_total = Var_charge + Var_harmonic + Var_smooth + Var_noise (C13 결과와 결합 시).
Validity metrics:     Per-representative: charge_verification (Q(generated) == Q_target), harmonic_shift (Δw_x, Δw_y), displacement from canonical
Statistical test:     Bootstrap CI on Var_rep/Var_charge ratio. Gate: upper 95% CI bound < τ_rep (0.25).
Sample-size rule:     3 seeds × ~50 pairs × 10 representatives = ~1,500 transplant evaluations. K=10은 representative variance의 bootstrap SE가 수렴하기에 충분.
Pass:                 Var_rep / Var_charge < 0.25 with 95% CI upper bound < 0.25.
Fail:                 Var_rep / Var_charge ≥ 0.25 — 대표 선택이 전하 부류 이상으로 중요. "Vortex charge가 causal agent"라는 주장이 약화됨.
Inconclusive:         CI가 0.25를 포함 (0.20–0.30). K 증가 또는 추가 seeds로 해상도 향상 필요.
Required code:        topological/v2/interventions.py: sample_representatives(field, n=10, method='harmonic_random')
                      topological/v2/evaluation.py: per-representative transplant + metric computation
Required tests:       test_interventions_v2.py: test_representative_same_charge_verification, test_representative_variance_decomposition
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id, charge_map_hash,
                      representatives: [{rep_id, method, harmonic_shift, smooth_distance,
                      transplant_outcome: {donor_ll, recipient_ll, margin, ...}}]}
                      per-seed: {Var_charge, Var_rep, Var_harmonic (if available), ratio, ratio_CI}
Compute formula:      3 seeds × 50 pairs × (1 canonical + 10 representatives) × (decomposition + transplant + evaluation) / throughput
                      ≈ 3 × 50 × 11 × ~3s ≈ 4,950s ≈ 82 min (~1.4 GPU-hr using trained models, evaluation only)
Dependency:            C05 (trained model 필요). C01-C04에서 topology baseline 확립.
Claim enabled:        "Vortex transplant 효과가 특정 대표 선택이 아닌 topological charge class 자체에 기인한다" — representative invariance gate (C2) calibration.
Claim prohibited:     "모든 대표가 동일한 효과를 낸다" — variance가 0이 아님을 인정; 단지 charge variance 대비 작음만 주장.
```

---

### C08: Minimal Sufficiency (최소 충분성)

```
Experiment ID:       C08
Scientific question: 최소 변위(minimal-displacement) vortex transplant가 canonical component-replacement transplant와 유사한 behavioral effect를 내는가?
Formal hypothesis:
  H0: Minimal surgery effect < 0.10 × canonical surgery effect (최소 수술이 효과를 거의 완전히 상실).
  H1: Minimal surgery effect ≥ 0.10 × canonical surgery effect (최소 수술이 canonical 효과의 상당 부분을 보존).
Alternative explanations:
  - Canonical transplant의 큰 displacement가 off-manifold perturbation을 유발하고, 이것이 behavioral effect의 진짜 원인일 수 있다 (minimal이 훨씬 작으면 이 가설 지지).
  - Minimal surgery optimization이 local minimum에 trapped되어 charge constraint를 충족하지 못할 수 있다.
  - Gradient energy 보존과 spectral similarity 보존이 상충하여 feasible solution이 없을 수 있다.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair (paired within-recipient)
Recipient selection:  C07과 동일 selection pipeline (~50 pairs per seed)
Donor selection:      Geometric selection (C07과 동일)
Intervention arms:
  - vortex_canonical: canonical component-replacement transplant (baseline)
  - vortex_minimal: minimal-surgery transplant (projected gradient descent로 displacement 최소화, constraints: charge preservation, magnitude preservation, harmonic sector preservation, gradient energy ≤ 1.1×, spectrum similarity ≥ 0.95, manifold distance ≤ Q_0.95)
Sham/controls:        vortex_sham: same-displacement sham surgery (charge-preserving, matched displacement magnitude) — displacement effect vs charge effect 분리
Null families:        N/A
Primary metric:       Recovery ratio: r_minimal = M_minimal / M_canonical (정규화된 효과 보존 비율). μ(r_minimal) across pairs. Per-pair displacement ratio: ||z* − z_r|| / ||z_canonical − z_r||.
Validity metrics:     Charge_verification (Q(z*) == Q_target), harmonic_preservation (|Δw_x|, |Δw_y| < 0.1), energy_ratio ≤ 1.1, spectrum_error ≤ 0.05, manifold_distance ≤ Q_0.95(D_nat), optimization convergence
Statistical test:     One-sided paired bootstrap: H0: mean(r_minimal) < 0.10. Bootstrap CI lower bound on r_minimal.
Sample-size rule:     3 seeds × ~50 pairs × 2 arms (canonical, minimal) × 1 sham ≈ 300 transplant evaluations
Pass:                 Mean r_minimal ≥ 0.50 with lower CI bound ≥ 0.10. 최소 수술이 canonical 효과의 절반 이상을 보존.
Fail:                 Mean r_minimal < 0.10 — 최소 수술 효과가 거의 소멸. Canonical 효과가 displacement artifact일 가능성.
Inconclusive:         0.10 ≤ mean r_minimal < 0.50 — 효과가 일부 보존되나 상당 부분 소멸. Minimal surgery optimization 개선 여지.
Required code:        topological/v2/decomposition.py: transplant_vortex_minimal(recipient, target_charge_map, constraints)
                      Projected gradient descent optimizer with charge/harmonic/magnitude constraints
Required tests:       test_decomposition_v2.py: test_minimal_transplant_charge_verification, test_minimal_transplant_constraints, test_sham_surgery_displacement_match
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id,
                      canonical: {displacement, energy_ratio, harmonic_shift, manifold_distance, margin, ...},
                      minimal:   {displacement, energy_ratio, harmonic_shift, manifold_distance, margin, optimization_steps, convergence, ...},
                      sham:      {displacement, energy_ratio, harmonic_shift, manifold_distance, margin, ...},
                      r_minimal, r_sham}
Compute formula:      3 seeds × 50 pairs × (1 canonical + 1 minimal[~50 optimization steps] + 1 sham) × evaluation / throughput
                      ≈ 3 × 50 × ~150 × ~0.1s (per-step, amortized) ≈ 2,250s ≈ 38 min (~0.6 GPU-hr)
Dependency:            C07 (trained model + canonical baseline 재사용). C11 (manifold model for manifold constraint).
Claim enabled:        "Vortex transplant 효과가 displacement artifact가 아니라 topological charge change에 기인한다" — minimal surgery가 canonical과 comparable하면 강화.
Claim prohibited:     "Minimal surgery가 항상 feasible하다" — 일부 pair에서 optimization이 실패할 수 있음 (NO_DECOM_PAIR 증가 가능).
```

---

### C09: Local Necessity (국소 필요성)

```
Experiment ID:       C09
Scientific question: 단일 defect pair를 소멸(annihilate)시키면 donor-specific behavior가 국소적으로(position-specifically) 감소하는가?
Formal hypothesis:
  H0: 단일 pair annihilation이 position-uniform한 효과를 가지거나 효과가 없다 (Δmargin_pos ≤ 0 for target position).
  H1: Annihilation 효과가 해당 defect 근처 output position에 집중된다 (Δmargin_target > Δmargin_other).
Alternative explanations:
  - Defect pair가 spatially correlated되어 있어 하나를 제거해도 다른 defect가 보상할 수 있다.
  - Target pair 선택이 임의적이면 (가장 가까운 pair), causal relevance가 없는 pair를 고를 가능성.
  - Annihilation 수술 자체의 displacement가 local behavioral change를 유발할 수 있다 → sham control 필수.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair, per-output-position
Recipient selection:  C07과 동일, 추가 조건: 최소 2개 이상의 ± pair 보유 (annihilation target이 존재해야 함)
Donor selection:      Geometric selection
Intervention arms:
  - vortex (canonical, no removal): full vortex transplant (baseline)
  - vortex_remove_pair_target: 가장 가까운 ± pair 제거 후 vortex transplant
  - vortex_remove_pair_distant: 가장 먼 ± pair 제거 후 vortex transplant (spatial specificity 확인)
  - vortex_sham_pair: matched-surgery sham (동일 displacement, charge 보존) for each removal
Sham/controls:        vortex_sham_pair (각 removal arm에 matched)
Null families:        N/A
Primary metric:       Per-position margin change: ΔM_pos = M_removed_pos − M_canonical_pos. Spatial specificity ratio: ΔM_target / mean(ΔM_other_pos). Distance-decay correlation: corr(distance_from_defect, ΔM_pos).
Validity metrics:     Charge_verification (Q_removed == Q_original minus target pair), pair_identification_correct (정확한 ± pair 제거 확인), displacement magnitude (sham과 matched인지)
Statistical test:     One-sided paired bootstrap on spatial specificity ratio: H0: ratio ≤ 1.0. Bootstrap CI lower bound > 1.0이면 target position에 집중된 효과.
Sample-size rule:     3 seeds × ~30 eligible pairs (2+ pairs 보유) × 3 arms (canonical, remove_target, remove_distant) × 2 sham ≈ 360 evaluations
Pass:                 Spatial specificity ratio > 1.5 (target position이 다른 position보다 50% 더 큰 효과 감소). Distance-decay correlation < −0.3.
Fail:                 Ratio ≈ 1.0 (균일한 효과) — defect pair가 국소적 인과 효과를 가지지 않음. 또는 0에 가까움 (annihilation이 전혀 효과 없음).
Inconclusive:         Ratio > 1.0 but CI가 1.0을 포함. 또는 30 eligible pairs 미만으로 통계적 검정력 부족.
Required code:        topological/v2/decomposition.py: annihilate_pair(field, pair_position) → modified field
                      topological/v2/interventions.py: identify_target_pairs(field, n=1, strategy='closest'|'farthest')
Required tests:       test_decomposition_v2.py: test_annihilate_pair_charge_change, test_annihilate_pair_magnitude_preserved, test_annihilate_pair_harmonic_preserved
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id, target_pair_position, distant_pair_position,
                      canonical: {margin_per_position: [4], ...},
                      remove_target: {margin_per_position: [4], displacement, charge_verification, ...},
                      remove_distant: {margin_per_position: [4], displacement, charge_verification, ...},
                      sham_target: {...}, sham_distant: {...},
                      spatial_specificity_ratio, distance_decay_corr}
Compute formula:      3 seeds × 30 pairs × (1 canonical + 2 removal + 2 sham) × evaluation / throughput
                      ≈ 3 × 30 × 5 × ~3s ≈ 1,350s ≈ 23 min GPU
Dependency:            C08 (minimal surgery 구현에 기반). C07 (trained model + canonical baseline).
Claim enabled:        "Vortex defect가 국소적이고 position-specific한 인과 효과를 가진다" — local necessity evidence.
Claim prohibited:     "모든 defect가 causally necessary하다" — single-pair test는 sufficiency의 보완재일 뿐 global necessity를 증명하지 않음.
```

---

### C10: Donor Specificity (기증자 특이성)

```
Experiment ID:       C10
Scientific question: Vortex transplant가 recipient의 행동을 특정 donor 방향으로 이동시키는가, 아니면 아무 non-recipient 방향으로 이동시키는가?
Formal hypothesis:
  H0: Target donor gain ≤ max alternate donor gain (Δlog p(y_d) ≤ max_{j≠d} Δlog p(y_j)).
  H1: Target donor gain > max alternate donor gain (donor-specific behavioral shift).
Alternative explanations:
  - Vortex transplant가 단순히 recipient의 output distribution을 무작위화(entropy increase)시키면, 우연히 특정 donor가 선택될 수 있다 → entropy change control 필수.
  - Donor와 recipient가 유사한 output distribution을 가지면 (task-competent models), Δlog p(y_d) 자체가 작아 specificity 비교가 의미 없음.
  - Donor catalog의 다른 donor들이 유사한 output을 생성하면, max alternate gain도 높아져 specificity signal이 묻힘.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair (per-recipient, tested against all 8 donors)
Recipient selection:  C07과 동일 (~50 per seed). 추가: donor catalog 내 최소 4개 donor가 서로 다른 output token sequence를 가져야 specificity test가 meaningful.
Donor selection:      All 8 donors in catalog (no geometric filter — full donor set test)
Intervention arms:
  - vortex_d: vortex transplant from donor d, for d = 1..8 (all donors)
  - natural_recipient (no intervention, baseline)
  - whole_state_d: whole-state transplant from donor d (positive control)
Sham/controls:        natural_recipient (Δlog p 계산의 기준)
Null families:        N/A (specificity는 IUT null family가 아닌 behavioral metric decomposition)
Primary metric:       Specificity-adjusted donor margin: M_spec(r,d) = Δlog p(y_d) − max_{j≠d, j≠r} Δlog p(y_j).
                      Donor specificity rate: fraction of pairs where target donor is top-1 gain (rank=1 among all 8 donors).
                      Entropy change: ΔH(p) = H(p_intervened) − H(p_recipient).
Validity metrics:     Δlog p(y_d) > 0 (positive direction), Δlog p(y_r) < 0 (away from recipient), ΔH(p) small (targeted perturbation), donor rank ≤ 2 (strong specificity)
Statistical test:     Per-pair M_spec > 0 비율의 bootstrap CI. Entropy change vs M_spec correlation (큰 entropy change → 큰 specificity는 의심).
Sample-size rule:     3 seeds × 50 recipients × 8 donors = 1,200 transplant evaluations
Pass:                 Mean M_spec > 0 with lower CI > 0. Donor specificity rate (top-1 rank) > 0.30 (chance = 1/8 = 0.125).
Fail:                 Mean M_spec ≤ 0 — vortex transplant이 specific donor를 향하지 않음. 또는 entropy change가 M_spec과 강한 양의 상관관계.
Inconclusive:         M_spec > 0 but CI가 0을 포함. 또는 specificity rate가 chance보다 높지만 0.15–0.30 (약한 특이성).
Required code:        topological/v2/evaluation.py: compute_behavioral_outcome_v2 with 5-metric decomposition (§7.2)
Required tests:       test_evaluation_v2.py: test_donor_specificity_metric, test_entropy_change_small
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id,
                      per_donor: [{donor_id, donor_ll, recipient_ll, margin, delta_logp_yd, delta_logp_yr,
                      max_alt_gain, entropy_change, donor_rank}],
                      M_spec, specificity_pass}
Compute formula:      3 seeds × 50 recipients × 8 donors × (decompose + transplant + intervention_continuation + evaluation) / throughput
                      ≈ 1,200 × ~3s ≈ 3,600s ≈ 60 min GPU (~1.0 GPU-hr)
Dependency:            C07 (trained model + canonical transplant). C08 (minimal surgery가 specificity를 바꾸는지 비교 가능).
Claim enabled:        "Vortex transplant는 generic perturbation이 아니라 donor-specific information을 전달한다" — patch efficacy checklist criterion 2.
Claim prohibited:     "Vortex가 donor의 full identity를 encoding한다" — Δlog p(y_d) > 0이 whole_state 수준임을 의미하지 않음.
```

---

### C11: Manifold Projection (다양체 사영)

```
Experiment ID:       C11
Scientific question: 어떤 다양체 타당성 지표(manifold validity metric)가 자연적 은닉 상태(natural hidden state)와 off-manifold 상태를 가장 잘 구분하는가?
Formal hypothesis:
  H0: 모든 manifold metric이 natural state pool 내 분포와 off-manifold state 분포를 구분하지 못한다 (AUROC ≈ 0.5 for all candidates).
  H1: 적어도 하나의 manifold metric이 natural vs off-manifold를 유의미하게 구분한다 (AUROC > 0.7).
Alternative explanations:
  - Natural state pool의 크기/다양성이 부족하면 metric이 과도하게 optimistic한 separation을 보일 수 있다.
  - Off-manifold state를 무엇으로 정의하느냐에 따라 metric ranking이 달라진다 (random perturbation, adversarial perturbation, component-swapped state 등).
  - PCA basis가 training data에 overfit되면, test set natural state조차 off-manifold로 분류될 수 있다.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Hidden state (individual field, not pair)
Recipient selection:  Natural pool: training/validation hidden states ~10,000 states. Off-manifold states: (a) random Gaussian perturbation, (b) component-swapped reconstruction, (c) adversarial direction perturbation.
Donor selection:      N/A
Intervention arms:    N/A (manifold metric evaluation only)
Sham/controls:        N/A
Null families:        N/A
Primary metric:       AUROC for each candidate metric in separating natural vs off-manifold:
                        (1) nearest_natural_neighbor_distance: min_{h∈pool} ||h' − h||₂
                        (2) PCA_reconstruction_error: ||h' − PCA_project(h')||₂
                        (3) kNN_density_ratio: log(knn_dist_natural(h') / knn_dist_intervened(h'))
                        (4) relaxation_drift: ||F₀^T(h') − h'||₂ after T=5 blank recurrence steps
Validity metrics:     Natural pool coverage (PCA explained variance ratio at k components), kNN stability (k=5, 10, 20), relaxation convergence (drift at t=1,3,5)
Statistical test:     AUROC comparison across methods (DeLong test). Best method selection: max AUROC with penalty for complexity.
Sample-size rule:     3 seeds × ~3,000 natural states + 500 off-manifold states = ~10,500 states. AUROC estimation에 충분.
Pass:                 Best method AUROC > 0.80. Clear separation between natural distance distribution and off-manifold distance distribution.
Fail:                 모든 method AUROC < 0.60 — manifold metric이 유의미한 구분을 제공하지 못함. Manifold gate를 사용할 수 없음.
Inconclusive:         Best AUROC 0.60–0.80 — moderate separation. Gate threshold calibration이 어려움. Multiple methods의 ensemble 고려.
Required code:        topological/v2/interventions.py: manifold_distance(field, pca_model, knn_index), compute_relaxation_drift
                      topological/v2/evaluation.py: build_natural_state_pool, generate_off_manifold_states
Required tests:       test_interventions_v2.py: test_manifold_distance_natural_small, test_manifold_distance_off_manifold_large
Raw artifact schema:  per-method: {method_name, AUROC, AP, threshold_at_q95,
                      natural_distances: {mean, std, q50, q95, q99},
                      off_manifold_distances: {mean, std, q50, q95, q99}}
                      per-seed: {pca_explained_variance, knn_k, relaxation_convergence}
Compute formula:      3 seeds × (3000 natural states + 500 off-manifold) × (PCA project + kNN query + relaxation) / CPU
                      PCA basis training: ~30s (SVD on ~10k states × 2048-dim)
                      Distance computation: 10,500 × ~0.01s ≈ 105s
                      Total: ~2.5 min CPU
Dependency:            C05 (trained model의 training/validation states pool). C07 (component-swapped states를 off-manifold 예시로 사용 가능).
Claim enabled:        "선택된 manifold metric이 vortex-transplanted state의 on/off-manifold 판별에 사용될 수 있다" — manifold validity gate (C3) calibration.
Claim prohibited:     "Manifold metric이 완벽히 on/off-manifold를 구분한다" — 모든 metric은 연속적이며, threshold 선택에 따라 false positive/negative 존재.
```

---

### C12: Natural Neighbor Control (자연 이웃 통제)

```
Experiment ID:       C12
Scientific question: 자연적 hidden state pool에서 vortex topology가 유사한 nearest neighbor를 찾아 transplant했을 때, 합성적(synthetic) vortex transplant와 비교하여 어떤 behavioral effect를 보이는가?
Formal hypothesis:
  H0: Natural neighbor margin ≥ vortex transplant margin (자연 상태가 합성 transplant보다 donor-specific effect가 크거나 같다).
  H1: Vortex transplant margin > natural neighbor margin (합성 transplant가 자연 상태보다 더 강한 donor-specific effect를 낸다).
Alternative explanations:
  - Natural pool에 target topology와 정확히 일치하는 상태가 없으면, neighbor가 proxy에 불과하여 unfair comparison.
  - Natural neighbor도 donor의 hidden state이면, neighbor margin이 높을 수 있으나 그것은 donor identity의 자연적 전달이지 vortex의 인과적 역할이 아니다.
  - Nearest neighbor search가 high-dimensional space에서 unstable할 수 있다 (hubness 문제).
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair
Recipient selection:  C07과 동일 (~50 pairs per seed). 추가: natural pool이 충분히 커야 neighbor search meaningful.
Donor selection:      Geometric selection
Intervention arms:
  - vortex (canonical synthetic transplant)
  - natural_neighbor: natural state with closest topology (signed charge map Jaccard) + similar harmonic sector, from calibration natural pool
  - natural_neighbor_continue: natural neighbor state를 그대로 continuation (개입 없음, upper bound)
Sham/controls:        N/A (natural neighbor 자체가 quasi-control)
Null families:        N/A
Primary metric:       effect_difference = M_vortex − M_natural_neighbor. Pair-level bootstrap on mean difference.
                      Topology similarity: signed_Jaccard(vortex_target_charge, neighbor_charge), harmonic_sector_distance.
Validity metrics:     Neighbor topology similarity (signed Jaccard > 0.5), neighbor manifold distance (on-manifold by construction), neighbor behavioral baseline (M_neighbor > M_recipient?)
Statistical test:     One-sided paired bootstrap: H0: mean(effect_difference) ≤ 0. Bootstrap CI lower bound > 0.
Sample-size rule:     3 seeds × 50 pairs = 150 vortex-vs-neighbor comparisons
Pass:                 Vortex margin > natural neighbor margin significantly (lower CI > 0). Vortex transplant이 단순히 "유사 topology의 자연 상태를 찾는 것"보다 강한 인과 효과.
Fail:                 Natural neighbor margin ≥ vortex margin — 합성 transplant가 자연 상태보다 효과가 약하거나 동등. Vortex transplant의 인과적 추가 가치 없음.
Inconclusive:         CI가 0을 포함. 또는 neighbor topology similarity가 낮아 (signed Jaccard < 0.5) 비교 자체가 invalid.
Required code:        topological/v2/interventions.py: find_natural_neighbor(field, pool, target_topology, top_k=5)
Required tests:       test_interventions_v2.py: test_natural_neighbor_topology_similarity
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id,
                      neighbor_id, neighbor_source_seed, neighbor_signed_jaccard, neighbor_harmonic_distance,
                      vortex_outcome: {margin, donor_ll, ...},
                      neighbor_outcome: {margin, donor_ll, ...},
                      effect_difference}
Compute formula:      3 seeds × 50 pairs × (natural neighbor search + 2 evaluations) / throughput
                      Neighbor search: 50 × ~0.5s (kNN over ~10k pool) ≈ 25s
                      Evaluations: 150 × 2 × ~3s ≈ 900s
                      Total: ~15 min GPU baseline. Neighbor search는 CPU 최적화 가능.
Dependency:            C11 (natural state pool + manifold model). C07 (canonical transplant baseline).
Claim enabled:        "Vortex transplant 효과가 단순히 자연적 유사 상태로의 근사가 아니다" — on-manifold mechanism claim 강화.
Claim prohibited:     "Natural neighbor가 항상 존재한다" — 일부 topology는 natural pool에 없을 수 있음 (missingness 보고).
```

---

### C13: Harmonic Sector Intervention (조화 섹터 개입)

```
Experiment ID:       C13
Scientific question: 국소 vortex charge가 아닌 전역 조화 섹터(global harmonic sector, torus winding holonomy)를 교환했을 때 behavioral effect가 vortex transplant와 comparable한가?
Formal hypothesis:
  H0: harmonic_margin ≥ vortex_margin (조화 섹터 효과가 vortex 효과보다 크거나 동등 — 전역 holonomy가 진짜 인과 요인).
  H1: vortex_margin > harmonic_margin (국소 vortex charge가 전역 harmonic winding보다 더 강한 인과 효과).
Alternative explanations:
  - Harmonic sector swap이 vortex field 자체를 변경할 수 있다 (v_Q가 (0,0) harmonic sector 대표이므로, 다른 harmonic sector와 결합 시 canonical vortex field가 왜곡됨).
  - Harmonic decomposition 구현이 잘못되어 exact/harmonic 분리가 불완전하면, harmonic swap이 smooth component도 변경.
  - Harmonic sector가 행동에 전혀 영향을 주지 않으면 (margin ≈ 0), vortex와의 비교가 trivially 통과하지만 이는 harmonic이 irrelevant하다는 다른 claim.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair
Recipient selection:  C07과 동일 (~50 pairs per seed)
Donor selection:      Geometric selection
Intervention arms:
  - vortex: canonical vortex transplant (baseline)
  - harmonic: harmonic sector swap — donor의 (w_x, w_y)를 recipient에 이식, 동일 vortex charge 유지
  - harmonic_sham: harmonic swap with same harmonic sector (zero change, displacement check)
Sham/controls:        harmonic_sham (zero-effect harmonic swap for displacement calibration)
Null families:        harmonic은 IUT null family 중 하나. 이 실험은 harmonic null family의 calibration + effect size 추정을 겸함.
Primary metric:       harmonic_vs_vortex_difference = M_vortex − M_harmonic. Per-pair, per-seed.
Validity metrics:     Harmonic extraction correctness: (w_x, w_y) via cycle holonomy integration, consistency with FFT-based harmonic component. Harmonic preservation: H(z') == H_target.
Statistical test:     One-sided paired bootstrap: H0: mean(harmonic_vs_vortex_difference) ≤ 0. Bootstrap CI lower bound.
Sample-size rule:     3 seeds × 50 pairs × 2 arms (vortex, harmonic) ≈ 300 evaluations
Pass:                 Vortex margin significantly > harmonic margin (CI lower bound > 0). Harmonic margin > 0이면 harmonic도 일부 정보를 전달하나 vortex가 더 강함.
Fail:                 Harmonic margin ≥ vortex margin — 전역 topology가 국소 topology보다 강한 인과 효과. "Vortex charge"가 아니라 "harmonic holonomy"가 메커니즘이라는 대안 가설 지지.
Inconclusive:         두 margin이 유사하고 CI가 겹침. 또는 harmonic decomposition이 일부 pair에서 실패 (missingness).
Required code:        topological/v2/decomposition.py: full Hodge decomposition (exact + coexact + harmonic 분리), extract_harmonic_sector(field) → (w_x, w_y)
                      topological/v2/interventions.py: transplant_harmonic(donor_harmonic, recipient)
Required tests:       test_decomposition_v2.py: test_harmonic_extraction_torus, test_harmonic_swap_preserves_charge, test_harmonic_swap_cycle_holonomy
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id,
                      donor_harmonic: {wx, wy}, recipient_harmonic: {wx, wy},
                      vortex_outcome: {margin, ...},
                      harmonic_outcome: {margin, ...},
                      harmonic_vs_vortex_difference,
                      harmonic_sham_outcome: {margin, displacement, ...}}
Compute formula:      3 seeds × 50 pairs × (1 vortex + 1 harmonic + 1 sham) × evaluation / throughput
                      ≈ 3 × 50 × 3 × ~3s ≈ 1,350s ≈ 23 min GPU
Dependency:            C06 (수학적 기반 §6.4: full Hodge decomposition with harmonic sector 분리 구현 필요). C07 (canonical vortex baseline).
Claim enabled:        "국소 vortex charge가 전역 harmonic winding보다 행동에 더 큰 인과적 영향을 미친다" — IUT harmonic null family에 대한 priors.
Claim prohibited:     "Harmonic sector가 행동에 무관하다" — harmonic margin > 0일 가능성 열어둠.
```

---

### C14: Multichannel Phase Locking (다채널 위상 잠금)

```
Experiment ID:       C14
Scientific question: Trained U1ConvRNN (C=8)에서 채널 간 위상이 상관되어 있는가 (phase-locked submanifold 형성)?
Formal hypothesis:
  H0: Inter-channel phase correlation ≤ 0.8 (채널 간 위상이 독립적 — C>1에서 well-defined topology 부재).
  H1: Inter-channel phase correlation > 0.8 (위상잠금 부분다양체 존재 — C>1에서 채널별 plaquette charge가 근사적 topological object로 해석 가능).
Alternative explanations:
  - Phase correlation이 training이 아니라 initialization에서 비롯될 수 있다 (PlainConvRNN과 비교).
  - High correlation이 특정 spatial scale에만 국한될 수 있다 (local vs global phase coherence).
  - Correlation이 magnitude-weighted되어야 의미 있다 (near-zero magnitude 채널은 phase가 무의미).
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8), 3 trained seeds each from calibration
Task:                 Copy task
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Hidden state (per-example, per-position)
Recipient selection:  N/A (all test example hidden states, ~100 per seed)
Donor selection:      N/A
Intervention arms:    N/A (observational analysis only)
Sham/controls:        PlainConvRNN (C=8) — same channel count, no U(1) equivariance → 낮은 phase correlation 예상
Null families:        N/A
Primary metric:       Inter-channel phase correlation matrix (C×C) averaged over examples and spatial positions. PCA on phase space (per spatial position: C-dimensional phase vector → top-k explained variance ratio). Effective dimension: 1 / Σ λ_i² (normalized eigenvalues). Phase coherence: mean resultant length of channel phases at each spatial position.
Validity metrics:     Minimum magnitude filter (exclude positions where any channel has |z| < threshold, phase undefined). PlainConvRNN baseline (낮은 correlation이 예상되며, 높게 나오면 metric이 artifact).
Statistical test:     U1 phase correlation vs Plain phase correlation: two-sample bootstrap on mean correlation. U1의 effective dimension이 유의미하게 1에 가까운지 (bootstrap CI).
Sample-size rule:     3 seeds × 2 model types × 100 examples × 256 spatial positions = 153,600 phase vectors. Correlation matrix 안정화에 충분.
Pass:                 U1 mean inter-channel phase correlation > 0.8, Plain < 0.5. U1 effective dimension < 1.5 (C=8임에도 near-1D manifold).
Fail:                 U1 correlation < 0.5 (채널 간 독립적). C>1에서 channelwise charge가 진정한 topological object가 아님 — C=1 gateway에 의존해야 함.
Inconclusive:         U1 correlation 0.5–0.8 — moderate locking. C>1 분석 가능하나 channel-mode ambiguity 존재. Conservative approach: C=1 primary, C=8 secondary.
Required code:        topological/v2/topology.py: inter_channel_phase_correlation(field), phase_pca_effective_dimension(field)
Required tests:       test_topology_v2.py: test_phase_correlation_u1_gt_plain, test_effective_dimension_near_one
Raw artifact schema:  per-seed: {seed, model_type, mean_correlation, correlation_matrix: [C×C],
                      effective_dimension, per_example_correlation: [{example_id, mean_corr, eff_dim}],
                      plain_baseline: {mean_correlation, effective_dimension}}
Compute formula:      3 seeds × 2 models × 100 examples × (~256 phase comparisons + PCA) / CPU
                      ≈ 600 × ~0.05s ≈ 30s CPU (분석은 purely observational, GPU 불필요)
Dependency:            C05 (trained model). C06 (C=1 topology context). C02 (untrained correlation baseline).
Claim enabled:        "Trained U1ConvRNN은 phase-locked submanifold를 형성하며, C>1에서도 well-defined topology를 가진다" — C>1 분석의 수학적 정당화.
Claim prohibited:     "Phase locking이 U(1) equivariance의 필연적 결과다" — 학습된 특성일 수 있으며, C05의 C=1과 구분되는 다채널-specific 현상.
```

---

### C15: Channel Basis Robustness (채널 기저 강건성)

```
Experiment ID:       C15
Scientific question: Vortex transplant 결과가 채널 순열(channel permutation)이나 무작위 유니터리 혼합(random unitary mixing)에 강건한가? 
                      구분: hidden-state perturbation (채널 기저 변경 후 재분해) vs function-preserving model reparameterization (모델 가중치에 동일 유니터리 적용).
Formal hypothesis:
  H0: Channel permutation/unitary mixing 후에도 vortex transplant 효과가 원본 채널 기저와 동일하다 (δ_perm = 0, basis-independent).
  H1: Vortex transplant 효과가 원본 채널 기저에서 가장 크다 (δ_perm < 0, basis-specific — 학습된 채널 구조가 topology를 encoding).
Alternative explanations:
  - Hidden-state perturbation (사후 채널 섞기)가 manifold를 벗어나게 하면, 효과 감소가 basis-specificity가 아니라 off-manifold artifact.
  - Function-preserving model reparameterization (모델 가중치에 동일 permutation 적용 후 재학습/재추론)이 진정한 basis-independence test — hidden-state perturbation과 구분 필요.
  - Random unitary mixing이 C>1 topology를 파괴할 수 있으며 (π₁ = 0), 이것이 C=1 gateway의 필요성을 재확인.
Model:               U1ConvRNN (C=8), 3 trained seeds from calibration
Task:                 Copy task, test split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default)
Split:                Calibration
Unit of analysis:     Donor-recipient pair
Recipient selection:  C07과 동일 (~50 pairs per seed)
Donor selection:      Geometric selection
Intervention arms:
  【Hidden-state perturbation path】 (모델 가중치 불변):
  - vortex_original: 원본 채널 기저 vortex transplant (baseline)
  - vortex_permuted: 채널 순열 (random permutation of 8 channels) 적용 후 vortex transplant → 재분해
  - vortex_unitary: 무작위 Haar-distributed U(8) unitary 혼합 적용 후 vortex transplant → 재분해 (B=20 random unitaries per pair)
  【Function-preserving reparameterization path】 (모델 가중치 변경):
  - vortex_reparam: 모델 가중치에 채널 순열 적용 (equivariant reparameterization) 후 동일 입력에 대한 hidden state 추출 → vortex transplant → 원본 평가 metric으로 측정
Sham/controls:        vortex_original + manifold distance check for permuted/unitary states (off-manifold 여부 확인)
Null families:        N/A (channel basis robustness는 IUT null family가 아닌 sensitivity analysis)
Primary metric:       Margin degradation: ΔM = M_perturbed − M_original (음수 기대). Per-arm mean ΔM.
                      Manifold distance: d_M(h_perturbed) vs d_M(h_original). Correlation between ΔM and manifold distance increase.
Validity metrics:     Permutation/unitary preserves net_charge_zero, magnitude distribution, and total field energy. Reparameterization preserves model output (accuracy unchanged).
Statistical test:     One-sided paired bootstrap per perturbation type: H0: ΔM ≥ 0 (no degradation). Bootstrap CI upper bound < 0이면 유의미한 degradation.
Sample-size rule:     3 seeds × 50 pairs × (1 original + 1 permuted + 20 unitary × 1 reparam) ≈ 3 × 50 × 22 ≈ 3,300 evaluations
Pass:                 Permutation degradation small (ΔM > −0.1 × M_original, CI includes 0) — vortex effect가 channel basis에 robust. Unitary degradation도 유사하게 작음. Reparameterization 결과가 hidden-state perturbation과 일치.
Fail:                 Permutation 또는 unitary mixing 시 vortex 효과가 크게 감소하거나 소멸 — 효과가 학습된 특정 채널 기저에 의존. C>1 topology 해석의 validity 의심.
Inconclusive:         Hidden-state perturbation은 degradation 보이나, reparameterization은 보존 — off-manifold artifact 가능성 (C11 manifold gate로 확인).
Required code:        topological/v2/interventions.py: permute_channels(field, permutation), apply_unitary_mixing(field, unitary)
                      topological/v2/model.py: reparameterize_model_channels(model, permutation) — equivariant weight transformation
Required tests:       test_interventions_v2.py: test_permutation_preserves_charge, test_permutation_preserves_energy
                      test_model_v2.py: test_reparameterization_output_identical, test_reparameterization_equivariant
Raw artifact schema:  per-pair: {seed, recipient_id, donor_id,
                      original: {margin, manifold_distance, ...},
                      permuted: {permutation_seed, margin, margin_delta, manifold_distance},
                      unitary: [{unitary_seed, margin, margin_delta, manifold_distance} × 20],
                      reparam: {margin, margin_delta}}
                      per-arm: {mean_delta, delta_CI, correlation_with_manifold_distance}
Compute formula:      3 seeds × 50 pairs × (1 original + 1 perm + 20 unitary + 1 reparam) × evaluation
                      ≈ 3 × 50 × 23 × ~3s ≈ 10,350s ≈ 173 min ≈ 2.9 GPU-hr
                      (Unitary B=20이 dominant. Calibration 후 confirmatory에서는 B=5로 축소 가능)
Dependency:            C11 (manifold model — off-manifold artifact 판별). C14 (phase locking 결과 — 강한 locking이면 degradation이 작을 것).
Claim enabled:        "C>1 vortex effect가 특정 채널 기저의 인공물이 아니라 genuinely topological하다" — C=8 분석의 internal validity.
Claim prohibited:     "채널 기저 robustness가 C=1의 필요성을 제거한다" — C=1은 channel-mode ambiguity를 원천 제거하는 수학적 gateway.
```

---

### C16: Factorial Baseline (요인 기저선 — 2×2 설계)

```
Experiment ID:       C16
Scientific question: U1ConvRNN의 vortex transplant 효과에서 U(1)-commuting linear convolution과 radial nonlinearity가 각각 어느 정도 기여하는가?
Formal hypothesis:
  H0: 4개 factorial variant 모두에서 comparable한 causal vortex effect가 관찰된다 (δ_i ≈ δ_j for all i,j).
  H1: Full U(1)-equivariant variant (U1CommutingLinear × RadialNonlinear)가 가장 강한 vortex 효과를 보이며, equivariance component가 하나라도 빠지면 효과가 감소한다.
Alternative explanations:
  - Elementwise nonlinearity variant가 학습에 실패하면 (low accuracy), 공정한 비교가 불가.
  - 각 variant의 parameter count가 다르면 capacity 차이로 인한 confound.
  - Vortex effect가 전적으로 radial nonlinearity에 기인할 경우, RealLinear × RadialNonlinear가 U1과 유사한 효과를 보일 것.
Model:               2×2 Factorial variants (all C=8):
                      (A1,B1): U1CommutingLinear_RadialNonlinear (= U1ConvRNN)
                      (A1,B2): U1CommutingLinear_ElementwiseNonlinear
                      (A2,B1): RealLinear_RadialNonlinear
                      (A2,B2): RealLinear_ElementwiseNonlinear (= PlainConvRNN)
Task:                 Copy task, delay_range=[16,32], vocab=10, copy_length=4
Input parameterization: Standard token embedding (V1 default). 참고: embedding lookup은 U(1)-equivariant가 아니므로 모든 variant 동일.
Split:                Calibration
Unit of analysis:     Seed (3 seeds per variant)
Recipient selection:  C07 pipeline (~50 pairs per seed, per variant)
Donor selection:      Geometric selection (variant-agnostic — 동일 selection function)
Intervention arms:    vortex (canonical) for each variant
Sham/controls:        C07 null families (10개의 null family per variant — 모든 variant에서 동일 null family set)
Null families:        Full null family set {smooth, magnitude, global_phase, zero_charge_phase, fourier_low, fourier_high, pca, random_direction, harmonic, charge_arrangement_shuffle}
                      — per variant, per family
Primary metric:       Per-variant Δ_mech (IUT primary estimand). Interaction effect: ANOVA-style decomposition:
                      M(vortex) ~ μ + α_linear + β_nonlinear + γ_interaction + ε
                      where α = U1CommutingLinear − RealLinear, β = RadialNonlinear − ElementwiseNonlinear.
Validity metrics:     Per-variant training convergence (모든 variant가 ≥90% validation accuracy, convergence 확인), per-variant analyzable fraction ψ_analyze
Statistical test:     Per-variant IUT. Cross-variant comparison: bootstrap CI on (U1 − variant) for each variant.
                      Factorial decomposition: α, β, γ interaction estimate with bootstrap CI.
Sample-size rule:     3 seeds × 4 variants = 12 trained models. Per variant: ~50 pairs × 10 null families = ~500 data points.
Pass:                 U1 IUT 통과, Plain IUT 실패 (C6). U1이 가장 큰 Δ_mech. α > 0 (linear equivariance effect), β > 0 (radial nonlinearity effect).
Fail:                 모든 variant에서 IUT 실패 (C1 false for all) — vortex effect가 전반적으로 부재. 또는 Plain이 U1과 유사한 효과 (C6 false).
Inconclusive:         일부 variant만 IUT 통과 — equivariance component의 contribution이 ambiguous. Interaction effect의 CI가 0을 포함.
Required code:        topological/v2/model.py: ComplexNoEquivConvRNN (A1,B2), RealWithEquivConvRNN (A2,B1)
Required tests:       test_model_v2.py: test_factorial_variants_forward, test_factorial_equivariance_gradient
                      test_evaluation_v2.py: test_factorial_iut_per_variant
Raw artifact schema:  per-variant, per-seed: standard V2 artifact schema (training.json, model.pt, evaluation/*.json)
                      cross-variant: {variant, Δ_mech, Δ_mech_CI, per_family_margins,
                      ψ_analyze, training_accuracy, interaction_effect}
Compute formula:      Training: 3 seeds × 4 variants × 30k updates × 28 steps × batch 64 / GPU_throughput
                      ≈ 12 × 30,000 × 28 × 64 / ~6400 steps/s ≈ 12 × ~8,400s ≈ 100,800s ≈ 28 GPU-hr
                      Evaluation: 12 seeds × ~50 pairs × 10 null families × ~3s ≈ 18,000s ≈ 5 GPU-hr
                      Total: ~33 GPU-hr (가장 compute-intensive calibration 실험)
Dependency:            C05-C15 (모든 calibration 결과가 factorial 설계의 해석에 context 제공).
                      C14 (phase locking이 variant 간 어떻게 다른지). C07 (canonical effect size baseline).
Claim enabled:        "U1ConvRNN의 vortex 효과는 linear U(1)-commutation과 radial nonlinearity의 결합에 고유하며, 단일 component만으로는 재현되지 않는다" — mechanism specificity claim.
Claim prohibited:     "U1ConvRNN 이외의 architecture는 vortex mechanism을 가질 수 없다" — factorial 설계는 4개 variant만 test; 다른 architecture는 open question.
```

---

## Confirmatory Phase (F01–F05)

Confirmatory phase의 모든 실험은 frozen contract 하에서 실행된다. Calibration이 완료되고 PI가 contract을 freeze한 후에만 실행 가능. 모든 parameter, threshold, selection function, sample size는 사전등록된다.

---

### F01: Confirmatory Pilot (확인적 본실험)

```
Experiment ID:       F01
Scientific question: U1ConvRNN에서 vortex transplantation이 donor-specific output을 causally control하는가? (representative-invariant, manifold-valid, 모든 null family 능가)
Formal hypothesis:
  H0: max_f p_f > 0.05 (적어도 하나의 null family에 대해 vortex가 능가하지 못함) — IUT global null.
  H1: max_f p_f ≤ 0.05 AND Var_rep/Var_charge < τ_rep AND median(d_M(H')) ≤ Q_0.95(D_nat) AND ψ_analyze ≥ 0.70
      — vortex가 모든 null family를 능가하고, 대표 불변이며, 다양체 위에 있고, 분석 가능 충분.
Alternative explanations:
  - Selection function이 effect modifier로 작용하여 selected-subpopulation에서만 효과 — all-admissible sensitivity로 확인 (§7.10).
  - Missingness가 differential하게 작용 (특정 arm에서 더 많은 실패) — pattern-mixture sensitivity (§7.7).
  - PlainConvRNN도 IUT를 통과하면 (C6 false), vortex 효과가 U(1)-specific하지 않음 → INCONCLUSIVE_BASELINE.
Model:               U1ConvRNN (C=8), N_confirm seeds (calibration power analysis로 결정)
Task:                 Copy task, confirmatory split (heldout delay=64)
Input parameterization: Standard token embedding (V1 default), calibration에서 결정된 variant
Split:                Confirmatory (frozen — no peeking)
Unit of analysis:     Seed (N_confirm independent seeds, sole generalization unit)
Recipient selection:  Frozen selection pipeline: accuracy ≥ 0.95 → magnitude > threshold → charge exists → decomposition valid → donor exists. Per seed: 100 test examples.
Donor selection:      Frozen geometric selection function (calibration에서 확정된 signed count, energy, spectrum, displacement criteria)
Intervention arms:
  Primary: vortex (canonical vortex transplant from donor to recipient)
  Sufficiency: vortex_minimal, vortex_alternate (K representatives = calibration-validated K)
  Necessity: vortex_remove_all, vortex_remove_pair, vortex_sham
  Specificity: vortex_sign_flip
  Harmonic: harmonic_swap
  Natural: natural_recipient, whole_state
  Null families (10 families × per-family null draws): smooth, magnitude, global_phase, zero_charge_phase,
    fourier_low, fourier_high, pca, random_direction (B≥199), harmonic, charge_arrangement_shuffle (B≥199)
Sham/controls:        vortex_sham (per-pair), natural_recipient (baseline), whole_state (positive control)
Null families:        Full 10-family set (frozen list from contract §7.9). Per-family null draws: calibration-validated B (≥199).
Primary metric:       Δ_mech: seed-level vortex-over-max-null-f advantage. Per-family δ_f = E_s[M_vortex − M_f].
                      Donor-transfer metric M (specificity-adjusted, §7.2): M(r,d) = Δlog p(y_d) − max_{j≠d,r} Δlog p(y_j).
                      ψ_analyze: per-seed analyzable fraction.
Validity metrics:     Decomposition reconstruction error < 1e-10, net_charge_zero, min(|field|) > magnitude_threshold,
                      branch_margin_q01 > stability_threshold (calibration-frozen), whole_state_sanity (WS_margin > 0),
                      directional_sanity (<10% pairs with negative denominator)
Statistical test:     Primary: IUT with hierarchical bootstrap (B=9999, seed-level resampling), α=0.05.
                      Per-family: one-sided bootstrap p-value p_f. Reject H0 iff max_f p_f ≤ 0.05.
                      Simultaneous CI: max-T bootstrap (optional, for reporting).
                      Validity gates: C2 (Var_rep/Var_charge < τ_rep), C3 (median d_M ≤ Q_0.95).
Sample-size rule:     N_confirm determined by calibration power analysis (§8.4). Target: power ≥ 0.80 at SESOI.
                      Simulation-based power curve with calibration variance estimates.
                      Minimum 20 seeds. Candidate range: 20–100.
Pass:                 GO_CONFIRMATORY (§8.7): C1 ∧ C2 ∧ C3 ∧ C4 ∧ C6 ∧ C7.
                      C1: max_f p_f ≤ 0.05. C2: representative gate 통과. C3: manifold gate 통과.
                      C4: ψ_analyze ≥ 0.70. C6: Plain IUT 실패. C7: U1 > Plain cross-model.
Fail:                 NO_GO_MECHANISM: C1 false (IUT 실패). 또는 C5 false (ψ_analyze < 0.50).
Inconclusive:         INCONCLUSIVE_REPRESENTATIVE (¬C2), INCONCLUSIVE_MANIFOLD (¬C3),
                      INCONCLUSIVE_STATISTICS (¬C5), INCONCLUSIVE_POWER (C4 false, C5 true),
                      INCONCLUSIVE_BASELINE (C6 false), 등 (§8.7.2)
Required code:        topological/v2/pilot.py: run_confirmatory_pilot (frozen contract executor)
                      topological/v2/evaluation.py: full per-family IUT pipeline
                      topological/v2/_contract.py: load/verify frozen contract
Required tests:       test_evaluation_v2.py: test_iut_type1_error, test_iut_seed_level_bootstrap, test_confirmatory_no_peeking
                      test_contract_v2.py: test_contract_hash_verification, test_split_isolation
Raw artifact schema:  Per-seed: {seed, model_type='u1', split='confirm',
                      participants_flow: {accuracy_pass, magnitude_pass, charge_pass, decomposition_pass, donor_pass → ψ},
                      pairs.json: {pair_id, recipient_id, donor_id, donor_selection_reason,
                        per_arm: {arm_name, donor_ll, recipient_ll, margin, M_spec, manifold_distance,
                                  representative_id (if applicable), sham_displacement, ...}},
                      outcomes.json: {per_family: {family_name, per_pair_margins: [...], seed_mean_margin}},
                      summary.json: {ψ_analyze, Δ_mech, per_family_δ_f, per_family_p_f, iut_max_p,
                                     Var_rep/Var_charge, median_manifold_distance, ρ_sel,
                                     Δ_mech_composite, tipping_point Δ_c}}
                      Per-split: decision.json {verdict, all_conditions, trace}
Compute formula:      Training: N_confirm × 30k × 28 × 64 / throughput.
                      Evaluation: N_confirm × ~50 analyzable pairs × (1 vortex + 10 null families × B≥199 draws + ~8 other arms) × evaluation_time / throughput.
                      Training ≈ N × ~14 min. Evaluation ≈ N × ~50 × ~2000 × ~0.02s (amortized) ≈ N × ~33 min.
                      Total per seed ≈ ~47 min. At N=50: ~39 GPU-hr. At N=100: ~78 GPU-hr.
Dependency:            ALL calibration experiments (C01–C16) 완료 + contract frozen + PI sign-off.
                      F02–F05는 F01과 동일 confirmatory split에서 실행 (seeds 공유).
Claim enabled:        "U1ConvRNN hidden state의 vortex defect가 output behavior를 causally control하며, 그 효과는 모든 null perturbation을 능가하고, 대표-불변이며, 자연 다양체 위에 있다."
Claim prohibited:     "Vortex가 copy task computation의 유일한 또는 필수적인 메커니즘이다" — sufficiency만 검증, necessity는 C09로 제한적 검증.
```

---

### F02: Cross-Model Comparison (교차 모델 비교)

```
Experiment ID:       F02
Scientific question: U1ConvRNN의 causal vortex effect가 PlainConvRNN의 vortex effect보다 통계적으로 유의미하게 큰가?
Formal hypothesis:
  H0: δ_U1 − δ_Plain ≤ 0 (U1 vortex advantage가 Plain보다 크지 않다 — U(1) equivariance가 추가적 이득 없음).
  H1: δ_U1 − δ_Plain > 0 (U1ConvRNN이 PlainConvRNN보다 유의미하게 큰 vortex causal effect를 보인다).
Alternative explanations:
  - U1과 Plain이 다른 training dynamics를 가지면, accuracy나 convergence 차이가 confound.
  - PlainConvRNN에서 C>1 topology가 정의되지 않으면 (C14 결과), Plain에서의 "vortex transplant" 자체가 invalid → cross-model 비교 불가.
  - Plain의 vortex effect가 0이면 비교는 trivial하지만, U1만의 효과라는 specificity claim은 강화.
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8) — F01과 동일 confirmatory seeds
Task:                 Copy task, confirmatory split (heldout delay=64)
Input parameterization: Both models: standard token embedding (matched)
Split:                Confirmatory (frozen)
Unit of analysis:     Paired seed (U1_seed_i, Plain_seed_i share same input sequences)
Recipient selection:  F01 frozen pipeline, per model type
Donor selection:      Frozen geometric selection (per model type)
Intervention arms:    F01 full arm set, per model type
Sham/controls:        Per-model sham controls
Null families:        F01 full 10-family set, per model type
Primary metric:       Cross-model difference: Δ_cross = Δ_mech(U1) − Δ_mech(Plain). Per-family cross-model difference: δ_f(U1) − δ_f(Plain).
Validity metrics:     Per-model accuracy matched (both ≥ 0.95), per-model ψ_analyze comparable, per-model training convergence
Statistical test:     Paired hierarchical bootstrap (seed-level pairing): one-sided p_cross ≤ 0.05.
                      Equivalence test (TOST) with pre-specified cross-model SESOI if one-sided not rejected.
Sample-size rule:     N_confirm (F01과 동일). Paired design이 variance를 줄이므로 F01과 동일 sample size로 충분.
Pass:                 p_cross ≤ 0.05 (U1 > Plain). Plain IUT fails (C6 true) — U1만 causal vortex effect 보유.
Fail:                 p_cross > 0.05 (U1 ≈ Plain) — cross-model 차이 없음.
                      If Plain also passes IUT (C6 false) → INCONCLUSIVE_BASELINE (두 모델 모두 vortex effect).
Inconclusive:         p_cross > 0.05 but equivalence test TOST rejects (difference within SESOI) → practically equivalent.
                      TOST cannot reject → underpowered for equivalence.
Required code:        F01 pipeline + paired cross-model bootstrap aggregator
Required tests:       test_evaluation_v2.py: test_cross_model_paired_bootstrap, test_cross_model_type1_error
Raw artifact schema:  Per-model: F01 schema. Cross-model: {Δ_cross, Δ_cross_CI, p_cross, per_family_cross_difference,
                      cross_model_equivalence_margin, tost_result, plain_iut_passed (C6)}
Compute formula:      F01 × 2 (U1 + Plain). Training: 2 × N × ~14 min. Evaluation: 2 × N × ~33 min.
                      Total: 2 × N × ~47 min. At N=50: ~78 GPU-hr. At N=100: ~157 GPU-hr.
Dependency:            F01 (동일 confirmatory split). C16 (calibration cross-model priors).
Claim enabled:        "U(1)-equivariant architecture가 PlainConvRNN보다 유의미하게 강한 causal vortex effect를 생성한다" — architecture specificity.
Claim prohibited:     "PlainConvRNN에는 vortex mechanism이 전혀 없다" — effect가 0이 아니더라도 U1보다 작으면 H1 성립.
```

---

### F03: Delay OOD Generalization (지연 OOD 일반화)

```
Experiment ID:       F03
Scientific question: Causal vortex effect가 training delay range [16,32]를 벗어난 OOD delay (128, 256)에서도 지속되는가?
Formal hypothesis:
  H0: OOD delay (128, 256)에서 max_f p_f > 0.05 — vortex effect가 OOD로 일반화되지 않는다.
  H1: OOD delay에서도 max_f p_f ≤ 0.05 — vortex effect가 training distribution 밖의 delay로 일반화된다.
Alternative explanations:
  - Delay가 길어질수록 model accuracy 자체가 저하되어, analyzable pair 수가 감소 (ψ_analyze ↓). Missingness가 증가하면 F01 대비 검정력 저하.
  - Delay 64 (F01 primary)가 이미 OOD이므로, 128/256은 "farther OOD" — 실패해도 primary claim에 영향 없음.
  - Long delay에서 hidden state dynamics가 qualitatively 달라져 topology 자체가 변할 수 있음.
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8) — F01 confirmatory trained models (delay [16,32]로 학습됨)
Task:                 Copy task, confirmatory split, OOD delays: 128, 256
Input parameterization: Standard token embedding. 새 delay로 test example 생성 (동일 hash namespace, split='confirm', delay ∈ {128, 256})
Split:                Confirmatory (frozen)
Unit of analysis:     Seed (F01과 동일 seeds)
Recipient selection:  F01 frozen pipeline, delay=128/256 test examples only
Donor selection:      Frozen geometric selection
Intervention arms:    F01 subset (vortex, primary null families only — full 10-family set)
                      Note: OOD evaluation에 전체 arm set을 적용하면 compute 폭증 → arm 수 축소 가능 (contract에 사전등록)
Sham/controls:        natural_recipient, whole_state (positive control for OOD accuracy check)
Null families:        Full 10-family set
Primary metric:       Per-delay Δ_mech_OOD(delay). OOD degradation: Δ_mech(128)/Δ_mech(64), Δ_mech(256)/Δ_mech(64).
Validity metrics:     Per-delay model accuracy (accuracy ≥ 0.50 at delay=256이면 분석 가능, 그 이하이면 task failure로 배제),
                      per-delay ψ_analyze (OOD filtering으로 더 낮을 것)
Statistical test:     Per-delay IUT (α=0.05, no correction — 별도 secondary hypothesis).
                      OOD degradation: bootstrap CI on ratio Δ_mech(delay)/Δ_mech(64).
Sample-size rule:     N_confirm (F01과 동일 seeds). Per-delay test examples: 100 (F01과 동일).
Pass:                 Both delay=128 and delay=256 IUT reject (max_f p_f ≤ 0.05 each). OOD degradation ratio > 0.5 at delay=256.
Fail:                 IUT fails at delay=256 — OOD generalization 없음. Primary claim은 delay=64로 유지.
Inconclusive:         delay=128은 통과, delay=256은 실패. 또는 ψ_analyze가 너무 낮아 (< 0.30) 통계적 검정력 부족.
Required code:        F01 pipeline + delay parameter override
Required tests:       test_evaluation_v2.py: test_ood_delay_iut, test_ood_psi_analyze
Raw artifact schema:  Per-delay: F01 summary.json schema. OOD summary: {delay, Δ_mech, iut_max_p, ψ_analyze,
                      degradation_ratio, accuracy, plain_iut_result}
Compute formula:      F01 evaluation × 2 delays. Training 재사용 (F01 trained models).
                      N × 2 delays × ~33 min ≈ N × ~66 min. At N=50: ~55 GPU-hr.
Dependency:            F01 (동일 confirmatory split + trained models).
Claim enabled:        "Vortex causal effect가 OOD delay로 일반화된다" — robustness claim.
Claim prohibited:     "Vortex effect가 모든 delay에서 일정하다" — delay 증가에 따른 degradation 허용.
```

---

### F04: Direct Predecessor Replication (직접 선행연구 재현)

```
Experiment ID:       F04
Scientific question: V2 pipeline이 Iqbal et al. (2026)의 vortex transplant setup을 정확히 재현하는가?
Formal hypothesis:
  H0: V2 pipeline의 vortex transplant effect size가 predecessor pipeline과 systematic하게 다르다 (|Δ_mech_V2 − Δ_mech_pred| > equivalence_margin).
  H1: V2 pipeline이 predecessor pipeline과 equivalent한 결과를 생산한다 (|Δ_mech_V2 − Δ_mech_pred| ≤ equivalence_margin).
Alternative explanations:
  - V2의 pipeline 변화 (branch margin check, manifold gate, representative invariant 등 V2 addition)가 predecessor 대비 pair filtering을 강화하여 analyzable pair 수를 줄일 수 있다.
  - Predecessor는 V1 codebase에서 실행되었으며, V2의 topological/v2/ 패키지는 parallel implementation — implementation 차이가 결과 차이를 유발할 수 있다.
  - Replication 실패가 V2의 문제인지 predecessor의 문제인지 구분이 필요.
Model:               U1ConvRNN (C=8), PlainConvRNN (C=8) — F01 confirmatory seeds 재사용
Task:                 Copy task, matched to predecessor setup: copy_length=4, vocab=10, delay=64 evaluation
Input parameterization: Predecessor-matched: standard V1 full-spatial random embedding (charge-free smooth embedding이 아님)
Split:                Confirmatory (frozen, 별도 replication namespace optional)
Unit of analysis:     Seed
Recipient selection:  Predecessor-matched: V1 selection pipeline (accuracy ≥ 0.95, minimum magnitude, charge exists).
                      No V2 additions: branch_margin gate OFF, manifold gate OFF, representative gate OFF.
Donor selection:      Predecessor-matched: V1 geometric selection (동일 signed count, energy, spectrum criteria)
Intervention arms:    Predecessor-matched (V1 arm set: vortex, smooth, magnitude, whole_phase, global_phase, zero_charge_phase)
                      No V2 additions (vortex_minimal, harmonic, vortex_remove_pair 등 제외)
Sham/controls:        Predecessor-matched controls
Null families:        Predecessor-matched null families (smooth, magnitude, global_phase, zero_charge_phase — 4 families)
Primary metric:       Δ_mech_pred: predecessor-matched estimand (V1-style, 4 null families, no IUT).
                      V2 vs predecessor comparison: equivalence test on Δ_mech.
Validity metrics:     Replication fidelity: pipeline match rate (what fraction of pairs have identical charge extraction, decomposition results between V1 and V2 code), accuracy match (V2 trained model accuracy vs reported predecessor accuracy)
Statistical test:     Equivalence test (TOST): H0: |Δ_V2 − Δ_pred| > ε_equiv. Two one-sided tests at α=0.05.
                      ε_equiv: pre-specified equivalence margin (e.g., 0.2 × Δ_pred observed in calibration).
Sample-size rule:     N_confirm (F01과 동일 seeds). Predecessor replication은 같은 모델에서 pipeline만 변경.
Pass:                 TOST rejects (V2 ≈ predecessor within equivalence margin). V2 pipeline이 predecessor를 faithfully 재현.
Fail:                 TOST does not reject — V2 pipeline이 predecessor와 systematic하게 다른 결과. 원인 진단 필요 (code audit).
Inconclusive:         TOST inconclusive (underpowered). 또는 predecessor reference 값이 single-point estimate라 variance를 알 수 없음.
Required code:        topological/v2/evaluation.py: predecessor_matched pipeline (V2 additions OFF mode)
                      V1 vs V2 code comparison: charge extraction, decomposition, transplant output hash comparison
Required tests:       test_replication_v2.py: test_v1_v2_charge_identical, test_v1_v2_decomposition_identical, test_v1_v2_transplant_identical
Raw artifact schema:  Per-model: {Δ_mech_pred, Δ_mech_pred_CI, per_family_margins (4 families),
                      pipeline_match_rate, v1_v2_discrepancy_log: [{pair_id, discrepancy_type, v1_value, v2_value}]}
                      Cross-version: {Δ_diff, Δ_diff_CI, tost_pvalue, equivalence_margin, tost_result}
Compute formula:      F01 evaluation + predecessor-matched evaluation (추가 비용 minimal — pipeline만 변경, model 재사용).
                      N × ~10 min (reduced arms).
Dependency:            F01 (confirmatory trained models). V1 predecessor benchmark (reference Δ_mech 값 필요 — V1 논문 또는 사전 실행).
Claim enabled:        "V2 pipeline이 predecessor와 consistent하며, V2 additions가 결과를 artifact로 왜곡하지 않는다" — pipeline validity.
Claim prohibited:     "V2가 predecessor와 identical하다" — equivalence within margin만 주장.
```

---

### F05: Charge-Free Input Experiment (전하-영 입력 실험)

```
Experiment ID:       F05
Scientific question: Token embedding이 charge-free (모든 plaquette Q_p=0, everywhere positive magnitude, smooth)로 설계된 경우, trained model이 vortex structure를 학습을 통해 출현시키는가?
Formal hypothesis:
  H0: Charge-free input model이 trained 후에도 analyzable vortex topology를 가지지 않는다 (defect_density ≈ 0, 또는 trainability 실패).
  H1: Charge-free input model이 trained 후 analyzable vortex topology를 출현시킨다 — vortex가 initialization artifact가 아니라 학습으로 생성된다.
Alternative explanations:
  - Charge-free embedding이 model capacity를 제한하여 task accuracy가 저하될 수 있다 → accuracy가 낮으면 causal analysis 불가.
  - C=1 (C05-C06)에서 이미 charge-free embedding의 효과를 일부 검증 — F05는 C=8에서의 replication.
  - Charge-free embedding이 완벽하지 않으면 (일부 plaquette에서 near-zero magnitude), 학습 전에도 미세한 charge가 존재 → 완전한 "charge-free at init" 조건 위반.
Model:               U1ConvRNN (C=8) — charge-free smooth input embedding variant
                      대조: Standard embedding U1ConvRNN (F01 control)
Task:                 Copy task, confirmatory split
Input parameterization: Charge-free smooth positive-magnitude embedding:
                        - Every plaquette Q_p = 0 (NOT merely ΣQ_p = 0)
                        - min(|z(x)|) > 0.1 (strictly positive magnitude)
                        - Smooth spatial profile (low Fourier bandwidth)
                        대조군: Standard random full-spatial embedding (F01)
Split:                Confirmatory (frozen)
Unit of analysis:     Seed
Recipient selection:  F01 frozen pipeline
Donor selection:      Frozen geometric selection
Intervention arms:    Full F01 arm set
Sham/controls:        F01 sham controls
Null families:        Full 10-family set
Primary metric:       Δ_mech (IUT primary estimand) for charge-free model. Charge-free vs standard comparison: Δ_mech_charge_free − Δ_mech_standard.
                      Topology emergence: defect_density_trained − defect_density_untrained (charge-free model, within-model).
Validity metrics:     Training convergence (accuracy ≥ 0.90), charge-free verification at init (모든 plaquette Q_p=0, 모든 lattice point |z|>0.1), charge-free degradation (post-training에도 charge-free가 유지되는 비율 — 학습이 charge를 생성하면 감소)
Statistical test:     Charge-free IUT (α=0.05, secondary hypothesis). Charge-free vs standard comparison: two-sided paired bootstrap on Δ_mech difference.
                      Untrained vs trained topology comparison (within charge-free model): two-sided on defect_density.
Sample-size rule:     N_confirm/2 (charge-free model만 추가 training). Power: F01과 동일한 Δ_mech 가정 시 half seeds로도 IUT 검출 가능.
                      최소 10 seeds (confirmatory power analysis로 확정).
Pass:                 Charge-free model IUT 통과 (vortex effect 존재) AND charge-free > standard (Δ_mech_charge_free ≥ Δ_mech_standard) — vortex가 학습 emergent.
                      또는: charge-free model도 IUT 통과, standard와 equivalent → vortex가 initialization-independent하게 존재.
Fail:                 Charge-free model IUT 실패 (vortex effect 부재) — vortex effect가 전적으로 initialization artifact. Primary claim에致命.
Inconclusive:         Charge-free model training 실패 (accuracy < 0.90) → task-architecture mismatch로 인한 confound.
Required code:        topological/v2/model.py: charge-free embedding constructor (C05에서 검증된 구현)
Required tests:       test_model_v2.py: test_charge_free_embedding_all_plaquettes_zero, test_charge_free_embedding_positive_magnitude
                      test_training_v2.py: test_charge_free_model_convergence
Raw artifact schema:  Charge-free model: F01 schema + charge_free_verification {init_prevalence, init_defect_density, post_train_prevalence, post_train_defect_density}.
                      Comparison: {charge_free_vs_standard_difference, charge_free_vs_standard_CI, emergence_effect_size}
Compute formula:      Training: N_charge_free × 30k × 28 × 64 / throughput (N_charge_free ≈ N/2).
                      Evaluation: F01 evaluation × (N_charge_free + N_standard).
                      Additional: ~0.5 × F01 compute.
Dependency:            F01 (confirmatory standard model). C05-C06 (C=1 charge-free embedding 선행 검증).
Claim enabled:        "Vortex causal effect가 initialization artifact가 아니라 학습을 통해 emergence한다" — training-vs-untrained claim의 가장 강한 evidence.
Claim prohibited:     "Charge-free embedding이 모든 initialization artifact를 제거한다" — 학습 중 수치적 instability로 미세한 charge가 생성될 수 있음.
```

---

## Experiment Dependency Graph

```
C01 (Random Null) ─────────────────────┐
C02 (Untrained Topology) ──────────────┤
C03 (Token Embedding) ─────────────────┤
C04 (Branch Stability) ────────────────┤
                                        ├──► C05 (C=1 Trainability)
                                        │       │
                                        │       ▼
                                        ├──► C06 (C=1 Topology)
                                        │
C07 (Representative) ◄─────────────────┤
C08 (Minimal Surgery) ◄────────────────┤
C09 (Local Necessity) ◄────────────────┤
C10 (Donor Specificity) ◄──────────────┤
                                        │
C11 (Manifold Projection) ◄────────────┤
C12 (Natural Neighbor) ◄───────────────┤──► F01 (Confirmatory)
C13 (Harmonic Sector) ◄────────────────┤       │
C14 (Multichannel Phase Lock) ◄────────┤       ├──► F02 (Cross-Model)
C15 (Channel Basis Robustness) ◄───────┤       ├──► F03 (OOD Generalization)
C16 (Factorial Baseline) ◄─────────────┘       ├──► F04 (Predecessor Replication)
                                                └──► F05 (Charge-Free Input)
```

## Execution Order by Compute Cost

| Order | Experiment | Compute | Blocks |
|:-----:|-----------|:-------:|:------:|
| 1 | C01 | <1 CPU-min | None |
| 2 | C03 | <1 CPU-min | None |
| 3 | C02 | ~4 CPU-min | None (V1 진단 완료) |
| 4 | C04 | ~5 CPU-min | C02 |
| 5 | C11 | ~3 CPU-min | C05 (trained states) |
| 6 | C05 | ~24 GPU-min | C01–C04 |
| 7 | C06 | ~10 CPU-sec | C05 |
| 8 | C14 | ~1 CPU-min | C05 |
| 9 | C08 | ~38 GPU-min | C07, C11 |
| 10 | C07 | ~1.4 GPU-hr | C05 |
| 11 | C10 | ~1.0 GPU-hr | C07 |
| 12 | C09 | ~0.4 GPU-hr | C08 |
| 13 | C13 | ~0.4 GPU-hr | C12 |
| 14 | C12 | ~0.3 GPU-hr | C11 |
| 15 | C15 | ~2.9 GPU-hr | C11, C14 |
| 16 | C16 | ~33 GPU-hr | C05–C15 |
| 17 | F01 | ~39–78 GPU-hr | ALL C |
| 18 | F02 | ~78–157 GPU-hr | F01 |
| 19 | F03 | ~55–110 GPU-hr | F01 |
| 20 | F04 | ~5–10 GPU-hr | F01 |
| 21 | F05 | ~20–40 GPU-hr | F01 |

**Calibration 총계**: ~40 GPU-hr + ~15 CPU-min
**Confirmatory 총계** (N=50): ~200 GPU-hr
**Grand total** (N=50): ~240 GPU-hr (168-hr confirmatory wall time 내 실행 가능, calibration은 별도 48-hr window)
