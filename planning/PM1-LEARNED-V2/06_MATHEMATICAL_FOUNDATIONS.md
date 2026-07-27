# 수학적 기반 (Mathematical Foundations)

## 6.1 동변성 범위 (Equivariance Scope)

### 모델 구성요소별 동변성 분석

U1ConvRNN은 세 연산으로 구성되며, 각각의 U(1) 동변성은 다음과 같이 구분된다:

1. **토큰 임베딩 (token embedding) \(E_x\)**: 토큰 인덱스 → \(T^2\) 상 복소장 (complex field)
   - \(E_x\)는 학습된 lookup table이며, 전역 위상회전 (global phase rotation) \(e^{i\alpha}\)와 교환되지 않는다.
   - **U(1) 동변성 없음**: \(E_x\)는 특정 초기장으로 매핑되는 절대적 연산이므로, \(E_x \neq e^{i\alpha} E_x\)이다.

2. **자율 순환 (autonomous recurrence) \(F_0\)**: 은닉상태 (hidden state) \(h\) → 은닉상태 (blank transition, \(E_x = 0\))
   - 합성곱 (convolution)은 실수 커널로 구성된 교차결합 (cross-coupled) 구조: \(W_r * \Re(h) - W_i * \Im(h) + i(W_i * \Re(h) + W_r * \Im(h))\)
   - 비선형성 (nonlinearity)은 `radial_tanh`: \(\tanh(\|z\|) \cdot z/\|z\|\)
   - **U(1) 동변성 성립**: \(F_0(e^{i\alpha} h) = e^{i\alpha} F_0(h)\). 합성곱은 \(\mathbb{C}\) 상에서 스칼라 곱셈 \(e^{i\alpha}\)와 교환되며, `radial_tanh`는 \(\|z\|\)에만 의존하고 방향을 보존하므로 교환된다.
   - **현재 검증 상태**: 단일 위상값 (single phase value), 소수 토큰으로 `torch.allclose` 통과 (F-NEW-I). 자율 순환 (blank transition)만 분리하여 격자 위상 (grid of phase values) 및 장시간 전개 (long recurrence chains)에 대한 검증은 미수행.

3. **비공백 토큰 전이 (nonblank token transition) \(F_x, x \neq \text{blank}\)**:
   - \(F_x(h) = F_0(h) + \Delta_x\)이며, 여기서 \(\Delta_x\)는 토큰 임베딩 \(E_x\)에 의존하는 비균질항 (inhomogeneous term)이다.
   - **U(1) 동변성 없음**: \(E_x \neq 0\)이므로 \(F_x(e^{i\alpha} h) \neq e^{i\alpha} F_x(h)\). 전역 위상회전이 토큰 입력과 교환되지 않는다.

4. **판독 (readout) \(R\)**: 은닉상태 → 어휘 로짓 (vocabulary logits)
   - 판독은 `Linear(Re(h) ⊕ Im(h)) + b` 형태의 일반적인 실수 선형 사상 (real linear map)이다.
   - **U(1) 불변성 없음**: \(R(e^{i\alpha} h) \neq R(h)\) — 실수부와 허수부가 혼합되어 위상회전에 따라 출력이 변한다. 판독은 입력 위상에 민감한 (phase-sensitive) 일반 사상이며, 동변적이거나 불변적이라고 주장할 수 없다.

### 동변성 범위 요약 (Equivariance Scope Summary)

| 연산 (Operation) | U(1) 거동 (Behavior) | 검증 상태 |
|:---|---|:---:|
| 토큰 임베딩 \(E_x\) | 비동변적 (not equivariant) | 구조적으로 자명 |
| 자율 순환 \(F_0\) (blank) | **동변적** (equivariant) | 단일 위상 smoke-test 통과, 격자/장시간 검증 미수행 |
| 비공백 전이 \(F_x\) | **비동변적** (not equivariant) | 구조적으로 자명 |
| 판독 \(R\) | 비불변적 (not invariant) | 구조적으로 자명 — 일반 선형 사상 |
| 전체 순방향 (full forward pass) | 혼합 (mixed) | 단일 위상 smoke-test 통과 |

**핵심 표현 (correct phrase)**: 모델은 **"U(1)-동변적 자율 순환 코어 (U(1)-equivariant autonomous recurrent core)와 대칭깨짐 입력 및 판독 인터페이스 (symmetry-breaking input and readout interfaces)"** 로 구성된다. 모델 전체가 U(1)-동변적이라고 기술해서는 안 되며, 자율 역학 (autonomous dynamics)만이 동변성을 갖는다.

---

## 6.2 위상적 객체 분류 (Topological Object Classification)

### C=1: 스칼라 복소장 (Scalar Complex Field)

\(\mathbb{C} \setminus \{0\} \simeq S^1 \times \mathbb{R}_+\)

\(\pi_1(\mathbb{C} \setminus \{0\}) = \mathbb{Z}\) — 천공 평면 (punctured plane)은 비자명 기본군 (non-trivial fundamental group)을 갖는다.

볼텍스 (vortex)는 \(T^2\) 상에서 축약가능 루프 (contractible loop)에 대해 비자명 권선 (nonzero winding)을 갖는 배치 \(z(x)\)이다. \(T^2\)가 단일연결 (simply connected)이 아니므로 "축약가능"이 중요하다 — 토러스 전체를 감는 권선은 **사이클 홀로노미 (cycle holonomy)** 이지 **국소 볼텍스 (local vortex)** 가 아니다.

### C>1: 다채널 벡터장 (Multichannel Vector Field)

\(\mathbb{C}^C \setminus \{0\} \simeq S^{2C-1} \times \mathbb{R}_+\)

\(\pi_1(S^{2C-1}) = 0\) for \(C > 1\) — 3차원 이상의 홀수 구면 (odd sphere)은 단일연결이다.

**의미 (implication)**: 단일 채널의 플라켓 전하 (channelwise plaquette charge)는 전체 벡터장의 위상적 결함 (topological defect)이 **아니다**. 다른 채널 차원을 통해 연속적으로 풀어낼 수 있다 (continuously unwound).

**그러나 이것이 진공 다양체 (vacuum manifold)가 전체 주변공간 (ambient space) \(\mathbb{C}^C \setminus \{0\}\)임을 의미하지는 않는다.** 신경망 동역학 (neural dynamics)은 상태를 위상잠금 부분다양체 (phase-locked submanifold)로 제약할 수 있다:

\[
z(x) \approx e^{i\theta(x)} a(x)
\]

여기서 내부 배향 (internal orientation) \(a(x) \in S^{2C-1}/U(1)\)이 충분히 경직되어 (rigid) 있다면, 채널간 위상 상관관계 (inter-channel phase correlation)가 높고 배향 분산 (orientation variance)이 낮으며 유효 차원 (effective dimension)이 1에 가까워진다.

**이것은 귀납적 검증이 필요한 경험적 가설 (empirical hypothesis)이다.** 현재 파이프라인은 다음을 측정하지 않는다:
- 채널간 위상 상관 행렬 (inter-channel phase correlation matrix)
- 채널별 배향 분산 (per-example orientation variance)
- 국소 차원 추정 (local effective dimension via PCA on \(S^{2C-1}\) patches)

이러한 측정 없이는 C>1에서 관찰된 채널별 플라켓 전하가 진정한 위상적 객체인지, 아니면 섭동에 의해 소멸될 수 있는 인공물 (artifact)인지 판단할 수 없다.

### C=1 진단적 관문 (Diagnostic Gateway) 및 주의사항

C=1은 **논리적으로 필수적이지 않다** — 위상잠금된 다채널장은 채널별 위상 전하를 지지할 수 있다. 그러나 C=1은 **진단적으로 필수적 (diagnostically essential)** 이다:
1. 채널 혼합을 통한 전하 해소 (charge unwinding through channel mixing) 가능성을 제거
2. 채널-모드 선택 모호성 (channel-mode selection ambiguity) 제거
3. 질문을 격리: "U(1) 동변성이 단독으로 인과적으로 유의미한 볼텍스를 생성하는가?"

**C=1 주의사항 (caution)**: 현재 랜덤 전역-공간 임베딩 (random full-spatial embedding)을 사용하는 C=1 모델은 **초기화 시점부터 밀집 볼텍스 플라즈마 (dense vortex plasma)** 상태일 수 있다 (F-NEW-F: ~340 쌍/채널, ~10.6 쌍/플라켓). C=1만으로는 입력-초기화 교란 (input-initialization confound)이 제거되지 않는다.

C=1 분석에 필요한 추가 조건:
1. **전하-영 평활 양의 크기 임베딩 (charge-free smooth positive-magnitude embedding)**: 토큰 임베딩이 0의 플라켓 전하를 갖고, 모든 지점에서 양의 크기 (positive magnitude)를 유지하며, 공간적으로 평활해야 한다. 이 조건 없이는 학습 전에 이미 존재하는 볼텍스와 학습으로 생성된 볼텍스를 구분할 수 없다.
2. **크기 하한 검증 (magnitude lower-bound verification)**: `min(|field|) > 0` — 영크기 (zero magnitude)는 위상을 정의 불가능하게 만들어 모든 위상 분석을 불가능하게 한다 (F-NEW-E).

---

## 6.3 위상 주장의 다섯 단계 (Five Levels of Topology Claim)

| 단계 | 주장 (Claim) | 필요 증거 (Required Evidence) | 현재 파이프라인 상태 |
|:---:|------|------|:---:|
| L1 | **수치적 정수성 (numerical integrality)**: 플라켓 합은 \(2\pi\)의 정수배 | 코드 구성 검증 (`round(curl/2π)`) | **포함됨** (covered) |
| L2 | **격자 권선 (lattice winding)**: 특정 플라켓이 0 아닌 정수 전하를 가짐 | `extract_charge()` with tolerance | **포함됨** (covered) |
| L3 | **섭동 안정성 (perturbation stability)**: 전하가 작은 위상 섭동에 강건함 | `branch_margin` 분포, 전하반전 반경 (charge-flip radius), 섭동 강건성 측정 | **미포함** (NOT covered) |
| L4 | **동역학적 지속성 (dynamical persistence)**: 전하가 순환 동역학을 통해 지속됨 | 시간 단계별 부호 자카드 (signed Jaccard) 외에 전체 전하 수명 분포 (charge lifetime distribution), 전하 생성률 (creation rate), 전하 소멸률 (annihilation rate) | **부분적 대리지표** (partial proxy only) |
| L5 | **위상적 보호 (topological protection)**: 전하 생성/소멸에 유한 에너지 장벽 필요 | 기울기 에너지 장벽 측정 (gradient energy barrier), 전하반전 에너지 비용 (charge-flip energy cost) | **부재** (absent) |

**L3 상세 (what L3 requires, not currently measured)**:
- `branch_margin` 분포: 링크 위상 (link phase)이 \(\pm\pi\) 가지절단 (branch cut)까지의 여유 거리. 현재 진단은 최솟값만 보고하며 (F-NEW-D: 0.0000–0.0016 rad), 이는 수치적 불안정 영역에 진입했음을 나타낸다.
- 전하반전 반경 (charge-flip radius): 단일 플라켓 전하를 반전시키는 데 필요한 최소 전역 위상 섭동 크기 \(|\delta\theta|_{\min}\).
- 섭동 강건성 (perturbation robustness): 가우시안 위상 잡음 \(\delta\theta \sim \mathcal{N}(0,\sigma^2)\)를 가했을 때 전하 지도 \(Q\)가 변하지 않는 최대 \(\sigma\).

**L4 상세 (what L4 requires beyond signed Jaccard)**:
- 부호 자카드는 "현재 전하 배치가 원래 배치와 얼마나 겹치는가"만 측정한다 — 개별 전하의 수명, 생성/소멸 경로, 전하 쌍의 시공간 궤적을 포착하지 않는다.
- 전체 측정을 위해 필요한 지표: 전하 수명 히스토그램 (charge lifetime histogram), 전하 쌍 생성률 (pair creation rate per step), 전하 쌍 소멸률 (pair annihilation rate per step), 국소 전하 밀도 자기상관 (local charge density autocorrelation).

**L5**: L5를 측정하기 전까지 "위상적으로 보호된 (topologically protected)"이라는 표현을 원고에서 사용해서는 안 된다. 대신 "위상적으로 양자화된 (topologically quantized)"을 사용해야 한다 (Finding 5).

---

## 6.4 토러스 분해 — \(T^2\) 상의 이산 미분형식 (Hodge Theory on \(T^2\))

### 이론적 분해 (Theoretical Decomposition)

평활 연속체 (smooth continuum)에서 1-형식 (1-form) \(d\theta\)의 호지 분해 (Hodge decomposition)는 다음과 같다:
\[
d\theta = d\phi + \delta A + h
\]
여기서:
- \(d\phi\): 완전 성분 (exact component) — 회전-영 (curl-free), 스칼라 퍼텐셜 \(\phi\)로부터 유도됨
- \(\delta A\): 공완전 성분 (coexact component) — 발산-영 (divergence-free), \(\mathrm{curl}(\delta A) = 2\pi Q\)를 통해 볼텍스 전하를 부호화
- \(h\): 조화 성분 (harmonic component) — 회전-영 및 발산-영. \(T^2\)에서 \(\dim H^1 = 2\), 두 독립 사이클 홀로노미 \((w_x, w_y)\)에 대응

### 이산 격자에서의 경고 (Discrete Lattice Caveat)

**격자 위의 콤팩트 U(1) 링크 분해 (compact U(1) link decomposition on the lattice)는 단순한 \(a = d\phi + \delta A + h\) 이상을 필요로 한다.** 콤팩트 변수는 주치 (principal value) \((-\pi, \pi]\)에 제약되므로, 링크 변수 (link variable) \(\Delta\theta\)는 정수 여차원 (integer cochain) \(n_{ij} \in \mathbb{Z}\)을 수반한다:
\[
(\Delta\theta)_{ij} = (\theta_j - \theta_i)_{\text{mod }2\pi} + 2\pi n_{ij}
\]
여기서 \(n_{ij}\)는 가지절단 (branch cut) 정보를 부호화한다. \(n_{ij}\) 없이는 링크 변수를 단순히 \(d\phi + \delta A + h\)로 분해할 수 없다 — 가지절단은 비국소적 (nonlocal)이며 국소 연산자 (local operator)만으로 복원할 수 없다.

### 현재 코드가 수행하는 것 (What the Current Code Does)

현재 `decompose()` (decomposition.py:43-64)는 다음을 수행한다:
1. 단위장 (unit field) \(z/|z|\)으로 정규화 (compactification)
2. `extract_charge()`로 정수 전하 \(Q\) 추출
3. 포아송 방정식 (Poisson equation) \(\nabla^2 \psi = 2\pi Q\)를 풀어 표준 볼텍스장 (canonical vortex field) \(v_Q = e^{i\psi}\) 구성
4. 평활 성분 (smooth component)을 \(s = (z/|z|) \cdot \overline{v_Q}\)로 정의하고 재정규화

**현재 코드가 수행하지 않는 것 (What the Current Code Does NOT Do)**:
- **조화 섹터 (harmonic sector)를 분리하지 않는다.** \(s\)는 국소 전하가 0인 몫 (zero-local-charge quotient)이며, 이는 평활 완전 모드 (smooth exact modes)와 조화 모드 (harmonic modes) **모두**를 포함한다. \(s\) 내에서 완전 성분과 조화 성분을 구분하려면 전역 사이클 홀로노미 \((w_x, w_y)\)를 추출하고 이에 대응하는 선형 위상 기울기 (linear phase ramp)를 제거하는 추가 단계가 필요하다.
- **가지절단 추적 (branch cut tracking)을 수행하지 않는다.** `extract_charge()`는 `round(curl/2π)`를 사용하여 정수 여차원을 근사하지만, 이는 개별 링크 변수의 가지절단이 아닌 플라켓 수준의 근사이다. `branch_margin`이 0에 가까울 때 (F-NEW-D), 이 근사는 신뢰할 수 없다.

**정정된 기술 (corrected description)**: "표준 볼텍스 제거는 국소 전하 영 몫 (zero-local-charge quotient)을 생성하며, 이 몫은 평활 완전 모드와 조화 모드 모두를 포함할 수 있다. 완전 성분과 조화 성분의 분리는 현재 구현에 포함되어 있지 않다."

### 국소 볼텍스 존재 시 경로 의존성 (Path Dependence with Local Vortices)

장에 국소 볼텍스 (nonzero \(Q\))가 있을 때, 특정 호몰로지류 (homology class)의 대표 사이클을 따라 측정된 사이클 홀로노미는 경로 선택에 의존한다 — 볼텍스 주위를 통과할 때 \(\pm 2\pi\)의 기여가 발생한다.

**수정된 기술**: 표준 볼텍스장 \(v_Q\)는 \(Q\)와 동일한 국소 전하를 가지며, 전역 홀로노미 \((w_x, w_y) = (0, 0)\)에 해당하는 **특정 조화 섹터 대표 (one specific harmonic-sector representative)** 이다. 표준 볼텍스장 제거 후 남은 평활 몫 \(s\)는 원본 장의 조화 섹터와 완전 성분을 보존한다. 따라서 \(s\)에서 조화 성분을 추출하는 것은 원칙적으로 가능하지만, 현재 코드는 이를 수행하지 않는다.

---

## 6.5 대표 모호성 (Representative Ambiguity)

### 문제 설정 (Problem Setup)

전하 지도 \(Q\)가 주어지면, \(T^2\) 상에서 \(\nabla^2 \psi = 2\pi Q\)의 해는 가산 상수 (additive constant, 전역 위상 이동)까지 유일하다. 표준 볼텍스장 \(v_Q = e^{i\psi}\)는 **하나의 특정 대표 (one specific representative)** 이다. 동일한 위상 (topology)을 갖는 모든 장은 다음과 같은 형태를 갖는다:
\[
v'_Q = v_Q \cdot g, \quad \text{where } Q(g) = 0 \text{ (국소 전하 영, zero local charge)}
\]

### 용어 구분 (Terminology Distinction)

모든 전하-영 인자 \(g\)를 "게이지 선택 (gauge choice)"이라고 부르는 것은 오해의 소지가 있다 — 모델은 임의의 국소 게이지 (arbitrary local gauge)가 아닌 **전역 U(1) 대칭 (global U(1) symmetry)** 만을 갖는다. 네 가지 구분이 필요하다:

| 분류 | 정의 | 예시 | 인과 개입에 미치는 영향 |
|------|------|------|------|
| **전역 위상 동치 (global phase equivalence)** | \(g = e^{i\alpha}\), 상수 | \(z\)와 \(e^{i\alpha}z\)는 동일한 전하, 동일한 조화 섹터, 동일한 평활 질감을 가짐 | 판독 \(R\)이 위상 민감이므로 행동 변화 가능; `align_global_phase`로 정규화 필요 |
| **조화 섹터 차이 (harmonic-sector difference)** | \(g = e^{i(k_x x + k_y y)}\), 선형 위상 기울기 | 사이클 홀로노미 \((w_x, w_y)\)가 다른 두 장 | 동일 전하, 다른 전역 위상 구조; 인과 효과가 조화 섹터인지 국소 전하인지 구분 필요 |
| **평활 질감 차이 (smooth-texture difference)** | \(g = e^{i\phi(x)}\), \(Q(g) = 0\)이고 \(\phi\)가 공간적으로 평활하며 비선형 | 동일 전하, 동일 조화 섹터, 다른 국소 위상 변조 | 대표 분산 (representative variance)의 원천; 전하 효과와 대표 효과 분리 필요 |
| **진정한 신경 상태 차이 (genuinely different neural state)** | \(g\)가 \(Q(g) = 0\)이면서도 신경망의 자연 다양체 상의 서로 다른 상태 | 위의 세 경우 중 어느 것이든 신경 상태 차이로 이어질 수 있음 | 개입의 타당성 (validity) 문제 — 대표 교체가 자연 상태를 벗어나는가? |

**권장 용어 (recommended terminology)**:
- 전하-영 인자 → **"동일-전하 위상 대표 (same-charge phase representative)"** 또는 **"국소-전하-영 인자 (zero-local-charge factor)"**
- "게이지 선택"이라는 용어는 사용하지 않는다. 전역 위상 동치, 조화 섹터 차이, 평활 질감 차이를 별도로 명시한다.

### 대표 분산의 통계 모형 (Statistical Model for Representative Variance)

\[
Y(r, d, \rho) = \mu + \alpha_{\text{charge}}(Q_d - Q_r) + \beta_{\text{repr}}(\rho_d - \rho_r) + \gamma_{\text{harmonic}}(w_d - w_r) + \varepsilon
\]

여기서 \(Y\)는 행동 결과 (behavioral outcome), \(\alpha\)는 전하 효과 (charge effect), \(\beta\)는 대표 효과 (representative effect), \(\gamma\)는 조화 섹터 효과 (harmonic-sector effect)이다.

인과적 주장 (causal claim)은 \(\alpha > 0\)이고 \(\beta\)와 \(\gamma\)가 작을 (대표-불변 및 조화-섹터-불변) 것을 요구한다. 조화 섹터를 제어하지 않고서는 국소 전하에 귀속된 효과가 실제로는 전역 홀로노미 차이에 기인한 것인지 배제할 수 없다.

---

## 6.6 최소 수술 설계 (Minimal Surgery Design)

### 현행 구현의 비최소성 (Non-Minimality of Current Implementation)

현행 `transplant_vortex` (decomposition.py:67-68)는 **최소 수술 (minimal surgery)이 아니다**:

```python
def transplant_vortex(recipient, donor):
    return recompose(recipient.magnitude, donor.vortex, recipient.smooth)
```

이 연산은 \(z^* = m_r \cdot v_{Q_{\text{target}}} \cdot s_r\)를 수행하며, 이는 **성분 교체 (component replacement)** 이다:
- 변위 (displacement) \(\|z^* - z_r\|\)를 최소화하지 않는다
- 기울기 에너지 (gradient energy) \(E(z^*)\)를 보존하지 않는다
- 전체 푸리에 스펙트럼 (full Fourier spectrum)을 보존하지 않는다
- 조화 섹터 (harmonic sector)를 보존하지 않는다 (\(v_{Q_{\text{target}}}\)은 \((0,0)\) 조화 섹터를 가정)
- 자연 다양체 위에 있음을 보장하지 않는다 (no on-manifold guarantee: 표준 볼텍스장 \(v_{Q_{\text{target}}}\)은 순환 동역학에 의해 생성된 자연 상태가 아닐 수 있음)

### 최소 수술의 정의 (Definition of Proper Minimal Surgery)

수용자 장 (recipient field) \(z_r\)과 목표 전하 지도 (target charge map) \(Q_{\text{target}}\)이 주어졌을 때, 다음 최적화 문제를 푼다:

\[
z^* = \arg\min_{z'} \; d(z', z_r)
\]

**제약조건 (constraints)**:
1. \(Q(z') = Q_{\text{target}}\) (정확한 전하 조작, exact charge manipulation)
2. \(M(z') = M(z_r)\) (크기 보존, magnitude preservation, pointwise)
3. \(H(z') = H(z_r)\) (조화 섹터 보존, harmonic sector preservation, \((w_x, w_y)\) 보존)
4. \(E(z') \leq E(z_r) \cdot (1 + \epsilon)\) (기울기 에너지 상한, gradient energy bounded increase)
5. \(S(z', z_r) \geq 1 - \delta\) (스펙트럼 유사도 하한, spectral similarity lower bound, `normalized_spectrum_error` ≤ δ)
6. \(d_{\mathcal{M}}(z') \leq \eta\) (자연 다양체 거리 상한, natural manifold distance)

**목적함수 후보 (candidate objective \(d\))**:
- \(\ell_2\) 거리: \(\|z' - z_r\|_2\) — 단순하지만 위상 정보를 포착하지 못함
- 단위장 각도 거리 (unit-field angular distance): \(\|z'/|z'| - z_r/|z_r|\|_2\) — 위상 변화에 집중
- 기울기 거리 (gradient distance): \(\|\nabla z' - \nabla z_r\|_2\) — 국소 구조 보존

**최적화 변수 (optimization variables)**: \(z'\)은 \(T^2\) 상의 복소장이며, 실수부와 허수부로 분해하여 \(2 \cdot H \cdot W\) 차원의 비제약 연속 변수로 취급할 수 있다. 크기 제약 \(M(z') = M(z_r)\)은 변수를 \(z' = m_r \cdot u'\) (여기서 \(|u'| = 1\))로 매개화하여 엄격히 부과할 수 있다.

**알고리즘 개요 (algorithm outline)**:
1. 초기화: \(z'_0 = m_r \cdot v_{Q_{\text{target}}} \cdot v_{Q_r}^* \cdot (z_r/|z_r|) = m_r \cdot v_{Q_{\text{target}}} \cdot s_r\) (현행 성분 교체)
2. 사영 경사하강 (projected gradient descent): \(z'\)을 제약 영역으로 사영하며 \(d(z', z_r)\) 최소화
3. 수렴 판정: \(\|z'_{k+1} - z'_k\| < \tau\) 또는 \(k > K_{\max}\)
4. 사후 다양체 이완 (post-surgery manifold relaxation): \(z^*\)를 자율 순환 \(F_0\)으로 \(T_{\text{relax}}\) 단계 전개

### 단일 결함 제거를 위한 특수화 (Specialization for Single-Defect Removal)

\(Q_r\)에서 하나의 \(\pm 1\) 쌍을 제거하여 \(Q_r'\)을 얻은 후, \(Q_{\text{target}} = Q_r'\)에 대해 위 최소 수술을 수행한다. 현행 성분 교체는 초기화 \(z'_0\)를 제공할 뿐이며, 이 초기화로부터 최적화를 통해 최소 변위 해를 찾아야 한다.

### 사기 수술 (Sham Surgery) 정의

**사기 수술 (sham surgery)은 다음을 보존하면서 위상을 보존해야 한다** (topology-preserving while matching displacement, energy, spectrum, and procedure):
- 전하 지도: \(Q(z') = Q(z_r)\) (동일 전하)
- 변위 규모: \(\|z' - z_r\| \approx \|z_{\text{active}} - z_r\|\)
- 기울기 에너지 변화: \(|E(z') - E(z_r)| \approx |E(z_{\text{active}}) - E(z_r)|\)
- 스펙트럼 변화: \(S(z', z_r) \approx S(z_{\text{active}}, z_r)\)
- 시술 절차: 동일한 최적화 알고리즘, 동일한 반복 횟수, 동일한 다양체 이완 단계

사기 수술은 위상은 변하지 않았으나 동일한 규모의 섭동을 가했을 때 행동이 변하는지를 측정하여, 관찰된 효과가 위상 변화 자체 때문인지 아니면 섭동의 크기 때문인지를 구분한다.

### 부호 반전 (Sign Flip)의 분류

**부호 반전 \(Q \to -Q\)는 사기 수술이 아니라 능동적 특이성 개입 (active specificity intervention)이다.**

- 부호 반전은 전하 지도를 변경한다 (\(Q \neq -Q\) in general) — 위상 보존 실패.
- 부호 반전은 전하 밀도와 쌍 구조 (pair structure)는 보존하지만 모든 전하의 부호를 반전시키므로, 방향성 정보 부호화 (directional information encoding)를 검증한다.
- 특이성 (specificity) 질문에 답한다: 볼텍스의 방향 (부호)이 인과적 관련이 있는가, 아니면 전하 밀도 (charge density)만으로 충분한가?

부호 반전은 인과 분류 체계 (causal taxonomy)에서 **특이성 팔 (specificity arm)** 에 속하며, 사기 수술 팔 (sham arm)과 구분되어야 한다 (7.3절 참조).

---

## 6.7 요약 및 미해결 위험 (Summary and Outstanding Risks)

| 항목 | 현재 상태 | 위험 |
|------|------|------|
| 동변성 범위 (equivariance scope) | 자율 코어만 동변적임을 인식하였으나, 판독의 비불변성과 비공백 전이의 비동변성은 분리 검증 미수행 | "U(1)-동변적 모델"이라는 과장된 주장으로 오인될 위험 |
| 위상 단계 L1–L2 | 포함됨 (covered) | — |
| 위상 단계 L3 (섭동 안정성) | 미포함 (NOT covered) | 가지절단 근방에서 전하 추출이 수치적으로 불안정할 위험 (F-NEW-D) |
| 위상 단계 L4 (동역학적 지속성) | 부분적 대리지표만 존재 | 개별 전하가 아닌 전역 중첩만 측정; 전하 수명 미측정 |
| 위상 단계 L5 (위상적 보호) | 부재 (absent) | "위상적으로 보호된"이라는 용어 사용 금지 |
| 호지 분해 (Hodge decomposition) | 조화 섹터 미분리; 국소 전하 영 몫에 완전 성분과 조화 성분 혼재 | 조화 섹터 효과와 국소 전하 효과의 혼동 위험 |
| 대표 용어 (representative terminology) | "게이지 선택"이라는 부정확한 용어 사용 중 | 전역 U(1) 대칭을 국소 게이지와 혼동할 위험 |
| 최소 수술 (minimal surgery) | 성분 교체만 구현됨; 최소화 없음, 제약조건 없음 | 관찰된 효과가 전하 차이 때문인지, 불필요하게 큰 섭동 때문인지 구분 불가 |
| C>1 분석 | \(\pi_1 = 0\)은 인지되었으나, 귀납적 위상잠금 검증 미수행 | 채널별 전하를 진정한 위상적 객체로 오인할 위험 |
| C=1 주의사항 | 밀집 볼텍스 플라즈마 및 영크기 문제 인지됨 | 전하-영 평활 초기화 없이는 학습-생성 볼텍스와 초기화 인공물 구분 불가 |
| 부호 반전 분류 | 특이성 개입으로 올바르게 분류됨 (7.3절) | 사기 수술과 혼동 시 논리적 오류 |
