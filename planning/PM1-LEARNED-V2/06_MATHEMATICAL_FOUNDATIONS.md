# Mathematical Foundations

## 6.1 Equivariance Scope

### Model Architecture

The U1ConvRNN has three operations:

1. **Token embedding** \(E_x\): token index \(\rightarrow\) complex field on \(T^2\)
   - **NOT equivariant**: \(E_x\) is a learned lookup table. A global phase rotation \(e^{i\alpha}\) does not commute with the embedding operation because the embedding maps to a specific initial field, not a phase-relational quantity.

2. **Autonomous recurrence** \(F\): hidden state \(\rightarrow\) hidden state (no input)
   - The convolution is explicitly constructed as cross-coupled real/imaginary: \(F(h) = W_r * \Re(h) - W_i * \Im(h) + i(W_i * \Re(h) + W_r * \Im(h))\)
   - The nonlinearity is `radial_tanh`: \(\tanh(\|z\|) \cdot z/\|z\|\)
   - **Claim**: \(F(e^{i\alpha} h) = e^{i\alpha} F(h)\)
   - **Proof sketch**: Convolution is linear over \(\mathbb{C}\) with real kernels → commutes with scalar multiplication by \(e^{i\alpha}\). The `radial_tanh` only depends on \(\|z\|\) and preserves direction → commutes.

3. **Readout** \(R\): hidden state \(\rightarrow\) vocabulary logits
   - The readout uses real-valued operations (reshaping + linear layer)
   - \(R(e^{i\alpha} h) = R(h)\) **should hold** but needs verification
   - If the readout uses only real-valued features (like magnitude or phase differences), it's invariant. If it uses raw phase values, it's not.

### Equivariance Scope Summary

| Operation | U(1) Behavior | Verified? |
|-----------|:---:|:---------:|
| Token embedding | Input-specific, not equivariant | By construction |
| Blank recurrence | Equivariant | Claimed, smoke-tested |
| Token-input recurrence | Possibly equivariant | Not separately tested |
| Readout | Should be invariant | Not separately tested |
| Full forward pass | Model-dependent | Smoke-tested (single phase) |

**Key gap**: The autonomous recurrent core is claimed equivariant, but the blank vs input transitions are not separately verified. The current equivariance test mixes all operations into one allclose check.

---

## 6.2 Topological Object Classification

### C=1: Scalar Complex Field

\(\mathbb{C} \setminus \{0\} \simeq S^1 \times \mathbb{R}_+\)

\(\pi_1(\mathbb{C} \setminus \{0\}) = \mathbb{Z}\) — punctured plane has non-trivial fundamental group.

A vortex is a configuration \(z(x)\) on \(T^2\) with nonzero winding on a contractible loop. Since \(T^2\) is not simply connected, "contractible" matters — a winding around the entire torus is a *cycle holonomy*, not a *local vortex*.

### C>1: Multichannel Vector Field

\(\mathbb{C}^C \setminus \{0\} \simeq S^{2C-1} \times \mathbb{R}_+\)

\(\pi_1(S^{2C-1}) = 0\) for \(C > 1\) — the real 3-sphere and higher odd spheres are simply connected.

**Implication**: A channelwise vortex (nonzero plaquette charge in one channel) is **not** a topological defect of the full vector field. It can be continuously deformed to zero by moving through other channel dimensions.

**For a channelwise vortex to be a genuine topological defect, the neural dynamics must constrain the state to an effective phase-locked submanifold**:
\[
z(x) \approx e^{i\theta(x)} a(x)
\]
where the internal orientation \(a(x) \in S^{2C-1}/U(1)\) is sufficiently rigid (slowly varying or empirically fixed across examples).

### C=1 Gateway Necessity

C=1 is **not** logically necessary — a phase-locked multichannel field could support channelwise topological charges. However, C=1 is **diagnostically essential** because:
1. It removes the possibility of charge unwinding through channel mixing
2. It eliminates the channel-mode selection ambiguity
3. It isolates the question: "Does U(1) equivariance, in isolation, create causally relevant vortices?"

---

## 6.3 Five Levels of Topology Claim

| Level | Claim | Required Evidence |
|:-----:|-------|-------------------|
| L1 | **Numerical integrality**: plaquette sums are integer multiples of \(2\pi\) | Construction verification (code) |
| L2 | **Lattice winding**: specific plaquettes have nonzero integer charge | extract_charge() with tolerance |
| L3 | **Perturbation stability**: charge is robust to small phase perturbations | branch_margin distribution, charge-flip radius |
| L4 | **Dynamical persistence**: charge persists through recurrent dynamics | Signed Jaccard over time steps |
| L5 | **Topological protection**: charge requires finite energy to create/annihilate | Gradient energy barrier, not yet measured |

**Current code covers L1-L3. L4 is partially covered by `signed_jaccard` persistence. L5 is not addressed.**

---

## 6.4 Torus Decomposition — Hodge Theory on \(T^2\)

The discrete 1-cochain \(\Delta \theta\) (link variables) on a periodic grid decomposes as:
\[
\Delta\theta = d\phi + \delta A + h
\]
where:
- \(d\phi\): exact (gradient) component — curl-free, from a scalar potential \(\phi\)
- \(\delta A\): coexact component — divergence-free, encodes the vortex charges via \(\mathrm{curl}(\delta A) = 2\pi Q\)
- \(h\): harmonic component — both curl-free and divergence-free. On \(T^2\), \(\dim H^1 = 2\), corresponding to the two independent cycle holonomies \((w_x, w_y)\).

### Path Dependence in the Presence of Local Vortices

If a field has local vortices (nonzero \(Q\)), a cycle holonomy measured by integrating link variables along a cycle depends on *which representative of the homology class* is chosen. Going around a vortex contributes \(\pm 2\pi\) to the holonomy.

**Implication for the harmonic sector**: When local vortices exist, the "harmonic component" cannot be uniquely extracted without first removing the coexact (vortex) component. The current decomposition pipeline handles this correctly by removing the canonical vortex field before analyzing smooth/harmonic components.

---

## 6.5 Representative Ambiguity

Given a charge map \(Q\), the solution to \(\nabla^2 \psi = 2\pi Q\) on \(T^2\) is unique up to an additive constant (uniform phase shift). However, the *canonical vortex field* \(v_Q = e^{i\psi}\) is one specific representative. Any field of the form:
\[
v'_Q = v_Q \cdot g, \quad \text{where } Q(g) = 0 \text{ (charge-free)}
\]
has the same topology.

The representative ambiguity matters because:
1. The intervention transplants \(v_Q\) — a specific representative
2. A different representative \(v'_Q\) with the same charges but different smooth component would give different behavioral results
3. The zero-charge matched control must account for this: if \(v'_Q\) causes a different behavioral shift than \(v_Q\), is the charge or the representative responsible?

**Statistical model for representative variance**:
\[
Y(r, d, \rho) = \mu + \alpha_{\text{charge}}(Q_d - Q_r) + \beta_{\text{repr}}(\rho_d - \rho_r) + \varepsilon
\]
where \(Y\) is the behavioral outcome, \(\alpha\) is the charge effect, \(\beta\) is the representative effect, and \(\varepsilon\) is noise. The causal claim requires \(\alpha > 0\) and \(\beta\) small (representative-invariant).

---

## 6.6 Minimal Surgery Design

Given recipient field \(z_r\) and target charge map \(Q_{\text{target}}\), find:
\[
z^* = \arg\min_{z'} d(z', z_r)
\]
subject to constraints:
1. \(Q(z') = Q_{\text{target}}\) (exact charge manipulation)
2. \(M(z') \approx M(z_r)\) (magnitude preservation)
3. \(H(z') \approx H(z_r)\) (harmonic sector preservation)
4. \(E(z') \approx E(z_r)\) (gradient energy preservation)
5. \(d_{\mathcal{M}}(z')\) small (on-manifold)

**Proposed approach**: Use the decomposition \(z_r = m_r \cdot v_{Q_r} \cdot s_r\). Then:
\[
z^* = m_r \cdot v_{Q_{\text{target}}} \cdot s_r
\]
This replaces only the vortex component while preserving magnitude and smooth components. This is already implemented as `transplant_vortex` in `decomposition.py`.

**For single-defect removal**: Replace \(Q_r\) with \(Q_r'\) where one pair of \(\pm 1\) charges is removed. Then compute:
\[
z^* = m_r \cdot v_{Q_r'} \cdot s_r
\]

**Sham surgery**: Replace \(Q_r\) with its negative \(Q_r' = -Q_r\) (flip all charges). This preserves charge density and structure but inverts the sign pattern. If sign matters (directional information encoding), behavior should change.

**Relaxation constraint**: After surgery, run the autonomous recurrence \(F\) for a few steps to allow the field to relax to the natural manifold. Then measure behavior. This controls for off-manifold artifacts.
