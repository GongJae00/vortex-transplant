# Literature Landscape and Novelty Assessment

**Status**: Partial — primary sources identified; forward/backward citation search and comprehensive matrix TBD.

---

## Direct Predecessors

### 1. Iqbal & Welling (2025) — "Topological defects propagate information in deep neural networks"

- **Venue**: OpenReview (under review)
- **Link**: https://openreview.net/forum?id=fM5s2Tqe0t
- **Core claim**: ℤ₂ symmetry breaking creates domain walls that propagate information in toy recurrent systems
- **Overlap**: Topological defects as information carriers in neural networks
- **Difference**: ℤ₂ (domain walls) vs U(1) (vortices); observational vs causal intervention
- **Novelty threat**: **Medium** — they established the conceptual framework for "defects as computation." The present work must show that U(1) vortices are causally distinct from ℤ₂ domain walls, not just a different symmetry group.
- **Unknown**: Do they include any causal intervention? What is the evidence standard?

### 2. Iqbal et al. (2026) — "Spontaneous symmetry breaking and Goldstone modes for deep information propagation"

- **Venue**: arXiv:2605.14685
- **Core claim**: U(1)-equivariant 2D ConvRNN develops long-lived vortex defects on a copy task, but their causal functional role is not established
- **Overlap**: Same symmetry group (U(1)), same task class (copy), same architecture (ConvRNN), same hidden grid
- **Difference**: The present work claims *causal* evidence via transplantation, not just observational correlation
- **Novelty threat**: **High** — this is the direct predecessor. If they have already published causal evidence before V2 completes, novelty collapses. Must verify their current manuscript: do they have any intervention experiments?
- **Required check**: Read Appendix D.3 of arXiv:2605.14685. Verify their explicit claims about vortex causality.

---

## Adjacent Fields — Literature Matrix

| Category | Representative Work | Overlap | Difference | Novelty Threat |
|----------|-------------------|---------|------------|:---:|
| **Causal abstraction / interchange intervention** | Geiger et al. (2021–2024), "Causal abstraction" | Intervention methodology | Applied to DNNs, not topological fields | Low |
| **Activation patching faithfulness** | Makelov et al. (2024), "Is This the Subspace You Are Looking For?" | Intervention on hidden states | Critiques patching methodology; no topology | Low |
| **Complex/unitary RNNs** | Arjovsky et al. (2016), "Unitary Evolution RNNs"; Wisdom et al. (2016) | U(1)-like dynamics | No topological analysis, no causal intervention | Low |
| **Neural cellular automata** | Mordvintsev et al. (2020) | Self-organizing grid dynamics | No topology analysis, no causal study | Low |
| **Physical reservoir computing (vortex)** | Various | Vortex as computational medium | Physical systems, not learned neural dynamics | Low |
| **Discrete Hodge decomposition** | Grady & Polimeni (2010), "Discrete Calculus" | Mathematical framework | Applied to graphs, not learned dynamics | Low |
| **Compact U(1) lattice gauge theory / XY model** | Kosterlitz & Thouless (1973) | Vortex topology physics | Statistical physics, not machine learning | Low |
| **TDA in neural networks** | Various | Topological analysis of representations | No causal claim, different structure | Low |
| **Gauge-equivariant neural networks** | Cohen et al. (2019), Weiler et al. (2018) | Symmetry in ML | No topological defect analysis | Low |
| **Mechanistic interpretability** | Olah et al., Elhage et al. | General methodology | No topology focus | Low |

---

## Search Log

| Database | Query | Date | Status |
|----------|-------|------|--------|
| Semantic Scholar | "topological defects neural network causal intervention" | TBD | Not executed |
| Semantic Scholar | "vortex transplant hidden state" | TBD | Not executed |
| OpenReview | "topological defect information propagation" | TBD | Not executed |
| arXiv | "U(1) equivariant RNN vortex causal" | TBD | Not executed |
| Crossref | "compact field decomposition neural network" | TBD | Not executed |

---

## Novelty Judgment

**Current status**: **Plausible, not yet "strongly supported" or "defensible."**

The direct predecessor (Iqbal et al. 2026) explicitly leaves open the question of whether vortices are causally functional. The present work directly addresses this gap. However:

1. The 2025 domain-wall paper by the same group already claims "topological defects = information carriers" — if this is accepted as causal evidence, the present work's novelty is in the *type of defect* (vortex vs domain wall) and the *symmetry group* (U(1) vs ℤ₂), not in the core "defects \(\rightarrow\) computation" claim.
2. The causal intervention methodology used here (interchange/transplant) is not novel — it's adapted from the causal abstraction literature.
3. The decomposition (Hodge-like vortex/smooth/magnitude) is a mathematical standard, not a novel contribution.

**Tentative novelty claim**: *"To our knowledge, the first systematic causal intervention study targeting local U(1) vortex configurations in learned recurrent hidden fields, with representative-sensitivity, matched-manifold controls, and explicit sufficiency/necessity/specificity decomposition."*

**Before freezing this claim, the following must be completed:**
1. Full-text review of Iqbal et al. 2026 Appendix D.3 (vortex section)
2. Forward citation search from both Iqbal papers
3. Search for "vortex transplant neural network" or similar phrasing
4. Check NeurIPS/ICML/ICLR 2025-2026 proceedings for related work
5. Verify that activation patching papers have not already applied transplant methods to spatial fields

**Risk of novelty collapse**: **Medium** — the Iqbal group is actively working in this exact space. If their next paper includes causal evidence, novelty is preempted. Speed matters but does not justify skipping controls.
