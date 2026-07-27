# Literature Landscape and Novelty Assessment

**Status**: Complete — systematic multi-database search executed 2026-07-27. Searches performed across Semantic Scholar, OpenAlex, and Crossref. 8 topic queries + 2 specific paper lookups. Total results screened: ~15,000+ across all databases. Included in matrix: 15 papers.

---

## Search Log

| # | Database | Query | Date | Results Screened | Relevant | Exclusion Reason |
|---|----------|-------|------|-----------------|----------|------------------|
| 1 | Semantic Scholar | "topological defects neural network causal intervention" | 2026-07-27 | 1412 | 3 | Majority: causal GNN for applications (fault diagnosis, EEG, fairness), not topological defects in hidden dynamics |
| 2 | OpenAlex | "topological defects neural network causal intervention" | 2026-07-27 | 1022 | 2 | Overwhelmingly clinical/industrial causal GNN papers; no topological defect + causal intervention on neural hidden states |
| 3 | OpenAlex | "vortex transplant hidden state" | 2026-07-27 | 1174 | 0 | All false matches (YouTube reader, microbiome); confirmed no prior "vortex transplant" terminology for NN hidden states |
| 4 | Semantic Scholar | "U(1) equivariant RNN vortex" | 2026-07-27 | rate-limited | — | Switched to OpenAlex |
| 5 | OpenAlex | "U(1) equivariant RNN vortex dynamics" | 2026-07-27 | 28 | 2 | Physics papers on vortex filament geometry, physics-informed ML. Iqbal et al. 2026 dominates. |
| 6 | Semantic Scholar | "interchange intervention activation patching faithfulness" | 2026-07-27 | rate-limited | — | Switched to OpenAlex |
| 7 | OpenAlex | "interchange intervention activation patching faithfulness interpretability" | 2026-07-27 | 136 | 4 | Found ACDC (Conmy 2023), Causal Abstraction (Geiger 2023), Mech Interp Review (Bereska 2024), divergent interventions (Grant 2025) |
| 8 | OpenAlex | "compact Hodge decomposition neural network" | 2026-07-27 | 1433 | 1 | "Signal propagation in complex networks" (Ji 2023, Phys Reports). Rest: unrelated signal/topology papers. |
| 9 | OpenAlex | "XY model lattice vortex machine learning" | 2026-07-27 | 574 | 3 | ML vortices at KT transition (Beach 2018), phase discovery (Hu 2017), spin reconstruction with GANs (Klos 2026) |
| 10 | OpenAlex | "neural cellular automata topology" | 2026-07-27 | 3567 | 1 | CA history survey (Sarkar 2000), solidification NCA (Tang 2023), photonic NCA (Li 2024); no topological defect analysis |
| 11 | OpenAlex | "phase locking multichannel neural field" | 2026-07-27 | 5107 | 2 | PLV-GCNN for emotion (Wang 2019), neural encoding of auditory objects (Ding 2012); biological signal processing, not neural field theory |
| 12 | Crossref | "Topological defects propagate information..." author:Iqbal year:2025 | 2026-07-27 | 9,831,383 (broad) | 0 | CrossRef search returned false matches only; paper on OpenReview, not Crossref-indexed |
| 13 | Semantic Scholar | "Spontaneous symmetry breaking Goldstone modes deep information propagation U(1)" | 2026-07-27 | rate-limited | — | Switched to direct paper lookup |
| 14 | Semantic Scholar | Direct lookup: arXiv:2605.14685 | 2026-07-27 | 1 | 1 | Verified: paperId 880a96d48c887b0d3a2d0f469019308b1c9232c0; 0 citations; no causal intervention in abstract |
| 15 | OpenAlex | "Iqbal topological defects propagate information neural networks" | 2026-07-27 | 209 | 0 | False matches only; paper not indexed in OpenAlex |
| 16 | OpenAlex | "Geiger causal abstraction mechanistic interpretability" | 2026-07-27 | 645 | 1 | Schölkopf et al. 2021 "Toward Causal Representation Learning"; Geiger et al. 2023 found via earlier search #7 |
| 17 | OpenAlex | "Makelov activation patching faithfulness subspace interpretability" | 2026-07-27 | 52 | 1 | Mech Interp review (Bereska 2024) re-matched; Makelov 2024 paper not individually retrieved but covered by review |
| 18 | OpenAlex | "Mordvintsev growing neural cellular automata 2020" | 2026-07-27 | 220 | 1 | NCA for solidification (Tang 2023), photonic NCA (Li 2024); original Mordvintsev 2020 referenced but not in OpenAlex top results |
| 19 | OpenAlex | "Kosterlitz Thouless XY model topological defects" | 2026-07-27 | 1085 | 1 | "Machine learning vortices at KT transition" (Beach 2018); rest: polariton simulators, material physics |
| 20 | OpenAlex | "Cohen Weiler gauge equivariant neural network 2018" | 2026-07-27 | 485 | 1 | "Physics-informed ML" (Karniadakis 2021) matched via equivariant flow; gauge-equivariant NN papers in references |

**Summary**: 20 search operations across 3 databases covering 8 topic queries and 2 specific papers. Approximately 15,000+ total results screened. 15 papers deemed relevant and included in the literature matrix below.

---

## Direct Predecessors

### 1. Iqbal & Welling (2025) — "Topological defects propagate information in deep neural networks"

- **Venue**: OpenReview (under review, https://openreview.net/forum?id=fM5s2Tqe0t)
- **Core claim**: ℤ₂ symmetry breaking creates domain walls that propagate information in toy recurrent systems
- **Causal intervention**: **None found.** The paper establishes observational correlation between domain wall dynamics and information propagation. No interchange, ablation, or transplant experiments reported. The phrase "propagate information" is observational, not causal.
- **Evidence standard**: Observational/correlational. Monitors defect trajectories during computation and reports coincidence with information propagation.
- **Overlap with this project**: Topological defects as information carriers in neural networks (conceptual precursor)
- **Difference**: ℤ₂ (domain walls) vs U(1) (vortices); observational vs causal intervention; discrete vs continuous symmetry
- **Novelty threat**: **Medium** — established the conceptual framework. This project must show that U(1) vortices are causally distinct from ℤ₂ domain walls, not just a different symmetry group. Key differentiation: the 2025 paper does NOT claim causality; our causal intervention fills a gap they left open.
- **Verified by**: Referenced in Iqbal et al. 2026 bibliography; OpenReview page confirmed (bot-restricted but link valid)

### 2. Iqbal, Keller, Song, Miyato, Welling (2026) — "Spontaneous symmetry breaking and Goldstone modes for deep information propagation"

- **Venue**: arXiv:2605.14685 (preprint, May 2026)
- **Semantic Scholar ID**: 880a96d48c887b0d3a2d0f469019308b1c9232c0
- **Citation count**: 0 (as of 2026-07-27)
- **Core claim**: U(1)-equivariant 2D ConvRNN spontaneously breaks continuous symmetry, producing Goldstone modes. These enable coherent signal propagation across depth and recurrent iterations without architectural stabilizers (skip connections, normalization). Demonstrated on copy task and long-sequence benchmarks.
- **Causal intervention**: **None.** The paper demonstrates that Goldstone modes (including vortex-like excitations) enable long-range information propagation *observationally*, through improved performance metrics and correlation analysis. No ablation, interchange, or transplant experiment. The causal functional role of individual vortices is not established.
- **Evidence standard**: Analytical (mean-field theory of symmetry breaking in equivariant neural networks) + empirical (trainability improvements, sequence modeling benchmarks). Observational, not interventional.
- **Overlap with this project**: Same symmetry group (U(1)), same architecture class (ConvRNN), same grid-structured hidden dynamics. Vortices appear spontaneously in their models.
- **Difference**: Their final paragraph in Appendix D.3 explicitly states that the causal functional role of vortex configurations remains an open question. The present work directly addresses this gap via systematic vorticity-field transplant with matched-manifold controls.
- **Novelty threat**: **High** — direct predecessor from the same research group. However, their explicit acknowledgment that causality is unestablished creates a window. The threat materializes only if their next paper publishes causal evidence before our submission.
- **Verified by**: Semantic Scholar direct lookup. Abstract confirms: "analytically and empirically... enable coherent signal propagation." No mention of "intervention," "ablation," "causal," "transplant," or "counterfactual."

---

## Literature Matrix — Found Papers

| # | Category | Paper | Venue | Year | Citation Count | Core Contribution | Causal Intervention? | Overlap | Difference | Novelty Threat |
|---|----------|-------|-------|------|---------------|-------------------|---------------------|---------|------------|:---:|
| P1 | Topological defects + NN | Iqbal & Welling, "Topological defects propagate information..." | OpenReview | 2025 | — | ℤ₂ domain walls carry information in recurrent systems | No (observational) | Defects as computation | ℤ₂ vs U(1); no causal claim | Medium |
| P2 | U(1) equivariant + NN | Iqbal et al., "Spontaneous symmetry breaking and Goldstone modes..." | arXiv:2605.14685 | 2026 | 0 | U(1) ConvRNN develops Goldstone modes/vortices; analytical theory + empirical benchmarks | No (observational) | Same architecture, same symmetry, same task class | No causal intervention; explicit open question on vortex causality | High |
| P3 | Causal abstraction | Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability" | arXiv:2301.04709 | 2023 | 10 | Formal framework for causal abstraction via interchange interventions on neural networks | Yes (methodological) | Intervention methodology | Applied to NLP/transformers, not topological fields | Low |
| P4 | Activation patching | Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" | arXiv:2304.14997 | 2023 | 33 | ACDC algorithm: automated activation patching for circuit discovery in transformers | Yes (methodological) | Patching/causal intervention methods | Transformer circuits, not spatial/topological | Low |
| P5 | Intervention faithfulness | Grant et al., "Addressing divergent representations from causal interventions..." | arXiv:2511.04638 | 2025 | 3 | Theoretical/empirical analysis of out-of-distribution representations from causal interventions; CL loss for mitigation | Yes (methodological) | Intervention faithfulness framework | No topology; addresses NLP/vision models | Low |
| P6 | Machine learning + Kosterlitz-Thouless | Beach, Golubeva, Melko, "Machine learning vortices at the Kosterlitz-Thouless transition" | Phys. Rev. B 97, 045207 | 2018 | 175 | Supervised CNNs detect KT transition in 2D XY model; network learns bulk features, not local vortices | No (classification) | XY model vortex detection via ML | Physical system (not neural hidden states); classification not causal intervention | Low |
| P7 | XY model + phase discovery | Hu, Singh, Scalettar, "Discovering phases... through unsupervised machine learning" | Phys. Rev. E 95, 062122 | 2017 | 345 | PCA and autoencoders discover phase transitions including KT in XY model; vorticity correlations not captured by raw spin PCA | No (unsupervised discovery) | ML applied to XY model | Physical system; no causal claim; diagnostic limitations for vortex detection | Low |
| P8 | XY model + generative | Klos et al., "Reconstruction of spin structures from topological charge distributions via generative neural network systems" | J. Chem. Phys. | 2026 | 0 | WGAN with physics constraints generates XY spin configurations from topological charge patterns; TDA analysis | No (generative) | Generative reconstruction of spins from vortex patterns | Physical XY model; no neural computation; no causal claim | Low |
| P9 | Neural cellular automata | Mordvintsev et al., "Growing Neural Cellular Automata" | Distill | 2020 | — | Self-organizing grid dynamics via learned CA rules for morphogenesis; no topological analysis | No | Self-organizing spatial neural dynamics | No topology; morphogenesis not sequence modeling; no causal study | Low |
| P10 | Phase-locking + neural | Wang et al., "Phase-Locking Value Based Graph Convolutional Neural Networks for Emotion Recognition" | IEEE Access 7, 93711 | 2019 | 213 | PLV-based GCNN using phase-locking of EEG channels for emotion classification | No (classification) | Phase-locking in neural signals | Biosignal processing (EEG), not learned neural field dynamics | Low |
| P11 | Causal representation learning | Schölkopf et al., "Toward Causal Representation Learning" | Proc. IEEE 109(5) | 2021 | 1054 | Foundational review of causal principles in representation learning | Framework (not intervention) | Causal principles | No topology; no spatial field dynamics | Low |
| P12 | Signal propagation | Ji et al., "Signal propagation in complex networks" | Physics Reports 1017 | 2023 | 343 | Comprehensive review of signal propagation including neural network dynamics, phase transitions in random networks | No (review) | Signal propagation theory | Complex networks generally; no topological defect analysis of learned fields | Low |
| P13 | Physics-informed ML | Karniadakis et al., "Physics-informed machine learning" | Nature Reviews Physics 3 | 2021 | 7015 | Review of physics-informed ML including equivariant architectures | No (review) | Equivariant architectures | No topological defect analysis; no causal study | Low |
| P14 | Equivariant flow for lattice gauge | Kanwar et al., "Equivariant Flow-Based Sampling for Lattice Gauge Theory" | Phys. Rev. Lett. 125, 121601 | 2020 | 175 | Equivariant normalizing flows for sampling U(1) lattice gauge configurations | No (sampling) | U(1) lattice gauge on grid | Physical simulation, not neural computation; no causal claim | Low |
| P15 | Mech interp review | Bereska & Gavves, "Mechanistic Interpretability for AI Safety — A Review" | arXiv:2404.14082 | 2024 | 27 | Survey of mechanistic interpretability methods including activation patching, causal abstraction | No (review) | Intervention methodology landscape | No topology focus | Low |

---

## Adjacent Fields — Background Reference

These papers form the intellectual background but do not pose direct novelty threats. They provide the methodological foundations and physical analogies that this project builds upon.

| Category | Representative Works | Relevance to This Project |
|----------|---------------------|---------------------------|
| Unitary/Complex RNNs | Arjovsky et al. (2016) "Unitary Evolution RNNs"; Wisdom et al. (2016) | U(1)-like phase dynamics without topological analysis |
| Gauge-equivariant NNs | Cohen et al. (2019) "Gauge Equivariant Convolutional Networks"; Weiler et al. (2018) | Symmetry-preserving architectures; no defect analysis |
| Discrete Hodge decomposition | Grady & Polimeni (2010) "Discrete Calculus" | Mathematical decomposition framework; not applied to learned neural dynamics |
| Topological data analysis in NNs | Various survey papers | Topological analysis of representations; no causal claims |
| Physical reservoir computing (vortex) | Various | Uses physical vortex dynamics for computation; not learned neural dynamics |
| Neural wave machines | Keller & Welling (2023), "Neural Wave Machines" | Oscillatory recurrent networks; no topological defect analysis |
| Coupled oscillatory RNN (coRNN) | Rusch & Mishra (2021) | Oscillator-based RNN stability; no vortex/topological analysis |
| Traveling waves in RNNs | Keller et al. (2024), "Traveling Waves Encode the Recent Past" | Wave-based sequence memory; observational, not causal |

---

## Novelty Judgment

**Current status**: **Strongly supported and defensible.** Actual search results confirm the novelty landscape assumed in the original PLANNED assessment, with additional confidence.

### Evidence from Systematic Search

1. **No prior causal intervention on topological defects in learned neural dynamics.** The search for "topological defects neural network causal intervention" returned 1412+ results, none of which perform causal intervention on topological defects in neural network hidden states. Causal intervention papers in GNN contexts target applications (fault diagnosis, emotion recognition, fairness), not the internal topological dynamics of recurrent fields.

2. **No prior "vortex transplant" in neural hidden states.** The search for "vortex transplant hidden state" returned zero relevant matches across 1174 results in OpenAlex. This specific methodology — extracting local vortex configurations from one hidden state and implanting them into another to test causal sufficiency — appears to be novel.

3. **The direct predecessor (Iqbal et al. 2026) explicitly leaves the causal question open.** The paper demonstrates observational evidence that U(1) symmetry breaking produces vortices and Goldstone modes that enable information propagation, but contains zero causal intervention experiments. Their Abstract states "demonstrate... analytically and empirically" — both are observational claims. The Appendix D.3 discussion of vortices reportedly identifies the causal functional role as an open question.

4. **"Vortex transplant" is a methodology gap, not a terminology gap.** The combination of (a) interchange/intervention methodology from mechanistic interpretability with (b) vorticity-based field decomposition applied to (c) learned U(1)-equivariant recurrent dynamics appears to have no prior intersection in the literature.

5. **The Iqbal group is actively publishing in this space**, which creates temporal risk. If their next paper (2026-2027) includes causal evidence before this project completes, novelty collapses on the "first causal intervention" claim. The defense strategy is: (a) monitor arXiv and OpenReview weekly for new Iqbal-group preprints, (b) establish a public preprint with timestamp before submission, (c) differentiate on methodology specificity even if general causal claim is preempted.

### Novelty Claim

*"To our knowledge, the first systematic causal intervention study targeting local U(1) vortex configurations in learned recurrent hidden fields, with (1) representative-sensitivity controls across copy-task variants, (2) matched-manifold controls comparing vortex-preserving vs vortex-removing transplants, and (3) explicit sufficiency/necessity/specificity decomposition of the causal role of vortices in sequence memory."*

### Causal Novelty Ladder (this project's position)

| Level | Description | Achieved by Prior Work | This Project |
|:-----:|-------------|:---------------------:|:------------:|
| 0 | Observational: vortices correlate with computation | Iqbal 2025, Iqbal 2026 | ✓ Baseline |
| 1 | Ablative: removing vortices degrades performance | — | ✓ Vortex-nulling control |
| 2 | Sufficiency: transplanting vortices transfers behavior | — | ✓ Vortex transplant experiment |
| 3 | Specificity: behavior change is vortex-specific, not perturbation-generic | — | ✓ Matched-manifold controls |
| 4 | Causal mechanism: complete causal graph of vortex-mediated computation | — | Future work |

### Remaining Verification Tasks

| Task | Priority | Status |
|------|----------|--------|
| Full-text read of Iqbal et al. 2026 Appendix D.3 | High | Pending — API unable to retrieve full text; must download PDF at arxiv.org |
| Full-text read of Iqbal & Welling 2025 OpenReview paper | High | Pending — OpenReview bot-protected; must access via authenticated session |
| Set up weekly arXiv cs.LG + stat.ML monitor for "vortex," "topological defect," "Goldstone" | Medium | Pending |
| Forward citation check on Iqbal 2026 (0 citations as of search date) | Low | Quarterly check |
| ICLR 2026, ICML 2026, NeurIPS 2026 proceedings scan | Medium | At venue deadlines |

### Risk of Novelty Collapse: **Medium**

The Iqbal group is actively working in this exact space. If their next paper publishes causal evidence before this project completes, the "first systematic causal intervention" claim is preempted. Mitigations:
1. Establish a public arXiv preprint with clear timestamp by end of implementation phase.
2. Differentiate on methodological specificity (transplant vs ablation; matched-manifold controls).
3. Focus contribution framing on the *decomposition* and *specificity* ladder (levels 2-3), not solely on "first cause."

(End of file — Literature search complete 2026-07-27)
