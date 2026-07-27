# Experiment Registry

Each experiment has a frozen specification before execution. Calibration experiments inform V2 contract; confirmatory experiments test hypotheses.

## Calibration Phase (C)

### C01: Random Field Null Baseline
- **Question**: What is the null distribution of vortex prevalence, density, and branch margin?
- **Hypothesis**: Random phase fields saturate state-level prevalence but have low branch margins and unstable topology
- **Phase**: Calibration
- **Compute**: <1 CPU-minute
- **Metrics**: prevalence, defect_density, branch_margin distribution, charge-flip radius

### C02: Untrained Model Topology
- **Question**: What topology exists at initialization vs after brief recurrence?
- **Hypothesis**: Embeddings have non-trivial topology; one-step recurrence may not change it
- **Phase**: Calibration
- **Compute**: <1 CPU-minute per model type
- **Metrics**: same as C01, stratified by source (embedding, post_write, pre_go)

### C03: Token Embedding Topology Baseline
- **Question**: Do individual token embeddings carry vortex structure?
- **Hypothesis**: Yes, due to random initialization with near-zero components
- **Phase**: Calibration
- **Compute**: <1 CPU-minute
- **Metrics**: Per-token defect count, density, valid-channel fraction

### C04: Branch Stability Across Recurrence
- **Question**: How stable are branch margins through blank recurrence?
- **Hypothesis**: U(1) recurrence stabilizes margins; Plain recurrence does not (or does so differently)
- **Phase**: Calibration
- **Compute**: <5 CPU-minutes per model (100 recurrence steps × 8 examples)
- **Metrics**: branch_margin time series, signed Jaccard time series, defect count time series

### C05: C=1 Micro-Pilot (Trainability)
- **Question**: Can a C=1 U1ConvRNN learn the copy task?
- **Hypothesis**: Yes, with similar or slightly worse accuracy than C=8
- **Phase**: Calibration
- **Compute**: ~30 min GPU (30k updates × 3 seeds, reduced batch?)
- **Metrics**: validation accuracy, defect density, branch stability

### C06: C=1 Topology Emergence
- **Question**: What topology emerges in C=1 hidden field after training?
- **Hypothesis**: Sparse, stable defects emerge; distinct from C=8 dense plasma
- **Phase**: Calibration
- **Compute**: Use C05 models
- **Metrics**: defect count, spatial distribution, signed Jaccard across seeds

### C07: Representative Sensitivity
- **Question**: How much does transplant outcome vary with same-charge representative choice?
- **Hypothesis**: Variance is small relative to charge effect
- **Phase**: Calibration
- **Compute**: ~2 min per donor-recipient pair (10 representatives)
- **Metrics**: representative variance fraction, intra-class correlation of charge vs rep

### C08: Minimal Sufficiency
- **Question**: Does minimal-surgery vortex transplant (same charge, minimum displacement) produce similar effect as canonical?
- **Hypothesis**: Yes, within representative variance
- **Phase**: Calibration
- **Compute**: ~1 min per pair
- **Metrics**: Correlation between minimal and canonical transplant margins

### C09: Local Necessity
- **Question**: Does annihilating a single defect pair reduce donor-specific behavior?
- **Hypothesis**: Effect is localized (specific output positions affected)
- **Phase**: Calibration
- **Compute**: ~1 min per pair per target pair
- **Metrics**: Position-specific margin change, displacement of surgery

### C10: Donor Specificity
- **Question**: Does vortex transplant from a specific donor shift behavior toward THAT donor, or toward any non-recipient?
- **Hypothesis**: Toward the specific donor (signature of information encoding)
- **Phase**: Calibration
- **Compute**: ~2 min per recipient (test all 8 donors)
- **Metrics**: donor-vs-alternate contrast

### C11: Manifold Projection
- **Question**: Are vortex-transplanted states on the natural manifold of hidden states?
- **Phase**: Calibration
- **Compute**: Uses PCA from training data; projection cost ~1s per state
- **Methods to evaluate**: nearest natural neighbor distance, reconstruction error under PCA, recurrence relaxation, kNN density ratio

### C12: Natural Neighbor Control
- **Question**: How does a natural hidden state with similar topology compare to the synthetic transplant?
- **Phase**: Calibration
- **Compute**: Requires trained models with diverse hidden states
- **Metrics**: effect_difference = transplant_margin - neighbor_margin

### C13: Harmonic Sector Intervention
- **Question**: Does swapping the harmonic sector (global torus winding) produce comparable effects?
- **Phase**: Calibration
- **Compute**: ~1 min per pair
- **Metrics**: harmonic_margin vs vortex_margin

### C14: Multichannel Phase Locking
- **Question**: Are channel phases correlated in trained U1ConvRNN?
- **Hypothesis**: U1ConvRNN develops phase coherence across channels; Plain does not
- **Phase**: Calibration
- **Compute**: ~5 min per model (analyze trained hidden states)
- **Metrics**: inter-channel phase correlation, PCA on phase space, effective rank

### C15: Channel Basis Robustness
- **Question**: Do results change under channel permutation or random unitary mixing?
- **Phase**: Calibration
- **Compute**: ~5 min per model
- **Metrics**: intervention margin under channel shuffle, margin under random unitary rotation

### C16: Factorial Baseline Decomposition
- **Question**: What explains the U1-vs-Plain difference? Is it equivariance, complex convolution, radial_tanh, or combination?
- **Phase**: Calibration
- **Compute**: ~30 min GPU per model variant (3 seeds each × 4 variants: U1, Plain, ComplexNoEquiv, RealWithEquiv)
- **Metrics**: causal vortex margin per variant

---

## Confirmatory Phase (F)

### F01: Confirmatory Pilot
- **Question**: Does vortex transplantation causally control output in U1ConvRNN, with representative-invariant, manifold-valid, nuisance-superior effects?
- **Phase**: Confirmatory (frozen contract)
- **Seed count**: Determined by calibration power analysis
- **Primary test**: Hierarchical bootstrap on mechanism advantage
- **Gate clauses**: Replicated from V1 structure but with redesigned `defect_learned_not_innate` and `branch_stability` gates

### F02: Cross-Model Comparison
- **Question**: Does U1ConvRNN show stronger causal vortex effects than PlainConvRNN?
- **Phase**: Confirmatory
- **Primary test**: Paired hierarchical bootstrap on U1-advantage minus Plain-advantage

### F03: Delay OOD Generalization
- **Question**: Does causal vortex effect persist at delay=64 (OOD)?
- **Phase**: Confirmatory
- **Same as F01, evaluated on held-out delay**

### F04: Task Generalization (Reverse Copy)
- **Question**: Does causal vortex effect transfer to reverse copy task?
- **Phase**: Optional, depends on calibration results
- **Note**: Only if primary claim is established

### F05: Noise Robustness
- **Question**: Is the causal vortex effect robust to input/training noise?
- **Phase**: Optional, sensitivity analysis
