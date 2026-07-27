# Resource Schedule and Value of Information

**Status**: All throughput estimates are UNFROZEN. Hardware capacity limits are FROZEN_ENGINEERING (from V2 contract §resource_policy). This document provides the model structure and candidate scaling rules; actual numbers will be populated from smoke benchmarks.

---

## Throughput Model

**Not yet benchmarked.** The following decomposes the throughput into nine individually measurable components. Each entry is a template variable whose value is UNFROZEN until smoke benchmarking.

### 1. Recurrent cell-steps/sec (forward only)

```
T_cell_fwd = measured forward steps/second on target GPU × model_size_factor
```

A single forward `model.step()` call. This is the atomic unit for all recurrence-based metrics. Measured by profiling blank-step sequences at batch_size=1 and batch_size=64, with C=1 and C=8 model sizes.

### 2. Examples/sec (forward + backward)

```
T_examples = batch_size / (forward_time + backward_time)
```

Each training step (`one_training_step`) executes a full `run_copy` forward pass through all recurrent steps, then backpropagates through the unrolled computation graph. The number of recurrent cell-steps per forward example is:

```
n_recurrent_steps_per_example = L (write) + D (delay) + 1 (GO) + (L-1) (readout)
                               = D + 2L
```

For copy length L=4 with training delay D ∈ [16, 32] (approximate midpoint D≈24):

```
n_recurrent_steps_per_example = D + 8    (NOT D + 4)
```

Backward pass cost is approximately 2× the forward pass in FLOPs. Effective throughput must account for GPU utilization, host-device transfers, and gradient clipping overhead.

### 3. Optimizer updates/sec

```
T_update = T_examples  (one example = one update, batch-synchronized)
```

Each optimizer step processes one batch. The wall-clock time per update is `batch_size / T_examples`. With `updates = 30_000` and `batch_size = 64`, the training wall time per seed is:

```
t_train_seed = 30_000 × 64 / T_examples  =  30_000 × batch_size / T_examples
```

### 4. Evaluation continuations/sec

```
T_eval = measured continuations/second @ batch_size=64, D=24, L=4
```

An evaluation continuation (`continue_copy` starting from `post_write`) executes D delay steps + 1 GO step + (L-1) readout steps = D + L recurrent steps. For each test example, the number of continuations equals:

```
n_continuations = n_donors × n_arms + n_null_draws × n_null_families + 1 (recipient baseline)
```

### 5. Decomposition/sec

```
T_decomp = measured decompositions/second (single field, batch_size=1)
```

The `decompose()` call operates on a single hidden field (2C × H × W tensor) and extracts the multichannel phase field, branch decomposition, vortex charge map, and branch stability flag. This is a CPU-bound operation; profiling should measure on-host (NumPy) and on-device (GPU-accelerated) variants.

### 6. Surgery solver time

```
t_surgery = measured seconds per (donor, recipient) pair for canonical transplant
```

Covers `transplant_vortex_canonical`, `transplant_vortex_minimal`, `transplant_harmonic`, `annihilate_pair`, and `relax()`. These are per-pair operations; aggregate time scales linearly with the number of donor-recipient pairs. Profiling must distinguish:
- Canonical transplant (full charge map substitution)
- Minimal transplant (single-vortex displacement)
- Harmonic sector swap (global winding substitution)
- Pair annihilation + relaxation

### 7. Null-generation time

```
t_null = measured seconds per null family × n_null_draws
```

Each null family requires generating `n_null_draws = 199` matched bootstrap draws. Null-draw generation includes same-charge representative sampling, matched global-phase rotation, and matched zero-charge phase perturbation. Operations are embarrassingly parallel per draw.

### 8. Manifold-diagnostic time

```
t_manifold = measured seconds per (state, manifold_model) pair
```

Covers: PCA projection, kNN neighbor search, reconstruction error, and kNN density ratio. The PCA basis is precomputed once per seed from training-data hidden states; per-state projection is O(min(K, D)) where K is the number of PCA components and D is the flattened state dimension.

### 9. Artifact write time

```
t_artifact = measured seconds per seed × n_artifacts_per_seed
```

Per the V2 artifact schema (contract §artifact_schema), each seed writes: `training.json`, `model.pt`, `evaluation/{topology,pairs,outcomes,summary}.json`, `manifold/{pca_basis.npz,diagnostics.json}`. Per split: `decision.json`. Root: `manifest.sha256`. I/O is dominated by model checkpoint (~tens of MB) and PCA basis (~tens of MB); remaining JSON files are O(KB-MB).

### Aggregate Training Formula

```
t_train_seed = n_updates × n_recurrent_steps_per_example × batch_size / T_cell_fwd
             + n_updates × n_recurrent_steps_per_example × batch_size × backward_factor / T_cell_fwd
             = n_updates × batch_size × n_recurrent_steps_per_example × (1 + backward_factor) / T_cell_fwd
             ≈ n_updates × batch_size × (D + 2L) × 3 / T_cell_fwd
```

where `backward_factor ≈ 2` (empirically calibrated from smoke benchmarks).

### Aggregate Evaluation Formula

```
t_eval_seed = n_test_examples × (
      n_donors × n_arms × n_continuation_steps_per_eval / T_eval
    + n_null_families × n_null_draws × n_continuation_steps_per_eval / T_eval
    + 1 × n_continuation_steps_per_eval / T_eval           # recipient baseline
) + n_pairs × (t_surgery + t_manifold) + t_artifact
```

where:
- `n_test_examples` ≈ 100
- `n_donors` = 8
- `n_arms` = 15 (V1:13 + V2 additions: vortex_minimal, harmonic)
- `n_null_draws` = 199 (per family, matched bootstrap)
- `n_null_families` = 8
- `n_continuation_steps_per_eval` = D + L (not D + 2L; `continue_copy` starts from post_write)
- `n_pairs` = n_test_examples × n_donors × n_arms

### Aggregate Wall Time

```
t_total = Σ_n_model_types Σ_n_seeds (t_train_seed + t_eval_seed)
```

---

## Phase Estimates (TEMPLATE — Fill from Smoke Benchmarks)

All compute estimates in this table are UNFROZEN. Ranges are derived from the throughput model structure above with provisional T_* symbols. Actual ranges will be computed when T_cell_fwd, T_eval, T_decomp, t_surgery, t_null, t_manifold, and t_artifact are measured from smoke benchmarks.

| Phase | Runs | Updates/Seed | Eval Pairs | Dominant Cost | Estimated Range | Confidence |
|-------|-----:|------------:|----------:|:--------------|:---------------:|:----------:|
| Calibration C01-C04 (CPU) | 10 seeds × 4 diag | 0 | 100 | T_decomp | <1 min | High |
| Calibration C05-C06 (C=1) | 3 seeds × 2 models | 30,000 | 100 | T_examples | UNFROZEN | Unknown |
| Calibration C07-C15 (full) | 3 seeds × 2 models | 30,000 | 100 × 15 arms | t_train_seed + t_eval_seed | UNFROZEN | Unknown |
| Calibration C16 (factorial) | 3 seeds × 4 variants | 30,000 | 100 × 15 arms | T_examples (4× training) | UNFROZEN | Unknown |
| Confirmatory F01-F02 | N seeds × 2 models | 30,000 | 100 × 15 arms × 199 nulls | t_train_seed + t_eval_seed (with bootstrap) | UNFROZEN | Unknown |
| Confirmatory F03 (OOD D=64) | N seeds × 2 models | 30,000 | 100 × 15 arms × 199 nulls | Same as F01, D=64 ~2.7× eval steps | UNFROZEN | Unknown |

**Do not freeze these until smoke benchmarks are run.** The phase ordering (next section) is correct for VOI; the compute numbers backing each phase are pre-benchmark.

---

## Value of Information Ordering

Six-phase sequential gating, ordered from cheapest falsification to most expensive confirmation. Each phase must pass its candidate decision rules (next section) before proceeding. All compute estimates per phase remain UNFROZEN until smoke benchmarks.

### Phase 1: C01-C04 — CPU Diagnostics

- **What**: Random-field null baseline, untrained model topology, token embedding topology, branch stability across recurrence.
- **Falsifies**: "Defects are learned" (F-NEW-A), "defects are U1-specific" (F-NEW-C).
- **Cost**: <1 CPU-minute. Measured via T_decomp.
- **Go gate**: U1 shows different topological structure than null (prevalence, density, branch margin, or signed Jaccard distinguishable from random/init baselines).

### Phase 2: C05-C06 — C=1 Micro-Pilot

- **What**: Train C=1 U1ConvRNN, analyze emergent topology.
- **Falsifies**: "U1 equivariance alone produces causally relevant vortices" (a single scalar channel suffices to encode topological structure).
- **Cost**: Training dominates; 3 seeds × 30,000 updates × `t_train_seed`. Evaluation is lightweight.
- **Go gate**: C=1 trains to acceptable accuracy AND produces clean, sparse topological structure distinguishable from null.

### Phase 3: C07-C10 — Representative Sensitivity + Minimal Surgery

- **What**: Representative sensitivity (C07), minimal sufficiency (C08), local necessity (C09), donor specificity (C10).
- **Falsifies**: "Vortex effect is representative-invariant" AND "effect is local" AND "effect is donor-specific."
- **Cost**: Training reused from Phase 2. Evaluation dominated by surgery (t_surgery per pair) and continuations.
- **Go gate**: Representative variance is acceptably small relative to charge variance AND minimal-transplant effect is acceptably close to canonical-transplant effect.

### Phase 4: C14 — Multichannel Phase Locking

- **What**: Inter-channel phase coherence analysis using trained models from prior phases.
- **Falsifies**: "C>1 behaves like C=1 with phase-locked channels" (channel diversity is meaningful).
- **Cost**: Low — analysis only, no retraining. Uses T_decomp on existing model states.
- **Go gate**: Channels show phase coherence consistent with structured topological encoding (not uncorrelated random phases).

### Phase 5: C16 — Factorial Baseline Decomposition

- **What**: Train and evaluate all four model variants (U1, Plain, ComplexNoEquiv, RealWithEquiv), decompose causal contribution of equivariance vs complex convolution vs radial nonlinearity.
- **Falsifies**: "U1 equivariance, not just complex convolution or radial nonlinearity, is the causal factor."
- **Cost**: Training 4× that of a single model (4 variants × 3 seeds × 30,000 updates). This is the most expensive calibration step.
- **Go gate**: U1ConvRNN shows a unique causal vortex signature distinguishable from all three factorial baselines.

### Phase 6: F01-F03 — Confirmatory

- **What**: Full confirmatory runs with frozen contract, paired hierarchical bootstrap, cross-model comparison (F02), OOD delay generalization (F03).
- **Falsifies**: All primary claims in the frozen contract.
- **Cost**: Highest. N seeds determined by calibration power analysis. Training dominates (N × 30,000 updates per model). Evaluation includes full bootstrap with 199 null draws per family.
- **Only run after**: All calibration gates pass AND V2 contract is frozen.

---

## Candidate Decision Rules (UNFROZEN)

Every threshold below is a **candidate decision rule**, not a frozen stop condition. Each is presented with:
- The candidate threshold value
- A calibration method for determining the final threshold
- A scientific rationale explaining what the threshold guards

All thresholds must be calibrated against empirical distributions from C01-C04 diagnostic baselines and C05-C06 C=1 pilot data before being promoted to FROZEN state.

### Rule 1: C=1 Copy Accuracy

| Field | Value |
|:------|:------|
| Candidate threshold | >90% accuracy |
| Status | UNFROZEN |
| Calibration method | Train C=1 on copy task with L=4, D∈[16,32]. Compute the accuracy distribution across 3 seeds. If the distribution is bimodal (high-accuracy seeds vs low-accuracy seeds), threshold at the separation boundary. If unimodal, threshold at the lower 10th percentile minus 0.5σ to avoid rejecting viable seeds. Run power analysis for distinguishing "trainable" from "untrainable" initialization given 3 seeds. |
| Scientific rationale | A C=1 model that cannot learn the copy task to reasonable accuracy cannot host meaningful topological structure. However, the threshold must not be so strict that it excludes seeds with weaker accuracy but rich topology (the research question is about vortex structure, not peak benchmark performance). The 90% candidate value is a reasonable starting point, drawn from prior experience with C=8 models typically reaching >95%. |
| Guard | If C=1 accuracy is too low, the model does not encode the task. Topological analysis of a non-performing model is uninformative. |

### Rule 2: Defect Density Range

| Field | Value |
|:------|:------|
| Candidate threshold | Lower: 1% defect density. Upper: 90% defect density. |
| Status | UNFROZEN |
| Calibration method | Compute the empirical defect density distribution from C01 (random-field null), C02 (untrained model), and C03 (token embeddings). The lower bound should be the 95th percentile of null density plus a margin ensuring statistical distinguishability. The upper bound should be set where spatial structure becomes statistically indistinguishable from a uniform Poisson process (plasma regime). The 1%/90% candidates are order-of-magnitude provisional estimates; actual calibration requires empirical null distributions. |
| Scientific rationale | If defect density is below the null-distinguishable threshold, the topology is too sparse to analyze — individual defects are so rare that representative-sensitivity analysis (C07) lacks statistical power. If defect density exceeds the spatial-structure threshold, the charge map is a dense plasma with no coherent structures to transplant — the decomposition degenerates to noise. The lower threshold protects statistical power; the upper threshold protects structural interpretability. |
| Guard | Defect density < lower: topology too sparse. Density > upper: topology is plasma (no interpretable structure). Both conditions make causal vortex analysis meaningless. |

### Rule 3: Representative Variance Fraction

| Field | Value |
|:------|:------|
| Candidate threshold | Representative variance < charge variance (i.e., variance(R_representative) / variance(R_charge) < 1, with a candidate of < 50% of charge variance). |
| Status | UNFROZEN |
| Calibration method | From C07: compute the variance in transplant outcome across same-charge representatives (within-pool variance) and compare to the variance across different charge values (between-pool variance). Fit a variance-components model. The threshold should be set such that within-pool variance is a small fraction of between-pool variance, ensuring that representative choice does not dominate the causal signal. The 50% candidate is a heuristic; a formal criterion would be `σ_rep² / σ_charge² < α` where α is calibrated via bootstrap confidence intervals. |
| Scientific rationale | A vortex transplant that means something must not depend on which same-charge vortex the experimenter chose. If the outcome varies as much across same-charge representatives as across different charge values, the causal variable is not charge — it is the specific vortex instance, or the transplant procedure itself. This is a necessary condition for the sufficiency claim. |
| Guard | If representative variance exceeds an acceptable fraction of charge variance, the causal claim fails the representative-invariance requirement. |

### Rule 4: Minimal-Transplant Effect Preservation

| Field | Value |
|:------|:------|
| Candidate threshold | Minimal-transplant effect ≥ 90% of canonical-transplant effect (equivalently: attenuation < 10%). |
| Status | UNFROZEN |
| Calibration method | From C08: compute the correlation and effect-size ratio between minimal and canonical transplant margins across pairs. The threshold should account for measurement noise: if the canonical effect is small (near zero), the ratio is unstable. Calibrate the threshold in effect-size space (Hedge's g) rather than raw margin space, and require the minimal effect's confidence interval to overlap with the canonical effect's confidence interval. The 10% attenuation candidate is a provisional estimate; a formal criterion would be equivalence testing with a pre-specified equivalence bound. |
| Scientific rationale | If the minimal-transplant effect (changing only as much as necessary) is substantially weaker than the canonical transplant (which changes everything), the causal mechanism may be holistic rather than localized to the vortex. A strong attenuation suggests that the surgical decomposition is not the true causal unit — the effect depends on a larger, non-decomposable field region. |
| Guard | Large attenuation between minimal and canonical transplants undermines the locality/sufficiency decomposition. |

### Rule 5: Off-Manifold Intervention Rate

| Field | Value |
|:------|:------|
| Candidate threshold | ≤20% of interventions classified off-manifold. |
| Status | UNFROZEN |
| Calibration method | From C11: compute the manifold validity distribution for natural (non-intervened) states under PCA + kNN projection. Establish a natural-state baseline for reconstruction error and neighbor distance. Then compute the same metrics for transplant-intervened states. The threshold should be a percentile of the natural-state distribution (e.g., 95th percentile of natural reconstruction error) rather than a fixed percentage. The 20% candidate is a heuristic; actual calibration requires comparing intervenable-state distributions against natural-state distributions using a two-sample test. |
| Scientific rationale | If a large fraction of intervenable states fall outside the manifold of naturally occurring hidden states, the behavioral effect may be an OOD artifact rather than a causal mechanism. The transplanted state is a mathematical construct — if the network never naturally produces states resembling it, the output shift may reflect the network's extrapolation failure, not the vortex's computational role. |
| Guard | High off-manifold rate indicates that the transplant procedure constructs states the network was never trained to process, confounding the causal interpretation. |

### Rule 6: Factorial Baseline Effect Comparison

| Field | Value |
|:------|:------|
| Candidate threshold | PlainConvRNN vortex effect ≤ 80% of U1ConvRNN vortex effect (i.e., U1 advantage > 20%). |
| Status | UNFROZEN |
| Calibration method | From C16: compute the causal vortex margin for all four variants (U1, Plain, ComplexNoEquiv, RealWithEquiv). Estimate the difference U1_margin − Plain_margin with a hierarchical bootstrap confidence interval. The threshold should be a non-inferiority margin derived from the measurement noise (standard error of the difference), not a fixed percentage. The 80% candidate is a provisional estimate; the calibrated threshold requires that the U1-advantage CI excludes zero and that the lower bound exceeds the SESOI (smallest effect size of interest). |
| Scientific rationale | If PlainConvRNN (non-equivariant) shows a comparable causal vortex effect, then U1 equivariance is not the distinguishing causal factor — the vortex structure may be an artifact of complex convolution or radial nonlinearity alone. The factorial decomposition (C16) is designed to isolate which architectural component drives the effect. A strong Plain effect would falsify the "equivariance is necessary" claim. |
| Guard | Comparable Plain effect weakens or falsifies the claim that U1 equivariance architecture causes the topological computation. |

### Rule 7: Confirmatory Contract Gate

| Field | Value |
|:------|:------|
| Condition | Any frozen contract gate clause fails. |
| Status | Gated by V2 contract freeze (separate document, §gates). |
| Calibration method | Contract gates are calibrated during the calibration phase and frozen before confirmatory runs begin. Each gate has its own calibration procedure documented in the contract. |
| Guard | Confirmatory runs are invalid if any gate clause used to filter calibration data is not satisfied for confirmatory data. |

---

## Resource Budget

Resources are classified as:
- **FROZEN_ENGINEERING**: Hardware capacity limit of the target machine. Not negotiable without changing hardware.
- **UNFROZEN**: Runtime feasibility estimate. May be refined after smoke benchmarks.

| Resource | Limit | Classification | Source |
|----------|------:|:---------------|:-------|
| GPU VRAM | 14 GB | FROZEN_ENGINEERING | V2 contract §resource_policy/vram_limit_gb. RTX 5080 16 GB envelope minus 2 GB safety margin for CUDA context and OS overhead. |
| System RAM (RSS) | 32 GB | FROZEN_ENGINEERING | V2 contract §resource_policy/rss_limit_gb. Standard workstation configuration matching the PI's development machine. Training/eval typically <8 GB; bootstrap resampling may peak ~16 GB temporarily. |
| Disk | 50 GB | FROZEN_ENGINEERING | V2 contract §resource_policy/disk_limit_gb. Model checkpoints, evaluation artifacts, and analysis outputs across all splits. Per-pair artifacts ~MB scale; 10k pairs × 1 MB ≈ 10 GB provides ample headroom. |
| Calibration wall time | 48 hr | FROZEN_ENGINEERING | V2 contract §resource_policy/wall_limit_hours/calibration. Two calendar days with scheduling slack. |
| Confirmatory wall time | 168 hr | FROZEN_ENGINEERING | V2 contract §resource_policy/wall_limit_hours/confirmatory. One week including job-queue latency. |

**Important distinction**: Hardware capacity limits (VRAM, RSS, Disk, wall-time ceilings) are fixed by the target machine and are FROZEN_ENGINEERING. Runtime feasibility estimates (how much of the wall budget each phase actually consumes) are UNFROZEN and depend on the throughput benchmarks. Feasibility statements such as "calibration will complete in 48 hours" or "confirmatory requires 100 seeds" remain UNFROZEN until smoke benchmarks measure actual T_cell_fwd, T_examples, T_eval, t_surgery, t_null, t_manifold, and t_artifact on the target RTX 5080.

### RSS 32 GB Provenance

The 32 GB RSS limit in the original V1 planning was inherited as a workstation constant. In the V2 contract (`15_PM1_LEARNED_V2_CONTRACT.yaml`, §resource_policy/rss_limit_gb), the 32 GB limit is formally ratified as FROZEN_ENGINEERING with rationale: "32 GB system RAM is the standard workstation configuration. Training and evaluation together typically use <8 GB RSS; bootstrap resampling may peak at ~16 GB temporarily. The 32 GB limit matches the PI's development machine." This is therefore a hardware capacity limit, not an arbitrary threshold.

### Potential Bottlenecks (UNFROZEN)

Identified risks that could violate resource limits; each must be profiled during smoke benchmarks:

1. **C=8 model size at batch_size=64**: Does forward+backward BPTT through D+8=~32 steps × batch_size=64 fit in 14 GB VRAM? The C=8 model has 8 × 2 × H × W hidden state plus intermediate activations. If OOM, reduce batch size (to 32 or 16) and proportionally increase updates.
2. **Factorial baseline (C16)**: Training 4 variants × 3 seeds = 12 training runs dominates calibration wall time. If each training run is ~30 min, C16 alone is ~6 GPU-hours — feasible within 48 hr calibration budget. This estimate is UNFROZEN.
3. **Bootstrap memory**: 199 null draws × 8 families × 100 examples × 8 donors × 15 arms can produce O(10⁷) per-seed outcomes. If memory exceeds 32 GB at evaluation time, switch to chunked/streaming bootstrap or disk-backed storage (as noted in contract sensitivity analysis).
4. **Evaluation continuation scaling**: Evaluation continuations scale as n_test × n_donors × n_arms × n_null_families × n_null_draws. For confirmatory: 100 × 8 × 15 × 8 × 199 ≈ 19 million continuations per seed. If T_eval is slow, this dominates total time. Must benchmark T_eval before finalizing confirmatory seed count.
