# Resource Schedule and Value of Information

## Throughput Model

**Not yet benchmarked.** The following model is a template, not a prediction.

### Training
```
t_seed = n_updates × n_recurrent_steps_per_update × batch_size / throughput
       = 30,000 × (delay~24 + copy_len 4) × 64 / T_train
```
where T_train = measured steps/second on target GPU × model_size_factor.

### Evaluation
```
t_seed = n_pairs × (1 + n_arms + n_null_draws × n_null_families) × n_continuation_steps / T_eval
```
where:
- n_pairs ≈ batch_size × avg_admissible_rate
- n_arms = 15 (V1:13 + V2 additions)
- n_null_draws = 199 (per family, matched)
- n_null_families = 8

### Total
```
t_total = n_model_types × n_seeds × (t_seed_train + t_seed_eval)
```

---

## Phase Estimates (TEMPLATE — Fill from Smoke Benchmarks)

| Phase | Runs | Updates/Seed | Eval Pairs | Estimated Range | Confidence |
|-------|-----:|------------:|----------:|:---------------:|:----------:|
| Calibration C01-C04 (CPU) | 10 seeds × 4 diag | 0 | 100 | <1 min | High |
| Calibration C05-C06 (C=1) | 3 seeds × 2 models | 30,000 | 100 | ? hr | Unknown |
| Calibration C07-C15 (full) | 3 seeds × 2 models | 30,000 | 100 × 15 arms | ? hr | Unknown |
| Calibration C16 (factorial) | 3 seeds × 4 variants | 30,000 | 100 × 15 arms | ? hr | Unknown |
| Confirmatory F01-F02 | N seeds × 2 models | 30,000 | 100 × 15 arms × 199 nulls | ? hr | Unknown |
| Confirmatory F03 (OOD) | N seeds × 2 models | 30,000 | 100 × 15 arms × 199 nulls | ? hr | Unknown |

**Do not freeze these until smoke benchmarks are run.**

---

## Value of Information Ordering

Most valuable (cheapest falsification) → least valuable:

1. **C01-C04: CPU diagnostics** (~1 min)
   - Falsifies: "defects are learned" (F-NEW-A), "defects are U1-specific" (F-NEW-C)
   - Cost: negligible
   - Go: proceed to C05 if U1 shows different topology than null

2. **C05-C06: C=1 micro-pilot** (~? GPU-hr)
   - Falsifies: "U1 equivariance alone produces causally relevant vortices"
   - Cost: moderate
   - Go: proceed to C07-C15 if C=1 produces clean topological structure

3. **C07-C10: Representative sensitivity + minimal surgery** (~? GPU-hr)
   - Falsifies: "vortex effect is representative-invariant" and "effect is local"
   - Cost: moderate
   - Go: proceed to C16 if representative variance < charge variance

4. **C14: Multichannel phase locking** (~? GPU-hr)
   - Falsifies: "C>1 behaves like C=1 with phase-locked channels"
   - Cost: low (uses C05-C10 models)
   - Go: proceed to C16 if channels are phase-locked

5. **C16: Factorial baseline** (~? GPU-hr)
   - Falsifies: "U1 equivariance, not just complex convolution or radial nonlinearity, is the causal factor"
   - Cost: high
   - Go: proceed to confirmatory if U1 has unique causal signature

6. **F01-F03: Confirmatory** (large)
   - Falsifies: all primary claims
   - Cost: highest
   - Only run after all calibration gates pass and contract is frozen

---

## Stop Conditions Per Phase

| Phase | Stop If |
|-------|---------|
| CPU diagnostics | Defect density and branch margins identical between U1 and Plain null |
| C=1 micro-pilot | C=1 fails to achieve >90% accuracy on copy task |
| C=1 topology | Defect density < 1% (too sparse to analyze) or > 90% (plasma, no structure) |
| Representative | Representative variance > 50% of charge variance |
| Minimal surgery | Minimal transplant effect < 10% of canonical transplant |
| Manifold | >20% of interventions marked off-manifold |
| Factorial baseline | PlainConvRNN shows comparable causal vortex effect (>80% of U1) |
| Confirmatory | Frozen contract gate clause fails |

---

## Resource Budget (Placeholder)

| Resource | Limit | Source |
|----------|------:|--------|
| GPU VRAM | 14 GB | V1 config (RTX 5080 16 GB minus overhead) |
| Wall time (calibration) | 48 hr | Arbitrary; adjust after throughput benchmark |
| Wall time (confirmatory) | 168 hr | Arbitrary; adjust after calibration |
| Disk | 50 GB | Per-pair artifacts are ~MB; 10k pairs × 1MB ≈ 10GB |
| Memory (RSS) | 32 GB | V1 config |
