# CPU Diagnostic Results — Exact Replication

**Status**: COMPLETE (exact 128-example V1 gate replication)
**Script**: `diagnostics/run_untrained_topology.py`
**Raw data**: `diagnostics/raw/untrained_gate_replication.json`
**Wall time**: 139s CPU

---

## D-01: Test Suite

| Metric | Value |
|--------|-------|
| Command | `pytest tests/ -q` |
| Result | **71 passed**, 0 failed, 0 skipped |
| Environment | CPU, PyTorch 2.12+, NumPy 2.0+ |
| Commit | `67b216b` |

---

## D-02: Exact V1 Gate Replication (Untrained Topology)

**Method**: Replicated V1 gate exactly: `generate_copy_batch(seed, "test/heldout-delay-64", TEST_EXAMPLES=128, TEST_DELAY=64)`, `run_copy(model, symbols, 64)`, `analyze_topology(hidden.post_write)` per example. 10 seeds × 2 model types = 20 runs, 2560 states analyzed.

### Per-Seed State Prevalence (128 examples each)

| Model | Seed | Post-Write Prevalence | Pre-GO Prevalence | Site Density (PW) | Site Density (PG) |
|-------|-----:|---------------------:|------------------:|------------------:|------------------:|
| U1 | 0 | **1.000** | **1.000** | 0.3331 | 0.2480 |
| U1 | 1 | 1.000 | 1.000 | 0.3280 | 0.1892 |
| U1 | 2 | 1.000 | 1.000 | 0.3236 | 0.0599 |
| U1 | 3 | 1.000 | 1.000 | 0.3361 | 0.2921 |
| U1 | 4 | 1.000 | 1.000 | 0.3391 | 0.2384 |
| U1 | 5 | 1.000 | 1.000 | 0.3329 | 0.2447 |
| U1 | 6 | 1.000 | 1.000 | 0.3289 | 0.4831 |
| U1 | 7 | 1.000 | 1.000 | 0.3319 | 0.1182 |
| U1 | 8 | 1.000 | 1.000 | 0.3277 | 0.2007 |
| U1 | 9 | 1.000 | 1.000 | 0.3336 | 0.2163 |
| Plain | 0 | **1.000** | **1.000** | 0.3380 | 0.3503 |
| Plain | 1 | 1.000 | 1.000 | 0.3297 | 0.3361 |
| Plain | 2 | 1.000 | 1.000 | 0.3332 | 0.3428 |
| Plain | 3 | 1.000 | 1.000 | 0.3334 | 0.3019 |
| Plain | 4 | 1.000 | 1.000 | 0.3339 | 0.3110 |
| Plain | 5 | 1.000 | 1.000 | 0.3296 | 0.3108 |
| Plain | 6 | 1.000 | 1.000 | 0.3270 | 0.3302 |
| Plain | 7 | 1.000 | 1.000 | 0.3398 | 0.3606 |
| Plain | 8 | 1.000 | 1.000 | 0.3329 | 0.3413 |
| Plain | 9 | 1.000 | 1.000 | 0.3361 | 0.3098 |

### Key Findings

1. **State-level prevalence = 1.0 for all 20 seeds** (both U1 and Plain, post-write and pre-go).
   - At least one channel has both + and - defects in **every single state** (128 × 20 = 2560 states).
2. **Post-write site density ≈ 0.33**: ~33% of all 2048 plaquettes (8 channels × 256) have nonzero charge.
3. **Pre-GO site density varies**: U1 pre-GO density ranges from 0.06 to 0.48 (seed-dependent decay).
4. **U1 recurrence reduces defect count**: post-write density (~0.33) → pre-GO density (0.06–0.48). The recurrent dynamics either annihilate or stabilize defects during blank recurrence.
5. **Plain recurrence does NOT reduce defects**: post-write (~0.33) → pre-GO (~0.33). PlainConvRNN's LayerNorm+tanh preserves the rough phase texture.

### Gate Verdict

```text
U1 seed 0 untrained prevalence: 1.0 (exact)
Plain seed 0 untrained prevalence: 1.0 (exact)

defect_learned_not_innate: STRUCTURALLY IMPASSABLE
Reason: Gate requires trained_prevalence > 1.0 for ALL seeds.
       Prevalence bounded at [0, 1]. Mathematical impossibility.

Verdict: CONFIRMED P0 — exact gate-input replication complete.
```

---

## D-03: Token Embedding Topology

**Method**: Extracted raw token embeddings from untrained U1ConvRNN and PlainConvRNN (seed 0). Analyzed topology per token (tokens 1-8).

### Key Results

| Model | Token | nonzero_defect | Site Count | Site Density | Min Magnitude |
|-------|------:|:---:|-----------:|-------------:|:-------------:|
| U1 | 1-8 | True (8/8) | 660-706 | 0.322-0.345 | 0.0 |
| Plain | 1-8 | True (8/8) | 660-724 | 0.322-0.354 | 0.0 |

- **All 16 token embeddings have nonzero_defect=True** — topological charge exists at initialization.
- **Min magnitude ≈ 0**: Embeddings initialized from normal distribution; at least one grid point is near-zero per embedding.
- **Site count per token ≈ 660-724**: ~33% of all plaquettes across 8 channels have nonzero charge.

### Implications

- Vortex structure pre-exists in token embeddings (initialization artifact, not learned).
- The research question is "does training REORGANIZE vortex structure?" not "does training CREATE it?"
- U1 and Plain embeddings are very similar (same initialization scheme, comparable defect counts).

---

## D-04: Gate Reachability (Updated)

| Clause | Passable? | Evidence |
|--------|:---------:|----------|
| `defect_learned_not_innate` | **✗ IMPASSABLE** | Exact 128-example replication confirms untrained prevalence = 1.0 for all 20 seeds |
| All other clauses | ? | Not tested (requires trained models) |

**Recommendation**: Remove `defect_learned_not_innate` from mandatory prerequisites. Replace with analyzable-stable-topology gate and two-sided training-vs-untrained secondary analysis.
