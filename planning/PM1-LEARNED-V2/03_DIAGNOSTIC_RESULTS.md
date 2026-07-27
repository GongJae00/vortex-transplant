# Low-Cost Diagnostic Package Results

## D-01: Test Suite

| Metric | Value |
|--------|-------|
| Command | `pytest tests/ -q` |
| Result | **71 passed**, 0 failed, 0 skipped |
| Runtime | ~6s |
| Environment | CPU, Python 3.11+, PyTorch 2.12+ |
| Commit | `67b216b` |

---

## D-02: Random Phase Null Topology

**Method**: Generated 10 IID uniform phase fields and 10 band-limited smooth phase fields (16×16 grid, 8 channels). Extracted topology using `extract_charge`.

### Key Results

| Phase Family | State Prevalence | Defect Density (mean ± σ) | Branch Margin Min | Branch Margin Mean |
|-------------|:----------------:|:-------------------------:|------------------:|-------------------:|
| IID uniform (seed 0-9) | **1.0** | 1705 ± 28 pairs/state | 0.0001–0.0024 rad | 1.55–1.59 rad |
| Band-limited smooth (seed 0-9) | **0.0** | 0.0 | 2.49–2.85 rad | 3.09–3.12 rad |

### Implications

1. **IID random phase fields have 100% state-level prevalence.** The `nonzero_defect = any(channel has both + and -)` definition is trivially satisfied for any rough phase field.
2. **Branch margins are extremely small** (median ~0.0004 rad) for random fields — most edges are very close to a branch cut.
3. **Smooth fields correctly show zero defects** — the topological analysis pipeline works.
4. **Charge density is very high**: ~340 positive + 340 negative per channel = ~680 defects per channel = ~5440 defects per state across 8 channels on a 16×16 = 256-plaquette grid. This is ~2.7 defects per plaquette per channel on average.

---

## D-03: Token Embedding & Untrained Hidden Topology

**Method**: Extracted raw token embeddings and untrained hidden states from U1ConvRNN and PlainConvRNN (seed=0, no training).

### Key Results

| Source | nonzero_defect | Defect Pairs per Channel | Branch Margin Min | Min Magnitude |
|--------|:---:|------------------------:|------------------:|:-------------:|
| U1 token 1-8 embedding | **True** (8/8) | 330–353 | 0.0001–0.0016 rad | 0.0 |
| U1 post_write (ex0-1) | **True** (2/2) | 325–341 | 0.0002–0.0012 rad | 0.0000–0.0003 |
| U1 pre_go (ex0-1) | **True** (2/2) | 325 | 0.0002–0.0003 rad | 0.0000–0.0003 |
| Plain token 1-8 embedding | **True** (8/8) | 330–362 | 0.0001–0.0013 rad | 0.0 |
| Plain post_write (ex0-1) | **True** (2/2) | 332–361 | 0.0000–0.0003 rad | 0.055–0.062 |
| Plain pre_go (ex0-1) | **True** (2/2) | 332–361 | 0.0000–0.0002 rad | 0.304–0.341 |

### Implications

1. **Vortices in raw token embeddings are an initialization artifact** — not learned. All 16 token embeddings (8 tokens × 2 models) have nonzero_defect=True with ~330-360 pairs.
2. **PlainConvRNN also has vortices** — the defect phenomenon is not unique to U(1)-equivariance. This partially confirms the hypothesis that topological defects are generic in high-dimensional random complex fields, not specific to equivariant dynamics.
3. **U1 has lower magnitude** (0.0001 vs 0.055) — the U1ConvRNN tends to produce more nearly-unit-magnitude fields (~0.0 min magnitude), while PlainConvRNN has larger magnitude variation.
4. **Embedding magnitudes are exactly 0.0** — the raw embeddings approach zero amplitude somewhere (due to normal distribution initialization), making vortex extraction potentially unstable in those channels.
5. **Token embeddings already have ~340 pairs** — this is comparable to post-write hidden states. If trained states don't significantly *reduce* this number, the "learned" component of defect prevalence may be zero or negative.

---

## D-04: Gate Reachability Analysis

### Current V1 Gate Clauses vs. Diagnostic Evidence

| Clause | Can Pass? | Evidence | Recommended Action |
|--------|:---------:|----------|--------------------|
| `prerequisites_met` (accuracy ≥ 95%) | ✓ | Trainability not in question | — |
| `prerequisites_met` (defect ≥ 50%) | ✓ | Trivially satisfied | — |
| `prerequisites_met` (persistence ≥ 50%) | ? | Untested | Needs measurement |
| `pair_counts_sufficient` | ✓ | Decomposition seems robust | — |
| `topology_guards_hold` | ✓ | Smooth field test passed | — |
| `interventions_complete` | ✓ | Smoke verifies 13 arms | — |
| `directional_sanity` | ? | Needs trained model | — |
| `component_guards_pass` | ? | Needs trained model | — |
| `nuisance_guards_pass` | ? | Needs trained model | — |
| `majority_advantages_positive` | ? | Needs trained model | — |
| `bootstrap_lower_positive` | ? | Needs trained model | — |
| `split_hashes_unique` | ✓ | Hash namespace architecture correct | — |
| `no_evaluation_errors` | ? | Depends on training outcome | — |
| **`defect_learned_not_innate`** | **✗ STRUCTURALLY IMPOSSIBLE** | Untrained prevalence = 1.0 | **REDESIGN REQUIRED** |

### The `defect_learned_not_innate` Problem

```python
# learned_evaluation.py:711-712
"defect_learned_not_innate": untrained_record is None or all(
    record.get("defect_prevalence", 0.0) > untrained_record.get("defect_prevalence", 0.0)
    for record in seed_records
)
```

**Diagnostic evidence**: Untrained state-level prevalence = 1.0 (confirmed for both U1 and Plain, embedding and hidden state). The gate requires ALL trained seeds to have prevalence > 1.0. This cannot happen because prevalence is bounded at [0, 1].

### Proposed Redesign Options

1. **Defect density** (\( \rho_d \)): Use `total_defects / (C × H × W)` instead of `nonzero_defect`
   - Gate: `trained_density > untrained_density × k` (k > 1)
2. **Branch margin improvement**: `mean_branch_margin_trained > mean_branch_margin_untrained * k`
   - Training should stabilize defects (increase branch margin)
3. **Signed Jaccard stability**: Cross-seed topology reproducibility
   - Trained models of the same type should converge to similar topology
4. **Cross-model comparison**: U1 produces *qualitatively different* topology than Plain
   - Not just "more defects" but "different defect structure"
5. **Change null baseline**: Use smooth-phase-field null (prevalence=0) as a stricter baseline
   - But this confuses "defect existence" with "defect structure"

**Most conservative approach**: Combine (1) and (2) — require trained models to have higher density AND higher stability than untrained, tested as paired comparison per seed.
