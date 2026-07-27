# Additional Findings (Beyond Prior 43)

These are findings discovered during the current DRAFT-0 audit that were not in the original 43 findings.

---

## F-NEW-A: `defect_learned_not_innate` Structurally Impassable [P0]

- **Severity**: P0 — prevents V1 pilot from passing its own decision gate
- **Evidence**: CPU diagnostic confirms untrained state-level defect prevalence = 1.0 for both U1ConvRNN and PlainConvRNN, for token embeddings and untrained hidden states. The gate requires `all(trained > untrained)` but `untrained = 1.0` and prevalence is bounded at `[0, 1]`.
- **Impact**: V1 pilot cannot pass. V2 must redefine this gate before any training.
- **Root cause**: `nonzero_defect = any(channel has both + and -)` is too permissive for a 8×16×16 random-like field. ~40% of plaquettes have nonzero charge per channel.
- **Recommendation**: Use `defect_density` (per-plaquette) or `branch_margin` or `signed Jaccard stability` as the learned-not-innate metric.

---

## F-NEW-B: Vortices Are Present in Token Embeddings (Initialization Artifact) [P1]

- **Severity**: P1 — confounds "learned" vs "pre-existing" topology
- **Evidence**: All 16 token embeddings (8 tokens × 2 models) show nonzero_defect=True with ~330-360 pairs per token. This occurs at model initialization, before any training.
- **Impact**: Vortex topology exists *before* any learning occurs. The research question is not "does learning create vortices?" but "does learning alter pre-existing vortex structure in a causally meaningful way?"
- **Recommendation**: V2 should include token embedding topology as a baseline diagnostic. The research claim should be "trained dynamics reorganize/sculpt vortex structure," not "create it."

---

## F-NEW-C: PlainConvRNN Also Has Vortex-Like Defects [P1]

- **Severity**: P1 — weakens U1-specificity claim
- **Evidence**: Both U1ConvRNN and PlainConvRNN show nonzero_defect=True in all tested states (post_write, pre_go). Defect density is comparable: U1 (325-341 pairs/channel), Plain (332-361 pairs/channel).
- **Impact**: The mere presence of defects does not distinguish U(1)-equivariant from non-equivariant models. The causal intervention must demonstrate that U1 defects are *functionally different* (more causally relevant), not just more numerous.
- **Note**: The decision gate already accounts for this: "Either PlainConvRNN shows no vortex phenomenon, OR U1ConvRNN advantage exceeds PlainConvRNN." But "no vortex phenomenon" is clearly false — PlainConvRNN has comparable defect density.
- **Recommendation**: The cross-model comparison clause should focus on *causal efficacy* difference, not raw prevalence difference.

---

## F-NEW-D: Branch Margins Are Near-Critical for Untrained States [P2]

- **Severity**: P2 — decomposition numerical stability risk
- **Evidence**: Branch margin minima: 0.0000–0.0016 rad for untrained states. A margin of 0.0000 rad means the phase field has a genuine branch point (link variable = ±π). This makes charge extraction potentially unstable.
- **Impact**: Decomposition may fail or produce unreliable charges when branch margins are this small. The `extract_charge(tolerance=1e-10)` may give different results for numerically equivalent fields with different floating-point paths.
- **Recommendation**: V2 should add a `branch_margin_min > threshold` validity check at decomposition time. States with near-critical margins should be flagged or excluded.

---

## F-NEW-E: Token Embedding Minimum Magnitude = 0.0 [P2]

- **Severity**: P2 — causes invalid-channel exclusion in topology analysis
- **Evidence**: Token embeddings have `min_magnitude = 0.0` (confirmed: `emb_*` rows in diagnostic). This means at least one pixel per channel has zero amplitude, making the phase undefined there.
- **Impact**: `analyze_topology` skips channels with `min(|field|) ≤ MAGNITUDE_EPSILON`. This is a silent exclusion — channels may be valid at inference time even if embeddings have near-zero components after the initial state computation incorporates them.
- **Recommendation**: Document the exclusion. Consider whether embedding analysis should use a different magnitude epsilon or if the near-zero embedding regions are a real concern.

---

## F-NEW-F: Charge Density Is Extremely High [P2]

- **Severity**: P2 — affects interpretation of "defect" as a meaningful computational unit
- **Evidence**: ~340 pairs per channel × 8 channels = ~2720 pairs per state on 256 plaquettes = ~10.6 pairs per plaquette. This means the field is saturated with topological charge — it's more like a "vortex plasma" than a "few well-separated vortices."
- **Impact**: A vortex transplant that swaps ~2720 charge locations may succeed for reasons unrelated to specific computational function (e.g., it just happens to contain all information due to density). The specificity arm must demonstrate that *specific* vortices are causally relevant, not just any dense perturbation.
- **Recommendation**: V2 should include a "sparse defect" intervention variant — only a subset of vortices (e.g., top-k by some criterion) are transplanted.

---

## F-NEW-G: Clean-Process Replay Broken (Feasibility Path) [P2]

- **Severity**: P2 — reproducibility defect in feasibility pipeline
- **Evidence**: `topological/pilot.py:132` calls `aligned_mask_transplant.pm1_pilot --replay-hash`, which is an external module not present in this repository.
- **Impact**: The `clean_process_replay_exact` clause in feasibility decision will always evaluate to `False`. The feasibility pipeline cannot prove its own reproducibility.
- **Recommendation**: Either remove the stale dependency (preferred) or document it as an external, versioned dependency with a clear provenance trail.

---

## F-NEW-H: `minimum_magnitude` Unused in Decision Pipeline [P3]

- **Severity**: P3 — computed but not leveraged
- **Evidence**: `HiddenComponents.minimum_magnitude` is computed at `interventions.py:99` but never referenced in `learned_evaluation.py` or any decision gate.
- **Impact**: States with very low magnitude may have unreliable vortex decompositions, but no gate catches this.
- **Recommendation**: Either use `minimum_magnitude` in a magnitude validity gate, or remove it to avoid dead code.

---

## F-NEW-I: Equivariance Check Coverage Is Shallow [P2]

- **Severity**: P2 — over-trust in limited probe
- **Evidence**: `learned_smoke.py:254` uses `torch.allclose(rotated, rerotated)` with a single phase value and few tokens. It tests global phase rotation of the full forward pass, not the autonomous recurrent core in isolation.
- **Impact**: A model could pass this check while violating equivariance in edge cases (near-zero magnitude, specific token transitions, long recurrence chains).
- **Recommendation**: V2 should separate equivariance checks: embedding transformation, blank transition, token-input transition, readout invariance. Report maximum equivariance error over a grid of phase values and hidden state probes.

---

## Summary Table

| ID | Severity | Gate Impact | Requires V2 Change? |
|----|:--------:|:-----------:|:-------------------:|
| F-NEW-A (defect_learned_not_innate) | **P0** | Gate impassable | **Yes — must redesign** |
| F-NEW-B (embedding vortices) | P1 | Confounds claim | Yes — add baseline |
| F-NEW-C (Plain also has defects) | P1 | Cross-model clause | Mention in interpretation |
| F-NEW-D (near-critical branch margins) | P2 | Numerical risk | Add validity check |
| F-NEW-E (embedding zero magnitude) | P2 | Silent exclusion | Document or fix |
| F-NEW-F (high charge density) | P2 | Specificity claim | Add sparse intervention |
| F-NEW-G (stale replay dependency) | P2 | Reproducibility | Remove stale path |
| F-NEW-H (unused minimum_magnitude) | P3 | None | Use or remove |
| F-NEW-I (equivariance coverage) | P2 | Over-confidence | Deepen check |
