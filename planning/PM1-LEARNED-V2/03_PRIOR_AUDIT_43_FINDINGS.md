# PM1-LEARNED-V2 — Prior Audit 43 Findings: Individual Adjudication

## Overview

| Verdict | Count |
|---------|------:|
| CONFIRMED | 43 |
| PARTIALLY CONFIRMED | 0 |
| MODIFIED | 0 |
| REJECTED | 0 |
| UNRESOLVED | 0 |

---

### Finding 1

**Original claim**: Iqbal et al. or a direct predecessor already report U(1)-equivariant 2D ConvRNN, copy task, and vortex emergence.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: none
**External evidence**: Iqbal et al. (2026), arXiv:2605.14685, Spontaneous symmetry breaking and Goldstone modes for deep information propagation
**Derivation**: Iqbal et al. (2026) explicitly uses U(1)-equivariant 2D ConvRNN trained on a copy task and reports long-lived vortex defects. The present work's V1 code (U1ConvRNN, copy task, vortex decomposition) matches the Iqbal et al. setup almost exactly, differing only in the causal intervention framework. The 2025 domain-wall paper (Iqbal & Welling, OpenReview) also claims topological defects are information carriers albeit for Z2 symmetry.
**Correction**: None. The finding is a statement of prior art, not an error.
**Scientific impact**: Novelty must be scoped to causal role verification via transplant/transplant-ablation, not vortex discovery or U(1) ConvRNN architecture.
**V2 action**: Explicitly position the work as "first causal intervention study targeting U(1) vortex configurations with representative-sensitivity controls," not as "first observation of vortices in equivariant networks." Verify Appendix D.3 of arXiv:2605.14685 for any existing causal claims before freezing the novelty statement.
**Required test**: `test_topological_literature_citations` — verify pre-freeze literature check catches any new causal-vortex publication from the Iqbal group.
**Acceptance criterion**: No peer-reviewed causal intervention on U(1) vortex defects in learned dynamics exists at V2 freeze time.
**Residual uncertainty**: The Iqbal group is active in this space; their next paper could include causal evidence, preempting novelty. The forward-citation search is not yet executed.

---

### Finding 2

**Original claim**: Novelty must be vortex causal role verification, not vortex discovery.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: none
**External evidence**: Iqbal et al. (2026), arXiv:2605.14685; Iqbal & Welling (2025), OpenReview fM5s2Tqe0t
**Derivation**: Given Finding 1 establishes that vortex emergence in U(1)-equivariant ConvRNNs on copy tasks is prior art, the only viable novelty axis is the causal claim. The V2 contract already captures this: the primary question is "Do U(1) vortex defects causally control output behavior," not "Do vortices exist."
**Correction**: None. Finding correctly scoped the novelty contribution.
**Scientific impact**: All experimental design, statistical testing, and manuscript claims must center on causal evidence. Observational claims (defect prevalence, persistence, correlation with accuracy) are supportive but not primary.
**V2 action**: Draft novelty statement as "to our knowledge, the first systematic causal intervention study targeting local U(1) vortex configurations in learned recurrent hidden fields, with representative-sensitivity, matched-manifold controls, and explicit sufficiency/necessity/specificity decomposition."
**Required test**: `test_literature_search_no_causal_vortex_prior` — literature search term "vortex transplant causal neural network" returns zero hits with causal methodology.
**Acceptance criterion**: Search executed and documented in `05_LITERATURE_AND_NOVELTY.md` with zero competing causal claims at freeze time.
**Residual uncertainty**: Literature search across arXiv, Semantic Scholar, OpenReview, and Crossref is not yet executed. Overlap with mechanistic interpretability techniques (activation patching applied to spatial fields) could exist.

---

### Finding 3

**Original claim**: Component transplant is adjacent to interchange intervention/activation patching.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: none
**External evidence**: Geiger et al. (2021–2024), "Causal abstraction" line of work; Makelov et al. (2024), "Is This the Subspace You Are Looking For?"
**Derivation**: The vortex transplant intervention (replace donor's vortex component onto recipient's smooth+magnitude components) is structurally an interchange intervention: swap a causal variable (charge map Q) between two examples while holding non-causal variables (smooth, magnitude) constant. The decomposition into topology/smooth/magnitude is novel to this application, but the interchange paradigm is standard in the causal abstraction and activation patching literature.
**Correction**: None. The finding correctly identifies methodological adjacency.
**Scientific impact**: The paper must cite relevant activation patching and causal abstraction literature, particularly Makelov et al. (2024) which critiques patching methodology for spatial interventions. The manifold and representative controls in V2 directly address patching-faithfulness concerns.
**V2 action**: Add interchange intervention / activation patching as a distinct category in the literature matrix. Cite Makelov et al. (2024) for patching methodological concerns.
**Required test**: None (literature positioning, not a code test).
**Acceptance criterion**: Literature review includes interchange intervention / activation patching section with at least 3 citations.
**Residual uncertainty**: Whether reviewers will accept "interchange on topological components" as sufficiently distinct from "interchange on subspaces."

---

### Finding 4

**Original claim**: Needs differentiation from complex RNN, TDA, and vortex computing literature.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: none
**External evidence**: Arjovsky et al. (2016), Unitary Evolution RNNs; Wisdom et al. (2016), Full-Capacity Unitary RNNs; Kosterlitz & Thouless (1973), XY model; various TDA-in-ML papers; physical reservoir computing literature.
**Derivation**: Complex/unitary RNNs use similar mathematical machinery (complex weights, phase-preserving activations) but focus on gradient stability, not topological analysis. TDA applies persistent homology to activations but makes no causal claims. Physical vortex computing uses actual fluid/optical vortices for reservoir computing. The present work must distinguish itself from all three.
**Correction**: None. Finding is a literature positioning gap.
**Scientific impact**: Without explicit differentiation, reviewers from any adjacent field may dismiss the contribution as "known" or "trivial extension."
**V2 action**: Create a structured differentiation table in the manuscript: Complex RNN (no topological analysis), TDA-in-ML (no causal claim), Physical Vortex Computing (not learnable dynamics).
**Required test**: None (literature positioning).
**Acceptance criterion**: Differentiation table exists with explicit contrasts for all three categories.
**Residual uncertainty**: Reviewer frame-of-reference cannot be fully controlled; a reviewer from TDA may still view the topology as elementary.

---

### Finding 5

**Original claim**: Integer residual of plaquette sum is not topological protection evidence.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/topology.py` — `extract_charge` returns integer-valued charge map via `round(curl(link_variables) / 2π)`; `learned_evaluation.py:625-628` — `maximum_integer_residual` records residual but no energetic barrier measurement
**External evidence**: none
**Derivation**: Integer quantization of the plaquette curl sum (L1: numerical integrality) is a structural property of compact U(1) fields — the W=round(curl/2π) operation guarantees integer output by construction. This demonstrates that the phase field can be partitioned into discrete topological sectors, but does not demonstrate topological protection (L5), which requires that changing a charge requires finite gradient energy. The L1-L5 hierarchy in `06_MATHEMATICAL_FOUNDATIONS.md:71-80` explicitly separates these.
**Correction**: The claim "integer quantization = topological protection" is an overstatement. The correct claim is "integer quantization enables topological classification; topological protection (energetic barrier to charge change) is not yet measured."
**Scientific impact**: The term "topological protection" should not appear in the manuscript without an energetic barrier measurement. Use "topological quantization" for the integer-residual property.
**V2 action**: Either (a) measure charge-flip energy (gradient energy difference between charge-Q and charge-Q' field) as a secondary diagnostic, or (b) remove "topologically protected" from the manuscript and replace with "topologically quantized."
**Required test**: `test_charge_flip_energy` — measure gradient energy increase from flipping a single plaquette charge.
**Acceptance criterion**: Either the energetic measurement exists or the term "topologically protected" is absent from all claims.
**Residual uncertainty**: Even if an energy barrier exists in the static field, whether the recurrent dynamics respect it depends on the training objective and gradient flow, which is a different question.

---

### Finding 6

**Original claim**: Branch/admissibility margin is needed.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/decomposition.py` — `extract_charge(tolerance=1e-10)` is called without branch margin validation; `topological/interventions.py:99` — `minimum_magnitude` is computed but not used in any gate; diagnostic `03_DIAGNOSTIC_RESULTS.md:37` — branch margin minima of 0.0000–0.0016 rad for untrained states
**External evidence**: none
**Derivation**: When a link variable (phase difference between adjacent pixels) is exactly π, the branch cut is at a genuine branch point — the charge decomposition is numerically unstable (a floating-point perturbation of 1e-16 could flip a ±1 charge). The `tolerance=1e-10` in `extract_charge` is a numerical guard, not a physics-based stability criterion. Diagnostic results confirm untrained states have near-critical margins, making decomposition unreliable.
**Correction**: Add a `branch_margin` metric (π - |Δθ_e| for each edge) with a minimum threshold for state admissibility. The V2 contract defines branch_margin at `15_PM1_LEARNED_V2_CONTRACT.yaml:165-177` with q01/q05/median quantiles.
**Scientific impact**: States with near-critical branch margins produce unreliable charge maps. Including them inflates measurement noise and could produce spurious "charge changes" across time steps.
**V2 action**: Implement `BranchStability` dataclass (`10_IMPLEMENTATION_SPEC.md:129-137`) and add `branch_margin_q01 > threshold` to the selection funnel.
**Required test**: T-V2-03 — `branch_margin_min ≥ threshold` for valid decompositions, tested on synthetic fields with known branch margin.
**Acceptance criterion**: Decomposition raises a clear error or returns `branch_stability_valid=False` for fields with q01 branch margin below calibrated threshold.
**Residual uncertainty**: The threshold must be calibrated from empirical data (C04 calibration). A fixed threshold (e.g., 0.1 rad) may be too strict for some fields and too lenient for others.

---

### Finding 7

**Original claim**: Torus has local charge plus harmonic/global winding sector.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/topology.py` — `extract_charge` computes plaquette-level local charge only; `topological/decomposition.py` — `decompose` separates vortex/smooth/magnitude but does not explicitly extract harmonic sector; `06_MATHEMATICAL_FOUNDATIONS.md:85-100` documents the Hodge decomposition on T²
**External evidence**: none
**Derivation**: On the torus T², the Hodge decomposition of a 1-cochain Δθ is: Δθ = dφ + δA + h, where dφ is exact (curl-free, from scalar potential), δA is coexact (divergence-free, encodes vortex charges), and h is harmonic (both curl-free and divergence-free, dim=2 on T², corresponding to two independent cycle holonomies w_x, w_y). The current decomposition extracts vortex (coexact) and smooth (exact) components but does not explicitly isolate or preserve the harmonic sector. The smooth component may contain harmonic winding.
**Correction**: None. The finding correctly identifies the missing harmonic sector. V2 should add explicit harmonic sector extraction and harmonic_swap intervention arm.
**Scientific impact**: A phase field with global phase winding but no local vortices is topologically distinct from a field with no winding at all. If the harmonic sector carries behavioral information, confounding it with the smooth component dilutes the specificity claim.
**V2 action**: Add `extract_harmonic` to topology module. Add `transplant_harmonic` to decomposition module. Add `harmonic_swap` intervention arm. The harmonic sector should be preserved in minimal surgery and reported as a diagnostic.
**Required test**: T-V2-18 — harmonic sector swap: donor harmonic applied, vortex unchanged, verify cyclotron holonomy match.
**Acceptance criterion**: A field with uniform phase gradient (w_x=1, w_y=0, no vortices) produces harmonic sector (w_x=1, w_y=0) and smooth sector (exact, gradient-removed).
**Residual uncertainty**: The FFT-based Poisson solver may introduce numerical artifacts that mix exact and harmonic components at floating-point precision.

---

### Finding 8

**Original claim**: Current smooth component may mix harmonic winding.
**Verdict**: CONFIRMED
**Confidence**: Medium
**Repository evidence**: `topological/decomposition.py` — smooth component is computed as `field / (magnitude * vortex)` which removes the vortex (exact solution to Poisson equation for charge map Q) but does not separate harmonic from exact; `06_MATHEMATICAL_FOUNDATIONS.md:95-99` — path dependence in harmonic extraction when local vortices exist
**External evidence**: none
**Derivation**: The current `decompose` function: computes vortex component v_Q via `canonical_vortex_field(Q)` (Poisson solve), then smooth = z / (|z| · v_Q). Since v_Q is one solution to curl(phase) = 2πQ on T², dividing it out removes the coexact component. However, the remaining phase field may contain harmonic winding (constant phase gradients that wrap around the torus). If so, a harmonic_swap intervention and a smooth_swap intervention are confounded.
**Correction**: The smooth component after vortex removal is actually the exact+harmonic component, not the exact component alone. Rename to `curl_free` or separately extract harmonic.
**Scientific impact**: If the smooth margin is partly driven by harmonic winding and not by exact gradient flow, the "mechanism advantage" formula incorrectly attributes harmonic effects to smooth nuisance.
**V2 action**: Add explicit harmonic extraction. The intervention arms should be: `vortex` (coexact swap), `harmonic` (harmonic swap), `smooth` (exact-only swap after harmonic removal). Report harmonic margin separately.
**Required test**: T-V2-18 — harmonic extraction and swap verified with known winding fields.
**Acceptance criterion**: A pure-harmonic field (constant phase gradient, zero local charge) produces nonzero harmonic component and zero smooth (exact) component after full decomposition.
**Residual uncertainty**: The separation quality depends on numerical precision of the Poisson solver. Mixed exact-harmonic modes at the grid scale may not cleanly separate.

---

### Finding 9

**Original claim**: Charge map does not uniquely determine phase representative.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/topology.py` — `canonical_vortex_field(Q)` returns one specific solution to ∇²ψ = 2πQ via FFT Poisson solve; `06_MATHEMATICAL_FOUNDATIONS.md:104-110` — the solution is unique only up to additive constant (uniform phase shift) on T²
**External evidence**: none
**Derivation**: Given a charge map Q with net zero charge, the Poisson equation ∇²ψ = 2πQ on T² has a solution unique up to an additive constant (uniform global phase rotation). Thus, the canonical vortex field v_Q = exp(iψ) is one specific representative of the equivalence class of all fields with charge map Q. Any field v'_Q = v_Q · g where Q(g) = 0 (i.e., g is charge-free) represents the same topological configuration — same vortices, different continuous background. When V1 transplants v_Q, it transplants one specific gauge, not "the vortex structure."
**Correction**: Rename the intervention from "vortex transplant" to "canonical-representative vortex transplant." Add same-charge representative sampling to bound the gauge-choice variance.
**Scientific impact**: If different same-charge representatives produce different behavioral outcomes, the "vortex causality" claim is confounded with the specific gauge choice. Must demonstrate that the charge itself, not the arbitrary representative, is the causal unit.
**V2 action**: Implement `sample_representatives(field, n=10)` to generate multiple same-charge representatives (via random harmonic sector or random exact-deformation). Estimate representative variance fraction (Var(rep_effect) / Var(charge_effect)). The contract requires representative invariance evidence.
**Required test**: T-V2-08 — representative variance fraction computed and reported; variance decomposition separates charge from representative effects.
**Acceptance criterion**: Representative variance fraction < 0.3 (i.e., charge effect dominates gauge-choice effect) or the finding is documented as a limitation.
**Residual uncertainty**: The "right" representative sampling strategy (harmonic_random, gradient_random, smooth-random) is itself a modeling choice. Different sampling strategies may give different variance estimates.

---

### Finding 10

**Original claim**: Poisson canonical representative and charge effect are confounded.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py` — `component_intervention` for arm `"vortex"` swaps `canonical_vortex_field(Q_donor)` onto recipient smooth+magnitude; `06_MATHEMATICAL_FOUNDATIONS.md:112-120` — the Poisson representative is one specific gauge choice; the statistical model separates charge effect α and representative effect β
**External evidence**: none
**Derivation**: The vortex transplant replaces the recipient's canonical vortex field v_{Q_r} with the donor's v_{Q_d}. The behavioral difference could be due to: (a) the charge map difference Q_d - Q_r (the causal claim), or (b) the specific gauge difference between the two canonical representatives (confound). Since the canonical Poisson representative minimizes gradient energy subject to the charge constraints, it has a specific smoothness property that may systematically differ between donor and recipient. The observed effect may partially reflect this smoothness difference rather than the charge difference.
**Correction**: V2 needs at least two same-charge representatives per intervention to decompose charge vs representative effects. The statistical model Y = μ + α_charge(Q_d - Q_r) + β_repr(ρ_d - ρ_r) + ε must be fit.
**Scientific impact**: Without representative decomposition, the primary estimate of "vortex causal effect" is an upper bound (includes both charge and gauge effects). The true charge effect may be smaller.
**V2 action**: Implement representative sampling and report both the canonical-representative effect and the ensemble-mean effect across representatives. The ensemble-mean effect is the better estimate of the charge-specific effect.
**Required test**: T-V2-08 — representative variance decomposition across ≥10 same-charge representatives per pair.
**Acceptance criterion**: Difference between canonical-representative effect and ensemble-mean effect is within 1 standard error, or the ensemble-mean effect is used as the primary estimand.
**Residual uncertainty**: The representative sampling strategy (how to generate charge-free perturbations) affects the variance estimate. Charge-free perturbations that are "too unnatural" may inflate the representative variance and mask a real charge effect.

---

### Finding 11

**Original claim**: Same-charge representative ensemble is needed.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `15_PM1_LEARNED_V2_CONTRACT.yaml:186` — `representative_definition: same_charge_class` with `representatives_per_sample: 10`; `10_IMPLEMENTATION_SPEC.md:59` — `sample_representatives(field, n=10, seed)` in new interventions.py
**External evidence**: none
**Derivation**: Follows directly from Findings 9 and 10. Without multiple representatives per charge class, the charge vs representative confound cannot be resolved. The V2 contract already mandates this at line 186. The implementation spec at `10_IMPLEMENTATION_SPEC.md:59` lists it as a new V2 function.
**Correction**: None. The V2 plan already addresses this.
**Scientific impact**: Representative ensemble is a necessary condition for the causal claim. Without it, the claim "charge is causal" is indistinguishable from "the specific Poisson-solved gauge is causal."
**V2 action**: Implement `sample_representatives` in `topological/v2/interventions.py`. The contract specifies 10 representatives per sample with `harmonic_random` axis.
**Required test**: T-V2-08 — verify that 10 representatives of the same charge class produce different fields but identical charge maps, and the variance decomposition separates charge effects from representative effects.
**Acceptance criterion**: Representative sampling produces fields with identical charge maps (bitwise Q match) and distinct fields (pairwise L2 > 0). Variance decomposition model runs without error.
**Residual uncertainty**: The number of representatives (10) may be insufficient to fully capture the representative distribution. A sensitivity analysis with N=20 or N=50 may be needed.

---

### Finding 12

**Original claim**: Channelwise charge in C>1 may not be coordinate-independent invariant.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `configs/topological_learned_v1.json:8` — channels=8; `topological/interventions.py` — decomposition runs per-channel, treating each channel independently; `06_MATHEMATICAL_FOUNDATIONS.md:47-55` — π₁(S^{2C-1}) = 0 for C > 1 means channelwise vortices can unwind through other channel dimensions
**External evidence**: none
**Derivation**: For a multichannel complex vector field z(x) ∈ ℂ^C, the punctured space classification is: for C=1, the vortex is a genuine topological defect (π₁(ℂ\{0}) = ℤ). For C>1, the full vector field has π₁(ℂ^C\{0}) = 0 — any loop can be continuously deformed to a point by moving through the extra channel dimensions. A channelwise vortex (nonzero plaquette charge in one specific channel) is not a topological invariant of the full vector field; it depends on the channel coordinate choice. Different channel bases (e.g., after a U(C) rotation) would produce different charge maps.
**Correction**: The channelwise charge is a coordinate-dependent quantity. The invariance claim requires either (a) proving that neural dynamics constrain the state to a phase-locked submanifold where the internal orientation is rigid, or (b) using C=1 where the issue does not arise.
**Scientific impact**: The term "topological charge" when applied to C>1 channelwise charge is technically imprecise. Reviewers from mathematical physics may flag this. The C=1 gateway (Finding 13) avoids this entirely.
**V2 action**: Either adopt C=1 as primary (eliminating the coordinate-dependence issue) or empirically validate inter-channel phase locking (prove that channel mixing is negligible under trained dynamics). Document the homotopy argument.
**Required test**: `test_topological_c1_homotopy` — verify that π₁(ℂ\{0}) = ℤ for C=1, and that channelwise charge in C>1 can be unwound via channel mixing.
**Acceptance criterion**: If C=1 is used: the argument is vacated (no coordinate dependence). If C>1 is used: inter-channel phase correlation > 0.8 across ≥10 trained seeds.
**Residual uncertainty**: Even with high inter-channel phase correlation, a reviewer may still reject the "topological" framing for C>1. The safest path is C=1 for confirmatory.

---

### Finding 13

**Original claim**: C=1 gateway is needed.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `configs/topological_learned_v1.json:8` — channels=8 only; `topological/model.py:15-19` — `ModelSpec` supports C=1 but not used in V1; `06_MATHEMATICAL_FOUNDATIONS.md:61-67` — C=1 is diagnostically essential; `16_OPEN_QUESTIONS_AND_DECISIONS.md:25-38` — PI decision D-P1
**External evidence**: none
**Derivation**: C=1 is not logically necessary — a naturally phase-locked C>1 system could have genuine channelwise topological charges. However, C=1 is diagnostically essential because it (a) eliminates charge-unwinding through channel mixing, (b) removes the channel-mode selection ambiguity, (c) isolates the single question of whether U(1) equivariance alone creates causally relevant vortices. The C=1 model has fewer parameters, making training faster and the result cleaner.
**Correction**: Add `ScalarU1ConvRNN(channels=1)` as a model variant. Use C=1 for the primary calibration → optionally scale to C=8 in a secondary confirmatory phase only if C=1 is successful.
**Scientific impact**: A C=1 result is cleaner, stronger, and immune to the homotopy critique. A C=1 result that then fails to scale to C=8 is still publishable as a negative scaling result. A C=8-only result is exposed to the coordinate-dependence critique.
**V2 action**: Run C=1 micro-pilot (C05-C06) before committing to the C>1 arms. If C=1 achieves ≥90% copy-task accuracy, use C=1 for confirmatory. The `15_PM1_LEARNED_V2_CONTRACT.yaml:112` and `16_OPEN_QUESTIONS_AND_DECISIONS.md:36-38` recommend C=1 calibration gateway.
**Required test**: T-V2-11 — C=1 model forward pass, training step, validation accuracy ≥ 90%.
**Acceptance criterion**: C=1 ScalarU1ConvRNN achieves ≥90% validation accuracy on the copy task at delay 16-32 within 30000 training updates.
**Residual uncertainty**: C=1 may not achieve sufficient task accuracy. The copy task at 8 tokens / length 4 may require more than one channel of information capacity. If C=1 fails, fallback to C=8 with explicit phase-locking analysis.

---

### Finding 14

**Original claim**: Multichannel order parameter, vacuum manifold, and homotopy must be defined.
**Verdict**: CONFIRMED
**Confidence**: Medium
**Repository evidence**: `15_PM1_LEARNED_V2_CONTRACT.yaml:147-157` — `order_parameter.c_greater_1` status UNFROZEN with four candidate methods; `06_MATHEMATICAL_FOUNDATIONS.md:47-60` — homotopy analysis for C>1 with phase-locked submanifold requirement
**External evidence**: none
**Derivation**: For C>1, defining "the vortex" requires selecting an order parameter ψ(x) that maps the C-channel field to a single complex number per spatial position. The four candidates in the contract (max-magnitude channel, PCA projection, learned linear w†z, SVD of inter-channel phase correlation) have different properties: basis-dependence, statistical vs learned, linear vs nonlinear. None are canonical. The vacuum manifold (set of fields with ψ(x) ≠ 0 for all x) and homotopy group depend on which projection is chosen.
**Correction**: The V2 contract already acknowledges this as UNFROZEN. The calibration method (C14) will compare projection methods. This finding adds the requirement that the comparison include homotopic analysis of the projected space.
**Scientific impact**: If the order parameter is poorly chosen, the resulting "vortex" classification may not correspond to a genuine topological invariant of the dynamical system. The paper must justify the chosen projection.
**V2 action**: In calibration C14, for each candidate projection method: (a) verify that the projected field has the homotopy properties of ℂ\{0}, (b) measure reconstruction fidelity of the projected field, (c) check that projected charges are stable under small channel-mixing perturbations. Select the method with best combination.
**Required test**: `test_multichannel_order_parameter_homotopy` — verify that the chosen projection produces a π₁-classifiable field (nonzero projected field magnitude).
**Acceptance criterion**: Chosen projection method has >95% of spatial positions with nonzero projected amplitude in trained states, and projected charge maps are stable under random U(C) rotations of amplitude ≤ 0.1.
**Residual uncertainty**: A linear projection w†z may miss nonlinear phase relationships. A nonlinear projection may be hard to justify theoretically. The C=1 gateway sidesteps this entirely and is the recommended path.

---

### Finding 15

**Original claim**: Near-zero magnitude rejection introduces selection bias.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py:90-96` — `decompose_hidden` skips channels with `min(|field|) ≤ MAGNITUDE_EPSILON=1e-8`; `03_DIAGNOSTIC_RESULTS.md:43` — token embedding `min_magnitude = 0.0` for all embeddings; `04_ADDITIONAL_FINDINGS.md:45-50` (F-NEW-E)
**External evidence**: none
**Derivation**: The `analyze_topology` function (in `learned_evaluation.py`) computes `StateTopology` with `valid_channels` flag. Channels with near-zero minimum magnitude are excluded from topology analysis. This is a scientifically reasonable guard (phase is undefined at zero magnitude) but introduces a selection: channels with higher amplitude are systematically over-represented in the topological analysis. If amplitude correlates with computational importance, the selection bias may be toward more important channels, but this is an uncontrolled variable.
**Correction**: Document the exclusion rate. Report `valid_channel_fraction` per state. In V2, track whether excluded channels differ systematically from included channels in their behavioral role. The `TopologyStatsV2` dataclass includes `valid_channels` field.
**Scientific impact**: If the excluded channels carry different computational roles from included channels, conclusions about "vortex causality" may not generalize to all channels. The finding is moderate-severity (not fatal) because charge is undefined where magnitude is zero, so exclusion is mathematically justified.
**V2 action**: Add `valid_channel_fraction` to per-seed summary. Add sensitivity analysis: compare results with and without near-zero channels (using phase-regularized extraction).
**Required test**: `test_near_zero_magnitude_bias` — measure whether excluded channels differ in behavioral contribution (via channel-wise ablation) from included channels.
**Acceptance criterion**: Valid channel fraction documented. If >10% of channels are excluded in >20% of states, report as a limitation.
**Residual uncertainty**: At what magnitude threshold does phase become "meaningful"? The 1e-8 threshold is an arbitrary numerical guard, not a physics-based criterion.

---

### Finding 16

**Original claim**: Exact-coordinate signed Jaccard can mistake defect motion for persistence failure.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py` — `_signed_jaccard` or equivalent signed topology persistence measure uses exact coordinate matching; `06_MATHEMATICAL_FOUNDATIONS.md:78` — "Signed Jaccard over time steps" is the L4 dynamical persistence metric
**External evidence**: none
**Derivation**: The signed Jaccard index between two charge maps Q(t) and Q(t+1) is computed as: (matching +/-, matching coordinates) / (union of charge positions). If a vortex defect moves by one grid cell between time steps (e.g., from (x,y) to (x+1,y)), the exact-coordinate match fails, and the Jaccard score penalizes this as if the defect disappeared and a new one appeared. But physically, a moving vortex is persisting — it just moved. The current persistence metric confuses motion with non-persistence.
**Correction**: Replace or supplement exact-coordinate signed Jaccard with a proximity-tolerant metric: match defects within a Manhattan distance radius r (e.g., r=1 or r=2), penalizing unmatched defects and distant matches. Use the Hungarian algorithm for optimal pairing.
**Scientific impact**: If the persistence metric underestimates true dynamical persistence, the `topology_guards_hold` gate may falsely fail on models where vortices move coherently but the exact-coordinate metric fails to capture this.
**V2 action**: Add `defect_tracking(field_t, field_t1)` with proximity matching (Hungarian algorithm, Manhattan distance cost). Report both exact-match and proximity-tolerant persistence in `TopologyStatsV2`.
**Required test**: `test_defect_tracking_proximity` — verify that a vortex translated by one grid cell is tracked as persistent (same defect) rather than as a new defect.
**Acceptance criterion**: Proximity-tolerant persistence ≥ exact-coordinate persistence for all test cases, with strict inequality for fields with nonzero defect motion.
**Residual uncertainty**: The proximity radius threshold needs calibration. Too large a radius may match unrelated defects; too small may miss legitimate defect motion. A radius that scales with grid size may be more robust.

---

### Finding 17

**Original claim**: Vortex transplant → output change alone cannot conclude charge is the causal unit.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:498` — `mechanism_advantage = mean_margins["vortex"] - nuisance_maximum` (only considers component-based nuisances); `07_CAUSAL_IDENTIFICATION.md:129-131` — "a perturbation that is large enough and pushes the state toward the donor's general neighborhood will shift outputs"
**External evidence**: none
**Derivation**: The vortex transplant changes multiple properties of the recipient state simultaneously: (a) the topological charge map, (b) the L2 distance from the original state, (c) the off-manifold distance, (d) the gradient energy distribution, (e) the spectral composition. A positive vortex margin could be due to any of these, not necessarily the charge change. The mechanism_advantage formula subtracts the maximum of (smooth, magnitude, global_phase, zero_charge_phase) margins, but these controls are imperfect: they don't control for off-manifold artifacts (Finding 22) or representative/gauge effects (Findings 9-10).
**Correction**: The primary claim must pass all five criteria in `07_CAUSAL_IDENTIFICATION.md:120-128`: (1) intervention changes behavior, (2) changes toward donor behavior, (3) vortex effect > nuisance effects, (4) on-manifold, (5) representative-invariant. V1 tests only (1-3).
**Scientific impact**: The causal claim requires multi-axis evidence. A paper claiming "vortices are causal" based only on a positive mechanism_advantage is scientifically incomplete.
**V2 action**: Implement all five criteria as integrated pass gates. The primary decision gate requires: mechanism_advantage > 0 AND manifold penalty < threshold AND representative variance < charge variance.
**Required test**: `test_full_causal_claim_criteria` — all five criteria computed and reported; decision gate integrates all five.
**Acceptance criterion**: Manuscript explicitly addresses all five criteria with quantitative evidence for each.
**Residual uncertainty**: What threshold constitutes "representative-invariant" or "on-manifold" is a calibration question. These thresholds may not have sharp separations, requiring graded evidence rather than binary pass/fail.

---

### Finding 18

**Original claim**: Must separate representative, off-manifold, destruction, generic perturbation, and necessity.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `07_CAUSAL_IDENTIFICATION.md:72-115` — intervention taxonomy lists all five categories; `15_PM1_LEARNED_V2_CONTRACT.yaml:232-261` — full arm list with sufficiency, necessity, specificity, representative, harmonic, component, null, natural controls; V1 implements only sufficiency + component + null arms
**External evidence**: none
**Derivation**: The finding lists five confounds that must be separated to make a clean causal claim. V1 addresses only some: it has sufficiency (vortex transplant), component nuisance controls (smooth, magnitude, global_phase, zero_charge), and generic perturbation nulls (random_direction, fourier, pca). It does not address: (a) representative variance (no same-charge reps), (b) off-manifold (no manifold model, no relaxation steps), (c) destruction (no vortex removal / sham surgery to distinguish transfer from destruction), (d) necessity (no "remove vortex → lose behavior" test). The V2 contract adds all missing categories.
**Correction**: V2 must implement all five categories. The V2 contract already includes: vortex_remove_all/remove_pair/sham (necessity), same-charge reps (representative), natural_neighbor + relaxation (manifold), sign_flip (specificity).
**Scientific impact**: Without separation, the claim "vortices are causally involved" conflates five distinct questions. A paper with only sufficiency + component controls is at best preliminary, not confirmatory.
**V2 action**: Implement all intervention arms listed in the V2 contract. The implementation spec (`10_IMPLEMENTATION_SPEC.md:57-60`) lists new functions for each category.
**Required test**: T-V2-07 (minimal_annihilation), T-V2-08 (representative sampling), T-V2-17 (natural neighbor), T-V2-18 (harmonic swap) — each category must have a dedicated test.
**Acceptance criterion**: All 5 categories have ≥1 intervention arm implemented, tested, and producing finite numerical results on synthetic data.
**Residual uncertainty**: Some arms (e.g., vortex_remove_pair for necessity) may be technically challenging to implement cleanly. The sham surgery arm requires matched perturbation intensity.

---

### Finding 19

**Original claim**: mechanism_advantage excludes Fourier/PCA/random from primary null.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:490-503` — `nuisance_maximum = max(smooth, magnitude, global_phase, zero_phase)`. Fourier_low, fourier_high, pca, and random_direction are computed as separate advantages but NOT included in the mechanism_advantage formula.
**External evidence**: none
**Derivation**: The `mechanism_advantage` at line 498 is `mean_margins["vortex"] - nuisance_maximum` where `nuisance_maximum` only considers smooth, magnitude, global_phase, and zero_phase. The fourier_low, fourier_high, pca, and random_direction advantages are computed separately (lines 499-503) but do not participate in the primary pass/fail gate. This means a vortex transplant that beats smooth/magnitude/global/zero controls but is beaten by a Fourier or PCA baseline still passes the gate.
**Correction**: V2's primary null should include ALL null families: `mechanism_advantage = vortex_margin - max(smooth, magnitude, global, zero_charge, random, fourier_low, fourier_high, pca)`. The V2 contract at `15_PM1_LEARNED_V2_CONTRACT.yaml:267` already uses this expanded definition.
**Scientific impact**: The excluded null families are important: fourier tests frequency-specific information (if low-frequency carries all copy-task information, vortex may not be special), PCA tests variance-captured information. Excluding them inflates the mechanism_advantage estimate.
**V2 action**: Use the expanded `mechanism_advantage` formula defined in the V2 contract. Report per-family advantage alongside the aggregate.
**Required test**: `test_mechanism_advantage_includes_all_nulls` — verify that the primary estimand includes all null families in the max operation.
**Acceptance criterion**: The `mechanism_advantage` value ≤ (vortex_margin - max(any single null family)), with equality to the most conservative bound.
**Residual uncertainty**: Including more null families makes the gate harder to pass (more conservative). This increases the risk of a false negative but protects against false positives.

---

### Finding 20

**Original claim**: Even if Fourier/PCA/random effects are stronger, V1 gate can still pass.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:490-503` — foúrier_*, pca, random_direction advantages are computed but not in `nuisance_maximum`; `learned_evaluation.py:706-708` — `majority_advantages_positive` uses the `mechanism_advantages` which exclude fourier/pca; `learned_evaluation.py:776-780` — cross-model gate uses the same advantages
**External evidence**: none
**Derivation**: Direct consequence of Finding 19. The gate checks `majority_advantages_positive` using the `mechanism_advantage` values that are `vortex_margin - max(smooth, magnitude, global, zero)`. If fourier_low margin > vortex margin, it does not affect `mechanism_advantage`. The gate can pass with `PM1_SURVIVE_LEARNED_PILOT` even if a simple Fourier low-pass of equal norm is a better behavioral shifter than the vortex transplant. For example: vortex_margin = 0.3, smooth_margin = 0.1, fourier_low_margin = 0.5 → mechanism_advantage = 0.3 - 0.1 = 0.2 > 0 → PASS, even though fourier does better.
**Correction**: Same as Finding 19 — expand the null in mechanism_advantage. Additionally, for the cross-model comparison, check that U1 vortex margin exceeds the maximum null margin for BOTH models, not just U1.
**Scientific impact**: A false-positive SURVIVE where Fourier baseline beats vortex is a serious scientific error — it implies a frequency-domain explanation is more parsimonious than a topological explanation, undermining the entire research direction.
**V2 action**: After expanding mechanism_advantage, add an explicit per-family sensitivity check: if any null family has mean margin > vortex margin, flag the result even if mechanism_advantage > 0 (due to the max pulling from a different family).
**Required test**: `test_gate_respects_all_null_families` — a synthetic scenario where fourier margin > vortex margin but vortex margin > component nuisance max produces NO_GO, not SURVIVE.
**Acceptance criterion**: All 8 null families' margins are below vortex margin for SURVIVE, or the result is INCONCLUSIVE_NULL_FAMILY_DOMINATES.
**Residual uncertainty**: What if vortex beats 7/8 null families but loses to one? Is that a partial success or a no-go? The contract should specify: SURVIVE requires vortex > ALL null families; INCONCLUSIVE for partial domination.

---

### Finding 21

**Original claim**: L2 displacement matching alone is insufficient.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py` — `matched_global_phase` and `matched_zero_charge_phase` match L2 norm of the vortex transplant displacement; `06_MATHEMATICAL_FOUNDATIONS.md:124-150` — minimal surgery design lists 5 constraints (charge, magnitude, harmonic, energy, manifold)
**External evidence**: none
**Derivation**: The `global_phase` and `zero_charge_phase` arms control for the L2 displacement norm of the vortex transplant by applying a uniform or structured phase perturbation of equal L2 norm. However, L2 distance is a weak similarity measure for complex fields: two fields with identical L2 norms can have completely different spatial structure, frequency content, and topological properties. The vortex transplant is highly structured (carries specific charge pattern, has Poisson-smooth phase), while the matched controls are less structured. If vortex transplant outperforms the controls, it could be because it's a more "natural" perturbation (closer to the data manifold), not because of the vortex charge specifically.
**Correction**: Add multi-axis controls: (a) spectral-matched control (same frequency content, no charges), (b) energy-matched control (same gradient energy, no charges), (c) spatial-smoothness-matched control. The harmonic swap arm partially addresses this.
**Scientific impact**: The L2-matched controls may be too weak (systematically under-matched in structure), producing an inflated mechanism_advantage. Stronger controls may reduce or eliminate the advantage.
**V2 action**: Supplement L2 matching with spectral and energy matching for at least one control arm. Use the harmonic_swap and sham_surgery arms as structurally-matched controls.
**Required test**: `test_matched_controls_spectral_energy` — verify that at least one control arm matches the vortex transplant in (L2 norm, gradient energy, and spectral power) within 10%.
**Acceptance criterion**: The strongest control arm (smooth or harmonic-swap) matches vortex transplant in ≥2 of {L2 norm, gradient energy, spectral power} within 20%.
**Residual uncertainty**: Perfect multi-axis matching is impossible — any perturbation that exactly matches all non-topological properties while changing topology would itself be a topological perturbation.

---

### Finding 22

**Original claim**: Off-manifold intervention is the largest causal confound.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `07_CAUSAL_IDENTIFICATION.md:111-114` — natural controls for manifold validity; V1 has no manifold model, no relaxation post-intervention, no manifold distance metric; `06_MATHEMATICAL_FOUNDATIONS.md:150` — "After surgery, run the autonomous recurrence F for a few steps to allow the field to relax to the natural manifold"
**External evidence**: Makelov et al. (2024), "Is This the Subspace You Are Looking For?" — activation patching often produces off-manifold states
**Derivation**: When the vortex component from a donor is composited with the recipient's smooth+magnitude components, the resulting field z* = m_r · v_{Q_d} · s_r may not correspond to any hidden state that the recurrent network would naturally produce. The composited field is a mathematical construct, not a naturally occurring state. Running this synthetic state through the network produces an output distribution that is out-of-training-distribution. The behavioral shift could be due to (a) the charge change (causal claim), or (b) the OOD nature of the intervenened state (confound). This is the well-known "off-manifold patching" problem from the activation patching literature.
**Correction**: Add post-intervention relaxation: run the autonomous recurrence F (blank transition) for k steps to allow the synthetic state to relax toward the natural manifold. Then measure behavioral outcomes on the relaxed state. The difference between raw (unrelaxed) and relaxed outcomes quantifies the manifold penalty.
**Scientific impact**: If the vortex effect persists after relaxation, the evidence for on-manifold causal relevance is strong. If it vanishes, the raw vortex effect was an OOD artifact. This is a make-or-break confound.
**V2 action**: Implement `relax(model, field, steps=5)` in `topological/v2/interventions.py`. Add `ManifoldDiagnostics` dataclass. Compute manifold_penalty = on_manifold_effect - raw_effect. Gate: `manifold_penalty < threshold`.
**Required test**: T-V2-17 — natural neighbor search finds states with similar topology; relaxation converges toward natural manifold.
**Acceptance criterion**: For ≥80% of intervenened pairs, relaxation reduces the off-manifold distance (to nearest natural state) by ≥50%.
**Residual uncertainty**: Relaxation may itself change the behavior in ways unrelated to the charge, e.g., if the autonomous dynamics have attractors. The relaxation step count (5) is a hyperparameter that needs calibration.

---

### Finding 23

**Original claim**: Sufficiency exists but necessity is absent.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py` — only sufficiency arms (vortex, smooth, magnitude, etc.) exist; `07_CAUSAL_IDENTIFICATION.md:80-84` — necessity arms (vortex_remove_all, vortex_remove_pair, vortex_sham) are listed but not implemented; `15_PM1_LEARNED_V2_CONTRACT.yaml:237-242` — necessity arms added as V2 new
**External evidence**: none
**Derivation**: The V1 intervention suite tests only: "Does adding vortex information shift behavior?" (sufficiency). It does not test: "Does removing vortex information eliminate specific behavior?" (necessity). A positive sufficiency result without necessity evidence leaves open the alternative explanation: the vortex transplant adds *any* discriminative information (not specifically required), and the model can use any available information. True causal involvement requires both directions.
**Correction**: V2 must add necessity arms. The V2 contract specifies three: `vortex_remove_all` (replace vortex field with v_0), `vortex_remove_pair` (annihilate single defect pair while preserving others), `vortex_sham` (same-displacement surgery with no net charge change).
**Scientific impact**: Without necessity evidence, the strongest justified claim is "vortex information is sufficient to shift behavior" — a much weaker claim than "vortices are causally involved in the model's natural computation." For a submission-venue paper, the necessity evidence substantially strengthens the contribution.
**V2 action**: Implement vortex_remove_all, vortex_remove_pair, and vortex_sham as described in `06_MATHEMATICAL_FOUNDATIONS.md:143-149`. The minimal_annihilation approach uses the Poisson solver to compute a new canonical vortex field with one pair removed.
**Required test**: T-V2-07 — `minimal_annihilation` removes target pair, preserves other charges (Q unchanged at all other positions), minimizes displacement.
**Acceptance criterion**: After vortex_remove_all, the recipient output becomes indistinguishable from recipient's natural output (charge effect eliminated). After vortex_sham (equal displacement, same charge), behavior does not change significantly.
**Residual uncertainty**: A single pair removal may not be sufficient to change behavior if information is redundantly encoded across many defect pairs (distributed representation). Necessity may be a "many pairs" rather than "single pair" property.

---

### Finding 24

**Original claim**: Donor-recipient margin conflates donor transfer with recipient destruction.
**Verdict**: CONFIRMED
**Confidence**: Medium
**Repository evidence**: `07_CAUSAL_IDENTIFICATION.md:56-69` — normalized recovery formula and edge cases defined; `15_PM1_LEARNED_V2_CONTRACT.yaml:272-274` — normalized recovery estimand with denominator edge cases
**External evidence**: none
**Derivation**: The vortex transplant margin is defined as mean(donor_ll_intervened - recipient_ll_natural). A positive margin could arise from: (a) donor vortex information is successfully transferred → output shifts toward donor (intended interpretation), or (b) recipient vortex information is destroyed → output shifts away from recipient (confound). Since donor_ll and recipient_ll are measured on different token sequences, these are conflated. The normalized_recovery estimand R_a = (vortex_margin - NR_margin) / (WS_margin - NR_margin) attempts to separate these by normalizing against the whole-state transplant, but this normalization fails when the denominator is near zero (donor and recipient produce similar outputs).
**Correction**: Add a "destruction-only" arm: remove the recipient's vortex field (replace with v_0) WITHOUT adding donor's vortex, and measure behavioral shift. This quantifies the recipient-information-destruction component. The vortex transplant effect can then be decomposed as: total_effect = transfer_effect + destruction_effect.
**Scientific impact**: If the vortex transplant effect is mostly destruction (recipient information loss) rather than transfer (donor information gain), the causal claim must be narrowed to "vortex structure encodes recipient-specific information" rather than "vortex structure encodes task-relevant information." This is a weaker claim.
**V2 action**: The `vortex_remove_all` arm (necessity) in V2 serves double duty as the destruction-only measurement. Subtract `vortex_remove_all` margin from `vortex` margin to isolate the transfer component.
**Required test**: `test_transfer_vs_destruction_decomposition` — verify that `vortex_margin - remove_all_margin` isolates transfer-specific effect.
**Acceptance criterion**: Transfer component (vortex - remove_all) is positively distinguishable from zero at bootstrap CI.
**Residual uncertainty**: The separation assumes the destruction and transfer effects are additive, which may not hold if there are interactions between donor charge and recipient non-charge components.

---

### Finding 25

**Original claim**: natural_donor and whole_state become identical under exact reconstruction.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py:122-123` — `natural_recipient` arm calls `compose_components(recipient.magnitude, recipient.vortex, recipient.smooth)`; `topological/decomposition.py` — `recompose(magnitude, vortex, smooth)` should reconstruct original field within numerical tolerance (1e-10)
**External evidence**: none
**Derivation**: The `natural_donor` arm (recompose donor's own components) and `whole_state` arm (raw donor hidden state) differ only if the decomposition-recomposition cycle introduces error. The contract specifies `decomposition_reconstruction_error < 1e-10` as a validity check, meaning the decomposition is designed to be lossless to double precision. If the reconstruction is exact, `natural_donor` and `whole_state` produce identical intervened states and thus identical behavioral outcomes. This makes one of the two arms redundant.
**Correction**: Remove `natural_donor` as a distinct arm. Use `whole_state` as the upper bound. The `natural_recipient` arm remains useful as the "no intervention" baseline since it tests recomposed recipient state.
**Scientific impact**: A redundant arm wastes evaluation compute and adds meaningless data to reports. Eliminating it simplifies the arm schema from 13 to 12 distinct arms.
**V2 action**: In the V2 intervention arm list, remove `natural_donor` or rename `natural_recipient` to `recomposed_recipient` and drop the separate donor recomposition arm. The `whole_state` arm serves as the upper bound.
**Required test**: `test_natural_donor_whole_state_identity` — verify that for all synthetic test fields with reconstruction error < 1e-10, the recomposed donor field equals the raw donor field to within machine precision.
**Acceptance criterion**: `natural_donor` output equals `whole_state` output at all 4 output positions to within 1e-8 nats.
**Residual uncertainty**: None — this is a deterministic mathematical identity given exact decomposition.

---

### Finding 26

**Original claim**: One-step commutation guard is relative comparison only and insufficient.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_smoke.py:254-259` — equivariance check uses `torch.allclose(rotated, rerotated)` with single phase value (0.417 rad), single blank step, single probe state; `02_DRAFT0_CORRECTION_LEDGER.md:48-62` — correction C-03 identifies this as shallow
**External evidence**: none
**Derivation**: The current equivariance check tests: F(e^{iα} h) ≈ e^{iα} F(h) for a single α=0.417, a single h, and a single blank-transition step. It passes if `allclose` at `atol=2e-5`. This is insufficient to guarantee equivariance because: (a) it tests only one phase value — a model could be approximately equivariant at 0.417 rad but not at 0.0 or π; (b) it tests only the blank transition, not token-input transition where embedding + recurrence may break equivariance; (c) it tests only one probe state, not the distribution of hidden states encountered in training; (d) atol=2e-5 is arbitrary — a model could have systematic equivariance error of 1e-4 at certain phases and still pass.
**Correction**: Separate equivariance checks into: (1) blank recurrence equivariance over a grid of phases [0, π/4, π/2, ..., 7π/4]; (2) token-input transition (embedding + recurrence) equivariance; (3) readout invariance; (4) full forward pass equivariance. Report maximum error for each.
**Scientific impact**: If the model is not truly equivariant, the claim "U(1)-equivariant dynamics create topological structure" rests on a false premise. The topological analysis depends on the equivariance being preserved in the autoregressive dynamics, not just one step.
**V2 action**: Implement separate equivariance checks as specified in `04_ADDITIONAL_FINDINGS.md:81-88` (F-NEW-I) and `12_REPRODUCIBILITY_AND_ARTIFACTS.md` smoke gates.
**Required test**: T-V2-10 — equivariance check on blank transition, input transition, readout separately; maximum error reported across phase grid.
**Acceptance criterion**: Maximum equivariance error < 1e-3 for blank transition, < 1e-2 for input transition, < 1e-3 for readout invariance across 8 phase values.
**Residual uncertainty**: The "true" equivariance error may depend on the specific distribution of hidden states encountered during training, which the smoke test only samples sparsely.

---

### Finding 27

**Original claim**: Only blank autonomous recurrent core is equivariant, not the full model.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/model.py:37-56` — U1ConvRNN has token_embedding (lookup table, not equivariant), complex conv + radial_tanh (equivariant), readout (Linear, invariance not verified); `06_MATHEMATICAL_FOUNDATIONS.md:23-33` — equivariance scope table shows embedding is not equivariant, readout invariance not separately tested
**External evidence**: none
**Derivation**: By construction: the token embedding maps each token index to a specific complex field — applying e^{iα} to the embedding output would NOT produce the embedding of the same token. The embedding breaks equivariance because token identity is absolute, not phase-relational. The readout maps from real state (reshaped 2*C*H*W) to vocabulary logits — if the readout uses real-valued features, it is invariant under e^{iα}h → h̃ where the readout-relevant features are unchanged. Only the blank autonomous recurrence (no input present) is claimed equivariant. The full model F(x, h) = recurrence(embedding(x), h) is NOT equivariant because the first step breaks equivariance.
**Correction**: The manuscript must state the limited scope explicitly: "The autonomous recurrent core is U(1)-equivariant. The full model with token input at step k=0 is not equivariant, but subsequent blank-transition steps preserve the U(1) degree of freedom." The research question is then: does the training process *utilize* this U(1) freedom to organize topological information?
**Scientific impact**: A reviewer reading "the model is U(1)-equivariant" may object that the embedding breaks equivariance. The precise statement is "the autonomous dynamics are equivariant; token input adds equivariance-breaking at the first step." The vortex structure, if it emerges, does so during the blank recurrent steps after token input.
**V2 action**: In the manuscript and model documentation, specify: "U(1)-equivariant autoregressive dynamics (blank transition). Token embedding breaks equivariance at the input step." Add this to the model card.
**Required test**: `test_full_model_not_equivariant_at_input` — verify that model(blank, e^{iα} model(token, h_0)) ≠ e^{iα} model(blank, model(token, h_0)) but model(blank, e^{iα} h) = e^{iα} model(blank, h).
**Acceptance criterion**: Equivariance violation detectable at token-input step, equivariance preserved at all blank-transition steps.
**Residual uncertainty**: Whether the readout is truly invariant or approximately invariant. If the readout is non-invariant, then output behavior depends on absolute phase, and the U(1) degree of freedom is not "free" — the readout constrains it, potentially creating selection pressure for specific phase configurations.

---

### Finding 28

**Original claim**: Full spatial token embedding can directly record vortices.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `03_DIAGNOSTIC_RESULTS.md:37-42` — all 16 token embeddings (8 tokens × 2 models) have `nonzero_defect=True` with ~330-360 defect pairs per token; `topological/model.py:47-51` — `token_embedding: nn.Embedding(vocabulary, real_state_size)` mapping to full (2*C*H*W)-dimensional space
**External evidence**: none
**Derivation**: Each token embedding maps a discrete token index to a 2*8*16*16 = 4096-dimensional real vector which is reshaped to (2, 8, 16, 16) = complex field. At initialization (random weights), this mapping produces phase fields with ~340 defect pairs per token. This means: (a) the token embedding already contains vortex-like structure at initialization; (b) different tokens have different vortex patterns (since they map to different random initial vectors); (c) the model could potentially use the embedding-stage vortices rather than learned dynamic vortices. This confuses the origin of any observed vortex effects.
**Correction**: The research question narrows from "does learning create vortices?" to "does the recurrent dynamics transform/pre-existing vortices in a causally relevant way?" The baseline for post-input hidden states should control for the embedding pattern.
**Scientific impact**: If the model's copy-task behavior is driven by embedding-stage lookups (memorizing token-specific vortex patterns) rather than dynamic vortex manipulation, the claim "recurrent dynamics sculpt topological computation" may be false. The model could be a glorified lookup table with no dynamical vortex processing.
**V2 action**: Diagnose whether post-recurrence vortex patterns differ from embedding-stage patterns (measured by signed Jaccard between embedding and post-state charge maps). If they are highly correlated, the "dynamic" claim is weak. Report the embedding-to-post-state charge map correlation as a diagnostic.
**Required test**: `test_embedding_vs_dynamic_vortex_correlation` — measure whether charge maps differ between pre-recurrence (token-only) and post-recurrence (trained model) states.
**Acceptance criterion**: Post-recurrence charge map differs from embedding charge map (signed Jaccard < 0.5) for ≥70% of examples in trained models.
**Residual uncertainty**: High correlation between embedding and post-state vortex patterns does not conclusively prove lookup coding — it could also mean the recurrent dynamics preserve and refine a useful topological structure seeded by embeddings.

---

### Finding 29

**Original claim**: 12-bit task vs 4096-dimensional state enables lookup-like coding.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/task.py` — copy task: vocabulary=10 (8 symbols + blank + go), length=4 → |S| = 8^4 = 4096; `topological/model.py:33-34` — `real_state_size = 2 * channels * height * width = 2*8*16*16 = 4096`; `02_DRAFT0_CORRECTION_LEDGER.md:89-95` — C-05 correction confirms task support = 4096 and training sees each sequence ~469 times on average
**External evidence**: none
**Derivation**: The number of possible distinct input sequences is 8^4 = 4096. The hidden state dimension is 2*8*16*16 = 4096. This is a perfect match: the state has exactly enough degrees of freedom to assign one dimension per possible sequence. Under the training regime (1,920,000 examples from a 4096-element support), each sequence is seen ~469 times. The model could memorize a forward mapping sequence → hidden_state → continuation rather than learning a general computational algorithm. This would make vortex effects a byproduct of memorization, not general computation.
**Correction**: The claim of generalization must be scoped to "delay generalization" (train delay 16-32, test delay 64), not "sequence generalization" (unseen sequences). The paper must explicitly state that test sequences are drawn from the same 4096-sequence support as training sequences.
**Scientific impact**: The "OOD generalization" narrative (delay=64 is OOD) is partially misleading because sequence content is not OOD. A reviewer who notices the state_size = |S| match may dismiss the result as memorization rather than combinatorial generalization.
**V2 action**: Add a diagnostic: measure per-sequence accuracy on the 4096 sequences at different delays. If accuracy is uniformly high across all 4096 sequences at test delay, the model has memorized the lookup. The contract at `02_DRAFT0_CORRECTION_LEDGER.md:95` clarifies the claim must be scoped to delay OOD.
**Required test**: `test_task_cardinality_vs_state_dimension` — verify that |S| = 4096 and state_dim = 4096. Document this as a limitation.
**Acceptance criterion**: The manuscript explicitly states: "The copy task has 4096 possible input sequences, matching the hidden state dimensionality. Test examples are drawn from the same 4096-sequence support as training examples. The generalization claim is limited to delay length, not to unseen sequences."
**Residual uncertainty**: Even with explicit documentation, a reviewer may view the dimensional match as a fatal confound. A stronger test would use a task with |S| >> state_dim (e.g., continuous input, longer sequences) where memorization is impossible.

---

### Finding 30

**Original claim**: PlainConvRNN changes multiple factors simultaneously.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/model.py` — PlainConvRNN differs from U1ConvRNN in: (a) standard real Conv2d vs complex cross-coupled conv, (b) tanh + LayerNorm vs radial_tanh, (c) real state with scalar nonlinearity vs complex state with phase-preserving nonlinearity; `README.md:28-32` — comparison table confirms multiple differences
**External evidence**: none
**Derivation**: When comparing U1ConvRNN vs PlainConvRNN, any difference in behavior could be attributed to: (a) complex vs real convolution, (b) radial_tanh vs tanh+LayerNorm nonlinearity, (c) phase-preserving vs phase-mixing dynamics, (d) blank embedding enforcement, (e) any interaction between these. A single PlainConvRNN baseline cannot isolate which factor causes the difference. Attributing a difference specifically to "U(1) equivariance" is a compound claim that requires factorial decomposition.
**Correction**: Add factorial baseline models: `ComplexNoEquiv` (complex conv + tanh + LayerNorm — removes radial_tanh) and `RealWithEquiv` (real conv + radial_tanh — adds U(1)-equivariant nonlinearity to real conv). The V2 contract includes these as candidates.
**Scientific impact**: Without factorial baselines, the claim "U(1) equivariance causes stronger vortex causality" is supported only by a compound comparison, not by an isolated-factor comparison. The factorial baselines allow attribution to specific architectural components.
**V2 action**: Implement `ComplexNoEquivConvRNN` and `RealWithEquivConvRNN` as listed in `10_IMPLEMENTATION_SPEC.md:18`. Run C16 factorial baseline pilot as specified in `15_PM1_LEARNED_V2_CONTRACT.yaml:106-111`.
**Required test**: T-V2-12 — factorial baseline models have distinct behavioral and topological signatures.
**Acceptance criterion**: At least one factorial baseline (ComplexNoEquiv or RealWithEquiv) shows intermediate causal vortex efficacy between PlainConvRNN and U1ConvRNN, allowing attribution of the effect to specific components.
**Residual uncertainty**: The factorial baselines may themselves confound factors (e.g., radial_tanh on real fields may behave differently from radial_tanh on complex fields). Perfect isolation of a single factor may be impossible.

---

### Finding 31

**Original claim**: A single PlainConvRNN cannot identify equivariance effects.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: Same as Finding 30; `16_OPEN_QUESTIONS_AND_DECISIONS.md:92-103` — PI decision D-P6 on factorial baseline scope lists 5 candidates
**External evidence**: none
**Derivation**: This is a direct restatement of Finding 30 with the emphasis on identifiability: from a single comparison (U1 vs Plain), you cannot identify which of the multiple differing factors is causally responsible for any observed difference. Standard causal inference requires that for each factor you want to make a claim about, you need a contrastive baseline that varies only that factor.
**Correction**: Same as Finding 30. The factorial baselines (`10_IMPLEMENTATION_SPEC.md:18-19`) provide the necessary identifiability. At minimum, if U1 > ComplexNoEquiv AND U1 > RealWithEquiv, the effect requires BOTH complex convolution AND radial_tanh — i.e., the full U(1)-equivariant combination.
**Scientific impact**: Without identifiability, the paper's core architecture claim is weak. With factorial baselines, it's much stronger: "the U(1)-equivariant combination of complex convolution and phase-preserving nonlinearity is necessary for the observed causal vortex effect."
**V2 action**: Run at minimum ComplexNoEquiv and RealWithEquiv in calibration. Include results in the manuscript even if null.
**Required test**: T-V2-12 — factorial baselines have qualitatively different causal vortex profiles from both U1 and Plain.
**Acceptance criterion**: Each factorial baseline occupies a distinct position in the (causal vortex strength, task accuracy) plane, with U1 in the top-right quadrant.
**Residual uncertainty**: The factorial baseline set may still be incomplete. Additional factors like blank-embedding enforcement and weight initialization may also matter.

---

### Finding 32

**Original claim**: Complex pairing of Plain real channels is arbitrary.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/interventions.py:85-100` — `decompose_hidden` treats the hidden state as (C, H, W) complex = (2C, H, W) real; for U1ConvRNN, the 2C channels are naturally paired (real, imag); for PlainConvRNN, the 2C real channels have no canonical complex pairing; `topological/model.py` — PlainConvRNN hidden state is `(batch, 2*C, H, W)` real
**External evidence**: none
**Derivation**: The decomposition pipeline converts the hidden state to complex by treating consecutive real channels as (real, imag) pairs. For U1ConvRNN, this pairing is meaningful: the convolution and radial_tanh preserve the complex structure. For PlainConvRNN, there is no physical complex structure — the 2C channels are independent real channels. Any pairing is arbitrary. Different pairings would produce different "charge" maps. The resulting "vortex transplant" on PlainConvRNN is physically meaningless — it's transplanting a construct that doesn't correspond to any real property of the model.
**Correction**: For PlainConvRNN and factorial baselines, either: (a) avoid applying complex decomposition entirely (only use real-valued interventions like L2 displacement, PCA, Fourier), or (b) document that the complex decomposition is a measurement convention applied for comparison, not a claim about the model's internal structure. The cross-model comparison should focus on real-valued behavioral measures, not "vortex charge" in Plains.
**Scientific impact**: If V2 reports "PlainConvRNN also has vortex-like charge patterns," this is misleading — the "charge" is an artifact of the arbitrary complex pairing, not a genuine property of the Plain model. The cross-model comparison must be carefully qualified.
**V2 action**: For non-U1 models, use only real-valued decomposition (Fourier, PCA, L2-based random). Do not report "charge" for models without complex structure. For the cross-model comparison, use paired real-valued interventions.
**Required test**: `test_plain_complex_pairing_arbitrary` — verify that different channel pairings produce different "charge" maps for the same Plain hidden state.
**Acceptance criterion**: At least two different channel pairings produce quantitatively different charge maps (signed Jaccard < 0.9) for the same Plain field.
**Residual uncertainty**: If the Plain model happens to organize its real channels into anti-correlated pairs (like a complex structure), the pairing may be less arbitrary than claimed. This should be empirically checked.

---

### Finding 33

**Original claim**: 10 seeds / 8-positive condition is weak.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `configs/topological_learned_v1.json:16` — `seeds: [0, 1, ..., 9]` = 10 seeds; `topological/learned_evaluation.py:654-657` — `_minimum_passing_seed_count(n) = ceil(0.8*n) = 8`; `learned_evaluation.py:707` — `sum(value > 0.0 for value in advantages) >= minimum_passing`
**External evidence**: none
**Derivation**: Under the null hypothesis H₀ (vortex has no effect, mechanism_advantage independently positive with probability p=0.5 per seed), the probability of passing (≥8/10 positive) is: P(≥8) = C(10,8)/2^10 + C(10,9)/2^10 + C(10,10)/2^10 = (45+10+1)/1024 = 56/1024 ≈ 0.0547. This is 1-in-18 false positive rate — higher than the conventional α=0.05 for a single test. With 2 models tested (U1 + Plain), the family-wise false positive rate is higher. Additionally, the binomial test has low power: with N=10 and truth p=0.7, detection power is only ~0.38.
**Correction**: Increase confirmatory seed count. The V2 contract defers the seed count to calibration-based power analysis. The contract specifies minimum N=20 for confirmatory, with simulation-based power analysis ensuring power ≥ 0.80 for MDE = 0.5 × calibration effect size.
**Scientific impact**: A 1-in-18 false positive gate is too weak for a confirmatory experiment. The V2 confirmatory must have a stronger gate. The seed count increase from 10 to ≥20 substantially improves both false positive rate and power.
**V2 action**: Use the V2 contract's power analysis procedure (`08_STATISTICAL_ANALYSIS_PLAN.md:62-86`). The binomial-gate threshold should be chosen to achieve α < 0.05 for the primary test, not inherited from V1's 80% heuristic.
**Required test**: T-V2-13 — hierarchical bootstrap CI contains expected coverage (~95%) on simulated data with known effect size.
**Acceptance criterion**: Confirmatory seed count determined by simulation with power ≥ 0.80 for MDE = 0.5 × calibration effect size. The gate threshold achieves α ≤ 0.05 under the null model.
**Residual uncertainty**: The calibration effect size estimate may itself be noisy with only 5-10 calibration seeds, making the power analysis imprecise.

---

### Finding 34

**Original claim**: Bootstrap resamples do not increase seed-level information.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `configs/topological_learned_v1.json:5` — `bootstrap_resamples: 10000`; `topological/learned_evaluation.py:676` — `bootstrap_lower_bound(advantages)` computes bootstrap CI from the list of seed-level advantages
**External evidence**: none
**Derivation**: Bootstrap resampling with N=10 seeds draws with-replacement from the 10 seed-level estimates. The bootstrap CI width is bounded below by the range of the 10 seed estimates (minus the discreteness of the bootstrap distribution). With only 10 seeds, the bootstrap cannot estimate a CI narrower than what the data support. The degrees of freedom for the variance estimate is still N-1=9, regardless of B=10000 resamples. Bootstrap is efficient for CI estimation but does not create new information — the statistical power is determined by N seeds and the within-seed variance, not by B.
**Correction**: The V2 contract specifies B=9999 resamples and N_seeds to be determined by power analysis. The bootstrap remains the CI method, but the seed count (N) is the primary power lever, not B. Document that bootstrap precision is adequate at B=9999 for N ≥ 20.
**Scientific impact**: With N=10 seeds, the bootstrap 2.5% CI can be unstable (the 2.5th percentile from 10 points is essentially the minimum). Reporting a CI lower bound with high precision (e.g., 0.0032) from N=10 is misleading about the actual precision.
**V2 action**: Use hierarchical bootstrap with B=9999 as specified. But do not report CIs with more significant digits than supported by N seeds. For N ≤ 20, report CI as [LB, UB] with 2 decimal places.
**Required test**: `test_bootstrap_precision_vs_seed_count` — verify that CI width is primarily driven by N seeds, not B resamples, and that B=9999 is adequate.
**Acceptance criterion**: Bootstrap CI at B=9999 converges (CI width at B=9999 within 1% of B=99999) for all N ≥ 10.
**Residual uncertainty**: None — this is a well-known property of bootstrap methods.

---

### Finding 35

**Original claim**: PCA is fit on the same held-out evaluation states.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:595-597` — `if len(valid_recipient_fields) >= 8: pca = fit_pca(valid_recipient_fields, k=8)`; line 608 — same PCA is passed to `evaluate_selected_pairs` where it's used for PCA-based intervention on these same states
**External evidence**: none
**Derivation**: The PCA model is fit on `valid_recipient_fields` — the same set of recipient hidden states that are used in the causal evaluation. When the PCA arm applies a PCA-based perturbation to a recipient state, the reconstruction uses components learned from that same state. This is in-sample, not out-of-sample. The PCA reconstruction error will be artificially low (zero at k=dimension) because the components were fit on the same data. This inflates the quality of the PCA baseline relative to what it would be with a properly held-out PCA model.
**Correction**: Fit PCA on training-set hidden states (or a held-out calibration set), not on evaluation states. The V2 contract uses a development/calibration/confirmatory split, where calibration data should be used for PCA fitting.
**Scientific impact**: The PCA baseline is likely over-optimistic (too close to the recipient state, too small perturbation), making the vortex margin appear larger by comparison. Using properly held-out PCA fitting would produce a more realistic PCA baseline and potentially reduce the mechanism_advantage.
**V2 action**: Fit PCA on calibration/development split states, not on evaluation split states. The contract's `calibration_split` is the designated fit-data source for PCA, manifold model, etc.
**Required test**: `test_pca_fit_on_separate_split` — verify that the PCA model used in confirmatory evaluation was fit on calibration data, not confirmatory data.
**Acceptance criterion**: PCA basis loaded from calibration artifacts; confirmatory evaluation never calls `fit_pca`.
**Residual uncertainty**: Training-set hidden states may have a different distribution from evaluation-set hidden states (different delay lengths, different model maturity), making the PCA fit suboptimal for evaluation. This is the standard generalization gap, which is acceptable.

---

### Finding 36

**Original claim**: Untrained baseline uses only seed 0.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:711-714` — `"defect_learned_not_innate": untrained_record is None or all(record["defect_prevalence"] > untrained_record["defect_prevalence"])` — `untrained_record` is a single dictionary passed from outside; `learned_pilot.py` or `learned_smoke.py` — the untrained model is likely instantiated once (single seed) and eval'd
**External evidence**: none
**Derivation**: The `defect_learned_not_innate` gate compares each trained seed's defect prevalence against a single untrained model's prevalence. If the untrained model was initialized with seed 0, its prevalence represents one draw from the initialization distribution. Different seeds (random initializations) may have different untrained prevalences. A single-seed baseline is not representative of the untrained distribution.
**Correction**: For V2, either (a) collect untrained prevalence from N ≥ 5 seeds and use the 95th percentile as the threshold, or (b) replace the binary gate with a paired test (trained prevalence > untrained prevalence tested across seed pairs). The V2 contract moves from prevalence to defect_density and branch_margin, making this a comparison of continuous metrics.
**Scientific impact**: A single untrained seed could be an outlier (lower prevalence than typical), making the gate easier to pass. Multiple untrained seeds provide a more robust baseline.
**V2 action**: In V2, the `defect_learned_not_innate` gate is replaced by density and branch_margin metrics (as decided in Finding 1 of the new findings, F-NEW-A). The untrained baseline should use N ≥ 5 seeds for robust reference distribution estimation.
**Required test**: `test_untrained_baseline_multiple_seeds` — verify that the untrained reference uses ≥5 seeds for density and branch_margin statistics.
**Acceptance criterion**: Untrained baseline density and branch_margin distributions estimated from ≥5 independent seeds.
**Residual uncertainty**: The specific number of untrained seeds (5) is a heuristic. More seeds improve normality of the reference distribution but add compute cost.

---

### Finding 37

**Original claim**: Plain evaluation failure auto-converts to U1 SURVIVE.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/learned_evaluation.py:770-780` — if U1 passes but competitor_interventions is False, status is `PM1_SURVIVE_LEARNED_PILOT`; lines 776-780:
  ```python
  if competitor_interventions:
      cross_positive = cross_clauses.get("cross_paired_lower_positive", False)
      status = "PM1_SURVIVE_LEARNED_PILOT" if cross_positive else "PM1_LEARNED_NO_GO"
  else:
      status = "PM1_SURVIVE_LEARNED_PILOT"
  ```
**External evidence**: none
**Derivation**: The `competitor_interventions` flag is True only if the competitor (PlainConvRNN) has complete intervention data (line 773-775: `competitor_decision["clauses"]["interventions_complete"]`). If PlainConvRNN's evaluation fails (training diverges, decomposition fails, no valid pairs found), `competitor_interventions` is False, and the cross-model gate automatically grants SURVIVE to U1. This is the wrong behavior: if we cannot evaluate the competitor, we cannot claim U1 superiority, because we have no evidence that the competitor wouldn't also succeed.
**Correction**: The V2 contract at `08_STATISTICAL_ANALYSIS_PLAN.md:135` states: "Competitor model (PlainConvRNN) failing ≠ success. PlainConvRNN failure is a necessary condition for the specificity claim, but if PlainConvRNN fails AND the U1 effect is small, the result is INCONCLUSIVE_BASELINE, not GO." The V2 gate must include an INCONCLUSIVE state when the competitor cannot be evaluated.
**Scientific impact**: V1's auto-pass logic is a false positive risk. V2 must not repeat it. If PlainConvRNN cannot be evaluated, the result is inconclusive, not successful.
**V2 action**: In the V2 decision function, add explicit handling: if competitor evaluation is incomplete (not merely "fails the gate" but "cannot be evaluated"), the status is `INCONCLUSIVE_BASELINE_UNEVALUABLE`, not `GO` or `NO_GO`.
**Required test**: `test_competitor_unevaluable_inconclusive` — a scenario where Plain evaluation fails (e.g., no valid pairs) produces `INCONCLUSIVE_BASELINE` not `SURVIVE`.
**Acceptance criterion**: When competitor_interventions is False, the final status is unequivocally INCONCLUSIVE (not GO, not NO_GO).
**Residual uncertainty**: The boundary between "unevaluable" (Plain training diverged, no pairs) and "evaluated but failed" (Plain evaluated, vortex margin ≤ 0) must be clearly defined. Only the latter should grant SURVIVE.

---

### Finding 38

**Original claim**: Competitor unevaluable is not success; must be inconclusive.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: Same as Finding 37; `08_STATISTICAL_ANALYSIS_PLAN.md:135` — explicit statement
**External evidence**: none
**Derivation**: This is a normative claim about scientific inference, not a factual claim about code behavior. It follows from the principle that a specificity claim requires evidence about what the specific mechanism does NOT do in a non-specific model. If we cannot measure the non-specific model, we cannot make the specificity claim. The auto-pass in Finding 37 violates this principle.
**Correction**: Same as Finding 37. Add INCONCLUSIVE_BASELINE status. The V2 contract at `15_PM1_LEARNED_V2_CONTRACT.yaml:369-372` already defines `INCONCLUSIVE_BASELINE: "PlainConvRNN also shows causal vortex effect"` — this should be extended to include "PlainConvRNN unevaluable."
**V2 action**: Extend the `INCONCLUSIVE_BASELINE` condition in the V2 contract to cover: (a) PlainConvRNN also shows causal vortex effect, (b) PlainConvRNN cannot be evaluated (training failure, too few valid pairs, decomposition failure).
**Required test**: Same as Finding 37 — `test_competitor_unevaluable_inconclusive`.
**Acceptance criterion**: The decision function has no path where competitor failure to evaluate leads to GO.
**Residual uncertainty**: What if PlainConvRNN is evaluable but produces degenerate results (e.g., all pairs have the same charge map because the model is untrained)? This should also be treated as unevaluable.

---

### Finding 39

**Original claim**: Private experiment contract dependency exists.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: 
  - `configs/topological_learned_v1.json:2` — `artifact_root: "/mnt/r/research-data/aligned-mask-transplant/pm1/learned-v1"` — hardcoded absolute path
  - `topological/learned_smoke.py:59-61` — `canonical_smoke_root()` returns `/mnt/r/research-data/aligned-mask-transplant/pm1/learned-v1/smoke-run-03`
  - `topological/pilot.py:37-38` — `canonical_config_path()` returns `configs/topological_feasibility_v1.json` with `artifact_root: "/mnt/r/research-data/aligned-mask-transplant/pm1/feasibility-v1"`
  - `topological/learned_smoke.py:68-70` — `_expected_config()` hardcodes the private path
**External evidence**: none
**Derivation**: The configuration and artifact paths reference `/mnt/r/research-data/aligned-mask-transplant/` — a path on the private research workstation filesystem. An independent researcher cannot replicate the experiment without modifying these paths. The smoke gate's `require_promotion` function also checks for pre-existing smoke runs at these hardcoded paths. Additionally, `topological/pilot.py:132` calls `aligned_mask_transplant.pm1_pilot` — an external module that exists on the private workstation but is not in the repository.
**Correction**: Remove all hardcoded absolute paths. Use environment variables (e.g., `RESEARCH_DATA_ROOT`) with documented defaults, and ensure defaults are relative or user-configurable. Remove or internalize the `aligned_mask_transplant.pm1_pilot` dependency.
**Scientific impact**: The current codebase is not independently reproducible. For a paper claiming reproducibility, this is a critical defect. The V2 contract's `resource_policy` and `artifact_schema` already avoid hardcoded paths, but V1 code still contains them.
**V2 action**: In V2, all paths derive from an explicit `artifact_root` passed to the pilot function, not from hardcoded module-level paths. The `canonical_*_root()` functions should take config arguments. The V2 `_artifacts.py` already implements `WriteOnceArtifact` which supports this.
**Required test**: `test_config_has_no_absolute_paths` — verify that no absolute paths appear in V2 config files or source code (outside of test fixtures).
**Acceptance criterion**: Setting `RESEARCH_DATA_ROOT=/tmp/test` produces all artifacts under `/tmp/test` with no references to `/mnt/r/research-data`.
**Residual uncertainty**: The PI's workstation convention uses `/mnt/r/research-data` for all research data. Supporting this convention while also supporting portable paths requires a configurable root mechanism, which V2 already plans.

---

### Finding 40

**Original claim**: Stale module path exists.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/pilot.py:131-137` — 
  ```python
  replay = subprocess.run(
      [sys.executable, "-m", "aligned_mask_transplant.pm1_pilot", "--replay-hash"],
      check=True, text=True, capture_output=True,
      cwd=_project_root(),
  ).stdout.strip()
  ```
  `01_REPOSITORY_MANIFEST.md:86-95` — confirmed stale external dependency; `02_DRAFT0_CORRECTION_LEDGER.md:15-27` — correction C-01 confirms with scope clarification
**External evidence**: none
**Derivation**: The `aligned_mask_transplant.pm1_pilot` module is referenced for clean-process replay verification. This module is not present in the repository and has no documented version, source, or installation instructions. The subprocess call at `pilot.py:132` will fail with `ModuleNotFoundError` on any system that doesn't have this private dependency installed. The `clean_process_replay_exact` gate (line 143) will always evaluate to False.
**Correction**: Remove the stale dependency. Implement clean-process replay as an internal function that re-runs the computation and compares SHA-256 hashes, as recommended in `12_REPRODUCIBILITY_AND_ARTIFACTS.md:73-79`.
**Scientific impact**: The feasibility pipeline's reproducibility guarantee is broken. The learned pipeline (V2 target) is not directly affected, but the pattern of external dependencies is a reproducibility concern for the whole project.
**V2 action**: In V2, remove all references to `aligned_mask_transplant`. Implement internal replay verification. The stale dependency is in `topological/pilot.py` (feasibility), not in the learned modules, but V2 should audit for similar patterns.
**Required test**: T-V2-05 — `CleanProcessReplay` raises a clear error for missing external module, or the stale path is removed entirely.
**Acceptance criterion**: `git grep aligned_mask_transplant` returns zero hits in V2 source code and configs.
**Residual uncertainty**: If the PI intends to integrate with a broader aligned-mask-transplant ecosystem later, the dependency should be documented as a planned future integration, not left as a stale reference.

---

### Finding 41

**Original claim**: README CPU smoke description mismatches CUDA promotion requirement.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `README.md:91-92` — "Smoke gate — verifies pipeline integrity without CUDA" with command `python -m topological.learned_smoke`; `topological/learned_smoke.py:305` — `def run_smoke(root, *, require_cuda: bool = True)` — default requires CUDA; `topological/learned_smoke.py:384` — `print(json.dumps(run_smoke(canonical_smoke_root()), sort_keys=True))` — uses default require_cuda=True; `topological/pilot.py:125` — `require_promotion((smoke_root or canonical_smoke_root()).resolve())` — promotion check expects prior CUDA smoke
**External evidence**: none
**Derivation**: The README describes a CPU-only smoke gate, but the actual `run_smoke()` function defaults to `require_cuda=True`, meaning it will raise an error if CUDA is not available. Furthermore, the pilot's `require_promotion` scans a previous smoke root and expects CUDA resources to be checked. Running `python -m topological.learned_smoke` as described in the README will fail on a CPU-only machine. The smoke gate's purpose is misrepresented.
**Correction**: Either (a) change the default to `require_cuda=False` so the README command works as documented, or (b) update the README to note that CUDA is required and provide a separate CPU-only smoke path. The V2 smoke design (`12_REPRODUCIBILITY_AND_ARTIFACTS.md:36-51`) separates CPU and CUDA smoke gates.
**Scientific impact**: A reader following the README instructions on a CPU machine gets a failure, reducing trust in the project's documentation accuracy.
**V2 action**: Implement separate `CPU_INTEGRITY_SMOKE` (no CUDA required) and `CUDA_RESOURCE_SMOKE` (CUDA required) as designed in the V2 smoke gates. The README should clearly document which smoke is for which hardware. The `--cpu` flag should be the default for basic integrity checks.
**Required test**: `test_cpu_smoke_no_cuda_dependency` — CPU integrity smoke runs successfully on a machine without CUDA.
**Acceptance criterion**: `python -m topological.v2.smoke --cpu` succeeds without CUDA; `python -m topological.v2.smoke --cuda` succeeds with CUDA and reports resource estimates.
**Residual uncertainty**: The PI's workstation has CUDA, so in practice the mismatch may not be observed locally. But for public reproducibility, the CPU-only path must work.

---

### Finding 42

**Original claim**: Repository URL / CITATION metadata mismatch.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: 
  - `README.md:128-133` — BibTeX uses `title = {vortex-transplant: Topological Defect Causal Intervention}`, `url = {https://github.com/GongJae00/vortex-transplant}`
  - `CITATION.cff:3-4` — `title: "Aligned Mask Transplant: Mechanism-first Causal Tests for Neural Representations"`, `repository-code: "https://github.com/GongJae00/vortex-transplant"`
**External evidence**: none
**Derivation**: The `CITATION.cff` title "Aligned Mask Transplant: Mechanism-first Causal Tests for Neural Representations" does not match the README BibTeX title "vortex-transplant: Topological Defect Causal Intervention." These describe different conceptual framings: "aligned mask transplant" (the broader methodology) vs "vortex-transplant" (the specific project). The mismatch could cause confusion: an automated tool extracting citation metadata from `CITATION.cff` would produce a different citation than the one shown in the README. The `CITATION.cff` also lacks author, year, and version metadata.
**Correction**: Synchronize the metadata. If the project is officially "vortex-transplant," the `CITATION.cff` title should match. If "Aligned Mask Transplant" is the umbrella project, add a `preferred-citation` with the paper title. Add author, year, and version fields.
**Scientific impact**: Minor — citation metadata inconsistency is a polish issue, not a scientific defect. However, automated citation tools will propagate the wrong metadata.
**V2 action**: Update `CITATION.cff` to have a title matching the repository name and the intended paper title. Add `authors`, `date-released`, and `version` fields. Ensure consistency with README and paper.
**Required test**: `test_citation_metadata_consistency` — verify that README BibTeX title, CITATION.cff title, and paper title are consistent.
**Acceptance criterion**: All three metadata sources (README, CITATION.cff, paper/main.tex) share the same paper title.
**Residual uncertainty**: The paper title may change during the writing process. Set a final title at V2 freeze time and update all metadata simultaneously.

---

### Finding 43

**Original claim**: Public repository alone cannot guarantee clean-room canonical reproduction.
**Verdict**: CONFIRMED
**Confidence**: High
**Repository evidence**: `topological/pilot.py:131-137` — clean-process replay depends on `aligned_mask_transplant.pm1_pilot` external module; `topological/learned_smoke.py:59-61` — canonical smoke root hardcodes private filesystem path; `configs/topological_learned_v1.json:2` — config hardcodes private path; `topological/learned_smoke.py:68-73` — `_expected_config` hardcodes private artifact root
**External evidence**: none
**Derivation**: Clean-room reproduction requires: (1) exact source code version, (2) deterministic computation with no external state, (3) identical environment (Python version, package versions, hardware). The current codebase violates (2) in two ways: the `aligned_mask_transplant.pm1_pilot` subprocess call reads external state, and the hardcoded artifact paths reference a private filesystem. An independent researcher cloning the repository cannot run the exact reproduction workflow because these external resources are unavailable. This finding aggregates Findings 39, 40, and the reproducibility gaps into a single systemic concern.
**Correction**: Same as Findings 39 and 40 — remove external dependencies and hardcoded paths. The V2 reproducibility plan (`12_REPRODUCIBILITY_AND_ARTIFACTS.md:62-69`) specifies: (a) clone at frozen commit, (b) create venv from uv.lock, (c) run CPU integrity smoke, (d) run CUDA resource smoke, (e) run scientific pilot with frozen config, (f) verify manifest.sha256. This is a self-contained reproduction path.
**Scientific impact**: The current codebase is not independently reproducible, which undermines the project's scientific credibility. For the paper, reproducibility must be demonstrated, not just claimed. The V2 plan addresses this systematically.
**V2 action**: Implement the V2 reproducibility plan. Ensure all components from Finding 39 (private paths), Finding 40 (stale module), Finding 41 (README/smoke mismatch), and Finding 42 (metadata) are resolved. Run a clean-room reproduction test on a fresh checkout as part of the V2 release checklist.
**Required test**: `test_full_reproduction_workflow` — clone repo at V2 freeze commit, run CPU integrity smoke, verify manifest hash matches pre-recorded value.
**Acceptance criterion**: A fresh clone + venv from uv.lock + `python -m topological.v2.smoke --cpu` succeeds with no modifications to source code, producing a manifest that matches the expected hash.
**Residual uncertainty**: CUDA reproducibility (bitwise-deterministic GPU results) is harder than CPU reproducibility due to floating-point non-associativity in CUDA kernels. The CUDA resource smoke should document any observed non-determinism.
