# DRAFT-0 Correction Ledger

DRAFT-0 = the agent response labeled "PM1-LEARNED-V2 MASTER PLAN" under audit.

## Summary: 4 Factual Errors, 5 Over-specifications, 8 Omissions

| Outcome | Count |
|---------|------:|
| DRAFT-0 claim REJECTED (factually wrong) | 4 |
| DRAFT-0 over-specified (unbacked numbers) | 5 |
| DRAFT-0 incomplete (needs expansion) | 8 |

---

## C-01: Stale Module Path

- **DRAFT-0 statement**: "`aligned_mask_transplant` or `pm1_pilot` path does not exist — stale-module finding REJECTED"
- **Correct verdict**: **CONFIRMED** (stale path exists), scope correction required
- **Repository evidence**: `topological/pilot.py:132`:
  ```python
  [sys.executable, "-m", "aligned_mask_transplant.pm1_pilot", "--replay-hash"],
  ```
  Present in blob `41f70f7`, confirmed by `git grep` at HEAD.
- **Reasoning**: DRAFT-0 searched only the `topological/learned_*` modules and concluded "not present." The stale call is in `topological/pilot.py` (feasibility path), not in the learned path. The request was a *full repository* audit.
- **Impact on master plan**: The `clean_process_replay_exact` clause in feasibility pilot is non-functional. This is a reproducibility defect. Does **not** affect the learned pipeline directly, but does affect the clean-room replay guarantee.
- **Required change**: Add to D4 additional findings. Either remove the stale call or document the external dependency. The gate `clean_process_replay_exact` currently evaluates to `False`.

---

## C-02: Minimum Magnitude

- **DRAFT-0 statement**: "minimum magnitude is not recorded"
- **Correct verdict**: **REJECTED** (it IS computed), but **not used in gates**
- **Repository evidence**:
  1. `HiddenComponents.minimum_magnitude: float` — `interventions.py:31`
  2. Computed at `interventions.py:99`: `float(np.min(np.abs(field)))`
  3. **NOT** referenced in `learned_evaluation.py` (grep: zero hits)
  4. **NOT** used in any decision gate clause
- **Three-way distinction**:
  - ✓ Computed (in `decompose_hidden` → `HiddenComponents`)
  - ✗ Exposed in seed-level reports (not found in learned evaluation reports)
  - ✗ Used in decision gates (no clause references it)
- **Required change**: A `branch_margin` or `magnitude_margin` field does not exist. A `minimum_magnitude` field exists but is unused in the decision pipeline. V2 should either add `minimum_magnitude` to seed/pair reports and use it in a validity gate, or remove it from `HiddenComponents`.

---

## C-03: CPU/CUDA Equivariance

- **DRAFT-0 statement**: "equivariance check is only performed on CPU"
- **Correct verdict**: **REJECTED** (CUDA path exists and executes)
- **Repository evidence**:
  1. `_exercise_device(device, model_type)` — `learned_smoke.py:174` — takes `device` parameter, runs training + evaluation on that device
  2. `_exercise_all(device)` — `learned_smoke.py:293` — calls `_exercise_device` for u1 and plain on the passed device
  3. `run_smoke` logic — `learned_smoke.py:312-317`:
     - CPU: `_exercise_all(torch.device("cpu"))`
     - CUDA (if available): `_exercise_all(torch.device("cuda"))`
- **Actual legitimate concerns** (not the DRAFT-0 claim):
  - Uses a single phase value (not a distribution)
  - Few probe states (not exhaustive)
  - Does not separate blank/input/readout equivariance
  - Does not report maximum equivariance error across hidden state distribution
- **Required change**: Replace "CPU-only" finding with a refined finding about equivariance robustness test scope. File as `F-NEW-5b`.

---

## C-04: Training Curves

- **DRAFT-0 statement**: "training curves are not stored in artifacts"
- **Correct verdict**: **REJECTED** (they ARE stored)
- **Repository evidence**:
  1. `_training_record(result, elapsed_seconds)` — `learned_pilot.py:147-159`:
     ```python
     "history": [asdict(record) for record in result.history],
     "training_log": [asdict(snapshot) for snapshot in result.training_log],
     ```
  2. Written per seed: `learned_pilot.py:236` — `writer.write_json(f"{prefix}/training.json", training_record)`
- **Actual gaps** (not "not stored"):
  - No aggregated cross-seed training curve figure generator
  - No explicit index in decision artifact pointing to per-seed training files
  - No automated training convergence audit
- **Required change**: REPLACE "not stored" with: "Training records are stored per seed. V2 should add cross-seed aggregation plot + explicit artifact index."

---

## C-05: Train/Test Independence

- **DRAFT-0 statement**: "SHA-256 namespace collision is a contamination risk"
- **Correct verdict**: **MISDIAGNOSED** — the real issue is task-support overlap
- **Reasoning**:
  - Copy task vocabulary = 8 tokens, length = 4 → total possible sequences = \(8^4 = 4,096\)
  - Training: \(30,000 \times 64 = 1,920,000\) examples per seed
  - Average repeats per sequence: \(1,920,000 / 4,096 \approx 468.75\)
  - SHA-256 namespace separation prevents RNG-state collision, but the test sequences are **not unseen** — they're drawn from the same finite support
- **What "held-out" actually means**: Unseen **delay** (delay=64 vs training range 16-32), not unseen sequences. The generalization claim is "longer delay generalization," not "new sequence generalization."
- **Required change**: Clarify that `train_hash != test_hash` is a provenance sanity check, not a scientific contamination guard. The claim must be scoped to "delay OOD generalization" not "unseen sequence generalization."

---

## C-06: Defect Prevalence

- **DRAFT-0 statement**: "wrapped identity guarantees nonzero charge everywhere" / "defect prevalence is conservatively 1"
- **Correct verdict**: **OVER-STATED** — integer residual ≠ nonzero charge at every plaquette
- **Reasoning**:
  - Integer quantization of plaquette curl sum is a structural property for compact fields — it does not force each individual plaquette to have nonzero charge
  - A perfectly smooth pure-phase field has zero charge everywhere
  - Random phase fields WILL have high prevalence, but it's not "guaranteed" in the mathematical sense
  - The current prevalence definition (`nonzero_defect = any channel has both + and -`) makes state-level saturation at 1.0 easy for rough untrained fields
- **More precise statement**: "For untrained random-like phase fields, state-level defect prevalence is expected to be near 1.0 due to the permissive `any(channel)` definition. The key quantity is `defect_density` (per-plaquette) and `branch_margin` (stability), not just `nonzero_defect`."
- **Required change**: Separate channel-level prevalence from state-level. Consider density-based metrics.

---

## C-07: Untrained Gate Saturation

- **DRAFT-0 statement**: DRAFT-0 did not identify this issue
- **Correct verdict**: **P0 FINDING** — `defect_learned_not_innate` may be structurally impossible
- **Repository evidence**: `learned_evaluation.py:711-712`:
  ```python
  "defect_learned_not_innate": untrained_record is None or all(
      record.get("defect_prevalence", 0.0) > untrained_record.get("defect_prevalence", 0.0)
      for record in seed_records
  )
  ```
- **Problem**: If untrained `nonzero_defect` prevalence = 1.0 (likely for a random-like 16x16 field with 8 channels), then ALL trained seeds need prevalence > 1.0, which is mathematically impossible.
- **Required action**: Run CPU-only untrained diagnostic BEFORE any training. If prevalence = 1.0, the gate definition must change (e.g., use `defect_density > untrained_density`, or per-channel signed Jaccard stability, or branch margin improvement).
- **This is the single most urgent diagnostic to run.**

---

## C-08: Branch Margin Threshold

- **DRAFT-0 statement**: DRAFT-0 proposed `branch_margin > 0.1 rad` as a threshold
- **Correct verdict**: **UNFROZEN** — no empirical basis
- **Issues**:
  - `branch_margin` (global minimum of \(\pi - |\Delta\theta_e|\) over all edges) is sensitive to grid size — larger grids → smaller minimum via extreme value statistics
  - One outlier edge can invalidate an entire state
  - Different metrics needed: global minimum, defect-local minimum, 1%-quantile, median, charge-flip radius
- **Required change**: Define a `branch_stability` protocol using multiple quantiles + defect-local statistics. Threshold determined via calibration, not assertion.

---

## C-09: Null Ensemble Size

- **DRAFT-0 statement**: DRAFT-0 proposed `J=20` null draws for 95th percentile
- **Correct verdict**: **UNFROZEN** — insufficient for stable tail quantile
- **Reasoning**:
  - \(J=20\): the 95th percentile is the 19th order statistic out of 20 → effectively the sample max → high variance
  - For a stable q95 estimate: need \(J \geq 99\) (then order statistic ~94th)
  - For Monte Carlo p-value resolution: \(p_{\min} = 1/(B+1)\) — needs \(B \geq 199\) for p < 0.005
- **Required change**: Split null families (not pooled). Use \(B \geq 199\) per family. Use order-statistic confidence intervals.

---

## C-10: Confirmatory Sample Size

- **DRAFT-0 statement**: "30 seeds gives ~80% power for \(d=0.5\)"
- **Correct verdict**: **UNFROZEN** — no calibration data to base this on
- **Required before freezing**:
  1. Calibration seed variance estimate
  2. Within-seed intra-class correlation
  3. Representative variance component
  4. Null-draw variance
  5. Simulation-based power analysis with planned statistic
  6. Attrition rate (invalid pairs, manifold failures)
- **Required change**: Defer to calibration phase. Provide a simulation template, not a frozen number.

---

## C-11: Risk Probabilities

- **DRAFT-0 statement**: Used specific percentages (25%, 30%) for various failure modes
- **Correct verdict**: **UNSUPPORTED** — no empirical or elicited basis
- **Required change**: Replace with ordinal categories: `Unknown / Low / Medium / High` with explicit `Evidence basis:` field for each.

---

## C-12: GPU-Hour Estimates

- **DRAFT-0 statement**: "10–20 GPU-hours" for Phase 3, "100–200" for confirmatory
- **Correct verdict**: **UNSUPPORTED** — no throughput model or benchmark
- **Required model**:
  ```
  training_time = updates × recurrent_steps × batch_size / throughput
  evaluation_time = pairs × interventions × null_draws × continuation_length / throughput
  ```
- **Required change**: Build throughput model from smoke benchmarks. Provide range estimates with confidence intervals, not single-point claims.

---

## C-13: Venue Claims

- **DRAFT-0 statement**: Referenced "NeurIPS requires 3 tasks" as a rule
- **Correct verdict**: **HEURISTIC, NOT OFFICIAL** — venue-specific requirements should cite actual CFP language, not informal community rules
- **Required change**: For each venue: cite official CFP/review criteria, distinguish published deadlines from projected future deadlines, note heuristic recommendations separately from policy.

---

## C-14: Multichannel Order Parameter

- **DRAFT-0 statement**: Proposed \(\psi(x) = w^\dagger z(x)\) as a solution for \(C>1\)
- **Correct verdict**: **PREMISE INCOMPLETE** — projection has fundamental issues
- **Issues**:
  - \(\pi_1(\mathbb{C}^C \setminus \{0\}) = 0\) for \(C > 1\) — full vector field has no topological charge
  - Projection creates artificial zeros and basis-dependence
  - Need to establish that learned dynamics constrain state to a \(\approx S^1 \times \mathbb{R}_+\) submanifold (phase-locked across channels)
  - The phase-locked submanifold must be empirically demonstrated, not assumed
- **Required change**: Derive vacuum manifold and isotropy subgroup first. C=1 is a diagnostic necessity, not just convenience.

---

## C-15: Necessity Intervention

- **DRAFT-0 statement**: Suggested replacing the entire vortex field with \(v_0\) as a necessity test
- **Correct verdict**: **TOO COARSE** — global vortex removal ≠ local defect removal
- **Required design**:
  1. Target a single defect pair (not entire field)
  2. Matched sham surgery (same energy/spectrum/displacement, charge-preserving)
  3. Minimal displacement (local surgery, not global phase rewrite)
  4. Harmonic sector preservation
  5. Relaxation post-surgery (allow natural dynamics to stabilize)
  6. Verification: charge removal success + non-target component preservation
- **Required change**: Design `minimal_annihilation(target_pair)` intervention.

---

## C-16: Pooled Null q95

- **DRAFT-0 statement**: All null families pooled into one q95 computation
- **Correct verdict**: **INVALID POOLING** — different families test different hypotheses
- **Family separation required**:
  | Family | Hypothesis tested |
  |--------|-------------------|
  | random_direction | Any perturbation above noise? |
  | fourier_low/fourier_high | Frequency-specific? |
  | PCA | Variance-captured? |
  | smooth/magnitude | Component-specific? |
  | harmonic swap | Competing topological variable? |
  | same-charge representative | Representative-invariant? |
  | natural neighbor | On-manifold? |
- **Required change**: Each family gets its own hypothesis test. Pooled q95 is a secondary sensitivity analysis.

---

## C-17: Deliverable Completeness

- **DRAFT-0 statement**: Claimed all 14 deliverables were provided
- **Correct verdict**: **INCOMPLETE** — most deliverables were summarized in 1-2 paragraphs, not produced as full artifacts
- **Missing**:
  - Full repository manifest (D1) — not provided
  - Individual finding adjudication (D2) — aggregate counts only
  - File/function/test spec (D9/D10) — missing
  - API/dataclass specification (D10) — missing
  - Frozen V2 contract (D14) — key fields missing, not YAML format
- **Required change**: This correction ledger + the 17 artifact files constitute the corrected deliverables.
