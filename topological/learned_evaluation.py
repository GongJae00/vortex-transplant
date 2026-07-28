"""Frozen topology, donor selection, and causal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .interventions import (
    HiddenComponents,
    PCADecomposition,
    aggregate_gradient_energy,
    aggregate_spectrum_error,
    component_intervention,
    complex_to_hidden,
    decompose_hidden,
    fit_pca,
    fourier_field_intervention,
    hidden_to_complex,
    matched_global_phase,
    matched_zero_charge_phase,
    pca_field_intervention,
    random_direction_intervention,
    state_displacement,
)
from .task import (
    BLANK_TOKEN,
    continue_copy,
    donor_sequences,
    generate_copy_batch,
    run_copy,
    write_copy,
)
from .topology import extract_charge


TEST_DELAY = 64
TEST_EXAMPLES = 128
DONORS_PER_RECIPIENT = 8
PAIR_MINIMUM = 64
TOPOLOGY_TOLERANCE = 1e-5
MAGNITUDE_EPSILON = 1e-6
NUISANCE_TOLERANCE = 0.10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_722


@dataclass(frozen=True)
class StateTopology:
    signed_tuples: frozenset[tuple[int, int, int, int]]
    valid_channels: tuple[bool, ...]
    nonzero_defect: bool
    maximum_integer_residual: float
    net_charge_valid: bool


@dataclass(frozen=True)
class PairCandidate:
    donor_index: int
    recipient_hidden: torch.Tensor
    donor_hidden: torch.Tensor
    recipient: HiddenComponents
    donor: HiddenComponents
    recipient_field: np.ndarray
    donor_field: np.ndarray
    signed_count_difference: int
    relative_energy_difference: float
    spectrum_difference: float
    state_displacement: float
    displacement_median_difference: float
    admissible_catalog_count: int

    @property
    def score(self) -> tuple[float, ...]:
        return (
            float(self.signed_count_difference),
            self.relative_energy_difference,
            self.spectrum_difference,
            self.displacement_median_difference,
            float(self.donor_index),
        )


def analyze_topology(hidden: torch.Tensor) -> StateTopology:
    field = hidden_to_complex(hidden)
    signed: set[tuple[int, int, int, int]] = set()
    valid_channels: list[bool] = []
    residuals: list[float] = []
    net_charge_valid = True
    nonzero_defect = False
    for channel_index, channel in enumerate(field):
        valid = bool(np.min(np.abs(channel)) > MAGNITUDE_EPSILON)
        valid_channels.append(valid)
        if not valid:
            continue
        charge = extract_charge(channel, tolerance=TOPOLOGY_TOLERANCE)
        residuals.append(charge.residual_max)
        net_charge_valid = net_charge_valid and charge.net_charge == 0
        positive = np.argwhere(charge.charge > 0)
        negative = np.argwhere(charge.charge < 0)
        nonzero_defect = nonzero_defect or (len(positive) > 0 and len(negative) > 0)
        for x, y in positive:
            signed.add((channel_index, 1, int(x), int(y)))
        for x, y in negative:
            signed.add((channel_index, -1, int(x), int(y)))
    return StateTopology(
        signed_tuples=frozenset(signed),
        valid_channels=tuple(valid_channels),
        nonzero_defect=nonzero_defect,
        maximum_integer_residual=max(residuals, default=0.0),
        net_charge_valid=net_charge_valid,
    )


def signed_jaccard(first: StateTopology, second: StateTopology) -> float:
    union = first.signed_tuples | second.signed_tuples
    if not union:
        return 1.0
    return len(first.signed_tuples & second.signed_tuples) / len(union)


def _signed_count_difference(first: np.ndarray, second: np.ndarray) -> int:
    difference = 0
    for left, right in zip(first, second, strict=True):
        difference += abs(int(np.count_nonzero(left > 0)) - int(np.count_nonzero(right > 0)))
        difference += abs(int(np.count_nonzero(left < 0)) - int(np.count_nonzero(right < 0)))
    return difference


def select_donor_pair(
    recipient_hidden: torch.Tensor,
    donor_hidden: Sequence[torch.Tensor],
    *,
    precomputed_recipient: HiddenComponents | None = None,
    donor_decompose_failures: list[int] | None = None,
) -> PairCandidate | None:
    """Select a donor from hidden-state geometry only; labels are not accepted.

    Parameters
    ----------
    recipient_hidden, donor_hidden:
        Hidden states for the recipient and its eight candidate donors.
    precomputed_recipient:
        Optional pre-decomposed recipient to avoid redundant computation.
    donor_decouple_failures:
        Optional list to append indices of donor candidates that fail
        decomposition (enables diagnostic counting without pre-checks).
    """

    if len(donor_hidden) != DONORS_PER_RECIPIENT:
        raise ValueError("donor selection requires exactly eight hidden candidates")
    if precomputed_recipient is not None:
        recipient = precomputed_recipient
    else:
        try:
            recipient = decompose_hidden(
                recipient_hidden,
                magnitude_epsilon=MAGNITUDE_EPSILON,
                tolerance=TOPOLOGY_TOLERANCE,
            )
        except ValueError:
            return None
    recipient_field = hidden_to_complex(recipient_hidden)
    recipient_energy = aggregate_gradient_energy(recipient_field)
    if recipient_energy <= 1e-12:
        return None
    provisional: list[dict[str, Any]] = []
    for donor_index, hidden in enumerate(donor_hidden):
        try:
            donor = decompose_hidden(
                hidden,
                magnitude_epsilon=MAGNITUDE_EPSILON,
                tolerance=TOPOLOGY_TOLERANCE,
            )
        except ValueError:
            if donor_decompose_failures is not None:
                donor_decompose_failures.append(donor_index)
            continue
        if np.array_equal(recipient.charge, donor.charge):
            continue
        donor_field = hidden_to_complex(hidden)
        displacement = state_displacement(recipient_field, donor_field)
        provisional.append(
            {
                "donor_index": donor_index,
                "donor_hidden": hidden.detach().cpu().clone(),
                "donor": donor,
                "donor_field": donor_field,
                "signed_count_difference": _signed_count_difference(recipient.charge, donor.charge),
                "relative_energy_difference": abs(
                    aggregate_gradient_energy(donor_field) - recipient_energy
                )
                / recipient_energy,
                "spectrum_difference": aggregate_spectrum_error(recipient_field, donor_field),
                "state_displacement": displacement,
            }
        )
    if not provisional:
        return None
    median_displacement = float(np.median([item["state_displacement"] for item in provisional]))
    candidates = [
        PairCandidate(
            donor_index=int(item["donor_index"]),
            recipient_hidden=recipient_hidden.detach().cpu().clone(),
            donor_hidden=item["donor_hidden"],
            recipient=recipient,
            donor=item["donor"],
            recipient_field=recipient_field,
            donor_field=item["donor_field"],
            signed_count_difference=int(item["signed_count_difference"]),
            relative_energy_difference=float(item["relative_energy_difference"]),
            spectrum_difference=float(item["spectrum_difference"]),
            state_displacement=float(item["state_displacement"]),
            displacement_median_difference=abs(
                float(item["state_displacement"]) - median_displacement
            ),
            admissible_catalog_count=len(provisional),
        )
        for item in provisional
    ]
    return min(candidates, key=lambda candidate: candidate.score)


@torch.no_grad()
def post_write_in_chunks(
    model: nn.Module,
    symbols: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("post-write chunk size must be positive")
    copy_length = int(symbols.shape[1])
    outputs = []
    for start in range(0, len(symbols), chunk_size):
        outputs.append(
            write_copy(model, symbols[start : start + chunk_size], copy_length=copy_length)
            .detach()
            .cpu()
        )
    return torch.cat(outputs, dim=0)


def log_likelihood_margin(
    logits: torch.Tensor,
    donor_symbols: torch.Tensor,
    recipient_symbols: torch.Tensor,
) -> torch.Tensor:
    expected_length = int(donor_symbols.shape[1])
    if logits.ndim != 3 or logits.shape[1] != expected_length:
        raise ValueError(
            f"margin requires logits with {expected_length} output positions"
        )
    log_probability = torch.log_softmax(logits, dim=-1)
    donor = log_probability.gather(-1, donor_symbols.unsqueeze(-1)).squeeze(-1)
    recipient = log_probability.gather(-1, recipient_symbols.unsqueeze(-1)).squeeze(-1)
    return (donor - recipient).mean(dim=1)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _normalized_field_residual(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected.ravel())), 1e-12)
    return float(np.linalg.norm((actual - expected).ravel()) / denominator)


def _relative_energy_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    energy = aggregate_gradient_energy(reference)
    if energy <= 1e-12:
        raise ValueError("learned energy reference is degenerate")
    return abs(aggregate_gradient_energy(candidate) - energy) / energy


@torch.no_grad()
def evaluate_selected_pairs(
    model: nn.Module,
    pairs: Sequence[PairCandidate],
    recipient_symbols: torch.Tensor,
    donor_symbols: torch.Tensor,
    recipient_indices: Sequence[int],
    *,
    delay: int = TEST_DELAY,
    task_type: str = "copy",
    pca: PCADecomposition | None = None,
) -> dict[str, Any]:
    count = len(pairs)
    if count == 0 or recipient_symbols.shape != donor_symbols.shape or len(recipient_symbols) != count:
        raise ValueError("selected-pair evaluation requires aligned nonempty inputs")
    if len(recipient_indices) != count:
        raise ValueError("selected-pair indices are not aligned")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    fields_by_arm: dict[str, list[np.ndarray]] = {
        arm: []
        for arm in (
            "natural_recipient",
            "natural_donor",
            "vortex",
            "smooth",
            "magnitude",
            "global_phase",
            "zero_phase",
            "whole_phase",
            "whole_state",
            "fourier_low",
            "fourier_high",
            "pca",
            "random_direction",
        )
    }
    static_records: list[dict[str, Any]] = []
    for pair, recipient_index in zip(pairs, recipient_indices, strict=True):
        vortex = component_intervention(pair.recipient, pair.donor, "vortex")
        smooth = component_intervention(pair.recipient, pair.donor, "smooth")
        magnitude = component_intervention(pair.recipient, pair.donor, "magnitude")
        whole_phase = component_intervention(pair.recipient, pair.donor, "whole_phase")
        whole_state = component_intervention(pair.recipient, pair.donor, "whole_state")
        target_displacement = state_displacement(pair.recipient_field, vortex)
        global_control = matched_global_phase(pair.recipient_field, target_displacement)
        zero_control = matched_zero_charge_phase(
            pair.recipient_field,
            target_displacement,
            control_index=recipient_index * DONORS_PER_RECIPIENT + pair.donor_index,
        )
        fourier_low = fourier_field_intervention(
            pair.recipient_field, pair.donor_field, "fourier_low"
        )
        fourier_high = fourier_field_intervention(
            pair.recipient_field, pair.donor_field, "fourier_high"
        )
        pca_field = None
        random_direction_field = None
        if pca is not None:
            pca_field = pca_field_intervention(pair.recipient_field, pair.donor_field, pca)
        random_direction_field = random_direction_intervention(
            pair.recipient_field, pair.donor_field,
            seed=recipient_index * DONORS_PER_RECIPIENT + pair.donor_index,
            target_norm=target_displacement,
        )
        fields = {
            "natural_recipient": pair.recipient_field,
            "natural_donor": pair.donor_field,
            "vortex": vortex,
            "smooth": smooth,
            "magnitude": magnitude,
            "global_phase": global_control.field,
            "zero_phase": zero_control.field,
            "whole_phase": whole_phase,
            "whole_state": whole_state,
            "fourier_low": fourier_low,
            "fourier_high": fourier_high,
            "pca": pca_field if pca_field is not None else pair.recipient_field,
            "random_direction": random_direction_field if random_direction_field is not None else pair.recipient_field,
        }
        for arm, field in fields.items():
            fields_by_arm[arm].append(field)

        pca_displacement = state_displacement(
            pair.recipient_field, pca_field
        ) if pca_field is not None else 0.0
        random_direction_displacement = state_displacement(
            pair.recipient_field, random_direction_field
        ) if random_direction_field is not None else 0.0

        transplanted = decompose_hidden(
            complex_to_hidden(vortex),
            magnitude_epsilon=MAGNITUDE_EPSILON,
            tolerance=TOPOLOGY_TOLERANCE,
        )
        retained_magnitude_error = float(
            np.max(np.abs(transplanted.magnitude - pair.recipient.magnitude))
        )
        retained_smooth_error = float(
            np.max(np.abs(transplanted.smooth - pair.recipient.smooth))
        )
        static_records.append(
            {
                "recipient_index": int(recipient_index),
                "donor_index": pair.donor_index,
                "admissible_catalog_count": pair.admissible_catalog_count,
                "donor_score": list(pair.score),
                "target_state_displacement": target_displacement,
                "global_displacement_error": global_control.displacement_error,
                "zero_phase_displacement_error": zero_control.displacement_error,
                "vortex_relative_energy_error": _relative_energy_error(
                    pair.recipient_field, vortex
                ),
                "vortex_spectrum_error": aggregate_spectrum_error(
                    pair.recipient_field, vortex
                ),
                "unintended_charge_count": int(
                    np.count_nonzero(transplanted.charge != pair.donor.charge)
                ),
                "retained_magnitude_error": retained_magnitude_error,
                "retained_smooth_error": retained_smooth_error,
            "whole_state_roundtrip_error": float(
                np.max(np.abs(whole_state - pair.donor_field))
            ),
            "pca_displacement": float(pca_displacement),
            "random_direction_displacement": float(random_direction_displacement),
        }
        )

    recipient_targets = recipient_symbols.to(device=device)
    donor_targets = donor_symbols.to(device=device)
    if task_type == "reverse":
        recipient_targets = torch.flip(recipient_targets, dims=[1])
        donor_targets = torch.flip(donor_targets, dims=[1])
    copy_length = int(donor_symbols.shape[1])
    arm_margins: dict[str, np.ndarray] = {}
    state_batches: dict[str, torch.Tensor] = {}
    for arm, fields in fields_by_arm.items():
        hidden = torch.stack(
            [complex_to_hidden(field, device=device, dtype=dtype) for field in fields]
        )
        state_batches[arm] = hidden
        logits, _ = continue_copy(model, hidden, delay, copy_length=copy_length)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"{arm} continuation produced non-finite logits")
        arm_margins[arm] = (
            log_likelihood_margin(logits, donor_targets, recipient_targets)
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )

    recipient_batch = torch.stack([pair.recipient_hidden for pair in pairs]).to(
        device=device, dtype=dtype
    )
    donor_batch = torch.stack([pair.donor_hidden for pair in pairs]).to(device=device, dtype=dtype)
    blank = torch.full((count,), BLANK_TOKEN, dtype=torch.long, device=device)
    recipient_next = model.step(blank, recipient_batch)
    donor_next = model.step(blank, donor_batch)
    next_recipient_parts = [decompose_hidden(state) for state in recipient_next]
    next_donor_parts = [decompose_hidden(state) for state in donor_next]
    residuals: dict[str, list[float]] = {"vortex": [], "smooth": [], "magnitude": []}
    for arm in residuals:
        actual_next = model.step(blank, state_batches[arm])
        expected_fields = [
            component_intervention(recipient, donor, arm)
            for recipient, donor in zip(next_recipient_parts, next_donor_parts, strict=True)
        ]
        actual_fields = [hidden_to_complex(state) for state in actual_next]
        residuals[arm] = [
            _normalized_field_residual(actual, expected)
            for actual, expected in zip(actual_fields, expected_fields, strict=True)
        ]

    pair_records = []
    for index, static in enumerate(static_records):
        margins = {arm: float(values[index]) for arm, values in arm_margins.items()}
        vortex_residual = residuals["vortex"][index]
        reciprocal_best = min(residuals["smooth"][index], residuals["magnitude"][index])
        exact_components = (
            static["unintended_charge_count"] == 0
            and static["retained_magnitude_error"] <= TOPOLOGY_TOLERANCE
            and static["retained_smooth_error"] <= TOPOLOGY_TOLERANCE
            and static["whole_state_roundtrip_error"] <= TOPOLOGY_TOLERANCE
        )
        nuisance_pass = (
            static["vortex_relative_energy_error"] <= NUISANCE_TOLERANCE
            and static["vortex_spectrum_error"] <= NUISANCE_TOLERANCE
            and vortex_residual <= reciprocal_best
        )
        pair_records.append(
            {
                **static,
                "margins": margins,
                "one_step_residuals": {
                    "vortex": vortex_residual,
                    "smooth": residuals["smooth"][index],
                    "magnitude": residuals["magnitude"][index],
                },
                "exact_component_guard": exact_components,
                "nuisance_joint_guard": nuisance_pass,
            }
        )
    mean_margins = {
        arm: float(np.mean(values, dtype=np.float64)) for arm, values in arm_margins.items()
    }
    nuisance_maximum = max(
        mean_margins[arm] for arm in ("smooth", "magnitude", "global_phase", "zero_phase")
    )
    fourier_nuisance_max = max(
        mean_margins[arm] for arm in ("fourier_low", "fourier_high")
    )
    return {
        "pair_records": pair_records,
        "mean_margins": mean_margins,
        "mechanism_advantage": mean_margins["vortex"] - nuisance_maximum,
        "fourier_low_advantage": mean_margins["fourier_low"] - nuisance_maximum,
        "fourier_high_advantage": mean_margins["fourier_high"] - nuisance_maximum,
        "fourier_max_advantage": fourier_nuisance_max - nuisance_maximum,
        "pca_advantage": mean_margins["pca"] - nuisance_maximum,
        "random_direction_advantage": mean_margins["random_direction"] - nuisance_maximum,
        "exact_component_guards_pass": all(
            record["exact_component_guard"] for record in pair_records
        ),
        "nuisance_joint_pass_fraction": sum(
            record["nuisance_joint_guard"] for record in pair_records
        )
        / count,
        "continuation_calls_per_arm": count,
        "recurrent_steps_per_continuation": delay + copy_length,
        "readout_calls_per_continuation": copy_length,
    }


AMPLITUDE_SATURATION_THRESHOLD = 0.95


def _batch_amplitude_saturation(hidden: torch.Tensor) -> float:
    radius = torch.sqrt(
        hidden[:, 0].square() + hidden[:, 1].square()
    )
    return float((radius > AMPLITUDE_SATURATION_THRESHOLD).float().mean().cpu())


@torch.no_grad()
def evaluate_seed_model(model: nn.Module, seed: int, *, task_type: str = "copy") -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    batch = generate_copy_batch(
        seed,
        "test/heldout-delay-64",
        TEST_EXAMPLES,
        TEST_DELAY,
        device=device,
    )
    trace = run_copy(model, batch.symbols, TEST_DELAY, copy_length=batch.copy_length)
    targets = torch.flip(batch.symbols, dims=[1]) if task_type == "reverse" else batch.symbols
    accuracy = float((trace.logits.argmax(dim=-1) == targets).float().mean().cpu())
    post_topology = [analyze_topology(state) for state in trace.post_write]
    pre_go_topology = [analyze_topology(state) for state in trace.pre_go]
    persistence = [
        signed_jaccard(post, pre_go)
        for post, pre_go in zip(post_topology, pre_go_topology, strict=True)
    ]
    defect_prevalence = sum(record.nonzero_defect for record in post_topology) / TEST_EXAMPLES
    recipient_amplitude_saturation = _batch_amplitude_saturation(trace.post_write)

    catalog_symbols = donor_sequences(batch.symbols, copy_length=batch.copy_length)
    flat_donors = catalog_symbols.reshape(TEST_EXAMPLES * DONORS_PER_RECIPIENT, batch.copy_length)
    flat_hidden = post_write_in_chunks(model, flat_donors)
    donor_amplitude_saturation = _batch_amplitude_saturation(flat_hidden)
    recipient_hidden = trace.post_write.detach().cpu()
    recipient_valid_count = 0
    donor_candidate_total = 0
    donor_candidate_failures = 0
    valid_recipient_fields: list[np.ndarray] = []
    selected_pairs: list[PairCandidate] = []
    selected_recipient_symbols: list[torch.Tensor] = []
    selected_donor_symbols: list[torch.Tensor] = []
    selected_indices: list[int] = []
    for recipient_index in range(TEST_EXAMPLES):
        start = recipient_index * DONORS_PER_RECIPIENT
        try:
            precomputed_recipient = decompose_hidden(
                recipient_hidden[recipient_index],
                magnitude_epsilon=MAGNITUDE_EPSILON,
                tolerance=TOPOLOGY_TOLERANCE,
            )
        except ValueError:
            continue
        recipient_valid_count += 1
        valid_recipient_fields.append(hidden_to_complex(recipient_hidden[recipient_index]))
        donor_candidate_total += DONORS_PER_RECIPIENT
        candidates = [flat_hidden[start + offset] for offset in range(DONORS_PER_RECIPIENT)]
        donor_failures: list[int] = []
        selected = select_donor_pair(
            recipient_hidden[recipient_index],
            candidates,
            precomputed_recipient=precomputed_recipient,
            donor_decompose_failures=donor_failures,
        )
        donor_candidate_failures += len(donor_failures)
        if selected is None:
            continue
        selected_pairs.append(selected)
        selected_indices.append(recipient_index)
        selected_recipient_symbols.append(batch.symbols[recipient_index].detach().cpu())
        selected_donor_symbols.append(
            catalog_symbols[recipient_index, selected.donor_index].detach().cpu()
        )
    intervention: dict[str, Any] | None
    evaluation_error: str | None = None
    pca = None
    if len(valid_recipient_fields) >= 8:
        pca = fit_pca(valid_recipient_fields, k=8)
    if selected_pairs:
        try:
            intervention = evaluate_selected_pairs(
                model,
                selected_pairs,
                torch.stack(selected_recipient_symbols),
                torch.stack(selected_donor_symbols),
                selected_indices,
                delay=TEST_DELAY,
                task_type=task_type,
                pca=pca,
            )
        except (RuntimeError, ValueError) as error:
            intervention = None
            evaluation_error = f"{type(error).__name__}: {error}"
    else:
        intervention = None
        evaluation_error = "no admissible donor pairs"
    return {
        "seed": int(seed),
        "test_content_sha256": batch.content_sha256,
        "donor_catalog_sha256": _tensor_sha256(catalog_symbols),
        "heldout_accuracy": accuracy,
        "defect_prevalence": defect_prevalence,
        "median_signed_persistence": float(np.median(persistence)),
        "valid_channel_fraction": sum(sum(record.valid_channels) for record in post_topology)
        / (TEST_EXAMPLES * model.spec.channels),
        "maximum_integer_residual": max(
            [record.maximum_integer_residual for record in post_topology + pre_go_topology],
            default=0.0,
        ),
        "net_charge_valid": all(
            record.net_charge_valid for record in post_topology + pre_go_topology
        ),
        "admissible_pair_count": len(selected_pairs),
        "recipient_valid_count": recipient_valid_count,
        "recipient_amplitude_saturation": recipient_amplitude_saturation,
        "donor_amplitude_saturation": donor_amplitude_saturation,
        "donor_candidate_total": donor_candidate_total,
        "donor_candidate_failures": donor_candidate_failures,
        "intervention": intervention,
        "evaluation_error": evaluation_error,
    }


def bootstrap_lower_bound(advantages: Sequence[float]) -> float:
    values = np.asarray(advantages, dtype=np.float64)
    n = len(values)
    if n < 5 or not np.isfinite(values).all():
        raise ValueError(f"bootstrap requires at least five finite seed advantages, got {n}")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025, method="linear"))


def _minimum_passing_seed_count(n: int) -> int:
    """Eighty percent of seeds, preserving the original four-of-five gate."""

    return max(1, int(np.ceil(0.8 * n)))


def _per_model_decision(
    seed_records: Sequence[dict[str, Any]],
    *,
    untrained_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n = len(seed_records)
    if tuple(int(record["seed"]) for record in seed_records) != tuple(range(n)):
        raise ValueError(f"decision requires ordered seeds 0--{n - 1}")
    minimum_passing = _minimum_passing_seed_count(n)
    interventions = [record.get("intervention") for record in seed_records]
    complete = all(intervention is not None for intervention in interventions)
    advantages = (
        [float(intervention["mechanism_advantage"]) for intervention in interventions]
        if complete
        else []
    )
    bootstrap_lower = bootstrap_lower_bound(advantages) if complete else None
    prerequisite_seed_count = sum(
        record["heldout_accuracy"] >= 0.95
        and record["defect_prevalence"] >= 0.50
        and record["median_signed_persistence"] >= 0.50
        for record in seed_records
    )
    clauses = {
        "prerequisites_met": prerequisite_seed_count >= minimum_passing,
        "pair_counts_sufficient": all(record["admissible_pair_count"] >= PAIR_MINIMUM for record in seed_records),
        "topology_guards_hold": all(
            record["net_charge_valid"]
            and record["maximum_integer_residual"] <= TOPOLOGY_TOLERANCE
            for record in seed_records
        ),
        "interventions_complete": complete,
        "directional_sanity": complete
        and all(
            intervention["mean_margins"]["natural_recipient"] < 0.0
            and intervention["mean_margins"]["whole_state"] > 0.0
            and intervention["mean_margins"]["vortex"] > 0.0
            for intervention in interventions
        ),
        "component_guards_pass": complete
        and all(intervention["exact_component_guards_pass"] for intervention in interventions),
        "nuisance_guards_pass": complete
        and all(
            intervention["nuisance_joint_pass_fraction"] >= 0.90
            for intervention in interventions
        ),
        "majority_advantages_positive": complete
        and sum(value > 0.0 for value in advantages) >= minimum_passing,
        "bootstrap_lower_positive": bootstrap_lower is not None and bootstrap_lower > 0.0,
        "split_hashes_unique": len({record["test_content_sha256"] for record in seed_records}) == n,
        "no_evaluation_errors": all(record["evaluation_error"] is None for record in seed_records),
        "defect_learned_not_innate": untrained_record is None or all(
            record.get("defect_prevalence", 0.0) > untrained_record.get("defect_prevalence", 0.0)
            for record in seed_records
        ),
    }
    return {
        "clauses": clauses,
        "all_pass": all(clauses.values()),
        "prerequisite_seed_count": prerequisite_seed_count,
        "mechanism_advantages": advantages,
        "bootstrap_lower_95": bootstrap_lower,
    }


def decide_learned_pilot(
    model_seed_records: dict[str, Sequence[dict[str, Any]]],
    *,
    model_types: tuple[str, ...] = ("u1", "plain"),
    untrained_records: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    per_model: dict[str, dict[str, Any]] = {}
    all_adv: dict[str, list[float]] = {}
    for model_type in model_types:
        records = model_seed_records.get(model_type, [])
        if not records:
            per_model[model_type] = {
                "all_pass": False, "clauses": {}, "error": "no records",
                "mechanism_advantages": [], "bootstrap_lower_95": None,
            }
            all_adv[model_type] = []
            continue
        untrained = untrained_records.get(model_type) if untrained_records else None
        decision = _per_model_decision(records, untrained_record=untrained)
        per_model[model_type] = decision
        all_adv[model_type] = decision["mechanism_advantages"]

    cross_clauses: dict[str, bool] = {}
    cross_bootstrap_lower: float | None = None
    if len(model_types) >= 2:
        primary = model_types[0]
        competitor = model_types[1]
        adv_primary = all_adv.get(primary, [])
        adv_competitor = all_adv.get(competitor, [])
        if adv_primary and adv_competitor and len(adv_primary) == len(adv_competitor):
            cross_clauses[f"{primary}_advantage_exceeds_{competitor}"] = all(
                a > b for a, b in zip(adv_primary, adv_competitor, strict=True)
            )
            cross_clauses[f"{primary}_mean_advantage_exceeds_{competitor}"] = (
                float(np.mean(adv_primary)) > float(np.mean(adv_competitor))
            )
            diffs = [a - b for a, b in zip(adv_primary, adv_competitor, strict=True)]
            if len(diffs) >= 5 and all(np.isfinite(diffs)):
                cross_bootstrap_lower = bootstrap_lower_bound(diffs)
                cross_clauses["cross_paired_lower_positive"] = cross_bootstrap_lower > 0.0

    u1_pass = per_model.get(model_types[0], {}).get("all_pass", False)

    if not u1_pass:
        status = "PM1_LEARNED_NO_GO"
    elif len(model_types) >= 2:
        competitor = model_types[1]
        competitor_decision = per_model.get(competitor, {})
        competitor_interventions = competitor_decision.get(
            "clauses", {}
        ).get("interventions_complete", False)
        if competitor_interventions:
            cross_positive = cross_clauses.get("cross_paired_lower_positive", False)
            status = "PM1_SURVIVE_LEARNED_PILOT" if cross_positive else "PM1_LEARNED_NO_GO"
        else:
            status = "PM1_SURVIVE_LEARNED_PILOT"
    else:
        status = "PM1_SURVIVE_LEARNED_PILOT" if u1_pass else "PM1_LEARNED_NO_GO"
    model_fields: dict[str, Any] = {}
    for model_type in model_types:
        d = per_model[model_type]
        model_fields[f"{model_type}_pass"] = d["all_pass"]
        model_fields[f"{model_type}_advantages"] = d["mechanism_advantages"]
        model_fields[f"{model_type}_bootstrap_lower_95"] = d["bootstrap_lower_95"]
        model_fields[f"{model_type}_clauses"] = d.get("clauses", {})

    return {
        "status": status,
        "clauses": cross_clauses,
        "cross_paired_bootstrap_lower_95": cross_bootstrap_lower,
        "per_model": model_fields,
        **model_fields,
    }


# ── Optimized batched evaluation (all arms in one forward pass) ──

@torch.no_grad()
def evaluate_seed_optimized(model, seed, *, task_type="copy"):
    """Batched evaluation: stacks all intervention arms into one continue_copy.
    
    ~3x faster than evaluate_seed_model on GPU."""
    model.eval()
    device = next(model.parameters()).device
    batch = generate_copy_batch(seed, "test/heldout-delay-64", TEST_EXAMPLES, TEST_DELAY, device=device)
    trace = run_copy(model, batch.symbols, TEST_DELAY)
    targets = torch.flip(batch.symbols, dims=[1]) if task_type == "reverse" else batch.symbols
    accuracy = float((trace.logits.argmax(dim=-1) == targets).float().mean().cpu())
    post_topology = [analyze_topology(s) for s in trace.post_write]
    defect_prevalence = sum(r.nonzero_defect for r in post_topology) / TEST_EXAMPLES
    persistence = [signed_jaccard(post_topology[i], analyze_topology(trace.pre_go[i])) for i in range(TEST_EXAMPLES)]

    catalog_symbols = donor_sequences(batch.symbols)
    flat_donors = catalog_symbols.reshape(TEST_EXAMPLES * DONORS_PER_RECIPIENT, batch.copy_length)
    flat_hidden = post_write_in_chunks(model, flat_donors)
    recipient_hidden = trace.post_write.detach().cpu()

    selected, pair_indices = [], []
    for ri in range(TEST_EXAMPLES):
        donors = [flat_hidden[ri * DONORS_PER_RECIPIENT + di] for di in range(DONORS_PER_RECIPIENT)]
        pair = select_donor_pair(recipient_hidden[ri], donors)
        if pair is not None:
            selected.append(pair); pair_indices.append(ri)

    if not selected:
        return {"task_accuracy": accuracy, "defect_prevalence": defect_prevalence, "pair_count": 0, "status": "NO_PAIRS"}

    arms = ["vortex", "smooth", "magnitude", "whole_phase", "whole_state",
            "fourier_low", "fourier_high", "random_direction"]  # PCA excluded (needs fitted model)
    all_fields = {}
    for pair in selected:
        r, d = pair.recipient, pair.donor
        for arm in arms:
            if arm in ("vortex", "smooth", "magnitude", "whole_phase", "whole_state"):
                f = component_intervention(r, d, arm)
            elif arm == "fourier_low":
                f = fourier_field_intervention(pair.recipient_field, pair.donor_field, "fourier_low")
            elif arm == "fourier_high":
                f = fourier_field_intervention(pair.recipient_field, pair.donor_field, "fourier_high")
            elif arm == "random_direction":
                f = random_direction_intervention(pair.recipient_field, pair.donor_field, seed=0)
            all_fields.setdefault(arm, []).append(f)

    n_pairs = len(selected)
    arm_fields = []
    arm_names = sorted(all_fields)
    for arm in arm_names:
        for f in all_fields[arm]:
            arm_fields.append(complex_to_hidden(f).unsqueeze(0))
    arm_batch = torch.cat(arm_fields, dim=0).to(device)

    arm_logits, _ = continue_copy(model, arm_batch, TEST_DELAY, copy_length=batch.copy_length)
    donor_indices = [p.donor_index for p in selected]
    donor_targets = torch.stack([catalog_symbols.reshape(TEST_EXAMPLES, DONORS_PER_RECIPIENT, batch.copy_length)[pi, di]
                                  for pi, di in zip(pair_indices, donor_indices)]).to(device)
    recipient_targets = batch.symbols[pair_indices].to(device)

    margins = {}
    for arm in arm_names:
        start = arm_names.index(arm) * n_pairs
        ll = arm_logits[start:start + n_pairs]
        r_ll = torch.gather(ll.log_softmax(dim=-1), -1, recipient_targets.unsqueeze(-1).expand(-1, -1, 10)).squeeze(-1).mean(-1).cpu().numpy()
        d_ll = torch.gather(ll.log_softmax(dim=-1), -1, donor_targets.unsqueeze(-1).expand(-1, -1, 10)).squeeze(-1).mean(-1).cpu().numpy()
        margins[arm] = (d_ll - r_ll).tolist()

    return {"task_accuracy": accuracy, "defect_prevalence": defect_prevalence,
            "mean_persistence": float(np.mean(persistence)) if persistence else 0.0,
            "pair_count": n_pairs, "per_family_margins": margins,
            "vortex_mean_margin": float(np.mean(margins.get("vortex", [0.0]))),
            "whole_state_margins": margins.get("whole_state", [])}


# ── Calibration executor ──

def run_calibration(output_dir, device, n_seeds=3, n_updates=5000):
    """Train C=1 models and evaluate topology. ~1.5h on RTX 5080."""
    import time, json
    from pathlib import Path
    from .model import ModelSpec
    from ._artifacts import WriteOnceArtifact

    spec = ModelSpec(channels=1)
    output = Path(output_dir)
    writer = WriteOnceArtifact(output)
    results = {"seeds": {}}

    for seed in range(n_seeds):
        configure_determinism(seed, benchmark=True)
        from .model import make_scalar_u1_model
        model = make_scalar_u1_model(model_spec=spec, device=device)
        training = train_seed(seed, model_type="u1", model_spec=spec,
                              training_spec=TrainingSpec(updates=n_updates), device=device)
        eval_result = evaluate_seed_optimized(training.model, seed)
        results["seeds"][f"seed_{seed}"] = {
            "accuracy": float(training.selected_accuracy),
            "defect_prevalence": eval_result.get("defect_prevalence", 0),
            "vortex_margin": eval_result.get("vortex_mean_margin", 0),
            "pair_count": eval_result.get("pair_count", 0),
        }
        del model, training
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accs = [r["accuracy"] for r in results["seeds"].values()]
    results["mean_accuracy"] = float(np.mean(accs))
    results["overall_pass"] = results["mean_accuracy"] >= 0.90
    writer.write_json("results.json", results)
    writer.finalize()
    return results
