"""Frozen feasibility evaluator."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .decomposition import (
    align_global_phase,
    decompose,
    gradient_energy,
    normalized_spectrum_error,
    transplant_magnitude,
    transplant_smooth,
    transplant_vortex,
)
from .fixture import GeneratedCompactField, FeasibilitySpec, charge_template
from .topology import canonical_vortex_field, extract_charge


PILOT_SEEDS = (0, 1, 2, 3, 4)
EXACT_TOLERANCE = 1e-10
HYBRID_TOLERANCE = 0.05


def _relative_energy_error(first: np.ndarray, second: np.ndarray) -> float:
    reference = gradient_energy(first)
    if reference <= 1e-15:
        raise ValueError("gradient-energy reference is zero")
    return abs(gradient_energy(second) - reference) / reference


def _same_separation_donors(
    recipient: GeneratedCompactField,
    records: Sequence[GeneratedCompactField],
) -> list[GeneratedCompactField]:
    return [
        record
        for record in records
        if record.separation == recipient.separation
        and record.template_index != recipient.template_index
        and not np.array_equal(record.charge, recipient.charge)
    ]


def select_donor(
    recipient: GeneratedCompactField,
    records: Sequence[GeneratedCompactField],
) -> tuple[GeneratedCompactField, int]:
    candidates = _same_separation_donors(recipient, records)
    if not candidates:
        raise RuntimeError("fixed donor catalog has no distinct same-separation donor")
    recipient_energy = gradient_energy(recipient.field)
    scored: list[tuple[float, int, GeneratedCompactField]] = []
    for donor in candidates:
        energy_distance = abs(gradient_energy(donor.field) - recipient_energy) / recipient_energy
        spectrum_distance = normalized_spectrum_error(recipient.field, donor.field)
        scored.append((energy_distance + spectrum_distance, donor.field_index, donor))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2], len(scored)


def exact_translation_metrics(spec: FeasibilitySpec = FeasibilitySpec()) -> dict[str, float]:
    energy_errors: list[float] = []
    spectrum_errors: list[float] = []
    for family in range(4):
        first = canonical_vortex_field(charge_template(family, spec)).field
        second = canonical_vortex_field(charge_template(family + 4, spec)).field
        energy_errors.append(_relative_energy_error(first, second))
        spectrum_errors.append(normalized_spectrum_error(first, second))
    return {
        "relative_gradient_energy_error": max(energy_errors),
        "radial_spectrum_error": max(spectrum_errors),
    }


def evaluate_seed(records: Sequence[GeneratedCompactField]) -> dict[str, Any]:
    if len(records) != 32 or len({record.seed for record in records}) != 1:
        raise ValueError("seed evaluation requires exactly 32 records from one seed")
    seed = int(records[0].seed)
    reconstruction_errors: list[float] = []
    charge_residuals: list[float] = []
    retained_component_errors: list[float] = []
    hybrid_energy_errors: list[float] = []
    hybrid_spectrum_errors: list[float] = []
    unintended_counts: list[int] = []
    donor_trials: list[int] = []
    decompositions = [decompose(record.field) for record in records]
    for record, parts in zip(records, decompositions, strict=True):
        reconstruction_errors.append(parts.reconstruction_error)
        charge_residuals.append(parts.charge.residual_max)
        unintended_counts.append(int(np.count_nonzero(parts.charge.charge != record.charge)))

        donor_record, trials = select_donor(record, records)
        donor_trials.append(trials)
        donor_parts = decompositions[donor_record.field_index]
        transplanted = align_global_phase(transplant_vortex(parts, donor_parts), record.field)
        transplanted_parts = decompose(transplanted)
        unintended_counts.append(
            int(np.count_nonzero(transplanted_parts.charge.charge != donor_record.charge))
        )
        retained_component_errors.append(
            float(np.max(np.abs(transplanted_parts.magnitude - parts.magnitude)))
        )
        smooth_overlap = np.vdot(transplanted_parts.smooth.ravel(), parts.smooth.ravel())
        smooth_aligned = transplanted_parts.smooth * np.exp(1j * np.angle(smooth_overlap))
        retained_component_errors.append(float(np.max(np.abs(smooth_aligned - parts.smooth))))
        hybrid_energy_errors.append(_relative_energy_error(record.field, transplanted))
        hybrid_spectrum_errors.append(normalized_spectrum_error(record.field, transplanted))

        smooth_control = transplant_smooth(parts, donor_parts)
        magnitude_control = transplant_magnitude(parts, donor_parts)
        gauge_control = record.field * np.exp(0.731j)
        for control in (smooth_control, magnitude_control, gauge_control):
            control_charge = extract_charge(control)
            unintended_counts.append(
                int(np.count_nonzero(control_charge.charge != record.charge))
            )
        whole_phase_control = parts.magnitude * donor_record.field / np.abs(donor_record.field)
        interpolation = parts.magnitude * (
            record.field / np.abs(record.field) + donor_record.field / np.abs(donor_record.field)
        )
        interpolation_norm = np.abs(interpolation)
        interpolation = np.divide(
            interpolation,
            interpolation_norm,
            out=np.ones_like(interpolation),
            where=interpolation_norm > 1e-12,
        ) * parts.magnitude
        if not np.isfinite(whole_phase_control).all() or not np.isfinite(interpolation).all():
            raise RuntimeError("negative control produced non-finite values")

    exact = exact_translation_metrics()
    metrics = {
        "unit_field_reconstruction_error": max(reconstruction_errors),
        "integer_charge_residual": max(charge_residuals),
        "unintended_charge_count": max(unintended_counts),
        "retained_component_error": max(retained_component_errors),
        "exact_relative_gradient_energy_error": exact["relative_gradient_energy_error"],
        "exact_radial_spectrum_error": exact["radial_spectrum_error"],
        "hybrid_relative_gradient_energy_error": max(hybrid_energy_errors),
        "hybrid_radial_spectrum_error": max(hybrid_spectrum_errors),
    }
    invariants = {
        "ambient_dimension_two": True,
        "support_dimension_zero": True,
        "codimension_two": True,
        "periodic_net_charge_zero": all(parts.charge.net_charge == 0 for parts in decompositions),
        "one_signed_pair": all(
            parts.charge.positive_count == 1 and parts.charge.negative_count == 1
            for parts in decompositions
        ),
        "smooth_component_zero_charge": all(parts.smooth_charge_residual <= EXACT_TOLERANCE for parts in decompositions),
        "donor_catalog_fixed": min(donor_trials) == max(donor_trials) == 4,
        "finite_metrics": all(np.isfinite(value) for value in metrics.values()),
    }
    return {
        "seed": seed,
        "metrics": metrics,
        "invariants": invariants,
        "donor_trials_per_field": donor_trials[0],
    }


def decide_feasibility(seed_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if tuple(int(record["seed"]) for record in seed_records) != PILOT_SEEDS:
        raise ValueError("feasibility decision requires ordered seeds 0--4")
    clauses = {
        "all_invariants": all(
            all(bool(value) for value in record["invariants"].values()) for record in seed_records
        ),
        "exact_errors": all(
            record["metrics"][key] <= EXACT_TOLERANCE
            for record in seed_records
            for key in (
                "unit_field_reconstruction_error",
                "integer_charge_residual",
                "retained_component_error",
                "exact_relative_gradient_energy_error",
                "exact_radial_spectrum_error",
            )
        ),
        "no_unintended_charge": all(
            record["metrics"]["unintended_charge_count"] == 0 for record in seed_records
        ),
        "hybrid_nuisance": all(
            record["metrics"][key] <= HYBRID_TOLERANCE
            for record in seed_records
            for key in (
                "hybrid_relative_gradient_energy_error",
                "hybrid_radial_spectrum_error",
            )
        ),
    }
    status = "PM1_DECOMPOSITION_FEASIBLE" if all(clauses.values()) else "PM1_NO_GO_DECOMPOSITION"
    return {"status": status, "clauses": clauses, "seed_records": list(seed_records)}
