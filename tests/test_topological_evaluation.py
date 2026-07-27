from __future__ import annotations

from topological.evaluation import decide_feasibility, evaluate_seed
from topological.fixture import generate_fields


def _passing_record(seed: int) -> dict:
    return {
        "seed": seed,
        "metrics": {
            "unit_field_reconstruction_error": 1e-12,
            "integer_charge_residual": 1e-12,
            "unintended_charge_count": 0,
            "retained_component_error": 1e-12,
            "exact_relative_gradient_energy_error": 1e-12,
            "exact_radial_spectrum_error": 1e-12,
            "hybrid_relative_gradient_energy_error": 0.01,
            "hybrid_radial_spectrum_error": 0.01,
        },
        "invariants": {"all": True},
    }


def test_seed_evaluator_checks_all_frozen_fields_and_components() -> None:
    record = evaluate_seed(generate_fields(97))

    assert record["seed"] == 97
    assert record["donor_trials_per_field"] == 4
    assert all(record["invariants"].values())
    assert record["metrics"]["unit_field_reconstruction_error"] <= 1e-10
    assert record["metrics"]["unintended_charge_count"] == 0


def test_decision_requires_all_five_seeds_and_all_clauses() -> None:
    passing = [_passing_record(seed) for seed in range(5)]
    assert decide_feasibility(passing)["status"] == "PM1_DECOMPOSITION_FEASIBLE"

    passing[3]["metrics"]["hybrid_radial_spectrum_error"] = 0.051
    assert decide_feasibility(passing)["status"] == "PM1_NO_GO_DECOMPOSITION"
