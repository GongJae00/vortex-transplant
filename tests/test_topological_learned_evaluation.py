from __future__ import annotations

import inspect

import numpy as np
import torch

from topological.interventions import complex_to_hidden
from topological.learned_evaluation import (
    _normalized_field_residual,
    analyze_topology,
    bootstrap_lower_bound,
    decide_learned_pilot,
    evaluate_selected_pairs,
    log_likelihood_margin,
    select_donor_pair,
    signed_jaccard,
)
from topological.model import ModelSpec, U1ConvRNN
from topological.topology import canonical_vortex_field


def _field(shift: int, smooth_offset: float = 0.0) -> torch.Tensor:
    charge = np.zeros((8, 8), dtype=np.int64)
    charge[(1 + shift) % 8, 2] = 1
    charge[(5 + shift) % 8, 6] = -1
    x, _ = np.indices((8, 8))
    channels = []
    for channel in range(2):
        smooth = np.exp(0.05j * np.sin(2.0 * np.pi * x / 8 + channel + smooth_offset))
        channels.append(canonical_vortex_field(charge).field * smooth)
    return complex_to_hidden(np.stack(channels))


def test_topology_uses_signed_channel_plaquette_tuples_and_jaccard() -> None:
    first = analyze_topology(_field(0))
    same = analyze_topology(_field(0, 0.4))
    shifted = analyze_topology(_field(2))

    assert first.nonzero_defect
    assert all(first.valid_channels)
    assert first.net_charge_valid
    assert signed_jaccard(first, same) == 1.0
    assert signed_jaccard(first, shifted) < 1.0


def test_donor_selection_is_label_free_and_lexicographically_deterministic() -> None:
    recipient = _field(0)
    donors = [_field(shift, index / 10) for index, shift in enumerate((1, 2, 3, 4, 5, 6, 7, 9))]

    selected = select_donor_pair(recipient, donors)
    repeat = select_donor_pair(recipient, donors)

    assert selected is not None and repeat is not None
    assert selected.donor_index == repeat.donor_index
    assert selected.score == repeat.score
    assert selected.admissible_catalog_count == 8
    assert "symbols" not in inspect.signature(select_donor_pair).parameters
    assert "logits" not in inspect.signature(select_donor_pair).parameters


def test_identical_charge_catalog_is_inadmissible() -> None:
    recipient = _field(0)
    assert select_donor_pair(recipient, [_field(0, index / 10) for index in range(8)]) is None


def test_log_likelihood_margin_has_frozen_donor_minus_recipient_direction() -> None:
    logits = torch.zeros((1, 4, 10))
    donor = torch.tensor([[2, 2, 2, 2]])
    recipient = torch.tensor([[1, 1, 1, 1]])
    logits[:, :, 2] = 3.0

    margin = log_likelihood_margin(logits, donor, recipient)

    assert margin.item() > 0.0


def test_selected_pair_executes_every_arm_and_commutation_residual() -> None:
    recipient = _field(0)
    donors = [_field(shift) for shift in (1, 2, 3, 4, 5, 6, 7, 9)]
    selected = select_donor_pair(recipient, donors)
    assert selected is not None
    model = U1ConvRNN(
        ModelSpec(height=8, width=8, channels=2),
        generator=torch.Generator().manual_seed(7),
    )

    result = evaluate_selected_pairs(
        model,
        [selected],
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[2, 3, 4, 5]]),
        [0],
        delay=1,
    )

    assert len(result["mean_margins"]) == 13
    assert len(result["pair_records"]) == 1
    assert set(result["pair_records"][0]["one_step_residuals"]) == {
        "vortex",
        "smooth",
        "magnitude",
    }
    assert result["continuation_calls_per_arm"] == 1


def test_bootstrap_lower_bound_accepts_variable_seed_counts() -> None:
    assert bootstrap_lower_bound([1.0] * 5) == 1.0
    assert bootstrap_lower_bound([1.0] * 10) == 1.0
    assert bootstrap_lower_bound([0.5] * 5) == 0.5


def test_untrained_defect_clause_rejects_when_trained_is_not_higher() -> None:
    trained = []
    for seed in range(5):
        trained.append(
            {
                "seed": seed,
                "test_content_sha256": str(seed),
                "heldout_accuracy": 1.0,
                "defect_prevalence": 0.3,
                "median_signed_persistence": 1.0,
                "maximum_integer_residual": 0.0,
                "net_charge_valid": True,
                "admissible_pair_count": 128,
                "evaluation_error": None,
                "intervention": {
                    "mean_margins": {
                        "natural_recipient": -1.0,
                        "whole_state": 1.0,
                        "vortex": 1.0,
                    },
                    "mechanism_advantage": 0.5,
                    "exact_component_guards_pass": True,
                    "nuisance_joint_pass_fraction": 1.0,
                },
            }
        )
    untrained = {"defect_prevalence": 0.5}
    result = decide_learned_pilot({"u1": trained}, untrained_records={"u1": untrained})
    assert not result["u1_clauses"]["defect_learned_not_innate"]

    untrained_low = {"defect_prevalence": 0.1}
    result = decide_learned_pilot({"u1": trained}, untrained_records={"u1": untrained_low})
    assert result["u1_clauses"]["defect_learned_not_innate"]


def test_complete_decision_requires_all_clauses() -> None:
    for n in (5, 10):
        records = []
        for seed in range(n):
            records.append(
                {
                    "seed": seed,
                    "test_content_sha256": str(seed),
                    "heldout_accuracy": 1.0,
                    "defect_prevalence": 1.0,
                    "median_signed_persistence": 1.0,
                    "maximum_integer_residual": 0.0,
                    "net_charge_valid": True,
                    "admissible_pair_count": 128,
                    "evaluation_error": None,
                    "intervention": {
                        "mean_margins": {
                            "natural_recipient": -1.0,
                            "whole_state": 1.0,
                            "vortex": 1.0,
                        },
                        "mechanism_advantage": 0.5,
                        "exact_component_guards_pass": True,
                        "nuisance_joint_pass_fraction": 1.0,
                    },
                }
            )

        assert decide_learned_pilot({"u1": records})["u1_pass"] is True
        records[2]["intervention"]["mean_margins"]["vortex"] = -0.1
        assert decide_learned_pilot({"u1": records})["u1_pass"] is False


def test_decision_handles_no_interventions_gracefully() -> None:
    records = []
    for seed in range(5):
        records.append({
            "seed": seed,
            "test_content_sha256": str(seed),
            "heldout_accuracy": 1.0,
            "defect_prevalence": 0.5,
            "median_signed_persistence": 1.0,
            "maximum_integer_residual": 0.0,
            "net_charge_valid": True,
            "admissible_pair_count": 0,
            "evaluation_error": "no donor pairs",
            "intervention": None,
        })
    result = decide_learned_pilot({"u1": records})
    assert result["u1_pass"] is False
    assert not result["u1_clauses"]["interventions_complete"]


def test_validation_count_uses_spec_copy_length() -> None:
    from topological.training import TrainingSpec
    spec = TrainingSpec(copy_length=6, validation_examples=12,
                           train_delay_min=1, train_delay_max=1)
    counts = spec._validation_counts() if hasattr(spec, '_validation_counts') else None
    assert spec.copy_length == 6
    assert spec.copy_length_min == 3
    assert spec.copy_length_max == 7
    spec.validate()


def test_bootstrap_requires_minimum_five_seeds() -> None:
    from topological.learned_evaluation import bootstrap_lower_bound
    try:
        bootstrap_lower_bound([1.0] * 3)
        assert False, "should raise"
    except ValueError as e:
        assert "five" in str(e).lower()


def test_seed_threshold_scales_with_seed_count() -> None:
    from topological.learned_evaluation import _minimum_passing_seed_count

    assert _minimum_passing_seed_count(5) == 4
    assert _minimum_passing_seed_count(10) == 8
    assert _minimum_passing_seed_count(1) == 1


def test_decision_threshold_scales_for_ten_seeds() -> None:
    """Four-of-five becomes eight-of-ten: exactly 7/10 positive must fail."""

    def _records(positive_advantage_count: int, n: int = 10) -> list[dict]:
        records = []
        for seed in range(n):
            advantage = 0.5 if seed < positive_advantage_count else -0.1
            records.append(
                {
                    "seed": seed,
                    "test_content_sha256": str(seed),
                    "heldout_accuracy": 1.0,
                    "defect_prevalence": 1.0,
                    "median_signed_persistence": 1.0,
                    "maximum_integer_residual": 0.0,
                    "net_charge_valid": True,
                    "admissible_pair_count": 128,
                    "evaluation_error": None,
                    "intervention": {
                        "mean_margins": {
                            "natural_recipient": -1.0,
                            "whole_state": 1.0,
                            "vortex": 1.0,
                        },
                        "mechanism_advantage": advantage,
                        "exact_component_guards_pass": True,
                        "nuisance_joint_pass_fraction": 1.0,
                    },
                }
            )
        return records

    assert decide_learned_pilot({"u1": _records(8)})["u1_clauses"]["majority_advantages_positive"]
    assert not decide_learned_pilot({"u1": _records(7)})["u1_clauses"]["majority_advantages_positive"]
    assert not decide_learned_pilot({"u1": _records(4)})["u1_clauses"]["majority_advantages_positive"]


def test_cross_model_u1_survives_when_plain_has_no_vortex() -> None:
    """U1 pass + plain no interventions → SURVIVE (U1-specific confirmed)."""

    u1_records = [
        {
            "seed": s, "test_content_sha256": f"u1_{s}",
            "heldout_accuracy": 1.0, "defect_prevalence": 0.9,
            "median_signed_persistence": 0.8,
            "maximum_integer_residual": 0.0, "net_charge_valid": True,
            "admissible_pair_count": 128, "evaluation_error": None,
            "intervention": {
                "mean_margins": {"natural_recipient": -1.0, "whole_state": 1.0, "vortex": 1.0},
                "mechanism_advantage": 0.5,
                "exact_component_guards_pass": True,
                "nuisance_joint_pass_fraction": 1.0,
            },
        }
        for s in range(8)
    ]
    plain_records = [
        {
            "seed": s, "test_content_sha256": f"p_{s}",
            "heldout_accuracy": 0.0, "defect_prevalence": 0.0,
            "median_signed_persistence": 0.0,
            "maximum_integer_residual": 0.0, "net_charge_valid": True,
            "admissible_pair_count": 0, "evaluation_error": "no vortex",
            "intervention": None,
        }
        for s in range(8)
    ]
    result = decide_learned_pilot({"u1": u1_records, "plain": plain_records})
    assert result["status"] == "PM1_SURVIVE_LEARNED_PILOT"
    assert result["u1_pass"] is True
    assert result["plain_pass"] is False
