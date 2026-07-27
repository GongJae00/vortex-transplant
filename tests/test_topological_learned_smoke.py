from __future__ import annotations

import json

import pytest

from topological.learned_smoke import (
    _exercise_device,
    _warm_device,
    assert_result_blind,
    canonical_config_path,
    require_canonical_config,
)
import torch


@pytest.mark.slow
def test_exercise_device_passes_all_guards_for_both_model_types() -> None:
    """Regression: plain model previously failed the u1-equivariance guard."""

    for model_type in ("u1", "plain"):
        result = _exercise_device(torch.device("cpu"), model_type=model_type)

        assert all(result["guards"].values()), result["guards"]
        assert result["guards"]["equivariance_matches_design"]
        assert result["guards"]["intervention_arms_complete"]


def test_frozen_learned_config_matches_code_contract() -> None:
    digest = require_canonical_config(canonical_config_path())

    assert len(digest) == 64


def test_disposable_backend_warmup_executes_without_retained_output() -> None:
    assert _warm_device(torch.device("cpu")) is None


def test_result_blind_audit_rejects_scientific_output_keys(tmp_path) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "work.json").write_text(
        json.dumps({"optimizer_steps": 20, "intervention_arm_count": 9}),
        encoding="utf-8",
    )
    assert_result_blind(safe)

    (safe / "forbidden.json").write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden result key"):
        assert_result_blind(safe)
