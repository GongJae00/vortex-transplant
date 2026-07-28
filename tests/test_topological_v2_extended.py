"""Tests for V2 extended modules: topology, model, smoke."""
import numpy as np
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topological.v2.v2_topology import (
    compute_branch_margins, compute_per_channel_branch_margins,
    defect_tracking, per_channel_defect_prevalence, extract_charge_map,
)
from topological.v2.v2_model import (
    make_factorial_model, make_scalar_u1_model,
)
from topological.v2.smoke import run_cpu_integrity_smoke
from topological.model import ModelSpec


class TestV2Topology:
    def test_branch_margins_smooth(self):
        """Smooth field has high branch margins."""
        H, W = 16, 16
        field = np.ones((H, W), dtype=np.complex128)
        margins = compute_branch_margins(field)
        assert margins.min_margin > 2.0  # far from branch cut
        assert margins.q01_margin > 2.0

    def test_branch_margins_random(self):
        """Random phase field has low branch margins."""
        rng = np.random.default_rng(42)
        field = np.exp(1j * rng.uniform(-np.pi, np.pi, (16, 16)))
        margins = compute_branch_margins(field)
        assert margins.min_margin < 1.0  # near branch cut somewhere
        assert 0.0 <= margins.median_margin <= np.pi

    def test_per_channel_margins(self):
        C, H, W = 4, 16, 16
        field = np.ones((C, H, W), dtype=np.complex128)
        margins = compute_per_channel_branch_margins(field)
        assert len(margins) == C
        for m in margins:
            assert m.min_margin > 2.0

    def test_defect_tracking_identity(self):
        """Same field → all matched, no births/deaths."""
        H, W = 16, 16
        rng = np.random.default_rng(42)
        field = np.exp(1j * rng.uniform(-np.pi, np.pi, (H, W)))
        result = defect_tracking(field, field)
        assert result["births"] == []
        assert result["deaths"] == []
        assert 0.0 <= result["signed_jaccard"] <= 1.0

    def test_per_channel_prevalence(self):
        C, H, W = 4, 16, 16
        rng = np.random.default_rng(42)
        field = np.exp(1j * rng.uniform(-np.pi, np.pi, (C, H, W)))
        prev = per_channel_defect_prevalence(field)
        assert len(prev) == C
        # Random fields → most channels have both signs
        assert sum(prev) >= C // 2

    def test_extract_charge_map(self):
        C, H, W = 2, 16, 16
        field = np.ones((C, H, W), dtype=np.complex128)
        charge_map = extract_charge_map(field)
        assert charge_map.shape == (H, W)
        assert np.all(charge_map == 0)


class TestV2Model:
    def test_scalar_u1_creates(self):
        model = make_scalar_u1_model()
        # C=1 model has channels=1
        state = model.initial_state(1, device=torch.device("cpu"))
        # Shape: (1, 2, C=1, H, W)
        assert state.shape[2] == 1

    def test_factorial_variants_all_create(self):
        spec = ModelSpec(channels=2, height=8, width=8)
        names = ['U1CommutingLinear_RadialNonlinear', 'U1CommutingLinear_ElementwiseNonlinear',
                 'UnrestrictedLinear_RadialNonlinear', 'UnrestrictedLinear_ElementwiseNonlinear']
        for name in names:
            model = make_factorial_model(name, model_spec=spec)
            assert model is not None, f"Failed to create {name}"
            # Test step-based forward (models use step/logits, not forward)
            tokens = torch.randint(1, 9, (2, 4))
            hidden = model.initial_state(2, device=torch.device("cpu"))
            for t in range(4):
                hidden = model.step(tokens[:, t], hidden)
            logits = model.logits(hidden)
            assert logits.shape == (2, spec.vocabulary), f"{name} forward failed"

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError):
            make_factorial_model("NonexistentVariant")


class TestV2Smoke:
    def test_cpu_integrity_smoke(self):
        result = run_cpu_integrity_smoke()
        assert result["overall_pass"], f"Smoke failed: {result.get('errors', [])}"
        for check_name, check_val in result["checks"].items():
            assert check_val, f"Check '{check_name}' failed"
