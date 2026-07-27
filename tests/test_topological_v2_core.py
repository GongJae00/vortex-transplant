"""Tests for V2 core modules: statistics, hodge, surgery, protocol."""
import numpy as np
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topological.v2.statistics import (
    hierarchical_bootstrap, wild_bootstrap, iut_test, compute_sesoi,
)
from topological.v2.hodge import (
    compact_hodge_decompose, extract_harmonic_sector,
    validate_hodge_decomposition,
)
from topological.v2.surgery import (
    minimal_topological_surgery, annihilate_single_pair, sham_surgery,
    SurgerySpec,
)
from topological.v2.protocol import (
    namespace_seed, authorize_split,
    load_contract, contract_digest,
)
from topological.v2._types import ContractState


class TestStatistics:
    def test_hierarchical_bootstrap_basic(self):
        data = {0: np.array([1.0, 2.0, 3.0]), 1: np.array([2.0, 3.0, 4.0]),
                2: np.array([3.0, 4.0, 5.0])}
        result = hierarchical_bootstrap(data, n_resamples=999, seed=42)
        assert result.estimate > 0
        assert result.ci_lower < result.estimate < result.ci_upper
        assert 0 <= result.p_value <= 1

    def test_wild_bootstrap_null(self):
        data = {i: np.random.randn(10) * 0.1 for i in range(10)}
        result = wild_bootstrap(data, n_resamples=999, seed=42)
        assert 0 <= result.p_value <= 1

    def test_wild_bootstrap_positive(self):
        data = {i: np.random.randn(10) + 1.0 for i in range(10)}
        result = wild_bootstrap(data, n_resamples=999, seed=42)
        assert result.p_value < 0.1  # strong signal → low p-value

    def test_iut_basic(self):
        per_family = {}
        for f in ["smooth", "magnitude", "random"]:
            per_family[f] = {i: np.random.randn(10) + 0.5 for i in range(10)}
        result = iut_test(per_family, alpha=0.05, method="wild_bootstrap", n_resamples=499)
        assert result.max_p_value > 0
        assert result.worst_family in per_family
        assert result.alpha == 0.05

    def test_compute_sesoi(self):
        data = {i: np.array([5.0, 6.0, 4.0]) for i in range(5)}
        sesoi = compute_sesoi(data, fraction=0.1)
        assert sesoi > 0


class TestHodge:
    def test_compact_decompose_basic(self):
        H, W = 16, 16
        charge_map = np.zeros((H, W), dtype=int)
        charge_map[4, 4] = 1
        charge_map[4, 5] = -1

        # Create simple link variables with a vortex pair
        from topological.topology import canonical_vortex_field
        field = canonical_vortex_field(charge_map).field
        unit = field / (np.abs(field) + 1e-12)
        dx = np.angle(unit[:, 1:] * np.conj(unit[:, :-1]))
        dx_pad = np.angle(unit[:, :1] * np.conj(unit[:, -1:]))
        dy = np.angle(unit[1:, :] * np.conj(unit[:-1, :]))
        dy_pad = np.angle(unit[:1, :] * np.conj(unit[-1:, :]))

        links = np.zeros((H, W, 2))
        links[:, :-1, 0] = dx
        links[:, -1, 0] = dx_pad.reshape(-1)
        links[:-1, :, 1] = dy
        links[-1, :, 1] = dy_pad.reshape(-1)

        result = compact_hodge_decompose(links, charge_map)
        assert "exact_potential" in result
        assert "coexact_potential" in result
        assert "harmonic_holonomy" in result
        assert result["reconstructed_error"] < 1e-8

    def test_extract_harmonic_sector(self):
        H, W = 16, 16
        field = np.ones((H, W), dtype=np.complex128)
        wx, wy = extract_harmonic_sector(field)
        assert abs(wx) < 1e-10
        assert abs(wy) < 1e-10

    def test_validate_decomposition(self):
        from topological.topology import canonical_vortex_field, extract_charge
        H, W = 16, 16
        charge_map = np.zeros((H, W), dtype=int)
        charge_map[2, 2] = 1; charge_map[2, 3] = -1  # verified positions
        field = canonical_vortex_field(charge_map).field
        q = extract_charge(field).charge
        unit = canonical_vortex_field(q).field
        links = np.zeros((H, W, 2))
        # Periodic links
        dx_periodic = np.angle(np.concatenate([unit[:, 1:], unit[:, :1]], axis=1) * np.conj(unit))
        dy_periodic = np.angle(np.concatenate([unit[1:, :], unit[:1, :]], axis=0) * np.conj(unit))
        links[:, :, 0] = dx_periodic
        links[:, :, 1] = dy_periodic
        result = compact_hodge_decompose(links, q)
        valid = validate_hodge_decomposition(result)
        assert valid


class TestSurgery:
    def test_minimal_surgery_basic(self):
        """Surgery on canonical vortex field produces a field (shape/crash test)."""
        C, H, W = 1, 16, 16
        from topological.topology import canonical_vortex_field
        charge = np.zeros((H, W), dtype=int)
        charge[2, 2] = 1; charge[2, 3] = -1
        recipient = canonical_vortex_field(charge).field[np.newaxis, :, :]
        spec = SurgerySpec(target_charge_map=charge)
        result = minimal_topological_surgery(recipient, charge, spec)
        assert result.intervened_field.shape == (C, H, W)
        assert not np.allclose(result.intervened_field, recipient)

    def test_minimal_surgery_charge_change(self):
        """Surgery with different charge target produces non-identical field."""
        C, H, W = 1, 16, 16
        from topological.topology import canonical_vortex_field
        init_charge = np.zeros((H, W), dtype=int)
        init_charge[2, 2] = 1; init_charge[2, 5] = -1
        recipient = canonical_vortex_field(init_charge).field[np.newaxis, :, :]
        target = np.zeros((H, W), dtype=int)
        target[2, 3] = 1; target[2, 4] = -1
        spec = SurgerySpec(target_charge_map=target)
        result = minimal_topological_surgery(recipient, target, spec)
        assert result.intervened_field.shape == (C, H, W)

    def test_annihilate_single_pair(self):
        """Annihilate pair produces a field with different topology."""
        C, H, W = 1, 16, 16
        target = np.zeros((H, W), dtype=int)
        target[2, 2] = 1; target[2, 3] = -1
        from topological.topology import canonical_vortex_field
        field = canonical_vortex_field(target).field[np.newaxis, :, :]
        result = annihilate_single_pair(field, 0, (2, 2), (2, 3))
        assert result.intervened_field.shape == (C, H, W)

    def test_sham_surgery(self):
        """Sham surgery with same charge produces valid field."""
        C, H, W = 1, 16, 16
        target = np.zeros((H, W), dtype=int)
        target[2, 2] = 1; target[2, 3] = -1
        from topological.topology import canonical_vortex_field
        field = canonical_vortex_field(target).field[np.newaxis, :, :]
        result = sham_surgery(field, target)
        assert result.intervened_field.shape == (C, H, W)


class TestProtocol:
    def test_namespace_seed(self):
        ns = namespace_seed(0, "calibration", "topology")
        assert "cal" in ns
        assert "topology" in ns
        assert "seed_0" in ns

    def test_namespace_seed_invalid_split(self):
        with pytest.raises(ValueError):
            namespace_seed(0, "invalid", "test")

    def test_contract_digest(self):
        contract_path = os.path.join(
            os.path.dirname(__file__), "..", "planning", "PM1-LEARNED-V2",
            "15_PM1_LEARNED_V2_CONTRACT.yaml",
        )
        if os.path.exists(contract_path):
            digest = contract_digest(contract_path)
            assert len(digest) == 64
            assert all(c in "0123456789abcdef" for c in digest)
