from __future__ import annotations

import numpy as np
import pytest

from topological.topology import (
    canonical_vortex_field,
    extract_charge,
    phase_links,
    plaquette_curl,
    unit_field,
)


def _pair_charge(shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    charge = np.zeros(shape, dtype=np.int64)
    charge[8, 9] = 1
    charge[13, 12] = -1
    return charge


def test_canonical_periodic_field_recovers_signed_codimension_two_pair() -> None:
    charge = _pair_charge()

    canonical = canonical_vortex_field(charge)
    extracted = extract_charge(canonical.field)

    assert np.array_equal(extracted.charge, charge)
    assert extracted.net_charge == 0
    assert extracted.positive_count == 1
    assert extracted.negative_count == 1
    assert extracted.residual_max <= 1e-10
    assert canonical.poisson_residual_max <= 1e-10
    assert canonical.curl_residual_max <= 1e-10
    assert canonical.integration_residual_max <= 1e-10
    assert canonical.holonomy_x <= 1e-10
    assert canonical.holonomy_y <= 1e-10


def test_phase_links_use_counterclockwise_charge_orientation() -> None:
    charge = _pair_charge()
    canonical = canonical_vortex_field(charge)
    link_x, link_y = phase_links(canonical.field)
    curl = plaquette_curl(link_x, link_y)

    assert curl[8, 9] == pytest.approx(2.0 * np.pi, abs=1e-10)
    assert curl[13, 12] == pytest.approx(-2.0 * np.pi, abs=1e-10)


def test_global_gauge_and_positive_magnitude_leave_charge_unchanged() -> None:
    charge = _pair_charge()
    canonical = canonical_vortex_field(charge).field
    x, y = np.indices(charge.shape)
    magnitude = 0.8 + 0.1 * np.cos(2.0 * np.pi * x / charge.shape[0])
    transformed = magnitude * canonical * np.exp(0.73j)

    assert np.array_equal(extract_charge(transformed).charge, charge)
    assert np.allclose(unit_field(transformed), canonical * np.exp(0.73j), atol=1e-12)


def test_invalid_periodic_charge_and_near_zero_phase_fail_closed() -> None:
    nonneutral = _pair_charge()
    nonneutral[13, 12] = 0
    with pytest.raises(ValueError, match="zero total charge"):
        canonical_vortex_field(nonneutral)

    field = np.ones((8, 8), dtype=np.complex128)
    field[2, 3] = 0.0
    with pytest.raises(ValueError, match="near-zero magnitude"):
        extract_charge(field)
