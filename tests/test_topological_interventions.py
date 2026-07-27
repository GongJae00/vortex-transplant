from __future__ import annotations

import numpy as np
import torch

from topological.interventions import (
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
from topological.topology import canonical_vortex_field, extract_charge


def _field(shift: int) -> np.ndarray:
    charge = np.zeros((8, 8), dtype=np.int64)
    charge[(1 + shift) % 8, 2] = 1
    charge[(5 + shift) % 8, 6] = -1
    x, y = np.indices((8, 8))
    channels = []
    for channel in range(2):
        magnitude = 1.0 + 0.05 * np.cos(2.0 * np.pi * (x + y) / 8 + channel)
        smooth = np.exp(0.1j * np.sin(2.0 * np.pi * x / 8 + channel))
        channels.append(magnitude * canonical_vortex_field(charge).field * smooth)
    return np.stack(channels)


def test_hidden_conversion_and_self_recomposition_are_identity() -> None:
    field = _field(0)
    hidden = complex_to_hidden(field)
    parts = decompose_hidden(hidden)
    reconstructed = component_intervention(parts, parts, "natural_recipient")

    assert np.max(np.abs(hidden_to_complex(hidden) - field)) <= 1e-7
    assert np.max(np.abs(reconstructed - field)) <= 1e-6
    assert parts.smooth_charge_residual <= 1e-5


def test_component_interventions_preserve_only_the_declared_recipient_parts() -> None:
    recipient = decompose_hidden(complex_to_hidden(_field(0)))
    donor = decompose_hidden(complex_to_hidden(_field(2)))

    vortex = decompose_hidden(complex_to_hidden(component_intervention(recipient, donor, "vortex")))
    smooth = decompose_hidden(complex_to_hidden(component_intervention(recipient, donor, "smooth")))
    magnitude = decompose_hidden(complex_to_hidden(component_intervention(recipient, donor, "magnitude")))

    assert np.array_equal(vortex.charge, donor.charge)
    assert np.max(np.abs(vortex.magnitude - recipient.magnitude)) <= 1e-6
    assert np.array_equal(smooth.charge, recipient.charge)
    assert np.max(np.abs(magnitude.magnitude - donor.magnitude)) <= 1e-6


def test_matched_controls_reach_target_and_preserve_charge() -> None:
    recipient = _field(0)
    donor = _field(2)
    target = state_displacement(recipient, donor) * 0.5

    global_control = matched_global_phase(recipient, target)
    smooth_control = matched_zero_charge_phase(recipient, target, control_index=11)

    assert global_control.displacement_error <= 1e-10
    assert smooth_control.displacement_error <= 1e-10
    phase_ratio = smooth_control.field / recipient
    assert np.max(np.abs(phase_ratio - phase_ratio[:, :1, :1])) > 1e-4
    for perturbation_channel in phase_ratio:
        assert not np.any(extract_charge(perturbation_channel).charge)


def test_invalid_near_zero_hidden_state_is_rejected() -> None:
    hidden = torch.zeros((2, 2, 8, 8))

    with np.testing.assert_raises(ValueError):
        decompose_hidden(hidden)


def test_decompose_hidden_error_message_identifies_failing_channel() -> None:
    hidden = torch.ones((2, 2, 8, 8))
    hidden[:, 0, :, :] = 0.0
    try:
        decompose_hidden(hidden)
    except ValueError as error:
        message = str(error)
        assert "channel 0" in message


def test_fourier_field_intervention_preserves_shapes() -> None:
    field = _field(0)
    donor = _field(2)

    low_arm = fourier_field_intervention(field, donor, "fourier_low")
    high_arm = fourier_field_intervention(field, donor, "fourier_high")

    assert low_arm.shape == field.shape
    assert high_arm.shape == field.shape
    assert np.isfinite(low_arm).all()
    assert np.isfinite(high_arm).all()


def test_pca_intervention_preserves_shape_and_is_finite() -> None:
    fields = [_field(s) for s in range(5)]
    pca = fit_pca(fields, k=4)

    assert pca.components.shape[0] == 4
    assert sum(pca.explained_variance_ratio) <= 1.0

    result = pca_field_intervention(fields[0], fields[2], pca)
    assert result.shape == fields[0].shape
    assert np.isfinite(result).all()


def test_pca_intervention_moves_recipient_toward_donor() -> None:
    fields = [_field(s) for s in range(5)]
    pca = fit_pca(fields, k=2)

    result = pca_field_intervention(fields[0], fields[2], pca)
    distance = float(np.linalg.norm((result - fields[2]).ravel()))

    assert distance < float(np.linalg.norm((fields[0] - fields[2]).ravel()))


def test_random_direction_intervention_preserves_shape() -> None:
    field = _field(0)
    donor = _field(3)

    result = random_direction_intervention(field, donor, seed=42)

    assert result.shape == field.shape
    assert np.isfinite(result).all()
    assert not np.array_equal(result, field)
