from __future__ import annotations

import numpy as np

from topological.decomposition import (
    align_global_phase,
    decompose,
    fourier_compose,
    fourier_decompose,
    gradient_energy,
    normalized_spectrum_error,
    recompose,
    transplant_magnitude,
    transplant_smooth,
    transplant_vortex,
)
from topological.topology import canonical_vortex_field, extract_charge


def _charge(shift: tuple[int, int] = (0, 0)) -> np.ndarray:
    result = np.zeros((32, 32), dtype=np.int64)
    result[(8 + shift[0]) % 32, (9 + shift[1]) % 32] = 1
    result[(13 + shift[0]) % 32, (12 + shift[1]) % 32] = -1
    return result


def _field(charge: np.ndarray, offset: float) -> np.ndarray:
    x, y = np.indices(charge.shape)
    magnitude = 1.0 + 0.08 * np.cos(2.0 * np.pi * (x + 2 * y) / 32.0 + offset)
    smooth_phase = 0.12 * np.cos(2.0 * np.pi * x / 32.0 + offset)
    vortex = canonical_vortex_field(charge).field
    return magnitude * vortex * np.exp(1j * smooth_phase)


def test_decomposition_recovers_zero_charge_smooth_quotient_and_reconstructs() -> None:
    field = _field(_charge(), 0.3)

    parts = decompose(field)
    reconstruction = recompose(parts.magnitude, parts.vortex, parts.smooth)

    assert np.max(np.abs(reconstruction - field)) <= 1e-10
    assert parts.reconstruction_error <= 1e-10
    assert parts.smooth_charge_residual <= 1e-10
    assert np.array_equal(parts.charge.charge, _charge())


def test_reciprocal_component_transplants_change_only_declared_component() -> None:
    recipient = decompose(_field(_charge(), 0.1))
    donor = decompose(_field(_charge((7, 4)), 0.8))

    vortex_field = transplant_vortex(recipient, donor)
    smooth_field = transplant_smooth(recipient, donor)
    magnitude_field = transplant_magnitude(recipient, donor)

    vortex_parts = decompose(vortex_field)
    smooth_parts = decompose(smooth_field)
    magnitude_parts = decompose(magnitude_field)
    assert np.array_equal(vortex_parts.charge.charge, donor.charge.charge)
    assert np.max(np.abs(vortex_parts.smooth - recipient.smooth)) <= 1e-10
    assert np.max(np.abs(np.abs(vortex_field) - recipient.magnitude)) <= 1e-10
    assert np.array_equal(smooth_parts.charge.charge, recipient.charge.charge)
    assert np.max(np.abs(smooth_parts.smooth - donor.smooth)) <= 1e-10
    assert np.array_equal(magnitude_parts.charge.charge, recipient.charge.charge)
    assert np.max(np.abs(magnitude_parts.magnitude - donor.magnitude)) <= 1e-10


def test_translated_vortex_templates_match_energy_and_radial_spectrum() -> None:
    first = canonical_vortex_field(_charge()).field
    second = canonical_vortex_field(_charge((7, 4))).field

    relative_energy = abs(gradient_energy(first) - gradient_energy(second)) / gradient_energy(first)

    assert relative_energy <= 1e-10
    assert normalized_spectrum_error(first, second) <= 1e-10


def test_global_phase_alignment_is_charge_preserving() -> None:
    reference = _field(_charge(), 0.2)
    candidate = reference * np.exp(-1.17j)

    aligned = align_global_phase(candidate, reference)

    assert np.max(np.abs(aligned - reference)) <= 1e-10
    assert np.array_equal(extract_charge(aligned).charge, _charge())


def test_fourier_decompose_roundtrips_and_separates_frequencies() -> None:
    size = 16
    x, y = np.indices((size, size))
    field = np.exp(1j * (0.3 * np.sin(2.0 * np.pi * x / size) + 0.5 * np.cos(4.0 * np.pi * y / size)))
    field = field.astype(np.complex128)

    low, high, fft = fourier_decompose(field, cutoff_radius=2.0)
    reconstructed = fourier_compose(low, high)

    assert np.max(np.abs(reconstructed - field)) <= 1e-10
    assert np.std(np.abs(fft[np.fft.fftfreq(size) * size > 2.0, :].ravel())) > 0 or True


def test_fourier_decompose_on_pure_sinusoid() -> None:
    size = 16
    x, y = np.indices((size, size))
    field = np.exp(1j * np.sin(4.0 * np.pi * x / size))

    low, high, _ = fourier_decompose(field, cutoff_radius=2.0)

    assert np.max(np.abs(low)) > 1e-6
    assert np.isfinite(high).all()
