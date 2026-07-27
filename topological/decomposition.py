"""Causal component operations for periodic compact U(1) fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology import ChargeExtraction, canonical_vortex_field, extract_charge, phase_links


@dataclass(frozen=True)
class CompactDecomposition:
    magnitude: np.ndarray
    vortex: np.ndarray
    smooth: np.ndarray
    charge: ChargeExtraction
    reconstruction_error: float
    smooth_charge_residual: float


def recompose(
    magnitude: np.ndarray,
    vortex: np.ndarray,
    smooth: np.ndarray,
) -> np.ndarray:
    radius = np.asarray(magnitude, dtype=np.float64)
    vortex_unit = np.asarray(vortex, dtype=np.complex128)
    smooth_unit = np.asarray(smooth, dtype=np.complex128)
    if radius.shape != vortex_unit.shape or radius.shape != smooth_unit.shape:
        raise ValueError("component shapes must match")
    if np.any(radius <= 1e-12):
        raise ValueError("recomposition requires positive magnitude")
    vortex_norm = np.abs(vortex_unit)
    smooth_norm = np.abs(smooth_unit)
    if not np.allclose(vortex_norm, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("vortex component must be unit magnitude")
    if not np.allclose(smooth_norm, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("smooth component must be unit magnitude")
    return radius * vortex_unit * smooth_unit


def decompose(field: np.ndarray) -> CompactDecomposition:
    observed = np.asarray(field, dtype=np.complex128)
    magnitude = np.abs(observed)
    charge = extract_charge(observed)
    canonical = canonical_vortex_field(charge.charge)
    compact = observed / magnitude
    smooth = compact * np.conjugate(canonical.field)
    smooth = smooth / np.abs(smooth)
    reconstruction = recompose(magnitude, canonical.field, smooth)
    reconstruction_error = float(np.max(np.abs(reconstruction - observed)))
    smooth_charge = extract_charge(smooth)
    smooth_charge_residual = float(
        max(smooth_charge.residual_max, np.max(np.abs(smooth_charge.charge)))
    )
    return CompactDecomposition(
        magnitude=magnitude,
        vortex=canonical.field,
        smooth=smooth,
        charge=charge,
        reconstruction_error=reconstruction_error,
        smooth_charge_residual=smooth_charge_residual,
    )


def transplant_vortex(recipient: CompactDecomposition, donor: CompactDecomposition) -> np.ndarray:
    return recompose(recipient.magnitude, donor.vortex, recipient.smooth)


def transplant_smooth(recipient: CompactDecomposition, donor: CompactDecomposition) -> np.ndarray:
    return recompose(recipient.magnitude, recipient.vortex, donor.smooth)


def transplant_magnitude(recipient: CompactDecomposition, donor: CompactDecomposition) -> np.ndarray:
    return recompose(donor.magnitude, recipient.vortex, recipient.smooth)


def align_global_phase(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    first = np.asarray(candidate, dtype=np.complex128)
    second = np.asarray(reference, dtype=np.complex128)
    if first.shape != second.shape:
        raise ValueError("phase alignment requires matching fields")
    overlap = np.vdot(first.ravel(), second.ravel())
    if abs(overlap) <= 1e-15:
        raise ValueError("global phase is undefined for orthogonal fields")
    return first * np.exp(1j * np.angle(overlap))


def fourier_decompose(
    field: np.ndarray,
    cutoff_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a complex scalar field into low- and high-pass Fourier components.

    Returns (low_pass, high_pass, full_fft) where
    low_pass + high_pass ≈ field (up to numerical FFT roundtrip error).
    """
    observed = np.asarray(field, dtype=np.complex128)
    height, width = observed.shape
    fft = np.fft.fft2(observed)
    fx = np.fft.fftfreq(height) * height
    fy = np.fft.fftfreq(width) * width
    radius = np.sqrt(fx[:, None] ** 2 + fy[None, :] ** 2)
    mask = radius <= cutoff_radius
    low = np.fft.ifft2(fft * mask)
    high = np.fft.ifft2(fft * (~mask))
    return low, high, fft


def fourier_compose(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray(low, dtype=np.complex128) + np.asarray(high, dtype=np.complex128)


def gradient_energy(field: np.ndarray) -> float:
    link_x, link_y = phase_links(field)
    return float(np.mean(link_x**2 + link_y**2))


def radial_link_spectrum(field: np.ndarray) -> np.ndarray:
    link_x, link_y = phase_links(field)
    height, width = link_x.shape
    frequency_x = np.fft.fftfreq(height) * height
    frequency_y = np.fft.fftfreq(width) * width
    radius = np.floor(
        np.sqrt(frequency_x[:, None] ** 2 + frequency_y[None, :] ** 2)
    ).astype(np.int64)
    power = np.abs(np.fft.fft2(link_x)) ** 2 + np.abs(np.fft.fft2(link_y)) ** 2
    spectrum = np.bincount(radius.ravel(), weights=power.ravel()).astype(np.float64)
    spectrum[0] = 0.0
    total = float(spectrum.sum())
    if total <= 1e-15:
        raise ValueError("radial spectrum is undefined for a constant field")
    return spectrum / total


def normalized_spectrum_error(first: np.ndarray, second: np.ndarray) -> float:
    left = radial_link_spectrum(first)
    right = radial_link_spectrum(second)
    size = max(len(left), len(right))
    left = np.pad(left, (0, size - len(left)))
    right = np.pad(right, (0, size - len(right)))
    return float(0.5 * np.sum(np.abs(left - right)))
