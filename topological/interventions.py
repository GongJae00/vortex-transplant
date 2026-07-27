"""Channelwise compact-state interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .decomposition import (
    decompose,
    fourier_compose,
    fourier_decompose,
    gradient_energy,
    normalized_spectrum_error,
    recompose,
)
from .topology import extract_charge


@dataclass(frozen=True)
class HiddenComponents:
    magnitude: np.ndarray
    vortex: np.ndarray
    smooth: np.ndarray
    charge: np.ndarray
    reconstruction_error: float
    integer_residual: float
    smooth_charge_residual: float
    minimum_magnitude: float


@dataclass(frozen=True)
class MatchedControl:
    field: np.ndarray
    scale: float
    displacement_error: float


def hidden_to_complex(hidden: torch.Tensor) -> np.ndarray:
    if hidden.ndim != 4 or hidden.shape[0] != 2:
        raise ValueError("hidden state must have shape (2, channels, height, width)")
    array = hidden.detach().to(device="cpu", dtype=torch.float64).numpy()
    return array[0] + 1j * array[1]


def complex_to_hidden(
    field: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    array = np.asarray(field, dtype=np.complex128)
    if array.ndim != 3:
        raise ValueError("complex hidden field must have shape (channels, height, width)")
    planes = np.stack((array.real, array.imag), axis=0)
    return torch.from_numpy(planes).to(device=device, dtype=dtype)


def decompose_hidden(
    hidden: torch.Tensor,
    *,
    magnitude_epsilon: float = 1e-6,
    tolerance: float = 1e-5,
) -> HiddenComponents:
    field = hidden_to_complex(hidden)
    channel_errors: list[str] = []
    for channel_index, channel in enumerate(field):
        m = float(np.min(np.abs(channel)))
        if m <= magnitude_epsilon:
            channel_errors.append(f"channel {channel_index}: minimum magnitude {m:.2e} <= {magnitude_epsilon:.0e}")
    if channel_errors:
        raise ValueError("hidden decomposition failed: " + "; ".join(channel_errors))
    parts = [decompose(channel) for channel in field]
    reconstruction_error = max(part.reconstruction_error for part in parts)
    integer_residual = max(part.charge.residual_max for part in parts)
    smooth_charge_residual = max(part.smooth_charge_residual for part in parts)
    errors: list[str] = []
    if reconstruction_error > tolerance:
        errors.append(f"reconstruction_error {reconstruction_error:.2e} > {tolerance:.0e}")
    if integer_residual > tolerance:
        errors.append(f"integer_residual {integer_residual:.2e} > {tolerance:.0e}")
    bad_channels = [
        f"channel {i}" for i, part in enumerate(parts) if part.charge.net_charge != 0
    ]
    if bad_channels:
        errors.append(f"nonzero net charge in {', '.join(bad_channels)}")
    if errors:
        raise ValueError("hidden decomposition exceeds tolerance: " + "; ".join(errors))
    return HiddenComponents(
        magnitude=np.stack([part.magnitude for part in parts]),
        vortex=np.stack([part.vortex for part in parts]),
        smooth=np.stack([part.smooth for part in parts]),
        charge=np.stack([part.charge.charge for part in parts]),
        reconstruction_error=float(reconstruction_error),
        integer_residual=float(integer_residual),
        smooth_charge_residual=float(smooth_charge_residual),
        minimum_magnitude=float(np.min(np.abs(field))),
    )


def compose_components(
    magnitude: np.ndarray,
    vortex: np.ndarray,
    smooth: np.ndarray,
) -> np.ndarray:
    if magnitude.ndim != 3:
        raise ValueError("components must have three dimensions")
    return np.stack(
        [recompose(magnitude[index], vortex[index], smooth[index]) for index in range(len(magnitude))]
    )


def component_intervention(
    recipient: HiddenComponents,
    donor: HiddenComponents,
    arm: str,
) -> np.ndarray:
    if recipient.magnitude.shape != donor.magnitude.shape:
        raise ValueError("donor and recipient state shapes must match")
    if arm == "natural_recipient":
        return compose_components(recipient.magnitude, recipient.vortex, recipient.smooth)
    if arm == "vortex":
        return compose_components(recipient.magnitude, donor.vortex, recipient.smooth)
    if arm == "smooth":
        return compose_components(recipient.magnitude, recipient.vortex, donor.smooth)
    if arm == "magnitude":
        return compose_components(donor.magnitude, recipient.vortex, recipient.smooth)
    if arm == "whole_phase":
        return compose_components(recipient.magnitude, donor.vortex, donor.smooth)
    if arm == "whole_state":
        return compose_components(donor.magnitude, donor.vortex, donor.smooth)
    raise ValueError(f"unknown intervention arm: {arm}")


def state_displacement(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.shape != right.shape:
        raise ValueError("state displacement requires matching shapes")
    return float(np.linalg.norm((left - right).ravel()))


def _bisect_match(
    original: np.ndarray,
    phase_base: np.ndarray,
    target: float,
    upper: float,
    *,
    iterations: int = 80,
) -> MatchedControl:
    if target < 0.0 or upper <= 0.0:
        raise ValueError("displacement target and bisection bracket are invalid")
    maximum = original * np.exp(1j * upper * phase_base)
    if state_displacement(original, maximum) + 1e-10 < target:
        raise ValueError("matched control cannot bracket the target displacement")
    lower = 0.0
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        candidate = original * np.exp(1j * midpoint * phase_base)
        if state_displacement(original, candidate) < target:
            lower = midpoint
        else:
            upper = midpoint
    scale = 0.5 * (lower + upper)
    field = original * np.exp(1j * scale * phase_base)
    return MatchedControl(
        field=field,
        scale=float(scale),
        displacement_error=abs(state_displacement(original, field) - target),
    )


def matched_global_phase(original: np.ndarray, target: float) -> MatchedControl:
    """Uniform global phase rotation: same phase shift at every spatial site.

    Controls for non-specific effect of global phase displacement.
    Perturbation is a constant complex phase exp(i * scale) applied uniformly,
    which preserves all charge structure but changes every phase equally.
    """
    phase_base = np.ones_like(np.asarray(original, dtype=np.complex128).real)
    return _bisect_match(np.asarray(original, dtype=np.complex128), phase_base, target, math.pi)


def _zero_charge_phase_base(
    shape: tuple[int, int, int],
    control_index: int,
    epsilon: float,
) -> np.ndarray:
    channels, height, width = shape
    if min(height, width) < 4:
        raise ValueError("zero-charge control requires spatial size at least four")
    generator = np.random.default_rng(int(control_index) % (2**63 - 1))
    signs = generator.choice(np.array([-1.0, 1.0]), size=(channels, height, 1))
    signs[:, 0, 0] = -1.0
    signs[:, 1, 0] = 1.0
    noise = np.broadcast_to(signs, (channels, height, width)).copy()
    return 1.0 + epsilon * noise


ZERO_CHARGE_EPSILONS = (0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)


def matched_zero_charge_phase(
    original: np.ndarray,
    target: float,
    control_index: int,
    *,
    epsilons: tuple[float, ...] = ZERO_CHARGE_EPSILONS,
) -> MatchedControl:
    """Spatially structured zero-charge phase perturbation.

    In contrast to matched_global_phase (uniform rotation across all sites),
    this applies a sign-structured spatial pattern (±1 per row) scaled by
    epsilon, resulting in a spatially varying phase that carries zero total
    topological charge.  This controls for structured-nonuniform phase
    displacement without introducing new vortex defects.
    """
    observed = np.asarray(original, dtype=np.complex128)
    for epsilon in epsilons:
        base = _zero_charge_phase_base(observed.shape, control_index, epsilon)
        upper = math.pi / (1.0 + epsilon)
        try:
            control = _bisect_match(observed, base, target, upper)
        except ValueError:
            continue
        perturbation = np.exp(1j * control.scale * base)
        perturbation_charge = np.stack(
            [extract_charge(channel).charge for channel in perturbation]
        )
        if not np.any(perturbation_charge):
            return control
    raise ValueError("zero-charge control cannot match the target displacement")


def fourier_field_intervention(
    recipient: np.ndarray,
    donor: np.ndarray,
    arm: str,
    cutoff_radius: float = 2.0,
) -> np.ndarray:
    """Replace low- or high-frequency Fourier components between fields."""
    observed_recipient = np.asarray(recipient, dtype=np.complex128)
    observed_donor = np.asarray(donor, dtype=np.complex128)
    result = np.empty_like(observed_recipient)
    for channel in range(observed_recipient.shape[0]):
        r_low, r_high, _ = fourier_decompose(observed_recipient[channel], cutoff_radius)
        d_low, d_high, _ = fourier_decompose(observed_donor[channel], cutoff_radius)
        if arm == "fourier_low":
            result[channel] = fourier_compose(d_low, r_high)
        elif arm == "fourier_high":
            result[channel] = fourier_compose(r_low, d_high)
        else:
            raise ValueError(f"unknown Fourier intervention arm: {arm}")
    return result


def aggregate_gradient_energy(field: np.ndarray) -> float:
    return float(np.mean([gradient_energy(channel) for channel in np.asarray(field)]))


def aggregate_spectrum_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            [
                normalized_spectrum_error(left, right)
                for left, right in zip(np.asarray(first), np.asarray(second), strict=True)
            ]
        )
    )


@dataclass(frozen=True)
class PCADecomposition:
    components: np.ndarray
    mean: np.ndarray
    explained_variance_ratio: np.ndarray


def fit_pca(hidden_states: list[np.ndarray], k: int) -> PCADecomposition:
    fields = [np.asarray(h, dtype=np.complex128) for h in hidden_states]
    flat = np.stack(
        [np.concatenate([f.real.ravel().astype(np.float64), f.imag.ravel().astype(np.float64)])
         for f in fields]
    )
    mean = flat.mean(axis=0)
    centered = flat - mean
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:k]
    explained_var = (s[:k] ** 2) / (s**2).sum()
    return PCADecomposition(
        components=components, mean=mean, explained_variance_ratio=explained_var,
    )


def pca_field_intervention(
    recipient: np.ndarray,
    donor: np.ndarray,
    pca: PCADecomposition,
) -> np.ndarray:
    def _to_real(f: np.ndarray) -> np.ndarray:
        return np.concatenate([f.real.ravel().astype(np.float64), f.imag.ravel().astype(np.float64)])

    def _from_real(vec: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        half = vec.size // 2
        return (vec[:half] + 1j * vec[half:]).reshape(shape)

    r = np.asarray(recipient, dtype=np.complex128)
    d = np.asarray(donor, dtype=np.complex128)
    r_flat = _to_real(r) - pca.mean
    d_flat = _to_real(d) - pca.mean
    r_proj = pca.components @ r_flat
    d_proj = pca.components @ d_flat
    corrected = r_flat - pca.components.T @ r_proj + pca.components.T @ d_proj
    return _from_real(corrected + pca.mean, r.shape)


def random_direction_intervention(
    recipient: np.ndarray,
    donor: np.ndarray,
    *,
    seed: int = 0,
    target_norm: float | None = None,
) -> np.ndarray:
    """Replace diff vector with random direction of matched norm.

    If any perturbation of the same magnitude reproduces the causal effect,
    then the specific vortex structure is not special. This baseline tests
    exactly that: generate a random complex field with Frobenius norm equal
    to ``target_norm`` (defaults to ||donor - recipient||), add it to the
    recipient, and check whether the continuation still shifts toward donor.

    For a fair comparison against the vortex arm, callers should pass
    ``target_norm = state_displacement(recipient, vortex_field)`` so the
    random perturbation matches the vortex intervention's displacement.
    """
    r = np.asarray(recipient, dtype=np.complex128)
    d = np.asarray(donor, dtype=np.complex128)
    diff = d - r
    if target_norm is None:
        target_norm = float(np.linalg.norm(diff.ravel()))
    if target_norm <= 1e-12:
        return r.copy()
    rng = np.random.default_rng(seed)
    noise = (
        rng.standard_normal(diff.shape, dtype=np.float64)
        + 1j * rng.standard_normal(diff.shape, dtype=np.float64)
    )
    noise_norm = float(np.linalg.norm(noise.ravel()))
    return r + noise * (target_norm / (noise_norm + 1e-12))
