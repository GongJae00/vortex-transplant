"""Same-charge representative sampling.

Given a charge map Q, generates multiple complex fields v_Q · g
where g is a zero-local-charge multiplicative factor (smooth, charge-free).
Tests whether intervention effects are invariant to the specific
representative choice (gauge/section choice of the same topology class).
"""
import numpy as np
from .topology import canonical_vortex_field, extract_charge
from .types import RepresentativeSpec, RepresentativeSample


def sample_representatives(
    charge_map: np.ndarray,     # (H, W) integer
    spec: RepresentativeSpec | None = None,
    reference_field: np.ndarray | None = None,  # (C, H, W) complex, for matching harmonic
) -> RepresentativeSample:
    """Generate multiple same-charge fields with different smooth components.

    Each representative has the identical charge map Q but a different
    zero-charge smooth multiplicative factor. This allows testing whether
    the behavioral effect of a vortex transplant depends on the specific
    representative choice.

    Methods:
    - "harmonic_random": random harmonic sector (vary global winding)
    - "fourier_random": random band-limited smooth phase
    - "combined": random harmonic + random smooth
    """
    if spec is None:
        spec = RepresentativeSpec()

    H, W = charge_map.shape
    rng = np.random.default_rng(spec.seed_offset)

    # Base canonical vortex
    base_vortex = canonical_vortex_field(charge_map).field

    fields = []
    harmonic_sectors = []

    for i in range(spec.n_representatives):
        seed_i = spec.seed_offset + i

        if spec.method == "harmonic_random":
            # Add random harmonic winding
            wx = rng.uniform(-np.pi, np.pi)
            wy = rng.uniform(-np.pi, np.pi)
            # Construct harmonic phase gradient
            x = np.arange(H)[:, np.newaxis]
            y = np.arange(W)[np.newaxis, :]
            harmonic_phase = (wx * x / H + wy * y / W)
            rep = base_vortex * np.exp(1j * harmonic_phase)

        elif spec.method == "fourier_random":
            # Add random band-limited smooth phase
            smooth_phase = _generate_smooth_phase(H, W, rng)
            rep = base_vortex * np.exp(1j * smooth_phase)

        elif spec.method == "combined":
            wx = rng.uniform(-np.pi, np.pi)
            wy = rng.uniform(-np.pi, np.pi)
            x = np.arange(H)[:, np.newaxis]
            y = np.arange(W)[np.newaxis, :]
            harmonic_phase = (wx * x / H + wy * y / W)
            smooth_phase = _generate_smooth_phase(H, W, rng)
            rep = base_vortex * np.exp(1j * (harmonic_phase + smooth_phase))

        else:
            raise ValueError(f"Unknown representative method: {spec.method}")

        # Verify charge is unchanged
        charge_check = extract_charge(rep)
        if not np.array_equal(charge_check.charge, charge_map):
            # Repair: enforce exact charge
            rep = _enforce_charge(rep, charge_map)

        fields.append(rep)
        harmonic_sectors.append(_extract_harmonic(rep))

    # Compute representative variance metrics
    field_stack = np.stack([f.ravel() for f in fields])
    displacement_variance = float(np.var([np.linalg.norm(f - fields[0]) for f in fields]))

    energies = [_gradient_energy(f) for f in fields]
    energy_variance = float(np.var(energies))

    spectra = [np.mean(np.abs(np.fft.fft2(f))**2) for f in fields]
    spectrum_variance = float(np.var(spectra))

    return RepresentativeSample(
        fields=fields,
        charge_map=charge_map,
        harmonic_sectors=harmonic_sectors,
        displacement_variance=displacement_variance,
        energy_variance=energy_variance,
        spectrum_variance=spectrum_variance,
    )


def representative_variance_decomposition(
    transplant_margins: list[float],  # margin per representative
) -> dict:
    """Decompose variance into charge vs representative components.

    Returns: total_variance, within_class_variance, charge_effect_variance.
    within_class_variance is the variance across same-charge representatives.
    """
    margins = np.array(transplant_margins)
    total_var = float(np.var(margins))
    # The variance within the same charge class is the total variance
    # (all margins are from the same charge, different reps)
    within_class_var = total_var
    return {
        "total_variance": total_var,
        "within_class_variance": within_class_var,
        "representative_variance_fraction": 1.0,  # all variance is rep variance here
        "n_representatives": len(transplant_margins),
        "mean_margin": float(np.mean(margins)),
        "std_margin": float(np.std(margins)),
    }



def _generate_smooth_phase(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    """Generate band-limited smooth phase field."""
    # Random low-frequency Fourier modes
    k_cutoff = min(H, W) // 4
    phase_hat = np.zeros((H, W), dtype=np.complex128)
    for kx in range(-k_cutoff, k_cutoff + 1):
        for ky in range(-k_cutoff, k_cutoff + 1):
            if kx == 0 and ky == 0:
                continue
            if kx**2 + ky**2 <= k_cutoff**2:
                idx_x = kx % H
                idx_y = ky % W
                phase_hat[idx_x, idx_y] = rng.normal(0, 0.1) + 1j * rng.normal(0, 0.1)
    phase = np.real(np.fft.ifft2(phase_hat))
    # Scale to moderate amplitude
    phase = phase / (np.std(phase) + 1e-12) * 0.1
    return phase


def _extract_harmonic(field: np.ndarray) -> tuple[float, float]:
    """Extract global cycle holonomies."""
    H, W = field.shape
    unit = field / (np.abs(field) + 1e-12)
    dx = np.concatenate([unit[:, 1:], unit[:, :1]], axis=1)
    dy = np.concatenate([unit[1:, :], unit[:1, :]], axis=0)
    wx = float(np.sum(np.angle(dx * np.conj(unit))))
    wy = float(np.sum(np.angle(dy * np.conj(unit))))
    return wx, wy


def _gradient_energy(field: np.ndarray) -> float:
    unit = field / (np.abs(field) + 1e-12)
    dx = np.concatenate([unit[:, 1:], unit[:, :1]], axis=1)
    dy = np.concatenate([unit[1:, :], unit[:1, :]], axis=0)
    return float(np.mean(np.angle(dx * np.conj(unit))**2 + np.angle(dy * np.conj(unit))**2))


def _enforce_charge(field: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reconstruct field to have exact target charge map."""
    H, W = field.shape
    target_vortex = canonical_vortex_field(target).field
    # Extract unit phase and replace vortex component
    unit = field / (np.abs(field) + 1e-12)
    # Remove existing vortex, add target vortex
    current_charge = extract_charge(field).charge
    current_vortex = canonical_vortex_field(current_charge).field
    smooth = unit / (current_vortex + 1e-12)
    return np.abs(field) * target_vortex * smooth
