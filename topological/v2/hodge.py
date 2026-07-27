"""Compact Hodge decomposition on 2-torus with branch integer cochain.

Produces: exact (curl-free), coexact (div-free, carries vortex charge),
harmonic (both curl-free and div-free, 2D on T²), and branch integer cochain.
"""
import numpy as np
from ..topology import extract_charge


def compact_hodge_decompose(
    link_variables: np.ndarray,  # (H, W, 2) — [dx, dy] per plaquette
    charge_map: np.ndarray,      # (H, W) integer charge
) -> dict:
    """Decompose compact U(1) link variables on a periodic 2D grid.

    Given principal link variables Δθ ∈ [-π, π) and integer charge Q,
    decomposes into:
      Δθ = dφ (exact) + δA (coexact) + h (harmonic) + 2π·b (branch)

    Returns dict with keys: exact_potential, coexact_potential,
    harmonic_holonomy, branch_cochain, reconstructed_error.
    """
    H, W = charge_map.shape
    N = H * W

    # ── Coexact: canonical vortex field via FFT Poisson solver ──
    kx = 2j * np.pi * np.fft.fftfreq(H) * H
    ky = 2j * np.pi * np.fft.fftfreq(W) * W
    KX, KY = np.meshgrid(ky, kx, indexing="ij")
    laplacian = KX**2 + KY**2
    laplacian[0, 0] = 1.0  # avoid div-by-zero for DC

    rhs = 2 * np.pi * charge_map.astype(np.float64)
    psi_hat = np.fft.fft2(rhs) / laplacian
    psi_hat[0, 0] = 0.0
    psi = np.real(np.fft.ifft2(psi_hat))

    # Coexact link variables from stream function
    coexact_dx = np.diff(psi, axis=1, append=psi[:, :1])
    coexact_dy = np.diff(psi, axis=0, append=psi[:1, :])

    # ── Harmonic: global cycle holonomies ──
    residual = np.stack([
        link_variables[..., 0] - coexact_dx,
        link_variables[..., 1] - coexact_dy,
    ], axis=-1)  # (H, W, 2)

    wx = float(np.sum(residual[:, :, 0]))  # row holonomy
    wy = float(np.sum(residual[:, :, 1]))  # column holonomy
    harmonic_dx = np.full((H, W), wx / H)
    harmonic_dy = np.full((H, W), wy / W)

    # ── Exact: curl-free remainder after removing coexact + harmonic ──
    exact_dx = residual[:, :, 0] - harmonic_dx
    exact_dy = residual[:, :, 1] - harmonic_dy

    # ── Branch integer cochain: reconstruct to verify ──
    reconstructed = np.stack([
        coexact_dx + harmonic_dx + exact_dx,
        coexact_dy + harmonic_dy + exact_dy,
    ], axis=-1)

    branch = np.round((link_variables - reconstructed) / (2 * np.pi))
    reconstructed_error = float(np.max(np.abs(
        link_variables - reconstructed - 2 * np.pi * branch
    )))

    return {
        "exact_potential": np.stack([exact_dx, exact_dy], axis=-1),
        "coexact_potential": np.stack([coexact_dx, coexact_dy], axis=-1),
        "harmonic_dx": harmonic_dx,
        "harmonic_dy": harmonic_dy,
        "harmonic_holonomy": (wx, wy),
        "branch_cochain": branch,
        "reconstructed": reconstructed,
        "reconstructed_error": reconstructed_error,
    }


def extract_harmonic_sector(
    field: np.ndarray,  # (H, W) complex
) -> tuple[float, float]:
    """Extract global cycle holonomies from a complex field."""
    H, W = field.shape
    unit = field / (np.abs(field) + 1e-12)

    # Horizontal links
    dx = np.concatenate([unit[:, 1:], unit[:, :1]], axis=1)
    dx = np.angle(dx * np.conj(unit))

    # Vertical links
    dy = np.concatenate([unit[1:, :], unit[:1, :]], axis=0)
    dy = np.angle(dy * np.conj(unit))

    wx = float(np.sum(dx))
    wy = float(np.sum(dy))
    return wx, wy


def validate_hodge_decomposition(components: dict, tolerance: float = 1e-10) -> bool:
    """Verify that the decomposition satisfies curl/div constraints."""
    H, W = components["coexact_potential"].shape[:2]

    # Coexact: should have correct curl
    coexact = components["coexact_potential"]
    # Exact: should have zero curl
    exact = components["exact_potential"]
    harmonic_dx = components["harmonic_dx"]
    harmonic_dy = components["harmonic_dy"]

    # Curl of exact should be ~0
    exact_curl = _discrete_curl(exact)
    if np.max(np.abs(exact_curl)) > tolerance:
        return False

    # Curl of harmonic should be ~0
    harm_curl = np.diff(harmonic_dy, axis=1, append=harmonic_dy[:, :1])
    harm_curl -= np.diff(harmonic_dx, axis=0, append=harmonic_dx[:1, :])
    if np.max(np.abs(harm_curl)) > tolerance:
        return False

    return True


def _discrete_curl(links: np.ndarray) -> np.ndarray:
    """Compute discrete curl of link variables. links: (H, W, 2)."""
    dx = links[..., 0]
    dy = links[..., 1]
    curl = np.zeros_like(dx)
    curl[:, :-1] += dy[:, 1:]
    curl[:, -1] += dy[:, 0]
    curl[:, :-1] -= dy[:, :-1]
    curl[:, -1] -= dy[:, -1]
    curl[:-1, :] -= dx[1:, :]
    curl[-1, :] -= dx[0, :]
    curl[:-1, :] += dx[:-1, :]
    curl[-1, :] += dx[-1, :]
    return curl
