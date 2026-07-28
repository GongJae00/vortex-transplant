"""Compact U(1) topology primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TAU = 2.0 * np.pi


def wrap_phase(values: np.ndarray) -> np.ndarray:
    """Return principal phases in ``(-pi, pi]``."""

    array = np.asarray(values, dtype=np.float64)
    wrapped = np.angle(np.exp(1j * array))
    wrapped[wrapped <= -np.pi] = np.pi
    return wrapped


def unit_field(field: np.ndarray, magnitude_epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(field, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError("fields must be two-dimensional")
    magnitude = np.abs(array)
    if np.any(magnitude <= magnitude_epsilon):
        raise ValueError("compact phase is undefined at near-zero magnitude")
    return array / magnitude


def phase_links(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oriented periodic link phases: link_x along axis 0, link_y along axis 1."""

    compact = unit_field(field)
    link_x = np.angle(np.roll(compact, -1, axis=0) * np.conjugate(compact))
    link_y = np.angle(np.roll(compact, -1, axis=1) * np.conjugate(compact))
    return wrap_phase(link_x), wrap_phase(link_y)


def plaquette_curl(link_x: np.ndarray, link_y: np.ndarray) -> np.ndarray:
    x = np.asarray(link_x, dtype=np.float64)
    y = np.asarray(link_y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("link fields must be matching two-dimensional arrays")
    return x + np.roll(y, -1, axis=0) - np.roll(x, -1, axis=1) - y


@dataclass(frozen=True)
class ChargeExtraction:
    charge: np.ndarray
    residual_max: float
    net_charge: int
    positive_count: int
    negative_count: int


def extract_charge(field: np.ndarray, tolerance: float = 1e-10) -> ChargeExtraction:
    link_x, link_y = phase_links(field)
    curl = plaquette_curl(link_x, link_y)
    charge = np.rint(curl / TAU).astype(np.int64)
    residual = float(np.max(np.abs(curl - TAU * charge)))
    if residual > tolerance:
        raise RuntimeError(f"integer-charge residual exceeds tolerance: {residual}")
    return ChargeExtraction(
        charge=charge,
        residual_max=residual,
        net_charge=int(charge.sum()),
        positive_count=int(np.count_nonzero(charge > 0)),
        negative_count=int(np.count_nonzero(charge < 0)),
    )


def _poisson_stream(charge: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.asarray(charge, dtype=np.float64)
    if q.ndim != 2 or min(q.shape) < 4:
        raise ValueError("charge maps require a two-dimensional grid of size at least four")
    if not np.array_equal(q, np.rint(q)):
        raise ValueError("charge maps must be integer valued")
    if int(np.rint(q.sum())) != 0:
        raise ValueError("periodic charge maps must have zero total charge")
    height, width = q.shape
    frequency_x = TAU * np.fft.fftfreq(height)
    frequency_y = TAU * np.fft.fftfreq(width)
    eigenvalues = (
        2.0 * np.cos(frequency_x)[:, None]
        + 2.0 * np.cos(frequency_y)[None, :]
        - 4.0
    )
    right_hat = np.fft.fft2(TAU * q)
    stream_hat = np.zeros_like(right_hat, dtype=np.complex128)
    nonzero = np.abs(eigenvalues) > 1e-15
    stream_hat[nonzero] = right_hat[nonzero] / eigenvalues[nonzero]
    stream_hat[0, 0] = 0.0
    stream = np.fft.ifft2(stream_hat).real
    laplacian = (
        np.roll(stream, -1, axis=0)
        + np.roll(stream, 1, axis=0)
        + np.roll(stream, -1, axis=1)
        + np.roll(stream, 1, axis=1)
        - 4.0 * stream
    )
    residual = float(np.max(np.abs(laplacian - TAU * q)))
    return stream, residual


@dataclass(frozen=True)
class CanonicalVortexField:
    field: np.ndarray
    link_x: np.ndarray
    link_y: np.ndarray
    poisson_residual_max: float
    curl_residual_max: float
    holonomy_x: float
    holonomy_y: float
    integration_residual_max: float


def canonical_vortex_field(charge: np.ndarray) -> CanonicalVortexField:
    """Construct a canonical unit vertex field carrying a periodic charge map.

    Links are derived from the stream function psi solving Lap(psi) = 2*pi*q.
    Convention: link along axis 0 (x-direction) is backward diff of stream;
    link along axis 1 (y-direction) is backward diff of stream.  This differs
    from phase_links (which uses forward diff) but is self-consistent:
    plaquette_curl(link_along_axis1, link_along_axis0) recovers Lap(psi).
    """

    q = np.asarray(charge, dtype=np.int64)
    stream, poisson_residual = _poisson_stream(q)
    link_along_axis1 = np.roll(stream, 1, axis=1) - stream    # y-link (axis 1)
    link_along_axis0 = stream - np.roll(stream, 1, axis=0)    # x-link (axis 0)
    curl_residual = float(
        np.max(np.abs(plaquette_curl(link_along_axis1, link_along_axis0) - TAU * q))
    )

    height, width = q.shape
    holonomy_x = float(np.angle(np.exp(1j * np.sum(link_along_axis0[0, :]))))
    holonomy_y = float(np.angle(np.exp(1j * np.sum(link_along_axis1[:, 0]))))
    link_along_axis1 = link_along_axis1 - holonomy_y / height
    link_along_axis0 = link_along_axis0 - holonomy_x / width

    phasor_x = np.exp(1j * link_along_axis1)
    phasor_y = np.exp(1j * link_along_axis0)
    field = np.empty(q.shape, dtype=np.complex128)
    field[0, 0] = 1.0 + 0.0j
    for index_x in range(height - 1):
        field[index_x + 1, 0] = phasor_x[index_x, 0] * field[index_x, 0]
    for index_y in range(width - 1):
        field[:, index_y + 1] = phasor_y[:, index_y] * field[:, index_y]
    field = field / np.abs(field)

    reconstructed_x = np.roll(field, -1, axis=0) * np.conjugate(field)
    reconstructed_y = np.roll(field, -1, axis=1) * np.conjugate(field)
    integration_residual = float(
        max(
            np.max(np.abs(reconstructed_x - phasor_x)),
            np.max(np.abs(reconstructed_y - phasor_y)),
        )
    )
    final_holonomy_x = float(np.max(np.abs(np.prod(reconstructed_x, axis=0) - 1.0)))
    final_holonomy_y = float(np.max(np.abs(np.prod(reconstructed_y, axis=1) - 1.0)))
    return CanonicalVortexField(
        field=field,
        link_x=wrap_phase(link_along_axis0),
        link_y=wrap_phase(link_along_axis1),
        poisson_residual_max=poisson_residual,
        curl_residual_max=curl_residual,
        holonomy_x=final_holonomy_x,
        holonomy_y=final_holonomy_y,
        integration_residual_max=integration_residual,
    )



import numpy as np
from .types import BranchStability


def compute_branch_margins(field):
    """Compute branch margin statistics for a complex 2D field."""
    from .topology import extract_charge, phase_links
    H, W = field.shape
    unit = field / (np.abs(field) + 1e-12)
    dx, dy = phase_links(unit)
    dx_pad = np.concatenate([dx, dx[:, :1]], axis=1)
    dy_pad = np.concatenate([dy, dy[:1, :]], axis=0)
    all_margins = np.concatenate([(np.pi - np.abs(dx_pad)).ravel(), (np.pi - np.abs(dy_pad)).ravel()])
    charge = extract_charge(field)
    charge_positions = np.argwhere(np.abs(charge.charge) > 0)
    flip_radii = []
    if len(charge_positions) > 0:
        for px, py in charge_positions[:20]:
            radius = _measure_charge_flip_radius(field, px, py)
            flip_radii.append(radius)
    median_flip = float(np.median(flip_radii)) if flip_radii else 0.0
    return BranchStability(
        min_margin=float(np.min(all_margins)),
        q01_margin=float(np.quantile(all_margins, 0.01)),
        q05_margin=float(np.quantile(all_margins, 0.05)),
        median_margin=float(np.median(all_margins)),
        charge_flip_radius_median=median_flip,
    )


def compute_per_channel_branch_margins(field):
    """Compute branch margins for each channel in (C, H, W) field."""
    return [compute_branch_margins(field[ch]) for ch in range(field.shape[0])]


def _measure_charge_flip_radius(field, x, y, eps_values=None):
    if eps_values is None:
        eps_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    from .topology import extract_charge
    charge_ref = extract_charge(field)
    ref_val = charge_ref.charge[x, y]
    for eps in eps_values:
        for sign in [1.0, -1.0]:
            perturbed = field.copy()
            perturbed[x, y] *= np.exp(1j * sign * eps)
            if extract_charge(perturbed).charge[x, y] != ref_val:
                return eps
    return float("inf")


def defect_tracking(field_t, field_t1, max_displacement=2):
    """Track signed defects between consecutive time steps."""
    from .topology import extract_charge
    charge_t = extract_charge(field_t)
    charge_t1 = extract_charge(field_t1)
    H, W = charge_t.charge.shape
    defects_t = [(x, y, charge_t.charge[x, y]) for x in range(H) for y in range(W) if charge_t.charge[x, y] != 0]
    defects_t1 = [(x, y, charge_t1.charge[x, y]) for x in range(H) for y in range(W) if charge_t1.charge[x, y] != 0]
    matched = []
    unmatched_t1 = list(defects_t1)
    for dx, dy, dq in defects_t:
        best_dist, best_idx = float("inf"), -1
        for i, (dx1, dy1, dq1) in enumerate(unmatched_t1):
            if dq != dq1: continue
            dist = abs(dx - dx1) + abs(dy - dy1)
            dist = min(dist, abs(dx - dx1 - H), abs(dy - dy1 - W))
            if dist < best_dist and dist <= max_displacement:
                best_dist, best_idx = dist, i
        if best_idx >= 0:
            matched.append((dx, dy, unmatched_t1[best_idx][0], unmatched_t1[best_idx][1], dq))
            unmatched_t1.pop(best_idx)
    births = [(x, y, q) for x, y, q in unmatched_t1]
    deaths = [(dx, dy, dq) for dx, dy, dq in defects_t if not any(m[0] == dx and m[1] == dy for m in matched)]
    intersection = len(matched)
    union = len(defects_t) + len(defects_t1) - intersection
    return {"matched": matched, "births": births, "deaths": deaths, "signed_jaccard": intersection / max(union, 1),
            "n_defects_t": len(defects_t), "n_defects_t1": len(defects_t1)}


def per_channel_defect_prevalence(field):
    """Per-channel presence of both + and - defects."""
    from .topology import extract_charge
    result = []
    for ch in range(field.shape[0]):
        charge = extract_charge(field[ch])
        result.append(int(np.sum(charge.charge > 0)) > 0 and int(np.sum(charge.charge < 0)) > 0)
    return tuple(result)


def extract_charge_map(field):
    """Extract charge map summed across channels. (C,H,W) -> (H,W) integer."""
    from .topology import extract_charge
    C, H, W = field.shape
    total = np.zeros((H, W), dtype=int)
    for ch in range(C):
        total += extract_charge(field[ch]).charge
    return total
