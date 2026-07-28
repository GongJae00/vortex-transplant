"""V2 topology extensions: branch margin, charge-flip radius, defect tracking."""
import numpy as np
from ..topology import extract_charge, phase_links, plaquette_curl
from ._types import BranchStability


def compute_branch_margins(field: np.ndarray) -> BranchStability:
    """Compute branch margin statistics for a complex field.
    
    Branch margin = π - |Δθ_e| for each edge e.
    Measures how far each link variable is from the branch cut at ±π.
    Fields with small branch margins are close to charge instability.
    """
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape {field.shape}")

    H, W = field.shape
    unit = field / (np.abs(field) + 1e-12)

    dx, dy = phase_links(unit)
    # Periodic padding
    dx_pad = np.concatenate([dx, dx[:, :1]], axis=1)
    dy_pad = np.concatenate([dy, dy[:1, :]], axis=0)

    all_margins = np.concatenate([
        (np.pi - np.abs(dx_pad)).ravel(),
        (np.pi - np.abs(dy_pad)).ravel(),
    ])

    # Charge-flip radius: median epsilon to flip any charge
    charge = extract_charge(field)
    charge_positions = np.argwhere(np.abs(charge.charge) > 0)

    flip_radii = []
    if len(charge_positions) > 0:
        for px, py in charge_positions[:20]:  # sample up to 20
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


def compute_per_channel_branch_margins(field: np.ndarray) -> list[BranchStability]:
    """Compute branch margins for each channel in a (C, H, W) field."""
    C = field.shape[0]
    return [compute_branch_margins(field[ch]) for ch in range(C)]


def _measure_charge_flip_radius(
    field: np.ndarray, x: int, y: int,
    eps_values: list[float] | None = None,
) -> float:
    """Measure the minimum phase perturbation to flip charge at (x, y)."""
    if eps_values is None:
        eps_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    charge_ref = extract_charge(field)
    ref_val = charge_ref.charge[x, y]

    for eps in eps_values:
        for sign in [1.0, -1.0]:
            perturbed = field.copy()
            perturbed[x, y] *= np.exp(1j * sign * eps)
            charge_test = extract_charge(perturbed)
            if charge_test.charge[x, y] != ref_val:
                return eps
    return float("inf")


def defect_tracking(
    field_t: np.ndarray,   # (H, W) at time t
    field_t1: np.ndarray,  # (H, W) at time t+1
    max_displacement: int = 2,
) -> dict:
    """Track signed defects between consecutive time steps.

    Uses optimal bipartite matching with spatial proximity.
    Returns: matched, births, deaths, jaccard.
    """
    charge_t = extract_charge(field_t)
    charge_t1 = extract_charge(field_t1)

    defects_t = []
    for x in range(charge_t.charge.shape[0]):
        for y in range(charge_t.charge.shape[1]):
            q = charge_t.charge[x, y]
            if q != 0:
                defects_t.append((x, y, q))

    defects_t1 = []
    for x in range(charge_t1.charge.shape[0]):
        for y in range(charge_t1.charge.shape[1]):
            q = charge_t1.charge[x, y]
            if q != 0:
                defects_t1.append((x, y, q))

    # Greedy nearest-neighbor matching (simplified, OK for sparse defects)
    matched = []
    unmatched_t1 = list(defects_t1)

    for dx, dy, dq in defects_t:
        best_dist = float("inf")
        best_idx = -1
        for i, (dx1, dy1, dq1) in enumerate(unmatched_t1):
            if dq != dq1:
                continue
            dist = abs(dx - dx1) + abs(dy - dy1)  # Manhattan, periodic
            dist = min(dist, abs(dx - dx1 - charge_t.charge.shape[0]))
            dist = min(dist, abs(dy - dy1 - charge_t.charge.shape[1]))
            if dist < best_dist and dist <= max_displacement:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched.append((dx, dy, unmatched_t1[best_idx][0], unmatched_t1[best_idx][1], dq))
            unmatched_t1.pop(best_idx)

    births = [(x, y, q) for x, y, q in unmatched_t1]
    deaths = [(dx, dy, dq) for dx, dy, dq in defects_t
              if not any(m[0] == dx and m[1] == dy for m in matched)]

    # Signed Jaccard
    intersection = len(matched)
    union = len(defects_t) + len(defects_t1) - intersection
    jaccard = intersection / max(union, 1)

    return {
        "matched": matched,
        "births": births,
        "deaths": deaths,
        "signed_jaccard": jaccard,
        "n_defects_t": len(defects_t),
        "n_defects_t1": len(defects_t1),
    }


def per_channel_defect_prevalence(field: np.ndarray) -> tuple[bool, ...]:
    """Per-channel presence of both + and - defects."""
    C = field.shape[0]
    result = []
    for ch in range(C):
        charge = extract_charge(field[ch])
        has_pos = int(np.sum(charge.charge > 0)) > 0
        has_neg = int(np.sum(charge.charge < 0)) > 0
        result.append(has_pos and has_neg)
    return tuple(result)


def extract_charge_map(field: np.ndarray) -> np.ndarray:
    """Extract charge map summed across channels. (C,H,W) → (H,W) integer."""
    C, H, W = field.shape
    total = np.zeros((H, W), dtype=int)
    for ch in range(C):
        charge = extract_charge(field[ch])
        total += charge.charge
    return total
