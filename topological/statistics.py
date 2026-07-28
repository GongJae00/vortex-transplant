"""Hierarchical bootstrap and intersection-union test for V2.

Primary method: null-centered wild bootstrap at seed level.
IUT: reject global H₀ iff max(p_f) ≤ α for all null families.
"""
import numpy as np
from .types import BootstrapResult, IUTResult


def hierarchical_bootstrap(
    seed_level_data: dict[int, np.ndarray],  # seed -> array of per-pair margins
    n_resamples: int = 9999,
    seed: int = 0,
) -> BootstrapResult:
    """Hierarchical bootstrap CI for seed-level mean.

    Resamples seeds with replacement, then within resampled seeds,
    resamples pairs with replacement.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(seed_level_data.keys())
    n_seeds = len(seeds)

    # Observed seed-level means
    seed_means = np.array([np.mean(seed_level_data[s]) for s in seeds])
    theta_hat = np.mean(seed_means)

    # Bootstrap
    bootstrap_means = np.zeros(n_resamples)
    for b in range(n_resamples):
        seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        resampled_means = seed_means[seed_idx]
        bootstrap_means[b] = np.mean(resampled_means)

    ci_lower = np.percentile(bootstrap_means, 2.5)
    ci_upper = np.percentile(bootstrap_means, 97.5)

    # One-sided p-value (H₀: θ ≤ 0) via bootstrap percentile
    p_value = (np.sum(bootstrap_means <= 0) + 1) / (n_resamples + 1)

    return BootstrapResult(
        estimate=float(theta_hat),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        n_resamples=n_resamples,
        method="hierarchical_bootstrap_percentile",
    )


def wild_bootstrap(
    seed_level_data: dict[int, np.ndarray],
    n_resamples: int = 9999,
    seed: int = 0,
) -> BootstrapResult:
    """Null-centered wild bootstrap for seed-level contrasts.

    Uses Rademacher weights (±1) to generate null distribution centered at zero.
    Appropriate for testing H₀: θ = 0.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(seed_level_data.keys())
    n_seeds = len(seeds)

    seed_means = np.array([np.mean(seed_level_data[s]) for s in seeds])
    theta_hat = np.mean(seed_means)

    # Center at zero (null)
    centered = seed_means - theta_hat

    bootstrap_means = np.zeros(n_resamples)
    for b in range(n_resamples):
        weights = rng.choice([-1.0, 1.0], size=n_seeds)
        bootstrap_means[b] = np.mean(weights * centered)

    # One-sided p-value: fraction of null distribution ≥ observed
    p_value = (np.sum(bootstrap_means >= theta_hat) + 1) / (n_resamples + 1)

    ci_lower = np.percentile(bootstrap_means + theta_hat, 2.5)
    ci_upper = np.percentile(bootstrap_means + theta_hat, 97.5)

    return BootstrapResult(
        estimate=float(theta_hat),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        p_value=float(p_value),
        n_resamples=n_resamples,
        method="wild_bootstrap_rademacher",
    )


def iut_test(
    per_family_data: dict[str, dict[int, np.ndarray]],
    alpha: float = 0.05,
    method: str = "wild_bootstrap",
    n_resamples: int = 9999,
    seed: int = 0,
) -> IUTResult:
    """Intersection-union test across null families.

    H₀: ∃ f s.t. δ_f ≤ 0  (vortex fails to beat at least one family)
    H₁: ∀ f, δ_f > 0       (vortex beats every family)

    Reject H₀ iff max(p_f) ≤ α.
    """
    per_family = {}
    for family_name, seed_data in per_family_data.items():
        if method == "wild_bootstrap":
            result = wild_bootstrap(seed_data, n_resamples=n_resamples, seed=seed + hash(family_name) % (2**31))
        else:
            result = hierarchical_bootstrap(seed_data, n_resamples=n_resamples, seed=seed + hash(family_name) % (2**31))
        per_family[family_name] = result

    max_p = max(r.p_value for r in per_family.values())
    worst_family = max(per_family, key=lambda f: per_family[f].p_value)
    global_reject = max_p <= alpha

    return IUTResult(
        per_family=per_family,
        global_reject=global_reject,
        max_p_value=max_p,
        worst_family=worst_family,
        alpha=alpha,
    )


def compute_sesoi(whole_state_margins: dict[int, np.ndarray], fraction: float = 0.1) -> float:
    """Compute SESOI as fraction of whole-state recovery magnitude."""
    all_ws = np.concatenate(list(whole_state_margins.values()))
    return float(fraction * np.mean(np.abs(all_ws)))
