#!/usr/bin/env python3
"""Compare bootstrap-based tests for per-family contrasts under the IUT framework.

Validates Type-I error control for the primary test via Monte Carlo simulation
of seed-level data with configurable distribution families, correlations,
sample sizes, effect sizes, and missingness mechanisms.

Methods:
  1. Naive bootstrap percentile (centered at observed mean)
  2. Null-centered wild bootstrap at seed level
  3. One-sample t-test on seed-level D_{s,f}
  4. Bootstrap-t confidence bound inversion
  5. Exact sign-flip test (symmetry required)

IUT decision rule: reject global H_0  <->  max_f p_f <= alpha.

Under the global null (all delta_f = 0), P(IUT rejects) << alpha since it
requires ALL families to independently reject.  The IUT is self-adjusting:
P(max(p_f) <= alpha) <= P(p_f* <= alpha) <= alpha for any f* with delta_f* <= 0.
This simulation therefore tracks both IUT-global and per-family rejection rates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# parameter canonicalization
# ---------------------------------------------------------------------------

DISTRIBUTIONS = ("gaussian", "skewed", "heavy_tailed")
CORRELATIONS = {
    "independent": 0.0,
    "moderate": 0.3,
    "high": 0.7,
}
SEED_COUNTS = (10, 20, 30, 50)
EFFECT_SIZES = (0.0, 0.1, 0.3, 0.5)
MISSING_RATES = (0.0, 0.10, 0.30)
MISSING_TYPES = ("none", "mcar", "mar")

METHOD_NAMES = (
    "naive_bootstrap_percentile",
    "null_centered_wild_bootstrap",
    "one_sample_t_test",
    "bootstrap_t_inversion",
    "exact_sign_flip",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _native(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    return obj


def _family_correlation_matrix(n_families: int, rho: float) -> np.ndarray:
    """Compound-symmetry correlation matrix, shape (F, F)."""
    return np.eye(n_families) + rho * (1.0 - np.eye(n_families))


# Fixed theoretical centering/scaling constants per distribution so that
# sample-level natural variability is preserved (no in-place demeaning or
# re-scaling that would force the column mean / variance to their nominal
# values and kill all rejection rates).

_SKEW_C = 0.5
_SKEW_THEORETICAL_MEAN = float(np.exp(_SKEW_C ** 2 / 2.0))          # ≈ 1.133
_SKEW_THEORETICAL_STD = float(np.sqrt(1.0 + np.exp(2 * _SKEW_C ** 2)
                                      - np.exp(_SKEW_C ** 2)
                                      + 2 * _SKEW_C * np.exp(_SKEW_C ** 2 / 2.0)))  # ≈ 1.581

_HEAVY_DF = 3.0
_HEAVY_THEORETICAL_STD = float(np.sqrt(_HEAVY_DF / (_HEAVY_DF - 2.0)))  # ≈ 1.732


def _generate_raw_data(
    rng: np.random.Generator,
    n_seeds: int,
    n_families: int,
    rho: float,
    dist: str,
) -> np.ndarray:
    """Generate (n_seeds, n_families) seed-level data, approx unit variance.

    Uses fixed theoretical centering and scaling (NOT per-sample demeaning)
    so that natural sampling variability is preserved within each column.
    """
    Sigma = _family_correlation_matrix(n_families, rho)
    if dist == "gaussian":
        return rng.multivariate_normal(np.zeros(n_families), Sigma, size=n_seeds)
    elif dist == "skewed":
        E = rng.multivariate_normal(np.zeros(n_families), Sigma, size=n_seeds)
        D = E + np.exp(E * _SKEW_C)
        D = (D - _SKEW_THEORETICAL_MEAN) / _SKEW_THEORETICAL_STD
        return D
    elif dist == "heavy_tailed":
        E = rng.multivariate_normal(np.zeros(n_families), Sigma, size=n_seeds)
        chi2 = rng.chisquare(df=_HEAVY_DF, size=n_seeds)
        scale = np.sqrt(chi2 / _HEAVY_DF)
        D = E / scale[:, np.newaxis]
        D = D / _HEAVY_THEORETICAL_STD
        return D
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def _apply_missingness(
    rng: np.random.Generator,
    data: np.ndarray,
    missing_rate: float,
    missing_type: str,
) -> np.ndarray:
    """Return boolean mask of shape data.shape, True == observed."""
    n_seeds, n_families = data.shape
    if missing_rate == 0.0 or missing_type == "none":
        return np.ones_like(data, dtype=bool)
    if missing_type == "mcar":
        return rng.random((n_seeds, n_families)) > missing_rate
    if missing_type == "mar":
        abs_data = np.abs(data)
        ranks = scipy_stats.rankdata(abs_data, axis=0, method="average")
        p_miss = missing_rate * 2.0 * (ranks / (n_seeds + 1))
        p_miss = np.clip(p_miss, 0.0, 1.0)
        return rng.random((n_seeds, n_families)) > p_miss
    raise ValueError(f"Unknown missingness type: {missing_type}")


# ---------------------------------------------------------------------------
# vectorized bootstrap engine
# ---------------------------------------------------------------------------

def _bootstrap_indices(
    rng: np.random.Generator,
    n: int,
    n_bootstrap: int,
) -> np.ndarray:
    """Return (n_bootstrap, n) int64 array of bootstrap resample indices."""
    return rng.integers(0, n, size=(n_bootstrap, n))


def _bootstrap_means(data: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return (n_bootstrap,) array of bootstrap means."""
    return data[indices].mean(axis=1)


def _wild_means(data: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Return (n_bootstrap,) array of wild-bootstrap means (Rademacher)."""
    n = len(data)
    w = rng.choice([-1, 1], size=(n_bootstrap, n), p=[0.5, 0.5])
    return data[np.newaxis, :] * w  # broadcast, then mean over axis 1


# ---------------------------------------------------------------------------
# per-family test methods  (vectorized)
# ---------------------------------------------------------------------------

def _per_family_tests(
    family_data: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compute one-sided p-values for a single family.

    H_0: delta_f <= 0   vs  H_1: delta_f > 0.
    Returns dict mapping method name -> p-value.
    """
    n = len(family_data)
    if n < 2:
        return {m: 0.5 for m in METHOD_NAMES}

    obs_mean = float(np.mean(family_data))
    obs_sd = float(np.std(family_data, ddof=1))
    obs_se = obs_sd / np.sqrt(n)
    obs_t = obs_mean / obs_se if obs_se > 0 else (np.inf if obs_mean > 0 else -np.inf)

    idx = _bootstrap_indices(rng, n, n_bootstrap)
    results: dict[str, float] = {}

    # -- method 1: naive bootstrap percentile --
    bmeans_naive = _bootstrap_means(family_data, idx)
    p_naive = (1.0 + np.sum(bmeans_naive <= 0.0)) / (n_bootstrap + 1.0)
    results["naive_bootstrap_percentile"] = float(p_naive)

    # -- method 2: null-centered wild bootstrap --
    centered = family_data - obs_mean
    bmeans_null = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        w = rng.choice([-1, 1], size=n, p=[0.5, 0.5])
        bmeans_null[b] = np.mean(centered * w)
    p_null = (1.0 + np.sum(bmeans_null >= obs_mean)) / (n_bootstrap + 1.0)
    results["null_centered_wild_bootstrap"] = float(p_null)

    # -- method 3: one-sample t-test --
    p_t = float(scipy_stats.t.sf(obs_t, df=n - 1))
    results["one_sample_t_test"] = p_t

    # -- method 4: bootstrap-t confidence bound inversion --
    # t* = sqrt(n) * (theta* - theta_obs) / s*
    bmeans_bt = _bootstrap_means(family_data, idx)
    # bootstrap SE for each resample
    boot_sq = family_data[idx] - family_data[idx].mean(axis=1, keepdims=True)
    bse_bt = np.sqrt(np.sum(boot_sq ** 2, axis=1) / (n - 1)) / np.sqrt(n)
    valid = bse_bt > 0
    if np.any(valid):
        t_star = np.sqrt(n) * (bmeans_bt[valid] - obs_mean) / bse_bt[valid]
        # one-sided H_1: mu > 0 -> P*(t* <= -t_obs)
        p_bt = (1.0 + np.sum(t_star <= -obs_t)) / (len(t_star) + 1.0)
    else:
        p_bt = 0.5
    results["bootstrap_t_inversion"] = float(p_bt)

    # -- method 5: exact sign-flip test --
    if n <= 12:
        n_configs = 1 << n
        perm_means = np.empty(n_configs)
        for bits in range(n_configs):
            signs = np.where(((bits >> np.arange(n)) & 1) == 0, -1.0, 1.0)
            perm_means[bits] = np.mean(family_data * signs)
    else:
        n_configs = n_bootstrap
        perm_means = np.empty(n_configs)
        for b in range(n_configs):
            signs = rng.choice([-1.0, 1.0], size=n, p=[0.5, 0.5])
            perm_means[b] = np.mean(family_data * signs)
    p_sf = (1.0 + np.sum(perm_means >= obs_mean)) / (n_configs + 1.0)
    results["exact_sign_flip"] = float(p_sf)

    return results


def _iut_reject(p_per_family: dict[str, list[float]], alpha: float) -> dict[str, bool]:
    """IUT: reject global H_0 iff max_f p_f <= alpha."""
    return {
        m: bool(np.max(ps) <= alpha) if ps else False
        for m, ps in p_per_family.items()
    }


def _per_family_reject(p_per_family: dict[str, list[float]], alpha: float) -> dict[str, np.ndarray]:
    """Return boolean array per family: True if p_f <= alpha."""
    return {
        m: np.array(ps) <= alpha
        for m, ps in p_per_family.items()
    }


def _per_family_ci_coverage(
    family_data: np.ndarray,
    true_effect: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Two-sided CI coverage: True if true_effect inside CI."""
    n = len(family_data)
    if n < 2:
        return {m: True for m in METHOD_NAMES}

    obs_mean = float(np.mean(family_data))
    obs_sd = float(np.std(family_data, ddof=1))
    obs_se = obs_sd / np.sqrt(n)
    idx = _bootstrap_indices(rng, n, n_bootstrap)
    results: dict[str, bool] = {}

    # naive bootstrap
    bm = _bootstrap_means(family_data, idx)
    lo = np.percentile(bm, 2.5)
    hi = np.percentile(bm, 97.5)
    results["naive_bootstrap_percentile"] = bool(lo <= true_effect <= hi)

    # null-centered wild
    centered = family_data - obs_mean
    bm_n = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        w = rng.choice([-1, 1], size=n, p=[0.5, 0.5])
        bm_n[b] = np.mean(centered * w)
    lo_n = obs_mean + np.percentile(bm_n, 2.5)
    hi_n = obs_mean + np.percentile(bm_n, 97.5)
    results["null_centered_wild_bootstrap"] = bool(lo_n <= true_effect <= hi_n)

    # t-test
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    lo_t = obs_mean - t_crit * obs_se
    hi_t = obs_mean + t_crit * obs_se
    results["one_sample_t_test"] = bool(lo_t <= true_effect <= hi_t)

    # bootstrap-t
    boot_sq = family_data[idx] - family_data[idx].mean(axis=1, keepdims=True)
    bse = np.sqrt(np.sum(boot_sq ** 2, axis=1) / (n - 1)) / np.sqrt(n)
    valid = bse > 0
    if np.sum(valid) >= 2:
        t_s = np.sqrt(n) * (bm[valid] - obs_mean) / bse[valid]
        t_lo = np.percentile(t_s, 2.5)
        t_hi = np.percentile(t_s, 97.5)
        lo_bt = obs_mean - t_hi * obs_se
        hi_bt = obs_mean - t_lo * obs_se
        results["bootstrap_t_inversion"] = bool(lo_bt <= true_effect <= hi_bt)
    else:
        results["bootstrap_t_inversion"] = True

    # sign-flip
    results["exact_sign_flip"] = bool(lo_n <= true_effect <= hi_n)

    return results


# ---------------------------------------------------------------------------
# scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    dist: str
    correlation_name: str
    rho: float
    n_seeds: int
    effect_size: float
    missing_rate: float
    missing_type: str
    n_families: int
    alpha: float
    n_bootstrap: int
    n_sim: int
    method_results: dict = field(default_factory=dict)
    computation_time_sec: float = 0.0


def _run_single_scenario(
    *,
    dist: str,
    correlation_name: str,
    rho: float,
    n_seeds: int,
    effect_size: float,
    missing_rate: float,
    missing_type: str,
    n_families: int,
    n_sim: int,
    n_bootstrap: int,
    alpha: float,
    master_seed: int,
) -> ScenarioResult:
    """Monte Carlo evaluation of one scenario."""
    t0 = time.perf_counter()
    rng = _rng(master_seed)
    n_clip = max(n_bootstrap, n_seeds)
    effect_sizes = [effect_size] * n_families
    missing_str = "none" if missing_rate == 0.0 else missing_type

    iut_rejections: dict[str, int] = {m: 0 for m in METHOD_NAMES}
    pf_rejections: dict[str, int] = {m: 0 for m in METHOD_NAMES}
    pf_n_families: dict[str, int] = {m: 0 for m in METHOD_NAMES}
    ci_covered: dict[str, int] = {m: 0 for m in METHOD_NAMES}
    ci_total: int = 0
    method_elapsed: dict[str, float] = {m: 0.0 for m in METHOD_NAMES}

    is_global_null = effect_size == 0.0

    for sim_i in range(n_sim):
        seed_i = master_seed + sim_i
        rng_s = _rng(seed_i)

        raw = _generate_raw_data(rng_s, n_seeds, n_families, rho, dist)
        D = raw + np.array(effect_sizes)
        mask = _apply_missingness(rng_s, D, missing_rate, missing_str)

        per_family_ps: dict[str, list[float]] = {m: [] for m in METHOD_NAMES}
        per_family_cis: dict[str, list[bool]] = {m: [] for m in METHOD_NAMES}

        t_m_start = time.perf_counter()

        for f in range(n_families):
            observed = D[mask[:, f], f]
            if len(observed) < 2:
                for m in METHOD_NAMES:
                    per_family_ps[m].append(0.5)
                    per_family_cis[m].append(True)
                continue

            t_p_start = time.perf_counter()
            ps = _per_family_tests(observed, n_clip, rng_s)
            t_p_end = time.perf_counter()
            dt = (t_p_end - t_p_start) / len(METHOD_NAMES)
            for m in METHOD_NAMES:
                method_elapsed[m] += dt

            for m in METHOD_NAMES:
                per_family_ps[m].append(ps[m])

            ci_res = _per_family_ci_coverage(observed, effect_size, n_clip, rng_s, alpha)
            for m in METHOD_NAMES:
                per_family_cis[m].append(ci_res[m])

        # IUT global decision
        for m in METHOD_NAMES:
            if _iut_reject(per_family_ps, alpha)[m]:
                iut_rejections[m] += 1

        # per-family rejection rates (relevant for global-null scenarios)
        if is_global_null:
            for m in METHOD_NAMES:
                rej = _per_family_reject(per_family_ps, alpha)[m]
                pf_rejections[m] += int(np.sum(rej))
                pf_n_families[m] += len(rej)

        # CI coverage
        for m in METHOD_NAMES:
            ci_covered[m] += sum(per_family_cis[m])
        ci_total += len(per_family_cis[m])

    elapsed = time.perf_counter() - t0

    method_results: dict[str, dict] = {}
    for m in METHOD_NAMES:
        iut_rate = iut_rejections[m] / n_sim if n_sim > 0 else 0.0
        cov_rate = ci_covered[m] / ci_total if ci_total > 0 else 1.0
        pf_rate = pf_rejections[m] / pf_n_families[m] if pf_n_families[m] > 0 else None
        entry: dict[str, Any] = {
            "iut_rejection_rate": iut_rate,
            "iut_rejections": iut_rejections[m],
            "ci_coverage": cov_rate,
            "ci_covered": ci_covered[m],
            "ci_total": ci_total,
            "computation_time_sec": method_elapsed[m],
        }
        if pf_rate is not None:
            entry["per_family_false_positive_rate"] = pf_rate
        method_results[m] = _native(entry)

    return ScenarioResult(
        dist=dist,
        correlation_name=correlation_name,
        rho=rho,
        n_seeds=n_seeds,
        effect_size=effect_size,
        missing_rate=missing_rate,
        missing_type=missing_str,
        n_families=n_families,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        n_sim=n_sim,
        method_results=method_results,
        computation_time_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# grid builders
# ---------------------------------------------------------------------------

def build_smoke_grid() -> list[dict]:
    """Minimal grid for fast validation."""
    return [
        {
            "dist": "gaussian",
            "correlation_name": corr_name,
            "rho": rho,
            "n_seeds": 10,
            "effect_size": es,
            "missing_rate": 0.0,
            "missing_type": "none",
        }
        for corr_name, rho in [
            ("independent", 0.0),
            ("moderate", 0.3),
            ("high", 0.7),
        ]
        for es in (0.0, 0.5)
    ]


def build_full_grid() -> list[dict]:
    """Complete grid — expensive; use after smoke validation."""
    combos = []
    for dist in DISTRIBUTIONS:
        for corr_name, rho in CORRELATIONS.items():
            for n_seeds in SEED_COUNTS:
                for es in EFFECT_SIZES:
                    for mr in MISSING_RATES:
                        if mr == 0.0:
                            combos.append({
                                "dist": dist,
                                "correlation_name": corr_name,
                                "rho": rho,
                                "n_seeds": n_seeds,
                                "effect_size": es,
                                "missing_rate": 0.0,
                                "missing_type": "none",
                            })
                        else:
                            for mt in ("mcar", "mar"):
                                combos.append({
                                    "dist": dist,
                                    "correlation_name": corr_name,
                                    "rho": rho,
                                    "n_seeds": n_seeds,
                                    "effect_size": es,
                                    "missing_rate": mr,
                                    "missing_type": mt,
                                })
    return combos


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

OUTPUT_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simulation_results.json",
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IUT statistical method comparison simulation"
    )
    p.add_argument("--smoke", action="store_true", default=False)
    p.add_argument("--full", action="store_true", default=False)
    p.add_argument("--n-sim", type=int, default=1000)
    p.add_argument("--n-bootstrap", type=int, default=199)
    p.add_argument("--n-families", type=int, default=5)
    p.add_argument("--n-seeds-override", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=OUTPUT_PATH_DEFAULT)
    return p.parse_args(argv)


def _print_header(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def _print_results(results: list[ScenarioResult]) -> None:
    for sr in results:
        is_null = sr.effect_size == 0.0
        label = (f"dist={sr.dist} corr={sr.correlation_name}(ρ={sr.rho}) "
                 f"N={sr.n_seeds} δ={sr.effect_size} "
                 f"miss={sr.missing_rate:.0%}-{sr.missing_type}")
        print(f"\n  {label}  ({sr.n_sim} sims, {sr.computation_time_sec:.1f}s)")
        hdr = f"  {'Method':<38} {'IUT rej':>9} {'CI cov':>8}"
        if is_null:
            hdr += f" {'PF FPR':>8}"
        hdr += f"  {'Note'}"
        print(hdr)
        print(f"  {'-'*38} {'-'*9} {'-'*8}", end="")
        if is_null:
            print(f" {'-'*8}", end="")
        print(f"  {'-'*18}")

        for m in METHOD_NAMES:
            mr = sr.method_results[m]
            rate = mr["iut_rejection_rate"]
            ci = mr["ci_coverage"]
            row = f"  {m:<38} {rate:>9.6f} {ci:>8.4f}"
            if is_null and "per_family_false_positive_rate" in mr:
                pf = mr["per_family_false_positive_rate"]
                row += f" {pf:>8.5f}"
            note = ""
            if is_null:
                # per-family FPR should be close to alpha
                if "per_family_false_positive_rate" in mr:
                    pf = mr["per_family_false_positive_rate"]
                    se_pf = np.sqrt(0.05 * 0.95 / (sr.n_families * sr.n_sim))
                    if pf > 0.05 + 3 * se_pf:
                        note = "*** INFLATED PF ***"
            if "naive" in m and is_null:
                note = (note + " " if note else "") + "[naive]"
            print(row + f"  {note}" if note else row)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.smoke and not args.full:
        args.smoke = True

    grid = build_smoke_grid() if args.smoke else build_full_grid()
    n_families = args.n_families

    mode_label = "SMOKE MODE" if args.smoke else "FULL SIMULATION"
    _print_header(mode_label)
    print(f"  scenarios: {len(grid)}")
    print(f"  sim reps: {args.n_sim}  |  bootstrap: {args.n_bootstrap}")
    print(f"  families: {n_families}  |  alpha: {args.alpha}  |  seed: {args.seed}")

    if args.n_seeds_override:
        for g in grid:
            g["n_seeds"] = args.n_seeds_override

    results: list[ScenarioResult] = []
    t_total = time.perf_counter()

    for i, g in enumerate(grid):
        label = (f"[{i+1}/{len(grid)}] dist={g['dist']} "
                 f"corr={g['correlation_name']} N={g['n_seeds']} "
                 f"δ={g['effect_size']} miss={g['missing_rate']}/{g['missing_type']}")
        print(f"  {label} ... ", end="", flush=True)

        sr = _run_single_scenario(
            dist=g["dist"],
            correlation_name=g["correlation_name"],
            rho=g["rho"],
            n_seeds=g["n_seeds"],
            effect_size=g["effect_size"],
            missing_rate=g["missing_rate"],
            missing_type=g["missing_type"],
            n_families=n_families,
            n_sim=args.n_sim,
            n_bootstrap=args.n_bootstrap,
            alpha=args.alpha,
            master_seed=args.seed + i * 10000,
        )
        results.append(sr)
        print(f"{sr.computation_time_sec:.1f}s")

    total_elapsed = time.perf_counter() - t_total

    # -- results --
    _print_header("RESULTS SUMMARY")
    _print_results(results)

    # -- Type-I check --
    _print_header("TYPE-I ERROR CHECK (δ=0, per-family false-positive rates)")
    any_inflated = False
    for sr in results:
        if sr.effect_size != 0.0:
            continue
        for m in METHOD_NAMES:
            if "per_family_false_positive_rate" not in sr.method_results[m]:
                continue
            pf = sr.method_results[m]["per_family_false_positive_rate"]
            se = np.sqrt(0.05 * 0.95 / (sr.n_families * sr.n_sim))
            if pf > 0.05 + 3 * se:
                print(f"  ** {m}: PF-FPR={pf:.5f} > {0.05+3*se:.5f} "
                      f"(corr={sr.correlation_name})")
                any_inflated = True
    if not any_inflated:
        print("  All methods pass per-family Type-I error control (3-SE bound).")

    # -- save --
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    serializable = [_native(asdict(sr)) for sr in results]
    output_payload = {
        "config": {
            "smoke": args.smoke,
            "n_sim": args.n_sim,
            "n_bootstrap": args.n_bootstrap,
            "n_families": n_families,
            "alpha": args.alpha,
            "master_seed": args.seed,
        },
        "scenarios": serializable,
        "total_computation_time_sec": round(total_elapsed, 1),
    }
    with open(args.output, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    # -- recommendation --
    if args.smoke:
        _print_header("FULL-SIMULATION RECOMMENDATION")
        full_grid = build_full_grid()
        n_full = len(full_grid)
        per_scenario = total_elapsed / max(len(grid), 1)
        est_total = per_scenario * n_full
        print(f"  Full grid scenarios:  {n_full}")
        print(f"  Time per scenario:    {per_scenario:.1f}s")
        print(f"  Est. total (current): {est_total:.0f}s ({est_total/3600:.1f}h)")
        scaled = est_total * (5000 / args.n_sim) * (1999 / args.n_bootstrap)
        print(f"  Est. total (5k sim, 1999 boot): {scaled:.0f}s ({scaled/3600:.1f}h)")
        print(f"  Recommended: run with --smoke --n-sim 5000 --n-bootstrap 1999 first")
        print(f"  Then --full --n-sim 2000 --n-bootstrap 999 for reasonable runtime")

    return 0


if __name__ == "__main__":
    sys.exit(main())
