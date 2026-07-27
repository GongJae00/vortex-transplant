"""V2 evaluation — corrected decision gates.

Key changes from V1:
1. REMOVED: defect_learned_not_innate (structurally impassable)
2. ADDED: analyzable_stable_topology_gate
3. ADDED: per-family IUT mechanism test
4. ADDED: representative invariance gate
5. ADDED: manifold validity gate
6. FIXED: competitor failure ≠ success
"""
import hashlib
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Callable

from ..topology import extract_charge
from ..decomposition import decompose
from ..interventions import hidden_to_complex, decompose_hidden, component_intervention
from ..task import generate_copy_batch, run_copy, write_copy, donor_sequences
from ..learned_evaluation import (
    TEST_EXAMPLES, TEST_DELAY, MAGNITUDE_EPSILON, TOPOLOGY_TOLERANCE,
    analyze_topology, select_donor_pair, StateTopology,
)
from ._types import (
    BranchStability, TopologyStatsV2, SelectionFunnel,
    SeedEvaluationV2, DecisionClauseV2, PerModelDecisionV2, CrossModelDecisionV2,
)
from .statistics import wild_bootstrap, iut_test

# ── Topology Analysis (V2 extensions) ──


def compute_branch_stability(field: np.ndarray) -> BranchStability:
    """Compute branch margin statistics for a complex field."""
    C, H, W = field.shape
    all_margins = []
    for ch in range(C):
        channel = field[ch]
        mag = np.abs(channel)
        if np.min(mag) <= MAGNITUDE_EPSILON:
            continue
        unit = channel / (mag + 1e-12)
        dx = np.concatenate([unit[:, 1:], unit[:, :1]], axis=1)
        dx = np.pi - np.abs(np.angle(dx * np.conj(unit)))
        dy = np.concatenate([unit[1:, :], unit[:1, :]], axis=0)
        dy = np.pi - np.abs(np.angle(dy * np.conj(unit)))
        all_margins.extend(dx.ravel().tolist())
        all_margins.extend(dy.ravel().tolist())

    margins = np.array(all_margins)
    return BranchStability(
        min_margin=float(np.min(margins)) if len(margins) > 0 else 0.0,
        q01_margin=float(np.quantile(margins, 0.01)) if len(margins) > 0 else 0.0,
        q05_margin=float(np.quantile(margins, 0.05)) if len(margins) > 0 else 0.0,
        median_margin=float(np.median(margins)) if len(margins) > 0 else 0.0,
        charge_flip_radius_median=0.0,  # requires perturbation analysis
    )


def analyze_topology_v2(hidden: torch.Tensor) -> TopologyStatsV2:
    """Extended topology analysis with V2 metrics."""
    field = hidden_to_complex(hidden)
    C, H, W = field.shape

    # V1 analysis for backward compatibility
    v1_result = analyze_topology(hidden)

    # Site-level defect density
    total_sites = 0
    total_charge_units = 0
    per_channel_prev = []
    for ch in range(C):
        channel = field[ch]
        if np.min(np.abs(channel)) <= MAGNITUDE_EPSILON:
            per_channel_prev.append(False)
            continue
        charge = extract_charge(channel, tolerance=TOPOLOGY_TOLERANCE)
        pos = int(np.sum(charge.charge > 0))
        neg = int(np.sum(charge.charge < 0))
        total_sites += pos + neg
        total_charge_units += int(np.sum(np.abs(charge.charge)))
        per_channel_prev.append(pos > 0 and neg > 0)

    site_density = total_sites / (C * H * W)
    charge_unit_density = total_charge_units / (C * H * W)

    branch = compute_branch_stability(field)

    return TopologyStatsV2(
        n_channels=C, H=H, W=W,
        defect_site_density=site_density,
        charge_unit_density=charge_unit_density,
        per_channel_prevalence=tuple(per_channel_prev),
        state_prevalence=v1_result.nonzero_defect,
        branch_stability=branch,
        signed_tuples=v1_result.signed_tuples,
        valid_channels=v1_result.valid_channels,
        net_charge_valid=v1_result.net_charge_valid,
        maximum_integer_residual=v1_result.maximum_integer_residual,
        minimum_magnitude=float(np.min(np.abs(field))),
    )


# ── Decision Gates ──


def analyzable_stable_topology_gate(
    topology_stats: TopologyStatsV2,
    min_valid_channels: int = 4,
    min_branch_q01: float = 0.0,  # UNFROZEN — calibrate
) -> DecisionClauseV2:
    """Gate: sufficient analyzable topology exists."""
    valid_ch_count = sum(topology_stats.valid_channels)
    passed = (
        valid_ch_count >= min_valid_channels and
        topology_stats.branch_stability.q01_margin >= min_branch_q01
    )
    return DecisionClauseV2(
        name="analyzable_stable_topology",
        passed=passed,
        value=float(valid_ch_count),
        threshold=float(min_valid_channels),
        detail=f"valid_channels={valid_ch_count}/{topology_stats.n_channels}, "
               f"branch_q01={topology_stats.branch_stability.q01_margin:.4f}",
    )


def representative_invariance_gate(
    rep_variance: float, charge_variance: float,
    max_ratio: float = 0.5,  # UNFROZEN — calibrate
) -> DecisionClauseV2:
    """Gate: vortex effect is representative-invariant.

    Representative variance should be small relative to charge effect variance.
    """
    if charge_variance > 0:
        ratio = rep_variance / charge_variance
    else:
        ratio = float("inf")
    passed = ratio <= max_ratio
    return DecisionClauseV2(
        name="representative_invariance",
        passed=passed,
        value=float(ratio),
        threshold=max_ratio,
        detail=f"rep_var/charge_var={ratio:.4f}",
    )


def manifold_validity_gate(
    manifold_pass_rate: float,
    min_pass_rate: float = 0.80,  # UNFROZEN — calibrate
) -> DecisionClauseV2:
    """Gate: intervened states are on the natural manifold."""
    passed = manifold_pass_rate >= min_pass_rate
    return DecisionClauseV2(
        name="manifold_validity",
        passed=passed,
        value=float(manifold_pass_rate),
        threshold=min_pass_rate,
        detail=f"on_manifold_rate={manifold_pass_rate:.3f}",
    )


def whole_state_positive_control_gate(
    ws_margins: list[float],
) -> DecisionClauseV2:
    """Gate: whole_state transplant produces expected behavioral shift."""
    if len(ws_margins) == 0:
        return DecisionClauseV2(
            name="whole_state_positive_control",
            passed=False, value=0.0, threshold=0.0,
            detail="no whole_state margins available",
        )
    mean_ws = float(np.mean(ws_margins))
    passed = mean_ws > 0
    return DecisionClauseV2(
        name="whole_state_positive_control",
        passed=passed,
        value=mean_ws,
        threshold=0.0,
        detail=f"mean_WS_margin={mean_ws:.4f}",
    )


def per_model_decision_v2(
    seed_records: list[dict],
    model_type: str,
    family_set: list[str],
    alpha: float = 0.05,
) -> PerModelDecisionV2:
    """V2 per-model decision with corrected gates.

    REMOVED: defect_learned_not_innate
    ADDED: analyzable_stable_topology, representative_invariance, manifold_validity
    """
    clauses = []

    # ── Prerequisites ──
    accuracies = [r.get("task_accuracy", 0.0) for r in seed_records]
    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    acc_pass = mean_acc >= 0.95
    clauses.append(DecisionClauseV2(
        name="prerequisite_accuracy", passed=acc_pass,
        value=mean_acc, threshold=0.95,
        detail=f"mean_accuracy={mean_acc:.4f}",
    ))

    # ── Analyzable stable topology ──
    n_valid = sum(
        1 for r in seed_records
        if r.get("analyzability_rate", 0.0) >= 0.50
    )
    topo_pass = n_valid >= max(1, len(seed_records) * 0.5)
    clauses.append(DecisionClauseV2(
        name="analyzable_stable_topology", passed=topo_pass,
        value=float(n_valid), threshold=max(1, len(seed_records) * 0.5),
        detail=f"seeds_with_analyzable_topology={n_valid}/{len(seed_records)}",
    ))

    # ── Whole-state positive control ──
    all_ws = []
    for r in seed_records:
        ws_m = r.get("whole_state_margins", [])
        if ws_m:
            all_ws.extend(ws_m)
    ws_mean = float(np.mean(all_ws)) if all_ws else 0.0
    ws_pass = ws_mean > 0
    clauses.append(DecisionClauseV2(
        name="whole_state_positive_control", passed=ws_pass,
        value=ws_mean, threshold=0.0,
        detail=f"mean_WS_margin={ws_mean:.4f}",
    ))

    # ── Per-family IUT ──
    per_family_data = {}
    for f in family_set:
        margins = []
        for r in seed_records:
            fm = r.get("per_family_margins", {}).get(f, [])
            if fm:
                margins.extend(fm)
        if margins:
            per_family_data[f] = {0: np.array(margins)}  # single-seed aggregation

    if per_family_data:
        iut_result = iut_test(per_family_data, alpha=alpha, method="wild_bootstrap", n_resamples=9999)
        iut_pass = iut_result.global_reject
    else:
        iut_pass = False

    clauses.append(DecisionClauseV2(
        name="iut_mechanism", passed=iut_pass,
        value=float(iut_result.max_p_value) if per_family_data else 1.0,
        threshold=alpha,
        detail=f"max_p={iut_result.max_p_value:.4f} worst={iut_result.worst_family}" if per_family_data else "no_data",
    ))

    # ── Representative invariance ──
    rep_ratios = [r.get("representative_variance_fraction", float("inf")) for r in seed_records]
    mean_rep = float(np.mean(rep_ratios)) if rep_ratios else float("inf")
    rep_pass = mean_rep <= 0.5
    clauses.append(DecisionClauseV2(
        name="representative_invariance", passed=rep_pass,
        value=mean_rep, threshold=0.5,
        detail=f"mean_rep_var_fraction={mean_rep:.4f}",
    ))

    # ── Manifold validity ──
    manifold_rates = [r.get("manifold_pass_rate", 0.0) for r in seed_records]
    mean_manifold = float(np.mean(manifold_rates)) if manifold_rates else 0.0
    manifold_pass = mean_manifold >= 0.80
    clauses.append(DecisionClauseV2(
        name="manifold_validity", passed=manifold_pass,
        value=mean_manifold, threshold=0.80,
        detail=f"mean_on_manifold_rate={mean_manifold:.3f}",
    ))

    # ── Overall ──
    all_pass = all(c.passed for c in clauses)
    status = "GO" if all_pass else "NO_GO_MECHANISM"

    return PerModelDecisionV2(
        model_type=model_type,
        clauses=clauses,
        overall_pass=all_pass,
        status=status,
    )


def cross_model_decision_v2(
    u1_decision: PerModelDecisionV2,
    plain_decision: PerModelDecisionV2,
    u1_advantage_over_plain: float = 0.0,
    u1_advantage_ci: tuple[float, float] = (0.0, 0.0),
) -> CrossModelDecisionV2:
    """V2 cross-model comparison.

    Plain non-significance ≠ U1-specific absence.
    U1-specific claim requires:
    1. U1 primary IUT passes
    2. U1–Plain paired contrast exceeds SESOI
    """
    cross_pass = (
        u1_decision.overall_pass and
        u1_advantage_over_plain > 0 and
        u1_advantage_ci[0] > 0
    )

    if not u1_decision.overall_pass:
        overall = "NO_GO_MECHANISM"
    elif not cross_pass:
        overall = "INCONCLUSIVE_BASELINE"
    else:
        overall = "GO"

    return CrossModelDecisionV2(
        u1_decision=u1_decision,
        plain_decision=plain_decision,
        u1_advantage_over_plain=u1_advantage_over_plain,
        u1_advantage_ci=u1_advantage_ci,
        cross_model_pass=cross_pass,
        overall_status=overall,
    )
