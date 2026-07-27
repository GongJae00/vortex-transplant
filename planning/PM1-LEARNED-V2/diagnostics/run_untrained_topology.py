"""Exact V1 gate replication: 128-example untrained topology for U1 and Plain."""
import hashlib, json, sys, time, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from topological.model import ModelSpec
from topological.training import make_model, configure_determinism
from topological.task import generate_copy_batch, run_copy
from topological.topology import extract_charge
from topological.interventions import hidden_to_complex

# ── Replicate V1 gate constants ──
TEST_EXAMPLES = 128
TEST_DELAY = 64
SEEDS = list(range(10))
MODEL_TYPES = ["u1", "plain"]
C, H, W = 8, 16, 16
MAGNITUDE_EPSILON = 1e-8

def _analyze_one(field):
    """Mirror V1 analyze_topology but with full metric reporting."""
    positive_site_count = 0
    negative_site_count = 0
    positive_charge_units = 0
    negative_charge_units = 0
    valid_channels = 0
    nonzero_defect = False
    residuals = []
    branch_margins = []
    min_magnitudes = []

    for ch in range(field.shape[0]):
        channel = field[ch]
        mag = np.abs(channel)
        min_mag = float(np.min(mag))
        min_magnitudes.append(min_mag)
        if min_mag <= MAGNITUDE_EPSILON:
            continue
        valid_channels += 1
        charge = extract_charge(channel, tolerance=1e-10)
        residuals.append(charge.residual_max)
        positive_site_count += int(np.sum(charge.charge > 0))
        negative_site_count += int(np.sum(charge.charge < 0))
        positive_charge_units += int(np.sum(np.maximum(charge.charge, 0)))
        negative_charge_units += int(np.sum(np.maximum(-charge.charge, 0)))
        if int(np.sum(charge.charge > 0)) > 0 and int(np.sum(charge.charge < 0)) > 0:
            nonzero_defect = True

        # Branch margins from link variables
        unit = channel / (mag + 1e-12)
        dx = np.concatenate([unit[:, 1:], unit[:, :1]], axis=1)
        dx = np.pi - np.abs(np.angle(dx * np.conj(unit)))
        dy = np.concatenate([unit[1:, :], unit[:1, :]], axis=0)
        dy = np.pi - np.abs(np.angle(dy * np.conj(unit)))
        all_links = np.concatenate([dx.ravel(), dy.ravel()])
        branch_margins.append(float(np.min(all_links)))
        branch_margins.append(float(np.quantile(all_links, 0.01)))

    signed_site_count = positive_site_count + negative_site_count
    absolute_charge_units = positive_charge_units + negative_charge_units
    net_charge = positive_charge_units - negative_charge_units

    return {
        "nonzero_defect": nonzero_defect,
        "valid_channels": valid_channels,
        "positive_site_count": positive_site_count,
        "negative_site_count": negative_site_count,
        "signed_site_count": signed_site_count,
        "positive_charge_units": positive_charge_units,
        "negative_charge_units": negative_charge_units,
        "absolute_charge_units": absolute_charge_units,
        "net_charge": net_charge,
        "site_density": signed_site_count / (C * H * W),
        "charge_unit_density": absolute_charge_units / (C * H * W),
        "max_residual": float(max(residuals, default=0.0)),
        "branch_margin_min": float(min(branch_margins, default=float("inf"))),
    }

device = torch.device("cpu")
spec = ModelSpec()
t0 = time.perf_counter()
all_results = {}

for model_type in MODEL_TYPES:
    for seed in SEEDS:
        label = f"{model_type}_seed{seed}"
        print(f"[{label}] running...", flush=True)
        configure_determinism(seed)
        model = make_model(seed, model_type=model_type, model_spec=spec, device=device)

        batch = generate_copy_batch(seed, "test/heldout-delay-64", TEST_EXAMPLES, TEST_DELAY, device=device)
        trace = run_copy(model, batch.symbols, TEST_DELAY)
        hidden_pw = trace.post_write.detach().cpu()
        hidden_pg = trace.pre_go.detach().cpu()

        seed_results = {"post_write": [], "pre_go": []}
        for ex_idx in range(TEST_EXAMPLES):
            # Post-write
            f_pw = hidden_to_complex(hidden_pw[ex_idx])
            seed_results["post_write"].append(_analyze_one(f_pw))
            # Pre-GO
            f_pg = hidden_to_complex(hidden_pg[ex_idx])
            seed_results["pre_go"].append(_analyze_one(f_pg))

        prev_pw = sum(1 for r in seed_results["post_write"] if r["nonzero_defect"]) / TEST_EXAMPLES
        prev_pg = sum(1 for r in seed_results["pre_go"] if r["nonzero_defect"]) / TEST_EXAMPLES

        # site densities
        site_den_pw = [r["site_density"] for r in seed_results["post_write"]]
        site_den_pg = [r["site_density"] for r in seed_results["pre_go"]]

        all_results[label] = {
            "state_prevalence_post_write": prev_pw,
            "state_prevalence_pre_go": prev_pg,
            "site_density_post_write_mean": float(np.mean(site_den_pw)),
            "site_density_post_write_std": float(np.std(site_den_pw)),
            "site_density_pre_go_mean": float(np.mean(site_den_pg)),
            "site_density_pre_go_std": float(np.std(site_den_pg)),
        }

elapsed = time.perf_counter() - t0
print(f"\nElapsed: {elapsed:.1f}s")

# Write raw JSON
out_path = os.path.join(os.path.dirname(__file__), "raw", "untrained_gate_replication.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({"elapsed_s": elapsed, "results": all_results}, f, indent=2)
print(f"Saved to {out_path}")

# Summary
print("\n=== Per-Seed Summary ===")
print(f"{'Model':<10} {'Seed':>4} {'PW_Prev':>8} {'PG_Prev':>8} {'PW_Den':>8} {'PG_Den':>8}")
for model_type in MODEL_TYPES:
    for seed in SEEDS:
        label = f"{model_type}_seed{seed}"
        r = all_results[label]
        print(f"{model_type:<10} {seed:>4} {r['state_prevalence_post_write']:>8.3f} {r['state_prevalence_pre_go']:>8.3f} "
              f"{r['site_density_post_write_mean']:>8.4f} {r['site_density_pre_go_mean']:>8.4f}")

# Gate verdict
for model_type in MODEL_TYPES:
    seed0_label = f"{model_type}_seed0"
    untrained_prev = all_results[seed0_label]["state_prevalence_post_write"]
    is_exactly_one = abs(untrained_prev - 1.0) < 1e-12
    verdict = "STRUCTURALLY_IMPASSABLE" if is_exactly_one else "SATURATED_OR_POORLY_IDENTIFIED"
    print(f"\n{model_type} untrained prevalence (seed 0): {untrained_prev}")
    print(f"  Exact 1.0? {is_exactly_one}")
    print(f"  Gate verdict: {verdict}")
