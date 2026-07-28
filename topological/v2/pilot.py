"""V2 calibration pilot — executes calibration experiments in VOI order.

Contract: calibration split authorized, results NOT confirmatory evidence.

Optimized: uses bfloat16 AMP + cuDNN benchmark for 2-3x faster training.
"""
import hashlib, json, time, os, sys
from pathlib import Path
from typing import Any, Callable
import torch
import numpy as np

from ..model import ModelSpec
from .v2_model import make_scalar_u1_model
from .optimized_training import (
    configure_optimized, train_seed_optimized, OptimizedTrainingSpec,
)
from ..task import generate_copy_batch, run_copy
from ..interventions import hidden_to_complex
from .v2_topology import compute_branch_margins, per_channel_defect_prevalence
from .._artifacts import WriteOnceArtifact


def run_c05_trainability_screen(
    output_dir: Path,
    device: torch.device,
    n_seeds: int = 5,
    n_updates: int = 5000,
) -> dict:
    """C05: C=1 trainability screen (optimized).

    Uses AMP + cuDNN benchmark for ~2x faster training.
    """
    spec = ModelSpec(channels=1)
    train_spec = OptimizedTrainingSpec(
        updates=n_updates,
        validation_examples=128,   # reduced
        validation_interval=1000,  # frequent for screening
        use_amp=True,
        deterministic=False,
    )

    writer = WriteOnceArtifact(output_dir)
    results = {"seeds": {}, "overall_pass": False}

    for seed in range(n_seeds):
        configure_optimized(seed, deterministic=False)
        model = make_scalar_u1_model(model_spec=spec, device=device)
        training = train_seed_optimized(
            seed, model_type="u1", model_spec=spec,
            spec=train_spec, device=device,
        )
        acc = float(training.selected_accuracy)
        results["seeds"][f"seed_{seed}"] = {
            "accuracy": acc,
            "selected_update": training.selected_update,
        }
        del model, training
        if device.type == "cuda":
            torch.cuda.empty_cache()

    accuracies = [r["accuracy"] for r in results["seeds"].values()]
    mean_acc = float(np.mean(accuracies))
    results["mean_accuracy"] = mean_acc
    results["min_accuracy"] = float(np.min(accuracies))
    results["overall_pass"] = mean_acc >= 0.90

    writer.write_json("results.json", results)
    writer.finalize()
    return results


def run_c06_topology_emergence(
    output_dir: Path,
    device: torch.device,
    n_seeds: int = 5,
    n_updates: int = 5000,
) -> dict:
    """C06: C=1 topology emergence (optimized)."""
    spec = ModelSpec(channels=1)
    train_spec = OptimizedTrainingSpec(
        updates=n_updates,
        validation_examples=128,
        validation_interval=1000,
        use_amp=True,
        deterministic=False,
    )

    writer = WriteOnceArtifact(output_dir)
    results = {"seeds": {}}

    for seed in range(n_seeds):
        configure_optimized(seed, deterministic=False)
        model = make_scalar_u1_model(model_spec=spec, device=device)
        training = train_seed_optimized(
            seed, model_type="u1", model_spec=spec,
            spec=train_spec, device=device,
        )

        # Analyze topology on test examples (no full evaluate_seed_model — too slow)
        TEST_EXAMPLES = 128; TEST_DELAY = 64
        batch = generate_copy_batch(seed, "cal/topology", TEST_EXAMPLES, TEST_DELAY, device=device)
        trace = run_copy(training.model, batch.symbols, TEST_DELAY)
        hidden = trace.post_write.detach().cpu()

        topologies = []
        for ex in range(TEST_EXAMPLES):
            f = hidden_to_complex(hidden[ex])
            margins = compute_branch_margins(f[0])
            prev = per_channel_defect_prevalence(f)
            topologies.append({
                "branch_min": margins.min_margin,
                "branch_q01": margins.q01_margin,
                "branch_median": margins.median_margin,
                "prevalence": prev[0],
            })

        results["seeds"][f"seed_{seed}"] = {
            "accuracy": float(training.selected_accuracy),
            "mean_branch_min": float(np.mean([t["branch_min"] for t in topologies])),
            "mean_branch_median": float(np.mean([t["branch_median"] for t in topologies])),
            "prevalence": float(np.mean([t["prevalence"] for t in topologies])),
        }
        del model, training
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    prev_values = [r["prevalence"] for r in results["seeds"].values()]
    branch_values = [r["mean_branch_min"] for r in results["seeds"].values()]
    results["summary"] = {
        "mean_prevalence": float(np.mean(prev_values)),
        "mean_branch_min": float(np.mean(branch_values)),
        "defect_emerged": float(np.mean(prev_values)) > 0.5,
    }

    writer.write_json("results.json", results)
    writer.finalize()
    return results


def run_calibration_phase(
    output_root: Path,
    device: torch.device,
    experiments: list[str] | None = None,
) -> dict:
    """Execute calibration experiments in VOI order.

    Args:
        output_root: root directory for calibration artifacts
        device: torch device
        experiments: list of experiment IDs to run (default: all)
    """
    if experiments is None:
        experiments = ["C05", "C06"]

    results = {}
    for exp_id in experiments:
        exp_dir = output_root / exp_id
        print(f"\n=== {exp_id} ===")
        t0 = time.perf_counter()

        if exp_id == "C05":
            results[exp_id] = run_c05_trainability_screen(exp_dir, device)
        elif exp_id == "C06":
            results[exp_id] = run_c06_topology_emergence(exp_dir, device)
        else:
            print(f"  SKIP: {exp_id} not implemented")

        elapsed = time.perf_counter() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Pass: {results[exp_id].get('overall_pass', 'N/A')}")

    return results
