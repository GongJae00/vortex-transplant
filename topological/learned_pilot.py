"""One-shot ten-seed executor for the learned causal pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Callable

import numpy as np
import torch

from ._artifacts import WriteOnceArtifact, verify_manifest
from .learned_evaluation import decide_learned_pilot, evaluate_seed_model
from .learned_smoke import (
    VRAM_LIMIT_BYTES,
    WALL_LIMIT_SECONDS,
    canonical_config_path,
    canonical_smoke_root,
    require_canonical_config,
    require_promotion,
)
from .model import ModelSpec
from .training import TrainingSpec, make_model, train_seed


CONTRACT = "PM1-LEARNED-V1"
PILOT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
PILOT_MODEL_TYPES = ("u1", "plain")
RSS_LIMIT_BYTES = 14 * 1024**3


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_pilot_root() -> Path:
    data_root = Path(os.environ.get("RESEARCH_DATA_ROOT", "/mnt/r/research-data")).expanduser()
    return data_root / "aligned-mask-transplant" / "pm1" / "learned-v1" / "pilot-run-02"


def canonical_contract_path() -> Path:
    return (
        Path.home()
        / "research"
        / "private-projects"
        / "aligned-mask-transplant"
        / "ledgers"
        / "EXPERIMENT_EVALUATION_CONTRACT.md"
    )


def _source_payload() -> dict[str, Any]:
    root = _project_root()
    paths = (
        "topological/model.py",
        "topological/task.py",
        "topological/training.py",
        "topological/topology.py",
        "topological/decomposition.py",
        "topological/interventions.py",
        "topological/learned_evaluation.py",
        "topological/learned_smoke.py",
        "topological/learned_pilot.py",
        "configs/topological_learned_v1.json",
        "scripts/run_topological_smoke.sh",
        "scripts/run_topological_pilot.sh",
    )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "git_commit": commit,
        "files": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    }


def _environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_available": cuda_available,
    }


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def initialize_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("learned canonical pilot requires the admitted CUDA device")
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    return torch.device("cuda:0")


def hash_chain(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_bytes(model: torch.nn.Module, metadata: dict[str, Any]) -> bytes:
    state = {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }
    buffer = io.BytesIO()
    torch.save({"metadata": metadata, "state_dict": state}, buffer)
    return buffer.getvalue()


def checkpoint_roundtrip_exact(payload: bytes, seed: int, model_type: str = "u1") -> bool:
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    clone = make_model(seed, model_type=model_type, model_spec=ModelSpec(), device="cpu")
    clone.load_state_dict(saved["state_dict"])
    return all(
        torch.equal(value, saved["state_dict"][name])
        for name, value in clone.state_dict().items()
    )


def _training_record(result: Any, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "selected_update": result.selected_update,
        "selected_accuracy": result.selected_accuracy,
        "selected_cross_entropy": result.selected_cross_entropy,
        "history": [asdict(record) for record in result.history],
        "training_log": [asdict(snapshot) for snapshot in result.training_log],
        "update_count": result.update_count,
        "finite_gradient_steps": result.finite_gradient_steps,
        "train_batch_count": len(result.train_hashes),
        "train_hash_chain_sha256": hash_chain(result.train_hashes),
        "elapsed_seconds": elapsed_seconds,
    }


def run_canonical_pilot(
    root: Path,
    *,
    model_types: tuple[str, ...] = PILOT_MODEL_TYPES,
    task_type: str = "copy",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = root.resolve()
    if destination.exists():
        raise FileExistsError(f"learned pilot root already exists: {destination}")
    require_promotion(canonical_smoke_root())
    config_sha256 = require_canonical_config(canonical_config_path())
    writer = WriteOnceArtifact(destination)
    writer.write_json("config.json", json.loads(canonical_config_path().read_text(encoding="utf-8")))
    writer.write_text("contract.md", canonical_contract_path().read_text(encoding="utf-8"))
    writer.write_json("source.json", _source_payload())
    writer.write_json("environment.json", _environment())
    writer.write_json(
        "lineage.json",
        {
            "contract": CONTRACT,
            "config_sha256": config_sha256,
            "promotion_root": str(canonical_smoke_root()),
            "invalid_resource_run": "/mnt/r/research-data/aligned-mask-transplant/pm1/learned-v1/smoke",
            "invalid_pilot_run": "/mnt/r/research-data/aligned-mask-transplant/pm1/learned-v1/pilot",
        },
    )

    device = initialize_cuda_device()
    model_spec = ModelSpec()
    training_spec = TrainingSpec()
    total_start = time.perf_counter()

    untrained_records: dict[str, dict[str, Any]] = {}
    for model_type in model_types:
        untrained_model = make_model(0, model_type=model_type, model_spec=model_spec, device=device)
        untrained_records[model_type] = evaluate_seed_model(untrained_model, 0, task_type=task_type)
        writer.write_json(f"models/{model_type}/untrained_diagnostic.json", untrained_records[model_type])
        del untrained_model
        torch.cuda.empty_cache()
        if progress is not None:
            progress(f"{model_type} untrained diagnostic complete")

    model_seed_records: dict[str, list[dict[str, Any]]] = {mt: [] for mt in model_types}
    for model_type in model_types:
        if progress is not None:
            progress(f"training {model_type} models")
        for seed in PILOT_SEEDS:
            seed_start = time.perf_counter()
            training = train_seed(
                seed,
                model_type=model_type,
                task_type=task_type,
                model_spec=model_spec,
                training_spec=training_spec,
                device=device,
                progress=progress,
            )
            training_elapsed = time.perf_counter() - seed_start
            training_record = _training_record(training, training_elapsed)
            payload = checkpoint_bytes(
                training.model,
                {
                    "contract": CONTRACT,
                    "model_type": model_type,
                    "seed": seed,
                    "selected_update": training.selected_update,
                },
            )
            roundtrip = checkpoint_roundtrip_exact(payload, seed, model_type=model_type)
            if not roundtrip:
                raise RuntimeError(f"checkpoint roundtrip failed for {model_type} seed {seed}")
            prefix = f"models/{model_type}/seeds/{seed}"
            writer.write_bytes(f"{prefix}/checkpoint.pt", payload)
            writer.write_json(f"{prefix}/training.json", training_record)
            writer.write_json(
                f"{prefix}/train_hashes.json",
                {"content_sha256": training.train_hashes},
            )
            evaluation = evaluate_seed_model(training.model, seed, task_type=task_type)
            evaluation["checkpoint_roundtrip_exact"] = roundtrip
            writer.write_json(f"{prefix}/evaluation.json", evaluation)
            model_seed_records[model_type].append(evaluation)
            if progress is not None:
                progress(f"{model_type} seed {seed} held-out causal evaluation complete")
            del training
            torch.cuda.empty_cache()

    scientific = decide_learned_pilot(
        model_seed_records,
        model_types=model_types,
        untrained_records=untrained_records,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - total_start
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    rss = peak_rss_bytes()
    all_roundtrips = all(
        record["checkpoint_roundtrip_exact"]
        for records in model_seed_records.values()
        for record in records
    )
    resource_clauses = {
        "total_wall_under_limit": elapsed < WALL_LIMIT_SECONDS,
        "peak_vram_under_fourteen_gib": peak_vram < VRAM_LIMIT_BYTES,
        "peak_rss_under_fourteen_gib": rss < RSS_LIMIT_BYTES,
        "all_checkpoint_roundtrips": all_roundtrips,
    }
    if not all(resource_clauses.values()):
        status = "PM1_LEARNED_RESOURCE_NO_GO"
    else:
        status = scientific["status"]
    decision = {
        **scientific,
        "status": status,
        "clauses": {**scientific["clauses"], **resource_clauses},
    }
    writer.write_json("seed_records.json", model_seed_records)
    writer.write_json(
        "replay.json",
        {
            "seed_records_sha256": payload_sha256(model_seed_records),
            "test_content_sha256": {
                model_type: [record["test_content_sha256"] for record in records]
                for model_type, records in model_seed_records.items()
            },
            "scientific_reexecution": False,
        },
    )
    writer.write_json(
        "resource.json",
        {
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": peak_vram,
            "peak_rss_bytes": rss,
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "vram_limit_bytes": VRAM_LIMIT_BYTES,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
        },
    )
    writer.write_json("decision.json", decision)
    writer.write_json(
        "receipt.json",
        {
            "contract": CONTRACT,
            "status": status,
            "interpretation_boundary": (
                "One controlled copy-task causal pilot only; survival is not final novelty, "
                "independent-task transfer, manuscript, venue, or submission evidence."
            ),
            "strongest_residual_confound": (
                "A component transplant can leave the learned-state manifold even when "
                "energy, spectrum, and one-step commutation guards pass."
            ),
            "cleanup_state": (
                "Canonical checkpoints and evidence retained; no raw data, tracked file, "
                "checkpoint, or scientific output deleted."
            ),
        },
    )
    writer.finalize()
    if not verify_manifest(destination):
        raise RuntimeError("learned pilot manifest verification failed")
    return {
        "status": status,
        "manifest_verified": True,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "peak_rss_bytes": rss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run_canonical_pilot(canonical_pilot_root(), progress=print)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
