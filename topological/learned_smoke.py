"""Result-blind resource and path smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from typing import Any, Iterable

import numpy as np
import torch

from ._artifacts import WriteOnceArtifact, verify_manifest
from .interventions import (
    component_intervention,
    complex_to_hidden,
    fit_pca,
    fourier_field_intervention,
    hidden_to_complex,
    matched_global_phase,
    matched_zero_charge_phase,
    pca_field_intervention,
    random_direction_intervention,
    state_displacement,
)
from .learned_evaluation import select_donor_pair
from .model import ModelSpec
from .task import continue_copy, donor_sequences, generate_copy_batch, run_copy, write_copy
from .training import configure_determinism, make_model, one_training_step


CONTRACT = "PM1-LEARNED-V1"
TOTAL_CANONICAL_UPDATES = 10 * 30_000
WALL_LIMIT_SECONDS = 12.0 * 60.0 * 60.0
VRAM_LIMIT_BYTES = 14 * 1024**3
FORBIDDEN_RESULT_KEYS = (
    "loss",
    "accuracy",
    "logit",
    "charge",
    "persistence",
    "margin",
    "advantage",
    "checkpoint",
    "parameter_value",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_smoke_root() -> Path:
    data_root = Path(os.environ.get("RESEARCH_DATA_ROOT", "/mnt/r/research-data")).expanduser()
    return data_root / "aligned-mask-transplant" / "pm1" / "learned-v1" / "smoke-run-03"


def canonical_config_path() -> Path:
    return _project_root() / "configs" / "topological_learned_v1.json"


def _expected_config() -> dict[str, Any]:
    return {
        "artifact_root": "/mnt/r/research-data/aligned-mask-transplant/pm1/learned-v1",
        "batch_size": 64,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20_260_722,
        "channels": 8,
        "contract": CONTRACT,
        "copy_length": 4,
        "gradient_clip": 1.0,
        "grid": [16, 16],
        "heldout_delay": 64,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "pair_donors": 8,
        "recipients": 128,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "smoke_batch_size": 8,
        "smoke_delay": 16,
        "smoke_updates": 20,
        "train_delay": [16, 32],
        "updates_per_seed": 30_000,
        "validation_examples": 512,
        "validation_interval": 2_000,
        "vocabulary": 10,
        "vram_limit_bytes": VRAM_LIMIT_BYTES,
        "wall_limit_seconds": int(WALL_LIMIT_SECONDS),
    }


def require_canonical_config(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_config():
        raise RuntimeError("learned config differs from the frozen code contract")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "configs/topological_learned_v1.json",
        "scripts/run_topological_smoke.sh",
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


def _walk_keys(payload: Any) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield str(key).lower()
            yield from _walk_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from _walk_keys(value)


def assert_result_blind(root: Path) -> None:
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in _walk_keys(payload):
            if any(term in key for term in FORBIDDEN_RESULT_KEYS):
                raise RuntimeError(f"smoke invariant contains forbidden result key: {key}")


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _rotate(hidden: torch.Tensor, phase: float) -> torch.Tensor:
    cosine = float(np.cos(phase))
    sine = float(np.sin(phase))
    return torch.stack(
        (cosine * hidden[:, 0] - sine * hidden[:, 1], sine * hidden[:, 0] + cosine * hidden[:, 1]),
        dim=1,
    )


def _warm_device(device: torch.device, model_type: str = "u1") -> None:
    """Pay one-time backend initialization on a disposable model."""

    configure_determinism(0)
    model = make_model(0, model_type=model_type, model_spec=ModelSpec(), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    batch = generate_copy_batch(0, "smoke/backend-warmup", 8, 16, device=device)
    one_training_step(model, optimizer, batch, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _exercise_device(device: torch.device, model_type: str = "u1") -> dict[str, Any]:
    seed = 0
    updates = 20
    batch_size = 8
    delay = 16
    configure_determinism(seed)
    model = make_model(seed, model_type=model_type, model_spec=ModelSpec(), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    content_hashes: list[str] = []
    for update in range(1, updates + 1):
        batch = generate_copy_batch(seed, f"smoke/train/{update}", batch_size, delay, device=device)
        one_training_step(model, optimizer, batch, 1.0)
        content_hashes.append(batch.content_sha256)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    evaluation = generate_copy_batch(seed, "smoke/evaluation", batch_size, delay, device=device)
    recipient_trace = run_copy(model, evaluation.symbols, delay)
    donor_catalog = donor_sequences(evaluation.symbols)
    donor_hidden = write_copy(model, donor_catalog.reshape(batch_size * 8, 4)).detach().cpu()
    selected = None
    for recipient_index in range(batch_size):
        start_index = recipient_index * 8
        selected = select_donor_pair(
            recipient_trace.post_write[recipient_index].detach().cpu(),
            [donor_hidden[start_index + offset] for offset in range(8)],
        )
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("smoke invariant found no admissible decomposition pair")
    recipient = selected.recipient
    donor = selected.donor
    recipient_field = component_intervention(recipient, recipient, "natural_recipient")
    vortex_field = component_intervention(recipient, donor, "vortex")
    target = state_displacement(recipient_field, vortex_field)
    global_control = matched_global_phase(recipient_field, target).field
    zero_charge_control = matched_zero_charge_phase(recipient_field, target, control_index=0).field
    donor_field_complex = component_intervention(donor, donor, "natural_recipient")
    fourier_low_field = fourier_field_intervention(recipient_field, donor_field_complex, "fourier_low")
    fourier_high_field = fourier_field_intervention(recipient_field, donor_field_complex, "fourier_high")
    random_direction_field = random_direction_intervention(recipient_field, donor_field_complex, seed=0)
    donor_fields = [hidden_to_complex(donor_hidden[start_index + offset]) for offset in range(8)]
    donor_fields.append(recipient_field)
    pca_smoke = fit_pca(donor_fields, k=4)
    pca_field = pca_field_intervention(recipient_field, donor_field_complex, pca_smoke)
    fields = {
        "natural_recipient": recipient_field,
        "natural_donor": donor_field_complex,
        "vortex": vortex_field,
        "smooth": component_intervention(recipient, donor, "smooth"),
        "magnitude": component_intervention(recipient, donor, "magnitude"),
        "global_phase": global_control,
        "zero_phase": zero_charge_control,
        "whole_phase": component_intervention(recipient, donor, "whole_phase"),
        "whole_state": component_intervention(recipient, donor, "whole_state"),
        "fourier_low": fourier_low_field,
        "fourier_high": fourier_high_field,
        "pca": pca_field,
        "random_direction": random_direction_field,
    }
    continuation_shapes = set()
    for field in fields.values():
        hidden = complex_to_hidden(field, device=device).unsqueeze(0)
        output, _ = continue_copy(model, hidden, delay)
        if not torch.isfinite(output).all():
            raise RuntimeError("smoke invariant produced non-finite continuation output")
        continuation_shapes.add(tuple(output.shape))

    is_u1 = model_type == "u1"
    blank = torch.zeros((1,), dtype=torch.long, device=device)
    probe = selected.recipient_hidden.unsqueeze(0).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    phase = 0.417
    measured_equivariance = torch.allclose(
        model.step(blank, _rotate(probe, phase)),
        _rotate(model.step(blank, probe), phase),
        atol=2e-5,
        rtol=2e-5,
    )
    equivariance_matches_design = bool(measured_equivariance) == is_u1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_vram = 0
    guards = {
        "finite_training_steps": True,
        "blank_embedding_zero": bool(torch.count_nonzero(model.token_embedding.weight[0]) == 0),
        "state_shape_exact": tuple(recipient_trace.post_write.shape) == (8, 2, 8, 16, 16),
        "continuation_shape_exact": continuation_shapes == {(1, 4, 10)},
        "intervention_arms_complete": len(fields) == 13,
        "equivariance_matches_design": equivariance_matches_design,
        "split_namespaces_unique": len(set(content_hashes)) == updates,
    }
    if not all(guards.values()):
        failed = [name for name, value in guards.items() if not value]
        raise RuntimeError(f"smoke invariant invariant failure on {device} {model_type}: {failed}")
    return {
        "device": device.type,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "guards": guards,
        "split_sha256": hashlib.sha256("".join(content_hashes).encode()).hexdigest(),
        "parameter_count": model.parameter_count(),
        "state_bytes_per_sample": model.real_state_bytes(),
        "optimizer_steps": updates,
        "recurrent_steps_per_continuation": delay + 4,
        "readout_calls_per_continuation": 4,
        "intervention_arm_count": len(fields),
    }


def _exercise_all(device: torch.device) -> dict[str, Any]:
    cpu_u1 = _exercise_device(device, model_type="u1")
    cpu_plain = _exercise_device(device, model_type="plain")
    return {
        "u1": cpu_u1,
        "plain": cpu_plain,
        "diagnostic_warmup_steps_per_device": 1,
        "projected_seconds_u1": cpu_u1["elapsed_seconds"] / cpu_u1["optimizer_steps"] * TOTAL_CANONICAL_UPDATES,
        "projected_seconds_plain": cpu_plain["elapsed_seconds"] / cpu_plain["optimizer_steps"] * TOTAL_CANONICAL_UPDATES,
    }


def run_smoke(root: Path, *, require_cuda: bool = True) -> dict[str, Any]:
    destination = root.resolve()
    if destination.exists():
        raise FileExistsError(f"smoke invariant root already exists: {destination}")
    config_sha256 = require_canonical_config(canonical_config_path())
    _warm_device(torch.device("cpu"), model_type="u1")
    _warm_device(torch.device("cpu"), model_type="plain")
    cpu = _exercise_all(torch.device("cpu"))
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        _warm_device(torch.device("cuda"), model_type="u1")
        _warm_device(torch.device("cuda"), model_type="plain")
        cuda = _exercise_all(torch.device("cuda"))
    else:
        cuda = None
    timed = cuda["u1"] if cuda is not None else cpu["u1"]
    projected_seconds = timed["elapsed_seconds"] / timed["optimizer_steps"] * TOTAL_CANONICAL_UPDATES
    timed_plain = cuda["plain"] if cuda is not None else cpu["plain"]
    projected_seconds_plain = timed_plain["elapsed_seconds"] / timed_plain["optimizer_steps"] * TOTAL_CANONICAL_UPDATES
    resource_guards = {
        "cuda_available": cuda_available or not require_cuda,
        "projected_total_u1_under_wall_limit": projected_seconds < WALL_LIMIT_SECONDS,
        "projected_total_plain_under_wall_limit": projected_seconds_plain < WALL_LIMIT_SECONDS,
        "projected_combined_under_wall_limit": (
            projected_seconds + projected_seconds_plain
        ) < WALL_LIMIT_SECONDS,
        "peak_vram_under_fourteen_gib": cuda is not None
        and cuda["u1"]["peak_vram_bytes"] < VRAM_LIMIT_BYTES
        and cuda["plain"]["peak_vram_bytes"] < VRAM_LIMIT_BYTES,
        "peak_rss_under_fourteen_gib": peak_rss_bytes() < VRAM_LIMIT_BYTES,
    }
    promoted = all(resource_guards.values())
    status = "PROMOTE_PM1_LEARNED_PILOT" if promoted else "PM1_LEARNED_RESOURCE_NO_GO"
    writer = WriteOnceArtifact(destination)
    writer.write_json("config.json", {"contract": CONTRACT, "config_sha256": config_sha256})
    writer.write_json("source.json", _source_payload())
    writer.write_json(
        "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        },
    )
    writer.write_json(
        "work.json",
        {
            "cpu": cpu,
            "cuda": cuda,
            "projected_seconds": projected_seconds,
            "projected_seconds_plain": projected_seconds_plain,
            "peak_rss_bytes": peak_rss_bytes(),
            "resource_guards": resource_guards,
        },
    )
    writer.write_json("status.json", {"status": status})
    writer.finalize()
    assert_result_blind(destination)
    if not verify_manifest(destination):
        raise RuntimeError("smoke invariant manifest verification failed")
    return {"status": status, "manifest_verified": True, "guard_count": len(resource_guards)}


def require_promotion(root: Path) -> None:
    destination = root.resolve()
    if not verify_manifest(destination):
        raise RuntimeError("smoke invariant manifest is invalid")
    assert_result_blind(destination)
    status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PROMOTE_PM1_LEARNED_PILOT":
        raise RuntimeError("smoke invariant did not authorize the canonical pilot")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_smoke(canonical_smoke_root()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
