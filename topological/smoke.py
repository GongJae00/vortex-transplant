"""Result-blind smoke gate for feasibility pilot."""

from __future__ import annotations

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

from ._artifacts import WriteOnceArtifact, verify_manifest
from .decomposition import decompose
from .fixture import generate_fields


FORBIDDEN_RESULT_KEYS = (
    "metric",
    "error",
    "energy",
    "spectrum",
    "decision",
    "feasible",
    "no_go",
    "loss",
    "accuracy",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_smoke_root() -> Path:
    data_root = Path(os.environ.get("RESEARCH_DATA_ROOT", "/mnt/r/research-data")).expanduser()
    return data_root / "aligned-mask-transplant" / "pm1" / "feasibility-v1" / "smoke"


def _source_payload() -> dict[str, Any]:
    root = _project_root()
    paths = (
        "topological/topology.py",
        "topological/decomposition.py",
        "topological/fixture.py",
        "topological/evaluation.py",
        "topological/smoke.py",
        "topological/pilot.py",
        "configs/topological_feasibility_v1.json",
        "scripts/run_topological_feasibility.sh",
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
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "processor": platform.processor(),
        "device": "cpu",
    }


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


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
                raise RuntimeError(f"result-blind smoke contains forbidden key: {key}")


def run_smoke(root: Path) -> dict[str, Any]:
    destination = root.resolve()
    if destination.exists():
        raise FileExistsError(f"smoke root already exists: {destination}")
    start = time.perf_counter()
    first = generate_fields(0)[:4]
    second = generate_fields(0)[:4]
    parts = [decompose(record.field) for record in first]
    guards = {
        "field_count_four": len(first) == 4,
        "namespace_replay_exact": all(
            np.array_equal(left.field, right.field)
            for left, right in zip(first, second, strict=True)
        ),
        "signed_pair_roundtrip": all(
            np.array_equal(record.charge, component.charge.charge)
            for record, component in zip(first, parts, strict=True)
        ),
        "periodic_net_charge_zero": all(component.charge.net_charge == 0 for component in parts),
        "compact_roundtrip": all(component.reconstruction_error <= 1e-10 for component in parts),
        "smooth_zero_charge": all(component.smooth_charge_residual <= 1e-10 for component in parts),
        "finite_components": all(
            np.isfinite(component.magnitude).all()
            and np.isfinite(component.vortex).all()
            and np.isfinite(component.smooth).all()
            for component in parts
        ),
    }
    elapsed = time.perf_counter() - start
    projected_seconds = elapsed * (5 * 32) / 4
    guards.update(
        {
            "projected_under_budget": projected_seconds < 20.0 * 60.0,
            "rss_under_budget": peak_rss_bytes() < 4 * 1024**3,
        }
    )
    if not all(guards.values()):
        failed = [key for key, value in guards.items() if not value]
        raise RuntimeError(f"result-blind smoke failed: {failed}")
    writer = WriteOnceArtifact(destination)
    writer.write_json(
        "config.json",
        {"contract": "PM1-FEASIBILITY-V1", "device": "cpu", "seed": 0, "field_count": 4},
    )
    writer.write_json("source.json", _source_payload())
    writer.write_json("environment.json", _environment())
    writer.write_json(
        "guards.json",
        {
            "guards": guards,
            "elapsed_seconds": elapsed,
            "projected_seconds": projected_seconds,
            "peak_rss_bytes": peak_rss_bytes(),
        },
    )
    writer.write_json("status.json", {"status": "PROMOTE_PM1_FEASIBILITY_RUN"})
    writer.finalize()
    assert_result_blind(destination)
    if not verify_manifest(destination):
        raise RuntimeError("smoke manifest verification failed")
    return {"status": "PROMOTE_PM1_FEASIBILITY_RUN", "manifest_verified": True, "guard_count": len(guards)}


def require_promotion(root: Path) -> None:
    destination = root.resolve()
    if not verify_manifest(destination):
        raise RuntimeError("smoke manifest is invalid")
    assert_result_blind(destination)
    status = json.loads((destination / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PROMOTE_PM1_FEASIBILITY_RUN":
        raise RuntimeError("smoke did not authorize the feasibility run")
