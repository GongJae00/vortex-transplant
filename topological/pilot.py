"""One-shot immutable executor for feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np

from ._artifacts import WriteOnceArtifact, verify_manifest
from .evaluation import PILOT_SEEDS, decide_feasibility, evaluate_seed
from .fixture import FeasibilitySpec, generate_fields
from .smoke import canonical_smoke_root, peak_rss_bytes, require_promotion


WALL_LIMIT_SECONDS = 20.0 * 60.0
RSS_LIMIT_BYTES = 4 * 1024**3


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_pilot_root() -> Path:
    data_root = Path(os.environ.get("RESEARCH_DATA_ROOT", "/mnt/r/research-data")).expanduser()
    return data_root / "aligned-mask-transplant" / "pm1" / "feasibility-v1" / "pilot"


def canonical_config_path() -> Path:
    return _project_root() / "configs" / "topological_feasibility_v1.json"


def _expected_config() -> dict[str, Any]:
    return {
        "artifact_root": "/mnt/r/research-data/aligned-mask-transplant/pm1/feasibility-v1",
        "contract": "PM1-FEASIBILITY-V1",
        "device": "cpu",
        "field_count_per_seed": 32,
        "grid": [32, 32],
        "hybrid_tolerance": 0.05,
        "exact_tolerance": 1e-10,
        "magnitude_bounds": [0.75, 1.25],
        "seeds": list(PILOT_SEEDS),
        "smooth_link_amplitude": 0.10,
        "wall_limit_seconds": 1200.0,
        "rss_limit_bytes": 4 * 1024**3,
    }


def require_canonical_config(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_config():
        raise RuntimeError("config differs from the frozen code contract")
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    return {"git_commit": commit, "files": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}}


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "processor": platform.processor(),
        "device": "cpu",
    }


def _scientific_payload(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    spec = FeasibilitySpec()
    seed_records = []
    for seed in PILOT_SEEDS:
        seed_records.append(evaluate_seed(generate_fields(seed, spec)))
        if progress is not None:
            progress(f"feasibility seed {seed} complete")
    return decide_feasibility(seed_records)


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def replay_hash() -> str:
    return _payload_sha256(_scientific_payload())


def run_canonical_pilot(
    root: Path,
    *,
    smoke_root: Path | None = None,
    config_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    destination = root.resolve()
    if destination.exists():
        raise FileExistsError(f"pilot root already exists: {destination}")
    require_promotion((smoke_root or canonical_smoke_root()).resolve())
    config = (config_path or canonical_config_path()).resolve()
    config_sha256 = require_canonical_config(config)
    start = time.perf_counter()
    scientific = _scientific_payload(progress)
    scientific_sha256 = _payload_sha256(scientific)
    replay = subprocess.run(
        [sys.executable, "-m", "aligned_mask_transplant.pm1_pilot", "--replay-hash"],
        check=True,
        text=True,
        capture_output=True,
        cwd=_project_root(),
    ).stdout.strip()
    elapsed = time.perf_counter() - start
    rss = peak_rss_bytes()
    resource_clauses = {
        "runtime_under_budget": elapsed < WALL_LIMIT_SECONDS,
        "rss_under_budget": rss < RSS_LIMIT_BYTES,
        "clean_process_replay_exact": replay == scientific_sha256,
    }
    decision = {
        **scientific,
        "clauses": {**scientific["clauses"], **resource_clauses},
    }
    decision["status"] = (
        "PM1_DECOMPOSITION_FEASIBLE"
        if all(decision["clauses"].values())
        else "PM1_NO_GO_DECOMPOSITION"
    )
    writer = WriteOnceArtifact(destination)
    writer.write_json("config.json", json.loads(config.read_text(encoding="utf-8")))
    writer.write_json("source.json", _source_payload())
    writer.write_json("environment.json", _environment())
    writer.write_json("seed_records.json", scientific["seed_records"])
    writer.write_json(
        "replay.json",
        {"scientific_sha256": scientific_sha256, "clean_process_sha256": replay, "exact": replay == scientific_sha256},
    )
    writer.write_json(
        "resource.json",
        {"elapsed_seconds": elapsed, "peak_rss_bytes": rss, "wall_limit_seconds": WALL_LIMIT_SECONDS, "rss_limit_bytes": RSS_LIMIT_BYTES},
    )
    writer.write_json("decision.json", decision)
    writer.write_json(
        "receipt.json",
        {
            "contract": "PM1-FEASIBILITY-V1",
            "config_sha256": config_sha256,
            "status": decision["status"],
            "interpretation_boundary": "Mathematical decomposition feasibility only; no memory, novelty, performance, manuscript, or submission claim.",
            "strongest_residual_confound": "Synthetic compact fields do not establish learned-manifold validity or vortex-carried task information.",
        },
    )
    writer.finalize()
    if not verify_manifest(destination):
        raise RuntimeError("pilot manifest verification failed")
    return {
        "status": decision["status"],
        "manifest_verified": True,
        "scientific_sha256": scientific_sha256,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": rss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-hash", action="store_true")
    args = parser.parse_args()
    if args.replay_hash:
        print(replay_hash())
        return 0
    result = run_canonical_pilot(canonical_pilot_root(), progress=print)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
