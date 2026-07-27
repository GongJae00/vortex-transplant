from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class WriteOnceArtifact:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self._finalized = False

    def write_json(self, relative: str, payload: Any) -> Path:
        if self._finalized:
            raise RuntimeError("artifact is already finalized")
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path == self.root:
            raise ValueError("artifact path escapes its root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return path

    def write_text(self, relative: str, payload: str) -> Path:
        if self._finalized:
            raise RuntimeError("artifact is already finalized")
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path == self.root:
            raise ValueError("artifact path escapes its root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        return path

    def write_bytes(self, relative: str, payload: bytes) -> Path:
        if self._finalized:
            raise RuntimeError("artifact is already finalized")
        path = (self.root / relative).resolve()
        if self.root not in path.parents or path == self.root:
            raise ValueError("artifact path escapes its root")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
        return path

    def finalize(self) -> Path:
        if self._finalized:
            raise RuntimeError("artifact is already finalized")
        records: dict[str, str] = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "sha256.json":
                relative = path.relative_to(self.root).as_posix()
                records[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = self.write_json(
            "sha256.json",
            {"algorithm": "sha256", "files": records, "verified_count": len(records)},
        )
        self._finalized = True
        return manifest


def verify_manifest(root: Path) -> bool:
    manifest_path = root / "sha256.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["files"]
    if payload.get("verified_count") != len(records):
        return False
    for relative, expected in records.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False
    return True
