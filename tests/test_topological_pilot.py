from __future__ import annotations

import json

import pytest

from topological.pilot import (
    _expected_config,
    canonical_config_path,
    replay_hash,
    require_canonical_config,
)


def test_public_pm1_config_matches_frozen_contract(tmp_path) -> None:
    path = canonical_config_path()

    assert json.loads(path.read_text(encoding="utf-8")) == _expected_config()
    assert len(require_canonical_config(path)) == 64
    changed = _expected_config()
    changed["hybrid_tolerance"] = 0.051
    invalid = tmp_path / "changed.json"
    invalid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen"):
        require_canonical_config(invalid)


def test_scientific_replay_hash_is_deterministic() -> None:
    assert replay_hash() == replay_hash()
