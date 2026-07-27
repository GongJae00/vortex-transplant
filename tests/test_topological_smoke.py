from __future__ import annotations

import pytest

from topological._artifacts import verify_manifest
from topological.smoke import require_promotion, run_smoke


def test_pm1_smoke_is_result_blind_write_once_and_tamper_evident(tmp_path) -> None:
    root = tmp_path / "smoke"
    receipt = run_smoke(root)

    assert receipt["status"] == "PROMOTE_PM1_FEASIBILITY_RUN"
    assert verify_manifest(root)
    require_promotion(root)
    with pytest.raises(FileExistsError):
        run_smoke(root)
    (root / "status.json").write_text("{}\n", encoding="utf-8")
    assert not verify_manifest(root)
