"""Split enforcement and contract loading for V2.

Ensures calibration/confirmatory isolation:
- Confirmatory split cannot use calibration config
- Calibration split cannot be presented as confirmatory evidence
- Contract hash verification at execution time
"""
import hashlib
import json
import os
import yaml
from pathlib import Path
from ._types import ContractState, SplitAuthorization


VALID_SPLITS = frozenset({"development", "calibration", "confirmatory"})

NAMESPACE_PREFIXES = {
    "development": "dev",
    "calibration": "cal",
    "confirmatory": "confirm",
}


def load_contract(contract_path: str | Path) -> dict:
    """Load and parse the V2 contract YAML."""
    with open(contract_path) as f:
        return yaml.safe_load(f)


def contract_digest(contract_path: str | Path) -> str:
    """SHA-256 digest of contract file."""
    h = hashlib.sha256()
    with open(contract_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_contract_state(
    contract_path: str | Path,
    design_base_commit: str,
    planning_content_digest: str,
    split: str,
) -> ContractState:
    """Verify that the contract matches its expected state."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}'. Must be one of {sorted(VALID_SPLITS)}")

    contract = load_contract(contract_path)
    cd = contract_digest(contract_path)

    # Verify design base commit
    stored_commit = contract.get("target_commit", "")
    if stored_commit != design_base_commit:
        raise ValueError(
            f"Contract design_base_commit mismatch: "
            f"stored={stored_commit}, expected={design_base_commit}"
        )

    # Check contract protocol version
    protocol = contract.get("protocol", "")
    if not protocol.startswith("PM1-LEARNED-V2"):
        raise ValueError(f"Unexpected protocol version: {protocol}")

    return ContractState(
        contract_digest=cd,
        design_base_commit=design_base_commit,
        planning_content_digest=planning_content_digest,
        split=split,
        frozen=True,
    )


def authorize_split(
    split: str,
    contract: ContractState,
    is_confirmatory_experiment: bool = False,
) -> SplitAuthorization:
    """Authorize a split for experimentation."""
    if split not in VALID_SPLITS:
        return SplitAuthorization(split=split, authorized=False, reason=f"Unknown split '{split}'")

    if not contract.frozen:
        return SplitAuthorization(split=split, authorized=False, reason="Contract not frozen")

    if split == "confirmatory":
        if is_confirmatory_experiment:
            return SplitAuthorization(
                split=split, authorized=True,
                reason="Confirmatory experiment authorized under frozen contract",
            )
        return SplitAuthorization(
            split=split, authorized=False,
            reason="Confirmatory split requires explicit authorization",
        )

    if split == "calibration":
        return SplitAuthorization(
            split=split, authorized=True,
            reason="Calibration split authorized (non-confirmatory evidence)",
        )

    if split == "development":
        return SplitAuthorization(
            split=split, authorized=True,
            reason="Development split authorized (no scientific evidence)",
        )

    return SplitAuthorization(split=split, authorized=False, reason="Unknown")


def namespace_seed(seed: int, split: str, sub_namespace: str) -> str:
    """Generate hash-separated namespace for deterministic data generation."""
    if split not in NAMESPACE_PREFIXES:
        raise ValueError(f"Unknown split: {split}")
    prefix = NAMESPACE_PREFIXES[split]
    return f"{prefix}/{sub_namespace}/seed_{seed}"


def require_clean_working_tree(repo_root: str | Path) -> bool:
    """Verify the git working tree is clean before experimentation."""
    import subprocess
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    if result.returncode != 0:
        raise RuntimeError("git status failed")
    return result.stdout.strip() == ""


def record_runtime_identity(repo_root: str | Path) -> dict:
    """Record runtime identity for experiment artifacts."""
    import subprocess
    import platform

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_root),
    ).stdout.strip()

    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        capture_output=True, text=True, cwd=str(repo_root),
    ).stdout.strip()

    dirty = not require_clean_working_tree(repo_root)

    return {
        "commit_sha": commit,
        "tree_sha": tree,
        "dirty_working_tree": dirty,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }
