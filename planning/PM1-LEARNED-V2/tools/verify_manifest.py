#!/usr/bin/env python3
"""Verify SHA-256 manifest against actual file content."""

import hashlib
import json
import os
import sys

PLAN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA256_PATH = os.path.join(PLAN_DIR, "diagnostics", "sha256.json")
SELF_PATH = os.path.relpath(__file__, PLAN_DIR)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_manifest():
    if not os.path.isfile(SHA256_PATH):
        return False, [f"Manifest not found at {SHA256_PATH}"], []

    try:
        with open(SHA256_PATH) as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Manifest JSON parse error: {e}"], []

    errors = []
    warnings = []
    verified = 0
    missing = 0
    mismatch = 0

    for rel_path, expected_hash in sorted(manifest.items()):
        abs_path = os.path.join(PLAN_DIR, rel_path)
        if not os.path.isfile(abs_path):
            errors.append(f"Manifested file missing: {rel_path}")
            missing += 1
            continue

        try:
            actual_hash = sha256_file(abs_path)
        except IOError as e:
            errors.append(f"Cannot hash {rel_path}: {e}")
            continue

        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {rel_path}: expected {expected_hash[:16]}..., got {actual_hash[:16]}...")
            mismatch += 1
        else:
            verified += 1

    if SELF_PATH in manifest:
        if os.path.abspath(__file__) != os.path.join(PLAN_DIR, SELF_PATH):
            warnings.append("SELF_PATH resolved differently on disk")
    else:
        verified += 1

    detail = f"{verified} verified, {missing} missing, {mismatch} mismatched"
    all_pass = len(errors) == 0
    return all_pass, detail, warnings, errors


def main():
    all_pass, detail, warnings, errors = verify_manifest()
    report = {"pass": all_pass, "detail": detail, "warnings": warnings, "errors": errors}
    print(json.dumps(report, indent=2))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
