#!/usr/bin/env python3
"""Validate PM1-LEARNED-V2 planning package integrity."""

import hashlib
import json
import os
import re
import sys

import yaml  # type: ignore

PLAN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PLAN_DIR, "00_INDEX.md")
CONTRACT_PATH = os.path.join(PLAN_DIR, "15_PM1_LEARNED_V2_CONTRACT.yaml")
FINDINGS_PATH = os.path.join(PLAN_DIR, "03_PRIOR_AUDIT_43_FINDINGS.md")
CORRECTION_PATH = os.path.join(PLAN_DIR, "02_DRAFT0_CORRECTION_LEDGER.md")
CAUSAL_PATH = os.path.join(PLAN_DIR, "07_CAUSAL_IDENTIFICATION.md")
SAP_PATH = os.path.join(PLAN_DIR, "08_STATISTICAL_ANALYSIS_PLAN.md")
DIAG_RAW_DIR = os.path.join(PLAN_DIR, "diagnostics", "raw")
OUTPUT_PATH = os.path.join(PLAN_DIR, "PLAN_VALIDATION.json")
SHA256_PATH = os.path.join(PLAN_DIR, "diagnostics", "sha256.json")

VALID_STATUSES = frozenset(("FROZEN_MATHEMATICAL", "FROZEN_ENGINEERING",
                             "UNFROZEN_CALIBRATION", "DERIVED"))
TBD_PATTERNS = re.compile(r"\b(TBD|placeholder|not executed|\?\?\?)\b", re.IGNORECASE)
MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
FINDING_PATTERN = re.compile(r"^### Finding (\d+)", re.MULTILINE)
CORRECTION_PATTERN = re.compile(r"^## C-(\d{2})", re.MULTILINE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _strip_backticks(s):
    s = s.strip()
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1]
    return s


def parse_index():
    with open(INDEX_PATH) as f:
        lines = f.readlines()

    entries = []
    in_table = False
    header_start = False
    for line in lines:
        if line.strip().startswith("| #"):
            header_start = True
            in_table = True
            continue
        if header_start and line.strip().startswith("| -"):
            header_start = False
            continue
        if in_table:
            if not line.strip().startswith("|"):
                in_table = False
                continue
            if line.strip().startswith("| -"):
                continue
            parts = [c.strip() for c in line.split("|")[1:-1]]
            if len(parts) < 3:
                continue
            parts = [_strip_backticks(p) for p in parts]
            if parts[0] in ("#", "---"):
                continue
            entries.append({"code": parts[0], "file": parts[1], "status": parts[2],
                            "content": parts[3] if len(parts) > 3 else ""})

    status_totals = {"COMPLETE": 0, "PARTIAL": 0, "BLOCKED": 0, "NOT_STARTED": 0}
    in_totals = False
    header_seen = False
    for line in lines:
        if line.strip() == "## Status Totals":
            in_totals = True
            header_seen = False
            continue
        if in_totals:
            if not line.strip().startswith("|") and not line.strip() == "":
                in_totals = False
                continue
            if line.strip() == "":
                continue
            stripped = line.strip()
            if "Status" in stripped and "Count" in stripped:
                header_seen = True
                continue
            if not header_seen:
                continue
            if stripped.startswith("|-"):
                continue
            parts = [c.strip() for c in stripped.split("|")[1:-1]]
            if len(parts) >= 2:
                status = parts[0]
                try:
                    count = int(parts[1])
                except ValueError:
                    continue
                if status in status_totals:
                    status_totals[status] = count

    return entries, status_totals


def check_01_index_consistency():
    entries, status_totals = parse_index()

    errors = []
    warnings = []

    actual_statuses = {"COMPLETE": 0, "PARTIAL": 0, "BLOCKED": 0}
    for e in entries:
        if e["status"] in actual_statuses:
            actual_statuses[e["status"]] += 1
        full_path = os.path.join(PLAN_DIR, e["file"])
        if e["file"].endswith("/"):
            if not os.path.isdir(full_path):
                errors.append(f"Index lists directory '{e['file']}' (status {e['status']}) but not found on disk")
        else:
            if not os.path.isfile(full_path):
                errors.append(f"Index lists '{e['file']}' (status {e['status']}) but file not found on disk")

    for status in ("COMPLETE", "PARTIAL", "BLOCKED"):
        declared = status_totals.get(status, 0)
        actual = actual_statuses.get(status, 0)
        if declared != actual:
            errors.append(f"Status total mismatch for {status}: index declares {declared}, actual count is {actual}")

    if errors:
        return False, "; ".join(errors), warnings
    return True, "All index entries exist on disk; status totals match", warnings


def check_02_artifact_count():
    entries, _ = parse_index()

    indexed_files = set()
    for e in entries:
        fp = os.path.normpath(os.path.join(PLAN_DIR, e["file"]))
        indexed_files.add(fp)

    plan_files = []
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith((".md", ".yaml", ".json", ".py", ".txt")):
                plan_files.append(os.path.normpath(os.path.join(root, f)))

    indexed_on_disk = [fp for fp in plan_files if fp in indexed_files]
    not_in_index = [fp for fp in plan_files if fp not in indexed_files]

    detail = f"{len(indexed_on_disk)} indexed, {len(not_in_index)} unindexed, {len(entries)} entries"
    warnings = []
    for fp in not_in_index:
        rel = os.path.relpath(fp, PLAN_DIR)
        warnings.append(f"Unindexed artifact: {rel}")

    if len(indexed_on_disk) != len(entries):
        return False, detail, warnings
    return True, detail, warnings


def check_03_no_tbd_in_complete():
    entries, _ = parse_index()

    complete_files = set()
    for e in entries:
        if e["status"] == "COMPLETE":
            fp = os.path.normpath(os.path.join(PLAN_DIR, e["file"]))
            if os.path.isfile(fp) and fp.endswith(".md"):
                complete_files.add(fp)

    errors = []
    for fpath in sorted(complete_files):
        with open(fpath) as f:
            for lineno, line in enumerate(f, 1):
                if TBD_PATTERNS.search(line):
                    rel = os.path.relpath(fpath, PLAN_DIR)
                    stripped = line.strip()
                    errors.append(f"{rel}:{lineno} COMPLETE but contains '{stripped}'")

    if errors:
        return False, f"{len(errors)} TBD/placeholder matches in COMPLETE files. " + "; ".join(errors), []
    return True, "No TBD/placeholder in COMPLETE files", []


def check_04_markdown_links():
    errors = []
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if not f.endswith(".md"):
                continue
            fpath = os.path.join(root, f)
            with open(fpath) as fh:
                content = fh.read()
            for match in MD_LINK_PATTERN.finditer(content):
                target = match.group(2)
                if target.startswith("http://") or target.startswith("https://"):
                    continue
                if target.startswith("#"):
                    continue
                target = target.split("#")[0]
                if not target:
                    continue
                target_path = os.path.normpath(os.path.join(os.path.dirname(fpath), target))
                if not os.path.exists(target_path):
                    rel_src = os.path.relpath(fpath, PLAN_DIR)
                    errors.append(f"{rel_src}: broken link '{match.group(2)}' (-> {os.path.relpath(target_path, PLAN_DIR)})")

    if errors:
        return False, f"{len(errors)} broken internal links. " + "; ".join(errors), []
    return True, "All internal markdown links valid", []


def check_05_yaml_parsing():
    try:
        with open(CONTRACT_PATH) as f:
            raw = f.read()
        documents = list(yaml.safe_load_all(raw))
        if not documents:
            return False, "YAML parsed but empty", []
        return True, "Contract parses without error", []
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}", []


def check_06_finding_ids():
    with open(FINDINGS_PATH) as f:
        content = f.read()
    matches = FINDING_PATTERN.findall(content)
    numbers = [int(m) for m in matches]
    expected = set(range(1, 44))
    actual = set(numbers)
    missing = expected - actual
    extra = actual - expected

    errors = []
    if missing:
        errors.append(f"Missing finding sections: {sorted(missing)}")
    if extra:
        errors.append(f"Extra finding sections: {sorted(extra)}")

    if len(numbers) != len(set(numbers)):
        errors.append("Duplicate finding numbers detected")

    if errors:
        return False, f"Found {len(matches)} findings, expected 43. " + "; ".join(errors), []
    if len(matches) != 43:
        return False, f"Found {len(matches)} findings, expected 43", []
    return True, f"43 finding sections confirmed (1-43)", []


def check_07_correction_ids():
    with open(CORRECTION_PATH) as f:
        content = f.read()
    matches = CORRECTION_PATTERN.findall(content)
    ids_found = [int(m) for m in matches]
    expected = set(range(1, 18))
    actual = set(ids_found)
    missing = expected - actual
    extra = actual - expected

    errors = []
    if missing:
        errors.append(f"Missing correction IDs: C-{sorted(missing)}")
    if extra:
        errors.append(f"Extra correction IDs: C-{sorted(extra)}")

    if errors:
        return False, f"Found {len(ids_found)} correction IDs, expected 17. " + "; ".join(errors), []
    if len(ids_found) != 17:
        return False, f"Found {len(ids_found)} correction IDs, expected 17", []
    return True, "All 17 correction IDs (C-01 through C-17) present", []


def check_08_diagnostic_raw_json():
    if not os.path.isdir(DIAG_RAW_DIR):
        return False, "diagnostics/raw/ directory not found", []

    json_files = [f for f in os.listdir(DIAG_RAW_DIR) if f.endswith(".json")]
    if not json_files:
        return False, "No .json files in diagnostics/raw/", []

    errors = []
    for jf in sorted(json_files):
        fpath = os.path.join(DIAG_RAW_DIR, jf)
        try:
            with open(fpath) as f:
                json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            errors.append(f"{jf}: {e}")

    if errors:
        return False, f"{len(errors)} parse failures. " + "; ".join(errors), []
    return True, f"{len(json_files)} raw JSON files parse without error", []


def _traverse_for_status(node, path="", errors=None, warnings=None):
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []

    if isinstance(node, dict):
        has_value = "value" in node
        has_status = "status" in node

        if has_value:
            if not has_status:
                errors.append(f"Missing status at {path}")
            else:
                s = node["status"]
                if s not in VALID_STATUSES:
                    errors.append(f"Invalid status '{s}' at {path}")
                if s == "UNFROZEN_CALIBRATION":
                    if "calibration_method" not in node:
                        errors.append(f"UNFROZEN_CALIBRATION at {path} missing calibration_method")
                    if "freeze_condition" not in node:
                        errors.append(f"UNFROZEN_CALIBRATION at {path} missing freeze_condition")

        for key, val in node.items():
            new_path = f"{path}.{key}" if path else str(key)
            _traverse_for_status(val, new_path, errors, warnings)

    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _traverse_for_status(item, f"{path}[{idx}]", errors, warnings)

    return errors, warnings


def check_09_contract_status_tags():
    try:
        with open(CONTRACT_PATH) as f:
            raw = f.read()
        docs = list(yaml.safe_load_all(raw))
        if not docs:
            return False, "Contract YAML is empty", []
        contract = docs[0]
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}", []

    errors, warnings = _traverse_for_status(contract)

    if errors:
        return False, f"{len(errors)} status-tag issues. " + "; ".join(errors[:5]) + (f" ... ({len(errors)} total)" if len(errors) > 5 else ""), warnings
    return True, "All contract scalars have valid status tags; unfrozen values have calibration_method/freeze_condition", warnings


def check_10_source_hash():
    try:
        with open(CONTRACT_PATH) as f:
            raw = f.read()
        docs = list(yaml.safe_load_all(raw))
        if not docs:
            return False, "Contract YAML is empty", []
        contract = docs[0]
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}", []

    source_hash = contract.get("source_hash", {})
    val = source_hash.get("value", "") if isinstance(source_hash, dict) else ""

    if not val:
        return False, "source_hash value is empty or missing", []

    content_addressed = False
    lower_val = str(val).lower()
    if re.search(r"(git|sha|sha-?256|tree.?sha|content.?address)", lower_val):
        content_addressed = True

    if content_addressed:
        return True, f"Source hash is content-addressed: '{val}'", []
    else:
        return False, f"Source hash does not appear content-addressed: '{val}'", []


def check_11_prim_estimand_consistency():
    try:
        with open(CONTRACT_PATH) as f:
            raw = f.read()
        docs = list(yaml.safe_load_all(raw))
        contract = docs[0] if docs else {}
    except yaml.YAMLError:
        return False, "Contract YAML parse failed", []

    with open(CAUSAL_PATH) as f:
        causal_text = f.read()
    with open(SAP_PATH) as f:
        sap_text = f.read()

    warnings = []

    estimand = contract.get("primary_estimand", {})
    formula = ""
    if isinstance(estimand, dict):
        formula = str(estimand.get("formula", {}).get("value", "")) if isinstance(estimand.get("formula"), dict) else str(estimand.get("formula", ""))
    test_method = ""
    pst = contract.get("primary_statistical_test", {})
    if isinstance(pst, dict):
        test_method = str(pst.get("method", {}).get("value", "")) if isinstance(pst.get("method"), dict) else str(pst.get("method", ""))

    formula_lower = formula.lower()
    test_lower = test_method.lower()

    uses_max = "max" in formula_lower

    in_causal = "intersection" in causal_text.lower() or "iut" in causal_text.lower()
    in_sap = ("intersection" in sap_text.lower() or "iut" in sap_text.lower() or
              "mechanism advantage" in sap_text.lower())

    has_iut_estimand = uses_max and ("vortex" in formula_lower and "margin" in formula_lower)

    if not in_causal:
        warnings.append("IUT not explicitly referenced in causal identification document")
    if not ("iut" in sap_text.lower() or "intersection" in sap_text.lower()):
        warnings.append("IUT not explicitly named in statistical analysis plan (mechanism advantage described via hierarchical bootstrap)")

    if not has_iut_estimand:
        return False, f"Primary estimand formula does not match IUT pattern: {formula[:120]}", warnings

    if "intersection_union" not in test_lower and "iut" not in test_lower:
        warnings.append(f"Primary test method '{test_method}' does not explicitly name IUT")

    if warnings:
        return True, f"Estimand uses max(nulls) consistent with IUT (formula present, test method: {test_method})", warnings
    return True, f"Primary estimand and IUT test consistent across contract, causal ID, and SAP", []


def check_12_required_artifacts():
    entries, _ = parse_index()

    expected_artifact_count = 28

    if len(entries) == expected_artifact_count:
        return True, f"{len(entries)} entries in index, expected {expected_artifact_count}", []
    else:
        return False, f"Index has {len(entries)} entries, expected {expected_artifact_count}", []


def compute_sha256_hashes():
    result = {}
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in sorted(files):
            if f.endswith((".md", ".yaml", ".json", ".py", ".txt")):
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, PLAN_DIR)
                result[rel] = sha256_file(fpath)
    return result


def main():
    checks = []

    all_errors = []
    all_warnings = []

    def _record_check(name, passed, detail, warnings):
        checks.append({"name": name, "pass": passed, "detail": detail})
        all_warnings.extend(warnings)
        if not passed:
            all_errors.append(f"[{name}] {detail}")

    _record_check("01_index_consistency", *check_01_index_consistency())
    _record_check("02_artifact_count", *check_02_artifact_count())
    _record_check("03_no_tbd_in_complete", *check_03_no_tbd_in_complete())
    _record_check("04_markdown_links", *check_04_markdown_links())
    _record_check("05_yaml_parsing", *check_05_yaml_parsing())
    _record_check("06_43_finding_ids", *check_06_finding_ids())
    _record_check("07_17_correction_ids", *check_07_correction_ids())
    _record_check("08_diagnostic_raw_json", *check_08_diagnostic_raw_json())
    _record_check("09_contract_status_tags", *check_09_contract_status_tags())
    _record_check("10_source_hash", *check_10_source_hash())
    _record_check("11_prim_estimand_consistency", *check_11_prim_estimand_consistency())
    _record_check("12_required_artifacts", *check_12_required_artifacts())

    overall = all(c["pass"] for c in checks)

    report = {
        "overall_pass": overall,
        "checks": checks,
        "errors": sorted(set(all_errors)),
        "warnings": sorted(set(all_warnings)),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    sha256_data = compute_sha256_hashes()
    os.makedirs(os.path.dirname(SHA256_PATH), exist_ok=True)
    with open(SHA256_PATH, "w") as f:
        json.dump(sha256_data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Validation report written to {OUTPUT_PATH}")
    print(f"SHA-256 hashes written to {SHA256_PATH}")
    print(f"Overall pass: {overall}")
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['name']}: {c['detail']}")
    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  - {err}")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"  - {w}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
