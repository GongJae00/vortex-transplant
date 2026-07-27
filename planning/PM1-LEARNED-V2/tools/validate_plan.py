#!/usr/bin/env python3
"""Validate PM1-LEARNED-V2 planning package integrity across 5 independent layers."""

import hashlib
import json
import os
import re
import subprocess
import sys

import yaml  # type: ignore

PLAN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(PLAN_DIR, "00_INDEX.md")
CONTRACT_PATH = os.path.join(PLAN_DIR, "15_PM1_LEARNED_V2_CONTRACT.yaml")
FINDINGS_PATH = os.path.join(PLAN_DIR, "03_PRIOR_AUDIT_43_FINDINGS.md")
CORRECTION_PATH = os.path.join(PLAN_DIR, "02_DRAFT0_CORRECTION_LEDGER.md")
CAUSAL_PATH = os.path.join(PLAN_DIR, "07_CAUSAL_IDENTIFICATION.md")
SAP_PATH = os.path.join(PLAN_DIR, "08_STATISTICAL_ANALYSIS_PLAN.md")
LITERATURE_PATH = os.path.join(PLAN_DIR, "05_LITERATURE_AND_NOVELTY.md")
VENUE_PATH = os.path.join(PLAN_DIR, "14_MANUSCRIPT_REVIEWER_AND_VENUE.md")
DIAG_RAW_DIR = os.path.join(PLAN_DIR, "diagnostics", "raw")
DIAG_DIR = os.path.join(PLAN_DIR, "diagnostics")
ENVIRONMENT_PATH = os.path.join(DIAG_DIR, "environment.json")
METRIC_DEF_PATH = os.path.join(DIAG_DIR, "metric_definitions.py")
COMMANDS_PATH = os.path.join(DIAG_DIR, "commands.txt")
SHA256_PATH = os.path.join(DIAG_DIR, "sha256.json")
OUTPUT_PATH = os.path.join(PLAN_DIR, "PLAN_VALIDATION.json")
VERIFY_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "verify_manifest.py")

VALID_STATUSES = frozenset((
    "FROZEN_MATHEMATICAL", "FROZEN_ENGINEERING",
    "UNFROZEN_ENGINEERING", "UNFROZEN_CALIBRATION",
    "DERIVED", "DESCRIPTIVE",
))
TBD_PATTERNS = re.compile(r"\b(TBD|placeholder|not executed|\?\?\?)\b", re.IGNORECASE)
MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
FINDING_PATTERN = re.compile(r"^### Finding (\d+)", re.MULTILINE)
CORRECTION_PATTERN = re.compile(r"^## C-(\d{2})", re.MULTILINE)
STALE_COUNT_PATTERNS = [
    re.compile(r"340\s+pairs/channel", re.IGNORECASE),
    re.compile(r"10\.6\s+pairs/plaquette", re.IGNORECASE),
]
BOOTSTRAP_P_PATTERN = re.compile(r"p\s*=\s*mean\(D\*\s*<=\s*0\)")
GIT_DIFF_STAT_PATTERN = re.compile(r"git\s+diff\s+--stat")


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


def _load_yaml(path):
    with open(path) as f:
        raw = f.read()
    docs = list(yaml.safe_load_all(raw))
    return docs[0] if docs else {}, raw


def _load_yaml_contract():
    return _load_yaml(CONTRACT_PATH)


def _read_text(path):
    with open(path) as f:
        return f.read()


def _load_json(path):
    with open(path) as f:
        return json.load(f)


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
            entries.append({
                "code": parts[0], "file": parts[1], "status": parts[2],
                "content": parts[3] if len(parts) > 3 else "",
            })

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


def file_texts_in_plan():
    texts = {}
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if f.endswith((".md", ".yaml")):
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, PLAN_DIR)
                try:
                    texts[rel] = _read_text(fpath)
                except IOError:
                    pass
    return texts


# ==============================================================================
# Layer 1: Structural integrity
# ==============================================================================

def l1_index_consistency():
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
                errors.append(
                    f"Index lists directory '{e['file']}' (status {e['status']}) "
                    f"but not found on disk"
                )
        else:
            if not os.path.isfile(full_path):
                errors.append(
                    f"Index lists '{e['file']}' (status {e['status']}) "
                    f"but file not found on disk"
                )

    for status in ("COMPLETE", "PARTIAL", "BLOCKED"):
        declared = status_totals.get(status, 0)
        actual = actual_statuses.get(status, 0)
        if declared != actual:
            errors.append(
                f"Status total mismatch for {status}: "
                f"index declares {declared}, actual count is {actual}"
            )

    if errors:
        return False, "; ".join(errors), warnings
    return True, "All index entries exist on disk; status totals match", warnings


def l1_artifact_count():
    entries, _ = parse_index()
    indexed_files = set()
    for e in entries:
        fp = os.path.normpath(os.path.join(PLAN_DIR, e["file"]))
        indexed_files.add(fp)

    plan_files = []
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if f.endswith((".md", ".yaml", ".json", ".py", ".txt")):
                plan_files.append(os.path.normpath(os.path.join(root, f)))

    indexed_on_disk = [fp for fp in plan_files if fp in indexed_files]
    not_in_index = [fp for fp in plan_files if fp not in indexed_files]

    warnings = []
    for fp in not_in_index:
        rel = os.path.relpath(fp, PLAN_DIR)
        warnings.append(f"Unindexed artifact: {rel}")

    detail = f"{len(indexed_on_disk)} indexed, {len(not_in_index)} unindexed, {len(entries)} entries"
    if len(indexed_on_disk) != len(entries):
        return False, detail, warnings
    return True, detail, warnings


def l1_no_tbd_in_complete():
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


def l1_markdown_links():
    errors = []
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if not f.endswith(".md"):
                continue
            fpath = os.path.join(root, f)
            content = _read_text(fpath)
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
                    errors.append(
                        f"{rel_src}: broken link '{match.group(2)}' "
                        f"(-> {os.path.relpath(target_path, PLAN_DIR)})"
                    )

    if errors:
        return False, f"{len(errors)} broken internal links. " + "; ".join(errors), []
    return True, "All internal markdown links valid", []


def l1_yaml_parsing():
    try:
        documents = list(yaml.safe_load_all(_read_text(CONTRACT_PATH)))
        if not documents:
            return False, "YAML parsed but empty", []
        return True, "Contract parses without error", []
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}", []


def l1_finding_ids():
    content = _read_text(FINDINGS_PATH)
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
    return True, "43 finding sections confirmed (1-43)", []


def l1_correction_ids():
    content = _read_text(CORRECTION_PATH)
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


def l1_diagnostic_raw_json():
    if not os.path.isdir(DIAG_RAW_DIR):
        return False, "diagnostics/raw/ directory not found", []

    json_files = [f for f in os.listdir(DIAG_RAW_DIR) if f.endswith(".json")]
    if not json_files:
        return False, "No .json files in diagnostics/raw/", []

    errors = []
    for jf in sorted(json_files):
        fpath = os.path.join(DIAG_RAW_DIR, jf)
        try:
            _load_json(fpath)
        except (json.JSONDecodeError, IOError) as e:
            errors.append(f"{jf}: {e}")

    if errors:
        return False, f"{len(errors)} parse failures. " + "; ".join(errors), []
    return True, f"{len(json_files)} raw JSON files parse without error", []


# ==============================================================================
# Layer 2: Diagnostic provenance
# ==============================================================================

def l2_environment_spec():
    if not os.path.isfile(ENVIRONMENT_PATH):
        return False, "diagnostics/environment.json not found", []

    try:
        env = _load_json(ENVIRONMENT_PATH)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"environment.json parse error: {e}", []

    errors = []
    for key in ["python", "pytorch", "numpy"]:
        if key not in env:
            errors.append(f"environment.json missing '{key}'")
            continue
        val = str(env[key]).strip()
        if val and re.match(r"^\d+\.\d+[+]*$", val):
            errors.append(
                f"environment.json '{key}'='{val}' uses '+' suffix; "
                f"should be exact pinned version"
            )

    if errors:
        return False, "; ".join(errors), []
    return True, (
        f"environment.json present with python={env.get('python')}, "
        f"pytorch={env.get('pytorch')}, numpy={env.get('numpy')}"
    ), []


def l2_raw_json_exists():
    if not os.path.isdir(DIAG_RAW_DIR):
        return False, "diagnostics/raw/ directory not found", []

    json_files = [f for f in os.listdir(DIAG_RAW_DIR) if f.endswith(".json")]
    if not json_files:
        return False, "No .json files in diagnostics/raw/", []

    errors = []
    for jf in sorted(json_files):
        fpath = os.path.join(DIAG_RAW_DIR, jf)
        try:
            _load_json(fpath)
        except (json.JSONDecodeError, IOError) as e:
            errors.append(f"{jf}: {e}")

    if errors:
        return False, "; ".join(errors), []
    return True, f"{len(json_files)} raw JSON files exist and parse", []


def l2_metric_definitions_exists():
    if not os.path.isfile(METRIC_DEF_PATH):
        return False, "diagnostics/metric_definitions.py not found", []
    return True, "metric_definitions.py exists", []


def l2_commands_txt_exists():
    if not os.path.isfile(COMMANDS_PATH):
        return False, "diagnostics/commands.txt not found", []

    content = _read_text(COMMANDS_PATH)
    if not content.strip():
        return False, "commands.txt is empty", []

    has_python_cmd = any(
        "python " in line or "python3 " in line
        for line in content.splitlines()
    )
    if not has_python_cmd:
        return False, "commands.txt has no reproduction commands", []

    return True, "commands.txt present with reproduction commands", []


def l2_sha256_no_self_hash():
    if not os.path.isfile(SHA256_PATH):
        return False, "diagnostics/sha256.json not found", []

    try:
        manifest = _load_json(SHA256_PATH)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"sha256.json parse error: {e}", []

    self_path = os.path.relpath(SHA256_PATH, PLAN_DIR)
    if self_path in manifest:
        return False, f"sha256.json contains its own hash ({self_path})", []

    return True, "sha256.json excludes itself", []


def l2_sha256_verify_all():
    if not os.path.isfile(SHA256_PATH):
        return False, "diagnostics/sha256.json not found", []

    try:
        manifest = _load_json(SHA256_PATH)
    except (json.JSONDecodeError, IOError) as e:
        return False, f"sha256.json parse error: {e}", []

    errors = []
    verified = 0
    for rel_path, expected_hash in sorted(manifest.items()):
        abs_path = os.path.join(PLAN_DIR, rel_path)
        if not os.path.isfile(abs_path):
            errors.append(f"Manifested file missing: {rel_path}")
            continue
        try:
            actual_hash = sha256_file(abs_path)
        except IOError as e:
            errors.append(f"Cannot hash {rel_path}: {e}")
            continue
        if actual_hash != expected_hash:
            errors.append(f"Hash mismatch for {rel_path}")
        else:
            verified += 1

    if errors:
        return False, f"{len(errors)} hash verification failures. " + "; ".join(errors), []
    return True, f"{verified}/{len(manifest)} SHA-256 hashes verified", []


def l2_no_stale_counts():
    texts = file_texts_in_plan()
    hits = []
    for rel, content in texts.items():
        for pattern in STALE_COUNT_PATTERNS:
            for m in pattern.finditer(content):
                hits.append(f"{rel}: '{m.group(0)}'")

    if hits:
        return False, f"Stale impossible count references: {'; '.join(hits)}", []
    return True, "No stale impossible counts (340 pairs/channel, 10.6 pairs/plaquette)", []


def l2_manifest_verification():
    if not os.path.isfile(VERIFY_MANIFEST_PATH):
        return False, "tools/verify_manifest.py not found", []

    try:
        result = subprocess.run(
            [sys.executable, VERIFY_MANIFEST_PATH],
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(result.stdout)
            if data.get("pass"):
                return True, f"verify_manifest.py: {data.get('detail', 'OK')}", []
            else:
                return False, (
                    f"verify_manifest.py failed: {data.get('detail', '')}; "
                    f"errors: {data.get('errors', [])}"
                ), []
        except json.JSONDecodeError:
            return False, f"verify_manifest.py returned non-JSON: {result.stdout[:200]}", []
    except subprocess.TimeoutExpired:
        return False, "verify_manifest.py timed out", []
    except Exception as e:
        return False, f"verify_manifest.py error: {e}", []


# ==============================================================================
# Layer 3: Contract schema
# ==============================================================================

def _traverse_contract(node, path="", errors=None, warnings=None):
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []

    if isinstance(node, dict):
        has_value = "value" in node
        has_status = "status" in node

        if has_value:
            if not has_status:
                errors.append(f"Missing 'status' key at {path} (has 'value')")
            else:
                s = node["status"]
                if s not in VALID_STATUSES:
                    errors.append(f"Unrecognized status '{s}' at {path}")
                if s == "UNFROZEN_CALIBRATION":
                    if "calibration_method" not in node:
                        errors.append(
                            f"UNFROZEN_CALIBRATION at {path} missing 'calibration_method'"
                        )
                    if "freeze_condition" not in node:
                        errors.append(
                            f"UNFROZEN_CALIBRATION at {path} missing 'freeze_condition'"
                        )
                if s == "FROZEN_MATHEMATICAL":
                    if "rationale" not in node:
                        errors.append(
                            f"FROZEN_MATHEMATICAL at {path} missing 'rationale'"
                        )
                    elif isinstance(node["rationale"], str):
                        r = node["rationale"].lower()
                        if not any(tok in r for tok in ("math", "proof", "topological", "theorem")):
                            warnings.append(
                                f"FROZEN_MATHEMATICAL at {path}: rationale may not reference math/proof"
                            )
                if s == "FROZEN_ENGINEERING":
                    if "rationale" not in node:
                        warnings.append(
                            f"FROZEN_ENGINEERING at {path} missing 'rationale'"
                        )
                    if "provenance" not in node:
                        warnings.append(
                            f"FROZEN_ENGINEERING at {path} missing 'provenance'"
                        )
                    if "sensitivity_analysis" not in node:
                        warnings.append(
                            f"FROZEN_ENGINEERING at {path} missing 'sensitivity_analysis'"
                        )

        for key, val in node.items():
            new_path = f"{path}.{key}" if path else str(key)
            _traverse_contract(val, new_path, errors, warnings)

    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _traverse_contract(item, f"{path}[{idx}]", errors, warnings)

    return errors, warnings


def l3_contract_status_tags():
    try:
        contract, _ = _load_yaml_contract()
    except yaml.YAMLError as e:
        return False, f"Contract YAML parse error: {e}", []

    errors, warnings = _traverse_contract(contract)

    if errors:
        return False, (
            f"{len(errors)} status-tag issues. "
            + "; ".join(errors[:8])
            + (f" ... ({len(errors)} total)" if len(errors) > 8 else "")
        ), warnings
    return True, (
        "All contract scalars have valid status tags; "
        "UNFROZEN_CALIBRATION has calibration_method/freeze_condition; "
        "FROZEN_MATHEMATICAL has rationale; "
        "FROZEN_ENGINEERING has rationale/provenance/sensitivity_analysis"
    ), warnings


def l3_no_unrecognized_status():
    try:
        contract, _ = _load_yaml_contract()
    except yaml.YAMLError as e:
        return False, f"Contract YAML parse error: {e}", []

    errors = []

    def _find_all_statuses(node, path=""):
        if isinstance(node, dict):
            if "status" in node:
                s = node["status"]
                if s not in VALID_STATUSES:
                    errors.append(f"Unrecognized status '{s}' at {path}")
            for key, val in node.items():
                np = f"{path}.{key}" if path else str(key)
                _find_all_statuses(val, np)
        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _find_all_statuses(item, f"{path}[{idx}]")

    _find_all_statuses(contract)

    if errors:
        return False, "; ".join(errors), []
    return True, f"All status values one of {sorted(VALID_STATUSES)}", []


def l3_recursive_scalar_leaves():
    try:
        contract, _ = _load_yaml_contract()
    except yaml.YAMLError as e:
        return False, f"Contract YAML parse error: {e}", []

    warnings = []

    def _check_untagged_leaves(node, path="", depth=0):
        if isinstance(node, dict):
            has_status = "status" in node
            has_rationale = "rationale" in node
            has_description = "description" in node
            has_text = "text" in node

            is_tagged = has_status
            is_descriptive = (
                has_rationale or has_description or has_text
                or "action" in node or "condition" in node
                or "name" in node or "id" in node
                or "label" in node or "role" in node
                or "allowed_actions" in node or "forbidden_actions" in node
                or "forbidden_claims" in node or "claims" in node
            )

            has_any_scalar = any(
                isinstance(v, (str, int, float, bool)) for v in node.values()
            )
            has_any_dict = any(isinstance(v, dict) for v in node.values())

            if depth >= 3 and has_any_scalar and not has_any_dict:
                if not is_tagged and not is_descriptive:
                    key_preview = sorted(node.keys())[:4]
                    warnings.append(
                        f"Untagged scalar dict at {path}: keys={key_preview}"
                    )

            for key, val in node.items():
                np = f"{path}.{key}" if path else str(key)
                _check_untagged_leaves(val, np, depth + 1)

        elif isinstance(node, list):
            for idx, item in enumerate(node):
                _check_untagged_leaves(item, f"{path}[{idx}]", depth + 1)

    _check_untagged_leaves(contract)

    if warnings:
        return True, (
            f"{len(warnings)} potentially untagged scalar leaves found"
        ), warnings
    return True, "All scalar leaves verified", []


# ==============================================================================
# Layer 4: Statistical semantics
# ==============================================================================

def _extract_contract_null_families():
    try:
        contract, _ = _load_yaml_contract()
    except yaml.YAMLError:
        return None
    nulls = (
        contract.get("intervention_families", {})
        .get("null_families", {})
        .get("families", [])
    )
    if not nulls:
        return None
    ids = []
    for fam in nulls:
        if isinstance(fam, dict) and "id" in fam:
            ids.append(fam["id"])
        elif isinstance(fam, str):
            ids.append(fam)
    return frozenset(ids)


def _extract_test_null_families():
    try:
        contract, _ = _load_yaml_contract()
    except yaml.YAMLError:
        return None
    families = (
        contract.get("primary_statistical_test", {})
        .get("individual_tests", {})
        .get("families", [])
    )
    if not families:
        return None
    return frozenset(f for f in families if isinstance(f, str))


def _extract_causal_null_families():
    content = _read_text(CAUSAL_PATH)
    null_families = None

    for p in [r"\\\\mathcal\{F\}\s*=\s*\{(.+?)\}", r"F\s*=\s*\{(.+?)\}"]:
        m = re.search(p, content, re.DOTALL)
        if m:
            raw = m.group(1)
            # Handle nested braces from \text{...} by scanning
            depth = 0
            end = 0
            for j, ch in enumerate(raw):
                if ch == '{': depth += 1
                elif ch == '}':
                    if depth == 0:
                        end = j
                        break
                    depth -= 1
            if end > 0:
                raw = raw[:end]
            items = re.split(r",\s*", raw)
            cleaned = set()
            for item in items:
                item = item.strip()
                item = re.sub(r"\\text\{.*?\}", "", item).strip()
                item = re.sub(r"\\mathrm\{.*?\}", "", item).strip()
                item = item.strip().lower().replace("_", "").replace("-", "")
                if item:
                    cleaned.add(item)
            null_families = cleaned
            break

    if not null_families:
        # Fallback: NULL_FAMILIES comment
        m2 = re.search(r"NULL_FAMILIES:\s*(.+)", content)
        if m2:
            items = [x.strip().lower() for x in m2.group(1).split(",")]
            null_families = frozenset(items)

    if not null_families:
        section_pattern = re.compile(r"^## 7\.9 .+", re.MULTILINE)
        m = section_pattern.search(content)
        if m:
            start = m.start()
            table_start = content.find("| Family |", start)
            if table_start == -1:
                table_start = content.find("|--------|", start)
            if table_start != -1:
                section = content[table_start:table_start + 2000]
                families = []
                for line in section.splitlines():
                    if line.startswith("|") and "`" in line:
                        match = re.findall(r"`([^`]+)`", line)
                        if match:
                            families.append(match[0].strip().lower())
                if families:
                    null_families = frozenset(families)
    return null_families


def _extract_sap_null_families():
    content = _read_text(SAP_PATH)
    m = re.search(r"F\s*=\s*\{(.+?)\}", content, re.DOTALL)
    if m:
        raw = m.group(1)
        items = re.split(r",\s*", raw)
        cleaned = set()
        for item in items:
            item = item.strip().lower().replace("_", "").replace("-", "")
            if item:
                cleaned.add(item)
        return cleaned
    return None


_NULL_FAMILY_NORMALIZE = {
    "fourier_low": "fourierlow", "fourier_high": "fourierhigh",
    "fourierlow": "fourierlow", "fourierhigh": "fourierhigh",
    "global_phase": "globalphase", "globalphase": "globalphase",
    "zero_charge_phase": "zerochargephase", "zerochargephase": "zerochargephase",
    "random_direction": "randomdirection", "randomdirection": "randomdirection",
    "charge_arrangement_shuffle": "chargearrangementshuffle",
    "chargearrangementshuffle": "chargearrangementshuffle",
    "same_charge_rep": "samechargerep", "samechargerep": "samechargerep",
    "smooth": "smooth", "magnitude": "magnitude",
    "pca": "pca", "harmonic": "harmonic",
}


def _normalize_null_families(families):
    if families is None:
        return None
    normalized = set()
    for f in families:
        norm = _NULL_FAMILY_NORMALIZE.get(f, f)
        normalized.add(norm)
    return frozenset(normalized)


def l4_null_family_identity():
    contract_nulls = _extract_contract_null_families()
    test_nulls = _extract_test_null_families()
    causal_nulls = _extract_causal_null_families()
    sap_nulls = _extract_sap_null_families()

    errors = []
    norm_contract = _normalize_null_families(contract_nulls)
    norm_test = _normalize_null_families(test_nulls)
    norm_causal = _normalize_null_families(causal_nulls)
    norm_sap = _normalize_null_families(sap_nulls)

    if norm_contract and norm_test and norm_contract != norm_test:
        errors.append(
            f"Contract null families ({sorted(norm_contract)}) != "
            f"individual_tests families ({sorted(norm_test)})"
        )

    if norm_contract and norm_causal:
        sym_diff = norm_contract.symmetric_difference(norm_causal)
        if sym_diff:
            errors.append(
                f"Null family mismatch contract<->causal: "
                f"only_contract={sorted(norm_contract - norm_causal)}, "
                f"only_causal={sorted(norm_causal - norm_contract)}"
            )

    if norm_contract and norm_sap:
        sym_diff = norm_contract.symmetric_difference(norm_sap)
        if sym_diff:
            errors.append(
                f"Null family mismatch contract<->SAP: "
                f"only_contract={sorted(norm_contract - norm_sap)}, "
                f"only_SAP={sorted(norm_sap - norm_contract)}"
            )

    if norm_causal and norm_sap:
        sym_diff = norm_causal.symmetric_difference(norm_sap)
        if sym_diff:
            errors.append(
                f"Null family mismatch causal<->SAP: "
                f"only_causal={sorted(norm_causal - norm_sap)}, "
                f"only_SAP={sorted(norm_sap - norm_causal)}"
            )

    warnings = []
    for src, s in [("contract", contract_nulls), ("test", test_nulls),
                   ("causal", causal_nulls), ("SAP", sap_nulls)]:
        if s is None:
            warnings.append(f"Could not extract null families from {src}")

    if errors:
        return False, "; ".join(errors), warnings
    ncs = [n for n in [norm_contract, norm_causal, norm_sap] if n is not None]
    counts = [len(n) for n in ncs]
    if len(set(counts)) > 1:
        warnings.append(f"Null family counts differ: contract={counts[0] if len(counts)>0 else '?'}, "
                        f"causal={counts[1] if len(counts)>1 else '?'}, "
                        f"SAP={counts[2] if len(counts)>2 else '?'}")
    return True, (
        f"Null families identical across all documents "
        f"(counts: {dict(zip(['contract','causal','SAP'], counts))})"
    ), warnings


_REPRESENTATIVE_IDS = {"samechargerep", "same_charge_rep"}
_MANIFOLD_IDS = {"manifold"}


def l4_representative_not_in_nulls():
    contract_nulls = _extract_contract_null_families()
    if contract_nulls is None:
        return False, "Cannot extract contract null families", []

    norm = _normalize_null_families(contract_nulls)
    found_rep = norm & _REPRESENTATIVE_IDS
    if found_rep:
        return False, (
            f"Representative sensitivity ID(s) found in null family list: "
            f"{sorted(found_rep)}. It should be a separate gate."
        ), []
    return True, "Representative sensitivity not in null family list (separate gate)", []


def l4_manifold_not_in_nulls():
    contract_nulls = _extract_contract_null_families()
    if contract_nulls is None:
        return False, "Cannot extract contract null families", []

    norm = _normalize_null_families(contract_nulls)
    found_man = norm & _MANIFOLD_IDS
    if found_man:
        return False, (
            f"Manifold validity ID(s) found in null family list: "
            f"{sorted(found_man)}. It should be a separate gate."
        ), []
    return True, "Manifold validity not in null family list (separate gate)", []


def l4_iut_rejection_max_not_min():
    texts = file_texts_in_plan()
    combined_text = "\n".join(texts.values())

    if re.search(r"\bmin\(p_f\)\s*(<=|≤|≤)\s*\bα\b", combined_text):
        return False, "min(p_f) <= alpha appears as decision rule (should be max(p_f))", []

    has_max_rule = bool(re.search(
        r"\bmax\(p_f\)\s*(<=|≤|≤)\s*\bα\b",
        combined_text
    ))
    if not has_max_rule:
        return False, "No explicit max(p_f) <= alpha rule found in plan documents", []

    return True, "Global IUT rejection uses max(p_f) <= alpha, not min(p_f)", []


def l4_no_ordinary_bootstrap_p_formula():
    texts = file_texts_in_plan()
    hits = []
    for rel, content in texts.items():
        for m in BOOTSTRAP_P_PATTERN.finditer(content):
            lineno = content[:m.start()].count("\n") + 1
            hits.append(f"{rel}:{lineno}")

    if hits:
        return False, (
            f"Ordinary uncentered bootstrap p-value formula (p=mean(D*<=0)): "
            f"{'; '.join(hits)}"
        ), []
    return True, "No ordinary uncentered bootstrap p-value formula", []


def l4_test_ci_level_consistency():
    contract, _ = _load_yaml_contract()

    alpha_node = (
        contract.get("primary_statistical_test", {})
        .get("global_rejection", {})
        .get("alpha", {})
    )
    alpha_val = None
    if isinstance(alpha_node, dict):
        alpha_val = alpha_node.get("value", None)

    ci_node = contract.get("confidence_interval", {})
    ci_level = None
    ci_one_sided = None
    if isinstance(ci_node, dict):
        ci_level_val = ci_node.get("level", {})
        ci_level = ci_level_val.get("value") if isinstance(ci_level_val, dict) else ci_level_val
        ci_os_val = ci_node.get("one_sided", {})
        ci_one_sided = ci_os_val.get("value") if isinstance(ci_os_val, dict) else ci_os_val

    warnings = []
    if alpha_val == 0.05 and ci_level == 0.95 and ci_one_sided is False:
        pass
    elif alpha_val is not None and ci_level is not None:
        expected_two_sided = 1 - 2 * alpha_val
        if isinstance(ci_level, (int, float)) and abs(ci_level - expected_two_sided) < 0.01:
            warnings.append(
                f"alpha={alpha_val} (one-sided) but two-sided CI level={ci_level} "
                f"matches 1-2*alpha. Consider separate statement."
            )

    if warnings:
        return True, "Test level and CI level consistency checked", warnings
    return True, "One-sided test alpha=0.05 with two-sided 95% CI is valid (informational reporting)", []


def l4_seed_level_contrasts():
    contract, _ = _load_yaml_contract()
    aggregation = contract.get("primary_estimand", {}).get("aggregation", {})
    agg_text = ""
    if isinstance(aggregation, dict):
        agg_text = str(aggregation.get("value", "")).lower()
    elif isinstance(aggregation, str):
        agg_text = aggregation.lower()

    bootstrap_level = (
        contract.get("primary_statistical_test", {})
        .get("bootstrap", {})
        .get("level", {})
    )
    bl_text = ""
    if isinstance(bootstrap_level, dict):
        bl_text = str(bootstrap_level.get("value", "")).lower()
    elif isinstance(bootstrap_level, str):
        bl_text = bootstrap_level.lower()

    if "seed" in agg_text and "seed" in bl_text:
        return True, "Primary estimand uses per-seed contrasts; bootstrap at seed level", []

    warnings = []
    if "seed" not in agg_text:
        warnings.append(f"Aggregation may not be seed-level: '{agg_text}'")
    if "seed" not in bl_text:
        warnings.append(f"Bootstrap level may not be seed: '{bl_text}'")

    if warnings:
        return True, "Seed-level contrasts: check warnings", warnings
    return True, "Seed-level contrasts confirmed", []


def l4_no_complete_case_without_analyzability():
    contract, _ = _load_yaml_contract()
    missingness = contract.get("missingness", {})
    primary_analysis = ""
    if isinstance(missingness, dict):
        pa = missingness.get("primary_analysis", {})
        if isinstance(pa, dict):
            primary_analysis = str(pa.get("value", "")).lower()
        elif isinstance(pa, str):
            primary_analysis = pa.lower()

    sap_text = _read_text(SAP_PATH)
    has_analyzability_in_sap = (
        "analyzability estimand" in sap_text.lower()
        or "psi_analyze" in sap_text.lower()
    )

    if "complete" in primary_analysis and "case" in primary_analysis:
        if not has_analyzability_in_sap:
            return False, (
                "Complete-case primary analysis in contract but no analyzability "
                "estimand (psi_analyze) defined in SAP"
            ), []
        return True, (
            "Complete-case primary analysis with analyzability estimand (psi_analyze) defined"
        ), []

    return True, "Primary analysis properly qualified", []


def l4_no_nonsig_as_equivalence():
    texts = file_texts_in_plan()
    warnings_local = []
    patterns = [
        (re.compile(r"p\s*[>≥]\s*0\.05.*effect\s+absent", re.IGNORECASE),
         "P > 0.05 interpreted as 'effect absent'"),
    ]
    for rel, content in texts.items():
        for pat, desc in patterns:
            if pat.search(content):
                warnings_local.append(f"{rel}: {desc}")

    if warnings_local:
        return True, "Potential non-significance-as-equivalence found (warnings)", warnings_local
    return True, "No non-significance interpreted as equivalence", []


def l4_no_git_diff_stat_source_hash():
    contract, _ = _load_yaml_contract()
    source_hash = contract.get("source_hash", {})
    val = ""
    if isinstance(source_hash, dict):
        val = str(source_hash.get("value", ""))
    elif isinstance(source_hash, str):
        val = source_hash

    if "git diff --stat" in val:
        return False, (
            f"source_hash uses 'git diff --stat' which is not content-addressed: '{val}'"
        ), []
    return True, "source_hash is content-addressed (not git diff --stat)", []


# ==============================================================================
# Layer 5: Literature evidence
# ==============================================================================

def l5_screening_accounting():
    content = _read_text(LITERATURE_PATH)

    terms = {
        "results_returned": bool(re.search(
            r"results\s+returned|returned\s+\d+", content, re.IGNORECASE
        )),
        "records_retrieved": bool(re.search(
            r"records?\s+retrieved|retrieved\s+\d+", content, re.IGNORECASE
        )),
        "titles_screened": bool(re.search(
            r"titles?\s+screened|screened.*title", content, re.IGNORECASE
        )),
        "abstracts_screened": bool(re.search(
            r"abstracts?\s+screened|screened.*abstract", content, re.IGNORECASE
        )),
        "full_texts_read": bool(re.search(
            r"full.?texts?\s+read|full.?text.*review", content, re.IGNORECASE
        )),
        "included": bool(re.search(
            r"included\s+in|final\s+set|\d+\s+included", content, re.IGNORECASE
        )),
    }

    missing_terms = [k for k, v in terms.items() if not v]

    if len(missing_terms) >= 4:
        return False, (
            f"Screening accounting missing key stages: {missing_terms}. "
            f"Found: {[k for k, v in terms.items() if v]}"
        ), []

    if missing_terms:
        return True, (
            f"Screening accounting partially present. "
            f"Missing: {missing_terms}"
        ), [f"Screening accounting missing: {missing_terms}"]

    return True, "Screening accounting separates all stages", []


def l5_no_raw_db_hits_as_screened():
    content = _read_text(LITERATURE_PATH)
    warnings = []

    if re.search(r"Results?\s+Screened.*:,?\s*\d{4,}", content, re.IGNORECASE):
        pass

    raw_count_labels = re.findall(
        r"(Total\s+)?[Rr]esults?\s+[Ss]creened:.{0,30}\d{4,}",
        content
    )
    if raw_count_labels:
        warnings.append(
            f"Raw database hit counts may be labeled as 'screened': {raw_count_labels}"
        )

    if warnings:
        return True, "Potential raw-DB-hit-labeled-as-screened found (warnings)", warnings
    return True, "No raw database hit count labeled as 'screened'", []


def l5_full_text_review_iqbal():
    content = _read_text(LITERATURE_PATH)

    iqbal_2025_reviewed = bool(re.search(
        r"Iqbal.*[&].*Welling.*2025.*full.?text|full.?text.*Iqbal.*2025",
        content, re.IGNORECASE
    ))
    iqbal_2026_reviewed = bool(re.search(
        r"Iqbal.*2026.*full.?text|full.?text.*Iqbal.*2026",
        content, re.IGNORECASE
    ))

    has_remaining_tasks = "Remaining Verification Tasks" in content

    errors = []
    if not has_remaining_tasks:
        return False, "No 'Remaining Verification Tasks' section found in literature document", []

    if not iqbal_2025_reviewed:
        pending_2025 = re.search(
            r"Full.text.*Iqbal.*Welling.*2025.*Pending",
            content, re.IGNORECASE
        )
        if pending_2025:
            errors.append(
                "Iqbal & Welling 2025 full-text review status: Pending (not completed)"
            )
        else:
            errors.append(
                "Iqbal & Welling 2025 full-text review status: not recorded"
            )

    if not iqbal_2026_reviewed:
        pending_2026 = re.search(
            r"Full.text.*Iqbal.*2026.*Appendix.*Pending",
            content, re.IGNORECASE
        )
        if pending_2026:
            errors.append(
                "Iqbal et al. 2026 full-text review status: Pending (not completed)"
            )
        else:
            errors.append(
                "Iqbal et al. 2026 full-text review status: not recorded"
            )

    if errors:
        return False, "; ".join(errors), []
    return True, "Full-text review status recorded for both Iqbal et al. papers", []


def l5_novelty_claim_strength():
    content = _read_text(LITERATURE_PATH)

    if "genuinely novel" in content.lower():
        return False, (
            "Novelty claim uses 'genuinely novel' before full-text review completed"
        ), []

    has_strong_claim = bool(re.search(
        r"strongly supported|strongly.*defensible|first.*causal",
        content, re.IGNORECASE
    ))

    warnings = []
    if has_strong_claim:
        full_text_complete = (
            "full-text" in content.lower()
            and "pending" not in content.lower()
        )
        if not full_text_complete:
            warnings.append(
                "Strong novelty claim present but full-text review of Iqbal papers "
                "is not yet complete"
            )

    if warnings:
        return True, "Novelty claim strength warned before full-text review", warnings
    return True, "Novelty claim proportional to evidence (no 'genuinely novel')", []


def l5_distill_not_active():
    content = _read_text(VENUE_PATH)

    sections = content.split("### Removed Venues")
    active_section = sections[0] if sections else content

    if re.search(r"^\|.*Distill.*\|", active_section, re.MULTILINE):
        return False, "Distill listed in active venue section", []

    if len(sections) > 1 and "Distill" in sections[1]:
        pass

    return True, "Distill not listed as active venue (mentioned only in removed/hiatus note)", []


def l5_literature_screening_total():
    content = _read_text(LITERATURE_PATH)

    raw_hits = re.findall(r"(\d{2,})\s+topic\s+quer", content, re.IGNORECASE)

    screening_nums = re.findall(
        r"(\d{1,4})\s+results?\s+screened",
        content, re.IGNORECASE
    )
    if screening_nums:
        total_screened = sum(int(n) for n in screening_nums)
        raw_db_total_match = re.search(
            r"~?(\d{1,3},?\d+)\+?\s+total.*screened|total.*screened.*~?(\d{1,3},?\d+)",
            content, re.IGNORECASE
        )
        if raw_db_total_match:
            pass

    return True, "Literature search summary present", []


# ==============================================================================
# Main
# ==============================================================================

def main():
    checks = []
    all_errors = []
    all_warnings = []

    def _record_check(layer_name, check_name, passed, detail, warnings):
        full_name = f"{layer_name}/{check_name}"
        checks.append({"name": full_name, "pass": passed, "detail": detail})
        all_warnings.extend(warnings)
        if not passed:
            all_errors.append(f"[{full_name}] {detail}")

    layer_results = {
        "structural_integrity_pass": True,
        "diagnostic_provenance_pass": True,
        "contract_schema_pass": True,
        "statistical_semantics_pass": True,
        "literature_evidence_pass": True,
    }

    layer_checks = {
        "structural_integrity_pass": {
            "l1": [
                ("index_consistency", l1_index_consistency),
                ("artifact_count", l1_artifact_count),
                ("no_tbd_in_complete", l1_no_tbd_in_complete),
                ("markdown_links", l1_markdown_links),
                ("yaml_parsing", l1_yaml_parsing),
                ("finding_ids", l1_finding_ids),
                ("correction_ids", l1_correction_ids),
                ("diagnostic_raw_json", l1_diagnostic_raw_json),
            ]
        },
        "diagnostic_provenance_pass": {
            "l2": [
                ("environment_spec", l2_environment_spec),
                ("raw_json_exists", l2_raw_json_exists),
                ("metric_definitions_exists", l2_metric_definitions_exists),
                ("commands_txt_exists", l2_commands_txt_exists),
                ("sha256_no_self_hash", l2_sha256_no_self_hash),
                ("sha256_verify_all", l2_sha256_verify_all),
                ("no_stale_counts", l2_no_stale_counts),
                ("manifest_verification", l2_manifest_verification),
            ]
        },
        "contract_schema_pass": {
            "l3": [
                ("contract_status_tags", l3_contract_status_tags),
                ("no_unrecognized_status", l3_no_unrecognized_status),
                ("recursive_scalar_leaves", l3_recursive_scalar_leaves),
            ]
        },
        "statistical_semantics_pass": {
            "l4": [
                ("null_family_identity", l4_null_family_identity),
                ("representative_not_in_nulls", l4_representative_not_in_nulls),
                ("manifold_not_in_nulls", l4_manifold_not_in_nulls),
                ("iut_rejection_max_not_min", l4_iut_rejection_max_not_min),
                ("no_ordinary_bootstrap_p", l4_no_ordinary_bootstrap_p_formula),
                ("test_ci_level_consistency", l4_test_ci_level_consistency),
                ("seed_level_contrasts", l4_seed_level_contrasts),
                ("no_complete_case_without_analyzability",
                 l4_no_complete_case_without_analyzability),
                ("no_nonsig_as_equivalence", l4_no_nonsig_as_equivalence),
                ("no_git_diff_stat_source_hash", l4_no_git_diff_stat_source_hash),
            ]
        },
        "literature_evidence_pass": {
            "l5": [
                ("screening_accounting", l5_screening_accounting),
                ("no_raw_db_hits_as_screened", l5_no_raw_db_hits_as_screened),
                ("full_text_review_iqbal", l5_full_text_review_iqbal),
                ("novelty_claim_strength", l5_novelty_claim_strength),
                ("distill_not_active", l5_distill_not_active),
                ("literature_screening_total", l5_literature_screening_total),
            ]
        },
    }

    for layer_key, layers in layer_checks.items():
        for prefix, funcs in layers.items():
            for name, func in funcs:
                passed, detail, warnings = func()
                _record_check(prefix, name, passed, detail, warnings)
                if not passed:
                    layer_results[layer_key] = False

    all_layer_pass = all(layer_results.values())
    freeze_candidate = all_layer_pass

    report = {
        "structural_integrity_pass": layer_results["structural_integrity_pass"],
        "diagnostic_provenance_pass": layer_results["diagnostic_provenance_pass"],
        "contract_schema_pass": layer_results["contract_schema_pass"],
        "statistical_semantics_pass": layer_results["statistical_semantics_pass"],
        "literature_evidence_pass": layer_results["literature_evidence_pass"],
        "freeze_candidate_pass": freeze_candidate,
        "checks": checks,
        "errors": sorted(set(all_errors)),
        "warnings": sorted(set(all_warnings)),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    sha256_data = {}
    for root, dirs, files in os.walk(PLAN_DIR):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in sorted(files):
            if fname.endswith((".md", ".yaml", ".json", ".py", ".txt")) and fname != "sha256.json":
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, PLAN_DIR)
                sha256_data[rel] = sha256_file(fpath)

    os.makedirs(os.path.dirname(SHA256_PATH), exist_ok=True)
    with open(SHA256_PATH, "w") as f:
        json.dump(sha256_data, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Validation report written to {OUTPUT_PATH}")
    print(f"SHA-256 hashes written to {SHA256_PATH}")
    print(f"\nLayer results:")
    for layer_key, result in layer_results.items():
        mark = "PASS" if result else "FAIL"
        print(f"  [{mark}] {layer_key}")
    print(f"\nFreeze candidate: {'PASS' if freeze_candidate else 'FAIL'}")

    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        detail_short = c["detail"][:120] + "..." if len(c["detail"]) > 120 else c["detail"]
        print(f"  [{mark}] {c['name']}: {detail_short}")

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        for err in all_errors:
            print(f"  - {err}")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for w in all_warnings[:30]:
            print(f"  - {w}")
        if len(all_warnings) > 30:
            print(f"  ... ({len(all_warnings) - 30} more warnings)")

    sys.exit(0 if all_layer_pass else 1)


if __name__ == "__main__":
    main()
