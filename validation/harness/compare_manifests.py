#!/usr/bin/env python3
"""Compare reproducibility-relevant CASCADE manifest content."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


VOLATILE_PATHS = (
    "created_at_utc",
    "settings_path",
    "inputs.settings_sha256",
    "settings.outputs",
    "resolved.prefix",
    "environment.python_executable",
    "timings",
    "outputs",
)


def _remove_path(value: dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(value[key], child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    stable_reference = copy.deepcopy(reference)
    stable_candidate = copy.deepcopy(candidate)
    for path in VOLATILE_PATHS:
        _remove_path(stable_reference, path)
        _remove_path(stable_candidate, path)

    ref_flat = _flatten(stable_reference)
    cand_flat = _flatten(stable_candidate)
    all_paths = sorted(set(ref_flat) | set(cand_flat))
    mismatches = [
        {
            "path": path,
            "reference": ref_flat.get(path, "<missing>"),
            "candidate": cand_flat.get(path, "<missing>"),
        }
        for path in all_paths
        if ref_flat.get(path, "<missing>") != cand_flat.get(path, "<missing>")
    ]
    result = {
        "status": "pass" if not mismatches else "fail",
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "excluded_volatile_paths": list(VOLATILE_PATHS),
        "stable_leaf_count": len(all_paths),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
