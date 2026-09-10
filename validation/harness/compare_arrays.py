"""Stream-compare isolated legacy and CASCADE array evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXACT_ARRAYS = {
    "segment_id",
    "left_child_id",
    "right_child_id",
    "parent_id",
    "terminal_mask",
    "starts_cm",
    "ends_cm",
    "radii_cm",
    "lengths_cm",
    "tissue_points_cm",
}
PHYSICAL_ARRAYS = {
    "flows_cm3_s",
    "pressures_pa",
    "cin",
    "cout",
    "discharge_hematocrit",
    "tube_hematocrit",
    "tissue_values",
}


def _load_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1 or not isinstance(data.get("arrays"), dict):
        raise ValueError(f"Unsupported array evidence manifest: {path}")
    return data


def _chunks(length: int, chunk_rows: int):
    for start in range(0, length, chunk_rows):
        yield slice(start, min(start + chunk_rows, length))


def compare_exact(reference: np.ndarray, candidate: np.ndarray, *, chunk_rows: int) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"passed": False, "reason": "shape_mismatch"}
    mismatches = 0
    first_mismatch = None
    length = int(reference.shape[0]) if reference.ndim else 1
    for section in _chunks(length, chunk_rows):
        left = np.asarray(reference[section]) if reference.ndim else np.asarray(reference)
        right = np.asarray(candidate[section]) if candidate.ndim else np.asarray(candidate)
        equal = np.equal(left, right) | (np.isnan(left) & np.isnan(right)) if left.dtype.kind in "fc" else np.equal(left, right)
        count = int(equal.size - np.count_nonzero(equal))
        if count and first_mismatch is None:
            first_mismatch = int(section.start or 0) + int(np.flatnonzero(~equal.reshape(equal.shape[0], -1).all(axis=1))[0])
        mismatches += count
    return {
        "passed": mismatches == 0,
        "mismatched_values": mismatches,
        "first_mismatched_row": first_mismatch,
    }


def _reference_scale(reference: np.ndarray, *, chunk_rows: int) -> float:
    scale = 0.0
    length = int(reference.shape[0]) if reference.ndim else 1
    for section in _chunks(length, chunk_rows):
        values = np.asarray(reference[section], dtype=np.float64) if reference.ndim else np.asarray(reference, dtype=np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            scale = max(scale, float(np.max(np.abs(values[finite]))))
    return scale


def compare_physical(reference: np.ndarray, candidate: np.ndarray, *, chunk_rows: int) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {"passed": False, "reason": "shape_mismatch"}
    scale = _reference_scale(reference, chunk_rows=chunk_rows)
    absolute_tolerance = 1e-6 * scale
    max_absolute_error = 0.0
    max_relative_error = 0.0
    sum_squared_error = 0.0
    finite_values = 0
    finite_mask_mismatches = 0
    tolerance_failures = 0
    length = int(reference.shape[0]) if reference.ndim else 1
    for section in _chunks(length, chunk_rows):
        left = np.asarray(reference[section], dtype=np.float64) if reference.ndim else np.asarray(reference, dtype=np.float64)
        right = np.asarray(candidate[section], dtype=np.float64) if candidate.ndim else np.asarray(candidate, dtype=np.float64)
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        finite_mask_mismatches += int(np.count_nonzero(left_finite != right_finite))
        both = left_finite & right_finite
        if not np.any(both):
            continue
        error = np.abs(right[both] - left[both])
        relative = error / np.maximum(np.abs(left[both]), absolute_tolerance or np.finfo(float).tiny)
        accepted = (error <= absolute_tolerance) | (relative <= 1e-3)
        tolerance_failures += int(np.count_nonzero(~accepted))
        max_absolute_error = max(max_absolute_error, float(np.max(error)))
        max_relative_error = max(max_relative_error, float(np.max(relative)))
        sum_squared_error += float(np.dot(error, error))
        finite_values += int(error.size)
    return {
        "passed": finite_mask_mismatches == 0 and tolerance_failures == 0,
        "reference_case_scale": scale,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": 1e-3,
        "finite_mask_mismatches": finite_mask_mismatches,
        "tolerance_failures": tolerance_failures,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "rmse": math.sqrt(sum_squared_error / finite_values) if finite_values else 0.0,
        "finite_values": finite_values,
    }


def compare_directories(
    legacy_dir: Path,
    cascade_dir: Path,
    *,
    chunk_rows: int = 262_144,
) -> dict[str, Any]:
    legacy_manifest = _load_manifest(legacy_dir)
    cascade_manifest = _load_manifest(cascade_dir)
    names = sorted(EXACT_ARRAYS | PHYSICAL_ARRAYS)
    comparisons = []
    for name in names:
        legacy_entry = legacy_manifest["arrays"].get(name)
        cascade_entry = cascade_manifest["arrays"].get(name)
        if legacy_entry is None or cascade_entry is None:
            comparisons.append({"array": name, "passed": False, "reason": "missing_array"})
            continue
        reference = np.load(legacy_dir / legacy_entry["path"], mmap_mode="r", allow_pickle=False)
        candidate = np.load(cascade_dir / cascade_entry["path"], mmap_mode="r", allow_pickle=False)
        result = (
            compare_exact(reference, candidate, chunk_rows=chunk_rows)
            if name in EXACT_ARRAYS
            else compare_physical(reference, candidate, chunk_rows=chunk_rows)
        )
        comparisons.append(
            {
                "array": name,
                "rule": "exact" if name in EXACT_ARRAYS else "relative_or_scale_floor",
                "reference_shape": list(reference.shape),
                "candidate_shape": list(candidate.shape),
                "reference_dtype": str(reference.dtype),
                "candidate_dtype": str(candidate.dtype),
                **result,
            }
        )
    return {
        "status": "pass" if all(item["passed"] for item in comparisons) else "fail",
        "legacy_dir": str(legacy_dir.resolve()),
        "cascade_dir": str(cascade_dir.resolve()),
        "chunk_rows": int(chunk_rows),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=262_144)
    args = parser.parse_args()
    result = compare_directories(args.legacy_dir, args.cascade_dir, chunk_rows=args.chunk_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
