"""Compare frozen legacy and CASCADE heart VTP products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


IDENTITY_FIELDS = {
    "point_id",
    "inside_tissue",
    "tree_id",
    "local_segment_id",
    "global_segment_id",
}
COORDINATE_FIELDS = {"mesh_points", "x", "y", "z"}


def _field_comparison(name: str, reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    result = {
        "field": name,
        "shape_equal": reference.shape == candidate.shape,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        result.update({"passed": False, "reason": "shape_mismatch"})
        return result

    finite_equal = np.array_equal(np.isfinite(reference), np.isfinite(candidate))
    result["finite_mask_equal"] = bool(finite_equal)
    if name in IDENTITY_FIELDS or name in COORDINATE_FIELDS:
        mismatches = int(np.count_nonzero(reference != candidate))
        result.update(
            {
                "rule": "exact",
                "mismatch_count": mismatches,
                "passed": finite_equal and mismatches == 0,
            }
        )
        return result

    finite = np.isfinite(reference) & np.isfinite(candidate)
    reference_scale = float(np.max(np.abs(reference[finite]))) if np.any(finite) else 0.0
    # Heart concentration fields are float32 accelerator products.  A literal
    # 1e-6 field-unit floor keeps sub-micro-unit GPU ordering noise from being
    # promoted into a scientific mismatch when the reference value is zero.
    absolute_tolerance = max(1.0e-6 * reference_scale, 1.0e-6)
    absolute_error = np.abs(candidate[finite] - reference[finite])
    allowed = absolute_tolerance + 1.0e-3 * np.abs(reference[finite])
    failed = int(np.count_nonzero(absolute_error > allowed))
    result.update(
        {
            "rule": "rtol_1e-3_plus_max_case_scale_1e-6_or_field_1e-6",
            "reference_case_scale": reference_scale,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": 1.0e-3,
            "max_absolute_error": float(np.max(absolute_error)) if absolute_error.size else 0.0,
            "failure_count": failed,
            "passed": finite_equal and failed == 0,
        }
    )
    return result


def _mesh_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    require_same_fields: bool,
    require_point_order: bool,
) -> dict:
    reference = pv.read(reference_path)
    candidate = pv.read(candidate_path)
    reference_points = np.asarray(reference.points)
    candidate_points = np.asarray(candidate.points)
    if not require_point_order and reference_points.shape == candidate_points.shape:
        reference_points = reference_points[np.lexsort(reference_points.T[::-1])]
        candidate_points = candidate_points[np.lexsort(candidate_points.T[::-1])]
    result = {
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "point_count_equal": reference.n_points == candidate.n_points,
        "cell_count_equal": reference.n_cells == candidate.n_cells,
        "reference_point_count": reference.n_points,
        "candidate_point_count": candidate.n_points,
        "reference_cell_count": reference.n_cells,
        "candidate_cell_count": candidate.n_cells,
        "point_order_required": require_point_order,
        "points": _field_comparison("mesh_points", reference_points, candidate_points),
        "fields": [],
    }
    common = sorted(set(reference.point_data) & set(candidate.point_data))
    for name in common:
        result["fields"].append(
            _field_comparison(name, reference.point_data[name], candidate.point_data[name])
        )
    result["reference_only_fields"] = sorted(set(reference.point_data) - set(candidate.point_data))
    result["candidate_only_fields"] = sorted(set(candidate.point_data) - set(reference.point_data))
    required_fields_equal = (
        not require_same_fields
        or (not result["reference_only_fields"] and not result["candidate_only_fields"])
    )
    result["same_field_set_required"] = require_same_fields
    result["passed"] = bool(
        result["point_count_equal"]
        and result["cell_count_equal"]
        and result["points"]["passed"]
        and required_fields_equal
        and all(field["passed"] for field in result["fields"])
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--cascade-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    suffixes = ("forest_vessels.vtp", "forest_oxygen_points.vtp", "domain_boundary.vtp")
    meshes = {}
    for suffix in suffixes:
        name = f"{args.prefix}_{suffix}"
        meshes[suffix] = _mesh_comparison(
            args.legacy_dir / name,
            args.cascade_dir / name,
            require_same_fields=suffix != "domain_boundary.vtp",
            require_point_order=suffix != "domain_boundary.vtp",
        )

    tissue_path = args.cascade_dir / f"{args.prefix}_forest_oxygen_points.vtp"
    legacy_tissue_path = args.legacy_dir / f"{args.prefix}_forest_oxygen_points.vtp"
    legacy_tissue = pv.read(legacy_tissue_path)
    cascade_tissue = pv.read(tissue_path)
    fractions = {}
    for threshold_name, threshold in (("above_1pct", 0.01), ("above_5pct", 0.05)):
        legacy_fraction = float(np.mean(np.asarray(legacy_tissue["local_concentration_norm"]) >= threshold))
        cascade_fraction = float(np.mean(np.asarray(cascade_tissue["local_concentration_norm"]) >= threshold))
        error = abs(cascade_fraction - legacy_fraction)
        fractions[threshold_name] = {
            "legacy": legacy_fraction,
            "cascade": cascade_fraction,
            "absolute_error": error,
            "absolute_tolerance": 0.001,
            "passed": error <= 0.001,
        }

    report = {
        "schema_version": 1,
        "tracker_ids": ["VAL-06", "VAL-07"],
        "rules": {
            "identity_and_coordinates": "exact",
            "physical_fields": "rtol=1e-3, atol=max(1e-6 * reference case scale, 1e-6 field units)",
            "fractions": "absolute tolerance 0.001",
            "finite_masks": "exact",
        },
        "meshes": meshes,
        "fractions": fractions,
    }
    report["status"] = "pass" if (
        all(mesh["passed"] for mesh in meshes.values())
        and all(item["passed"] for item in fractions.values())
    ) else "fail"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
