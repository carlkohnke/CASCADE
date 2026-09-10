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


def _field_comparison(
    name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    max_sparse_outlier_fraction: float = 0.0,
) -> dict:
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
    failure_fraction = float(failed / absolute_error.size) if absolute_error.size else 0.0
    result.update(
        {
            "rule": "rtol_1e-3_plus_max_case_scale_1e-6_or_field_1e-6",
            "reference_case_scale": reference_scale,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": 1.0e-3,
            "max_absolute_error": float(np.max(absolute_error)) if absolute_error.size else 0.0,
            "failure_count": failed,
            "failure_fraction": failure_fraction,
            "max_sparse_outlier_fraction": float(max_sparse_outlier_fraction),
            "passed": finite_equal and failure_fraction <= float(max_sparse_outlier_fraction),
        }
    )
    return result


def _mesh_comparison(
    reference_path: Path,
    candidate_path: Path,
    *,
    require_same_fields: bool,
    require_point_order: bool,
    max_sparse_outlier_fraction: float,
    align_by_coordinates: bool = False,
    max_coordinate_set_difference_fraction: float = 0.0,
) -> dict:
    reference = pv.read(reference_path)
    candidate = pv.read(candidate_path)
    reference_points = np.asarray(reference.points)
    candidate_points = np.asarray(candidate.points)
    aligned_reference_rows = None
    aligned_candidate_rows = None
    coordinate_alignment = None
    if align_by_coordinates:
        ref_contiguous = np.ascontiguousarray(reference_points)
        cand_contiguous = np.ascontiguousarray(candidate_points)
        if ref_contiguous.ndim != 2 or ref_contiguous.shape[1] != 3:
            raise ValueError("Coordinate alignment requires three-component point arrays")
        if ref_contiguous.dtype != cand_contiguous.dtype:
            cand_contiguous = cand_contiguous.astype(ref_contiguous.dtype)
        key_dtype = np.dtype((np.void, ref_contiguous.dtype.itemsize * 3))
        ref_keys = ref_contiguous.view(key_dtype).reshape(-1)
        cand_keys = cand_contiguous.view(key_dtype).reshape(-1)
        if np.unique(ref_keys).size != ref_keys.size or np.unique(cand_keys).size != cand_keys.size:
            raise ValueError("Coordinate alignment requires unique point coordinates")
        _, aligned_reference_rows, aligned_candidate_rows = np.intersect1d(
            ref_keys,
            cand_keys,
            assume_unique=True,
            return_indices=True,
        )
        reference_only = int(ref_keys.size - aligned_reference_rows.size)
        candidate_only = int(cand_keys.size - aligned_candidate_rows.size)
        difference_fraction = float(
            max(reference_only / max(ref_keys.size, 1), candidate_only / max(cand_keys.size, 1))
        )
        coordinate_alignment = {
            "mode": "exact_coordinate_intersection",
            "common_points": int(aligned_reference_rows.size),
            "reference_only_points": reference_only,
            "candidate_only_points": candidate_only,
            "difference_fraction": difference_fraction,
            "max_difference_fraction": float(max_coordinate_set_difference_fraction),
            "passed": difference_fraction <= float(max_coordinate_set_difference_fraction),
        }
        reference_points = reference_points[aligned_reference_rows]
        candidate_points = candidate_points[aligned_candidate_rows]
    elif not require_point_order and reference_points.shape == candidate_points.shape:
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
        "coordinate_alignment": coordinate_alignment,
        "points": _field_comparison("mesh_points", reference_points, candidate_points),
        "fields": [],
    }
    common = sorted(set(reference.point_data) & set(candidate.point_data))
    for name in common:
        if align_by_coordinates and name == "point_id":
            continue
        reference_values = np.asarray(reference.point_data[name])
        candidate_values = np.asarray(candidate.point_data[name])
        if align_by_coordinates:
            reference_values = reference_values[aligned_reference_rows]
            candidate_values = candidate_values[aligned_candidate_rows]
        result["fields"].append(
            _field_comparison(
                name,
                reference_values,
                candidate_values,
                max_sparse_outlier_fraction=max_sparse_outlier_fraction,
            )
        )
    result["alignment_ignored_fields"] = ["point_id"] if align_by_coordinates else []
    result["reference_only_fields"] = sorted(set(reference.point_data) - set(candidate.point_data))
    result["candidate_only_fields"] = sorted(set(candidate.point_data) - set(reference.point_data))
    required_fields_equal = (
        not require_same_fields
        or (not result["reference_only_fields"] and not result["candidate_only_fields"])
    )
    result["same_field_set_required"] = require_same_fields
    result["passed"] = bool(
        (coordinate_alignment["passed"] if align_by_coordinates else result["point_count_equal"])
        and (result["cell_count_equal"] or align_by_coordinates)
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
    parser.add_argument(
        "--allow-sparse-physical-outlier-fraction",
        type=float,
        default=0.0,
        help=(
            "Allow this fraction of physical-field values to exceed the pointwise "
            "tolerance; identity fields, coordinates, finite masks, and fractions "
            "remain strict. The default is zero."
        ),
    )
    parser.add_argument(
        "--skip-vessel-mesh",
        action="store_true",
        help=(
            "Compare tissue and domain outputs only. Use for explicitly compact "
            "production-scale campaigns after vessel schema is certified on HEART-S."
        ),
    )
    parser.add_argument(
        "--align-tissue-by-coordinates",
        action="store_true",
        help="Align tissue fields on their exact shared coordinates before comparison.",
    )
    parser.add_argument(
        "--allow-tissue-coordinate-set-difference-fraction",
        type=float,
        default=0.0,
        help="Allowed fraction of vessel-boundary tissue points present on only one side.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.allow_sparse_physical_outlier_fraction <= 1.0:
        raise ValueError("--allow-sparse-physical-outlier-fraction must be in [0, 1]")
    if not 0.0 <= args.allow_tissue_coordinate_set_difference_fraction <= 1.0:
        raise ValueError("--allow-tissue-coordinate-set-difference-fraction must be in [0, 1]")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    suffixes = ("forest_vessels.vtp", "forest_oxygen_points.vtp", "domain_boundary.vtp")
    meshes = {}
    for suffix in suffixes:
        if args.skip_vessel_mesh and suffix == "forest_vessels.vtp":
            continue
        name = f"{args.prefix}_{suffix}"
        meshes[suffix] = _mesh_comparison(
            args.legacy_dir / name,
            args.cascade_dir / name,
            require_same_fields=suffix != "domain_boundary.vtp",
            require_point_order=suffix != "domain_boundary.vtp",
            max_sparse_outlier_fraction=float(args.allow_sparse_physical_outlier_fraction),
            align_by_coordinates=(
                bool(args.align_tissue_by_coordinates)
                and suffix == "forest_oxygen_points.vtp"
            ),
            max_coordinate_set_difference_fraction=float(
                args.allow_tissue_coordinate_set_difference_fraction
            ),
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
            "max_sparse_physical_outlier_fraction": float(
                args.allow_sparse_physical_outlier_fraction
            ),
            "tissue_coordinate_alignment": (
                "exact shared-coordinate intersection"
                if args.align_tissue_by_coordinates
                else "disabled"
            ),
            "max_tissue_coordinate_set_difference_fraction": float(
                args.allow_tissue_coordinate_set_difference_fraction
            ),
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
