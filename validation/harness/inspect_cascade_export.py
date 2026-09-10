"""Inspect a bounded CASCADE export and cross-check CSV and VTK fields."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyvista as pv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vtk_record(path: Path) -> tuple[object, dict[str, object]]:
    dataset = pv.read(path)
    record = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "type": type(dataset).__name__,
        "n_points": int(dataset.n_points),
        "n_cells": int(dataset.n_cells),
        "bounds": [float(value) for value in dataset.bounds],
        "point_data": {
            name: {"dtype": str(np.asarray(dataset.point_data[name]).dtype), "shape": list(np.asarray(dataset.point_data[name]).shape)}
            for name in sorted(dataset.point_data.keys())
        },
        "cell_data": {
            name: {"dtype": str(np.asarray(dataset.cell_data[name]).dtype), "shape": list(np.asarray(dataset.cell_data[name]).shape)}
            for name in sorted(dataset.cell_data.keys())
        },
    }
    return dataset, record


def max_absolute_error(first: np.ndarray, second: np.ndarray) -> float:
    difference = np.abs(first - second)
    finite = difference[np.isfinite(difference)]
    return float(np.max(finite)) if finite.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-domain", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    required = ["manifest.json", "summary.csv", "segments.csv", "points.csv", "vessels.vtp", "oxygen_points.vtp"]
    if args.require_domain:
        required.append("domain_boundary.vtp")
    missing = [name for name in required if not (out_dir / name).is_file()]
    checks: list[dict[str, object]] = []
    datasets: dict[str, object] = {}
    vtk_files: dict[str, dict[str, object]] = {}
    for path in sorted(out_dir.glob("*.vt*")):
        dataset, record = vtk_record(path)
        datasets[path.name] = dataset
        vtk_files[path.name] = record
        checks.append({"check": f"{path.name}:nonempty", "passed": dataset.n_points > 0})

    with (out_dir / "segments.csv").open(newline="", encoding="utf-8") as handle:
        segment_rows = list(csv.DictReader(handle))
    vessels = datasets.get("vessels.vtp")
    if vessels is not None:
        ids = np.asarray(vessels.point_data["global_segment_id"])
        unique_ids, first = np.unique(ids, return_index=True)
        checks.append({"check": "vessels:one-cell-per-segment", "passed": int(vessels.n_cells) == len(segment_rows)})
        checks.append({"check": "vessels:all-segment-ids", "passed": unique_ids.tolist() == list(range(len(segment_rows)))})
        for name in ("flow_cm3_s", "flow_ul_min", "pressure_pa", "radius_cm", "length_cm", "concentration"):
            present = name in vessels.point_data
            checks.append({"check": f"vessels:field:{name}", "passed": present})
        for name in ("flow_cm3_s", "flow_ul_min", "pressure_pa", "radius_cm", "length_cm"):
            if name in vessels.point_data:
                vtk_values = np.asarray(vessels.point_data[name])[first]
                csv_values = np.asarray([float(row[name]) for row in segment_rows])
                checks.append(
                    {
                        "check": f"vessels:csv-match:{name}",
                        "passed": bool(np.allclose(vtk_values, csv_values, rtol=1e-5, atol=1e-7, equal_nan=True)),
                        "max_absolute_error": max_absolute_error(vtk_values, csv_values),
                    }
                )

    with (out_dir / "points.csv").open(newline="", encoding="utf-8") as handle:
        point_rows = list(csv.DictReader(handle))
    oxygen = datasets.get("oxygen_points.vtp")
    if oxygen is not None:
        checks.append({"check": "oxygen:point-count", "passed": int(oxygen.n_points) == len(point_rows)})
        for name in ("point_id", "local_concentration", "viability"):
            checks.append({"check": f"oxygen:field:{name}", "passed": name in oxygen.point_data})
        if "point_id" in oxygen.point_data:
            expected_ids = np.asarray([int(row["point_id"]) for row in point_rows])
            checks.append(
                {
                    "check": "oxygen:csv-match:point_id",
                    "passed": bool(np.array_equal(np.asarray(oxygen.point_data["point_id"]), expected_ids)),
                }
            )
        if "local_concentration" in oxygen.point_data:
            vtk_values = np.asarray(oxygen.point_data["local_concentration"])
            csv_values = np.asarray([float(row["local_concentration"]) for row in point_rows])
            checks.append(
                {
                    "check": "oxygen:csv-match:local_concentration",
                    "passed": bool(np.allclose(vtk_values, csv_values, rtol=1e-5, atol=1e-7, equal_nan=True)),
                    "max_absolute_error": max_absolute_error(vtk_values, csv_values),
                }
            )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    status = "pass" if not missing and checks and all(bool(check["passed"]) for check in checks) else "fail"
    result = {
        "schema_version": 1,
        "status": status,
        "out_dir": str(out_dir),
        "missing_required_files": missing,
        "manifest_sha256": sha256(out_dir / "manifest.json"),
        "manifest": manifest,
        "vtk_files": vtk_files,
        "segment_rows": len(segment_rows),
        "point_rows": len(point_rows),
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "checks": len(checks), "missing": missing}, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
