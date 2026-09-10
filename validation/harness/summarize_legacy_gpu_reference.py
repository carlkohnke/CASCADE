"""Freeze a compact summary of the historical GPU timing CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


TIMING_FIELDS = (
    "t_assembly_s_mean",
    "t_kirchhoff_s_mean",
    "t_concentration_s_mean",
    "t_tissue_s_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    actual_hash = _sha256(source)
    if actual_hash != args.sha256.lower():
        raise RuntimeError(f"Expected {args.sha256.lower()}, got {actual_hash}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        vessels = int(round(float(row["total_segments_mean"])))
        grouped.setdefault(vessels, []).append(row)
    cases = []
    for vessels, group in sorted(grouped.items()):
        samples = []
        for row in group:
            components = {name: float(row[name]) for name in TIMING_FIELDS}
            samples.append(
                {
                    "fluid": row["fluid"],
                    "target_terminals": int(round(float(row["target_terminals"]))),
                    "distance_sample_count": int(round(float(row["distance_sample_count"]))),
                    "components_s": components,
                    "total_compute_s": sum(components.values()),
                }
            )
        totals = [sample["total_compute_s"] for sample in samples]
        cases.append(
            {
                "number_vessels": vessels,
                "samples": samples,
                "total_compute_mean_s": statistics.mean(totals),
                "total_compute_min_s": min(totals),
                "total_compute_max_s": max(totals),
            }
        )

    report = {
        "schema_version": 1,
        "tracker_ids": ["PERF-01", "PERF-02", "PERF-04", "PERF-07"],
        "source_path": str(source),
        "source_sha256": actual_hash,
        "source_rows": len(rows),
        "timing_definition": " + ".join(TIMING_FIELDS),
        "excluded_from_timing": [
            "tree load",
            "Python/package startup",
            "domain construction",
            "output/export",
            "external monitoring",
        ],
        "profile": {
            "tissue_sample_count": 1_000_000,
            "working_float_dtype": "float32",
            "working_index_dtype": "int32",
            "concentration_solver": "topdown_ext_hybrid_bg",
            "cext_accel": "gpu",
            "cext_frozen_accel": "gpu",
            "tissue_accel": "gpu",
            "cext_quadrature": 1,
            "tissue_quadrature": 5,
            "cext_grid": 128,
            "cext_iterations": 1,
            "window_factor": 4,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "rows": len(rows), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
