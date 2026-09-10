"""Consolidate cube-pair evidence into a compact reviewable report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


CASE_RE = re.compile(r"^cube-(\d+)-(blood|water)-arrays-comparison\.json$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", action="append", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("validation/runs"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    cases: dict[tuple[int, str], tuple[Path, Path]] = {}
    for campaign_dir in args.campaign_dir:
        for array_path in sorted(campaign_dir.glob("cube-*-arrays-comparison.json")):
            match = CASE_RE.match(array_path.name)
            if match:
                cases[(int(match.group(1)), match.group(2))] = (campaign_dir, array_path)

    records = []
    for (target, fluid), (campaign_dir, array_path) in sorted(cases.items()):
        stem = f"cube-{target}-{fluid}"
        arrays = load_json(array_path)
        summary = load_json(campaign_dir / f"{stem}-summary-comparison.json")
        legacy_monitor = load_json(campaign_dir / f"{stem}-legacy-monitor.json")
        cascade_monitor = load_json(campaign_dir / f"{stem}-cascade-monitor.json")
        cascade = load_json(campaign_dir / f"{stem}-cascade.json")
        legacy = load_json(campaign_dir / f"{stem}-legacy.json")
        raw_case = args.runs_root / campaign_dir.name / stem
        legacy_row = load_csv_row(raw_case / "legacy-summary.csv")

        exact_mismatches = 0
        tolerance_failures = 0
        max_abs = 0.0
        max_rel = 0.0
        for comparison in arrays["comparisons"]:
            exact_mismatches += int(comparison.get("mismatched_values", 0))
            tolerance_failures += int(comparison.get("tolerance_failures", 0))
            max_abs = max(max_abs, float(comparison.get("max_absolute_error", 0.0)))
            max_rel = max(max_rel, float(comparison.get("max_relative_error", 0.0)))

        records.append(
            {
                "target_terminals": target,
                "total_segments": 2 * target + 1,
                "fluid": fluid,
                "backend": legacy["tissue_backend"],
                "tree_sha256": legacy["tree_sha256"],
                "array_status": arrays["status"],
                "summary_status": summary["status"],
                "arrays_compared": len(arrays["comparisons"]),
                "summary_fields_compared": len(summary["comparisons"]),
                "exact_mismatches": exact_mismatches,
                "tolerance_failures": tolerance_failures,
                "max_absolute_error": max_abs,
                "max_relative_error": max_rel,
                "legacy_wall_s_with_evidence": legacy_monitor["elapsed_wall_s"],
                "cascade_wall_s_with_evidence": cascade_monitor["elapsed_wall_s"],
                "legacy_peak_rss_gib": legacy_monitor["peak_rss_bytes"] / 1024**3,
                "cascade_peak_rss_gib": cascade_monitor["peak_rss_bytes"] / 1024**3,
                "legacy_network_load_s": float(legacy_row["t_load_s_mean"]),
                "cascade_network_load_s": cascade["build_timings"]["network_load_s"],
                "performance_certifying": False,
                "campaign": campaign_dir.name,
            }
        )

    status = "pass" if records and all(
        row["array_status"] == "pass" and row["summary_status"] == "pass" for row in records
    ) else "fail"
    report = {
        "schema_version": 1,
        "status": status,
        "case_count": len(records),
        "scope": "existing-tree cube numerical characterization",
        "performance_certifying": False,
        "records": records,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Cube existing-tree characterization",
        "",
        f"Overall status: **{status.upper()}** ({len(records)} fluid/scale cases).",
        "",
        "Every row used one explicit SHA-256-checked tree and the shared one-million-point coordinate fixture. "
        "Wall times include evidence-array writes and are diagnostic, not M3 certification.",
        "",
        "| Terminals | Segments | Fluid | Backend | Arrays | Summary | Exact mismatches | Tol failures | Legacy/CASCADE wall (s) | Legacy/CASCADE RSS (GiB) | Legacy/CASCADE load (s) |",
        "|---:|---:|:---|:---|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['target_terminals']:,} | {row['total_segments']:,} | {row['fluid']} | "
            f"{row['backend']} | {row['array_status']} ({row['arrays_compared']}) | "
            f"{row['summary_status']} ({row['summary_fields_compared']}) | "
            f"{row['exact_mismatches']} | {row['tolerance_failures']} | "
            f"{row['legacy_wall_s_with_evidence']:.2f}/{row['cascade_wall_s_with_evidence']:.2f} | "
            f"{row['legacy_peak_rss_gib']:.2f}/{row['cascade_peak_rss_gib']:.2f} | "
            f"{row['legacy_network_load_s']:.3f}/{row['cascade_network_load_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The maximum reported relative error is not used alone for pass/fail near zero; each physical field "
            "uses the frozen relative-or-reference-scale-floor rule, with finite-mask equality required.",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
