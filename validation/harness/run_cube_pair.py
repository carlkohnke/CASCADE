"""Run one hash-frozen legacy/CASCADE cube pair sequentially and compare it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ORACLE = Path(
    "/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/TissueSim_cube_local.py"
)
ORACLE_SHA256 = "f2c4c89c8826b11246a239f7ae947872b955e5e47b44290033c13d1cef68bfda"
POINTS = Path(
    "/home/carl/svv_sweeps/GFM/validation/runs/2026-09-09_m2-staging_20744c8/fixtures/"
    "cube_seed42_points_1000000.npy"
)
POINTS_SHA256 = "251ca61e9a082c3ae10f46a882a77fde7cacc0312faaf0b2834d14e72004e6b7"


def _run(command: list[str], *, cwd: Path, quiet: bool = False) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL if quiet else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--fluid", choices=("blood", "water"), required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--tree-sha256", required=True)
    parser.add_argument("--tissue-backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--cascade-python", type=Path, required=True)
    parser.add_argument("--max-rss-gib", type=float, default=45.0)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    harness = repo / "validation" / "harness"
    results = repo / "validation" / "results" / args.campaign_id
    run_dir = repo / "validation" / "runs" / args.campaign_id / f"cube-{args.target}-{args.fluid}"
    prefix = f"cube-{args.target}-{args.fluid}"
    pair_result = results / f"{prefix}-pair.json"
    if pair_result.exists() or run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing pair evidence for {prefix}")
    results.mkdir(parents=True, exist_ok=True)

    legacy_monitor = results / f"{prefix}-legacy-monitor.json"
    legacy_metadata = results / f"{prefix}-legacy.json"
    cascade_monitor = results / f"{prefix}-cascade-monitor.json"
    cascade_metadata = results / f"{prefix}-cascade.json"
    array_comparison = results / f"{prefix}-arrays-comparison.json"
    summary_comparison = results / f"{prefix}-summary-comparison.json"

    monitor = [
        sys.executable,
        str(harness / "run_monitored.py"),
        "--max-rss-gib",
        str(args.max_rss_gib),
    ]
    legacy_command = [
        *monitor,
        "--metadata",
        str(legacy_monitor),
        "--log",
        str(run_dir / "legacy.log"),
        "--",
        str(args.legacy_python.resolve()),
        str(harness / "run_legacy_cube.py"),
        "--oracle",
        str(ORACLE),
        "--oracle-sha256",
        ORACLE_SHA256,
        "--points",
        str(POINTS),
        "--points-sha256",
        POINTS_SHA256,
        "--tree",
        str(args.tree.resolve()),
        "--tree-sha256",
        args.tree_sha256,
        "--target",
        str(args.target),
        "--fluid",
        args.fluid,
        "--solver",
        "topdown",
        "--tissue-backend",
        args.tissue_backend,
        "--output",
        str(run_dir / "legacy-summary.csv"),
        "--metadata",
        str(legacy_metadata),
        "--arrays-dir",
        str(run_dir / "legacy-arrays"),
    ]
    _run(legacy_command, cwd=repo)

    cascade_command = [
        *monitor,
        "--metadata",
        str(cascade_monitor),
        "--log",
        str(run_dir / "cascade.log"),
        "--",
        str(args.cascade_python.resolve()),
        str(harness / "run_cascade_cube.py"),
        "--settings",
        str(args.settings.resolve()),
        "--metadata",
        str(cascade_metadata),
        "--arrays-dir",
        str(run_dir / "cascade-arrays"),
    ]
    _run(cascade_command, cwd=repo)

    _run(
        [
            sys.executable,
            str(harness / "compare_arrays.py"),
            "--legacy-dir",
            str(run_dir / "legacy-arrays"),
            "--cascade-dir",
            str(run_dir / "cascade-arrays"),
            "--output",
            str(array_comparison),
        ],
        cwd=repo,
        quiet=True,
    )
    _run(
        [
            sys.executable,
            str(harness / "compare_csv.py"),
            "--legacy",
            str(run_dir / "legacy-summary.csv"),
            "--cascade",
            str(run_dir / "cascade-output" / "summary.csv"),
            "--fluid",
            args.fluid,
            "--output",
            str(summary_comparison),
        ],
        cwd=repo,
        quiet=True,
    )

    legacy_resource = json.loads(legacy_monitor.read_text(encoding="utf-8"))
    cascade_resource = json.loads(cascade_monitor.read_text(encoding="utf-8"))
    arrays = json.loads(array_comparison.read_text(encoding="utf-8"))
    summary = json.loads(summary_comparison.read_text(encoding="utf-8"))
    record = {
        "case": f"CUBE-{args.target}",
        "fluid": args.fluid,
        "tissue_backend": args.tissue_backend,
        "tree_path": str(args.tree.resolve()),
        "tree_sha256": args.tree_sha256,
        "settings_path": str(args.settings.resolve()),
        "array_status": arrays["status"],
        "summary_status": summary["status"],
        "legacy_wall_s_with_evidence": legacy_resource["elapsed_wall_s"],
        "cascade_wall_s_with_evidence": cascade_resource["elapsed_wall_s"],
        "legacy_peak_rss_bytes": legacy_resource["peak_rss_bytes"],
        "cascade_peak_rss_bytes": cascade_resource["peak_rss_bytes"],
        "legacy_peak_gpu_memory_bytes": legacy_resource["peak_gpu_memory_bytes"],
        "cascade_peak_gpu_memory_bytes": cascade_resource["peak_gpu_memory_bytes"],
        "performance_certifying": False,
        "performance_note": "Array-evidence writes are enabled; internal component timings are diagnostic only.",
    }
    pair_result.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
