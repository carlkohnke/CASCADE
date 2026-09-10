"""Run sequential, alternating legacy/CASCADE cube performance pairs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from urllib.parse import unquote, urlparse


ORACLE = Path("/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/TissueSim_cube_local.py")
ORACLE_SHA256 = "f2c4c89c8826b11246a239f7ae947872b955e5e47b44290033c13d1cef68bfda"
TREE_ROOT = Path("/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/trees_cache")
TREE_FAMILY = "a50c006ac491d07e399f94f1be37cfe0"
TREE_HASHES = {
    1: "8d0e6b8a3f6b9c0e28384337888d43c1df135d0a33292b4b7e98e426a63352be",
    10: "02e1c3e563f27481a1462108f2e8fb0a8ed1c996c23b4d54fcb556c3dadb385d",
    100: "ad547a2d40ddfcc93880d0dd51860c4390c67eb818540b72fb01a27ef5218283",
    1_000: "f7d0c209be4cd62e08a72be4e3e8fbf7f5604a7fd0dbea89e9962fc562b833b7",
    10_000: "e5051e152e6ca0b2773e40f0305b8babfcb04c3f305cf59c59e390b3e057da35",
    100_000: "245423388a9e97781c439eef12312ad479d7624bdd5a54e5945783ff1b64f062",
    1_000_000: "eda0008cbc0fcabd894adc917a221330ce1aaa7440adccedcd6f393d3db2d170",
    5_000_000: "8f1126a60b93ba777feea4b0f05e2127e623b22060baf4ad913567459c0af8c7",
}
POINTS = Path("/home/carl/svv_sweeps/GFM/validation/runs/2026-09-09_m2-staging_20744c8/fixtures/cube_seed42_points_1000000.npy")
POINTS_SHA256 = "251ca61e9a082c3ae10f46a882a77fde7cacc0312faaf0b2834d14e72004e6b7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path,
    quiet: bool = False,
    allowed_returncodes: tuple[int, ...] = (0,),
) -> int:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if completed.returncode not in allowed_returncodes:
        if quiet:
            print(completed.stdout, end="", flush=True)
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _installed_wheel_path(cascade_meta: dict) -> Path | None:
    direct_url = cascade_meta.get("cascade_direct_url")
    if not isinstance(direct_url, dict):
        return None
    url = direct_url.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _legacy_components(path: Path) -> dict[str, float]:
    row = next(csv.DictReader(path.open(newline="", encoding="utf-8")))
    result = {}
    for key, value in row.items():
        if key.startswith("t_") and key.endswith("_mean"):
            try:
                result[key.removesuffix("_mean")] = float(value)
            except (TypeError, ValueError):
                pass
    return result


def _stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--base-settings", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--fluid", choices=("blood", "water"), default="blood")
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--solver", default="topdown")
    parser.add_argument(
        "--finite-radius-o2-terms",
        choices=("none", "intravascular", "extravascular", "both"),
        default=None,
    )
    parser.add_argument("--lumen-wall-closure", choices=("wellmixed", "graetz"), default=None)
    parser.add_argument("--cext-grid", type=int, default=256)
    parser.add_argument("--cext-window-factor", type=float, default=6.0)
    parser.add_argument("--tissue-window-factor", type=float, default=6.0)
    parser.add_argument(
        "--cext-tissue-quadrature-mode",
        choices=("independent", "legacy_cext"),
        default="independent",
    )
    parser.add_argument(
        "--comparison-policy",
        choices=("required", "record"),
        default="required",
        help="Abort on a scientific comparison failure, or retain it and continue.",
    )
    parser.add_argument("--legacy-python", type=Path, required=True)
    parser.add_argument("--cascade-python", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--max-rss-gib", type=float, default=45.0)
    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.cext_grid <= 0:
        raise ValueError("--cext-grid must be positive")
    if args.cext_window_factor <= 0 or args.tissue_window_factor <= 0:
        raise ValueError("Cext and tissue window factors must be positive")
    unknown = sorted(set(args.targets) - set(TREE_HASHES))
    if unknown:
        raise ValueError(f"No frozen tree hash for targets: {unknown}")

    repo = args.repo.expanduser().resolve()
    harness = repo / "validation" / "harness"
    results = repo / "validation" / "results" / args.campaign_id
    runs = repo / "validation" / "runs" / args.campaign_id
    if results.exists() or runs.exists():
        raise FileExistsError(f"Refusing to overwrite campaign {args.campaign_id}")
    results.mkdir(parents=True)
    base = json.loads(args.base_settings.read_text(encoding="utf-8"))
    finite_radius_o2_terms = args.finite_radius_o2_terms or (
        "none" if args.solver == "topdown" else "both"
    )
    lumen_wall_closure = args.lumen_wall_closure or (
        "wellmixed" if args.solver == "topdown" else "graetz"
    )

    wheel = args.wheel.expanduser().resolve()
    actual_wheel_hash = _sha256(wheel)
    if actual_wheel_hash != args.wheel_sha256.lower():
        raise RuntimeError(f"Wheel hash mismatch: expected {args.wheel_sha256}, got {actual_wheel_hash}")

    pairs = []
    monitor_prefix = [
        sys.executable,
        str(harness / "run_monitored.py"),
        "--max-rss-gib",
        str(args.max_rss_gib),
        "--quiet-child",
    ]
    for target in args.targets:
        tree = TREE_ROOT / f"tree_{TREE_FAMILY}_t{target}.tree.npz"
        for repetition in range(1, args.repetitions + 1):
            label = f"cube-{target}-{args.fluid}-r{repetition:02d}"
            run_dir = runs / label
            raw = copy.deepcopy(base)
            raw["network"]["input_path"] = str(tree)
            raw["network"]["target_terminal_count"] = int(target)
            raw["simulation"]["fluid"] = args.fluid
            raw["simulation"]["concentration_solver"] = args.solver
            raw["simulation"]["tissue_accel"] = args.backend
            raw["simulation"]["tissuesim"]["finite_radius_o2_terms"] = finite_radius_o2_terms
            raw["simulation"]["tissuesim"]["lumen_wall_closure"] = lumen_wall_closure
            raw["simulation"]["tissuesim"]["window_factor"] = float(args.tissue_window_factor)
            raw["simulation"]["cext"] = {
                "accel_mode": args.backend,
                "frozen_accel_mode": args.backend,
                "init_mode": "decoupled_greens",
                "lambda_source": "lambda_t",
                "vess_coupling_accel": "anderson",
                "hybrid_bg_mode": "fft",
                "hybrid_bg_solver": "fft",
            }
            raw["settings"]["cext"]["window_factor"] = float(args.cext_window_factor)
            raw["settings"]["cext"]["vess_coupling_max_iter"] = 1
            raw["settings"]["cext"]["hybrid_bg_grid"] = int(args.cext_grid)
            raw["settings"].setdefault("tissue", {})["cext_tissue_quadrature_mode"] = (
                args.cext_tissue_quadrature_mode
            )
            raw["outputs"]["out_dir"] = str(run_dir / "cascade-output")
            raw["outputs"]["prefix"] = label.replace("-", "_")
            raw["outputs"]["write_paraview"] = False
            raw["outputs"]["write_summary_csv"] = True
            raw["outputs"]["write_segments_csv"] = False
            raw["outputs"]["write_points_csv"] = False
            raw["outputs"]["save_network"] = False
            raw["outputs"]["use_cache"] = False
            settings = results / f"{label}-settings.json"
            settings.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

            legacy_summary = run_dir / "legacy-summary.csv"
            legacy_record = results / f"{label}-legacy.json"
            cascade_record = results / f"{label}-cascade.json"
            legacy_monitor = results / f"{label}-legacy-monitor.json"
            cascade_monitor = results / f"{label}-cascade-monitor.json"
            comparison = results / f"{label}-summary-comparison.json"
            legacy_command = [
                *monitor_prefix, "--metadata", str(legacy_monitor), "--log", str(run_dir / "legacy.log"), "--",
                str(args.legacy_python.expanduser().absolute()), str(harness / "run_legacy_cube.py"),
                "--oracle", str(ORACLE), "--oracle-sha256", ORACLE_SHA256,
                "--points", str(POINTS), "--points-sha256", POINTS_SHA256,
                "--tree", str(tree), "--tree-sha256", TREE_HASHES[target],
                "--target", str(target), "--fluid", args.fluid, "--solver", args.solver,
                "--tissue-backend", args.backend, "--output", str(legacy_summary),
                "--cext-grid", str(args.cext_grid),
                "--cext-window-factor", str(args.cext_window_factor),
                "--tissue-window-factor", str(args.tissue_window_factor),
                "--cext-accel", args.backend,
                "--cext-frozen-accel", args.backend,
                "--finite-radius-o2-terms", finite_radius_o2_terms,
                "--lumen-wall-closure", lumen_wall_closure,
                "--metadata", str(legacy_record),
            ]
            cascade_command = [
                *monitor_prefix, "--metadata", str(cascade_monitor), "--log", str(run_dir / "cascade.log"), "--",
                # Do not resolve a virtual environment's Python symlink: invoking
                # its base interpreter path would discard the environment.
                str(args.cascade_python.expanduser().absolute()), str(harness / "run_cascade_cube_benchmark.py"),
                "--settings", str(settings), "--metadata", str(cascade_record),
            ]
            order = ["legacy", "cascade"] if repetition % 2 else ["cascade", "legacy"]
            for implementation in order:
                _run(legacy_command if implementation == "legacy" else cascade_command, cwd=repo)
            _run(
                [sys.executable, str(harness / "compare_csv.py"), "--legacy", str(legacy_summary),
                 "--cascade", str(run_dir / "cascade-output" / "summary.csv"), "--fluid", args.fluid,
                 "--output", str(comparison)],
                cwd=repo,
                quiet=True,
                allowed_returncodes=(0, 1) if args.comparison_policy == "record" else (0,),
            )

            legacy_meta = json.loads(legacy_record.read_text(encoding="utf-8"))
            cascade_meta = json.loads(cascade_record.read_text(encoding="utf-8"))
            installed_wheel = _installed_wheel_path(cascade_meta)
            if installed_wheel != wheel:
                raise RuntimeError(
                    "CASCADE interpreter is not running the declared wheel: "
                    f"expected {wheel}, installation metadata reports {installed_wheel}; "
                    f"module={cascade_meta.get('cascade_module')}"
                )
            legacy_resource = json.loads(legacy_monitor.read_text(encoding="utf-8"))
            cascade_resource = json.loads(cascade_monitor.read_text(encoding="utf-8"))
            comparison_data = json.loads(comparison.read_text(encoding="utf-8"))
            pair = {
                "target": target,
                "fluid": args.fluid,
                "backend": args.backend,
                "repetition": repetition,
                "classification": "cold_compile" if not pairs else "warmed_compile_cache",
                "execution_order": order,
                "summary_status": comparison_data["status"],
                "legacy_process_s": legacy_meta["elapsed_process_s"],
                "cascade_process_s": cascade_meta["elapsed_process_s"],
                "legacy_monitored_wall_s": legacy_resource["elapsed_wall_s"],
                "cascade_monitored_wall_s": cascade_resource["elapsed_wall_s"],
                "legacy_peak_rss_bytes": legacy_resource["peak_rss_bytes"],
                "cascade_peak_rss_bytes": cascade_resource["peak_rss_bytes"],
                "legacy_components": _legacy_components(legacy_summary),
                "cascade_components": cascade_meta["component_timings"],
                "cascade_build_wall_s": cascade_meta["build_wall_s"],
                "cascade_solve_wall_s": cascade_meta["solve_wall_s"],
                "cascade_export_wall_s": cascade_meta["export_wall_s"],
            }
            legacy_compute_fields = ("t_assembly_s", "t_kirchhoff_s", "t_concentration_s", "t_tissue_s")
            pair["legacy_comparable_compute_s"] = sum(
                float(pair["legacy_components"][name]) for name in legacy_compute_fields
            )
            cascade_summary = pair["cascade_components"][0]
            pair["cascade_comparable_compute_s"] = (
                float(cascade_meta["simulation_timings"].get("tree_0_tissue_cache_s", 0.0))
                + sum(float(cascade_summary[name]) for name in legacy_compute_fields)
            )
            pair["comparable_compute_ratio_cascade_over_legacy"] = (
                pair["cascade_comparable_compute_s"] / pair["legacy_comparable_compute_s"]
            )
            pair["process_ratio_cascade_over_legacy"] = pair["cascade_process_s"] / pair["legacy_process_s"]
            pair["monitored_ratio_cascade_over_legacy"] = pair["cascade_monitored_wall_s"] / pair["legacy_monitored_wall_s"]
            pair_path = results / f"{label}-pair.json"
            pair_path.write_text(json.dumps(pair, indent=2, allow_nan=True) + "\n", encoding="utf-8")
            pairs.append(pair)

    summaries = {}
    for target in args.targets:
        selected = [pair for pair in pairs if pair["target"] == target]
        summaries[str(target)] = {
            "all_process_s": {
                "legacy": _stats([pair["legacy_process_s"] for pair in selected]),
                "cascade": _stats([pair["cascade_process_s"] for pair in selected]),
            },
            "all_monitored_wall_s": {
                "legacy": _stats([pair["legacy_monitored_wall_s"] for pair in selected]),
                "cascade": _stats([pair["cascade_monitored_wall_s"] for pair in selected]),
            },
            "median_process_ratio": statistics.median(
                [pair["process_ratio_cascade_over_legacy"] for pair in selected]
            ),
            "median_monitored_ratio": statistics.median(
                [pair["monitored_ratio_cascade_over_legacy"] for pair in selected]
            ),
            "median_comparable_compute_ratio": statistics.median(
                [pair["comparable_compute_ratio_cascade_over_legacy"] for pair in selected]
            ),
            "comparable_compute_s": {
                "legacy": _stats([pair["legacy_comparable_compute_s"] for pair in selected]),
                "cascade": _stats([pair["cascade_comparable_compute_s"] for pair in selected]),
            },
            "all_scientific_summaries_pass": all(pair["summary_status"] == "pass" for pair in selected),
        }
        warmed = [pair for pair in selected if pair["classification"] == "warmed_compile_cache"]
        if warmed:
            summaries[str(target)]["warmed_process_s"] = {
                "legacy": _stats([pair["legacy_process_s"] for pair in warmed]),
                "cascade": _stats([pair["cascade_process_s"] for pair in warmed]),
            }
            summaries[str(target)]["warmed_median_process_ratio"] = statistics.median(
                [pair["process_ratio_cascade_over_legacy"] for pair in warmed]
            )
            summaries[str(target)]["warmed_median_monitored_ratio"] = statistics.median(
                [pair["monitored_ratio_cascade_over_legacy"] for pair in warmed]
            )
            summaries[str(target)]["warmed_median_comparable_compute_ratio"] = statistics.median(
                [pair["comparable_compute_ratio_cascade_over_legacy"] for pair in warmed]
            )

    report = {
        "schema_version": 1,
        "tracker_ids": [
            "PERF-01",
            "PERF-02",
            "PERF-04" if args.backend == "gpu" else "PERF-03",
            "PERF-07",
        ],
        "campaign_id": args.campaign_id,
        "wheel_path": str(wheel),
        "wheel_sha256": actual_wheel_hash,
        "oracle_path": str(ORACLE),
        "oracle_sha256": ORACLE_SHA256,
        "points_path": str(POINTS),
        "points_sha256": POINTS_SHA256,
        "repetitions_per_target": args.repetitions,
        "first_pair_policy": (
            "The first pair in the campaign is labeled cold_compile; later pairs use the "
            "persistent Numba/CUDA compile cache. Every pair is an isolated fresh process."
        ),
        "strictly_sequential": True,
        "alternating_order": True,
        "array_evidence_writes": False,
        "targets": args.targets,
        "fluid": args.fluid,
        "backend": args.backend,
        "solver": args.solver,
        "finite_radius_o2_terms": finite_radius_o2_terms,
        "lumen_wall_closure": lumen_wall_closure,
        "cext_grid": args.cext_grid,
        "cext_window_factor": args.cext_window_factor,
        "tissue_window_factor": args.tissue_window_factor,
        "cext_tissue_quadrature_mode": args.cext_tissue_quadrature_mode,
        "comparison_policy": args.comparison_policy,
        "comparable_compute_definition": (
            "legacy: t_assembly+t_kirchhoff+t_concentration+t_tissue; CASCADE: "
            "tree tissue-cache build plus the same four solver fields"
        ),
        "summaries": summaries,
        "pair_files": [f"cube-{p['target']}-{p['fluid']}-r{p['repetition']:02d}-pair.json" for p in pairs],
    }
    (results / "benchmark-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
