from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


CASCADE_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("CASCADE_AUDIT_PYTHON", sys.executable))

COMPARE_KEYS = (
    "total_segments_mean",
    "terminal_segments_mean",
    "total_volume_mean",
    "total_length_mean",
    "avg_radius_mean",
    "radius_min_mean",
    "radius_max_mean",
    "dRnet_mean",
    "Qmin_over_Qinlet_mean",
    "C_LQ_over_Cmax_mean",
    "C_tiss_over_Cmax_mean",
    "FracAbove1pct_mean",
    "FracAbove5pct_mean",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the CASCADE CLI against legacy TissueSim paths.")
    parser.add_argument("--work-dir", default=".cascade_audit_runs", help="Scratch output directory under CASCADE.")
    parser.add_argument("--targets", nargs="*", type=int, default=[1, 100], help="Tree target counts to compare.")
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-cext", action="store_true")
    args = parser.parse_args(argv)

    work_dir = resolve_work_dir(args.work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    results: list[dict[str, Any]] = []
    for target in args.targets:
        results.append(run_tree_parity_case(target, int(args.sample_count), work_dir, timeout=int(args.timeout)))
    if not args.skip_cext:
        cext_target = min(max(int(args.targets[0]), 1), 25)
        results.append(run_tree_parity_case(
            cext_target,
            int(args.sample_count),
            work_dir,
            timeout=int(args.timeout),
            solver="topdown_ext_hybrid_bg",
            name=f"tree_cext_t{cext_target}",
        ))
    results.append(run_equal_geometry_case(work_dir, timeout=int(args.timeout)))
    results.append(run_grid_export_case(work_dir, timeout=int(args.timeout)))
    results.append(run_simple_channel_case(work_dir, timeout=int(args.timeout)))
    forest_result = run_forest_smoke_case(work_dir, timeout=int(args.timeout))
    results.append(forest_result)
    results.append(run_forest_cache_reload_case(work_dir, timeout=int(args.timeout)))
    results.append(run_forest_direct_simcache_case(work_dir, timeout=int(args.timeout)))
    results.append(run_scheduled_growth_case(work_dir, timeout=int(args.timeout)))
    results.append(run_nearest_tree_growth_case(work_dir, timeout=int(args.timeout)))
    report = build_report(results)
    report_path = work_dir / "audit_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nAudit report: {report_path}")
    return 0 if all(r.get("status") == "pass" for r in results) else 1


def resolve_work_dir(value: str) -> Path:
    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else CASCADE_ROOT / raw
    path = path.resolve()
    try:
        path.relative_to(CASCADE_ROOT)
    except ValueError as exc:
        raise ValueError(f"Audit work directory must stay under {CASCADE_ROOT}: {path}") from exc
    return path


def run_tree_parity_case(
    target: int,
    sample_count: int,
    work_dir: Path,
    *,
    timeout: int,
    solver: str = "topdown",
    name: str | None = None,
) -> dict[str, Any]:
    case = name or f"tree_{solver}_t{target}"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = case_dir / "legacy_tissuesim.csv"
    legacy_output_name = f"cascade_audit_{case}.csv"
    legacy_output_path = CASCADE_ROOT / legacy_output_name
    if legacy_output_path.exists():
        legacy_output_path.unlink()
    cli_dir = case_dir / "cli"
    settings_path = case_dir / "settings.json"
    settings = tree_settings(target, sample_count, cli_dir, solver=solver)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    legacy_cmd = [
        str(PYTHON),
        "TissueSim_cube_local.py",
        "--target-counts",
        str(int(target)),
        "--distance-sample-count",
        str(int(sample_count)),
        "--tissue-gpu-validate-points",
        "0",
        "--concentration-solver",
        solver,
        "--finite-radius-o2-terms",
        "none" if solver == "topdown" else "both",
        "--lumen-wall-closure",
        "wellmixed" if solver == "topdown" else "graetz",
        "--output-csv",
        legacy_output_name,
    ]
    cli_cmd = [str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)]
    legacy = run_cmd(legacy_cmd, timeout=timeout)
    cli = run_cmd(cli_cmd, timeout=timeout)
    out: dict[str, Any] = {
        "case": case,
        "kind": "tree_parity",
        "target": int(target),
        "solver": solver,
        "legacy_elapsed_s": legacy["elapsed_s"],
        "cli_elapsed_s": cli["elapsed_s"],
        "legacy_returncode": legacy["returncode"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if legacy["returncode"] != 0 or cli["returncode"] != 0:
        out["notes"].append("Command failure; inspect stdout/stderr in the work directory.")
        write_logs(case_dir, legacy=legacy, cli=cli)
        return out
    if legacy_output_path.exists():
        legacy_output_path.replace(baseline_csv)
    legacy_row = select_csv_row(baseline_csv, fluid="blood")
    cli_row = select_csv_row(cli_dir / "summary.csv", fluid="blood")
    out["comparisons"] = compare_rows(legacy_row, cli_row)
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, legacy=legacy, cli=cli)
    return out


def run_equal_geometry_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "tree_equal_geometry"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "target_terminal_count": 100,
            "root": {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
        },
        "growth": {
            "n_closest_vessels": 2,
            "n_points": 50,
            "ignore_collisions": True,
            "allow_inside_vessels": True,
            "n_equal_bifurcations": 2,
            "equal_terminal": {"length": 0.05, "length_shrink": 0.5, "report_timings": False},
        },
        "simulation": {"fluid": "blood", "build_fluid": "blood", "distance_sample_count": 0, "geometry_only": True},
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "equal", "write_paraview": True, "save_network": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "equal_geometry",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI equal-terminal geometry run failed.")
        write_logs(case_dir, cli=cli)
        return out
    summary = select_csv_row(case_dir / "cli" / "summary.csv", fluid="blood")
    segments = read_csv(case_dir / "cli" / "segments.csv")
    expected_segments = 201
    actual_segments = int(float(summary.get("total_segments_mean", "nan")))
    out["comparisons"].append({
        "key": "total_segments_mean",
        "legacy": expected_segments,
        "cli": actual_segments,
        "abs_diff": abs(actual_segments - expected_segments),
        "rel_diff": 0.0 if actual_segments == expected_segments else math.inf,
        "ok": actual_segments == expected_segments and len(segments) == expected_segments,
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_forest_smoke_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "forest_smoke"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "forest",
            "target_terminal_counts": [100, 100],
            "roots": [
                {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
                {"start": [-0.49, 0.49, 0.49], "direction": [0.49, -0.49, -0.49]},
            ],
        },
        "growth": {"n_closest_vessels": 2, "n_points": 50, "ignore_collisions": True, "allow_inside_vessels": True},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "flow_source": "total_split",
            "concentration_solver": "topdown",
            "distance_sample_count": 1000,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "forest", "write_paraview": True, "save_network": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "forest_smoke",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI forest smoke failed.")
        write_logs(case_dir, cli=cli)
        return out
    cli_dir = case_dir / "cli"
    required = ["summary.csv", "segments.csv", "points.csv", "vessels.vtp", "oxygen_points.vtp", "manifest.json", "forest.forest", "forest.forest.simcache"]
    for name in required:
        exists = (cli_dir / name).exists()
        out["comparisons"].append({"key": f"exists:{name}", "legacy": True, "cli": exists, "abs_diff": 0, "rel_diff": 0, "ok": exists})
    summary_rows = read_csv(cli_dir / "summary.csv")
    segment_rows = read_csv(cli_dir / "segments.csv")
    out["comparisons"].append({
        "key": "summary_rows",
        "legacy": 2,
        "cli": len(summary_rows),
        "abs_diff": abs(len(summary_rows) - 2),
        "rel_diff": 0.0,
        "ok": len(summary_rows) == 2,
    })
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": 402,
        "cli": len(segment_rows),
        "abs_diff": abs(len(segment_rows) - 402),
        "rel_diff": 0.0,
        "ok": len(segment_rows) == 402,
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_forest_cache_reload_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "forest_cache_reload"
    source_dir = work_dir / "forest_smoke" / "cli"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "forest",
            "input_path": str(source_dir / "forest.forest"),
            "target_terminal_counts": [100, 100],
        },
        "growth": {"enabled": False, "n_closest_vessels": 2, "n_points": 50},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "flow_source": "total_split",
            "concentration_solver": "topdown",
            "distance_sample_count": 1000,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "forest_reload", "write_paraview": True, "save_network": False, "use_cache": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "forest_cache_reload",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI forest cache reload failed.")
        write_logs(case_dir, cli=cli)
        return out

    baseline_summary = read_csv(source_dir / "summary.csv")
    reload_summary = read_csv(case_dir / "cli" / "summary.csv")
    baseline_segments = read_csv(source_dir / "segments.csv")
    reload_segments = read_csv(case_dir / "cli" / "segments.csv")
    out["comparisons"].append({
        "key": "summary_rows",
        "legacy": len(baseline_summary),
        "cli": len(reload_summary),
        "abs_diff": abs(len(reload_summary) - len(baseline_summary)),
        "rel_diff": 0.0,
        "ok": len(reload_summary) == len(baseline_summary),
    })
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": len(baseline_segments),
        "cli": len(reload_segments),
        "abs_diff": abs(len(reload_segments) - len(baseline_segments)),
        "rel_diff": 0.0,
        "ok": len(reload_segments) == len(baseline_segments),
    })
    for row_idx, (baseline, reloaded) in enumerate(zip(baseline_summary, reload_summary)):
        for comp in compare_rows(baseline, reloaded):
            comp["key"] = f"tree_{row_idx}:{comp['key']}"
            out["comparisons"].append(comp)
    out["status"] = "pass" if out["comparisons"] and all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_forest_direct_simcache_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "forest_direct_simcache"
    source_dir = work_dir / "forest_smoke" / "cli"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "forest",
            "input_path": str(source_dir / "forest.forest.simcache"),
            "target_terminal_counts": [100, 100],
        },
        "growth": {"enabled": False, "n_closest_vessels": 2, "n_points": 50},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "flow_source": "total_split",
            "concentration_solver": "topdown",
            "distance_sample_count": 1000,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "forest_direct_simcache", "write_paraview": True, "save_network": False, "use_cache": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "forest_direct_simcache",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI direct simulation-cache load failed.")
        write_logs(case_dir, cli=cli)
        return out

    baseline_summary = read_csv(source_dir / "summary.csv")
    reload_summary = read_csv(case_dir / "cli" / "summary.csv")
    baseline_segments = read_csv(source_dir / "segments.csv")
    reload_segments = read_csv(case_dir / "cli" / "segments.csv")
    out["comparisons"].append({
        "key": "summary_rows",
        "legacy": len(baseline_summary),
        "cli": len(reload_summary),
        "abs_diff": abs(len(reload_summary) - len(baseline_summary)),
        "rel_diff": 0.0,
        "ok": len(reload_summary) == len(baseline_summary),
    })
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": len(baseline_segments),
        "cli": len(reload_segments),
        "abs_diff": abs(len(reload_segments) - len(baseline_segments)),
        "rel_diff": 0.0,
        "ok": len(reload_segments) == len(baseline_segments),
    })
    for row_idx, (baseline, reloaded) in enumerate(zip(baseline_summary, reload_summary)):
        for comp in compare_rows(baseline, reloaded):
            comp["key"] = f"tree_{row_idx}:{comp['key']}"
            out["comparisons"].append(comp)
    out["status"] = "pass" if out["comparisons"] and all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_grid_export_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "grid_export"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "target_terminal_count": 10,
            "root": {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
            "repair_connectivity": True,
            "validate_connectivity": True,
        },
        "growth": {"n_closest_vessels": 2, "n_points": 20, "ignore_collisions": True, "allow_inside_vessels": True},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": "topdown",
            "sample_mode": "grid",
            "tissue_grid": {"nx": 8, "ny": 8, "nz": 8, "chunk_points": 512},
            "tissue_gpu_validate_points": 0,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {
            "out_dir": str(case_dir / "cli"),
            "prefix": "grid",
            "write_paraview": True,
            "save_network": True,
            "export_float_dtype": "float32",
        },
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "grid_export",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI grid export failed.")
        write_logs(case_dir, cli=cli)
        return out
    cli_dir = case_dir / "cli"
    required = ["summary.csv", "segments.csv", "points.csv", "vessels.vtp", "oxygen_points.vtp", "manifest.json"]
    for name in required:
        exists = (cli_dir / name).exists()
        out["comparisons"].append({"key": f"exists:{name}", "legacy": True, "cli": exists, "abs_diff": 0, "rel_diff": 0, "ok": exists})
    manifest = json.loads((cli_dir / "manifest.json").read_text(encoding="utf-8"))
    tissue = manifest.get("resolved", {}).get("tissue_sample", {})
    point_rows = read_csv(cli_dir / "points.csv")
    segment_rows = read_csv(cli_dir / "segments.csv")
    out["comparisons"].append({
        "key": "sample_mode",
        "legacy": "grid",
        "cli": tissue.get("sample_mode"),
        "abs_diff": 0,
        "rel_diff": 0,
        "ok": tissue.get("sample_mode") == "grid",
    })
    out["comparisons"].append({
        "key": "grid_total_points",
        "legacy": 512,
        "cli": int(tissue.get("total_grid_points", -1)),
        "abs_diff": abs(int(tissue.get("total_grid_points", -1)) - 512),
        "rel_diff": 0.0,
        "ok": int(tissue.get("total_grid_points", -1)) == 512,
    })
    out["comparisons"].append({
        "key": "point_rows_positive",
        "legacy": True,
        "cli": len(point_rows) > 0,
        "abs_diff": 0,
        "rel_diff": 0,
        "ok": len(point_rows) > 0,
    })
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": 21,
        "cli": len(segment_rows),
        "abs_diff": abs(len(segment_rows) - 21),
        "rel_diff": 0.0,
        "ok": len(segment_rows) == 21,
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_simple_channel_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "simple_channel"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "box", "dimensions": [1.06, 0.811, 0.46], "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {
                "mode": "onechannel",
                "axis": "x",
                "radius_cm": 0.015,
                "z_from_bottom_cm": 0.243,
                "flow_ul_min": 24.0384615385,
                "concentration_inlet": 0.22471,
                "edge_extension_frac": 0.25,
            },
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "water",
            "build_fluid": "water",
            "qin_target_ul_min": 24.0384615385,
            "concentration_solver": "simple_channel",
            "distance_sample_count": 250,
            "tissue_gpu_validate_points": 0,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "simple", "write_paraview": True, "save_network": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "simple_channel",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI simple-channel run failed.")
        write_logs(case_dir, cli=cli)
        return out
    cli_dir = case_dir / "cli"
    for name in ("summary.csv", "segments.csv", "points.csv", "vessels.vtp", "oxygen_points.vtp", "simple.simple.npz"):
        exists = (cli_dir / name).exists()
        out["comparisons"].append({"key": f"exists:{name}", "legacy": True, "cli": exists, "abs_diff": 0, "rel_diff": 0, "ok": exists})
    segments = read_csv(cli_dir / "segments.csv")
    points = read_csv(cli_dir / "points.csv")
    summary = select_csv_row(cli_dir / "summary.csv", fluid="water")
    out["comparisons"].append({"key": "segment_rows", "legacy": 1, "cli": len(segments), "abs_diff": abs(len(segments) - 1), "rel_diff": 0.0, "ok": len(segments) == 1})
    out["comparisons"].append({"key": "points_positive", "legacy": True, "cli": len(points) > 0, "abs_diff": 0, "rel_diff": 0, "ok": len(points) > 0})
    total_flow = to_float(summary.get("total_flowrate"))
    out["comparisons"].append({
        "key": "total_flowrate_positive",
        "legacy": True,
        "cli": total_flow > 0.0,
        "abs_diff": 0,
        "rel_diff": 0,
        "ok": total_flow > 0.0,
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_scheduled_growth_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "scheduled_growth"
    source_dir = work_dir / "forest_smoke" / "cli"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "forest",
            "input_path": str(source_dir / "forest.forest"),
            "target_terminal_counts": [102, 102],
            "repair_connectivity": True,
            "validate_connectivity": True,
        },
        "growth": {
            "enabled": True,
            "assignment": "scheduled",
            "bulk_growth_mode": "never",
            "n_closest_vessels": 2,
            "n_points": 20,
            "ignore_collisions": True,
            "allow_inside_vessels": True,
            "checkpoint_path": str(case_dir / "cli" / "checkpoint.forest"),
            "checkpoint_every_adds": 2,
            "save_target_counts": [203, 204],
        },
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "flow_source": "total_split",
            "concentration_solver": "topdown",
            "distance_sample_count": 100,
            "tissue_gpu_validate_points": 0,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "scheduled", "write_paraview": True, "save_network": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "scheduled_growth",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI scheduled growth failed.")
        write_logs(case_dir, cli=cli)
        return out
    cli_dir = case_dir / "cli"
    required = [
        "summary.csv",
        "segments.csv",
        "scheduled.forest",
        "scheduled.forest.simcache",
        "checkpoint.forest",
        "checkpoint.forest.json",
        "scheduled_t203.forest",
        "scheduled_t204.forest",
    ]
    for name in required:
        exists = (cli_dir / name).exists()
        out["comparisons"].append({"key": f"exists:{name}", "legacy": True, "cli": exists, "abs_diff": 0, "rel_diff": 0, "ok": exists})
    segment_rows = read_csv(cli_dir / "segments.csv")
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": 410,
        "cli": len(segment_rows),
        "abs_diff": abs(len(segment_rows) - 410),
        "rel_diff": 0.0,
        "ok": len(segment_rows) == 410,
    })
    manifest = json.loads((cli_dir / "manifest.json").read_text(encoding="utf-8"))
    segments = manifest.get("network", {}).get("segments", [])
    out["comparisons"].append({
        "key": "manifest_segments",
        "legacy": [205, 205],
        "cli": segments,
        "abs_diff": 0,
        "rel_diff": 0,
        "ok": segments == [205, 205],
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def run_nearest_tree_growth_case(work_dir: Path, *, timeout: int) -> dict[str, Any]:
    case = "nearest_tree_growth"
    case_dir = work_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "forest",
            "target_terminal_counts": [4, 4],
            "roots": [
                {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
                {"start": [-0.49, 0.49, 0.49], "direction": [0.49, -0.49, -0.49]},
            ],
            "repair_connectivity": True,
            "validate_connectivity": True,
        },
        "growth": {
            "enabled": True,
            "assignment": "nearest-tree",
            "bulk_growth_mode": "never",
            "n_closest_vessels": 2,
            "n_points": 8,
            "nearest_tree_batch_points": 32,
            "ignore_collisions": False,
            "n_ignore_collisions": -1,
            "collision_retry_limit": 20,
            "collision_failure_mode": "error",
            "allow_inside_vessels": True,
        },
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "flow_source": "total_split",
            "concentration_solver": "topdown",
            "distance_sample_count": 100,
            "tissue_gpu_validate_points": 0,
            "tissuesim": {"finite_radius_o2_terms": "none", "lumen_wall_closure": "wellmixed"},
        },
        "outputs": {"out_dir": str(case_dir / "cli"), "prefix": "nearest", "write_paraview": True, "save_network": True},
    }
    settings_path = case_dir / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    cli = run_cmd([str(PYTHON), "-m", "cascade.cli", "run", "--settings", str(settings_path)], timeout=timeout)
    out = {
        "case": case,
        "kind": "nearest_tree_growth",
        "cli_elapsed_s": cli["elapsed_s"],
        "cli_returncode": cli["returncode"],
        "status": "fail",
        "comparisons": [],
        "notes": [],
    }
    if cli["returncode"] != 0:
        out["notes"].append("CLI nearest-tree growth failed.")
        write_logs(case_dir, cli=cli)
        return out
    cli_dir = case_dir / "cli"
    segment_rows = read_csv(cli_dir / "segments.csv")
    manifest = json.loads((cli_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("summary.csv", "segments.csv", "points.csv", "vessels.vtp", "oxygen_points.vtp", "nearest.forest"):
        exists = (cli_dir / name).exists()
        out["comparisons"].append({"key": f"exists:{name}", "legacy": True, "cli": exists, "abs_diff": 0, "rel_diff": 0, "ok": exists})
    out["comparisons"].append({
        "key": "segment_rows",
        "legacy": 18,
        "cli": len(segment_rows),
        "abs_diff": abs(len(segment_rows) - 18),
        "rel_diff": 0.0,
        "ok": len(segment_rows) == 18,
    })
    reports = manifest.get("network", {}).get("connectivity_reports", [])
    reports_clean = bool(reports) and all(
        int(report.get("bad_parent", 0)) == 0
        and int(report.get("bad_child", 0)) == 0
        and int(report.get("bad_geom", 0)) == 0
        and int(report.get("reachable", 0)) == int(report.get("segments", -1))
        for report in reports
    )
    out["comparisons"].append({
        "key": "connectivity_reports_clean",
        "legacy": True,
        "cli": reports_clean,
        "abs_diff": 0,
        "rel_diff": 0,
        "ok": reports_clean,
    })
    out["status"] = "pass" if all(c["ok"] for c in out["comparisons"]) else "fail"
    write_logs(case_dir, cli=cli)
    return out


def tree_settings(target: int, sample_count: int, out_dir: Path, *, solver: str) -> dict[str, Any]:
    return {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "target_terminal_count": int(target),
            "root": {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
        },
        "growth": {"n_closest_vessels": 2, "n_points": 50, "ignore_collisions": True, "allow_inside_vessels": True, "n_equal_bifurcations": None},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": solver,
            "distance_sample_count": int(sample_count),
            "tissue_gpu_validate_points": 0,
            "tissuesim": {
                "finite_radius_o2_terms": "none" if solver == "topdown" else "both",
                "lumen_wall_closure": "wellmixed" if solver == "topdown" else "graetz",
            },
        },
        "outputs": {"out_dir": str(out_dir), "prefix": f"tree_t{target}", "write_paraview": True, "save_network": True},
    }


def run_cmd(cmd: list[str], *, timeout: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(CASCADE_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "elapsed_s": time.perf_counter() - start}


def write_logs(case_dir: Path, **logs: dict[str, Any]) -> None:
    for name, data in logs.items():
        if not data:
            continue
        (case_dir / f"{name}.log").write_text(
            "$ " + " ".join(data.get("cmd", [])) + "\n\n" + data.get("stdout", ""),
            encoding="utf-8",
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_csv_row(path: Path, *, fluid: str) -> dict[str, str]:
    rows = read_csv(path)
    matching = [row for row in rows if str(row.get("fluid", "")).lower() == fluid.lower()]
    if matching:
        return matching[0]
    if len(rows) == 1:
        return rows[0]
    raise RuntimeError(f"Could not select fluid={fluid!r} from {path}")


def compare_rows(legacy: dict[str, str], cli: dict[str, str]) -> list[dict[str, Any]]:
    comparisons = []
    for key in COMPARE_KEYS:
        lv = to_float(legacy.get(key))
        cv = to_float(cli.get(key))
        if math.isnan(lv) and math.isnan(cv):
            ok = True
            abs_diff = 0.0
            rel_diff = 0.0
        else:
            abs_diff = abs(cv - lv)
            denom = max(abs(lv), 1.0)
            rel_diff = abs_diff / denom
            strict_keys = {"total_segments_mean", "terminal_segments_mean"}
            tol = 0.0 if key in strict_keys else 1.0e-10
            ok = abs_diff <= tol or rel_diff <= tol
        comparisons.append({
            "key": key,
            "legacy": lv,
            "cli": cv,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "ok": ok,
        })
    return comparisons


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def build_report(results: list[dict[str, Any]]) -> str:
    lines = ["# CASCADE Refactor Audit", ""]
    for result in results:
        status = result.get("status", "unknown").upper()
        lines.append(f"## {result.get('case')} - {status}")
        elapsed_bits = []
        if result.get("legacy_elapsed_s") is not None:
            elapsed_bits.append(f"legacy={result['legacy_elapsed_s']:.3f}s")
        if result.get("cli_elapsed_s") is not None:
            elapsed_bits.append(f"cli={result['cli_elapsed_s']:.3f}s")
        if elapsed_bits:
            lines.append("")
            lines.append("Runtime: " + ", ".join(elapsed_bits))
        for note in result.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")
        lines.append("| key | legacy/expected | cli | abs diff | rel diff | ok |")
        lines.append("| --- | ---: | ---: | ---: | ---: | :--: |")
        for comp in result.get("comparisons", []):
            lines.append(
                f"| {comp['key']} | {fmt(comp['legacy'])} | {fmt(comp['cli'])} | "
                f"{fmt(comp['abs_diff'])} | {fmt(comp['rel_diff'])} | {'yes' if comp['ok'] else 'NO'} |"
            )
        lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
