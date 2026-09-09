from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import statistics
import sys
from time import perf_counter
from typing import Any

import numpy as np

CASCADE_ROOT = Path(__file__).resolve().parents[1]
if str(CASCADE_ROOT) not in sys.path:
    sys.path.insert(0, str(CASCADE_ROOT))

import TissueSim_cube_local as ts
from cascade.config import parse_config
from cascade.growth import apply_tissuesim_settings, build_domain, terminal_flow_for_target
from cascade.simulation import run_simulation
from cascade.svv_adapter import Tree


DEFAULT_LEGACY_TREE = Path(
    "/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/"
    "svv/SCRIPTS/trees_cache_heart/tree_ebf58642624d04b2ffebd2f0b12a3739_t10000.tree.npz"
)
DEFAULT_LOCAL_TREE = CASCADE_ROOT / ".cascade_benchmark_cache" / "legacy_cube_t10000.tree.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare direct TissueSim simulation speed against CASCADE wrapper speed.")
    parser.add_argument("--tree", default=str(DEFAULT_LOCAL_TREE), help="Tree .npz path used by both benchmarks.")
    parser.add_argument("--target", type=int, default=10000, help="Target terminal count represented by the tree.")
    parser.add_argument("--sample-count", type=int, default=10000, help="Shared tissue sample count.")
    parser.add_argument("--solver", default="topdown", help="Concentration solver to benchmark.")
    parser.add_argument("--fluid", default="blood", choices=("blood", "water"), help="Analysis fluid.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per implementation.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per implementation.")
    parser.add_argument("--output-json", default=None, help="Optional JSON result path.")
    args = parser.parse_args()

    tree_path = _ensure_tree(Path(args.tree).expanduser())
    result = run_benchmark(
        tree_path=tree_path,
        target=int(args.target),
        sample_count=int(args.sample_count),
        solver=str(args.solver),
        fluid=str(args.fluid),
        runs=int(args.runs),
        warmup=int(args.warmup),
    )
    print(json.dumps(result, indent=2, allow_nan=True))
    if args.output_json:
        out = Path(args.output_json).expanduser()
        if not out.is_absolute():
            out = CASCADE_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    return 0


def run_benchmark(
    *,
    tree_path: Path,
    target: int,
    sample_count: int,
    solver: str,
    fluid: str,
    runs: int,
    warmup: int,
) -> dict[str, Any]:
    config = _make_config(tree_path=tree_path, target=target, sample_count=sample_count, solver=solver, fluid=fluid)
    apply_tissuesim_settings(ts, config)
    domain = build_domain(config, ts=ts)
    sample_points = np.asarray(ts.sample_domain_points(domain, sample_count), dtype=float)

    legacy_times: list[float] = []
    cascade_times: list[float] = []
    legacy_summaries: list[dict[str, Any]] = []
    cascade_rows: list[dict[str, Any]] = []

    total = max(int(warmup), 0) + max(int(runs), 1)
    for idx in range(total):
        measured = idx >= int(warmup)

        legacy_s, legacy_summary = _time_legacy(config, domain, tree_path, target, sample_points)
        cascade_s, cascade_row = _time_cascade(config, domain, tree_path, target, sample_points)

        if measured:
            legacy_times.append(float(legacy_s))
            cascade_times.append(float(cascade_s))
            legacy_summaries.append(legacy_summary)
            cascade_rows.append(cascade_row)

    metrics = _metric_diffs(legacy_summaries[-1], cascade_rows[-1]) if legacy_summaries and cascade_rows else {}
    legacy_median = statistics.median(legacy_times)
    cascade_median = statistics.median(cascade_times)
    return {
        "tree_path": str(tree_path),
        "target": int(target),
        "sample_count": int(sample_count),
        "solver": str(solver),
        "fluid": str(fluid),
        "runs": int(runs),
        "warmup": int(warmup),
        "legacy_times_s": legacy_times,
        "cascade_times_s": cascade_times,
        "legacy_median_s": legacy_median,
        "cascade_median_s": cascade_median,
        "cascade_over_legacy_ratio": cascade_median / legacy_median if legacy_median > 0.0 else float("nan"),
        "metric_diffs": metrics,
    }


def _make_config(*, tree_path: Path, target: int, sample_count: int, solver: str, fluid: str):
    raw = {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {"mode": "tree", "input_path": str(tree_path), "target_terminal_count": int(target)},
        "growth": {"enabled": False, "n_equal_bifurcations": 200000},
        "simulation": {
            "fluid": str(fluid).lower(),
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": str(solver).lower(),
            "distance_sample_count": int(sample_count),
            "sample_mode": "random",
            "tissue_accel": "gpu",
            "tissue_gpu_validate_points": 0,
            "nearest_tissue_vessels": 250,
            "window_factor": 6,
            "solver_timing_details": True,
            "cext": {
                "accel_mode": "gpu",
                "frozen_accel_mode": "gpu",
                "vess_coupling_accel": "anderson",
                "hybrid_bg_mode": "fft",
                "hybrid_bg_solver": "fft",
            },
        },
        "outputs": {
            "out_dir": ".cascade_benchmark_cache/out",
            "write_paraview": False,
            "write_summary_csv": True,
            "write_segments_csv": False,
            "write_points_csv": False,
            "save_network": False,
        },
    }
    config = parse_config(deepcopy(raw))
    config.settings_path = CASCADE_ROOT / ".cascade_benchmark_cache" / "benchmark_settings.json"
    return config


def _time_legacy(config, domain, tree_path: Path, target: int, sample_points: np.ndarray) -> tuple[float, dict[str, Any]]:
    apply_tissuesim_settings(ts, config)
    tree = _load_tree(config, domain, tree_path, target)
    stream = io.StringIO()
    t0 = perf_counter()
    with redirect_stdout(stream):
        tissue_cache = ts.build_tissue_cache_from_tree(tree, sample_points)
        summary, _details = ts.summarize_tree(
            tree,
            sample_points,
            int(target),
            side_length=float(config.domain.side_length),
            fluid=config.simulation.fluid,
            inlet_flow_cm3_s=_qin_cm3_s(config),
            tissue_cache=tissue_cache,
            concentration_solver=config.simulation.concentration_solver,
            return_details=True,
        )
    return perf_counter() - t0, dict(summary)


def _time_cascade(config, domain, tree_path: Path, target: int, sample_points: np.ndarray) -> tuple[float, dict[str, Any]]:
    apply_tissuesim_settings(ts, config)
    tree = _load_tree(config, domain, tree_path, target)
    stream = io.StringIO()
    t0 = perf_counter()
    with redirect_stdout(stream):
        result = run_simulation(domain, [tree], [int(target)], config, sample_points=sample_points)
    return perf_counter() - t0, dict(result.summary_rows[0])


def _load_tree(config, domain, tree_path: Path, target: int):
    tree = Tree.load(str(tree_path), domain=domain, data_dtype=np.float64, index_dtype=np.int64)
    if hasattr(ts, "_prepare_loaded_tree"):
        ts._prepare_loaded_tree(tree)
    if hasattr(ts, "_sync_loaded_tree_params"):
        ts._sync_loaded_tree_params(
            tree,
            side_length=float(config.domain.side_length),
            terminal_flow_override=terminal_flow_for_target(config, _qin_cm3_s(config), int(target)),
        )
    return tree


def _qin_cm3_s(config) -> float:
    return float(config.simulation.qin_target_ul_min) * 1.0e-3 / 60.0 * (float(config.domain.side_length) ** 3)


def _metric_diffs(legacy: dict[str, Any], cascade: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_segments",
        "terminal_segments",
        "dRnet",
        "C_LQ_over_Cmax",
        "C_tiss_over_Cmax",
        "cext_concentration_mean",
    )
    out = {}
    for key in keys:
        if key not in legacy or key not in cascade:
            continue
        a = legacy.get(key)
        b = cascade.get(key)
        try:
            diff = abs(float(a) - float(b))
        except Exception:
            diff = None
        out[key] = {"legacy": a, "cascade": b, "abs_diff": diff}
    return out


def _ensure_tree(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        return path
    if path == DEFAULT_LOCAL_TREE.resolve() and DEFAULT_LEGACY_TREE.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DEFAULT_LEGACY_TREE, path)
        return path
    raise FileNotFoundError(f"Tree file not found: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
