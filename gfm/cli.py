from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from .config import example_config, load_config
from .export import export_run
from .gpu import require_gpu_runtime
from .growth import build_or_load_network
from .simulation import run_simulation
from .sweep import run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gfm.cli", description="GFM public-svv simulation/export CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a JSON-configured tree or forest simulation/export.")
    run_parser.add_argument("--settings", required=True, help="Path to settings JSON.")

    sweep_parser = sub.add_parser("sweep", help="Run a JSON-configured target/fluid sweep into one summary CSV.")
    sweep_parser.add_argument("--settings", required=True, help="Path to sweep settings JSON.")

    init_parser = sub.add_parser("init-settings", help="Write an example settings JSON.")
    init_parser.add_argument("path", help="Output settings path.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing file.")

    args = parser.parse_args(argv)
    if args.command == "init-settings":
        return _init_settings(args)
    if args.command == "run":
        return _run(args)
    if args.command == "sweep":
        return _sweep(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _init_settings(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser()
    if path.exists() and not bool(args.force):
        raise FileExistsError(f"{path} already exists; pass --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(example_config(), indent=2), encoding="utf-8")
    print(f"Wrote example settings: {path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    t0 = perf_counter()
    config = load_config(args.settings)
    print(f"Loaded settings: {config.settings_path}", flush=True)
    gpu = require_gpu_runtime(config)
    if gpu is not None:
        print(f"GPU preflight passed: {gpu.summary}", flush=True)
    build = build_or_load_network(config)
    print(
        f"Network ready: mode={config.network_mode} trees={len(build.trees)} "
        f"segments={[int(getattr(t, 'segment_count', 0) or 0) for t in build.trees]}",
        flush=True,
    )
    simulation = run_simulation(
        build.domain,
        build.trees,
        build.target_counts,
        config,
        sample_points=build.sample_points,
    )
    outputs = export_run(config, build, simulation)
    elapsed = perf_counter() - t0
    print(f"Done in {elapsed:.3f}s")
    print("Wrote:")
    for key in sorted(outputs):
        print(f"  {key}: {outputs[key]}")
    return 0


def _sweep(args: argparse.Namespace) -> int:
    t0 = perf_counter()
    outputs = run_sweep(args.settings)
    elapsed = perf_counter() - t0
    print(f"Done in {elapsed:.3f}s")
    print("Wrote:")
    for key in sorted(outputs):
        print(f"  {key}: {outputs[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
