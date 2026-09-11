"""Parse CASCADE command-line arguments and dispatch runs, diagnostics, and self-tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from cascade import __version__
from cascade.configuration.schema import example_config, load_config
from cascade.utils.execution import guard_simulation


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="cascade", description="CASCADE vascular simulation and export CLI."
    )
    parser.add_argument("--version", action="version", version=f"CASCADE {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="Run a JSON-configured tree or forest simulation/export."
    )
    run_parser.add_argument("--settings", required=True, help="Path to settings JSON.")

    batch_parser = sub.add_parser(
        "batch",
        help="Run multiple settings files serially with warm reusable state.",
    )
    batch_parser.add_argument(
        "--settings",
        required=True,
        nargs="+",
        help="Settings JSON files, in execution order.",
    )
    batch_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later cases when one settings file fails.",
    )

    sweep_parser = sub.add_parser(
        "sweep", help="Run a JSON-configured target/fluid sweep into one summary CSV."
    )
    sweep_parser.add_argument(
        "--settings", required=True, help="Path to sweep settings JSON."
    )

    sub.add_parser(
        "worker",
        help="Run the persistent local worker used by CASCADE Studio.",
    )

    init_parser = sub.add_parser(
        "init-settings", help="Write an example settings JSON."
    )
    init_parser.add_argument("path", help="Output settings path.")
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing file."
    )

    doctor_parser = sub.add_parser(
        "doctor", help="Inspect dependencies and run a compiled GPU preflight."
    )
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--no-gpu-probe", action="store_true")
    doctor_parser.add_argument("--require-gpu", action="store_true")

    self_test_parser = sub.add_parser(
        "self-test",
        help="Run a bounded installed-package solver and export check.",
    )
    self_test_parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exercise the CUDA Cext and tissue paths; fail if they are unavailable.",
    )
    self_test_parser.add_argument(
        "--output-dir", help="Directory in which to retain the test run."
    )
    self_test_parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep an automatically created test directory.",
    )
    self_test_parser.add_argument(
        "--json", action="store_true", help="Print the final report as JSON."
    )

    args = parser.parse_args(argv)
    if args.command == "init-settings":
        return _init_settings(args)
    if args.command == "run":
        return _run(args)
    if args.command == "batch":
        return _batch(args)
    if args.command == "sweep":
        return _sweep(args)
    if args.command == "worker":
        from cascade.commands.worker import serve

        return serve()
    if args.command == "doctor":
        from cascade.diagnostics.environment import main as doctor_main

        doctor_args = []
        if args.json:
            doctor_args.append("--json")
        if args.no_gpu_probe:
            doctor_args.append("--no-gpu-probe")
        if args.require_gpu:
            doctor_args.append("--require-gpu")
        return doctor_main(doctor_args)
    if args.command == "self-test":
        from cascade.diagnostics.self_test import main as self_test_main

        self_test_args = []
        if args.require_gpu:
            self_test_args.append("--require-gpu")
        if args.output_dir:
            self_test_args.extend(["--output-dir", args.output_dir])
        if args.keep:
            self_test_args.append("--keep")
        if args.json:
            self_test_args.append("--json")
        return self_test_main(self_test_args)
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


@guard_simulation("cascade run")
def _run(args: argparse.Namespace) -> int:
    execute_configured_run(args.settings, cleanup_policy="hard")
    return 0


@guard_simulation("cascade batch")
def _batch(args: argparse.Namespace) -> int:
    """Run arbitrary serial cases without repaying import/build startup costs."""
    from cascade.commands.workspace import InteractiveRunWorkspace
    from cascade.configuration.bridge import load_runtime_module
    from cascade.utils.execution import release_completed_case_memory

    workspace = InteractiveRunWorkspace()
    failures = 0
    total_start = perf_counter()
    try:
        for index, settings in enumerate(args.settings, start=1):
            print(f"\nBatch case {index}/{len(args.settings)}: {settings}", flush=True)
            try:
                execute_configured_run(
                    settings, cleanup_policy="adaptive", workspace=workspace
                )
            except Exception as exc:
                failures += 1
                if not args.continue_on_error:
                    raise
                print(f"Case failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        workspace.clear()
        release_completed_case_memory(
            load_runtime_module(), trim_accelerator_pools=True
        )
    print(
        f"Batch finished: cases={len(args.settings)} failures={failures} "
        f"elapsed={perf_counter() - total_start:.3f}s",
        flush=True,
    )
    return 1 if failures else 0


def execute_configured_run(
    settings_path: str | Path,
    *,
    cleanup_policy: str = "adaptive",
    workspace=None,
) -> tuple[dict[str, str], float]:
    """Execute and export one case inside either a CLI or persistent worker.

    Requested outputs are fully written before case-owned arrays are released.
    ``hard`` returns all unused accelerator blocks to the driver; ``adaptive``
    keeps safe reusable capacity warm for the next serial case.
    """
    if cleanup_policy not in {"hard", "adaptive"}:
        raise ValueError("cleanup_policy must be 'hard' or 'adaptive'")
    from cascade.accelerators.backend import require_gpu_runtime
    from cascade.configuration.bridge import load_runtime_module
    from cascade.exporting.run import export_run
    from cascade.simulation.engine import run_simulation
    from cascade.utils.execution import release_completed_case_memory
    from cascade.vessels.build import build_or_load_network
    t0 = perf_counter()
    config = None
    build = None
    simulation = None
    outputs: dict[str, str] = {}
    try:
        config = load_config(settings_path)
        print(f"Loaded settings: {config.settings_path}", flush=True)
        gpu = require_gpu_runtime(config)
        if gpu is not None:
            print(f"GPU preflight passed: {gpu.summary}", flush=True)
        if workspace is None:
            build = build_or_load_network(config)
            build_reused = False
        else:
            build, build_reused = workspace.resolve_build(config)
        print(
            f"Network ready: mode={config.network_mode} trees={len(build.trees)} "
            f"segments={[int(getattr(t, 'segment_count', 0) or 0) for t in build.trees]} "
            f"reused={str(build_reused).lower()}",
            flush=True,
        )
        tissue_caches = (
            workspace.resolve_tissue_caches(config) if workspace is not None else None
        )
        simulation_key = None
        if workspace is not None:
            simulation, simulation_key = workspace.resolve_simulation(config)
        if simulation is None:
            simulation = run_simulation(
                build.domain,
                build.trees,
                build.target_counts,
                config,
                sample_points=build.sample_points,
                tissue_caches=tissue_caches,
                reuse_geometry=workspace is not None,
                materialize_segment_rows=False,
            )
            if workspace is not None and simulation_key is not None:
                workspace.store_simulation(config, simulation_key, simulation)
        outputs = export_run(config, build, simulation)
        elapsed = perf_counter() - t0
        print(f"Done in {elapsed:.3f}s")
        print("Wrote:")
        for key in sorted(outputs):
            print(f"  {key}: {outputs[key]}")
        return outputs, elapsed
    finally:
        # Export completes before this point.  Drop the large object graph first
        # so the cleanup routine can actually return unused host/device memory.
        simulation = None
        build = None
        config = None
        try:
            cleanup = release_completed_case_memory(
                load_runtime_module(),
                trim_accelerator_pools=True if cleanup_policy == "hard" else None,
            )
            if workspace is not None:
                eviction = workspace.enforce_memory_budget(cleanup)
                if eviction:
                    print(
                        f"Interactive memory pressure: evicted {eviction}",
                        flush=True,
                    )
                    release_completed_case_memory(
                        load_runtime_module(), trim_accelerator_pools=True
                    )
        except Exception:
            # Cleanup must never replace the original simulation/export error.
            pass


@guard_simulation("cascade sweep")
def _sweep(args: argparse.Namespace) -> int:
    from cascade.simulation.sweep import run_sweep

    t0 = perf_counter()
    outputs = run_sweep(args.settings)
    elapsed = perf_counter() - t0
    print(f"Done in {elapsed:.3f}s")
    print("Wrote:")
    for key in sorted(outputs):
        print(f"  {key}: {outputs[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
