"""Implement the ``cascade`` command-line interface.

Users normally call the installed ``cascade`` command to create settings, run
or batch simulations, inspect and prepare large inputs, and diagnose an
installation. CASCADE Studio also starts the internal ``worker`` subcommand so
serial GUI jobs can reuse bounded scientific state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from cascade import __version__
from cascade.configuration.schema import example_config, load_config
from cascade.utils.execution import guard_simulation


def main(argv: list[str] | None = None) -> int:
    command_args = list(sys.argv[1:] if argv is None else argv)
    debug = "--debug" in command_args
    command_args = [value for value in command_args if value != "--debug"]
    try:
        return _main(command_args)
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        if debug:
            raise
        print(f"CASCADE error: {exc}", file=sys.stderr, flush=True)
        return 2


def _main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="cascade", description="CASCADE vascular simulation and export CLI."
    )
    parser.add_argument("--version", action="version", version=f"CASCADE {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show Python tracebacks for command failures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="Run a JSON-configured tree or forest simulation/export."
    )
    run_parser.add_argument("--settings", required=True, help="Path to settings JSON.")

    inspect_parser = sub.add_parser(
        "inspect",
        help="Estimate scale and preparation cost without loading the simulation.",
    )
    inspect_parser.add_argument(
        "--settings", required=True, help="Path to settings JSON."
    )

    prepare_parser = sub.add_parser(
        "prepare",
        help="Build persistent memory-mapped caches for large saved inputs.",
    )
    prepare_parser.add_argument(
        "--settings", required=True, help="Path to settings JSON."
    )
    prepare_parser.add_argument(
        "--cache-dir", help="Optional prepared-cache directory override."
    )
    prepare_parser.add_argument("--json", action="store_true")

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
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "prepare":
        return _prepare(args)
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


def _inspect(args: argparse.Namespace) -> int:
    """Print the shared CLI/Studio metadata-only resource plan."""
    from cascade.runtime.planning import decide_implicit_preparation

    config = load_config(args.settings)
    decision = decide_implicit_preparation(config)
    print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
    return 0


@guard_simulation("cascade prepare")
def _prepare(args: argparse.Namespace) -> int:
    """Build explicit persistent caches without running a simulation."""
    from cascade.configuration.runtime import RuntimeConfiguration
    from cascade.utils.resources import resolve_path
    from cascade.vessels.prepared import prepare_tree_archive

    config = load_config(args.settings)
    source = resolve_path(
        config.network.input_path,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if source is None:
        raise ValueError("cascade prepare requires network.input_path.")
    numerics = RuntimeConfiguration.from_run_config(config).sections.get(
        "numerics", {}
    )
    prepared = prepare_tree_archive(
        source,
        data_dtype=numerics.get("tree_data_dtype", "float64"),
        index_dtype=numerics.get("tree_index_dtype", "int64"),
        cache_root=args.cache_dir,
    )
    payload = prepared.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        action = "Reused" if prepared.reused else "Prepared"
        print(f"{action} memory-mapped tree cache: {prepared.root}")
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
                print(
                    f"Batch case {index} failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
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
    from cascade.configuration.bridge import load_runtime_module
    from cascade.exporting.run import export_run
    from cascade.utils.execution import release_completed_case_memory

    t0 = perf_counter()
    config = None
    build = None
    simulation = None
    outputs: dict[str, str] = {}
    try:
        config, build, simulation = _resolve_configured_simulation(
            settings_path, workspace=workspace, allow_detailed_cache=False
        )
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


def prepare_configured_run(
    settings_path: str | Path, *, workspace
) -> dict[str, object]:
    """Prepare one bounded immutable Studio case without writing its outputs.

    The persistent worker calls this as soon as a case is queued.  Geometry,
    spatial indexes, imports, and (for interactive-sized cases) the exact
    numerical result are then ready before the user presses Run.  Detailed
    results are retained only when the workspace's byte cap allows it.
    """
    from cascade.configuration.bridge import load_runtime_module
    from cascade.exporting.run import warm_export_metadata
    from cascade.utils.execution import release_completed_case_memory

    config = None
    build = None
    simulation = None
    started = perf_counter()
    try:
        from cascade.runtime.planning import decide_implicit_preparation

        candidate = load_config(settings_path)
        decision = decide_implicit_preparation(candidate)
        if not decision.allowed:
            warm_export_metadata()
            print(
                f"Interactive preparation skipped: {decision.reason}",
                flush=True,
            )
            return {
                "elapsed_s": float(perf_counter() - started),
                "cache": workspace.cache_state(),
                "skipped": True,
                "plan": decision.to_dict(),
            }
        config, build, simulation = _resolve_configured_simulation(
            settings_path,
            workspace=workspace,
            allow_detailed_cache=True,
            presolve_if_bounded=True,
        )
        warm_export_metadata()
        return {
            "elapsed_s": float(perf_counter() - started),
            "cache": workspace.cache_state(),
            "plan": decision.to_dict(),
        }
    finally:
        del simulation, build, config
        try:
            cleanup = release_completed_case_memory(
                load_runtime_module(), trim_accelerator_pools=None
            )
            eviction = workspace.enforce_memory_budget(cleanup)
        except Exception:
            # Preparation is opportunistic.  Cleanup diagnostics must never
            # turn a usable prepared case into a failed worker request.
            eviction = None
        if eviction:
            release_completed_case_memory(
                load_runtime_module(), trim_accelerator_pools=True
            )


def _resolve_configured_simulation(
    settings_path: str | Path,
    *,
    workspace=None,
    allow_detailed_cache: bool,
    presolve_if_bounded: bool = False,
):
    """Load, build, and optionally solve one case using reusable state."""
    from cascade.accelerators.backend import require_gpu_runtime
    from cascade.simulation.engine import run_simulation
    from cascade.vessels.build import build_or_load_network

    config = load_config(settings_path)
    print(f"Loaded settings: {config.settings_path}", flush=True)
    from cascade.utils.resources import resolve_path
    out_dir = resolve_path(
        config.outputs.out_dir,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if (
        out_dir is not None
        and (out_dir / "manifest.json").exists()
        and not config.outputs.overwrite
    ):
        raise FileExistsError(
            f"{out_dir} already contains a CASCADE run; choose a new outputs.out_dir "
            "or set outputs.overwrite=true."
        )
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
    simulation = None
    simulation_key = None
    if workspace is not None:
        simulation, simulation_key = workspace.resolve_simulation(config)
    should_solve = not presolve_if_bounded or (
        workspace is not None and workspace.should_presolve(config)
    )
    if simulation is None and should_solve:
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
            workspace.store_simulation(
                config,
                simulation_key,
                simulation,
                allow_detailed=allow_detailed_cache,
            )
    return config, build, simulation


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
