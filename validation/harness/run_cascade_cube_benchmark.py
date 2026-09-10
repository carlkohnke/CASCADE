"""Run one installed-wheel CASCADE cube benchmark without array-evidence writes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
from time import perf_counter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    settings_path = args.settings.expanduser().resolve()
    metadata_path = args.metadata.expanduser().resolve()
    if metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite {metadata_path}")

    import cascade
    from cascade.config import load_config
    from cascade.execution import single_simulation
    from cascade.export import export_run
    from cascade.gpu import require_gpu_runtime
    from cascade.growth import build_or_load_network
    from cascade.resources import resolve_path
    from cascade.simulation import run_simulation

    config = load_config(settings_path)
    out_dir = resolve_path(
        config.outputs.out_dir,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if out_dir is None:
        raise ValueError("settings outputs.out_dir did not resolve")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite CASCADE run directory: {out_dir}")

    started = perf_counter()
    with single_simulation("M3 installed-wheel cube benchmark"):
        gpu = require_gpu_runtime(config)
        build_started = perf_counter()
        build = build_or_load_network(config)
        build_wall_s = perf_counter() - build_started
        solve_started = perf_counter()
        simulation = run_simulation(
            build.domain,
            build.trees,
            build.target_counts,
            config,
            sample_points=build.sample_points,
        )
        solve_wall_s = perf_counter() - solve_started
        export_started = perf_counter()
        outputs = export_run(config, build, simulation)
        export_wall_s = perf_counter() - export_started
    elapsed_s = perf_counter() - started

    summaries = [dict(result.summary) for result in simulation.tree_results]
    component_timings = [
        {key: value for key, value in summary.items() if str(key).startswith("t_")}
        for summary in summaries
    ]
    distribution = importlib.metadata.distribution("cascade-vascular")
    direct_url_path = Path(distribution._path) / "direct_url.json"
    direct_url = None
    if direct_url_path.is_file():
        direct_url = json.loads(direct_url_path.read_text(encoding="utf-8"))
    record = {
        "settings_path": str(settings_path),
        "cascade_version": cascade.__version__,
        "cascade_module": str(Path(cascade.__file__).resolve()),
        "cascade_distribution": str(Path(distribution._path).resolve()),
        "cascade_direct_url": direct_url,
        "gpu": None if gpu is None else gpu.summary,
        "elapsed_process_s": elapsed_s,
        "build_wall_s": build_wall_s,
        "solve_wall_s": solve_wall_s,
        "export_wall_s": export_wall_s,
        "build_timings": build.build_timings,
        "simulation_timings": simulation.timings,
        "component_timings": component_timings,
        "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "outputs": outputs,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(record, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
