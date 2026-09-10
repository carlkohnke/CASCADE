"""Bounded installed-package verification for CASCADE users and release CI."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np

from . import __version__
from .config import example_config, parse_config
from .gpu import gpu_requested


def _run_case(root: Path, *, require_gpu: bool) -> dict[str, Any]:
    # Validate the generated starter contract independently, then use explicit
    # geometry for the executable smoke so this check measures simulation and
    # export rather than stochastic CCO growth.
    starter = parse_config(example_config())
    if gpu_requested(starter):
        raise RuntimeError("The generated starter configuration unexpectedly requires a GPU.")

    # Create a deterministic one-segment input through CASCADE's public-svv
    # adapter. This avoids stochastic growth while still exercising the same
    # loaded-tree solver path used by production validation cases.
    from .runtime import tissuesim as ts

    domain = ts.build_domain(1.0, random_seed=42)
    tree = ts.Tree(data_dtype=np.float64, index_dtype=np.int64, preallocation_step=4)
    tree.set_domain(domain)
    tree.parameters.root_pressure = ts.ROOT_PRESSURE
    tree.parameters.terminal_pressure = ts.TERMINAL_PRESSURE
    tree.parameters.terminal_flow = 900.0 * 0.00001666666666
    tree.set_root(
        np.asarray([0.49, -0.49, -0.49]),
        np.asarray([-0.49, 0.49, 0.49]),
    )
    tree_path = root / "self_test.tree.npz"
    tree.save(str(tree_path), include_domain=False)

    settings: dict[str, Any] = {
        "schema_version": 1,
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "tree",
            "input_path": str(tree_path),
            "target_terminal_count": 1,
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "blood",
            "build_fluid": "blood",
            "qin_target_ul_min": 900.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 64,
            "tissue_accel": "cpu",
        },
        "settings": {
            "oxygen": {"gl_order": 5, "gl_order_cext": 1},
            "cext": {"accel_mode": "cpu", "window_factor": 6.0},
            "tissue": {"accel_mode": "cpu"},
        },
        "outputs": {
            "out_dir": str(root / "run"),
            "prefix": "self_test",
            "write_paraview": True,
            "save_network": True,
        },
    }

    backend = "gpu" if require_gpu else "cpu"
    if require_gpu:
        settings["simulation"]["concentration_solver"] = "network_ext"
        settings["simulation"]["tissue_accel"] = "gpu"
        settings["settings"]["oxygen"].update(
            {
                "gl_order": 5,
                "gl_order_cext": 1,
                "finite_radius_o2_terms": "none",
                "lumen_wall_closure": "wellmixed",
            }
        )
        settings["settings"]["cext"].update(
            {
                "accel_mode": "gpu",
                "vess_coupling_max_iter": 1,
                "window_factor": 6.0,
            }
        )
        settings["settings"]["tissue"]["accel_mode"] = "gpu"

    parsed = parse_config(settings)
    if gpu_requested(parsed) is not require_gpu:
        raise RuntimeError(f"Self-test settings did not select the expected {backend} backend.")

    settings_path = root / "settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    command = [sys.executable, "-m", "cascade.cli", "run", "--settings", str(settings_path)]
    started = perf_counter()
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    elapsed = perf_counter() - started
    command_log = root / "command.log"
    command_log.write_text(
        "$ " + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-30:])
        raise RuntimeError(f"CASCADE {backend} self-test failed (see {command_log}):\n{detail}")

    output_dir = root / "run"
    required = [
        output_dir / "manifest.json",
        output_dir / "summary.csv",
        output_dir / "segments.csv",
        output_dir / "points.csv",
        output_dir / "vessels.vtp",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"CASCADE self-test did not create required outputs: {missing}")

    import pyvista as pv

    vessels = pv.read(output_dir / "vessels.vtp")
    if int(vessels.n_points) <= 0:
        raise RuntimeError("CASCADE self-test produced an empty vessels.vtp file.")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("cascade", {}).get("version") != __version__:
        raise RuntimeError("Self-test manifest version does not match the installed CASCADE package.")
    if manifest.get("dependencies", {}).get("svv_version") != "0.0.48":
        raise RuntimeError("Self-test did not run against the qualified public svv==0.0.48 dependency.")
    if not manifest.get("inputs", {}).get("settings_sha256"):
        raise RuntimeError("Self-test manifest is missing the settings SHA-256 provenance field.")

    return {
        "status": "passed",
        "backend": backend,
        "cascade_version": __version__,
        "svv_version": metadata.version("svv"),
        "elapsed_seconds": elapsed,
        "vessel_points": int(vessels.n_points),
        "output_dir": str(output_dir),
        "command_log": str(command_log),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    holder: tempfile.TemporaryDirectory[str] | None = None
    if args.output_dir:
        root = Path(args.output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        retained = True
    elif args.keep:
        root = Path(tempfile.mkdtemp(prefix="cascade-self-test-"))
        retained = True
    else:
        holder = tempfile.TemporaryDirectory(prefix="cascade-self-test-")
        root = Path(holder.name)
        retained = False

    try:
        report = _run_case(root, require_gpu=bool(args.require_gpu))
        report["retained"] = retained
        if not retained:
            report["output_dir"] = None
            report["command_log"] = None
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            backend = report["backend"].upper()
            print(
                f"CASCADE installed-package self-test passed ({backend}, "
                f"{report['elapsed_seconds']:.3f}s, svv {report['svv_version']})."
            )
            if retained:
                print(f"Evidence retained under: {root}")
        return 0
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
