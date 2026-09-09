#!/usr/bin/env python3
"""Install a CASCADE wheel in isolation and exercise the public CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import venv


def run(command: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install a wheel in a temporary venv and execute CASCADE release smoke cases."
    )
    parser.add_argument("--wheel", required=True, help="Exact path to one CASCADE wheel.")
    parser.add_argument("--constraints", help="Optional pip constraints/lock file.")
    parser.add_argument(
        "--gpu-extra",
        choices=("gpu-cu11", "gpu-cu12", "gpu-cu13"),
        help="Install and require this GPU extra.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep the temporary directory.")
    args = parser.parse_args(argv)

    wheel = Path(args.wheel).expanduser().resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")
    constraints = Path(args.constraints).expanduser().resolve() if args.constraints else None
    if constraints is not None and not constraints.is_file():
        parser.error(f"constraints file not found: {constraints}")

    holder = None if args.keep else tempfile.TemporaryDirectory(prefix="cascade-release-smoke-")
    root = Path(tempfile.mkdtemp(prefix="cascade-release-smoke-")) if holder is None else Path(holder.name)
    try:
        env_dir = root / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cascade = env_dir / ("Scripts/cascade.exe" if sys.platform == "win32" else "bin/cascade")

        extras = ["gui"]
        if args.gpu_extra:
            extras.append(args.gpu_extra)
        wheel_spec = f"{wheel}[{','.join(extras)}]"
        install = [str(python), "-m", "pip", "install"]
        if constraints is not None:
            install.extend(["-c", str(constraints)])
        install.append(wheel_spec)
        run(install, cwd=root)

        run([str(cascade), "--version"], cwd=root)
        doctor = [str(cascade), "doctor", "--json"]
        if args.gpu_extra:
            doctor.append("--require-gpu")
        run(doctor, cwd=root)
        run([str(cascade), "export-heart", "--help"], cwd=root)
        run(
            [str(python), "-c", "import PySide6, cascade.gui; print('GUI import passed')"],
            cwd=root,
        )

        geometry = root / "custom_y_channel.csv"
        geometry.write_text(
            "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm,prox_id,dist_id\n"
            "-0.45,0,0,0,0,0,0.02,0,1\n"
            "0,0,0,0.45,0.2,0,0.015,1,2\n"
            "0,0,0,0.45,-0.2,0,0.015,1,3\n",
            encoding="utf-8",
        )

        cases = {
            "tree": {
                "schema_version": 1,
                "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
                "network": {
                    "mode": "tree",
                    "target_terminal_count": 1,
                    "root": {
                        "start": [0.49, -0.49, -0.49],
                        "direction": [-0.49, 0.49, 0.49],
                    },
                },
                "growth": {"enabled": False},
                "simulation": {
                    "fluid": "blood",
                    "qin_target_ul_min": 900.0,
                    "concentration_solver": "topdown",
                    "distance_sample_count": 32,
                    "tissue_accel": "cpu",
                },
                "outputs": {"out_dir": str(root / "tree-out"), "write_paraview": True},
            },
            "custom": {
                "schema_version": 1,
                "domain": {"type": "box", "dimensions": [1, 1, 1], "random_seed": 42},
                "network": {
                    "mode": "simple",
                    "simple": {
                        "mode": "custom",
                        "path": str(geometry),
                        "flow_ul_min": 24.0,
                        "concentration_inlet": 0.22471,
                    },
                },
                "growth": {"enabled": False},
                "simulation": {
                    "fluid": "water",
                    "build_fluid": "water",
                    "qin_target_ul_min": 24.0,
                    "concentration_solver": "network",
                    "distance_sample_count": 32,
                    "tissue_accel": "cpu",
                },
                "outputs": {"out_dir": str(root / "custom-out"), "write_paraview": True},
            },
            "forest": {
                "schema_version": 1,
                "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
                "network": {
                    "mode": "forest",
                    "target_terminal_counts": [1, 1],
                    "roots": [
                        {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]},
                        {"start": [-0.49, 0.49, -0.49], "direction": [0.49, -0.49, 0.49]},
                    ],
                },
                "growth": {"enabled": False},
                "simulation": {"geometry_only": True, "tissue_accel": "cpu"},
                "outputs": {
                    "out_dir": str(root / "forest-out"),
                    "prefix": "forest_smoke",
                    "write_paraview": True,
                },
            },
        }
        for name, settings in cases.items():
            path = root / f"{name}.json"
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            run([str(cascade), "run", "--settings", str(path)], cwd=root)
            output = Path(settings["outputs"]["out_dir"])
            required = (output / "manifest.json", output / "summary.csv", output / "vessels.vtp")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise RuntimeError(f"{name} smoke case is missing outputs: {missing}")

        heart_out = root / "heart-out"
        heart_command = [
            str(cascade),
            "export-heart",
            "--forest", str(root / "forest-out" / "forest_smoke.forest"),
            "--domain", str(root / "forest-out" / "domain_boundary.vtp"),
            "--out-dir", str(heart_out),
            "--prefix", "heart_smoke",
            "--nx", "4", "--ny", "4", "--nz", "4",
            "--finite-radius-o2-terms", "none",
            "--lumen-wall-closure", "wellmixed",
            "--vessel-resolution", "2",
        ]
        if args.gpu_extra:
            heart_command.extend(
                [
                    "--cext-bg-grid", "8",
                    "--cext-bg-lambda-bins", "2",
                    "--cext-vess-coupling-max-iter", "1",
                    "--cext-active-set-enable", "false",
                    "--cext-target-active-set-enable", "false",
                    "--tissue-accel", "gpu",
                    "--tissue-gpu-validate-points", "0",
                ]
            )
        else:
            heart_command.extend(["--no-cext", "--tissue-accel", "cpu"])
        run(heart_command, cwd=root)

        validation = (
            "import json, pathlib, pyvista as pv; "
            f"root=pathlib.Path({str(root)!r}); "
            "assert pv.read(root/'tree-out'/'vessels.vtp').n_points > 0; "
            "assert pv.read(root/'custom-out'/'vessels.vtp').n_points > 0; "
            "assert pv.read(root/'heart-out'/'heart_smoke_forest_vessels.vtp').n_points > 0; "
            "assert pv.read(root/'heart-out'/'heart_smoke_forest_oxygen_points.vtp').n_points > 0; "
            "m=json.loads((root/'custom-out'/'manifest.json').read_text()); "
            "assert m['cascade']['version'] and m['dependencies']['svv_version']=='0.0.48'; "
            "assert m['inputs']['settings_sha256']; "
            "print('VTK and manifest validation passed')"
        )
        run([str(python), "-c", validation], cwd=root)
        print(f"CASCADE wheel smoke test passed: {wheel}")
        return 0
    finally:
        if holder is None:
            print(f"Kept smoke directory: {root}")
        else:
            holder.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
