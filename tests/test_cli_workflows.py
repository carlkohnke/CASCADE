from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyvista as pv


def _run_cli(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "cascade.commands.main", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    return completed


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _base_simple_config(out_dir: Path) -> dict:
    return {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {
                "mode": "onechannel",
                "radius_cm": 0.02,
                "flow_ul_min": 2.0,
            },
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "water",
            "qin_target_ul_min": 2.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 8,
            "tissue_accel": "cpu",
        },
        "settings": {
            "hematocrit": {"flow_iterations": 0},
            "cext": {"accel_mode": "cpu"},
            "tissue": {"accel_mode": "cpu"},
        },
        "outputs": {
            "out_dir": str(out_dir),
            "prefix": "native_cli",
            "write_paraview": True,
            "save_network": False,
        },
    }


def _assert_full_exports(out_dir: Path) -> None:
    required = {
        "summary.csv",
        "segments.csv",
        "points.csv",
        "vessels.vtp",
        "oxygen_points.vtp",
        "domain_boundary.vtp",
        "domain_mesh.vtu",
        "manifest.json",
    }
    assert required.issubset(path.name for path in out_dir.iterdir())

    vessels = pv.read(out_dir / "vessels.vtp")
    oxygen = pv.read(out_dir / "oxygen_points.vtp")
    boundary = pv.read(out_dir / "domain_boundary.vtp")
    volume = pv.read(out_dir / "domain_mesh.vtu")
    assert vessels.n_points > 0
    assert {"concentration", "flow_ul_min", "pressure_pa"}.issubset(
        vessels.point_data
    )
    assert oxygen.n_points == 8
    assert {"local_concentration", "viability"}.issubset(oxygen.point_data)
    assert boundary.n_points > 0
    assert volume.n_cells > 0


def test_custom_csv_and_npz_runs_cover_flow_pressure_and_custom_fluid(
    tmp_path: Path,
) -> None:
    starts = np.asarray([[-0.4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ends = np.asarray([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    radii = np.asarray([0.02, 0.018])
    prox_ids = np.asarray([0, 1], dtype=np.int64)
    dist_ids = np.asarray([1, 2], dtype=np.int64)

    csv_path = tmp_path / "network.csv"
    csv_path.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm,prox_id,dist_id\n"
        "-0.4,0,0,0,0,0,0.02,0,1\n"
        "0,0,0,0.4,0,0,0.018,1,2\n",
        encoding="utf-8",
    )
    npz_path = tmp_path / "network.npz"
    np.savez(
        npz_path,
        starts=starts,
        ends=ends,
        radii=radii,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
    )

    flow_config = _base_simple_config(tmp_path / "csv-flow-output")
    flow_config["network"]["simple"] = {"mode": "custom", "path": "network.csv"}
    _run_cli(
        tmp_path,
        "run",
        "--settings",
        str(_write_json(tmp_path / "csv-flow.json", flow_config)),
    )
    _assert_full_exports(tmp_path / "csv-flow-output")

    pressure_config = _base_simple_config(tmp_path / "npz-pressure-output")
    pressure_config["network"]["simple"] = {"mode": "custom", "path": "network.npz"}
    pressure_config["settings"]["kirchhoff"] = {
        "solver": "tree",
        "bc_mode": "pressure_pressure",
    }
    pressure_config["settings"]["hemodynamics"] = {
        "root_pressure": 7000.0,
        "terminal_pressure": 6000.0,
        "scale_dp_by_volume": False,
    }
    _run_cli(
        tmp_path,
        "run",
        "--settings",
        str(_write_json(tmp_path / "npz-pressure.json", pressure_config)),
    )
    _assert_full_exports(tmp_path / "npz-pressure-output")
    with (tmp_path / "npz-pressure-output" / "summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = next(csv.DictReader(handle))
    assert float(summary["pressure_drop_mean"]) == 1000.0
    assert float(summary["inlet_flow_ul_per_min_mean"]) > 0.0

    custom_config = _base_simple_config(tmp_path / "custom-fluid-output")
    custom_config["network"]["simple"] = {"mode": "custom", "path": "network.csv"}
    custom_config["simulation"]["fluid"] = "custom"
    custom_config["settings"]["hemodynamics"] = {
        "custom_fluid_density_g_cm3": 1.05,
        "custom_fluid_dynamic_viscosity_cp": 1.25,
    }
    _run_cli(
        tmp_path,
        "run",
        "--settings",
        str(_write_json(tmp_path / "custom-fluid.json", custom_config)),
    )
    _assert_full_exports(tmp_path / "custom-fluid-output")


def test_batch_and_sweep_complete_with_serial_cpu_reuse(tmp_path: Path) -> None:
    first = _base_simple_config(tmp_path / "batch-one")
    first["simulation"].update(
        {"distance_sample_count": 0, "skip_tissue_oxygen": True}
    )
    first["outputs"].update({"write_paraview": False})
    second = deepcopy(first)
    second["simulation"]["fluid"] = "custom"
    second["settings"]["hemodynamics"] = {
        "custom_fluid_density_g_cm3": 1.02,
        "custom_fluid_dynamic_viscosity_cp": 1.1,
    }
    second["outputs"]["out_dir"] = str(tmp_path / "batch-two")
    first_path = _write_json(tmp_path / "batch-one.json", first)
    second_path = _write_json(tmp_path / "batch-two.json", second)

    batch = _run_cli(
        tmp_path,
        "batch",
        "--settings",
        str(first_path),
        str(second_path),
    )
    assert "Batch finished: cases=2 failures=0" in batch.stdout
    assert (tmp_path / "batch-one" / "manifest.json").is_file()
    assert (tmp_path / "batch-two" / "manifest.json").is_file()

    sweep = deepcopy(first)
    sweep["outputs"]["out_dir"] = str(tmp_path / "sweep-work")
    sweep["outputs"]["overwrite"] = True
    sweep["sweep"] = {
        "target_terminal_counts": [1],
        "fluids": ["water", "custom"],
        "qin_target_ul_min_values": [1.0, 2.0],
        "distance_sample_counts": [0],
        "output_csv": str(tmp_path / "sweep-summary.csv"),
        "work_dir": str(tmp_path / "sweep-work"),
        "save_final_network": False,
    }
    _run_cli(
        tmp_path,
        "sweep",
        "--settings",
        str(_write_json(tmp_path / "sweep.json", sweep)),
    )
    with (tmp_path / "sweep-summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["fluid"] for row in rows} == {"water", "custom"}
    assert (tmp_path / "sweep-summary_manifest.json").is_file()
