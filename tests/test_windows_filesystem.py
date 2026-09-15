from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from cascade.runtime import tissuesim as ts
from cascade.utils.execution import SimulationAlreadyRunningError, single_simulation
from cascade.vessels.prepared import prepare_tree_archive

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows semantics")


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


def _geometry_config(output: str, network_path: str) -> dict:
    return {
        "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
        "network": {
            "mode": "simple",
            "simple": {"mode": "custom", "path": network_path},
        },
        "growth": {"enabled": False},
        "simulation": {
            "fluid": "water",
            "qin_target_ul_min": 2.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 0,
            "tissue_accel": "cpu",
            "geometry_only": True,
        },
        "settings": {
            "hematocrit": {"flow_iterations": 0},
            "cext": {"accel_mode": "cpu"},
            "tissue": {"accel_mode": "cpu"},
        },
        "outputs": {
            "out_dir": output,
            "prefix": "path_case",
            "write_paraview": True,
            "save_network": False,
        },
    }


def test_cli_handles_spaces_unicode_deep_paths_and_different_cwd(
    tmp_path: Path,
) -> None:
    case_root = tmp_path / "Path With Spaces" / "Unicode 血管 Ω"
    for index in range(7):
        case_root /= f"nested-level-{index:02d}-abcdefgh"
    inputs = case_root / "inputs"
    settings_dir = case_root / "configuration"
    inputs.mkdir(parents=True)
    settings_dir.mkdir()
    network = inputs / "vascular geometry.csv"
    network.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm\n"
        "-0.4,0,0,0.4,0,0,0.02\n",
        encoding="utf-8",
    )
    settings = settings_dir / "case settings.json"
    settings.write_text(
        json.dumps(
            _geometry_config(
                "../results/Output With Spaces Ω",
                "../inputs/vascular geometry.csv",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    assert len(str(settings)) > 200
    _run_cli(tmp_path, "run", "--settings", str(settings.resolve()))
    output = case_root / "results" / "Output With Spaces Ω"
    assert (output / "manifest.json").is_file()
    assert (output / "vessels.vtp").is_file()

    relative_settings = settings.relative_to(tmp_path)
    inspected = _run_cli(
        tmp_path,
        "inspect",
        "--settings",
        str(relative_settings),
    )
    assert '"kind": "custom-csv"' in inspected.stdout


def test_prepared_tree_mmaps_release_for_windows_directory_rename(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CASCADE_PREPARED_CACHE_MIN_FREE_GB", "0")
    domain = ts.build_domain(1.0, random_seed=42)
    tree = ts.Tree(data_dtype=np.float64, index_dtype=np.int64, preallocation_step=4)
    tree.set_domain(domain)
    tree.set_root(
        np.asarray([0.49, -0.49, -0.49]),
        np.asarray([-0.49, 0.49, 0.49]),
    )
    source = tmp_path / "mmap-source.tree.npz"
    tree.save(str(source), include_domain=False)
    prepared = prepare_tree_archive(
        source,
        data_dtype=np.float64,
        index_dtype=np.int64,
        cache_root=tmp_path / "prepared",
    )
    monkeypatch.setenv("CASCADE_PREPARED_CACHE_DIR", str(tmp_path / "prepared"))
    loaded = ts.Tree.load(str(source), domain=domain, analysis_only=True)
    assert loaded._cascade_prepared_mmap

    del loaded
    gc.collect()
    renamed = prepared.root.with_name(prepared.root.name + "-renamed")
    prepared.root.rename(renamed)
    assert (renamed / "data.npy").is_file()


def test_stale_domain_cache_rebuilds_and_output_replacement_is_repeatable(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "surface.stl"
    pv.Sphere(
        radius=0.5,
        theta_resolution=12,
        phi_resolution=12,
    ).triangulate().save(surface)
    network = tmp_path / "network.csv"
    network.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm\n"
        "-0.4,0,0,0.4,0,0,0.02\n",
        encoding="utf-8",
    )
    config = _geometry_config("first-output", "network.csv")
    config["domain"] = {
        "type": "file",
        "path": "surface.stl",
        "side_length": 1.0,
        "random_seed": 42,
        "use_cache": True,
        "cache_dir": "domain-cache",
    }
    settings = tmp_path / "stale-cache.json"
    settings.write_text(json.dumps(config, indent=2), encoding="utf-8")
    _run_cli(tmp_path, "run", "--settings", str(settings))

    cached = list((tmp_path / "domain-cache").iterdir())
    assert len(cached) == 1
    cached[0].write_bytes(b"deliberately stale cache")
    config["outputs"]["out_dir"] = "replacement-output"
    config["outputs"]["overwrite"] = True
    settings.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rebuilt = _run_cli(tmp_path, "run", "--settings", str(settings))
    assert "rebuilding" in rebuilt.stdout.lower()
    _run_cli(tmp_path, "run", "--settings", str(settings))
    manifest = json.loads(
        (tmp_path / "replacement-output" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["settings"]["domain"]["type"] == "file"


def test_simulation_lock_rejects_concurrent_process_and_recovers(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "state" / "simulation.lock"
    code = (
        "from cascade.utils.execution import single_simulation\n"
        "import time\n"
        f"with single_simulation('holder', lock_path={str(lock_path)!r}):\n"
        "    print('LOCKED', flush=True)\n"
        "    time.sleep(30)\n"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "LOCKED"
        with pytest.raises(SimulationAlreadyRunningError):
            with single_simulation("contender", lock_path=lock_path):
                pass
    finally:
        holder.terminate()
        holder.wait(timeout=10)

    with single_simulation("recovered", lock_path=lock_path):
        pass
