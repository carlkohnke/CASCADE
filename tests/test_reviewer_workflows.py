"""Regression coverage for reviewer entry points and serial scientific runs."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.configuration import load_config, parse_config
from cascade.configuration.bridge import apply_runtime_settings
from cascade.configuration.settings import default_settings


def _runtime():
    return SimpleNamespace(
        **{
            name: deepcopy(value)
            for values in default_settings().values()
            for name, value in values.items()
        }
    )


@pytest.mark.parametrize("frontend", ["cli", "gui"])
def test_physics_defaults_are_independent_of_previous_run(frontend):
    from cascade.configuration.schema import example_config
    from cascade.gui.model import default_project

    factory = example_config if frontend == "cli" else default_project
    prior = factory()
    prior["settings"]["oxygen"].update(
        {
            "vmax_mm": 0.001,
            "solute_diffusivity": 1e-5,
            "lumen_diffusivity_cm2_s": 9e-5,
        }
    )
    prior["settings"]["hematocrit"]["hd_discharge"] = 0.2
    reused = _runtime()
    apply_runtime_settings(reused, parse_config(prior))
    reused.CONCENTRATION_INLET_BY_FLUID["blood"] = 99
    next_run = factory()
    next_run["simulation"]["qin_target_ul_min"] = 123
    next_run["simulation"]["concentration_solver"] = "topdown"
    fresh = _runtime()
    apply_runtime_settings(fresh, parse_config(next_run))
    apply_runtime_settings(reused, parse_config(next_run))
    for name in (
        "VMAX_MM",
        "SOLUTE_DIFFUSIVITY",
        "LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S",
        "LUMEN_DIFFUSIVITY_CM2_S",
        "HD_DISCHARGE",
        "CONCENTRATION_INLET_BY_FLUID",
    ):
        assert getattr(reused, name) == getattr(fresh, name)
    assert reused.QIN_TARGET == 123
    assert reused.CONCENTRATION_SOLVER == "topdown"
    assert default_settings()["oxygen"]["CONCENTRATION_INLET_BY_FLUID"]["blood"] == 0.14


@pytest.mark.parametrize("mode", ["tree", "forest"])
def test_disabled_growth_requires_a_saved_network(mode):
    with pytest.raises(ValueError, match="network.input_path.*growth.enabled=false"):
        parse_config({"network": {"mode": mode}, "growth": {"enabled": False}})


def test_custom_geometry_alias_resolves_relative_to_settings(tmp_path):
    case = tmp_path / "case with spaces"
    case.mkdir()
    (case / "vessels.csv").write_text("radius_cm\n0.01\n")
    settings = case / "run.json"
    settings.write_text(
        json.dumps(
            {
                "network": {
                    "mode": "simple",
                    "simple": {
                        "mode": "custom",
                        "geometry_path": "vessels.csv",
                    },
                },
                "growth": {"enabled": False},
            }
        )
    )
    assert load_config(settings).network.simple["geometry_path"] == "vessels.csv"


def test_missing_network_fails_before_preparation(tmp_path):
    settings = tmp_path / "run.json"
    settings.write_text(json.dumps({"network": {"input_path": "absent.tree.npz"}}))
    with pytest.raises(FileNotFoundError, match="Network input file not found"):
        load_config(settings)


def test_resume_uses_existing_checkpoint_when_original_input_is_missing(tmp_path):
    (tmp_path / "checkpoint.tree.npz").touch()
    settings = tmp_path / "run.json"
    settings.write_text(
        json.dumps(
            {
                "network": {"input_path": "absent.tree.npz"},
                "growth": {
                    "resume_from_checkpoint": True,
                    "checkpoint_path": "checkpoint.tree.npz",
                },
            }
        )
    )
    assert load_config(settings).growth.resume_from_checkpoint


def _simple_settings(tmp_path):
    settings = tmp_path / "run.json"
    settings.write_text(
        json.dumps(
            {
                "network": {"mode": "simple"},
                "growth": {"enabled": False},
                "simulation": {"qin_target_ul_min": 123},
                "settings": {"oxygen": {"vmax_mm": 0.003}},
                "outputs": {"out_dir": "results"},
            }
        )
    )
    return settings


def test_unwritable_output_fails_before_gpu_or_network_work(tmp_path, monkeypatch):
    import cascade.accelerators.backend as backend
    from cascade.commands.main import _resolve_configured_simulation

    settings = _simple_settings(tmp_path)
    (tmp_path / "results").write_text("existing file")
    monkeypatch.setattr(
        backend, "require_gpu_runtime", lambda config: pytest.fail("late preflight")
    )
    with pytest.raises(OSError, match="Choose a writable output directory"):
        _resolve_configured_simulation(settings, allow_detailed_cache=False)


def test_cached_gui_build_applies_current_settings_and_manifest(tmp_path, monkeypatch):
    import cascade.accelerators.backend as backend
    import cascade.configuration.bridge as bridge
    import cascade.simulation.engine as engine
    from cascade.commands.main import _resolve_configured_simulation

    state = _runtime()
    state.QIN_TARGET = 999
    state.VMAX_MM = 999
    monkeypatch.setattr(backend, "require_gpu_runtime", lambda config: None)
    monkeypatch.setattr(bridge, "load_runtime_module", lambda: state)
    build = SimpleNamespace(trees=[], domain=None, target_counts=[], sample_points=None)
    workspace = SimpleNamespace(
        resolve_build=lambda config: (build, True),
        resolve_tissue_caches=lambda config: None,
        resolve_simulation=lambda config: (None, "key"),
        store_simulation=lambda *args, **kwargs: None,
    )

    def solve(*args, **kwargs):
        assert state.QIN_TARGET == 123
        assert state.VMAX_MM == 0.003
        return "solved"

    monkeypatch.setattr(engine, "run_simulation", solve)
    config, _, result = _resolve_configured_simulation(
        _simple_settings(tmp_path),
        workspace=workspace,
        allow_detailed_cache=False,
    )
    assert result == "solved"
    assert config.runtime_setting_overrides["oxygen"]["VMAX_MM"] == 0.003


def test_publication_demo_allows_gpu_with_cpu_fallback():
    from cascade.accelerators.backend import gpu_requested, _gpu_required

    source = Path(__file__).resolve().parents[1] / "examples" / "publication"
    config = load_config(source / "demo.json")
    assert gpu_requested(config)
    assert not _gpu_required(config)


def test_publication_demo_runs_from_an_unrelated_directory(tmp_path):
    import pyvista as pv

    source = Path(__file__).resolve().parents[1] / "examples" / "publication"
    case = tmp_path / "demo with spaces"
    case.mkdir()
    shutil.copy(source / "demo.tree.npz", case)
    raw = json.loads((source / "demo.json").read_text())
    raw["outputs"]["out_dir"] = "results"
    settings = case / "demo.json"
    settings.write_text(json.dumps(raw))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cascade.commands.main",
            "run",
            "--settings",
            str(settings),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "GPU preflight passed" in completed.stdout
        or "using CPU-compatible solver paths" in completed.stdout
    )
    outputs = case / "results"
    with (outputs / "summary.csv").open(newline="") as handle:
        summary = next(csv.DictReader(handle))
    expected_backend = "gpu" if "GPU preflight passed" in completed.stdout else "cpu"
    assert summary["t_tissue_backend"].startswith(expected_backend)
    assert float(summary["total_segments"]) == 201
    assert float(summary["terminal_segments"]) == 101
    assert float(summary["inlet_flow_ul_per_min"]) == pytest.approx(100)
    # Rounded references intentionally tolerate numerical variation across platforms.
    assert float(summary["C_LQ_over_Cmax"]) == pytest.approx(0.889, abs=0.005)
    assert float(summary["C_tiss_over_Cmax"]) == pytest.approx(0.0345, abs=0.002)
    oxygen = pv.read(outputs / "oxygen_points.vtp")
    assert 9900 <= oxygen.n_points <= 10000
    assert np.isfinite(oxygen.point_data["local_concentration"]).all()
    assert pv.read(outputs / "domain_mesh.vtu").n_cells > 0
    assert (outputs / "publication_demo.tree.npz").is_file()
    manifest = json.loads((outputs / "manifest.json").read_text())
    assert manifest


def test_blood_batch_matches_independent_runs_after_physics_and_flow_changes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples" / "publication"
    shutil.copy(source / "demo.tree.npz", tmp_path)
    base = json.loads((source / "demo.json").read_text())
    base["simulation"]["distance_sample_count"] = 128
    base["outputs"].update({"write_paraview": False, "save_network": False})
    prior = deepcopy(base)
    prior["outputs"]["out_dir"] = "prior"
    second = deepcopy(base)
    second["settings"]["oxygen"].pop("vmax_mm")
    second["outputs"]["out_dir"] = "second"
    third = deepcopy(second)
    third["simulation"]["qin_target_ul_min"] = 200
    third["outputs"]["out_dir"] = "third"

    def run(*args):
        completed = subprocess.run(
            [sys.executable, "-m", "cascade.commands.main", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return completed.stdout

    def summary(folder):
        with (tmp_path / folder / "summary.csv").open(newline="") as handle:
            return next(csv.DictReader(handle))

    paths = []
    for label, raw in (("prior", prior), ("second", second), ("third", third)):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(raw))
        paths.append(str(path))
    independent = []
    independent_points = []
    for path, label in zip(paths[1:], ("second", "third")):
        run("run", "--settings", path)
        independent.append(summary(label))
        independent_points.append(
            np.genfromtxt(
                tmp_path / label / "points.csv",
                delimiter=",",
                names=True,
            )
        )
    log = run("batch", "--settings", *paths)
    # On a busy host the memory guard can intentionally discard geometry.
    # Both cached and rebuilt batches must retain the same scientific results.
    assert (
        "Reusing interactive geometry" in log
        or "Interactive memory pressure: evicted all" in log
    )
    for expected, label in zip(independent, ("second", "third")):
        actual = summary(label)
        for field in (
            "inlet_flow_ul_per_min",
            "pressure_drop",
            "C_LQ_over_Cmax",
            "C_tiss_over_Cmax",
            "FracAbove1pct",
        ):
            assert float(actual[field]) == pytest.approx(
                float(expected[field]), rel=1e-6, abs=1e-9
            )
    for expected, label in zip(independent_points, ("second", "third")):
        actual = np.genfromtxt(
            tmp_path / label / "points.csv", delimiter=",", names=True
        )
        np.testing.assert_allclose(
            actual["local_concentration"], expected["local_concentration"], rtol=1e-6
        )
        np.testing.assert_array_equal(actual["viability"], expected["viability"])
