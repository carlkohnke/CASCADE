from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from cascade import __version__
from cascade.cli import main as cli_main
from cascade.config import example_config, parse_config
from cascade.doctor import collect_diagnostics
from cascade.heart_export import (
    CEXT_BG_GRID_DEFAULT,
    CEXT_GL_ORDER_DEFAULT,
    CEXT_VESS_COUPLING_MAX_ITER_DEFAULT,
    EXPORT_FLOAT_DTYPE_DEFAULT,
    GL_ORDER_DEFAULT,
    WINDOW_FACTOR_DEFAULT,
    _load_tissuesim,
    _should_use_simulation_cache,
)
from cascade.resources import resolve_domain_path
from cascade.settings.oxygen import DEFAULTS as OXYGEN_DEFAULTS
from cascade.simulation import DYN_PER_CM2_TO_PA, _segment_rows
from cascade.simple import _load_custom_geometry


def test_release_version_and_cli_contract(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])
    assert exc.value.code == 0
    assert f"CASCADE {__version__}" in capsys.readouterr().out


def test_unknown_settings_fail_loudly():
    raw = example_config()
    raw["simluation"] = {"geometry_only": True}
    with pytest.raises(ValueError, match=r"Unknown CASCADE setting\(s\): simluation"):
        parse_config(raw)

    raw = example_config()
    raw["simulation"]["geometery_only"] = True
    with pytest.raises(ValueError, match="simulation.geometery_only"):
        parse_config(raw)


def test_custom_csv_geometry_contract(tmp_path: Path):
    geometry = tmp_path / "y-channel.csv"
    geometry.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm\n"
        "0,0,0,1,0,0,0.02\n"
        "1,0,0,2,1,0,0.01\n"
        "1,0,0,2,-1,0,0.01\n",
        encoding="utf-8",
    )

    starts, ends, radii, lengths, inlets, outlets, prox, dist = _load_custom_geometry(
        geometry, default_radius_cm=0.015
    )

    assert starts.shape == ends.shape == (3, 3)
    np.testing.assert_allclose(radii, [0.02, 0.01, 0.01])
    np.testing.assert_allclose(lengths, [1.0, np.sqrt(2.0), np.sqrt(2.0)])
    assert len(inlets) == 1
    assert len(outlets) == 2
    assert dist[0] == prox[1] == prox[2]


def test_packaged_heart_export_uses_packaged_runtime():
    runtime = _load_tissuesim()
    assert runtime.__name__ == "cascade.runtime.tissuesim"
    assert callable(runtime.build_domain)
    assert callable(runtime.compute_tissue_samples_greens)


def test_m2_heart_profile_defaults_are_frozen():
    assert EXPORT_FLOAT_DTYPE_DEFAULT == "float32"
    assert CEXT_BG_GRID_DEFAULT == 256
    assert CEXT_GL_ORDER_DEFAULT == 1
    assert GL_ORDER_DEFAULT == 5
    assert CEXT_VESS_COUPLING_MAX_ITER_DEFAULT == 1
    assert WINDOW_FACTOR_DEFAULT == 6.0
    assert OXYGEN_DEFAULTS["CONCENTRATION_INLET_BY_FLUID"]["water"] == 0.2211
    assert OXYGEN_DEFAULTS["CONCENTRATION_INLET_BY_FLUID"]["cell media"] == 0.2211


def test_packaged_bivent3_domain_is_resolvable_and_frozen():
    path = resolve_domain_path("bivent3.stl")
    assert path is not None and path.is_file()
    assert path.stat().st_size == 652_584
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "10fd497650eb881b37390861cb960db88e0c7e6f0b18267ffbb35d564c24a060"
    )


def test_heart_export_recognizes_cascade_simulation_cache(tmp_path: Path):
    forest = tmp_path / "heart.forest"
    cache = tmp_path / "heart.forest.simcache"
    forest.write_bytes(b"forest")
    with zipfile.ZipFile(cache, "w") as archive:
        archive.writestr("simulation_meta.pkl", b"placeholder")
    assert _should_use_simulation_cache(forest, cache)


def test_segment_pressure_export_converts_cgs_solver_values_to_pa():
    details = {
        "starts": np.zeros((1, 3)),
        "ends": np.ones((1, 3)),
        "radii": np.asarray([0.01]),
        "lengths": np.asarray([1.0]),
        "flows": np.asarray([0.001]),
        "pressures": np.asarray([666_610.0]),
        "cin": np.asarray([0.14]),
        "cout": np.asarray([0.13]),
    }
    rows, _ = _segment_rows(details, {}, tree_id=0, start_global_id=0)
    assert DYN_PER_CM2_TO_PA == 0.1
    assert rows[0]["pressure_pa"] == pytest.approx(66_661.0)


def test_doctor_machine_readable_contract():
    report = collect_diagnostics(probe_gpu=False)
    assert report["cascade_version"] == __version__
    assert report["gpu"] is None
    assert report["python_executable"]
    assert "svv" in report["packages"]
    json.dumps(report)
