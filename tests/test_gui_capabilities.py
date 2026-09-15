from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _select(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_arbitrary_network_is_a_top_level_source_and_previews_exact_geometry(
    qapp, tmp_path
):
    from cascade.gui.model import default_project
    from cascade.gui.pages.vessels import VesselsPage
    from cascade.gui.visualization.geometry import network_geometry

    graph = tmp_path / "branch.csv"
    graph.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm\n"
        "0,0,0,1,0,0,0.01\n"
        "1,0,0,1,1,0,0.02\n",
        encoding="utf-8",
    )
    page = VesselsPage()
    assert page.source.findData("custom") >= 0
    assert page.simple_mode.findData("custom") < 0
    _select(page.source, "custom")
    page.custom_path.setText(str(graph))
    config = default_project()
    page.write(config)
    starts, ends, radii, *_rest = network_geometry(config)
    np.testing.assert_allclose(starts, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(ends, [[1, 0, 0], [1, 1, 0]])
    np.testing.assert_allclose(radii, [0.01, 0.02])


def test_network_export_controls_round_trip(qapp, tmp_path):
    from cascade.gui.model import default_project
    from cascade.gui.pages.vessels import VesselsPage

    path = tmp_path / "saved-network.npz"
    page = VesselsPage()
    config = default_project()
    page.load(config)
    page.save_network.setChecked(True)
    page.network_save_path.setText(str(path))
    page.write(config)
    assert config["outputs"]["save_network"] is True
    assert config["network"]["save_path"] == str(path)


def test_custom_constant_viscosity_fluid_round_trip(qapp, tmp_path):
    from cascade.configuration.parsing import load_config
    from cascade.gui.model import default_project
    from cascade.gui.pages.physics import PhysicsPage

    page = PhysicsPage()
    config = default_project()
    page.load(config)
    _select(page.fluid, "custom")
    page.custom_density.setValue(1.12)
    page.custom_viscosity.setValue(3.4)
    page.write(config)
    assert config["simulation"]["fluid"] == "custom"
    assert config["simulation"]["build_fluid"] == "custom"
    hemo = config["settings"]["hemodynamics"]
    assert hemo["custom_fluid_density_g_cm3"] == pytest.approx(1.12)
    assert hemo["custom_fluid_dynamic_viscosity_cp"] == pytest.approx(3.4)
    path = tmp_path / "custom-fluid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.simulation.fluid == "custom"
    assert loaded.runtime_settings["hemodynamics"][
        "custom_fluid_dynamic_viscosity_cp"
    ] == pytest.approx(3.4)


def test_solver_has_no_preset_and_uses_one_stop_rule(qapp):
    from cascade.gui.model import default_project
    from cascade.gui.pages.solver import SolverPage

    page = SolverPage()
    config = default_project()
    page.load(config)
    assert not hasattr(page, "preset")
    assert page.flow_solver.currentData() == "spsolve"
    _select(page.hct_stop, "iterations")
    _select(page.cext_stop, "tolerance")
    page.write(config)
    assert config["settings"]["hematocrit"]["hdtol"] == -1.0
    assert config["settings"]["hematocrit"]["qtol_nl_min"] == -1.0
    assert config["settings"]["cext"]["vess_coupling_max_iter"] == 1000
    assert config["settings"]["cext"]["vess_coupling_tol"] >= 0.0


def test_root_endpoint_rows_follow_inlet_count(qapp):
    from cascade.gui.model import default_project
    from cascade.gui.pages.vessels import VesselsPage

    page = VesselsPage()
    config = default_project()
    page.load(config)
    _select(page.topology, "forest")
    page.inlet_count.setValue(3)
    page.auto_roots.setChecked(False)
    assert len(page.root_rows) == 3
    page.root_rows[0]["proximal"].setText("0, 0, 0")
    page.root_rows[0]["distal"].setText("1, 2, 3")
    page.write(config)
    assert len(config["network"]["roots"]) == 3
    assert config["network"]["roots"][0] == {
        "start": [0.0, 0.0, 0.0],
        "direction": [1.0, 2.0, 3.0],
    }


def test_solver_only_changes_do_not_rebuild_network_preview(qapp):
    from copy import deepcopy
    from cascade.gui.main import MainWindow
    from cascade.gui.model import default_project

    window = MainWindow()
    first = default_project()
    second = deepcopy(first)
    second.setdefault("settings", {}).setdefault("hemodynamics", {})[
        "kirchhoff_solver"
    ] = "cg"
    second.setdefault("settings", {}).setdefault("hematocrit", {}).update(
        {"flow_iterations": 30, "hdtol": 1e-8}
    )
    assert window._case_preview_signature(first) == window._case_preview_signature(second)
    window.close()
