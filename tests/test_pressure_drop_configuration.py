import pytest

from cascade.configuration.runtime import RuntimeConfiguration
from cascade.configuration.schema import parse_config
from cascade.flow.topology import _normalize_kirchhoff_bc_mode
from cascade.gui.model import default_project, validate_project


def test_cli_configuration_selects_pressure_pressure_runtime_mode():
    raw = default_project()
    raw["simulation"]["kirchhoff_bc_mode"] = "pressure_pressure"
    raw["settings"]["hemodynamics"].update(
        {
            "root_pressure": 7080.0,
            "terminal_pressure": 6000.0,
            "scale_dp_by_volume": False,
        }
    )
    config = parse_config(raw)
    resolved = RuntimeConfiguration.from_run_config(config)

    assert resolved.sections["runtime"]["kirchhoff_bc_mode"] == "pressure_pressure"
    assert _normalize_kirchhoff_bc_mode("pressure_pressure") == "pressure_pressure"


def test_studio_project_validation_accepts_pressure_pressure_mode():
    project = default_project()
    project["gui"]["boundary_conditions"]["mode"] = "pressure_pressure"
    project["settings"]["hemodynamics"].update(
        {"root_pressure": 7080.0, "terminal_pressure": 6000.0}
    )

    report = validate_project(project)

    assert report.runnable
    assert any("Inlet flow will be calculated" in note for note in report.notes)


def test_studio_pressure_page_writes_enabled_pressure_mode(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from cascade.gui.pages.physics import PhysicsPage

    app = QApplication.instance() or QApplication([])
    page = PhysicsPage()
    project = default_project()
    page.load(project)
    index = page.bc_mode.findData("pressure_pressure")
    page.bc_mode.setCurrentIndex(index)
    page.write(project)

    assert not page.flow.isEnabled()
    assert (
        project["settings"]["hemodynamics"]["kirchhoff_bc_mode"]
        == "pressure_pressure"
    )
    assert validate_project(project).runnable
    page.deleteLater()
    app.processEvents()


def test_studio_recognizes_cli_authored_pressure_mode(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from cascade.gui.pages.physics import PhysicsPage

    app = QApplication.instance() or QApplication([])
    project = default_project()
    project["gui"]["boundary_conditions"]["mode"] = "flow_pressure"
    project["simulation"]["kirchhoff_bc_mode"] = "pressure_pressure"
    page = PhysicsPage()
    page.load(project)

    assert page.bc_mode.currentData() == "pressure_pressure"
    assert not page.flow.isEnabled()
    page.deleteLater()
    app.processEvents()
