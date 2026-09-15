from __future__ import annotations

import json

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtTest import QSignalSpy

from cascade.gui.pages.analysis import AnalysisPage
from cascade.gui.pages.physics import PhysicsPage
from cascade.gui.runner import JobRunner
from cascade.gui.window import MainWindow
from cascade.vessels.simple import _load_custom_geometry


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_non_blood_fields_display_na_without_losing_values(app):
    page = PhysicsPage()
    original = (page.hematocrit.value(), page.hb_capacity.value())
    for fluid in ("water", "custom"):
        page.fluid.setCurrentIndex(page.fluid.findData(fluid))
        app.processEvents()
        for display in (
            page.hematocrit_model_display,
            page.hematocrit_display,
            page.hb_capacity_display,
        ):
            assert display.currentIndex() == 1
            assert display.currentWidget().text() == "N/a"
    assert (page.hematocrit.value(), page.hb_capacity.value()) == original


def test_pressure_pressure_banner_omits_delta_p_explanation(app):
    page = PhysicsPage()
    page.bc_mode.setCurrentIndex(page.bc_mode.findData("pressure_pressure"))
    app.processEvents()
    assert all("solved from" not in label.text() for label in page.findChildren(QLabel))


def test_custom_geometry_requires_radii(tmp_path):
    csv_path = tmp_path / "network.csv"
    csv_path.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z\n0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="radius_cm"):
        _load_custom_geometry(csv_path, default_radius_cm=0.015)

    npz_path = tmp_path / "network.npz"
    np.savez(npz_path, starts=np.zeros((1, 3)), ends=np.ones((1, 3)))
    with pytest.raises(ValueError, match="radii array"):
        _load_custom_geometry(npz_path, default_radius_cm=0.015)


def test_unsaved_results_do_not_discover_old_result_folders(app, tmp_path):
    result_dir = tmp_path / "results" / "old-run"
    result_dir.mkdir(parents=True)
    (result_dir / "manifest.json").write_text(
        json.dumps({"settings": {"gui": {"project_name": "Old run"}}}),
        encoding="utf-8",
    )
    runner = JobRunner(tmp_path)
    page = AnalysisPage()
    page.project_path = None
    page.set_runner(runner)
    assert page.result_job.count() == 0
    page.project_path = tmp_path / "saved-project.json"
    page.refresh()
    assert page.result_job.count() == 1
    runner.shutdown()


def test_results_page_has_back_but_no_completion_button(app):
    window = MainWindow()
    window.show()
    window.nav.setCurrentRow(7)
    app.processEvents()
    assert window.back_btn.isVisible()
    assert not window.next_btn.isVisible()
    window.nav.setCurrentRow(6)
    app.processEvents()
    assert window.next_btn.isVisible()
    window.runner.shutdown()
    window.close()


def test_results_refreshes_automatically_when_page_is_shown(app, tmp_path):
    runner = JobRunner(tmp_path)
    page = AnalysisPage()
    page.set_runner(runner)
    refreshes = QSignalSpy(page.refresh_btn.clicked)
    page.show()
    app.processEvents()
    assert refreshes.count() == 1
    page.hide()
    page.show()
    app.processEvents()
    assert refreshes.count() == 2
    page.close()
    runner.shutdown()


def test_vessel_visibility_disables_normalization_control(app):
    page = AnalysisPage()
    page.vessel_visible.setChecked(False)
    app.processEvents()
    assert not page.normalize_fields.isEnabled()
    assert "QCheckBox:disabled" in page.normalize_fields.styleSheet()
