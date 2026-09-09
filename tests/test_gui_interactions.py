from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QProcess, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from gfm.gui.main import (
    AnalysisPage,
    APP_STYLE,
    DomainPage,
    MainWindow,
    OutputsPage,
    PhysicsPage,
    QueuePage,
    SolverPage,
    VesselsPage,
    QMessageBox as CascadeMessageBox,
)
from gfm.gui.widgets import InfoTip, NumberInput, labeled, row_of
from gfm.config import parse_config
from gfm.growth import flow_for_tree, pressures_for_tree
from gfm.gui.model import JobRecord, create_jobs, default_project, oxygen_to_concentration
from gfm.gui.runner import JobRunner
import gfm.gui.widgets as gui_widgets
from gfm.gui.preview import (
    CasePreview,
    GeometryCanvas,
    _count_status,
    _normalize_with_scale,
    _legend_tick_values,
    _layer_field_label,
    _mesh_wireframe,
    _svv_placeholder_geometry,
    domain_wireframe,
    domain_surface_triangles,
    limit_near_inlets,
    network_geometry,
    preview_tissue_geometry,
    select_tissue_points,
)
from gfm.gui.preview_worker import (
    _fast_file_preview_domain,
    main as preview_worker_main,
)
from gfm.gui.widgets import ChoiceComboBox, FocusPlainTextEdit, IconButton, PathPicker, StatusPill
from gfm.lattice import channel_count, generate_lattice


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_combo_hover_is_enabled_and_click_closes_popup(app):
    window = QWidget()
    layout = QVBoxLayout(window)
    combo = ChoiceComboBox(window)
    layout.addWidget(combo)
    combo.addItems(["First", "Second", "Third"])
    combo.resize(220, 36)
    window.resize(320, 240)
    window.show()
    combo.showPopup()
    app.processEvents()

    frame = combo._choice_frame
    view = combo._choice_list
    assert frame is not None and view is not None
    assert frame.parent() is window
    assert frame.isVisible()
    assert not frame.isWindow()
    assert view.hasMouseTracking()
    assert view.viewport().hasMouseTracking()
    assert "item:hover" in frame.styleSheet()
    assert view.verticalScrollBar().maximum() == 0
    last_rect = view.visualItemRect(view.item(view.count() - 1))
    assert last_rect.bottom() == view.viewport().rect().bottom()

    item = view.item(1)
    QTest.mouseMove(view.viewport(), view.visualItemRect(item).center())
    app.processEvents()
    assert view.currentRow() == 1
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualItemRect(item).center())
    QTest.qWait(20)
    app.processEvents()
    assert combo.currentIndex() == 1
    assert not frame.isVisible()

    # Clicking the already-selected entry must also dismiss the popup; in that
    # case currentIndexChanged does not fire, so the viewport handler matters.
    combo.showPopup()
    app.processEvents()
    item = view.item(1)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=view.visualItemRect(item).center())
    QTest.qWait(20)
    app.processEvents()
    assert combo.currentIndex() == 1
    assert not frame.isVisible()


def test_issue_pill_is_clickable(app):
    pill = StatusPill("2 issues", "danger")
    clicks = []
    pill.clicked.connect(lambda: clicks.append(True))
    pill.resize(100, 30)
    pill.show()
    QTest.mouseClick(pill, Qt.LeftButton)
    assert clicks == [True]


def test_info_tip_wraps_and_stays_within_window_bounds(app, monkeypatch):
    window = QWidget()
    window.resize(300, 180)
    tip = InfoTip("A deliberately long explanation that must wrap instead of extending past the application window.", parent=window)
    tip.move(275, 160)
    window.show()
    app.processEvents()
    shown = {}
    monkeypatch.setattr(
        gui_widgets.QToolTip,
        "showText",
        lambda pos, text, parent=None: shown.update(pos=pos, text=text, parent=parent),
    )

    tip._show_tip()

    assert "white-space:normal" in shown["text"]
    assert "width:" in shown["text"]
    bounds = window.frameGeometry()
    assert bounds.left() <= shown["pos"].x() <= bounds.right()
    assert bounds.top() <= shown["pos"].y() <= bounds.bottom()


def test_tab_moves_from_multiline_editor_without_inserting_text(app):
    window = QWidget()
    layout = QVBoxLayout(window)
    multiline = FocusPlainTextEdit()
    following = QLineEdit()
    layout.addWidget(multiline)
    layout.addWidget(following)
    window.show()
    window.activateWindow()
    multiline.setFocus()
    app.processEvents()

    QTest.keyClick(multiline, Qt.Key_Tab)
    app.processEvents()
    assert following.hasFocus()
    assert multiline.toPlainText() == ""


def test_lattice_solver_lock_survives_preset_changes(app):
    page = SolverPage()
    page.set_lattice_mode(True)
    fast_index = page.preset.findData("fast")
    page.preset.setCurrentIndex(fast_index)
    page.preset.activated.emit(fast_index)

    assert page.conc_solver.currentData() == "network_ext"
    assert page.flow_solver.currentData() == "spsolve"
    assert page.conc_solver.isEnabled()
    assert not page.flow_solver.isEnabled()
    assert page.cext_card.isEnabled()
    assert page.closure.currentData() == "wellmixed"
    assert not page.closure.isEnabled()
    assert "required for lattices" in page.solver_hint.label.text()


def test_concentration_solver_choices_hide_treecode_and_include_network_fft(app):
    page = SolverPage()
    values = [page.conc_solver.itemData(i) for i in range(page.conc_solver.count())]
    labels = [page.conc_solver.itemText(i) for i in range(page.conc_solver.count())]

    assert "topdown_ext_treecode" not in values
    assert "network_ext_hybrid_bg" in values
    assert values == [
        "network_ext",
        "network_ext_hybrid_bg",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "network",
        "topdown",
    ]
    assert labels == [
        "Network direct solver (best for large domains)",
        "Network FFT solver (best for primitive-shape domains)",
        "Top-down direct solver (for tree structures only)",
        "Top-down FFT solver (for tree structures only)",
        "Network, no vessel-vessel coupling (faster but less accurate)",
        "Top-down, no vessel-vessel coupling (faster but less accurate)",
    ]


def test_finite_radius_choices_use_backend_terms(app):
    page = SolverPage()
    values = {page.finite_radius.itemData(i) for i in range(page.finite_radius.count())}
    assert values == {"none", "monopole", "dipole", "both"}
    assert page.tissue_gl_order.value() == 5
    assert page.cext_gl_order.value() == 1
    assert page.axial_steps.value() == 5


def test_tissue_and_external_field_quadrature_orders_are_independent(app):
    page = SolverPage()
    config = default_project()
    config["settings"]["oxygen"]["gl_order"] = 3
    config["settings"]["oxygen"]["gl_order_cext"] = 7
    page.load(config)

    assert page.tissue_gl_order.value() == 3
    assert page.cext_gl_order.value() == 7
    page.tissue_gl_order.setValue(4)
    page.cext_gl_order.setValue(9)
    page.write(config)
    assert config["settings"]["oxygen"]["gl_order"] == 4
    assert config["settings"]["oxygen"]["gl_order_cext"] == 9


def test_lumen_closure_labels_are_experimentalist_facing(app):
    page = SolverPage()
    assert [page.closure.itemText(i) for i in range(page.closure.count())] == [
        "Resolved intralumen radial transport (recommended)",
        "Well mixed lumen",
        "Custom wall exchange expression",
    ]


def test_custom_kappa_only_appears_for_custom_wall_exchange(app):
    page = SolverPage()
    assert page.kappa_row.isHidden()

    page.closure.setCurrentIndex(page.closure.findData("custom_kappa"))
    assert not page.kappa_row.isHidden()
    assert page.kappa.isEnabled()

    page.closure.setCurrentIndex(page.closure.findData("graetz"))
    assert page.kappa_row.isHidden()


def test_viability_defaults_on_and_oxygen_kinetics_support_si_and_mmhg(app):
    page = PhysicsPage()
    config = default_project()
    page.load(config)

    assert page.viability_enabled.isChecked()
    assert page.viability.unit() == "mmHg"
    assert page.vmax.unit() == "mol/m³/s"
    assert page.km.unit() == "mol/m³"
    page.vmax.setUnit("mmHg/s")
    page.vmax.setValue(2.0)
    page.km.setUnit("mmHg")
    page.km.setValue(3.0)
    page.write(config)
    oxygen = config["settings"]["oxygen"]
    assert oxygen["vmax_mm"] == pytest.approx(oxygen_to_concentration(2.0, "mmHg/s"))
    assert oxygen["k_m_mm"] == pytest.approx(oxygen_to_concentration(3.0, "mmHg"))
    page.close()


def test_results_page_has_no_separate_full_viewer_action(app):
    page = AnalysisPage()
    assert page.open_folder_btn.text() == "Open results folder"
    assert not hasattr(page, "viewer_btn")
    assert not hasattr(page, "summary_table")
    assert page.vessel_limit.findData("none") >= 0
    assert page.tissue_mode.findData("none") >= 0
    assert page.vessel_scale.currentData() == "linear"
    assert page.tissue_scale.currentData() == "linear"


def test_outputs_page_has_no_orphaned_resource_banner(app):
    page = OutputsPage()
    assert not hasattr(page, "resource_banner")
    page.close()


def test_structured_grid_selection_writes_its_shape_and_total(app):
    page = OutputsPage()
    page.sample_mode.setCurrentIndex(page.sample_mode.findData("grid"))
    page.grid_x.setValue(64)
    page.grid_y.setValue(64)
    page.grid_z.setValue(64)
    config = default_project()
    page.write(config)

    assert config["simulation"]["sample_mode"] == "grid"
    assert config["simulation"]["tissue_grid"]["shape"] == [64, 64, 64]
    assert page.grid_total.text() == "64 × 64 × 64 = 262,144 tissue points"
    page.close()


def test_results_run_picker_includes_every_queued_simulation(app, tmp_path):
    page = AnalysisPage()
    page.runner = SimpleNamespace(
        jobs=[
            JobRecord("queued", "Queued case", "", str(tmp_path / "queued")),
            JobRecord("done", "Finished case", "", str(tmp_path / "done"), status="Completed"),
        ]
    )

    page.refresh()

    labels = [page.result_job.itemText(index) for index in range(page.result_job.count())]
    assert labels == ["Queued case  —  Queued", "Finished case  —  Completed"]
    assert "All completed runs" not in labels
    page.close()


def test_log_scale_replaces_a_zero_lower_bound_with_available_positive_data():
    normalized, limits = _normalize_with_scale(
        np.asarray([0.0, 1.0e-3, 1.0, 10.0]), 0.0, 10.0, "log"
    )
    assert normalized is not None
    assert limits is not None
    assert limits[0] > 0.0
    assert limits[1] == 10.0


def test_sphere_defaults_to_fast_surface_detail(app):
    page = DomainPage()
    assert [page.sphere_detail.itemData(i) for i in range(page.sphere_detail.count())] == [
        "12x8",
        "64x64",
    ]
    sphere_index = page.kind.findData("sphere")
    page.kind.setCurrentIndex(sphere_index)
    config = {}
    page.write(config)

    assert config["domain"]["theta_resolution"] == 12
    assert config["domain"]["phi_resolution"] == 8


def test_domain_uses_one_box_choice_and_loads_legacy_cubes(app):
    page = DomainPage()
    assert [page.kind.itemText(i) for i in range(page.kind.count())] == [
        "Box",
        "Sphere",
        "Upload mesh / .dmn",
    ]
    assert page.kind.currentData() == "box"
    assert (page.box_x.value(), page.box_y.value(), page.box_z.value()) == (
        1.0,
        1.0,
        1.0,
    )

    page.load({"domain": {"type": "cube", "side_length": 2.5}})
    assert page.kind.currentData() == "box"
    assert (page.box_x.value(), page.box_y.value(), page.box_z.value()) == (
        2.5,
        2.5,
        2.5,
    )

    config = {}
    page.write(config)
    assert config["domain"] == {
        "type": "box",
        "random_seed": 42,
        "side_length": 2.5,
        "x_length": 2.5,
        "y_length": 2.5,
        "z_length": 2.5,
    }


def test_box_dimensions_share_one_equal_three_column_row(app):
    page = DomainPage()
    page.resize(840, 600)
    page.show()
    app.processEvents()

    positions = [
        field.mapTo(page, QPoint())
        for field in (page.box_x, page.box_y, page.box_z)
    ]
    widths = [field.width() for field in (page.box_x, page.box_y, page.box_z)]
    assert len({position.y() for position in positions}) == 1
    assert positions[0].x() < positions[1].x() < positions[2].x()
    assert max(widths) - min(widths) <= 2
    page.close()


def test_tissue_points_are_random_or_structured_cartesian(app):
    page = OutputsPage()
    assert [page.sample_mode.itemData(i) for i in range(page.sample_mode.count())] == [
        "random",
        "grid",
    ]
    assert page.sample_mode.itemText(1) == "Structured Cartesian grid"

    page.sample_mode.setCurrentIndex(page.sample_mode.findData("grid"))
    page._sampling_changed()
    page.grid_x.setValue(12)
    page.grid_y.setValue(13)
    page.grid_z.setValue(14)
    config = {}
    page.write(config)

    assert page.sample_controls.currentWidget() is page.grid_fields
    assert config["simulation"]["sample_mode"] == "grid"
    assert config["simulation"]["tissue_grid"]["shape"] == [12, 13, 14]


def test_flow_sweep_parameter_and_values_can_be_edited(app):
    page = OutputsPage()
    page.resize(1100, 850)
    page.show()
    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    app.processEvents()

    parameter = page.sweep_rows[0]["combo"]
    assert isinstance(parameter, ChoiceComboBox)
    assert parameter.currentData() == "simulation.qin_target_ul_min"
    parameter.showPopup()
    app.processEvents()
    first = parameter._choice_list.item(0)
    QTest.mouseClick(
        parameter._choice_list.viewport(),
        Qt.LeftButton,
        pos=parameter._choice_list.visualItemRect(first).center(),
    )

    values = page.sweep_rows[0]["values"]
    assert isinstance(values, QLineEdit)
    values.setFocus()
    QTest.keyClicks(values, "10, 20, 30")
    app.processEvents()

    assert values.text() == "10, 20, 30"
    assert page.job_count.text() == "3 simulations"


def test_sweep_rows_have_direct_remove_buttons_and_compact_add_control(app):
    page = OutputsPage()
    page.resize(700, 800)
    page.show()
    assert page.add_sweep_btn.text() == "+"
    assert not {"Use", "Parameter", "Values"}.intersection(
        label.text() for label in page.sweep_rows_panel.findChildren(QLabel)
    )

    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    app.processEvents()
    first_path = page.sweep_rows[0]["combo"].currentData()
    second_remove = page.sweep_rows[1]["remove"]
    assert second_remove.text() == "×"
    plus_right = page.add_sweep_btn.mapTo(
        page, page.add_sweep_btn.rect().topRight()
    ).x()
    remove_right = second_remove.mapTo(page, second_remove.rect().topRight()).x()
    assert remove_right == plus_right

    QTest.mouseClick(second_remove, Qt.LeftButton)
    app.processEvents()

    assert len(page.sweep_rows) == 1
    assert page.sweep_rows[0]["combo"].currentData() == first_path


def test_sweep_units_convert_values_to_canonical_project_units(app):
    page = OutputsPage()
    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    record = page.sweep_rows[0]
    assert record["combo"].currentData() == "simulation.qin_target_ul_min"
    assert [record["unit"].itemText(i) for i in range(record["unit"].count())] == [
        "µL/min",
        "cm³/s",
        "mL/min",
        "m³/s",
    ]
    assert record["unit"].width() == 96

    record["unit"].setCurrentIndex(record["unit"].findText("cm³/s"))
    record["values"].setText("0.001, 0.002")
    config = {}
    page.write(config)

    sweep = config["gui"]["sweeps"][0]
    assert sweep["unit"] == "cm³/s"
    assert sweep["values"] == pytest.approx([60.0, 120.0])

    restored = OutputsPage()
    restored.load(config)
    restored_record = restored.sweep_rows[0]
    assert restored_record["unit"].currentText() == "cm³/s"
    assert restored_record["values"].text() == "0.001, 0.002"


def test_sweep_unit_change_preserves_physical_values(app):
    page = OutputsPage()
    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    record = page.sweep_rows[0]
    record["values"].setText("60000, 120000")

    record["unit"].setCurrentIndex(record["unit"].findText("cm³/s"))
    app.processEvents()

    assert record["values"].text() == "1, 2"


def test_combined_sweep_csv_controls_follow_sweep_and_summary_outputs(app):
    page = OutputsPage()
    page.load(default_project())
    assert not page.combined_sweep_csv.isEnabled()
    assert not page.combined_sweep_filename.isEnabled()

    QTest.mouseClick(page.add_sweep_btn, Qt.LeftButton)
    assert page.combined_sweep_csv.isEnabled()
    assert page.combined_sweep_filename.isEnabled()

    page.summary_csv.setChecked(False)
    assert not page.combined_sweep_csv.isChecked()
    assert not page.combined_sweep_filename.isEnabled()

    page.combined_sweep_csv.setChecked(True)
    assert page.summary_csv.isChecked()
    page.combined_sweep_filename.setText("oxygen_study.csv")
    config = {}
    page.write(config)
    assert config["outputs"]["write_combined_sweep_csv"] is True
    assert config["outputs"]["combined_sweep_filename"] == "oxygen_study.csv"


def test_runner_refreshes_combined_csv_when_a_job_finishes(app, tmp_path):
    config = default_project()
    config["outputs"]["out_dir"] = str(tmp_path / "results")
    config["gui"]["sweeps"] = [
        {
            "enabled": True,
            "path": "simulation.qin_target_ul_min",
            "unit": "µL/min",
            "values": [10, 20],
        }
    ]
    records = create_jobs(config, tmp_path)
    runner = JobRunner(tmp_path)
    runner.add(records)
    combined = Path(records[0].combined_csv_path)
    assert combined.is_file()

    output = Path(records[0].output_dir)
    output.mkdir(parents=True)
    (output / "summary.csv").write_text(
        "number_of_trees,target_terminals,total_volume_mean\n1,100,1.5\n",
        encoding="utf-8",
    )
    runner.current = records[0]
    records[0].status = "Running"
    runner._finished(0, QProcess.NormalExit)
    app.processEvents()

    with combined.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["run_status"] == "Completed"
    assert rows[0]["total_volume_mean"] == "1.5"
    assert rows[1]["run_status"] == "Queued"


def test_typing_sweep_values_is_safe_with_live_main_window_refresh(app):
    """Regression: a sweep-value edit must survive debounced live callbacks."""
    window = MainWindow()
    window.show()
    window.nav.setCurrentRow(5)
    QTest.qWait(250)
    outputs = window.pages[5]
    QTest.mouseClick(outputs.add_sweep_btn, Qt.LeftButton)
    assert not window._preview_timer.isActive()
    assert not window._status_timer.isActive()
    values = outputs.sweep_rows[0]["values"]
    values.setFocus()
    QTest.keyClicks(values, "10, 20, 30")
    QTest.qWait(3000)
    app.processEvents()

    assert values.text() == "10, 20, 30"
    assert outputs.job_count.text() == "3 simulations"
    window.close()


def test_multiple_sweep_dimensions_are_safe_with_live_refresh_and_expand(app, tmp_path):
    """Exercise the exact second-row interaction that previously crashed WSLg."""
    window = MainWindow()
    window.project_dir = tmp_path
    window.show()
    window.nav.setCurrentRow(5)
    outputs = window.pages[5]
    QTest.mouseClick(outputs.add_sweep_btn, Qt.LeftButton)
    QTest.mouseClick(outputs.add_sweep_btn, Qt.LeftButton)
    assert len(outputs.sweep_rows) == 2
    assert outputs.sweep_rows[0]["combo"].currentData() != outputs.sweep_rows[1]["combo"].currentData()

    first = outputs.sweep_rows[0]["values"]
    second = outputs.sweep_rows[1]["values"]
    first.setFocus()
    QTest.keyClicks(first, "10, 20")
    second.setFocus()
    QTest.keyClicks(second, "0.001, 0.002, 0.003")
    QTest.qWait(6500)
    app.processEvents()

    config = window._collect(show_error=False)
    assert config is not None
    assert outputs.job_count.text() == "6 simulations"
    assert len(create_jobs(config, tmp_path)) == 6
    window.close()


def test_preview_cap_is_distributed_across_inlets():
    left = np.column_stack((np.linspace(-1.0, -0.1, 4000), np.zeros(4000), np.zeros(4000)))
    right = np.column_stack((np.linspace(0.1, 1.0, 4000), np.zeros(4000), np.zeros(4000)))
    starts = np.vstack((left, right))
    ends = starts + np.array([0.001, 0.0, 0.0])
    shown_starts, shown_ends, _values, _alpha = limit_near_inlets(
        starts,
        ends,
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        limit=5000,
    )

    assert shown_starts.shape == shown_ends.shape == (5000, 3)
    assert np.count_nonzero(shown_starts[:, 0] < 0) == 2500
    assert np.count_nonzero(shown_starts[:, 0] > 0) == 2500


def test_domain_preview_supports_box_and_sphere():
    assert domain_wireframe({"type": "box", "x_length": 1, "y_length": 2, "z_length": 3}).shape == (12, 2, 3)
    sphere = domain_wireframe({"type": "sphere", "radius": 0.5})
    assert sphere.ndim == 3 and sphere.shape[1:] == (2, 3)
    assert sphere.shape[0] > 100


def test_svv_network_is_immediately_visible_and_uses_automatic_sphere_root(app):
    page = VesselsPage()
    assert not hasattr(page, "equal_bif")
    config = {
        "domain": {"type": "sphere", "radius": 0.5, "side_length": 1.0},
        "gui": {},
        "growth": {},
    }
    page.write(config)
    root = config["network"]["roots"][0]
    assert np.linalg.norm(root["start"]) < 0.5

    starts, ends, alpha, inlets = _svv_placeholder_geometry(
        config, config["network"]["roots"]
    )
    assert starts.shape == ends.shape == (31, 3)
    assert alpha.shape == (31,)
    assert inlets.shape == (1, 3)


def test_forest_automatically_creates_one_root_per_requested_tree(app):
    page = VesselsPage()
    page.topology.setCurrentIndex(page.topology.findData("forest"))
    page.inlet_count.setValue(3)
    config = {
        "domain": {"type": "sphere", "radius": 0.5, "side_length": 1.0},
        "gui": {},
        "growth": {},
    }
    page.write(config)

    roots = config["network"]["roots"]
    assert config["network"]["mode"] == "forest"
    assert len(roots) == 3
    assert page.configured_inlet_count() == 3
    assert all(np.linalg.norm(root["start"]) < 0.5 for root in roots)
    assert len({tuple(np.round(root["start"], 6)) for root in roots}) == 3


def test_physics_supports_distinct_conditions_for_each_forest_inlet(app):
    config = default_project()
    vessels = VesselsPage()
    vessels.topology.setCurrentIndex(vessels.topology.findData("forest"))
    vessels.inlet_count.setValue(2)
    vessels.write(config)

    physics = PhysicsPage()
    physics.load(config)
    physics.set_inlet_count(2)
    physics.use_inlet_conditions.setChecked(True)
    physics._inlet_widgets[0]["flow"].setValue(80.0)
    physics._inlet_widgets[1]["flow"].setValue(160.0)
    physics._inlet_widgets[0]["inlet_pressure"].setValue(480.0)
    physics._inlet_widgets[1]["inlet_pressure"].setValue(520.0)
    physics.write(config)

    parsed = parse_config(config)
    assert parsed.simulation.flow_source == "per_inlet"
    assert [item["flow_ul_min"] for item in parsed.simulation.inlet_conditions] == [80.0, 160.0]
    assert flow_for_tree(parsed, 0, 2) * 60.0 / 1e-3 == pytest.approx(80.0)
    assert flow_for_tree(parsed, 1, 2) * 60.0 / 1e-3 == pytest.approx(160.0)
    p0, _ = pressures_for_tree(parsed, 0)
    p1, _ = pressures_for_tree(parsed, 1)
    assert p0 == pytest.approx(480.0 * 133.322, rel=1e-5)
    assert p1 == pytest.approx(520.0 * 133.322, rel=1e-5)


def test_forest_preview_worker_builds_every_tree_from_gui_configuration(app, tmp_path):
    config = default_project()
    vessels = VesselsPage()
    vessels.topology.setCurrentIndex(vessels.topology.findData("forest"))
    vessels.inlet_count.setValue(2)
    vessels.terminals.setValue(2)
    vessels.write(config)

    physics = PhysicsPage()
    physics.load(config)
    physics.set_inlet_count(2)
    physics.use_inlet_conditions.setChecked(True)
    physics._inlet_widgets[0]["flow"].setValue(80.0)
    physics._inlet_widgets[1]["flow"].setValue(160.0)
    physics.write(config)

    request = tmp_path / "request.json"
    output = tmp_path / "preview"
    request.write_text(json.dumps(config), encoding="utf-8")
    assert preview_worker_main(["--request", str(request), "--output", str(output)]) == 0

    response = json.loads((output / "response.json").read_text(encoding="utf-8"))
    geometry = np.load(response["geometry_path"])
    assert response["trees"] == 2
    assert len(response["segments"]) == 2
    assert set(geometry["tree_ids"].tolist()) == {0, 1}
    assert response["root_radii_cm"][1] > response["root_radii_cm"][0]


def test_preview_line_width_tracks_vessel_radius(app):
    canvas = GeometryCanvas()
    canvas.set_geometry(
        vessel_starts=np.zeros((2, 3)),
        vessel_ends=np.ones((2, 3)),
        vessel_radii=np.asarray([0.01, 0.0025]),
    )
    widths = canvas._radius_widths()
    assert widths[0] > widths[1] > 0


def test_lattice_preview_contains_boundary_connections_and_absolute_radii(app):
    config = default_project()
    config["gui"]["network_source"] = "lattice"
    config["network"] = {
        "mode": "simple",
        "simple": {
            "mode": "lattice",
            "lattice_type": "cubic",
            "cells": 2,
            "radius_cm": 0.0005,
            "inlet_points_cm": [[-0.62, -0.19, -0.11]],
            "outlet_points_cm": [[0.63, 0.18, 0.12]],
        },
    }
    starts, ends, radii, inlets, outlets, _alpha, detail = network_geometry(config)
    assert len(starts) == channel_count(2, "cubic") + 2
    assert np.allclose(radii, 0.0005)
    assert np.allclose(inlets, [[-0.62, -0.19, -0.11]])
    assert np.allclose(outlets, [[0.63, 0.18, 0.12]])
    assert "2 boundary connections" in detail

    canvas = GeometryCanvas()
    canvas.resize(700, 700)
    boundary = domain_wireframe(config["domain"])
    canvas.set_geometry(
        domain_lines=boundary,
        vessel_starts=starts,
        vessel_ends=ends,
        vessel_radii=radii,
        inlet_points=inlets,
        outlet_points=outlets,
    )
    small_width = float(np.mean(canvas._radius_widths()))
    canvas.set_geometry(
        domain_lines=boundary,
        vessel_starts=starts,
        vessel_ends=ends,
        vessel_radii=radii * 10.0,
        inlet_points=inlets,
        outlet_points=outlets,
    )
    assert float(np.mean(canvas._radius_widths())) > small_width
    assert np.allclose(canvas.inlet_points, inlets)
    assert np.allclose(canvas.outlet_points, outlets)


def test_settings_inspector_is_wide_and_splitter_remains_draggable(app):
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.resize(1580, 900)
    window.show()
    app.processEvents()

    assert window.windowTitle() == "CASCADE O2 Simulation Studio"
    assert window.windowFlags() & Qt.FramelessWindowHint
    assert not hasattr(window.title_bar, "title")
    visible_labels = [label.text() for label in window.findChildren(QLabel)]
    assert "GFM / CASCADE" not in visible_labels
    assert "CASCADE O₂ Simulation Studio" not in visible_labels
    assert not any(text.lower().startswith("estimated host") for text in visible_labels)
    assert [action.text() for action in window.app_menu.actions()] == ["File", "Run"]
    assert all(handle.isVisible() for handle in window.resize_handles)
    assert window.title_bar.minimize_button.accessibleName() == "Minimize"
    assert window.title_bar.maximize_button.accessibleName() == "Maximize"
    assert window.title_bar.close_button.accessibleName() == "Close"
    assert window.pages[6].table.verticalHeader().isHidden()
    queue_header = window.pages[6].table.horizontalHeader()
    assert all(
        queue_header.sectionResizeMode(column) == QHeaderView.Interactive
        for column in range(window.pages[6].table.columnCount())
    )
    assert [window.pages[6].table.columnWidth(column) for column in range(5)] == [
        270,
        155,
        88,
        545,
        255,
    ]
    window.pages[6].table.setColumnWidth(0, 333)
    window.pages[6].refresh()
    assert window.pages[6].table.columnWidth(0) == 333
    assert window.pages_stack.width() >= 650
    initial_width = window.pages_stack.width()
    window.instrument_splitter.moveSplitter(540, 1)
    app.processEvents()
    assert window.pages_stack.width() > initial_width
    assert window.workspace_stack.currentIndex() == 0
    window.nav.setCurrentRow(1)
    app.processEvents()
    assert window.pages[1].horizontalScrollBar().maximum() == 0
    window.nav.setCurrentRow(6)
    app.processEvents()
    assert window.workspace_stack.currentIndex() == 1
    window.close()


def test_path_picker_uses_windows_native_bridge_in_wsl(app, monkeypatch):
    captured = {}

    def fake_windows_picker(**kwargs):
        captured.update(kwargs)
        return "/mnt/c/Users/carl/Downloads/domain.stl"

    monkeypatch.setattr(gui_widgets, "_running_in_wsl", lambda: True)
    monkeypatch.setattr(gui_widgets, "_windows_native_picker", fake_windows_picker)
    picker = PathPicker()
    picker._browse()
    assert picker.text() == "/mnt/c/Users/carl/Downloads/domain.stl"
    assert captured["mode"] == "file"


def test_non_wsl_path_picker_allows_platform_native_dialog(app, monkeypatch):
    captured = {}

    def fake_open(*args, **kwargs):
        captured.update(kwargs)
        return "", ""

    monkeypatch.setattr(gui_widgets, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_open)
    PathPicker()._browse()
    assert not (captured["options"] & QFileDialog.DontUseNativeDialog)


def test_qt_file_filters_are_translated_for_windows_explorer():
    converted = gui_widgets._qt_filter_to_windows(
        "Domain (*.dmn *.stl *.vtk);;All files (*)"
    )
    assert converted == "Domain|*.dmn;*.stl;*.vtk|All files|*.*"


def test_numeric_inputs_show_meaningful_digits_without_losing_precision(app):
    spin = NumberInput(significant_digits=6)
    spin.setDecimals(10)
    spin.setRange(-1.0e12, 1.0e12)

    examples = (
        (1.0, "1"),
        (100.0, "100"),
        (0.04, "0.04"),
        (0.0069, "0.0069"),
        (2.41e-5, "2.41e-5"),
        (300.02463, "300.025"),
    )
    for value, expected in examples:
        spin.setValue(value)
        assert spin.text() == expected
        assert spin.value() == pytest.approx(value)


def test_vtk_memory_guidance_is_quiet_overview_text(app):
    window = MainWindow()
    overview = window.pages[0]
    overview.load(default_project())

    assert overview.vtk_memory_note.text() == "VTK export can increase peak RAM."
    assert overview.vtk_memory_note.isVisibleTo(overview)

    without_vtk = default_project()
    without_vtk["outputs"]["write_paraview"] = False
    overview.load(without_vtk)
    assert not overview.vtk_memory_note.isVisible()
    window.close()


def test_queue_actions_have_semantic_styles_icons_and_states(app):
    page = QueuePage()
    runner = SimpleNamespace(running=False, jobs=[])
    page.runner = runner
    page.refresh()

    assert page.add_btn.property("queueRole") == "neutral"
    assert page.add_run_btn.property("queueRole") == "run"
    assert page.run_selected_btn.property("queueRole") == "run"
    assert page.run_all_btn.property("queueRole") == "run"
    assert page.cancel_btn.property("queueRole") == "stop"
    assert all(
        not button.icon().isNull()
        for button in (
            page.add_btn,
            page.add_run_btn,
            page.run_selected_btn,
            page.run_all_btn,
            page.cancel_btn,
        )
    )
    assert not page.run_selected_btn.isEnabled()
    assert not page.run_all_btn.isEnabled()
    assert not page.cancel_btn.isEnabled()
    assert not page.remove_btn.isEnabled()
    assert not page.clear_btn.isEnabled()

    runner.jobs = [
        JobRecord("queued", "Queued case", "settings.json", "results")
    ]
    page.refresh()
    assert page.run_all_btn.isEnabled()
    assert not page.run_selected_btn.isEnabled()

    page.table.selectRow(0)
    app.processEvents()
    assert page.run_selected_btn.isEnabled()
    assert page.remove_btn.isEnabled()
    assert not page.clear_btn.isEnabled()

    runner.jobs[0].status = "Completed"
    page.refresh()
    assert page.clear_btn.isEnabled()

    runner.running = True
    runner.current = runner.jobs[0]
    page._update_action_states()
    assert not page.run_selected_btn.isEnabled()
    assert not page.run_all_btn.isEnabled()
    assert page.cancel_btn.isEnabled()
    assert not page.remove_btn.isEnabled()
    page.close()


def test_queue_keeps_paths_diagnostics_and_progress_contextual(app, tmp_path):
    page = QueuePage()
    output_dir = tmp_path / "results" / "case-001"
    job = JobRecord(
        "running",
        "Perfusion baseline",
        str(tmp_path / "settings.json"),
        str(output_dir),
        status="Running",
        progress=42,
        stage="Solving tissue oxygen",
        started_at="2026-09-08T12:00:00+00:00",
    )
    page.runner = SimpleNamespace(running=True, current=job, jobs=[job])
    page.refresh()

    assert page.table.columnCount() == 5
    assert [
        page.table.horizontalHeaderItem(column).text()
        for column in range(page.table.columnCount())
    ] == ["Run", "Timestamp", "Runtime", "Status", "Results"]
    visible_text = " ".join(
        page.table.item(0, column).text()
        for column in range(3)
    )
    assert str(output_dir) not in visible_text
    assert page.table.cellWidget(0, 3).property("jobStatus") == "Running"
    progress = page.table.cellWidget(0, 3).findChild(QProgressBar)
    assert progress is not None
    assert progress.value() == 42
    assert progress.property("activeProgress") is True
    results = page.table.cellWidget(0, 4)
    assert results.text() == "Open results folder"
    assert results.toolTip() == str(output_dir)
    assert not hasattr(page, "overall")
    assert page.diagnostics_panel.isHidden()
    page.details_btn.setChecked(True)
    assert not page.diagnostics_panel.isHidden()
    page.runtime_timer.stop()
    page.close()


def test_cascade_message_boxes_use_frameless_dark_application_chrome(app):
    observed = []

    def dismiss_dialog():
        dialogs = [
            widget for widget in app.topLevelWidgets()
            if isinstance(widget, QDialog)
            and widget.objectName() == "cascadeMessageDialog"
        ]
        assert len(dialogs) == 1
        dialog = dialogs[0]
        observed.append(dialog.windowFlags())
        dialog.accept()

    QTimer.singleShot(0, dismiss_dialog)
    CascadeMessageBox.warning(None, "Setup issues", "Example validation issue")

    assert observed[0] & Qt.FramelessWindowHint


def test_workflow_rail_uses_one_icon_family_without_step_numbers(app):
    window = MainWindow()

    assert [window.nav.item(index).text() for index in range(window.nav.count())] == (
        window.PAGE_NAMES
    )
    assert all(
        not window.nav.item(index).icon().isNull()
        for index in range(window.nav.count())
    )
    assert not hasattr(window, "step_label")
    assert window.project_label.text() == "Unsaved project"
    assert window.project_label.toolTip() == ""
    assert str(window.project_dir) not in window.project_label.text()
    window.close()


def test_sidebar_brand_is_complete_and_visually_balanced(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    sidebar = window.findChild(QWidget, "sidebar")
    brand = window.findChild(QLabel, "brand")
    subtitle = window.findChild(QLabel, "brandSub")

    assert sidebar.width() == 216
    assert subtitle.text() == "O₂ SIMULATION STUDIO"
    assert subtitle.width() >= subtitle.fontMetrics().horizontalAdvance(subtitle.text())
    assert brand.width() >= brand.fontMetrics().horizontalAdvance(brand.text())
    assert abs(
        brand.fontMetrics().horizontalAdvance(brand.text())
        - subtitle.fontMetrics().horizontalAdvance(subtitle.text())
    ) <= 8
    assert brand.font().pixelSize() > subtitle.font().pixelSize()
    window.close()


def test_parameter_columns_align_controls_when_one_has_help_text(app):
    left = QLineEdit()
    right = QLineEdit()
    fields = row_of(
        labeled("Discharge hematocrit", left),
        labeled("Hemoglobin capacity", right, "Additional explanation."),
    )
    fields.resize(700, 120)
    fields.show()
    app.processEvents()

    assert left.mapTo(fields, QPoint()).y() == right.mapTo(fields, QPoint()).y()
    fields.close()


def test_viewport_vessel_is_mouse_selectable(app):
    canvas = GeometryCanvas()
    canvas.resize(500, 400)
    canvas.set_geometry(
        vessel_starts=np.asarray([[-1.0, 0.0, 0.0]]),
        vessel_ends=np.asarray([[1.0, 0.0, 0.0]]),
    )
    selected = []
    canvas.selection_changed.connect(selected.append)
    canvas.show()
    app.processEvents()
    QTest.mouseClick(canvas, Qt.LeftButton, pos=canvas.rect().center())
    app.processEvents()

    assert selected and selected[-1]["kind"] == "vessel"
    assert selected[-1]["index"] == 0


def test_viewport_orbit_stays_directional_after_crossing_over_the_top(app):
    canvas = GeometryCanvas()
    canvas.resize(500, 400)
    for vertical_rotation in (np.pi * 0.5, np.pi * 0.75):
        canvas._rotation = np.eye(3)
        canvas._orbit(0.0, vertical_rotation)

        # A point directly along the current viewing direction begins at
        # screen center. A rightward orbit must still move it right both at
        # and beyond the old Euler-angle pole; it must not spin the top view.
        forward_world = canvas._rotation.T @ np.asarray([0.0, 0.0, 1.0])
        before, _ = canvas._project(forward_world[None, :])
        canvas._orbit(0.20, 0.0)
        after, _ = canvas._project(forward_world[None, :])

        assert after[0, 0] > before[0, 0]
        assert np.allclose(
            canvas._rotation @ canvas._rotation.T, np.eye(3), atol=1e-12
        )


def test_uploaded_mesh_preview_uses_sparse_contours_not_triangle_soup():
    import pyvista as pv

    surface = pv.Sphere(theta_resolution=48, phi_resolution=48).triangulate()
    all_edges = surface.extract_all_edges().n_cells
    preview_lines = _mesh_wireframe(surface)
    assert 0 < len(preview_lines) <= 3000
    assert len(preview_lines) < all_edges


def test_box_result_uses_only_the_twelve_outer_edges(app, tmp_path):
    import pyvista as pv

    domain_path = tmp_path / "domain_boundary.vtp"
    pv.Cube(x_length=1.0, y_length=2.0, z_length=3.0).triangulate().save(domain_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "settings": {
                    "domain": {
                        "type": "box",
                        "x_length": 1.0,
                        "y_length": 2.0,
                        "z_length": 3.0,
                        "side_length": 3.0,
                    }
                },
                "outputs": {"domain_boundary_vtp": str(domain_path)},
            }
        ),
        encoding="utf-8",
    )

    preview = CasePreview()
    result = preview._load_result(str(manifest_path))
    assert result["domain_lines"].shape == (12, 2, 3)
    assert result["domain_triangles"].shape == (12, 3, 3)
    preview.close()


def test_domain_surface_mode_is_lightweight_and_switchable(app):
    surface = domain_surface_triangles({"type": "sphere", "radius": 0.5})
    assert surface.shape == (1152, 3, 3)

    preview = CasePreview()
    assert preview.title.text() == "LIVE CASE PREVIEW"
    home = preview.findChild(IconButton, "previewHome")
    assert home is not None and home.text() == "" and not home.icon().isNull()
    assert home.iconSize().width() == 20
    assert preview.domain_view.width() == 104
    assert "STYLE" in [label.text() for label in preview.findChildren(QLabel)]
    preview.canvas.set_geometry(
        domain_lines=domain_wireframe({"type": "sphere", "radius": 0.5}),
        domain_triangles=surface,
    )
    preview.domain_view.setCurrentIndex(1)
    app.processEvents()
    assert preview.domain_view.currentData() == "surface"
    assert preview.canvas.domain_mode == "surface"
    assert len(preview.canvas.domain_triangles) == 1152
    preview.show_case(default_project(), include_network=False)
    assert preview.status.isHidden()
    assert preview._status_slot.minimumHeight() == 28
    preview.close()


def test_preview_status_contains_only_scientific_counts():
    assert _count_status(101, 1, 505, 10_000) == (
        "101 vessels  │  1 tree  │  505 quadrature nodes  │  10,000 tissue points"
    )


def test_scalar_legend_ticks_follow_the_active_scale():
    assert np.allclose(
        _legend_tick_values((1.0, 16.0), "linear"),
        [1.0, 4.75, 8.5, 12.25, 16.0],
    )
    assert np.allclose(
        _legend_tick_values((1.0, 16.0), "log"), [1.0, 2.0, 4.0, 8.0, 16.0]
    )
    assert _layer_field_label("Vessels", "Flow Rate (μL/min)") == "Vessels: Flow Rate (μL/min)"
    assert _layer_field_label("Tissue", "Tissue concentration (mol/m³)") == "Tissue: Tissue concentration (mol/m³)"


def test_viewport_scale_bar_tracks_zoom(app):
    canvas = GeometryCanvas()
    canvas.resize(700, 500)
    canvas.set_geometry(domain_lines=domain_wireframe({"type": "box", "side_length": 1.0}))
    pixels_before, label_before = canvas._scale_bar_spec()
    canvas._zoom *= 2.0
    pixels_after, label_after = canvas._scale_bar_spec()

    assert 60 <= pixels_before <= 150
    assert 60 <= pixels_after <= 150
    assert label_after != label_before


def test_output_tissue_preview_caps_near_inlets_and_fades():
    config = default_project()
    config["simulation"]["distance_sample_count"] = 15_000
    inlet = np.asarray([[-0.5, 0.0, 0.0]])
    points, alpha, requested = preview_tissue_geometry(config, inlet)

    assert requested == 15_000
    assert len(points) == 10_000
    assert alpha.max() == pytest.approx(1.0)
    assert alpha.min() == pytest.approx(0.0)


def test_analysis_tissue_selection_supports_near_random_and_all():
    points = np.column_stack((np.arange(20_000), np.zeros((20_000, 2))))
    inlet = np.zeros((1, 3))
    near, near_alpha = select_tissue_points(points, inlet, mode="near")
    random, random_alpha = select_tissue_points(points, inlet, mode="random")
    all_points, all_alpha = select_tissue_points(points, inlet, mode="all")

    assert len(near) == len(random) == 10_000
    assert near[-1] == 9_999
    assert near_alpha[-1] == pytest.approx(0.0)
    assert len(np.unique(random)) == 10_000
    assert np.all(random_alpha == 1.0)
    assert len(all_points) == 20_000
    assert np.all(all_alpha == 1.0)


def test_seed_error_summary_identifies_domain_meshing():
    message = MainWindow._preview_failure_summary(
        "TetGen worker failed\nRuntimeError: Unknown exception"
    )
    assert message == "The uploaded domain could not be volume-meshed."


def test_uploaded_preview_domain_uses_direct_tetrahedral_mesh(tmp_path):
    import pyvista as pv

    surface_path = tmp_path / "preview-domain.stl"
    pv.Sphere(theta_resolution=16, phi_resolution=16).save(surface_path)
    raw = default_project()
    raw["domain"] = {"type": "file", "path": str(surface_path), "random_seed": 42}
    config = parse_config(raw)

    domain = _fast_file_preview_domain(config)
    assert domain is not None
    assert domain.mesh.n_cells > 0
    assert len(domain.patches) == 0
    assert bool(domain.within(np.asarray([[0.0, 0.0, 0.0]]))[0])
    assert not bool(domain.within(np.asarray([[2.0, 2.0, 2.0]]))[0])


def test_abs_radius_expression_and_zero_guidance():
    lattice = generate_lattice(
        2,
        (1.0, 1.0, 1.0),
        0.01,
        lattice_type="cubic",
        radius_expression="abs(r0 * (1 + 0.25*x/L))",
    )
    assert (lattice["segment_radii_cm"] > 0).all()

    with pytest.raises(ValueError, match=r"abs\(0\).+small positive floor"):
        generate_lattice(
            2,
            (1.0, 1.0, 1.0),
            0.01,
            lattice_type="cubic",
            radius_expression="abs(r0 * (1 - 2*x/L))",
        )
