from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil
import pytest
import pyvista as pv
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QWidget

from cascade.gui import model as gui_model
from cascade.gui import viewer as viewer_module
from cascade.gui import widgets as gui_widgets
from cascade.gui.model import (
    PROJECT_FILENAME,
    JobRecord,
    QueueStore,
    create_jobs,
    default_project,
    load_project,
)
from cascade.gui.runner import JobRunner
from cascade.gui.window import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    instance = QApplication.instance()
    if instance is not None:
        assert isinstance(instance, QApplication)
        return instance
    return QApplication([])


def _wait_until(qapp: QApplication, predicate, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return bool(predicate())


def _write_network(root: Path) -> Path:
    network = root / "vascular network Ω.csv"
    network.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z,radius_cm\n"
        "-0.4,0,0,0.4,0,0,0.02\n",
        encoding="utf-8",
    )
    return network


def _quick_config(root: Path, run_name: str) -> dict:
    network = _write_network(root)
    config = default_project()
    config["domain"] = {
        "type": "cube",
        "side_length": 1.0,
        "random_seed": 42,
    }
    config["network"] = {
        "mode": "simple",
        "simple": {"mode": "custom", "path": str(network)},
    }
    config.setdefault("growth", {})["enabled"] = False
    config["simulation"].update(
        {
            "fluid": "water",
            "qin_target_ul_min": 2.0,
            "concentration_solver": "topdown",
            "distance_sample_count": 0,
            "tissue_accel": "cpu",
            "geometry_only": True,
        }
    )
    config["settings"]["hematocrit"]["flow_iterations"] = 0
    config["settings"]["cext"]["accel_mode"] = "cpu"
    config["settings"]["tissue"]["accel_mode"] = "cpu"
    config["outputs"].update(
        {
            "write_paraview": True,
            "write_summary_csv": True,
            "write_segments_csv": False,
            "write_points_csv": False,
            "save_network": False,
        }
    )
    config["gui"].update(
        {
            "network_source": "custom",
            "project_name": "Studio lifecycle Ω",
            "run_name": run_name,
        }
    )
    return config


def _generated_config(run_name: str) -> dict:
    config = default_project()
    config["network"]["target_terminal_count"] = 1000
    config["simulation"].update(
        {
            "distance_sample_count": 0,
            "tissue_accel": "cpu",
            "geometry_only": True,
        }
    )
    config["settings"]["hematocrit"]["flow_iterations"] = 0
    config["settings"]["cext"]["accel_mode"] = "cpu"
    config["settings"]["tissue"]["accel_mode"] = "cpu"
    config["outputs"].update(
        {
            "write_paraview": False,
            "write_summary_csv": False,
            "save_network": False,
        }
    )
    config["gui"].update(
        {
            "network_source": "svv_generated",
            "project_name": "Cancellation lifecycle",
            "run_name": run_name,
        }
    )
    return config


def _assert_process_stopped(pid: int) -> bool:
    return not pid or not psutil.pid_exists(pid)


def test_main_window_constructs_pages_and_round_trips_project(
    qapp: QApplication, tmp_path: Path
) -> None:
    project = tmp_path / "Project With Spaces 血管"
    project.mkdir()
    config = _quick_config(project, "Round trip")
    window = MainWindow()
    try:
        assert window.nav.count() == 8
        assert len(window.pages) == 8
        assert all(page.findChildren(QWidget) for page in window.pages)

        window.project_dir = project
        window.project_path = project / PROJECT_FILENAME
        window.config = config
        window._load_pages()
        assert "Not saved yet" in window.project_label.text()
        window.save()

        assert window.project_path.is_file()
        assert "Last saved" in window.project_label.text()
        restored = load_project(window.project_path)
        assert restored["gui"]["project_name"] == "Studio lifecycle Ω"
        assert not Path(restored["network"]["simple"]["path"]).is_absolute()

        window.config = restored
        window._load_pages()
        collected = window._collect(show_error=False)
        assert collected is not None
        assert collected["gui"]["project_name"] == "Studio lifecycle Ω"
        assert collected["network"]["simple"]["mode"] == "custom"
    finally:
        window._prepare_timer.stop()
        window.runner.shutdown()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_interrupted_queue_recovery_and_windows_openers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "recovery"
    running = JobRecord(
        id="interrupted-job",
        name="Interrupted job",
        settings_path=str(project / "settings.json"),
        output_dir=str(project / "runs" / "interrupted-job"),
        status="Running",
        stage="Solving oxygen transport",
    )
    store = QueueStore(project)
    store.save([running])

    recovered = QueueStore(project).load()
    assert len(recovered) == 1
    assert recovered[0].status == "Interrupted"
    assert "Previous session ended" in recovered[0].stage

    opened: list[str] = []
    monkeypatch.setattr(gui_model, "sys_platform", lambda: "win32")
    monkeypatch.setattr(
        gui_model.os,
        "startfile",
        lambda path: opened.append(str(path)),
        raising=False,
    )
    folder = tmp_path / "Folder With Spaces Ω"
    gui_model.open_folder(folder)
    result = folder / "result.csv"
    result.write_text("value\n1\n", encoding="utf-8")
    gui_model.open_path(result)

    assert opened == [str(folder.resolve()), str(result.resolve())]


def test_native_windows_dialogs_use_qt_without_wsl_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    selected = {
        "directory": str(tmp_path / "chosen folder"),
        "open": str(tmp_path / "chosen input.stl"),
        "save": str(tmp_path / "chosen project.json"),
    }
    monkeypatch.setattr(gui_widgets, "_running_in_wsl", lambda: False)
    monkeypatch.setattr(
        gui_widgets.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: selected["directory"],
    )
    monkeypatch.setattr(
        gui_widgets.QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (selected["open"], "Surface (*.stl)"),
    )
    monkeypatch.setattr(
        gui_widgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (selected["save"], "JSON (*.json)"),
    )

    assert gui_widgets.choose_native_path(
        None,
        mode="directory",
        caption="Choose folder",
        start=str(tmp_path),
    ) == selected["directory"]
    assert gui_widgets.choose_native_path(
        None,
        mode="open",
        caption="Open surface",
        start=str(tmp_path),
        file_filter="Surface (*.stl)",
    ) == selected["open"]
    assert gui_widgets.choose_native_path(
        None,
        mode="save",
        caption="Save project",
        start=str(tmp_path),
        file_filter="JSON (*.json)",
    ) == selected["save"]


def test_opengl_canvas_factory_imports_renderer_from_gui_package(
    monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    from cascade.gui.gpu_preview import OpenGLGeometryCanvas
    from cascade.gui.visualization import support

    class _NativePlatform:
        @staticmethod
        def platformName() -> str:
            return "windows"

    monkeypatch.setattr(support, "QApplication", _NativePlatform)
    monkeypatch.setattr(support, "_opengl_33_available", lambda: True)
    monkeypatch.setenv("CASCADE_RENDER_BACKEND", "opengl")
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    parent = QWidget()
    canvas = support._create_geometry_canvas(parent)
    assert isinstance(canvas, OpenGLGeometryCanvas)
    parent.close()
    parent.deleteLater()
    qapp.processEvents()


def test_window_state_transitions_and_title_bar_controls(qapp: QApplication) -> None:
    window = MainWindow()
    try:
        window.show()
        assert _wait_until(qapp, window.isVisible, 5.0)
        window.resize(1280, 760)
        qapp.processEvents()
        assert window.width() == 1280
        assert window.height() == 760

        window.title_bar.toggle_maximized()
        assert _wait_until(qapp, window.isMaximized, 5.0)
        window.title_bar.toggle_maximized()
        assert _wait_until(qapp, lambda: not window.isMaximized(), 5.0)

        window.showMinimized()
        assert _wait_until(qapp, window.isMinimized, 5.0)
        window.showNormal()
        assert _wait_until(qapp, lambda: not window.isMinimized(), 5.0)
    finally:
        window._prepare_timer.stop()
        window.runner.shutdown()
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _pe_subsystem(path: Path) -> int:
    data = path.read_bytes()
    pe_offset = int.from_bytes(data[0x3C:0x40], "little")
    assert data[pe_offset : pe_offset + 4] == b"PE\0\0"
    optional_header = pe_offset + 24
    return int.from_bytes(
        data[optional_header + 68 : optional_header + 70], "little"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows PE launcher subsystems")
def test_normal_launchers_are_gui_subsystem_and_diagnostics_keep_console() -> None:
    scripts = Path(sys.executable).resolve().parent
    assert _pe_subsystem(scripts / "cascade-gui.exe") == 2
    assert _pe_subsystem(scripts / "cascade-viewer.exe") == 2
    assert _pe_subsystem(scripts / "cascade-gui-console.exe") == 3
    assert _pe_subsystem(scripts / "cascade-viewer-console.exe") == 3


def test_result_viewer_loads_exported_arrays_without_source_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vessels = pv.Line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), resolution=1)
    vessels.point_data["flow_ul_min"] = np.asarray([1.0, 2.0])
    vessels.point_data["concentration"] = np.asarray([2.0, 4.0])
    vessels_path = tmp_path / "vessels.vtp"
    vessels.save(vessels_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "outputs": {"vessels_vtp": "vessels.vtp"},
                "settings": {
                    "simulation": {"qin_target_ul_min": 2.0},
                    "settings": {
                        "oxygen": {"conc_max_for_normalization": 4.0}
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class _Plotter:
        def __init__(self, **_kwargs):
            self.background = None

        def set_background(self, value):
            self.background = value

    monkeypatch.setattr(viewer_module.pv, "Plotter", _Plotter)
    viewer = viewer_module.ResultViewer(manifest)
    assert viewer.vessels is not None
    assert np.allclose(
        viewer.vessels.point_data["flow / total inlet flow"], [0.5, 1.0]
    )
    assert np.allclose(
        viewer.vessels.point_data["oxygen / inlet oxygen"], [0.5, 1.0]
    )


@pytest.mark.skipif(os.name != "nt", reason="native Windows QProcess lifecycle")
def test_persistent_worker_completes_reuses_restarts_and_shuts_down(
    qapp: QApplication, tmp_path: Path
) -> None:
    project = tmp_path / "Persistent Worker Ω"
    project.mkdir()
    runner = JobRunner(project)
    first_pid = 0
    restarted_pid = 0
    try:
        first = create_jobs(_quick_config(project, "First"), project)[0]
        runner.add([first])
        runner.run([first.id])
        assert _wait_until(qapp, lambda: first.status == "Completed")
        assert Path(first.manifest_path or "").is_file()
        assert runner.process is not None
        assert runner.process.state() == QProcess.Running
        assert runner._worker_ready
        first_pid = int(runner.process.processId())

        second = create_jobs(_quick_config(project, "Second"), project)[0]
        runner.add([second])
        runner.run([second.id])
        assert _wait_until(qapp, lambda: second.status == "Completed")
        assert Path(second.manifest_path or "").is_file()
        assert runner.process is not None
        assert int(runner.process.processId()) == first_pid

        runner.shutdown()
        assert _wait_until(qapp, lambda: _assert_process_stopped(first_pid), 15.0)

        runner.warmup()
        assert _wait_until(
            qapp,
            lambda: runner.process is not None and runner._worker_ready,
            30.0,
        )
        assert runner.process is not None
        restarted_pid = int(runner.process.processId())
        assert restarted_pid != first_pid
    finally:
        runner.shutdown()
        qapp.processEvents()
    assert _wait_until(qapp, lambda: _assert_process_stopped(restarted_pid), 15.0)


@pytest.mark.skipif(os.name != "nt", reason="native Windows QProcess cancellation")
def test_worker_cancellation_allows_followup_job(
    qapp: QApplication, tmp_path: Path
) -> None:
    project = tmp_path / "Cancel And Recover"
    project.mkdir()
    runner = JobRunner(project)
    cancelled_pid = 0
    try:
        cancelled = create_jobs(_generated_config("Cancel me"), project)[0]
        runner.add([cancelled])
        runner.run([cancelled.id])
        assert _wait_until(
            qapp,
            lambda: (
                runner.current is cancelled
                and runner.process is not None
                and runner.process.state() == QProcess.Running
                and (
                    runner._preparing_id is not None or runner._job_dispatched
                )
            ),
            30.0,
        )
        assert runner.process is not None
        cancelled_pid = int(runner.process.processId())
        runner.cancel_current()
        assert _wait_until(
            qapp,
            lambda: cancelled.status == "Cancelled" and runner.current is None,
            20.0,
        )
        assert _wait_until(
            qapp, lambda: _assert_process_stopped(cancelled_pid), 15.0
        )

        recovery = create_jobs(_quick_config(project, "Recovery"), project)[0]
        runner.add([recovery])
        runner.run([recovery.id])
        assert _wait_until(qapp, lambda: recovery.status == "Completed")
        assert Path(recovery.manifest_path or "").is_file()
    finally:
        runner.shutdown()
        qapp.processEvents()


@pytest.mark.skipif(os.name != "nt", reason="native Windows preview QProcess")
def test_preview_worker_completes_cancels_and_recovers(
    qapp: QApplication, tmp_path: Path
) -> None:
    project = tmp_path / "Preview Worker Ω"
    project.mkdir()
    window = MainWindow()
    cancelled_pid = 0
    try:
        window.project_dir = project
        window.nav.setCurrentRow(6)
        base = _quick_config(project, "Preview")
        window.config = deepcopy(base)
        window._load_pages()

        first_signature = window._case_preview_signature(base)
        window._start_preview_seed(base, first_signature)
        assert _wait_until(
            qapp,
            lambda: (
                window._preview_process_job is not None
                and window._preview_process_job.active
            ),
            20.0,
        )
        assert _wait_until(
            qapp,
            lambda: (
                window._preview_process is None
                and window._preview_seed_signature == first_signature
            ),
            60.0,
        )
        window._prepare_timer.stop()
        assert Path(window._preview_seed_response["geometry_path"]).is_file()

        cancel_config = deepcopy(base)
        cancel_config["domain"]["random_seed"] = 43
        cancel_signature = window._case_preview_signature(cancel_config)
        window._start_preview_seed(cancel_config, cancel_signature)
        assert _wait_until(
            qapp,
            lambda: (
                window._preview_process is not None
                and window._preview_process_job is not None
                and window._preview_process_job.active
            ),
            20.0,
        )
        assert window._preview_process is not None
        cancelled_pid = int(window._preview_process.processId())
        window._cancel_preview_seed()
        assert window._preview_process is None
        assert _wait_until(
            qapp, lambda: _assert_process_stopped(cancelled_pid), 15.0
        )

        recovery_config = deepcopy(base)
        recovery_config["domain"]["random_seed"] = 44
        recovery_signature = window._case_preview_signature(recovery_config)
        window._start_preview_seed(recovery_config, recovery_signature)
        assert _wait_until(
            qapp,
            lambda: (
                window._preview_process is None
                and window._preview_seed_signature == recovery_signature
            ),
            60.0,
        )
        window._prepare_timer.stop()
        assert Path(window._preview_seed_response["geometry_path"]).is_file()
    finally:
        window._prepare_timer.stop()
        window._cancel_preview_seed()
        window.runner.shutdown()
        window.close()
        window.deleteLater()
        qapp.processEvents()
