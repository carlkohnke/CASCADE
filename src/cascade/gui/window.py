"""Main-window shell, title bar, and GUI state coordination.

The window synchronizes page models, preview workers, job queues, and result
navigation while leaving scientific calculations to the simulation services.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from PySide6.QtCore import (
    QEvent,
    QProcess,
    QSize,
    QTimer,
    Qt,
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenuBar,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import (
    create_jobs,
    default_project,
    estimate_resources,
    hardware_info,
    load_project,
    save_project,
    validate_project,
)
from cascade.gui.preview import (
    CasePreview,
    FlowBackdrop,
)
from cascade.gui.runner import JobRunner
from cascade.gui.theme import Tokens
from cascade.gui.widgets import (
    StatusPill,
    cancel_native_pickers,
    choose_native_path,
)
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from cascade.gui.ui_helpers import (
    QMessageBox,
    _default_project_directory,
    _workflow_icon,
)

from cascade.gui.pages.analysis import (
    AnalysisPage,
)

from cascade.gui.pages.vessels import (
    VesselsPage,
)

from cascade.gui.pages.queue import (
    QueuePage,
)

from cascade.gui.pages.domain import (
    DomainPage,
)

from cascade.gui.pages.physics import (
    PhysicsPage,
)

from cascade.gui.pages.solver import (
    SolverPage,
)

from cascade.gui.pages.outputs import (
    OutputsPage,
)

from cascade.gui.pages.overview import (
    OverviewPage,
)


class WindowResizeHandle(QWidget):
    """Narrow frameless-window edge that delegates resizing to the compositor."""

    def __init__(self, window: QMainWindow, edge, *, horizontal: bool):
        super().__init__(window)
        self.host_window = window
        self.edge = edge
        self.horizontal = horizontal
        if horizontal:
            self.setFixedHeight(4)
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setFixedWidth(4)
            self.setCursor(Qt.SizeHorCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            edges = self.edge
            if self.horizontal:
                if event.position().x() <= 10:
                    edges |= Qt.LeftEdge
                elif event.position().x() >= self.width() - 10:
                    edges |= Qt.RightEdge
            handle = self.host_window.windowHandle()
            if handle is not None and handle.startSystemResize(edges):
                event.accept()
                return
        super().mousePressEvent(event)


class WindowTitleBar(QFrame):
    """Dark client-side chrome with native compositor move behavior."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.host_window = window
        self.setObjectName("windowTitleBar")
        self.setFixedHeight(30)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addStretch(1)

        self.minimize_button = self._control("−", "Minimize")
        self.maximize_button = self._control("□", "Maximize")
        self.close_button = self._control("×", "Close", close=True)
        self.minimize_button.clicked.connect(window.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(window.close)
        row.addWidget(self.minimize_button)
        row.addWidget(self.maximize_button)
        row.addWidget(self.close_button)

    @staticmethod
    def _control(
        text: str, accessible_name: str, *, close: bool = False
    ) -> QPushButton:
        button = QPushButton(text)
        button.setFixedSize(44, 30)
        button.setFocusPolicy(Qt.NoFocus)
        button.setProperty("windowControl", "close" if close else "standard")
        button.setAccessibleName(accessible_name)
        return button

    def toggle_maximized(self) -> None:
        if self.host_window.isMaximized():
            self.host_window.showNormal()
        else:
            self.host_window.showMaximized()
        self.host_window._sync_window_chrome()

    def sync_state(self) -> None:
        maximized = self.host_window.isMaximized()
        self.maximize_button.setText("❐" if maximized else "□")
        self.maximize_button.setAccessibleName("Restore" if maximized else "Maximize")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            handle = self.host_window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class MainWindow(QMainWindow):
    PAGE_NAMES = [
        "Project",
        "Domain",
        "Network",
        "Physics",
        "Solver",
        "Outputs",
        "Run",
        "Results",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self._initializing = True
        self.setWindowTitle("CASCADE O2 Simulation Studio")
        self.resize(1580, 940)
        self.setMinimumSize(1180, 720)
        self.config = default_project()
        self.project_path: Path | None = None
        self.project_dir = _default_project_directory()
        self.hardware = hardware_info()
        self.runner = JobRunner(self.project_dir, self)
        self._preview_process: QProcess | None = None
        self._preview_seed_signature = ""
        self._preview_seed_response: dict[str, Any] = {}
        self._preview_pending_signature = ""
        self._preview_failed_signature = ""
        self._preview_failure_message = ""
        self._preview_output_buffer = ""
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(180)
        self._status_timer.timeout.connect(self._refresh_status)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(90)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._build_ui()
        self._build_menu()
        self._load_pages()
        self._initializing = False
        self._refresh_status()

    def _build_ui(self):
        self.title_bar = WindowTitleBar(self)
        self.app_menu = QMenuBar()
        self.app_menu.setObjectName("appMenu")
        chrome = QFrame()
        chrome.setObjectName("windowChrome")
        chrome_layout = QVBoxLayout(chrome)
        chrome_layout.setContentsMargins(0, 0, 0, 0)
        chrome_layout.setSpacing(0)
        chrome_layout.addWidget(self.title_bar)
        chrome_layout.addWidget(self.app_menu)

        root = FlowBackdrop()
        frame = QGridLayout(root)
        frame.setContentsMargins(0, 0, 0, 0)
        frame.setSpacing(0)
        top_resize = WindowResizeHandle(self, Qt.TopEdge, horizontal=True)
        bottom_resize = WindowResizeHandle(self, Qt.BottomEdge, horizontal=True)
        left_resize = WindowResizeHandle(self, Qt.LeftEdge, horizontal=False)
        right_resize = WindowResizeHandle(self, Qt.RightEdge, horizontal=False)
        self.resize_handles = (
            top_resize,
            bottom_resize,
            left_resize,
            right_resize,
        )
        frame.addWidget(top_resize, 0, 0, 1, 3)
        frame.addWidget(left_resize, 1, 0)
        frame.addWidget(right_resize, 1, 2)
        frame.addWidget(bottom_resize, 2, 0, 1, 3)
        surface = QWidget()
        surface.setObjectName("windowSurface")
        frame.addWidget(surface, 1, 1)
        shell = QVBoxLayout(surface)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        shell.addWidget(chrome)
        work_area = QWidget()
        shell.addWidget(work_area, 1)
        self.backdrop = root
        outer = QHBoxLayout(work_area)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(216)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(14, 20, 14, 14)
        brand = QLabel("CASCADE")
        brand.setObjectName("brand")
        sub = QLabel("O₂ SIMULATION STUDIO")
        sub.setObjectName("brandSub")
        side.addWidget(brand)
        side.addWidget(sub)
        side.addSpacing(18)
        self.nav = QListWidget()
        self.nav.setObjectName("navigation")
        self.nav.setIconSize(QSize(28, 28))
        stage_icons = (
            "project",
            "domain",
            "network",
            "physics",
            "solver",
            "outputs",
            "run",
            "results",
        )
        for icon_name, name in zip(stage_icons, self.PAGE_NAMES):
            item = QListWidgetItem(_workflow_icon(icon_name), name)
            item.setSizeHint(QSize(0, 54))
            self.nav.addItem(item)
        side.addWidget(self.nav, 1)
        outer.addWidget(sidebar)

        content = QWidget()
        content_col = QVBoxLayout(content)
        content_col.setContentsMargins(0, 0, 0, 0)
        content_col.setSpacing(0)
        top = QFrame()
        top.setObjectName("topbar")
        top_row = QHBoxLayout(top)
        top_row.setContentsMargins(16, 8, 16, 8)
        self.project_label = QLabel("Unsaved project")
        self.project_label.setStyleSheet(f"font-weight:650;color:{Tokens.TEXT_2};")
        self.validation_pill = StatusPill("Checking…")
        self.validation_pill.setToolTip("Click to review setup issues and warnings.")
        self.validation_pill.clicked.connect(self._show_validation_details)
        save = QPushButton("Save project")
        save.setProperty("secondary", True)
        save.clicked.connect(self.save)
        top_row.addWidget(self.project_label)
        top_row.addStretch()
        top_row.addWidget(self.validation_pill)
        top_row.addWidget(save)
        content_col.addWidget(top)

        self.pages = [
            OverviewPage(self.hardware),
            DomainPage(),
            VesselsPage(),
            PhysicsPage(),
            SolverPage(),
            OutputsPage(),
            QueuePage(),
            AnalysisPage(),
        ]

        # The model is the primary workspace.  Setup and result controls live in
        # a narrow contextual inspector rather than competing with the viewport.
        self.workspace_stack = QStackedWidget()
        instrument = QWidget()
        instrument_layout = QHBoxLayout(instrument)
        instrument_layout.setContentsMargins(0, 0, 0, 0)
        instrument_layout.setSpacing(0)
        splitter = QSplitter(Qt.Horizontal)
        self.instrument_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        self.preview = CasePreview()
        splitter.addWidget(self.preview)
        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(470)
        inspector.setMaximumWidth(1000)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        self.pages_stack = QStackedWidget()
        self._inspector_page_index = {}
        for page_index in (0, 1, 2, 3, 4, 5, 7):
            self._inspector_page_index[page_index] = self.pages_stack.addWidget(
                self.pages[page_index]
            )
        inspector_layout.addWidget(self.pages_stack)
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([680, 720])
        instrument_layout.addWidget(splitter)
        self.workspace_stack.addWidget(instrument)
        self.workspace_stack.addWidget(self.pages[6])
        content_col.addWidget(self.workspace_stack, 1)

        bottom = QFrame()
        bottom.setObjectName("solverStatus")
        bottom_row = QHBoxLayout(bottom)
        bottom_row.setContentsMargins(16, 7, 16, 7)
        bottom_row.setSpacing(18)

        def status_group(key: str, value: str) -> tuple[QWidget, QLabel]:
            group = QWidget()
            column = QVBoxLayout(group)
            column.setContentsMargins(0, 0, 0, 0)
            column.setSpacing(1)
            key_label = QLabel(key)
            key_label.setObjectName("statusKey")
            value_label = QLabel(value)
            value_label.setObjectName("statusValue")
            column.addWidget(key_label)
            column.addWidget(value_label)
            return group, value_label

        solve_group, self.status_solver = status_group("ACTIVE RUN", "Idle")
        bottom_row.addWidget(solve_group)
        bottom_row.addStretch()
        self.back_btn = QPushButton("Back")
        self.back_btn.setProperty("secondary", True)
        self.next_btn = QPushButton("Next")
        bottom_row.addWidget(self.back_btn)
        bottom_row.addWidget(self.next_btn)
        content_col.addWidget(bottom)
        outer.addWidget(content, 1)
        self.setCentralWidget(root)
        self.nav.currentRowChanged.connect(self._page_changed)
        self.nav.setCurrentRow(0)
        self.back_btn.clicked.connect(
            lambda: self.nav.setCurrentRow(max(0, self.nav.currentRow() - 1))
        )
        self.next_btn.clicked.connect(
            lambda: self.nav.setCurrentRow(
                min(len(self.pages) - 1, self.nav.currentRow() + 1)
            )
        )
        queue: QueuePage = self.pages[6]
        analysis: AnalysisPage = self.pages[7]
        queue.set_runner(self.runner)
        analysis.set_runner(self.runner)
        analysis.render_requested.connect(self._render_analysis)
        self.preview.view_settings_changed.connect(self._viewer_settings_changed)
        self.preview.result_fields_loaded.connect(analysis.set_render_fields)
        self.preview.selection_changed.connect(analysis.set_selection)
        self.runner.running_changed.connect(self._preview_running_changed)
        self.runner.jobs_changed.connect(self._update_solver_status)
        queue.add_requested.connect(self._enqueue)
        self.pages[2].open_physics_requested.connect(lambda: self.nav.setCurrentRow(3))
        self.pages[2].source.currentIndexChanged.connect(self._network_source_changed)
        self.pages[2].topology.currentIndexChanged.connect(
            self._sync_inlet_condition_count
        )
        self.pages[2].inlet_count.valueChanged.connect(self._sync_inlet_condition_count)
        self.pages[2].auto_roots.toggled.connect(self._sync_inlet_condition_count)
        self.pages[2].roots.textChanged.connect(self._sync_inlet_condition_count)
        self.pages[3].changed.connect(self._schedule_status_refresh)
        self.pages[5].preview_changed.connect(self._schedule_output_preview_refresh)
        self._wire_live_validation()

    def _build_menu(self):
        file_menu = self.app_menu.addMenu("File")
        actions = [
            ("New project", self.new),
            ("Open project…", self.open),
            ("Save", self.save),
            ("Save as…", self.save_as),
        ]
        for text, slot in actions:
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        run_menu = self.app_menu.addMenu("Run")
        add = QAction("Add current setup to queue", self)
        add.triggered.connect(lambda: self._enqueue(False))
        run_menu.addAction(add)
        run_all = QAction("Run queued simulations", self)
        run_all.triggered.connect(lambda: self.runner.run())
        run_menu.addAction(run_all)

    def _wire_live_validation(self):
        for page in self.pages[:6]:
            for widget in page.findChildren(QComboBox):
                widget.currentIndexChanged.connect(self._schedule_status_refresh)
            for spin_type in (QSpinBox, QDoubleSpinBox):
                for widget in page.findChildren(spin_type):
                    widget.valueChanged.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QCheckBox):
                widget.toggled.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QLineEdit):
                widget.textChanged.connect(self._schedule_status_refresh)
            for widget in page.findChildren(QPlainTextEdit):
                widget.textChanged.connect(self._schedule_status_refresh)

    def _schedule_status_refresh(self, *_):
        if not self._initializing:
            self._status_timer.start()
            # Outputs edits do not alter geometry. Avoid re-entering the WSLg
            # canvas while QTableWidget is changing its cell widgets.
            source = self.sender()
            outputs = self.pages[5] if len(self.pages) > 5 else None
            if outputs is None or source is None or not outputs.isAncestorOf(source):
                self._preview_timer.start()

    def _schedule_output_preview_refresh(self, *_):
        if not self._initializing and self.nav.currentRow() == 5:
            self._preview_timer.start()

    def _load_pages(self):
        for page in self.pages[:6]:
            page.load(self.config)
        self.pages[7].load(self.config)
        self.preview.load_view_settings(self.config)
        self._network_source_changed()
        self._sync_inlet_condition_count()
        self._update_project_label()

    def _update_project_label(self) -> None:
        if self.project_path is None:
            self.project_label.setText("Unsaved project")
            self.project_label.setToolTip("")
            self.project_label.setAccessibleDescription("Unsaved project")
            return

        display_path = self.project_path
        while display_path.suffix:
            display_path = display_path.with_suffix("")
        saved_at = datetime.fromtimestamp(
            self.project_path.stat().st_mtime
        ).astimezone()
        saved_text = saved_at.strftime("%b %-d, %Y, %-I:%M %p")
        label = f"Project {display_path.name}: Last saved {saved_text}"
        self.project_label.setText(label)
        self.project_label.setToolTip("")
        self.project_label.setAccessibleDescription(label)

    def _network_source_changed(self, *_):
        vessels: VesselsPage = self.pages[2]
        solvers: SolverPage = self.pages[4]
        is_lattice = vessels.source.currentData() == "lattice"
        solvers.set_lattice_mode(is_lattice)
        self._sync_inlet_condition_count()

    def _sync_inlet_condition_count(self, *_):
        if not getattr(self, "pages", None):
            return
        vessels: VesselsPage = self.pages[2]
        physics: PhysicsPage = self.pages[3]
        physics.set_inlet_count(vessels.configured_inlet_count())

    def _collect(self, show_error=True):
        updated = deepcopy(self.config)
        try:
            self._sync_inlet_condition_count()
            for page in self.pages[:6]:
                page.write(updated)
            self.pages[7].write(updated)
            updated.setdefault("gui", {})["viewer"] = self.preview.view_settings()
            if updated.get("gui", {}).get("network_source") == "lattice":
                if updated.setdefault("simulation", {}).get(
                    "concentration_solver"
                ) not in {
                    "network_ext",
                    "network",
                    "network_ext_hybrid_bg",
                }:
                    updated["simulation"]["concentration_solver"] = "network_ext"
                updated.setdefault("settings", {}).setdefault("hemodynamics", {})[
                    "kirchhoff_solver"
                ] = "spsolve"
        except Exception as exc:
            if show_error:
                QMessageBox.warning(self, "Check the current setup", str(exc))
            return None
        self.config = updated
        return updated

    def _page_changed(self, index):
        if index == 6:
            self.workspace_stack.setCurrentIndex(1)
        else:
            self.workspace_stack.setCurrentIndex(0)
            self.pages_stack.setCurrentIndex(self._inspector_page_index[index])
        self.back_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.pages) - 1)
        self.next_btn.setText(
            f"Continue to {self.PAGE_NAMES[index + 1]}"
            if index < len(self.pages) - 1
            else "Workflow complete"
        )
        if index == 6:
            self.preview.release()
        elif index == 7:
            self.pages[7]._request_render()
        else:
            self._preview_timer.start()
        if not self._initializing:
            self._refresh_status()

    def _refresh_preview(self):
        index = self.nav.currentRow()
        if index < 0 or index in {6, 7}:
            return
        config = self._collect(show_error=False)
        if config is not None:
            self.preview.show_case(
                config,
                include_network=index >= 2,
                include_tissue=index == 5,
            )
            source = config.get("gui", {}).get("network_source")
            if index >= 2 and source in {"svv_generated", "uploaded"}:
                signature = self._case_preview_signature(config)
                response = self._preview_seed_response
                geometry = response.get("geometry_path")
                if (
                    signature == self._preview_seed_signature
                    and geometry
                    and Path(str(geometry)).exists()
                ):
                    self.preview.show_seed(
                        config,
                        str(geometry),
                        response,
                        include_tissue=index == 5,
                    )
                elif signature == self._preview_failed_signature:
                    self.preview.status.setText(self._preview_failure_message)
                else:
                    self._start_preview_seed(config, signature)

    def _render_analysis(self, manifest_path, options):
        if self.nav.currentRow() == 7:
            self.preview.show_result(manifest_path, options)

    def _viewer_settings_changed(self):
        if self._initializing:
            return
        index = self.nav.currentRow()
        if index == 7:
            self.pages[7]._request_render()
        elif index not in {6}:
            self._preview_timer.start()

    def _preview_running_changed(self, running):
        self.backdrop.set_animation_enabled(running)
        self._update_solver_status()
        if running:
            self.preview.release()
        elif self.nav.currentRow() not in {6, 7}:
            self._preview_timer.start()
        elif self.nav.currentRow() == 7:
            self.pages[7]._request_render()

    def _update_solver_status(self):
        """Show only the global state that remains useful across pages."""
        job = self.runner.current
        if job is None:
            self.status_solver.setText("Idle")
        else:
            stage = str(job.stage or "Running")
            if len(stage) > 32:
                stage = stage[:31] + "…"
            self.status_solver.setText(f"{stage}  │  {int(job.progress or 0)}%")

    @staticmethod
    def _case_preview_signature(config):
        simulation = config.get("simulation", {})
        settings = config.get("settings", {})
        gui = config.get("gui", {})
        relevant = {
            "preview_version": 4,
            "domain": deepcopy(config.get("domain", {})),
            "network": deepcopy(config.get("network", {})),
            "growth": deepcopy(config.get("growth", {})),
            "simulation": {
                key: simulation.get(key)
                for key in (
                    "build_fluid",
                    "fluid",
                    "qin_target_ul_min",
                    "flow_source",
                    "inlet_conditions",
                )
            },
            "settings": {
                "hemodynamics": deepcopy(settings.get("hemodynamics", {})),
                "hematocrit": deepcopy(settings.get("hematocrit", {})),
            },
            "gui": {
                "network_source": gui.get("network_source"),
                "sweeps": [
                    deepcopy(sweep)
                    for sweep in gui.get("sweeps", [])
                    if sweep.get("path") == "network.target_terminal_count"
                ],
            },
        }
        if gui.get("network_source") == "svv_generated":
            relevant["network"].pop("input_path", None)
        payload = json.dumps(
            relevant, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _start_preview_seed(self, config, signature):
        if self._preview_process is not None:
            return
        if signature == self._preview_failed_signature:
            return
        preview_dir = self.project_dir / ".cascade_gui" / "previews" / signature[:16]
        response_path = preview_dir / "response.json"
        if response_path.exists():
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if (
                    Path(str(response.get("geometry_path", ""))).exists()
                    and Path(str(response.get("seed_path", ""))).exists()
                ):
                    self._preview_seed_signature = signature
                    self._preview_seed_response = response
                    self._preview_failed_signature = ""
                    self._preview_failure_message = ""
                    self.pages[2].set_radius_summary(response)
                    self._refresh_preview()
                    return
            except Exception:
                pass
        preview_dir.mkdir(parents=True, exist_ok=True)
        request = preview_dir / "request.json"
        preview_config = deepcopy(config)
        if preview_config.get("gui", {}).get("network_source") == "uploaded":
            raw_path = preview_config.get("network", {}).get("input_path")
            if raw_path:
                path = Path(str(raw_path)).expanduser()
                if not path.is_absolute():
                    preview_config["network"]["input_path"] = str(
                        (self.project_dir / path).resolve()
                    )
        request.write_text(json.dumps(preview_config, indent=2), encoding="utf-8")
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        process.readyReadStandardOutput.connect(self._preview_seed_progress)
        process.finished.connect(
            lambda code, status, sig=signature, folder=preview_dir: (
                self._preview_seed_finished(code, sig, folder)
            )
        )
        self._preview_process = process
        self._preview_pending_signature = signature
        self._preview_output_buffer = ""
        source = config.get("gui", {}).get("network_source")
        if source == "svv_generated":
            self.pages[2].set_radius_summary(calculating=True)
        uploaded_domain = str(config.get("domain", {}).get("type", "cube")) not in {
            "cube",
            "box",
            "sphere",
        }
        self.preview.status.setText(
            "Preparing the uploaded domain for preview…"
            if uploaded_domain
            else (
                "Growing reusable preview seed  │  up to about 1,000 vessel segments total…"
                if source == "svv_generated"
                else "Loading a bounded network preview…"
            )
        )
        process.start(
            sys.executable,
            [
                "-m",
                "cascade.gui.preview_worker",
                "--request",
                str(request),
                "--output",
                str(preview_dir),
            ],
        )

    def _cancel_preview_seed(self) -> None:
        process = self._preview_process
        self._preview_process = None
        self._preview_pending_signature = ""
        self._preview_output_buffer = ""
        if process is None:
            return
        try:
            process.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        if process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1500)
        process.deleteLater()

    def _preview_seed_progress(self) -> None:
        process = self._preview_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not chunk:
            return
        self._preview_output_buffer += chunk
        normalized = self._preview_output_buffer[-4_000:].replace("\r", "\n")
        percentages = re.findall(r"Adding vessels:\s*(\d+)%", normalized)
        if percentages:
            self.preview.status.setText(
                f"Growing reusable preview seed  │  {percentages[-1]}%"
            )
        elif "Preview domain ready:" in normalized:
            self.preview.status.setText(
                "Uploaded domain prepared  │  growing reusable preview seed…"
            )
        elif "repair" in normalized.lower() or "tetra" in normalized.lower():
            self.preview.status.setText(
                "Repairing and volume-meshing the uploaded domain…"
            )

    def _preview_seed_finished(self, exit_code, signature, preview_dir):
        process = self._preview_process
        output = self._preview_output_buffer
        if process is not None:
            output += bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )
            process.deleteLater()
        self._preview_process = None
        self._preview_pending_signature = ""
        self._preview_output_buffer = ""
        log_path = Path(preview_dir) / "preview.log"
        try:
            log_path.write_text(output, encoding="utf-8")
        except OSError:
            pass
        response_path = Path(preview_dir) / "response.json"
        if int(exit_code) == 0 and response_path.exists():
            try:
                self._preview_seed_response = json.loads(
                    response_path.read_text(encoding="utf-8")
                )
                self._preview_seed_signature = signature
                self._preview_failed_signature = ""
                self._preview_failure_message = ""
                self.pages[2].set_radius_summary(self._preview_seed_response)
            except Exception:
                self.preview.status.setText("preview seed unreadable")
        else:
            self._preview_failed_signature = signature
            summary = self._preview_failure_summary(output)
            self._preview_failure_message = f"Preview seed failed  │  {summary}"
            self.preview.status.setText(self._preview_failure_message)
            self.pages[2].radius_summary.setText(
                f"Radii could not be calculated. {summary}"
            )
        if self.nav.currentRow() not in {6, 7}:
            self._refresh_preview()

    @staticmethod
    def _preview_failure_summary(output: str) -> str:
        text = str(output or "")
        lowered = text.lower()
        if "tetgen" in lowered or "tetrahedraliz" in lowered:
            if "after automatic manifold repair" in lowered:
                return "The uploaded domain could not be volume-meshed after automatic repair."
            return "The uploaded domain could not be volume-meshed."
        meaningful = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
            and "unknown exception" not in line.lower()
            and not line.lstrip().startswith("Traceback")
        ]
        for line in reversed(meaningful):
            if line.startswith(("ValueError:", "RuntimeError:")):
                return line.split(":", 1)[1].strip()
        return (
            meaningful[-1][-180:]
            if meaningful
            else "The seed worker stopped unexpectedly."
        )

    def _refresh_status(self):
        config = self._collect(show_error=False)
        if config is None:
            self.validation_pill.set_status("Needs attention", "danger")
            self.validation_pill.setToolTip(
                "The current page contains a value that cannot be read. Click for details."
            )
            return
        report = validate_project(config)
        self._validation_report = report
        estimate = estimate_resources(config, self.hardware)
        if report.errors:
            self.validation_pill.set_status(
                f"{len(report.errors)} issue{'s' if len(report.errors) != 1 else ''}",
                "danger",
            )
        elif report.warnings:
            self.validation_pill.set_status(
                f"Ready  │  {len(report.warnings)} warning{'s' if len(report.warnings) != 1 else ''}",
                "warning",
            )
        else:
            self.validation_pill.set_status("Ready to queue", "success")
        details = self._validation_details(report)
        self.validation_pill.setToolTip(details)
        self.validation_pill.setAccessibleDescription(details)
        self._update_solver_status()
        self.pages[5].set_resource(estimate)

    @staticmethod
    def _validation_details(report) -> str:
        sections = []
        for title, messages in (
            ("Issues", report.errors),
            ("Warnings", report.warnings),
        ):
            if messages:
                sections.append(title + ":\n" + "\n".join(f"• {m}" for m in messages))
        return "\n\n".join(sections) or "Fully specified."

    def _show_validation_details(self) -> None:
        config = self._collect(show_error=False)
        if config is None:
            QMessageBox.warning(
                self,
                "Setup needs attention",
                "A value on the current page cannot be read. Review highlighted inputs or try queueing to reveal the exact field error.",
            )
            return
        report = validate_project(config)
        text = self._validation_details(report)
        if report.errors:
            QMessageBox.warning(self, "Setup issues", text)
        elif report.warnings:
            QMessageBox.information(self, "Setup warnings", text)
        else:
            QMessageBox.information(self, "Setup status", text)

    def _enqueue(self, run_now=False):
        config = self._collect()
        if config is None:
            return
        if config.get("gui", {}).get("network_source") == "svv_generated":
            signature = self._case_preview_signature(config)
            seed_path = self._preview_seed_response.get("seed_path")
            seed_ready = not (
                signature != self._preview_seed_signature
                or not seed_path
                or not Path(str(seed_path)).exists()
            )
            if seed_ready:
                config.setdefault("network", {})["input_path"] = str(seed_path)
                config.setdefault("gui", {})["preview_seed"] = {
                    "signature": signature,
                    "path": str(seed_path),
                }
            else:
                # Do not make a production run wait for a cosmetic preview.
                # The worker grows directly from the same roots and random seed.
                self._cancel_preview_seed()
                config.setdefault("network", {}).pop("input_path", None)
                config.setdefault("gui", {}).pop("preview_seed", None)
            config.setdefault("growth", {})["enabled"] = True
            config["growth"]["resume_from_checkpoint"] = False
        report = validate_project(config)
        if report.errors:
            QMessageBox.warning(
                self, "Cannot queue this setup", "\n\n".join(report.errors)
            )
            return
        if report.warnings:
            answer = QMessageBox.question(
                self,
                "Queue with warnings?",
                "\n\n".join(report.warnings) + "\n\nAdd the run anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.project_dir.mkdir(parents=True, exist_ok=True)
        if self.project_path is None:
            self.project_path = self.project_dir / "project.cascade.json"
        save_project(self.project_path, config)
        records = create_jobs(config, self.project_dir)
        self.runner.add(records)
        self._update_project_label()
        self.nav.setCurrentRow(6)
        if run_now:
            self.runner.run([job.id for job in records])

    def new(self):
        if self.runner.running:
            QMessageBox.warning(
                self,
                "Simulation running",
                "Cancel the active simulation before changing projects.",
            )
            return
        self.config = default_project()
        self.project_path = None
        self.project_dir = _default_project_directory()
        self.runner.replace_project(self.project_dir)
        self._load_pages()
        self.nav.setCurrentRow(0)
        self._refresh_status()

    def open(self):
        if self.runner.running:
            QMessageBox.warning(
                self,
                "Simulation running",
                "Cancel the active simulation before changing projects.",
            )
            return
        selected = choose_native_path(
            self,
            mode="file",
            caption="Open CASCADE project",
            start=str(self.project_dir),
            file_filter="CASCADE project (*.json *.cascade.json);;All JSON (*.json)",
        )
        if not selected:
            return
        try:
            self.config = load_project(selected)
            self.project_path = Path(selected).resolve()
            self.project_dir = self.project_path.parent
            self.runner.replace_project(self.project_dir)
            self._load_pages()
            self._refresh_status()
        except Exception as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))

    def save(self):
        if self.project_path is None:
            return self.save_as()
        config = self._collect()
        if config is None:
            return
        try:
            save_project(self.project_path, config)
            self._update_project_label()
            self.statusBar().showMessage("Project saved", 2500)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))

    def save_as(self):
        selected = choose_native_path(
            self,
            mode="save",
            caption="Save CASCADE project",
            start=str(self.project_dir / "project.cascade.json"),
            file_filter="CASCADE project (*.cascade.json);;JSON (*.json)",
        )
        if not selected:
            return
        self.project_path = Path(selected).resolve()
        self.project_dir = self.project_path.parent
        try:
            self.runner.replace_project(self.project_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot move project", str(exc))
            return
        self.save()

    def _sync_window_chrome(self):
        self.title_bar.sync_state()
        show_resize_handles = not (self.isMaximized() or self.isFullScreen())
        for handle in self.resize_handles:
            handle.setVisible(show_resize_handles)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange and hasattr(self, "title_bar"):
            self._sync_window_chrome()

    def closeEvent(self, event):
        cancel_native_pickers()
        self._cancel_preview_seed()
        if self.runner.running:
            answer = QMessageBox.question(
                self,
                "Simulation is running",
                "Cancel the active worker and exit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.cancel_pending()
            self.runner.cancel_current()
        event.accept()


__all__ = ("WindowResizeHandle", "WindowTitleBar", "MainWindow")
