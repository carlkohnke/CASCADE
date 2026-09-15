"""Simulation queue page."""

from __future__ import annotations

from PySide6.QtCore import (
    QSize,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from cascade.gui.model import (
    open_folder,
    open_path,
)
from cascade.gui.runner import JobRunner
from cascade.gui.theme import Tokens
from cascade.gui.widgets import (
    Card,
    FocusPlainTextEdit,
    labeled,
)
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from cascade.gui.ui_helpers import (
    QMessageBox,
    _queue_action_icon,
)

from cascade.gui.pages.base import (
    Page,
)


class QueuePage(Page):
    add_requested = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(
            "Run queue",
            "",
            parent,
        )
        control = Card("Queue controls")
        self.run_name = QLineEdit("Run 001")
        self.run_name.setPlaceholderText("Run 001")
        control.add(labeled("Run name", self.run_name))
        toolbar = QWidget()
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(0, 0, 0, 0)
        self.add_btn = QPushButton("Add current setup")
        self.add_btn.setProperty("queueRole", "neutral")
        self.add_btn.setIcon(_queue_action_icon("add"))
        self.add_run_btn = QPushButton("Add and run")
        self.add_run_btn.setProperty("queueRole", "run")
        self.add_run_btn.setIcon(_queue_action_icon("add_run"))
        self.run_selected_btn = QPushButton("Run selected")
        self.run_selected_btn.setProperty("queueRole", "run")
        self.run_selected_btn.setIcon(_queue_action_icon("play_circle"))
        self.run_selected_btn.setEnabled(False)
        self.run_all_btn = QPushButton("Run all")
        self.run_all_btn.setProperty("queueRole", "run")
        self.run_all_btn.setIcon(_queue_action_icon("fast_forward"))
        self.run_all_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel active")
        self.cancel_btn.setProperty("queueRole", "stop")
        self.cancel_btn.setIcon(_queue_action_icon("stop"))
        self.cancel_btn.setEnabled(False)
        for button in [
            self.add_btn,
            self.add_run_btn,
            self.run_selected_btn,
            self.run_all_btn,
            self.cancel_btn,
        ]:
            button.setIconSize(QSize(22, 18))
            row.addWidget(button)
        row.addStretch()
        control.add(toolbar)
        self.column.addWidget(control)

        jobs = Card("Simulations")
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Run", "Timestamp", "Runtime", "Status", "Results"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setMinimumSectionSize(64)
        header.setCascadingSectionResizes(False)
        header.setStretchLastSection(True)
        # Dense default proportions for the primary queue view.  Keep every
        # column interactive so users can still tailor the table to a study.
        # The status column is deliberately the largest: it carries active
        # progress, while the results action needs a dependable readable width.
        for column, width in enumerate((270, 155, 88, 545, 255)):
            self.table.setColumnWidth(column, width)
        self.table.setMinimumHeight(270)
        jobs.add(self.table)
        small = QWidget()
        small_row = QHBoxLayout(small)
        small_row.setContentsMargins(0, 0, 0, 0)
        self.remove_btn = QPushButton("Clear selected")
        self.remove_btn.setProperty("secondary", True)
        self.remove_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear finished from queue")
        self.clear_btn.setProperty("secondary", True)
        self.clear_btn.setEnabled(False)
        self.delete_btn = QPushButton("Delete run files")
        self.delete_btn.setProperty("secondary", True)
        self.delete_btn.setEnabled(False)
        self.open_sweep_csv_btn = QPushButton("Open sweep CSV")
        self.open_sweep_csv_btn.setProperty("secondary", True)
        self.open_sweep_csv_btn.setEnabled(False)
        small_row.addWidget(self.remove_btn)
        small_row.addWidget(self.clear_btn)
        small_row.addWidget(self.delete_btn)
        small_row.addWidget(self.open_sweep_csv_btn)
        small_row.addStretch()
        jobs.add(small)
        self.column.addWidget(jobs)

        self.details_btn = QPushButton("+  Solver details")
        self.details_btn.setProperty("secondary", True)
        self.details_btn.setCheckable(True)
        self.details_btn.setChecked(False)
        self.details_btn.setToolTip("Show the selected simulation's diagnostic output.")
        self.column.addWidget(self.details_btn, 0, Qt.AlignLeft)
        log = Card("Diagnostics")
        self.log = FocusPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setMinimumHeight(220)
        log.add(self.log)
        self.diagnostics_panel = log
        self.diagnostics_panel.setVisible(False)
        self.column.addWidget(self.diagnostics_panel)
        self.finish()
        self.runner: JobRunner | None = None
        self._setup_changed = True
        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(1000)
        self.runtime_timer.timeout.connect(self.refresh)
        self.add_btn.clicked.connect(lambda: self.add_requested.emit(False))
        self.add_run_btn.clicked.connect(lambda: self.add_requested.emit(True))
        self.run_selected_btn.clicked.connect(self._run_selected)
        self.run_all_btn.clicked.connect(lambda: self.runner and self.runner.run())
        self.cancel_btn.clicked.connect(
            lambda: self.runner and self.runner.cancel_current()
        )
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn.clicked.connect(
            lambda: self.runner and self.runner.clear_finished()
        )
        self.delete_btn.clicked.connect(self._delete)
        self.open_sweep_csv_btn.clicked.connect(self._open_sweep_csv)
        self.details_btn.toggled.connect(self._toggle_diagnostics)
        self.table.itemSelectionChanged.connect(self._show_selected_log)
        self.table.itemSelectionChanged.connect(self._update_action_states)

    def set_runner(self, runner: JobRunner):
        if self.runner:
            try:
                self.runner.jobs_changed.disconnect(self.refresh)
                self.runner.job_updated.disconnect(self._refresh_job)
            except Exception:
                pass
        self.runner = runner
        runner.jobs_changed.connect(self.refresh)
        runner.job_updated.connect(self._refresh_job)
        runner.log_line.connect(self._append_log)
        runner.running_changed.connect(self._running_changed)
        self.refresh()

    def selected_ids(self):
        return [
            self.table.item(row, 0).data(Qt.UserRole)
            for row in sorted({i.row() for i in self.table.selectedIndexes()})
        ]

    def refresh(self):
        if not self.runner:
            return
        selected = set(self.selected_ids())
        self.table.setRowCount(len(self.runner.jobs))
        tones = {
            "Completed": Tokens.SUCCESS,
            "Failed": Tokens.DANGER,
            "Running": Tokens.TEXT,
            "Queued": Tokens.AMBER,
            "Cancelled": Tokens.TEXT_3,
            "Interrupted": Tokens.AMBER,
        }
        for row, job in enumerate(self.runner.jobs):
            self._render_job_row(row, job, tones)
            if job.id in selected:
                self.table.selectRow(row)
        if self.runner.running and not self.runtime_timer.isActive():
            self.runtime_timer.start()
        elif not self.runner.running:
            self.runtime_timer.stop()
        self._update_action_states()

    def _refresh_job(self, job_id: str) -> None:
        """Update one queue row without rebuilding thousands of row widgets."""
        if not self.runner:
            return
        row = next(
            (
                index
                for index in range(self.table.rowCount())
                if self.table.item(index, 0)
                and self.table.item(index, 0).data(Qt.UserRole) == job_id
            ),
            None,
        )
        job = next((item for item in self.runner.jobs if item.id == job_id), None)
        if row is None or job is None:
            self.refresh()
            return
        tones = {
            "Completed": Tokens.SUCCESS,
            "Failed": Tokens.DANGER,
            "Running": Tokens.TEXT,
            "Queued": Tokens.AMBER,
            "Cancelled": Tokens.TEXT_3,
            "Interrupted": Tokens.AMBER,
        }
        self._render_job_row(row, job, tones)
        self._update_action_states()

    def _render_job_row(self, row, job, tones) -> None:
        values = [job.name, self._timestamp_text(job), self._runtime_text(job)]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.UserRole, job.id if col == 0 else None)
            if col == 1:
                item.setToolTip(str(getattr(job, "created_at", "") or ""))
            self.table.setItem(row, col, item)

        status_cell = QWidget()
        status_cell.setProperty("jobStatus", job.status)
        status_layout = QVBoxLayout(status_cell)
        status_layout.setContentsMargins(7, 4, 7, 4)
        status_layout.setSpacing(4)
        status_text = job.status
        if job.status == "Running":
            stage = str(job.stage or "Running")
            status_text = f"{stage}  │  {int(job.progress or 0)}%"
        status_label = QLabel(status_text)
        status_label.setStyleSheet(
            f"color:{tones.get(job.status, Tokens.TEXT_2)};font-weight:600;"
        )
        status_layout.addWidget(status_label)
        if job.status == "Running":
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(int(job.progress or 0))
            progress.setTextVisible(False)
            progress.setFixedHeight(5)
            progress.setProperty("activeProgress", True)
            status_layout.addWidget(progress)
        self.table.setCellWidget(row, 3, status_cell)

        results = QPushButton("Open results folder")
        results.setProperty("secondary", True)
        results.setToolTip(str(job.output_dir))
        results.clicked.connect(
            lambda _checked=False, path=str(job.output_dir): self._open_results_folder(
                path
            )
        )
        self.table.setCellWidget(row, 4, results)
        self.table.setRowHeight(row, 46 if job.status == "Running" else 38)

    @staticmethod
    def _timestamp_text(job) -> str:
        raw = str(getattr(job, "created_at", "") or "")
        try:
            return datetime.fromisoformat(raw).astimezone().strftime("%Y-%m-%d  %H:%M")
        except (TypeError, ValueError):
            return raw[:16].replace("T", " ") or "—"

    @staticmethod
    def _runtime_text(job) -> str:
        started = getattr(job, "started_at", None)
        if not started:
            return "—"
        try:
            start = datetime.fromisoformat(str(started))
            finished = getattr(job, "finished_at", None)
            end = (
                datetime.fromisoformat(str(finished))
                if finished
                else datetime.now(timezone.utc)
            )
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            seconds = max(0, int((end - start).total_seconds()))
        except (TypeError, ValueError):
            return "—"
        if seconds < 60:
            return f"{seconds}s"
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m {seconds:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    def _update_action_states(self):
        running = bool(self.runner and self.runner.running)
        selected_ids = set(self.selected_ids()) if self.runner else set()
        selected_runnable = bool(
            self.runner
            and any(
                job.id in selected_ids
                and job.status in {"Queued", "Failed", "Cancelled", "Interrupted"}
                for job in self.runner.jobs
            )
        )
        queued = bool(
            self.runner
            and any(
                job.status in {"Queued", "Failed", "Cancelled", "Interrupted"}
                for job in self.runner.jobs
            )
        )
        active_id = getattr(getattr(self.runner, "current", None), "id", None)
        removable = bool(selected_ids) and active_id not in selected_ids
        finished = bool(
            self.runner
            and any(
                job.status in {"Completed", "Failed", "Cancelled"}
                for job in self.runner.jobs
            )
        )
        can_add = not running and self._setup_changed
        self.add_btn.setEnabled(can_add)
        self.add_run_btn.setEnabled(can_add)
        self.run_selected_btn.setEnabled(not running and selected_runnable)
        self.run_all_btn.setEnabled(not running and queued)
        self.cancel_btn.setEnabled(running)
        self.remove_btn.setEnabled(removable)
        self.clear_btn.setEnabled(finished)
        self.delete_btn.setEnabled(removable)
        selected_job = next(
            (
                job
                for job in (self.runner.jobs if self.runner else [])
                if job.id in selected_ids and job.combined_csv_path
            ),
            None,
        )
        self.open_sweep_csv_btn.setEnabled(
            bool(selected_job and Path(str(selected_job.combined_csv_path)).is_file())
        )

    def set_setup_changed(self, changed: bool) -> None:
        self._setup_changed = bool(changed)
        self._update_action_states()

    def _run_selected(self):
        selected = self.selected_ids()
        if self.runner and selected:
            self.runner.run(selected)

    def _remove(self):
        if not self.runner:
            return
        try:
            self.runner.remove(self.selected_ids())
        except Exception as exc:
            QMessageBox.warning(self, "Cannot remove job", str(exc))

    def _delete(self):
        if not self.runner:
            return
        ids = self.selected_ids()
        if not ids:
            return
        answer = QMessageBox.question(
            self,
            "Delete run files?",
            "Permanently delete the selected run folders and their results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            self.runner.delete(ids)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot delete run files", str(exc))

    def requested_run_name(self) -> str:
        return self.run_name.text().strip() or self.run_name.placeholderText()

    def advance_run_name(self) -> None:
        count = len(self.runner.jobs) + 1 if self.runner else 1
        self.run_name.setText(f"Run {count:03d}")

    def _open_sweep_csv(self):
        if not self.runner:
            return
        selected = set(self.selected_ids())
        job = next(
            (
                item
                for item in self.runner.jobs
                if item.id in selected and item.combined_csv_path
            ),
            None,
        )
        if job is not None:
            try:
                open_path(job.combined_csv_path)
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open sweep CSV", str(exc))

    def _open_results_folder(self, path):
        try:
            open_folder(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open results folder", str(exc))

    def _append_log(self, job_id, line):
        if not self.selected_ids() or job_id in self.selected_ids():
            self.log.appendPlainText(line)

    def _show_selected_log(self):
        if not self.runner:
            return
        ids = self.selected_ids()
        job = next((j for j in self.runner.jobs if j.id in ids), None)
        if job and job.log_path and Path(job.log_path).exists():
            try:
                text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")
                self.log.setPlainText(text[-250_000:])
                self.log.verticalScrollBar().setValue(
                    self.log.verticalScrollBar().maximum()
                )
            except Exception:
                pass

    def _running_changed(self, running):
        if running:
            self.runtime_timer.start()
        else:
            self.runtime_timer.stop()
        self.refresh()

    def _toggle_diagnostics(self, visible: bool) -> None:
        self.details_btn.setText(
            "−  Solver details" if visible else "+  Solver details"
        )
        self.diagnostics_panel.setVisible(visible)
        if visible:
            self._show_selected_log()


__all__ = ("QueuePage",)
