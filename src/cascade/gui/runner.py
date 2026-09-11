from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from typing import Iterable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from .model import JobRecord, QueueStore
from .sweep_csv import rebuild_combined_sweep_csv


class JobRunner(QObject):
    """Sequential QProcess scheduler that keeps simulation memory out of the GUI."""

    jobs_changed = Signal()
    log_line = Signal(str, str)
    running_changed = Signal(bool)
    all_finished = Signal()

    def __init__(self, project_dir: str | Path, parent: QObject | None = None):
        super().__init__(parent)
        self.store = QueueStore(project_dir)
        self.jobs = self.store.load()
        self.process: QProcess | None = None
        self.current: JobRecord | None = None
        self._scheduled_ids: list[str] = []
        self._cancelled = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def replace_project(self, project_dir: str | Path) -> None:
        if self.running:
            raise RuntimeError("cannot change projects while a simulation is running")
        self.store = QueueStore(project_dir)
        self.jobs = self.store.load()
        self.jobs_changed.emit()

    def add(self, records: Iterable[JobRecord]) -> None:
        records = list(records)
        self.jobs.extend(records)
        self._refresh_combined_sweep_csvs(records)
        self._save_emit()

    def remove(self, job_ids: Iterable[str]) -> None:
        ids = set(job_ids)
        if self.current and self.current.id in ids:
            raise RuntimeError("cancel the active job before removing it")
        self.jobs = [job for job in self.jobs if job.id not in ids]
        self._save_emit()

    def clear_finished(self) -> None:
        self.jobs = [
            job
            for job in self.jobs
            if job.status not in {"Completed", "Failed", "Cancelled"}
        ]
        self._save_emit()

    def run(self, job_ids: Iterable[str] | None = None) -> None:
        if self.running:
            return
        allowed = set(job_ids) if job_ids is not None else None
        self._scheduled_ids = [
            job.id
            for job in self.jobs
            if job.status in {"Queued", "Failed", "Cancelled"}
            and (allowed is None or job.id in allowed)
        ]
        for job in self.jobs:
            if job.id in self._scheduled_ids:
                job.status, job.progress, job.stage, job.error = (
                    "Queued",
                    0,
                    "Waiting",
                    None,
                )
        self._refresh_combined_sweep_csvs()
        self._save_emit()
        self._start_next()

    def cancel_current(self) -> None:
        if not self.process or not self.current:
            return
        self._cancelled = True
        self.current.stage = "Cancelling"
        self.jobs_changed.emit()
        self.process.terminate()
        QTimer.singleShot(5000, self._kill_if_running)

    def cancel_pending(self) -> None:
        scheduled = set(self._scheduled_ids)
        self._scheduled_ids.clear()
        for job in self.jobs:
            if job.id in scheduled and job is not self.current:
                job.status, job.stage = "Cancelled", "Not started"
        self._refresh_combined_sweep_csvs()
        self._save_emit()

    def _start_next(self) -> None:
        if not self._scheduled_ids:
            self.current = None
            self.running_changed.emit(False)
            self.all_finished.emit()
            return
        job_id = self._scheduled_ids.pop(0)
        job = next((item for item in self.jobs if item.id == job_id), None)
        if job is None:
            self._start_next()
            return
        self.current = job
        self._cancelled = False
        job.status = "Running"
        job.stage = "Launching worker"
        job.progress = 1
        job.started_at = datetime.now(timezone.utc).isoformat()
        job.finished_at = None
        job.return_code = None
        job.error = None
        if job.log_path:
            path = Path(job.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("SVV_CEXT_GPU_POOL_TRIM", "true")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        self._save_emit()
        self.running_changed.emit(True)
        process.start(
            sys.executable, ["-m", "cascade.commands.main", "run", "--settings", job.settings_path]
        )

    def _read_output(self) -> None:
        if not self.process or not self.current:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if not data:
            return
        if self.current.log_path:
            with Path(self.current.log_path).open("a", encoding="utf-8") as handle:
                handle.write(data)
        for line in data.splitlines():
            self._update_stage(line)
            self.log_line.emit(self.current.id, line)
        self._save_emit()

    def _update_stage(self, line: str) -> None:
        if not self.current:
            return
        text = line.lower()
        stage, progress = self.current.stage, self.current.progress
        if "loaded settings" in text:
            stage, progress = "Reading settings", 5
        elif "gpu preflight passed" in text:
            stage, progress = "GPU ready", max(progress, 8)
        elif (
            "allocating patches" in text
            or "creating patches" in text
            or "solving patches" in text
            or "domain meshing" in text
        ):
            stage, progress = "Preparing tissue domain", max(progress, 12)
        elif (
            "growth" in text
            or "growing" in text
            or "bifurcation" in text
            or "adding vessels" in text
        ):
            stage, progress = "Building vessels", max(progress, 18)
        elif "network ready" in text:
            stage, progress = "Network ready", 32
        elif "kirchhoff" in text or "hematocrit" in text or "flow solver" in text:
            stage, progress = "Solving flow and hematocrit", max(progress, 43)
        elif "cext" in text or "coupling" in text:
            match = re.search(r"(?:iter(?:ation)?)[ =:]*(\d+)", text)
            iteration = int(match.group(1)) if match else 1
            stage, progress = (
                f"Coupling vessel and tissue oxygen · iteration {iteration}",
                min(55 + 4 * iteration, 82),
            )
        elif "concentration" in text or "oxygen" in text:
            stage, progress = "Solving oxygen transport", max(progress, 55)
        elif "tissue" in text or "sample" in text:
            stage, progress = "Evaluating tissue", max(progress, 84)
        elif "wrote" in text or "export" in text:
            stage, progress = "Writing results", max(progress, 94)
        self.current.stage, self.current.progress = stage, progress

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        job = self.current
        if job is not None:
            job.return_code = int(exit_code)
            job.finished_at = datetime.now(timezone.utc).isoformat()
            if self._cancelled:
                job.status, job.stage = "Cancelled", "Stopped by user"
            elif exit_code == 0:
                job.status, job.stage, job.progress = "Completed", "Finished", 100
            else:
                job.status, job.stage = "Failed", "Worker exited with an error"
                if not job.error:
                    job.error = (
                        f"Simulation worker exited with code {exit_code}. See run.log."
                    )
        process = self.process
        self.process = None
        self.current = None
        if process is not None:
            process.deleteLater()
        self._refresh_combined_sweep_csvs([job] if job is not None else None)
        self._save_emit()
        QTimer.singleShot(0, self._start_next)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self.current:
            self.current.error = f"Could not run simulation worker: {error.name}"
            self.log_line.emit(self.current.id, self.current.error)

    def _kill_if_running(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _save_emit(self) -> None:
        self.store.save(self.jobs)
        self.jobs_changed.emit()

    def _refresh_combined_sweep_csvs(
        self, changed_jobs: Iterable[JobRecord] | None = None
    ) -> None:
        paths = {
            str(job.combined_csv_path)
            for job in (changed_jobs if changed_jobs is not None else self.jobs)
            if job is not None and job.combined_csv_path
        }
        for path in paths:
            try:
                rebuild_combined_sweep_csv(self.jobs, path)
            except Exception as exc:
                owner = next(
                    (job for job in self.jobs if job.combined_csv_path == path),
                    None,
                )
                if owner is not None:
                    self.log_line.emit(
                        owner.id,
                        f"Warning: could not update combined sweep CSV: {exc}",
                    )
