"""Queue simulation jobs, launch worker processes, and report their progress to the GUI."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from cascade.utils.processes import ChildProcessJob

from .model import JobRecord, QueueStore
from .sweep_csv import rebuild_combined_sweep_csv


class JobRunner(QObject):
    """Sequential QProcess scheduler that keeps simulation memory out of the GUI."""

    jobs_changed = Signal()
    job_updated = Signal(str)
    log_line = Signal(str, str)
    running_changed = Signal(bool)
    all_finished = Signal()

    def __init__(self, project_dir: str | Path, parent: QObject | None = None):
        super().__init__(parent)
        self.store = QueueStore(project_dir)
        self.jobs = self.store.load()
        self.process: QProcess | None = None
        self._process_job: ChildProcessJob | None = None
        self._process_job_error = ""
        self.current: JobRecord | None = None
        self._scheduled_ids: list[str] = []
        self._cancelled = False
        self._worker_ready = False
        self._worker_buffer = ""
        self._job_dispatched = False
        self._pending_prepare: tuple[str, str] | None = None
        self._preparing_id: str | None = None
        self._preparing_job_id: str | None = None
        self._state_dirty = False
        self._state_timer = QTimer(self)
        self._state_timer.setSingleShot(True)
        self._state_timer.setInterval(500)
        self._state_timer.timeout.connect(self._flush_save_emit)
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(
            max(int(os.environ.get("CASCADE_STUDIO_WORKER_IDLE_MS", "300000")), 1000)
        )
        self._idle_timer.timeout.connect(self._shutdown_idle_worker)
        self._pending_sweep_csvs: set[str] = set()
        self._sweep_csv_timer = QTimer(self)
        self._sweep_csv_timer.setSingleShot(True)
        self._sweep_csv_timer.setInterval(5000)
        self._sweep_csv_timer.timeout.connect(self._flush_combined_sweep_csvs)

    @property
    def running(self) -> bool:
        return self.current is not None

    def warmup(self) -> None:
        """Start scientific imports in the background before the first Run click."""
        if self.current is None:
            self._ensure_worker()
            if self._pending_prepare is None and self._preparing_id is None:
                first = next((job for job in self.jobs if job.status == "Queued"), None)
                if first is not None:
                    self._queue_preparation(first)

    def prepare(self, settings_path: str | Path, cache_id: str) -> None:
        """Prepare the latest stable Studio setup without starting a queue row."""
        if self.current is not None:
            return
        self._pending_prepare = (str(cache_id), str(settings_path))
        self._idle_timer.stop()
        self._ensure_worker()
        self._dispatch_preparation()

    def replace_project(self, project_dir: str | Path) -> None:
        if self.running:
            raise RuntimeError("cannot change projects while a simulation is running")
        # An idle worker still owns the previous project's geometry, samples,
        # and allocator pools.  Do not carry those references across projects.
        self.shutdown()
        self.store = QueueStore(project_dir)
        self.jobs = self.store.load()
        self.jobs_changed.emit()
        QTimer.singleShot(250, self.warmup)

    def add(self, records: Iterable[JobRecord]) -> None:
        records = list(records)
        self.jobs.extend(records)
        self._refresh_combined_sweep_csvs(records)
        self._save_emit()
        if (
            self.current is None
            and self._pending_prepare is None
            and self._preparing_id is None
        ):
            first = next((job for job in records if job.status == "Queued"), None)
            if first is not None:
                self._queue_preparation(first)

    def remove(self, job_ids: Iterable[str]) -> None:
        ids = set(job_ids)
        if self.current and self.current.id in ids:
            raise RuntimeError("cancel the active job before removing it")
        removed = [job for job in self.jobs if job.id in ids]
        self.store.archive_results(removed)
        self.jobs = [job for job in self.jobs if job.id not in ids]
        if self._pending_prepare is not None and self._pending_prepare[0] in ids:
            self._pending_prepare = None
        self._save_emit()

    def delete(self, job_ids: Iterable[str]) -> None:
        ids = set(job_ids)
        if self.current and self.current.id in ids:
            raise RuntimeError("cancel the active job before deleting it")
        removed = [job for job in self.jobs if job.id in ids]
        self.store.delete_results(removed)
        self.jobs = [job for job in self.jobs if job.id not in ids]
        self._save_emit()

    def clear_finished(self) -> None:
        self.store.archive_results(
            job
            for job in self.jobs
            if job.status in {"Completed", "Failed", "Cancelled"}
        )
        self.jobs = [
            job
            for job in self.jobs
            if job.status not in {"Completed", "Failed", "Cancelled"}
        ]
        self._save_emit()

    def result_jobs(self) -> list[JobRecord]:
        results = {job.id: job for job in self.store.load_results()}
        for job in self.jobs:
            if job.manifest_path and Path(job.manifest_path).is_file():
                results[job.id] = job
        return sorted(results.values(), key=lambda job: job.created_at)

    def run(self, job_ids: Iterable[str] | None = None) -> None:
        if self.running:
            return
        self._idle_timer.stop()
        allowed = set(job_ids) if job_ids is not None else None
        self._scheduled_ids = [
            job.id
            for job in self.jobs
            if job.status in {"Queued", "Failed", "Cancelled", "Interrupted"}
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
        self.job_updated.emit(self.current.id)
        if os.name == "nt" and self._process_job is not None:
            self._terminate_process_tree(self.process)
        else:
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

    def shutdown(self) -> None:
        """Stop the persistent worker when Studio closes."""
        self._idle_timer.stop()
        self._flush_combined_sweep_csvs()
        self._flush_save_emit()
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            return
        if self.current is None and self._worker_ready:
            request = {"command": "shutdown", "id": "studio-shutdown"}
            process.write((json.dumps(request) + "\n").encode("utf-8"))
            process.waitForFinished(1500)
        if process.state() != QProcess.NotRunning:
            if os.name == "nt" and self._process_job is not None:
                self._terminate_process_tree(process)
                process.waitForFinished(1500)
            else:
                process.terminate()
                if not process.waitForFinished(1500):
                    process.kill()
                    process.waitForFinished(1500)
        self._release_process_job()

    def _start_next(self) -> None:
        if not self._scheduled_ids:
            self._flush_save_emit()
            self._flush_combined_sweep_csvs()
            self.current = None
            self.running_changed.emit(False)
            self.all_finished.emit()
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                self._idle_timer.start()
            return
        self._idle_timer.stop()
        job_id = self._scheduled_ids.pop(0)
        job = next((item for item in self.jobs if item.id == job_id), None)
        if job is None:
            self._start_next()
            return
        self.current = job
        self.running_changed.emit(True)
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

        self._job_dispatched = False
        self._ensure_worker()
        if self._worker_ready:
            self._dispatch_current()

    def _ensure_worker(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            return
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        working_directory = self.store.root.parent
        working_directory.mkdir(parents=True, exist_ok=True)
        process.setWorkingDirectory(str(working_directory))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        # The worker's adaptive case cleanup enforces a VRAM headroom bound.
        # Per-kernel pool trimming would discard reusable CUDA allocations and
        # make every queued case pay cold-allocation costs again.
        env.insert("SVV_CEXT_GPU_POOL_TRIM", "false")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_output)
        process.started.connect(self._attach_worker_process)
        process.finished.connect(self._worker_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        self._worker_ready = False
        self._worker_buffer = ""
        if self.current is not None:
            self._persist_job_update(self.current.id)
        process.start(
            sys.executable,
            ["-m", "cascade.commands.main", "worker"],
        )

    def _attach_worker_process(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            self._process_job = ChildProcessJob(int(process.processId()))
            self._process_job_error = ""
        except OSError as exc:
            self._process_job = None
            self._process_job_error = f"Could not own worker process tree: {exc}"
            if self.current is not None:
                self.log_line.emit(self.current.id, self._process_job_error)

    def _queue_preparation(self, job: JobRecord) -> None:
        """Prepare the first immutable queued case while Studio is idle."""
        self.prepare(job.settings_path, job.id)

    def _dispatch_preparation(self) -> None:
        if (
            self.process is None
            or not self._worker_ready
            or self._pending_prepare is None
            or self._preparing_id is not None
        ):
            return
        job_id, settings_path = self._pending_prepare
        self._pending_prepare = None
        request_id = f"studio-prepare:{job_id}"
        request = {
            "command": "prepare",
            "id": request_id,
            "settings": settings_path,
        }
        self.process.write((json.dumps(request) + "\n").encode("utf-8"))
        self._preparing_id = request_id
        self._preparing_job_id = job_id

    def _dispatch_current(self) -> None:
        if (
            self.process is None
            or self.current is None
            or not self._worker_ready
            or self._job_dispatched
        ):
            return
        if self._preparing_id is not None:
            self.current.stage = "Preparing queued case"
            self._persist_job_update(self.current.id)
            return
        if (
            self._pending_prepare is not None
            and self._pending_prepare[0] == self.current.id
        ):
            self._dispatch_preparation()
            self.current.stage = "Preparing queued case"
            self._persist_job_update(self.current.id)
            return
        # A Run selection supersedes an idle preparation that has not started.
        self._pending_prepare = None
        request = {
            "command": "run",
            "id": self.current.id,
            "settings": self.current.settings_path,
        }
        if self._process_job_error:
            self.log_line.emit(self.current.id, self._process_job_error)
            self._process_job_error = ""
        self.process.write((json.dumps(request) + "\n").encode("utf-8"))
        self._job_dispatched = True
        self.current.stage = "Starting simulation"
        self._persist_job_update(self.current.id)

    def _read_output(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if not data:
            return
        # Progress reporters commonly rewrite a line with carriage returns.
        # Treat them as records too so a long run cannot accumulate one giant
        # unterminated string in the GUI process.
        self._worker_buffer += data.replace("\r", "\n")
        lines = self._worker_buffer.split("\n")
        self._worker_buffer = lines.pop()
        if len(self._worker_buffer) > 1_048_576:
            lines.append(self._worker_buffer)
            self._worker_buffer = ""
        log_lines: list[str] = []
        log_job = self.current
        for line in lines:
            if line.startswith("CASCADE_WORKER_READY "):
                self._worker_ready = True
                self._dispatch_preparation()
                self._dispatch_current()
                if (
                    self.current is None
                    and self._preparing_id is None
                    and self._pending_prepare is None
                ):
                    self._idle_timer.start()
                continue
            if line.startswith("CASCADE_WORKER_RESULT "):
                self._handle_worker_result(line[len("CASCADE_WORKER_RESULT ") :])
                continue
            if self.current is None:
                continue
            log_lines.append(line)
            self._update_stage(line)
        if log_job is not None and log_lines:
            block = "\n".join(log_lines)
            if log_job.log_path:
                with Path(log_job.log_path).open("a", encoding="utf-8") as handle:
                    handle.write(block + "\n")
            # One queued Qt signal per process batch instead of one per line.
            self.log_line.emit(log_job.id, block)
        self._schedule_save_emit()

    def _handle_worker_result(self, payload: str) -> None:
        try:
            result = json.loads(payload)
        except Exception as exc:
            if self.current is not None:
                self.current.error = f"Invalid worker response: {exc}"
            return
        result_id = str(result.get("id", ""))
        if result_id == self._preparing_id:
            self._preparing_id = None
            self._preparing_job_id = None
            self._dispatch_current()
            if self.current is None:
                self._dispatch_preparation()
                if self._preparing_id is None:
                    self._idle_timer.start()
            return
        if self.current is None or result_id != self.current.id:
            return
        self._complete_current(
            int(result.get("exit_code", 1)),
            str(result.get("error", "")) or None,
        )

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

    def _complete_current(self, exit_code: int, error: str | None = None) -> None:
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
                if error:
                    job.error = error
                elif not job.error:
                    job.error = (
                        f"Simulation worker exited with code {exit_code}. See run.log."
                    )
        if job is not None and job.status == "Completed":
            self.store.archive_results([job])
        self.current = None
        self._job_dispatched = False
        self._schedule_combined_sweep_csvs([job] if job is not None else None)
        if job is not None:
            self._persist_job_update(job.id)
        QTimer.singleShot(0, self._start_next)

    def _worker_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        process = self.process
        self.process = None
        self._worker_ready = False
        self._worker_buffer = ""
        self._preparing_id = None
        self._preparing_job_id = None
        self._idle_timer.stop()
        self._release_process_job()
        if process is not None:
            process.deleteLater()
        if self.current is not None:
            self._complete_current(
                int(exit_code) if exit_code else 1,
                None if self._cancelled else "Persistent simulation worker stopped.",
            )

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        """Complete a legacy one-process job without bypassing new queue cleanup."""
        self._complete_current(int(exit_code))
        self._flush_combined_sweep_csvs()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if self.current:
            self.current.error = f"Could not run simulation worker: {error.name}"
            self.log_line.emit(self.current.id, self.current.error)

    def _kill_if_running(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self._terminate_process_tree(self.process)

    def _terminate_process_tree(self, process: QProcess) -> None:
        job, self._process_job = self._process_job, None
        if job is not None and job.active:
            try:
                job.terminate()
                return
            except OSError:
                pass
        process.kill()

    def _release_process_job(self) -> None:
        job, self._process_job = self._process_job, None
        if job is not None and job.active:
            try:
                job.close()
            except OSError:
                pass

    def _save_emit(self) -> None:
        self._state_timer.stop()
        self._state_dirty = False
        self.store.save(self.jobs)
        self.jobs_changed.emit()

    def _schedule_save_emit(self) -> None:
        """Coalesce verbose worker updates into at most two UI saves per second."""
        self._state_dirty = True
        if not self._state_timer.isActive():
            self._state_timer.start()

    def _flush_save_emit(self) -> None:
        if not self._state_dirty:
            return
        self._state_dirty = False
        self.store.save(self.jobs)
        if self.current is not None:
            self.job_updated.emit(self.current.id)

    def _persist_job_update(self, job_id: str) -> None:
        # The in-memory row is authoritative during a live queue.  Persist at
        # the coalesced interval so sub-second cached jobs do not serialize a
        # thousand-row queue once per completion.
        self._state_dirty = True
        if not self._state_timer.isActive():
            self._state_timer.start()
        self.job_updated.emit(job_id)

    def _shutdown_idle_worker(self) -> None:
        """Release cached geometry and pools after a configurable idle period."""
        if self.current is None:
            self.shutdown()

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

    def _schedule_combined_sweep_csvs(
        self, changed_jobs: Iterable[JobRecord] | None = None
    ) -> None:
        """Bound live sweep aggregation cost for large, fast GUI queues."""
        self._pending_sweep_csvs.update(
            str(job.combined_csv_path)
            for job in (changed_jobs if changed_jobs is not None else self.jobs)
            if job is not None and job.combined_csv_path
        )
        if self._pending_sweep_csvs and not self._sweep_csv_timer.isActive():
            self._sweep_csv_timer.start()

    def _flush_combined_sweep_csvs(self) -> None:
        if not self._pending_sweep_csvs:
            return
        self._sweep_csv_timer.stop()
        paths = set(self._pending_sweep_csvs)
        self._pending_sweep_csvs.clear()
        for path in paths:
            self._refresh_combined_sweep_csvs(
                [job for job in self.jobs if job.combined_csv_path == path]
            )
