from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from PySide6.QtCore import QCoreApplication, QProcess

from cascade.gui.runner import JobRunner
from cascade.utils.processes import ChildProcessJob

_PARENT_CODE = r"""
import pathlib
import subprocess
import sys
import time

gate = pathlib.Path(sys.argv[1])
child_pid = pathlib.Path(sys.argv[2])
while not gate.exists():
    time.sleep(0.01)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
child_pid.write_text(str(child.pid), encoding="ascii")
time.sleep(60)
"""


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object semantics")
def test_windows_job_terminates_descendant_processes(tmp_path: Path) -> None:
    gate = tmp_path / "start"
    child_pid_file = tmp_path / "child.pid"
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT_CODE, str(gate), str(child_pid_file)],
        cwd=tmp_path,
    )
    job = ChildProcessJob(parent.pid)
    child_pid = 0
    try:
        assert job.active
        gate.write_bytes(b"")
        assert _wait_until(child_pid_file.is_file)
        child_pid = int(child_pid_file.read_text(encoding="ascii"))
        assert psutil.pid_exists(child_pid)

        job.terminate()

        assert parent.wait(timeout=5) != 0
        assert _wait_until(lambda: not psutil.pid_exists(child_pid))
    finally:
        if job.active:
            job.terminate()
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if child_pid and psutil.pid_exists(child_pid):
            psutil.Process(child_pid).kill()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object semantics")
def test_studio_worker_uses_project_directory_and_process_job(tmp_path: Path) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    runner = JobRunner(tmp_path)
    worker_pid = 0
    try:
        runner.warmup()

        assert _wait_until(
            lambda: (
                app.processEvents() is None
                and runner.process is not None
                and runner.process.state() == QProcess.Running
                and runner._process_job is not None
            ),
            timeout_s=10,
        )
        assert runner.process is not None
        worker_pid = int(runner.process.processId())
        assert Path(runner.process.workingDirectory()) == tmp_path.resolve()
        assert runner._process_job is not None and runner._process_job.active
    finally:
        runner.shutdown()
        app.processEvents()
    if worker_pid:
        assert _wait_until(lambda: not psutil.pid_exists(worker_pid))
