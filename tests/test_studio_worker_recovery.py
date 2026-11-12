from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from PySide6.QtWidgets import QApplication

from cascade.configuration.runtime import RuntimeConfiguration
from cascade.gui.model import JobRecord, prune_studio_caches
from cascade.gui.runner import JobRunner
from cascade.gui.window import MainWindow

# Speaks the persistent worker protocol.  Each settings file selects how the
# fake worker answers that request.
_FAKE_WORKER = r"""
import json
import os
import sys
import time

from cascade.commands.worker import exit_with_parent

exit_with_parent()


def emit(payload):
    print("CASCADE_WORKER_RESULT " + json.dumps(payload), flush=True)


print("CASCADE_WORKER_READY " + json.dumps({"pid": os.getpid()}), flush=True)
for raw in sys.stdin:
    request = json.loads(raw)
    request_id = request.get("id", "")
    command = request.get("command", "run")
    if command == "shutdown":
        emit({"id": request_id, "exit_code": 0})
        break
    with open(request["settings"], encoding="utf-8") as handle:
        settings = json.load(handle)
    behavior = settings.get(command, "ok")
    print(f"fake worker pid {os.getpid()} {command}", flush=True)
    if behavior == "ok":
        emit({"id": request_id, "exit_code": 0})
    elif behavior == "fail":
        print("RuntimeError: fake failure", flush=True)
        emit({"id": request_id, "exit_code": 1, "error": "RuntimeError: fake failure"})
    elif behavior == "malformed":
        print("CASCADE_WORKER_RESULT {not json", flush=True)
    elif behavior == "crash":
        os._exit(3)
    elif behavior == "hang":
        time.sleep(60)
"""


def _app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(predicate())


@pytest.fixture
def runner(tmp_path: Path):
    _app()
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER, encoding="utf-8")
    instance = JobRunner(tmp_path / "project")
    instance.worker_command = [sys.executable, str(script)]
    yield instance
    instance.shutdown()
    _app().processEvents()


def _job(tmp_path: Path, name: str, **behavior: str) -> JobRecord:
    folder = tmp_path / "runs" / name
    folder.mkdir(parents=True)
    settings = folder / "settings.json"
    settings.write_text(json.dumps(behavior), encoding="utf-8")
    return JobRecord(
        id=name,
        name=name,
        settings_path=str(settings),
        output_dir=str(folder),
        log_path=str(folder / "run.log"),
    )


def _worker_pid(job: JobRecord) -> int:
    text = Path(job.log_path or "").read_text(encoding="utf-8")
    return int(text.split("fake worker pid ", 1)[1].split()[0])


def _process_gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _finished(*jobs: JobRecord):
    return lambda: all(job.status in {"Completed", "Failed", "Cancelled"} for job in jobs)


def test_worker_that_fails_to_start_fails_the_job(
    runner: JobRunner, tmp_path: Path
) -> None:
    runner.worker_command = [str(tmp_path / "missing-python")]
    job = _job(tmp_path, "unstartable")
    runner.add([job])
    runner.run([job.id])

    assert _wait_until(lambda: job.status == "Failed" and not runner.running)
    assert "Could not start the simulation worker" in str(job.error)
    assert runner.process is None

    # The runner recovers as soon as a usable worker is configured.
    runner.worker_command = [sys.executable, str(tmp_path / "fake_worker.py")]
    runner.run([job.id])
    assert _wait_until(lambda: job.status == "Completed" and not runner.running)


def test_malformed_worker_result_fails_the_job(
    runner: JobRunner, tmp_path: Path
) -> None:
    job = _job(tmp_path, "malformed", run="malformed")
    runner.add([job])
    runner.run([job.id])

    assert _wait_until(lambda: job.status == "Failed" and not runner.running)
    assert str(job.error).startswith("Invalid worker response:")


def test_worker_crash_mid_job_fails_and_next_job_restarts_worker(
    runner: JobRunner, tmp_path: Path
) -> None:
    crashed = _job(tmp_path, "crashed", run="crash")
    recovered = _job(tmp_path, "recovered")
    runner.add([crashed, recovered])
    runner.run([crashed.id, recovered.id])

    assert _wait_until(_finished(crashed, recovered))
    assert crashed.status == "Failed"
    assert crashed.error == "Persistent simulation worker stopped."
    assert recovered.status == "Completed"
    assert _worker_pid(recovered) != _worker_pid(crashed)


def test_worker_is_replaced_after_failure_and_reused_after_success(
    runner: JobRunner, tmp_path: Path
) -> None:
    failed = _job(tmp_path, "failed", run="fail")
    fresh = _job(tmp_path, "fresh")
    reused = _job(tmp_path, "reused")
    runner.add([failed, fresh, reused])
    runner.run([failed.id, fresh.id, reused.id])

    assert _wait_until(_finished(failed, fresh, reused))
    assert [failed.status, fresh.status, reused.status] == [
        "Failed",
        "Completed",
        "Completed",
    ]
    assert failed.error == "RuntimeError: fake failure"
    failed_pid = _worker_pid(failed)
    assert _worker_pid(fresh) != failed_pid
    assert _worker_pid(reused) == _worker_pid(fresh)
    assert _wait_until(lambda: _process_gone(failed_pid))


def test_cancelled_job_releases_runner_for_next_job(
    runner: JobRunner, tmp_path: Path
) -> None:
    hanging = _job(tmp_path, "hanging", run="hang")
    followup = _job(tmp_path, "followup")
    runner.add([hanging, followup])
    runner.run([hanging.id])
    assert _wait_until(
        lambda: runner._job_dispatched and Path(hanging.log_path or "").stat().st_size
    )
    runner.cancel_current()
    assert _wait_until(lambda: hanging.status == "Cancelled" and not runner.running)

    runner.run([followup.id])
    assert _wait_until(lambda: followup.status == "Completed")


def test_failed_background_preparation_is_reported(
    runner: JobRunner, tmp_path: Path
) -> None:
    messages: list[str] = []
    logs: list[tuple[str, str]] = []
    runner.preparation_failed.connect(messages.append)
    runner.log_line.connect(lambda job_id, text: logs.append((job_id, text)))
    job = _job(tmp_path, "unpreparable", prepare="fail")
    runner.add([job])

    assert _wait_until(lambda: bool(messages))
    assert messages == ["Background preparation failed: RuntimeError: fake failure"]
    assert any(
        job_id == job.id and "RuntimeError: fake failure" in text
        for job_id, text in logs
    )
    assert runner.process is None


def test_preview_worker_that_fails_to_start_does_not_block_previews(
    tmp_path: Path,
) -> None:
    _app()
    window = MainWindow()
    try:
        window.project_dir = tmp_path
        window.preview_worker_command = [str(tmp_path / "missing-python")]
        config = window.config
        window._start_preview_seed(config, "a" * 64)
        assert _wait_until(lambda: window._preview_process is None)
        assert window._preview_failed_signature == "a" * 64
        assert "Could not start the preview worker" in window._preview_failure_message

        window._start_preview_seed(config, "b" * 64)
        # Windows may report FailedToStart synchronously from QProcess.start().
        assert _wait_until(lambda: window._preview_process is None)
        assert window._preview_pending_signature == ""
        assert window._preview_failed_signature == "b" * 64
        assert "Could not start the preview worker" in window._preview_failure_message
    finally:
        window._prepare_timer.stop()
        window._cancel_preview_seed()
        window.runner.shutdown()
        window.close()
        window.deleteLater()
        _app().processEvents()


def test_settings_runtime_overrides_do_not_leak_between_runs() -> None:
    runtime = SimpleNamespace(CHECKER_LINE_RESOLUTION=24, CM_TO_M=0.01)
    base = RuntimeConfiguration(
        fluid="water",
        build_fluid="water",
        inlet_flow_ul_min=2.0,
        distance_sample_count=0,
        compute_average_distance=False,
        concentration_solver="auto",
        equal_bifurcations=None,
        sections={},
    )
    replace(
        base, sections={"runtime": {"checker_line_resolution": 48}}
    ).apply_compatibility_state(runtime)
    assert runtime.CHECKER_LINE_RESOLUTION == 48

    base.apply_compatibility_state(runtime)
    assert runtime.CHECKER_LINE_RESOLUTION == 24

    with pytest.raises(ValueError, match="fixed unit conversion"):
        replace(
            base, sections={"runtime": {"cm_to_m": 1.0}}
        ).apply_compatibility_state(runtime)
    assert runtime.CM_TO_M == 0.01


def test_cache_pruning_keeps_newest_and_protected_entries(tmp_path: Path) -> None:
    previews = tmp_path / ".cascade_gui" / "previews"
    preloads = tmp_path / ".cascade_gui" / "preloads"
    preloads.mkdir(parents=True)
    for index in range(5):
        folder = previews / f"seed{index}"
        folder.mkdir(parents=True)
        (folder / "response.json").write_text("{}", encoding="utf-8")
        snapshot = preloads / f"case{index}.json"
        snapshot.write_text("{}", encoding="utf-8")
        for path in (folder, folder / "response.json", snapshot):
            os.utime(path, (1_000 + index, 1_000 + index))
    (preloads / "unused-output").mkdir()

    removed = prune_studio_caches(
        tmp_path, keep=2, protected=[previews / "seed0", preloads / "case1.json"]
    )

    assert removed == 4
    assert sorted(path.name for path in previews.iterdir()) == [
        "seed0",
        "seed3",
        "seed4",
    ]
    assert sorted(path.name for path in preloads.iterdir()) == [
        "case1.json",
        "case3.json",
        "case4.json",
        "unused-output",
    ]


_ORPHAN_PARENT = r"""
import os
import subprocess
import sys

env = dict(os.environ, CASCADE_STUDIO_PARENT_PID=str(os.getpid()))
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import time; from cascade.commands.worker import exit_with_parent; "
        "exit_with_parent(); print('armed', flush=True); time.sleep(30)",
    ],
    env=env,
    stdout=subprocess.PIPE,
    encoding="utf-8", errors="replace",
    start_new_session=True,
)
assert child.stdout.readline().strip() == "armed"
print(child.pid, flush=True)
sys.stdin.read()
"""


@pytest.mark.skipif(os.name == "nt", reason="Windows workers use a Job Object")
def test_worker_exits_when_studio_dies() -> None:
    parent = subprocess.Popen(
        [sys.executable, "-c", _ORPHAN_PARENT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        encoding="utf-8", errors="replace",
    )
    child_pid = 0
    try:
        child_pid = int(parent.stdout.readline())
        assert psutil.pid_exists(child_pid)
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=5)
        assert _wait_until(lambda: _process_gone(child_pid), timeout_s=3.0)
    finally:
        if parent.poll() is None:
            parent.kill()
        if child_pid and not _process_gone(child_pid):
            psutil.Process(child_pid).kill()


@pytest.mark.skipif(os.name == "nt", reason="Windows workers use a Job Object")
def test_worker_exits_when_studio_died_before_startup() -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait(timeout=5)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cascade.commands.worker import exit_with_parent; "
            "exit_with_parent(); print('still running')",
        ],
        env=dict(os.environ, CASCADE_STUDIO_PARENT_PID=str(gone.pid)),
        capture_output=True,
        encoding="utf-8", errors="replace",
        timeout=10,
    )
    assert result.returncode == 1
    assert "still running" not in result.stdout
