"""Persistent, memory-bounded worker protocol for CASCADE Studio."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any

from cascade.utils.execution import single_simulation


_READY_PREFIX = "CASCADE_WORKER_READY "
_RESULT_PREFIX = "CASCADE_WORKER_RESULT "
_PARENT_PID_ENV = "CASCADE_STUDIO_PARENT_PID"
_PR_SET_PDEATHSIG = 1


def exit_with_parent() -> None:
    """Stop this worker when the Studio process that launched it exits.

    Studio starts workers in their own session, so a crashed GUI would
    otherwise leave them running until their next stdin read.  Windows needs no
    handling: Studio's kill-on-close Job Object already owns the worker tree.
    """
    if os.name == "nt":
        return
    try:
        parent_pid = int(os.environ.pop(_PARENT_PID_ENV, "") or os.getppid())
    except ValueError:
        parent_pid = os.getppid()
    direct_child = os.getppid() == parent_pid
    armed = False
    if direct_child and sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            armed = libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) == 0
        except (AttributeError, OSError):
            armed = False
    # Re-check after arming: the parent may have exited before prctl ran.
    if not _parent_alive(parent_pid, direct_child):
        os._exit(1)
    if not armed:
        threading.Thread(
            target=_watch_parent,
            args=(parent_pid, direct_child),
            name="cascade-parent-watchdog",
            daemon=True,
        ).start()


def _parent_alive(parent_pid: int, direct_child: bool) -> bool:
    if direct_child:
        return os.getppid() == parent_pid
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _watch_parent(parent_pid: int, direct_child: bool) -> None:
    while _parent_alive(parent_pid, direct_child):
        time.sleep(0.5)
    os._exit(1)


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def serve() -> int:
    """Read newline-delimited jobs from stdin and keep imports/CUDA warm."""
    exit_with_parent()
    # Import the scientific path before advertising readiness.  Studio starts
    # this worker in the background, so imports are paid while the user edits
    # or reviews a case instead of after the Run click.
    from cascade.commands.main import (
        execute_configured_run,
        prepare_configured_run,
    )
    from cascade.commands.workspace import InteractiveRunWorkspace

    workspace = InteractiveRunWorkspace()
    _emit(_READY_PREFIX, {"pid": int(os.getpid()), "protocol": 3})
    # Record pristine solver constants before the first request can override
    # them.  Requests queue on stdin while the scientific runtime imports.
    try:
        from cascade.configuration.bridge import load_runtime_module
        from cascade.configuration.runtime import snapshot_runtime_constants

        snapshot_runtime_constants(load_runtime_module())
    except Exception:
        # The first request reports the import failure with its traceback.
        pass
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        request_id = ""
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("worker request must be a JSON object")
            request_id = str(request.get("id", ""))
            command = str(request.get("command", "run"))
            if command in {"status", "clear"}:
                from cascade.configuration.bridge import load_runtime_module
                from cascade.utils.execution import (
                    host_memory_snapshot,
                    release_completed_case_memory,
                )

                if command == "clear":
                    workspace.clear()
                    cleanup = release_completed_case_memory(
                        load_runtime_module(), trim_accelerator_pools=True
                    )
                else:
                    cleanup = host_memory_snapshot()
                _emit(
                    _RESULT_PREFIX,
                    {
                        "id": request_id,
                        "exit_code": 0,
                        "memory": cleanup,
                        "cache": workspace.cache_state(),
                    },
                )
                continue
            if command == "shutdown":
                workspace.clear()
                try:
                    from cascade.configuration.bridge import load_runtime_module
                    from cascade.utils.execution import release_completed_case_memory

                    release_completed_case_memory(
                        load_runtime_module(), trim_accelerator_pools=True
                    )
                except Exception:
                    pass
                _emit(_RESULT_PREFIX, {"id": request_id, "exit_code": 0})
                return 0
            if command == "prepare":
                settings_path = request.get("settings")
                if not isinstance(settings_path, str) or not settings_path:
                    raise ValueError("worker prepare request requires a settings path")
                with single_simulation("cascade worker preparation"):
                    prepared = prepare_configured_run(
                        settings_path, workspace=workspace
                    )
                _emit(
                    _RESULT_PREFIX,
                    {
                        "id": request_id,
                        "exit_code": 0,
                        "prepared": True,
                        **prepared,
                    },
                )
                continue
            if command != "run":
                raise ValueError(f"unsupported worker command: {command!r}")
            settings_path = request.get("settings")
            if not isinstance(settings_path, str) or not settings_path:
                raise ValueError("worker run request requires a settings path")

            with single_simulation("cascade worker case"):
                outputs, elapsed = execute_configured_run(
                    settings_path,
                    cleanup_policy="adaptive",
                    workspace=workspace,
                )
            from cascade.utils.execution import host_memory_snapshot

            _emit(
                _RESULT_PREFIX,
                {
                    "id": request_id,
                    "exit_code": 0,
                    "elapsed_s": float(elapsed),
                    "outputs": outputs,
                    "memory": host_memory_snapshot(),
                    "cache": workspace.cache_state(),
                },
            )
        except BaseException as exc:
            traceback.print_exc()
            _emit(
                _RESULT_PREFIX,
                {
                    "id": request_id,
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    return 0


__all__ = ["serve"]
