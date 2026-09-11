"""Persistent, memory-bounded worker protocol for CASCADE Studio."""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any

from cascade.utils.execution import single_simulation


_READY_PREFIX = "CASCADE_WORKER_READY "
_RESULT_PREFIX = "CASCADE_WORKER_RESULT "


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def serve() -> int:
    """Read newline-delimited jobs from stdin and keep imports/CUDA warm."""
    _emit(_READY_PREFIX, {"pid": int(os.getpid()), "protocol": 2})
    # Let Studio enqueue the first request immediately; scientific imports are
    # then paid once while the request waits in the stdin pipe.
    from cascade.commands.workspace import InteractiveRunWorkspace

    workspace = InteractiveRunWorkspace()
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
            if command != "run":
                raise ValueError(f"unsupported worker command: {command!r}")
            settings_path = request.get("settings")
            if not isinstance(settings_path, str) or not settings_path:
                raise ValueError("worker run request requires a settings path")

            # Import after the ready marker so Studio can distinguish worker
            # startup from the first scientific stack initialization.
            from cascade.commands.main import execute_configured_run

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

