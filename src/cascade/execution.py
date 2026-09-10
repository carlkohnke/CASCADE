from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import gc
import getpass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterator, TypeVar


class SimulationAlreadyRunningError(RuntimeError):
    """Raised when another memory-intensive CASCADE process owns the host lock."""


_F = TypeVar("_F", bound=Callable[..., Any])


def simulation_lock_path() -> Path:
    """Return the per-user lock used to prevent overlapping CASCADE simulations."""
    override = os.environ.get("CASCADE_SIMULATION_LOCK_PATH")
    if override:
        return Path(override).expanduser().resolve()
    try:
        identity = f"uid-{os.getuid()}"
    except AttributeError:  # pragma: no cover - Windows only
        identity = re.sub(r"[^A-Za-z0-9_.-]+", "-", getpass.getuser()) or "user"
    return Path(tempfile.gettempdir()) / f"cascade-simulation-{identity}.lock"


@contextmanager
def single_simulation(
    operation: str,
    *,
    lock_path: str | Path | None = None,
) -> Iterator[Path]:
    """Allow only one memory-intensive CASCADE operation per user at a time.

    The operating system owns the lock, so an abnormal process exit releases it;
    the small metadata file may remain and is safe to reuse.
    """
    path = Path(lock_path).expanduser().resolve() if lock_path is not None else simulation_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        try:
            _lock_nonblocking(handle)
            acquired = True
        except (BlockingIOError, OSError) as exc:
            holder = _read_lock_metadata(handle)
            detail = f" Current owner: {holder}." if holder else ""
            raise SimulationAlreadyRunningError(
                "Another CASCADE simulation is already active. CASCADE permits one "
                f"memory-intensive operation at a time on this user account.{detail} "
                f"Lock: {path}"
            ) from exc

        metadata = {
            "pid": int(os.getpid()),
            "operation": str(operation),
        }
        handle.seek(0)
        handle.truncate()
        handle.write((json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        yield path
    finally:
        if acquired:
            try:
                handle.seek(0)
                handle.truncate()
                handle.flush()
            finally:
                _unlock(handle)
        handle.close()


def guard_simulation(operation: str) -> Callable[[_F], _F]:
    """Decorate a CLI entry point with the single-simulation host guard."""
    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any):
            with single_simulation(operation):
                return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator


def release_completed_case_memory(
    runtime_module: Any | None = None,
    *,
    trim_accelerator_pools: bool = True,
) -> dict[str, Any]:
    """Release result state, optionally retaining accelerator pools for reuse."""
    released: dict[str, Any] = {
        "module_attrs_cleared": [],
        "cupy_pool_trimmed": False,
    }
    if runtime_module is not None:
        for name in ("_LAST_CEXT_SOURCE_STATE", "_LAST_CEXT_CONTEXT", "_LAST_TISSUE_TIMINGS"):
            if hasattr(runtime_module, name):
                try:
                    setattr(runtime_module, name, None)
                    released["module_attrs_cleared"].append(name)
                except Exception:
                    pass
        cp = getattr(runtime_module, "_cp", None)
        if cp is not None:
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            if trim_accelerator_pools:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    released["cupy_pool_trimmed"] = True
                except Exception:
                    pass
                try:
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
    gc.collect()
    return released


def _read_lock_metadata(handle) -> str:
    try:
        handle.seek(0)
        return handle.read(512).decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


if os.name == "nt":  # pragma: no cover - exercised on Windows installations
    import msvcrt

    def _lock_nonblocking(handle) -> None:
        handle.seek(0)
        if not handle.read(1):
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_nonblocking(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
