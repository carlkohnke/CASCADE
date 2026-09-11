"""Coordinate process-safe simulation locks and release completed-case memory."""

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
import sys
from typing import Any, Callable, Iterator, TypeVar


class SimulationAlreadyRunningError(RuntimeError):
    """Raised when another memory-intensive CASCADE process owns the host lock."""


_F = TypeVar("_F", bound=Callable[..., Any])
_CASE_CLEANUP_COUNT = 0


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
    path = (
        Path(lock_path).expanduser().resolve()
        if lock_path is not None
        else simulation_lock_path()
    )
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
    trim_accelerator_pools: bool | None = True,
    gpu_headroom_fraction: float = 0.20,
    gpu_pool_limit_fraction: float = 0.50,
) -> dict[str, Any]:
    """Release one case while optionally retaining safe allocator capacity.

    ``trim_accelerator_pools=None`` selects an adaptive policy: unused blocks
    stay warm while the device has adequate headroom and are returned when the
    pool grows too large or free VRAM is low.  Live arrays are never hidden by
    this helper; callers must drop case-owned references before invoking it.
    """
    global _CASE_CLEANUP_COUNT
    _CASE_CLEANUP_COUNT += 1
    full_collection = bool(trim_accelerator_pools) or _CASE_CLEANUP_COUNT % 16 == 0
    released: dict[str, Any] = {
        "module_attrs_cleared": [],
        "cupy_pool_trimmed": False,
        "cupy_pool_used_bytes": 0,
        "cupy_pool_total_bytes": 0,
        "gpu_free_bytes": None,
        "gpu_total_bytes": None,
        **host_memory_snapshot(),
        "gc_generation": 2 if full_collection else 0,
    }
    collected = False
    if runtime_module is not None:
        for name in (
            "_LAST_CEXT_SOURCE_STATE",
            "_LAST_CEXT_CONTEXT",
            "_LAST_TISSUE_TIMINGS",
        ):
            if hasattr(runtime_module, name):
                try:
                    setattr(runtime_module, name, None)
                    released["module_attrs_cleared"].append(name)
                except Exception:
                    pass
        gc.collect(2 if full_collection else 0)
        collected = True
        cp = getattr(runtime_module, "_cp", None)
        if cp is not None:
            try:
                cp.cuda.Stream.null.synchronize()
            except Exception:
                pass
            should_trim = bool(trim_accelerator_pools)
            try:
                pool = cp.get_default_memory_pool()
                released["cupy_pool_used_bytes"] = int(pool.used_bytes())
                released["cupy_pool_total_bytes"] = int(pool.total_bytes())
                free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
                free_bytes = int(free_bytes)
                total_bytes = int(total_bytes)
                released["gpu_free_bytes"] = free_bytes
                released["gpu_total_bytes"] = total_bytes
                if trim_accelerator_pools is None and total_bytes > 0:
                    should_trim = (
                        free_bytes < float(gpu_headroom_fraction) * total_bytes
                        or int(pool.total_bytes())
                        > float(gpu_pool_limit_fraction) * total_bytes
                    )
            except Exception:
                # Explicit trimming remains best-effort even when accounting is
                # unavailable on an older CuPy/runtime combination.
                should_trim = bool(trim_accelerator_pools)
            if should_trim:
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    released["cupy_pool_trimmed"] = True
                except Exception:
                    pass
                try:
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
    # NumPy/CuPy arrays are reference-counted; a full cyclic scan on every
    # sub-second case can cost more than the solve.  Sweep periodically and at
    # hard boundaries, while still collecting new cycles after every case.
    if not collected:
        gc.collect(2 if full_collection else 0)
    # CPython has already released unreachable objects at this point.  On
    # glibc, large freed arenas can nevertheless remain mapped in a long-lived
    # worker.  Return them only for an explicit hard cleanup or under genuine
    # host-memory pressure; keeping ordinary arenas warm is faster between
    # interactive cases.
    available = released.get("host_available_bytes")
    total = released.get("host_total_bytes")
    host_pressure = bool(
        isinstance(available, int)
        and isinstance(total, int)
        and total > 0
        and available < 0.15 * total
    )
    released["host_allocator_trimmed"] = (
        _trim_host_allocator()
        if bool(trim_accelerator_pools) or host_pressure
        else False
    )
    released.update(host_memory_snapshot())
    return released


def host_memory_snapshot() -> dict[str, int | None]:
    """Return current process RSS and system RAM headroom without dependencies."""
    rss: int | None = None
    available: int | None = None
    total: int | None = None
    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        with Path("/proc/self/statm").open("r", encoding="ascii") as handle:
            rss = int(handle.read().split()[1]) * page
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open("r", encoding="ascii") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
        available = values.get("MemAvailable")
        total = values.get("MemTotal")
    except (OSError, ValueError, IndexError):
        pass
    return {
        "process_rss_bytes": rss,
        "host_available_bytes": available,
        "host_total_bytes": total,
    }


def _trim_host_allocator() -> bool:
    """Best-effort glibc arena trim for long-lived Linux workers."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim")
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        return bool(malloc_trim(0))
    except (AttributeError, OSError):
        return False


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
