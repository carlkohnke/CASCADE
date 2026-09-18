"""Cross-platform ownership for subprocess trees launched by CASCADE."""

from __future__ import annotations

import os
import signal
from typing import Any

import psutil

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100

    class _JobObjectBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobObjectExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobObjectBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _KERNEL32.CreateJobObjectW.restype = wintypes.HANDLE
    _KERNEL32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _KERNEL32.SetInformationJobObject.restype = wintypes.BOOL
    _KERNEL32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _KERNEL32.OpenProcess.restype = wintypes.HANDLE
    _KERNEL32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _KERNEL32.AssignProcessToJobObject.restype = wintypes.BOOL
    _KERNEL32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _KERNEL32.TerminateJobObject.restype = wintypes.BOOL
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL


class ChildProcessJob:
    """Own a child process tree and terminate descendants as one unit.

    Windows uses a Job Object with kill-on-close semantics. POSIX platforms
    snapshot descendants before signaling the root so grandchildren cannot be
    orphaned when Studio cancels a worker.
    """

    def __init__(self, process_id: int) -> None:
        self.process_id = int(process_id)
        self._handle: Any | None = None
        self._closed = False
        self._create_time: float | None = None
        self._process_group: int | None = None
        if os.name == "nt":
            self._handle = self._attach_windows_job(self.process_id)
        else:
            try:
                process = psutil.Process(self.process_id)
                self._create_time = float(process.create_time())
            except psutil.Error as exc:
                raise OSError(
                    f"Could not attach to child process {self.process_id}."
                ) from exc
            process_group = os.getpgid(self.process_id)
            if process_group == self.process_id:
                self._process_group = process_group

    @property
    def active(self) -> bool:
        if self._closed:
            return False
        if os.name == "nt":
            return self._handle is not None
        # A session leader may exit before one of its grandchildren.  Keep the
        # ownership token active until close() has had a chance to reap the
        # whole process group.
        return self._process_group is not None or self._root_process() is not None

    def terminate(self, exit_code: int = 1) -> None:
        """Terminate the attached process tree and release its ownership token."""
        if self._closed:
            return
        if os.name == "nt":
            if self._handle is None:
                self._closed = True
                return
            handle, self._handle = self._handle, None
            try:
                if not _KERNEL32.TerminateJobObject(handle, int(exit_code)):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                _KERNEL32.CloseHandle(handle)
                self._closed = True
            return
        self._terminate_posix_tree(force=False)
        self._closed = True

    def kill(self, exit_code: int = 1) -> None:
        """Immediately kill the attached process tree."""
        if self._closed:
            return
        if os.name == "nt":
            self.terminate(exit_code)
            return
        self._terminate_posix_tree(force=True)
        self._closed = True

    def close(self) -> None:
        """Release ownership, killing descendants left after the root exits."""
        if self._closed:
            return
        if os.name == "nt":
            if self._handle is not None:
                handle, self._handle = self._handle, None
                if not _KERNEL32.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
        else:
            self._terminate_posix_tree(force=True)
        self._closed = True

    def _root_process(self) -> psutil.Process | None:
        try:
            process = psutil.Process(self.process_id)
            if self._create_time is not None and not abs(
                float(process.create_time()) - self._create_time
            ) < 1.0e-6:
                return None
            return process
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            return None

    def _terminate_posix_tree(self, *, force: bool) -> None:
        root = self._root_process()
        if root is None and self._process_group is None:
            return
        descendants: list[psutil.Process] = []
        if root is not None:
            try:
                descendants = root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                descendants = []
        processes = [*reversed(descendants), *([root] if root is not None else [])]
        if self._process_group is not None:
            try:
                os.killpg(
                    self._process_group,
                    signal.SIGKILL if force else signal.SIGTERM,
                )
            except ProcessLookupError:
                return
            except PermissionError:
                pass
        for process in processes:
            try:
                process.kill() if force else process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not force:
            _gone, alive = psutil.wait_procs(processes, timeout=0.5)
            for process in alive:
                try:
                    process.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if self._process_group is not None:
                try:
                    os.killpg(self._process_group, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    @staticmethod
    def _attach_windows_job(process_id: int):
        handle = _KERNEL32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _JobObjectExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            )
            if not _KERNEL32.SetInformationJobObject(
                handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            process = _KERNEL32.OpenProcess(
                _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
                False,
                int(process_id),
            )
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not _KERNEL32.AssignProcessToJobObject(handle, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                _KERNEL32.CloseHandle(process)
            return handle
        except BaseException:
            _KERNEL32.CloseHandle(handle)
            raise


__all__ = ["ChildProcessJob"]
