"""Cross-platform ownership for subprocess trees launched by CASCADE."""

from __future__ import annotations

import os
from typing import Any

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
    """Own a Windows process tree and terminate descendants as one unit.

    On non-Windows platforms this is a no-op token; existing POSIX process
    handling remains authoritative there. On Windows, every descendant of the
    attached process joins a Job Object configured with
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
    """

    def __init__(self, process_id: int) -> None:
        self.process_id = int(process_id)
        self._handle: Any | None = None
        if os.name == "nt":
            self._handle = self._attach_windows_job(self.process_id)

    @property
    def active(self) -> bool:
        return self._handle is not None

    def terminate(self, exit_code: int = 1) -> None:
        """Terminate the attached Windows process tree and release its job."""
        if os.name != "nt" or self._handle is None:
            return
        handle, self._handle = self._handle, None
        try:
            if not _KERNEL32.TerminateJobObject(handle, int(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            _KERNEL32.CloseHandle(handle)

    def close(self) -> None:
        """Release the job, killing any descendants left after the root exits."""
        if os.name != "nt" or self._handle is None:
            return
        handle, self._handle = self._handle, None
        if not _KERNEL32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

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
