"""Configure wheel-provided CUDA components before CuPy is imported.

Python 3.8 and newer deliberately excludes ``PATH`` from the default DLL
search used for extension modules on Windows.  CUDA component wheels install
their DLLs below ``site-packages``; retaining ``os.add_dll_directory`` handles
for those directories makes the dependencies visible without modifying the
user or system environment.
"""

from __future__ import annotations

import ctypes
import functools
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from cascade.runtime.paths import config_directory

_COMPONENT_DISTRIBUTIONS = (
    "nvidia-cublas",
    "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-cusolver",
    "nvidia-cusparse",
    "nvidia-nvjitlink",
)
_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()
_PRELOADED_LIBRARIES: list[Any] = []


def _windows_short_path(path: str | Path) -> str:
    """Return an ASCII-safe NTFS path when Windows exposes an 8.3 alias."""
    value = str(path)
    if os.name != "nt" or value.isascii():
        return value
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        buffer = ctypes.create_unicode_buffer(32768)
        written = int(get_short_path(value, buffer, len(buffer)))
    except (AttributeError, OSError, ValueError):
        return value
    return buffer.value if 0 < written < len(buffer) else value


def configure_cupy_compiler_paths(cupy_module: Any) -> bool:
    """Make CuPy's NVRTC include options safe for Unicode Windows installs."""
    if os.name != "nt":
        return False
    compiler = cupy_module.cuda.compiler
    current = compiler._compile_module_with_cache
    if getattr(current, "_cascade_windows_path_wrapper", False):
        return True

    @functools.wraps(current)
    def compile_with_windows_paths(source, options=(), **kwargs):
        safe_options = tuple(
            "-I" + _windows_short_path(option[2:])
            if option.startswith("-I") and not option.isascii()
            else option
            for option in options
        )
        return current(source, safe_options, **kwargs)

    compile_with_windows_paths._cascade_windows_path_wrapper = True
    compiler._compile_module_with_cache = compile_with_windows_paths
    return True


def _is_component_library(path: Path) -> bool:
    name = path.name.lower()
    if os.name == "nt":
        return name.endswith(".dll")
    return ".so" in name


def _component_distributions():
    for distribution_name in _COMPONENT_DISTRIBUTIONS:
        try:
            yield metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError:
            continue


def cuda_component_library_dirs() -> tuple[Path, ...]:
    """Return directories containing NVIDIA component-wheel libraries."""
    directories: list[Path] = []
    for distribution in _component_distributions():
        for relative in distribution.files or ():
            relative_path = Path(str(relative))
            if not _is_component_library(relative_path):
                continue
            directory = Path(distribution.locate_file(relative)).resolve().parent
            if directory.is_dir() and directory not in directories:
                directories.append(directory)
    return tuple(directories)


def _component_cuda_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for distribution in _component_distributions():
        for relative in distribution.files or ():
            relative_path = Path(str(relative))
            if relative_path.name.lower() != "cuda_fp16.h":
                continue
            include_directory = Path(distribution.locate_file(relative)).resolve().parent
            root = include_directory.parent
            if root not in roots:
                roots.append(root)
    return tuple(roots)


def _has_cuda_headers(path: Path) -> bool:
    return (path / "include" / "cuda_fp16.h").is_file()


def cuda_path_file() -> Path:
    """Return the persistent per-user CUDA toolkit selection file."""
    return config_directory() / ".cascade_cuda_path"


def _configured_file_root() -> Path | None:
    path_file = cuda_path_file()
    try:
        value = path_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(value).expanduser() if value else None


def _system_cuda_roots() -> tuple[Path, ...]:
    if os.name == "nt":
        program_files = os.environ.get("ProgramFiles")
        if not program_files:
            return ()
        root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        try:
            return tuple(sorted(root.glob("v*"), key=_cuda_version_key, reverse=True))
        except OSError:
            return ()
    return (
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-13"),
        Path("/usr/local/cuda-12"),
    )


def _cuda_version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def discover_cuda_path() -> Path | None:
    """Find a CUDA root containing headers needed by CuPy's NVRTC compiler."""
    candidates: list[Path] = []
    configured = os.environ.get("CUDA_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())
    file_root = _configured_file_root()
    if file_root is not None:
        candidates.append(file_root)
    candidates.append(Path(sys.prefix) / "targets" / "x86_64-linux")
    candidates.extend(_component_cuda_roots())
    candidates.extend(_system_cuda_roots())

    for candidate in candidates:
        if _has_cuda_headers(candidate):
            return candidate.resolve()
    return None


def preload_cuda_component_libraries() -> tuple[str, ...]:
    """Make component-wheel CUDA libraries visible to the current process."""
    directories = cuda_component_library_dirs()
    if os.name == "nt":
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is not None:
            for directory in directories:
                key = os.path.normcase(str(directory))
                if key in _DLL_DIRECTORY_PATHS:
                    continue
                _DLL_DIRECTORY_HANDLES.append(add_directory(str(directory)))
                _DLL_DIRECTORY_PATHS.add(key)
        return tuple(str(path) for path in directories)

    if not sys.platform.startswith("linux"):
        return tuple(str(path) for path in directories)

    # nvJitLink must be globally visible before cuFFT on CUDA 13 Linux wheels.
    for library_pattern in ("libnvJitLink.so*", "libcufft.so*"):
        library_path = next(
            (
                candidate
                for directory in directories
                for candidate in sorted(directory.glob(library_pattern))
                if candidate.is_file()
            ),
            None,
        )
        if library_path is None:
            continue
        try:
            _PRELOADED_LIBRARIES.append(
                ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not load CUDA component library {library_path}: {exc}"
            ) from exc
    return tuple(str(path) for path in directories)


def configure_cuda_runtime() -> tuple[str, ...]:
    """Configure CUDA headers and libraries without importing CuPy."""
    if not os.environ.get("CUDA_PATH"):
        cuda_path = discover_cuda_path()
        if cuda_path is not None:
            os.environ["CUDA_PATH"] = _windows_short_path(cuda_path)
    return preload_cuda_component_libraries()


def cuda_subprocess_environment() -> dict[str, str]:
    """Return an environment suitable for an isolated CUDA probe process."""
    configure_cuda_runtime()
    probe_env = os.environ.copy()
    directories = cuda_component_library_dirs()
    if directories:
        variable = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        existing = probe_env.get(variable, "")
        prefix = os.pathsep.join(str(path) for path in directories)
        probe_env[variable] = prefix if not existing else prefix + os.pathsep + existing
    return probe_env


__all__ = [
    "configure_cupy_compiler_paths",
    "configure_cuda_runtime",
    "cuda_component_library_dirs",
    "cuda_path_file",
    "cuda_subprocess_environment",
    "discover_cuda_path",
    "preload_cuda_component_libraries",
]
