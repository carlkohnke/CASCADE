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
# CUDA 13 component wheels are unsuffixed and install below nvidia/cu13; CUDA
# 11 and 12 wheels carry a -cu11/-cu12 suffix and one directory per component.
_COMPONENT_SUFFIXES = {13: "", 12: "-cu12", 11: "-cu11"}
_CUPY_DISTRIBUTIONS = ("cupy-cuda13x", "cupy-cuda12x", "cupy-cuda11x", "cupy")
# CuPy 14 finds component wheels through cuda-pathfinder. CuPy 13, the last
# release for CUDA 11, links them by soname, so they must already be loaded.
# Pathfinder also prefers nvidia/cu13 over the CUDA 12 directories, so leftover
# CUDA 13 wheels would hand cupy-cuda12x a CUDA 13 NVRTC that rejects sm_70;
# loading this family's libraries first makes pathfinder reuse them.
_LINUX_PRELOAD = ("libnvJitLink.so*", "libcufft.so*")
_FAMILY_LINUX_PRELOAD = (
    "libnvJitLink.so*",
    "libnvrtc.so*",
    "libcudart.so*",
    "libcublasLt.so*",
    "libcublas.so*",
    "libcufft.so*",
    "libcurand.so*",
    "libcusparse.so*",
    "libcusolver.so*",
)
_FAMILY_WINDOWS_PRELOAD = (
    "nvJitLink_*.dll",
    "nvrtc64_*.dll",
    "cudart64_*.dll",
    "cublasLt64_*.dll",
    "cublas64_*.dll",
    "cufft64_*.dll",
    "curand64_*.dll",
    "cusparse64_*.dll",
    "cusolver64_*.dll",
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

    # CuPy 13 passes arch, cache_dir, extra_source and backend positionally.
    @functools.wraps(current)
    def compile_with_windows_paths(source, options=(), *args, **kwargs):
        safe_options = tuple(
            "-I" + _windows_short_path(option[2:])
            if option.startswith("-I") and not option.isascii()
            else option
            for option in options
        )
        return current(source, safe_options, *args, **kwargs)

    compile_with_windows_paths._cascade_windows_path_wrapper = True
    compiler._compile_module_with_cache = compile_with_windows_paths
    return True


def _is_component_library(path: Path) -> bool:
    name = path.name.lower()
    if os.name == "nt":
        return name.endswith(".dll")
    return ".so" in name


def installed_cupy_distributions() -> tuple[tuple[str, str], ...]:
    """Return ``(name, version)`` for each installed CuPy distribution."""
    found = []
    for name in _CUPY_DISTRIBUTIONS:
        try:
            found.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    return tuple(found)


def cupy_cuda_major(name: str) -> int | None:
    """Return the CUDA major version of a ``cupy-cudaNNx`` distribution name."""
    match = re.fullmatch(r"cupy-cuda(\d+)x", name)
    return int(match.group(1)) if match else None


def _installed_cupy() -> tuple[int | None, int | None]:
    """Return the installed CuPy's CUDA major and CuPy major version."""
    installed = installed_cupy_distributions()
    if len(installed) != 1:
        return None, None
    name, version = installed[0]
    match = re.match(r"\d+", version)
    return cupy_cuda_major(name), int(match.group()) if match else None


def _component_distributions():
    # Use only the CUDA family matching CuPy, so stale wheels from a previous
    # variant cannot shadow the libraries this CuPy was built against.
    cuda_major, _cupy_major = _installed_cupy()
    if cuda_major in _COMPONENT_SUFFIXES:
        suffixes = (_COMPONENT_SUFFIXES[cuda_major],)
    else:
        suffixes = tuple(_COMPONENT_SUFFIXES.values())
    for suffix in suffixes:
        for distribution_name in _COMPONENT_DISTRIBUTIONS:
            try:
                yield metadata.distribution(distribution_name + suffix)
            except metadata.PackageNotFoundError:
                continue


def foreign_component_distributions() -> tuple[str, ...]:
    """Return CUDA component wheels of a family other than the installed CuPy's."""
    cuda_major, _cupy_major = _installed_cupy()
    if cuda_major not in _COMPONENT_SUFFIXES:
        return ()
    found = []
    for family, suffix in _COMPONENT_SUFFIXES.items():
        if family == cuda_major:
            continue
        for distribution_name in _COMPONENT_DISTRIBUTIONS:
            try:
                metadata.distribution(distribution_name + suffix)
            except metadata.PackageNotFoundError:
                continue
            found.append(distribution_name + suffix)
    return tuple(found)


def _library_files(directories, patterns):
    """Yield the first library matching each pattern, in pattern order."""
    for library_pattern in patterns:
        library_path = next(
            (
                candidate
                for directory in directories
                for candidate in sorted(directory.glob(library_pattern))
                if candidate.is_file()
            ),
            None,
        )
        if library_path is not None:
            yield library_path


def _load_library(path: Path, **kwargs) -> None:
    try:
        _PRELOADED_LIBRARIES.append(ctypes.CDLL(str(path), **kwargs))
    except OSError as exc:
        raise RuntimeError(
            f"Could not load CUDA component library {path}: {exc}"
        ) from exc


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
    try:
        versioned = sorted(
            Path("/usr/local").glob("cuda-*"), key=_cuda_version_key, reverse=True
        )
    except OSError:
        versioned = []
    return (Path("/usr/local/cuda"), *versioned)


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
    # Cluster environment modules commonly export only CUDA_HOME.
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home).expanduser())
    candidates.extend(_system_cuda_roots())

    for candidate in candidates:
        if _has_cuda_headers(candidate):
            return candidate.resolve()
    return None


def preload_cuda_component_libraries() -> tuple[str, ...]:
    """Make component-wheel CUDA libraries visible to the current process."""
    directories = cuda_component_library_dirs()
    _cuda_major, cupy_major = _installed_cupy()
    legacy = cupy_major is not None and cupy_major < 14
    pin_family = legacy or bool(foreign_component_distributions())
    if os.name == "nt":
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is not None:
            for directory in directories:
                key = os.path.normcase(str(directory))
                if key in _DLL_DIRECTORY_PATHS:
                    continue
                _DLL_DIRECTORY_HANDLES.append(add_directory(str(directory)))
                _DLL_DIRECTORY_PATHS.add(key)
        if pin_family:
            for library_path in _library_files(directories, _FAMILY_WINDOWS_PRELOAD):
                _load_library(library_path)
        return tuple(str(path) for path in directories)

    if not sys.platform.startswith("linux"):
        return tuple(str(path) for path in directories)

    # nvJitLink must be globally visible before cuFFT on CUDA 13 Linux wheels.
    patterns = _FAMILY_LINUX_PRELOAD if pin_family else _LINUX_PRELOAD
    for library_path in _library_files(directories, patterns):
        _load_library(library_path, mode=ctypes.RTLD_GLOBAL)
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
    # The probe prints the device name; keep its output UTF-8 under any locale.
    probe_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
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
    "cupy_cuda_major",
    "discover_cuda_path",
    "foreign_component_distributions",
    "installed_cupy_distributions",
    "preload_cuda_component_libraries",
]
