"""Packaged CUDA source resources for the CuPy/NVRTC runtime."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import PurePosixPath


def _validated_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.name != name or path.suffix not in {".cu", ".cuh"}:
        raise ValueError(f"Invalid CUDA resource name: {name!r}")
    return name


@cache
def _read_source(name: str) -> str:
    resource = files(__package__).joinpath(_validated_name(name))
    return resource.read_text(encoding="utf-8")


def load_cuda_source(name: str, *, prelude: str | None = None) -> str:
    """Load a CUDA translation unit, optionally prepending a shared header."""
    source = _read_source(name)
    if prelude is None:
        return source
    return _read_source(prelude).rstrip() + "\n\n" + source.lstrip()


__all__ = ["load_cuda_source"]
