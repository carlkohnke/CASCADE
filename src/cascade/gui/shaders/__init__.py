"""Packaged GLSL shader resources used by the OpenGL preview."""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from pathlib import PurePosixPath


@cache
def load_shader_source(name: str) -> str:
    """Return one self-contained vertex or fragment shader."""
    path = PurePosixPath(name)
    if path.name != name or path.suffix not in {".vert", ".frag"}:
        raise ValueError(f"Invalid GLSL shader resource name: {name!r}")
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


__all__ = ["load_shader_source"]
