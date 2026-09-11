"""Small helper for dependency-safe package-level exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def resolve_export(namespace: dict[str, Any], exports: dict[str, tuple[str, str]], name: str) -> Any:
    """Resolve and cache one explicitly declared package export."""
    try:
        module_name, attribute = exports[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    namespace[name] = value
    return value
