"""Thin command-line entry points for the unified simulation platform."""

from cascade.utils.lazy import resolve_export

_EXPORTS = {"main": ("cascade.commands.main", "main")}
__all__ = ["main"]


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
