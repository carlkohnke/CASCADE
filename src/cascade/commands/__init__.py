"""Command-line entry points; scientific work is delegated to workflows."""

from cascade.core.lazy import resolve_export

_EXPORTS = {"main": ("cascade.commands.main", "main")}
__all__ = ["main"]


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
