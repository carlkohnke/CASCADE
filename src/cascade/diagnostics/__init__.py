"""Environment inspection and runtime diagnostics."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "collect_diagnostics": ("cascade.diagnostics.environment", "collect_diagnostics"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
