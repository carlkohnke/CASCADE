"""CSV, VTK, manifest, and scientific-report output."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "export_run": ("cascade.exporting.run", "export_run"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
