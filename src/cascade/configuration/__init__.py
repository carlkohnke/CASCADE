"""Typed configuration models and runtime bridges."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "RunConfig": ("cascade.configuration.models", "RunConfig"),
    "RuntimeConfiguration": ("cascade.configuration.runtime", "RuntimeConfiguration"),
    "apply_runtime_settings": ("cascade.configuration.bridge", "apply_runtime_settings"),
    "load_config": ("cascade.configuration.models", "load_config"),
    "load_runtime_module": ("cascade.configuration.bridge", "load_runtime_module"),
    "parse_config": ("cascade.configuration.models", "parse_config"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
