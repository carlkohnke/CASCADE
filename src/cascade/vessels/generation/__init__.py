"""SVV vascular architecture generation and compatibility adapters."""

from cascade.utils.lazy import resolve_export

_EXPORTS = {
    "Domain": ("cascade.vessels.generation.svv_adapter", "Domain"),
    "Forest": ("cascade.vessels.generation.svv_adapter", "Forest"),
    "Tree": ("cascade.vessels.generation.svv_adapter", "Tree"),
    "grow_tree": ("cascade.vessels.generation.tree_ops", "grow_tree"),
    "set_tree_fluid": ("cascade.vessels.generation.tree_ops", "set_tree_fluid"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
