"""Vessel models, topology, architecture generation, and serialization."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "NetworkBuildResult": ("cascade.vessels.build", "NetworkBuildResult"),
    "SimpleNetwork": ("cascade.vessels.simple", "SimpleNetwork"),
    "build_or_load_network": ("cascade.vessels.build", "build_or_load_network"),
    "build_simple_network": ("cascade.vessels.simple", "build_simple_network"),
    "channel_count": ("cascade.vessels.lattice", "channel_count"),
    "connectivity_report": ("cascade.vessels.connectivity", "connectivity_report"),
    "generate_lattice": ("cascade.vessels.lattice", "generate_lattice"),
    "repair_tree_parent_columns": ("cascade.vessels.connectivity", "repair_tree_parent_columns"),
    "validate_trees": ("cascade.vessels.connectivity", "validate_trees"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
