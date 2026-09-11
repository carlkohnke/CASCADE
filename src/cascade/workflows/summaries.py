"""Compatibility names for tree simulation result aggregation.

The historical functions combine solver orchestration with result aggregation,
so their implementations now live in :mod:`cascade.workflows.tree_simulation`.
New workflow code should import from that module directly.
"""

from cascade.concentration.vessel.network import solve_network_concentrations

from .tree_simulation import summarize_tissue_only_from_details, summarize_tree

__all__ = [
    "solve_network_concentrations",
    "summarize_tree",
    "summarize_tissue_only_from_details",
]
