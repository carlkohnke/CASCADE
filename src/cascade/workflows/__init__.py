"""High-level simulation, sweep, and heart workflows."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "SimulationResult": ("cascade.workflows.simulation", "SimulationResult"),
    "TreeSimulation": ("cascade.workflows.simulation", "TreeSimulation"),
    "run_simulation": ("cascade.workflows.simulation", "run_simulation"),
    "run_sweep": ("cascade.workflows.sweep", "run_sweep"),
    "run_tree_simulation": ("cascade.workflows.tree_simulation", "run_tree_simulation"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
