"""Canonical CASCADE case execution and batch orchestration."""

from cascade.core.lazy import resolve_export

_EXPORTS = {
    "SimulationResult": ("cascade.simulation.engine", "SimulationResult"),
    "TreeSimulation": ("cascade.simulation.engine", "TreeSimulation"),
    "run_simulation": ("cascade.simulation.engine", "run_simulation"),
    "run_sweep": ("cascade.simulation.sweep", "run_sweep"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
