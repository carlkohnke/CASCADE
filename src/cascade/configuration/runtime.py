"""Resolved runtime configuration passed through the CASCADE workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cascade.configuration.settings import (
    apply_settings,
    collect_config_settings,
    default_settings,
)

# Runtime module id -> (module, import-time values of its uppercase constants).
_CONSTANT_SNAPSHOTS: dict[int, tuple[Any, dict[str, Any]]] = {}


def _copy_constant(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:
        return value


def snapshot_runtime_constants(runtime: Any) -> None:
    """Record the runtime's uppercase constants once, before any run overrides them."""
    if id(runtime) in _CONSTANT_SNAPSHOTS:
        return
    values = {
        name: _copy_constant(getattr(runtime, name))
        for name in list(vars(runtime))
        if name.isupper() and not name.startswith("_")
    }
    _CONSTANT_SNAPSHOTS[id(runtime)] = (runtime, values)


def restore_runtime_constants(runtime: Any) -> None:
    """Return every snapshotted constant, including ``settings.runtime`` overrides."""
    snapshot_runtime_constants(runtime)
    for name, value in _CONSTANT_SNAPSHOTS[id(runtime)][1].items():
        setattr(runtime, name, _copy_constant(value))


@dataclass(frozen=True)
class RuntimeConfiguration:
    """Immutable view of the settings resolved for one simulation run."""

    fluid: str
    build_fluid: str
    inlet_flow_ul_min: float
    distance_sample_count: int
    compute_average_distance: bool
    concentration_solver: str
    equal_bifurcations: int | None
    sections: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_run_config(cls, config: Any) -> "RuntimeConfiguration":
        simulation = config.simulation
        growth = config.growth
        sections = {
            name: dict(values)
            for name, values in collect_config_settings(config).items()
        }
        return cls(
            fluid=str(simulation.fluid),
            build_fluid=str(simulation.build_fluid),
            inlet_flow_ul_min=float(simulation.qin_target_ul_min),
            distance_sample_count=int(simulation.distance_sample_count),
            compute_average_distance=bool(simulation.compute_avg_distance_to_channel),
            concentration_solver=str(simulation.concentration_solver),
            equal_bifurcations=growth.n_equal_bifurcations,
            sections=sections,
        )

    def apply_compatibility_state(self, runtime: Any) -> dict[str, Any]:
        """Apply typed settings to the shared TissueSim runtime state."""
        # GUI workers and batch jobs share this state across runs. Restore even
        # nullable defaults before applying this run's typed and explicit values.
        # ``settings.runtime`` may override any constant, not only registered
        # settings, so the first-run snapshot is restored as well.
        restore_runtime_constants(runtime)
        for values in default_settings().values():
            for name, value in values.items():
                setattr(runtime, name, deepcopy(value))
        runtime.FLUID = self.fluid
        runtime.ACTIVE_FLUID = self.fluid
        runtime.BUILD_FLUID = self.build_fluid
        runtime.QIN_TARGET = self.inlet_flow_ul_min
        runtime.DISTANCE_SAMPLE_COUNT = self.distance_sample_count
        runtime.COMPUTE_AVG_DISTANCE_TO_CHANNEL = self.compute_average_distance
        runtime.CONCENTRATION_SOLVER = self.concentration_solver
        runtime.N_EQUAL_BIFURCATIONS = self.equal_bifurcations
        runtime.TREE_DATA_DTYPE = np.float64
        runtime.TREE_INDEX_DTYPE = np.int64
        return apply_settings(runtime, self.sections)


__all__ = [
    "RuntimeConfiguration",
    "restore_runtime_constants",
    "snapshot_runtime_constants",
]
