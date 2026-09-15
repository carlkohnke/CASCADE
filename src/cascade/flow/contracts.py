"""Validated input and result contracts for flow solving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _vector(value: Any, name: str, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class FlowProblem:
    """Network resistance problem with prescribed total inlet flow."""

    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray
    resistances: np.ndarray
    inlet_nodes: Sequence[int]
    outlet_nodes: Sequence[int]
    inlet_flow_cm3_s: float
    node_count: int | None = None
    solver: str = "tree"
    boundary_condition: str = "legacy_equal_terminal_flow"

    def __post_init__(self) -> None:
        proximal = _vector(self.proximal_nodes, "proximal_nodes", dtype=np.int64)
        distal = _vector(self.distal_nodes, "distal_nodes", dtype=np.int64)
        resistance = _vector(self.resistances, "resistances", dtype=float)
        if not (proximal.size == distal.size == resistance.size):
            raise ValueError(
                "Flow topology and resistance arrays must have equal lengths."
            )
        if proximal.size == 0:
            raise ValueError("A flow problem requires at least one segment.")
        if np.any(proximal < 0) or np.any(distal < 0):
            raise ValueError("Flow node indices must be nonnegative.")
        if np.any(resistance <= 0.0):
            raise ValueError("Flow resistances must be strictly positive.")
        inlet_flow = float(self.inlet_flow_cm3_s)
        if not np.isfinite(inlet_flow) or inlet_flow <= 0.0:
            raise ValueError("inlet_flow_cm3_s must be finite and strictly positive.")
        inlet_nodes = tuple(int(value) for value in self.inlet_nodes)
        outlet_nodes = tuple(int(value) for value in self.outlet_nodes)
        if not inlet_nodes:
            raise ValueError("A flow problem requires at least one inlet node.")
        if not outlet_nodes:
            raise ValueError("A flow problem requires at least one outlet node.")
        inferred_node_count = int(max(int(proximal.max()), int(distal.max())) + 1)
        node_count = (
            inferred_node_count if self.node_count is None else int(self.node_count)
        )
        if node_count < inferred_node_count:
            raise ValueError(
                f"node_count={node_count} does not include topology node "
                f"{inferred_node_count - 1}."
            )
        if any(node < 0 or node >= node_count for node in inlet_nodes + outlet_nodes):
            raise ValueError("Flow boundary node indices must lie within node_count.")
        solver = str(self.solver).strip().lower().replace("-", "_")
        solver = {"tree_neumann": "tree", "tree_current_bc": "tree"}.get(solver, solver)
        if solver not in {"tree", "auto", "cg", "gmres_ilu", "spsolve"}:
            raise ValueError(
                "solver must be one of 'tree', 'auto', 'cg', 'gmres_ilu', or 'spsolve'."
            )
        boundary_condition = (
            str(self.boundary_condition).strip().lower().replace("-", "_")
        )
        if boundary_condition not in {
            "legacy_equal_terminal_flow",
            "terminal_pressure",
        }:
            raise ValueError(
                "boundary_condition must be 'legacy_equal_terminal_flow' or "
                "'terminal_pressure'."
            )
        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)
        object.__setattr__(self, "resistances", resistance)
        object.__setattr__(self, "inlet_nodes", inlet_nodes)
        object.__setattr__(self, "outlet_nodes", outlet_nodes)
        object.__setattr__(self, "inlet_flow_cm3_s", inlet_flow)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "boundary_condition", boundary_condition)


@dataclass(frozen=True)
class FlowResult:
    """Pressures and signed segment flows returned by the flow subsystem."""

    pressures: np.ndarray
    flows_cm3_s: np.ndarray
    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray


@dataclass(frozen=True)
class PressureDropProblem:
    """Network problem with prescribed inlet and outlet pressures.

    Pressure values must use the pressure unit implied by ``resistances``.  For
    CASCADE's CGS vessel resistances this is dyn/cm², and the resulting flows
    are cm³/s.
    """

    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray
    resistances: np.ndarray
    inlet_nodes: Sequence[int]
    outlet_nodes: Sequence[int]
    outlet_pressure: float = 0.0
    pressure_drop: float | None = None
    inlet_pressure: float | None = None
    node_count: int | None = None
    solver: str = "tree"

    def __post_init__(self) -> None:
        proximal = _vector(self.proximal_nodes, "proximal_nodes", dtype=np.int64)
        distal = _vector(self.distal_nodes, "distal_nodes", dtype=np.int64)
        resistance = _vector(self.resistances, "resistances", dtype=float)
        if not (proximal.size == distal.size == resistance.size):
            raise ValueError(
                "Flow topology and resistance arrays must have equal lengths."
            )
        if proximal.size == 0:
            raise ValueError("A pressure-drop problem requires at least one segment.")
        if np.any(proximal < 0) or np.any(distal < 0):
            raise ValueError("Flow node indices must be nonnegative.")
        if np.any(resistance <= 0.0):
            raise ValueError("Flow resistances must be strictly positive.")

        outlet_pressure = float(self.outlet_pressure)
        if not np.isfinite(outlet_pressure):
            raise ValueError("outlet_pressure must be finite.")
        if self.pressure_drop is None:
            if self.inlet_pressure is None:
                raise ValueError("Either pressure_drop or inlet_pressure is required.")
            inlet_pressure = float(self.inlet_pressure)
            pressure_drop = inlet_pressure - outlet_pressure
        else:
            pressure_drop = float(self.pressure_drop)
            inlet_pressure = outlet_pressure + pressure_drop
            if self.inlet_pressure is not None and not np.isclose(
                float(self.inlet_pressure), inlet_pressure, rtol=1.0e-12, atol=0.0
            ):
                raise ValueError(
                    "inlet_pressure must equal outlet_pressure + pressure_drop."
                )
        if not np.isfinite(inlet_pressure) or not np.isfinite(pressure_drop):
            raise ValueError("Inlet pressure and pressure drop must be finite.")
        if pressure_drop <= 0.0:
            raise ValueError("pressure_drop must be strictly positive.")

        inlet_nodes = tuple(dict.fromkeys(int(value) for value in self.inlet_nodes))
        outlet_nodes = tuple(dict.fromkeys(int(value) for value in self.outlet_nodes))
        if not inlet_nodes:
            raise ValueError("A pressure-drop problem requires at least one inlet node.")
        if not outlet_nodes:
            raise ValueError("A pressure-drop problem requires at least one outlet node.")
        if set(inlet_nodes).intersection(outlet_nodes):
            raise ValueError("Inlet and outlet node sets must not overlap.")

        inferred_node_count = int(max(int(proximal.max()), int(distal.max())) + 1)
        node_count = (
            inferred_node_count if self.node_count is None else int(self.node_count)
        )
        if node_count < inferred_node_count:
            raise ValueError(
                f"node_count={node_count} does not include topology node "
                f"{inferred_node_count - 1}."
            )
        if any(node < 0 or node >= node_count for node in inlet_nodes + outlet_nodes):
            raise ValueError("Flow boundary node indices must lie within node_count.")

        solver = str(self.solver).strip().lower().replace("-", "_")
        solver = {"tree_neumann": "tree", "tree_current_bc": "tree"}.get(
            solver, solver
        )
        if solver not in {"tree", "auto", "cg", "gmres_ilu", "spsolve"}:
            raise ValueError(
                "solver must be one of 'tree', 'auto', 'cg', 'gmres_ilu', or 'spsolve'."
            )

        object.__setattr__(self, "proximal_nodes", proximal)
        object.__setattr__(self, "distal_nodes", distal)
        object.__setattr__(self, "resistances", resistance)
        object.__setattr__(self, "inlet_nodes", inlet_nodes)
        object.__setattr__(self, "outlet_nodes", outlet_nodes)
        object.__setattr__(self, "inlet_pressure", inlet_pressure)
        object.__setattr__(self, "outlet_pressure", outlet_pressure)
        object.__setattr__(self, "pressure_drop", pressure_drop)
        object.__setattr__(self, "node_count", node_count)
        object.__setattr__(self, "solver", solver)

@dataclass(frozen=True)
class PressureDropResult:
    """Direct pressure-pressure solution and its emergent boundary flow."""

    pressures: np.ndarray
    flows_cm3_s: np.ndarray
    proximal_nodes: np.ndarray
    distal_nodes: np.ndarray
    inlet_flow_cm3_s: float
    outlet_flow_cm3_s: float
    node_net_outflows_cm3_s: np.ndarray


__all__ = [
    "FlowProblem",
    "FlowResult",
    "PressureDropProblem",
    "PressureDropResult",
]
