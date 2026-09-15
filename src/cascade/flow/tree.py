"""Tree geometry assembly and flow recomputation."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.flow.hematocrit import _tree_exact_connectivity
from cascade.flow.kirchhoff import solve_kirchhoff
from cascade.flow.api import solve_pressure_drop
from cascade.flow.contracts import PressureDropProblem
from cascade.flow.rheology import (
    segment_viscosity_from_radius,
    segment_viscosity_from_radius_hd,
)
from cascade.flow.topology import _build_node_indices
from cascade.flow.topology import _normalize_kirchhoff_bc_mode
from cascade.vessels.generation.tree_ops import set_tree_fluid


def assemble_tree_segments(
    tree: _state.Tree,
    fluid: str,
    hd_per_segment: np.ndarray | None = None,
    *,
    compute_viscosity: bool = True,
    reuse_geometry: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
    np.ndarray,
    np.ndarray,
]:
    """Return flow-ready segment geometry, viscosity, and exact topology."""
    set_tree_fluid(tree, fluid)
    segment_count = int(getattr(tree, "segment_count", 0))
    tree_data = np.asarray(tree.data)
    data_address = int(tree_data.__array_interface__["data"][0])
    cached = getattr(tree, "_cascade_assembly_geometry", None)
    cache_valid = (
        reuse_geometry
        and isinstance(cached, dict)
        and int(cached.get("segment_count", -1)) == segment_count
        and int(cached.get("data_address", -1)) == data_address
    )
    if cache_valid:
        starts = cached["starts"]
        ends = cached["ends"]
        radii = cached["radii"]
        lengths = cached["lengths"]
        inlet_nodes = cached["inlet_nodes"]
        outlet_nodes = cached["outlet_nodes"]
        proximal_ids = cached["proximal_ids"]
        distal_ids = cached["distal_ids"]
    else:
        data = np.asarray(tree_data[:segment_count])
        starts = data[:, 0:3]
        ends = data[:, 3:6]
        radii = np.asarray(data[:, 21], dtype=float)
        lengths = np.asarray(data[:, 20], dtype=float).copy()
        bad_lengths = ~np.isfinite(lengths) | (lengths <= 0.0)
        if np.any(bad_lengths):
            lengths[bad_lengths] = np.linalg.norm(
                ends[bad_lengths] - starts[bad_lengths], axis=1
            )

        exact_node_ids = getattr(tree, "_cascade_node_ids", None)
        if exact_node_ids is not None and np.asarray(exact_node_ids).shape == (
            segment_count,
            2,
        ):
            exact_node_ids = np.asarray(exact_node_ids, dtype=np.int64)
            proximal_raw = exact_node_ids[:, 0]
            distal_raw = exact_node_ids[:, 1]
            have_node_ids = bool(np.all(proximal_raw >= 0) and np.all(distal_raw >= 0))
        else:
            proximal_raw = np.asarray(data[:, 18], dtype=float).reshape(-1)
            distal_raw = np.asarray(data[:, 19], dtype=float).reshape(-1)
            have_node_ids = bool(
                proximal_raw.size == segment_count
                and distal_raw.size == segment_count
                and np.all(np.isfinite(proximal_raw))
                and np.all(np.isfinite(distal_raw))
            )

        if have_node_ids:
            proximal_ids = proximal_raw.astype(np.int64, copy=False)
            distal_ids = distal_raw.astype(np.int64, copy=False)
        else:
            geometry = np.zeros((starts.shape[0], 6), dtype=float)
            geometry[:, 0:3] = starts
            geometry[:, 3:6] = ends
            proximal_ids, distal_ids, _ = _build_node_indices(geometry)

        inlet_nodes = [int(proximal_ids[0])] if proximal_ids.size else []
        connectivity = _tree_exact_connectivity(tree, data)
        terminal = (connectivity[:, 0] < 0) & (connectivity[:, 1] < 0)
        outlet_nodes = [int(value) for value in distal_ids[terminal]]
        if reuse_geometry:
            # Own the arrays so transient tree views and reversible solve-time
            # edits cannot change a cache behind our back.
            cached = {
                "segment_count": segment_count,
                "data_address": data_address,
                "starts": np.array(starts, copy=True),
                "ends": np.array(ends, copy=True),
                "radii": np.array(radii, copy=True),
                "lengths": np.array(lengths, copy=True),
                "inlet_nodes": list(inlet_nodes),
                "outlet_nodes": list(outlet_nodes),
                "proximal_ids": np.array(proximal_ids, copy=True),
                "distal_ids": np.array(distal_ids, copy=True),
            }
            try:
                tree._cascade_assembly_geometry = cached
            except Exception:
                pass
            starts = cached["starts"]
            ends = cached["ends"]
            radii = cached["radii"]
            lengths = cached["lengths"]
            proximal_ids = cached["proximal_ids"]
            distal_ids = cached["distal_ids"]

    if not compute_viscosity:
        # The caller will calculate viscosity from its converged hematocrit.
        # Avoid an O(N) nonlinear rheology pass whose values would be discarded.
        viscosity = np.empty(radii.shape, dtype=float)
    else:
        base_viscosity = float(tree.parameters.fluid_density) * float(
            tree.parameters.kinematic_viscosity
        )
        if hd_per_segment is None:
            viscosity = segment_viscosity_from_radius(
                radii, base_viscosity, fluid
            )
        else:
            viscosity = segment_viscosity_from_radius_hd(
                radii, base_viscosity, fluid, hd_per_segment
            )

    return (
        starts,
        ends,
        radii,
        lengths,
        viscosity,
        inlet_nodes,
        outlet_nodes,
        proximal_ids,
        distal_ids,
    )


def recompute_tree_flows(
    tree: _state.Tree,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    list[int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Solve pressures and flows for a tree using its current geometry."""
    (
        starts,
        ends,
        radii,
        lengths,
        viscosity,
        inlet_nodes,
        outlet_nodes,
        proximal_ids,
        distal_ids,
    ) = assemble_tree_segments(tree, fluid)
    if starts.size == 0:
        empty = np.empty((0,), dtype=float)
        return (
            empty,
            empty,
            inlet_nodes,
            outlet_nodes,
            starts,
            ends,
            radii,
            lengths,
            proximal_ids,
            distal_ids,
        )

    safe_radii = np.maximum(radii, 1e-12)
    resistances = (8.0 * viscosity * lengths) / (np.pi * safe_radii**4)
    pressure_pressure_mode = _normalize_kirchhoff_bc_mode() == "pressure_pressure"
    if pressure_pressure_mode:
        result = solve_pressure_drop(
            PressureDropProblem(
                proximal_nodes=proximal_ids,
                distal_nodes=distal_ids,
                resistances=resistances,
                inlet_nodes=inlet_nodes,
                outlet_nodes=outlet_nodes,
                outlet_pressure=float(tree.parameters.terminal_pressure)
                * _state.PA_TO_DYN_PER_CM2,
                pressure_drop=(
                    float(tree.parameters.root_pressure)
                    - float(tree.parameters.terminal_pressure)
                )
                * _state.PA_TO_DYN_PER_CM2,
                solver=str(_state.KIRCHHOFF_SOLVER),
            )
        )
        pressures = result.pressures
        flows = result.flows_cm3_s
    else:
        pressures, flows, _, _, _ = solve_kirchhoff(
            proximal_ids,
            distal_ids,
            resistances,
            inlet_nodes,
            inlet_flow_cm3_s,
            outlet_nodes,
        )
    if pressures.size and inlet_nodes and not pressure_pressure_mode:
        target = float(tree.parameters.root_pressure) * _state.PA_TO_DYN_PER_CM2
        pressures = pressures + target - float(pressures[inlet_nodes[0]])
    return (
        flows,
        pressures,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        proximal_ids,
        distal_ids,
    )


__all__ = ["assemble_tree_segments", "recompute_tree_flows"]
