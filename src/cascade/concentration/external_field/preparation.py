"""Prepare vascular-network state for external-field coupling."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np


def hematocrit_capacity(
    runtime: Any,
    network: Any,
    segment_count: int,
    flows: np.ndarray,
    *,
    fluid: str,
) -> np.ndarray:
    """Return oxygen-binding capacity for every segment."""
    if str(fluid).lower() != "blood":
        return np.zeros((int(segment_count),), dtype=np.float32)
    cached = runtime._get_tree_hematocrit_cache(
        network,
        int(segment_count),
        model=str(runtime.HEMATOCRIT_MODEL),
        flows=flows,
    )
    if cached is None:
        context = runtime._hematocrit_context_for_tree(network)
        discharge, tube = runtime.compute_tree_hematocrit(
            network,
            hd_root=float(runtime.HD_DISCHARGE),
            flows=flows,
            model=str(runtime.HEMATOCRIT_MODEL),
            order=np.asarray(context["order"], dtype=np.int64),
        )
        runtime._store_tree_hematocrit_cache(
            network,
            discharge,
            tube,
            model=str(runtime.HEMATOCRIT_MODEL),
            flows=flows,
        )
    else:
        _, tube = cached
    return np.asarray(tube, dtype=np.float32) * np.float32(
        float(runtime.O2_CAP_PER_HCT)
    )


def run_frozen_transport_step(
    runtime: Any,
    solution: dict,
    *,
    fluid: str,
) -> None:
    """Update vessel transport with a frozen external-field source state."""
    state = solution["cext_state"]
    context = solution["cext_context"]
    backend = "gpu"
    if hasattr(runtime, "_resolve_cext_frozen_accel_mode"):
        backend = str(runtime._resolve_cext_frozen_accel_mode())
    cin, cout, c_iv, backend, transfer = runtime._run_topdown_ext_frozen_step(
        context,
        state,
        inlet_concentration=float(solution["inlet_concentration"]),
        vmax=float(runtime.VMAX_MM),
        km=float(runtime.K_M_MM),
        chb_max=np.asarray(solution["chb_max"], dtype=np.float32),
        fluid_mode=str(fluid),
        frozen_backend=backend,
    )
    state["cin_seg"] = np.asarray(cin, dtype=np.float32)
    state["cout_seg"] = np.asarray(cout, dtype=np.float32)
    state["c_iv_gl"] = np.asarray(c_iv, dtype=np.float32)
    runtime._build_cext_iteration_cache(context, state)
    solution["cin"] = state["cin_seg"]
    solution["cout"] = state["cout_seg"]
    solution["cext_frozen_backend"] = str(backend)
    solution["cext_frozen_transfer_s"] = float(transfer)


def prepare_network_external_field(
    runtime: Any,
    transport_runtime: Any,
    network: Any,
    inlet_flow_cm3_s: float,
    *,
    fluid: str,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Solve flow and initialize one network for a shared Cext solve."""
    log = progress or (lambda _message: None)
    inlet_concentration = float(transport_runtime.get_concentration_inlet(fluid))
    log(
        f"Cext network prep: segments={int(network.segment_count)} "
        f"terminals={int(network.n_terminals)} qin={inlet_flow_cm3_s:.9g} cm3/s"
    )
    t0 = perf_counter()
    (
        flows,
        pressures,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        _prox_ids,
        _dist_ids,
    ) = transport_runtime.recompute_tree_flows(
        network, inlet_flow_cm3_s, fluid=fluid
    )
    log(f"flow recompute completed in {perf_counter() - t0:.2f}s")
    segment_count = int(np.asarray(starts).shape[0])
    context = runtime._build_cext_geometry_context(
        network,
        np.asarray(flows),
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(radii),
        np.asarray(lengths),
        inlet_concentration=inlet_concentration,
        diffusivity=float(transport_runtime.SOLUTE_DIFFUSIVITY),
        vmax=float(transport_runtime.VMAX_MM),
        km=float(transport_runtime.K_M_MM),
        build_candidate_index=False,
    )
    cache_key = (
        str(fluid),
        int(runtime.GL_ORDER_CEXT),
        float(inlet_concentration),
        float(transport_runtime.SOLUTE_DIFFUSIVITY),
        float(transport_runtime.VMAX_MM),
        float(transport_runtime.K_M_MM),
    )
    state = runtime._initialize_cext_state(
        network,
        cache_key=cache_key,
        nseg=segment_count,
        inlet_concentration=inlet_concentration,
        vmax=float(transport_runtime.VMAX_MM),
        km=float(transport_runtime.K_M_MM),
    )
    capacity = hematocrit_capacity(
        runtime,
        network,
        segment_count,
        np.asarray(flows),
        fluid=fluid,
    )
    p_in = (
        float(pressures[inlet_nodes[0]])
        if pressures.size and inlet_nodes
        else float("nan")
    )
    p_out = (
        float(np.mean(pressures[outlet_nodes]))
        if pressures.size and outlet_nodes
        else float("nan")
    )
    solution = {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(state["cin_seg"]),
        "cout": np.asarray(state["cout_seg"]),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
        "cext_context": context,
        "cext_state": state,
        "chb_max": capacity,
    }
    run_frozen_transport_step(runtime, solution, fluid=fluid)
    log(f"initial frozen transport and Cext geometry completed in {perf_counter() - t0:.2f}s")
    return solution


__all__ = [
    "hematocrit_capacity",
    "prepare_network_external_field",
    "run_frozen_transport_step",
]
