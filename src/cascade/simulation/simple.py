"""Standard solver orchestration for simple, lattice, and custom networks."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from cascade.concentration.external_field.coupling_steps import (
    _run_network_ext_frozen_step,
)
from cascade.concentration.external_field.geometry import _build_cext_geometry_context
from cascade.concentration.properties import get_concentration_inlet
from cascade.configuration.schema import RunConfig
from cascade.flow import PressureDropProblem, solve_pressure_drop
from cascade.flow.topology import _normalize_kirchhoff_bc_mode
from cascade.vessels.simple import SimpleNetwork, _simple_tree_data


def solve_simple_network(
    network: SimpleNetwork, ts, config: RunConfig
) -> SimpleNetwork:
    """Solve one explicit geometry with the standard flow and concentration kernels."""
    raw = dict(config.network.simple or {})
    fluid = config.simulation.fluid
    flow_ul_min = float(network.metadata["flow_ul_min"])
    q_inlet_cm3_s = flow_ul_min * 1.0e-3 / 60.0
    configured_inlet = network.metadata.get("concentration_inlet")
    inlet_conc = float(
        get_concentration_inlet(fluid) if configured_inlet is None else configured_inlet
    )
    solve_separate = bool(network.metadata.get("solve_channels_separately", True))

    if network.mode == "lattice" and str(
        getattr(ts, "KIRCHHOFF_SOLVER", "tree")
    ).strip().lower().startswith("tree"):
        print(
            "Lattice graph selected: switching Kirchhoff solver from tree to spsolve.",
            flush=True,
        )
        ts.KIRCHHOFF_SOLVER = "spsolve"

    mu_vals = ts.segment_viscosity_from_radius(
        network.radii,
        _fluid_mu_base(ts, fluid),
        fluid,
    )
    resistances = (8.0 * mu_vals * network.lengths) / (
        np.pi * np.maximum(network.radii, 1.0e-12) ** 4
    )
    diffusivity = float(raw.get("diffusivity", ts.SOLUTE_DIFFUSIVITY))
    vmax = float(raw.get("vmax", ts.VMAX_MM))
    km = float(raw.get("km", ts.K_M_MM))
    pressure_pressure_mode = _normalize_kirchhoff_bc_mode() == "pressure_pressure"
    if (
        not pressure_pressure_mode
        and
        solve_separate
        and network.starts.shape[0] > 1
        and network.mode == "multichannel"
    ):
        flows, cin, cout = _solve_independent_channels(
            ts,
            network.starts,
            network.radii,
            network.lengths,
            q_inlet_cm3_s,
            inlet_conc,
            fluid,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
        )
    else:
        if pressure_pressure_mode:
            solver = str(getattr(ts, "KIRCHHOFF_SOLVER", "spsolve"))
            if len(network.inlet_nodes) != 1:
                solver = "spsolve"
            pressure_result = solve_pressure_drop(
                PressureDropProblem(
                    proximal_nodes=network.prox_ids,
                    distal_nodes=network.dist_ids,
                    resistances=resistances,
                    inlet_nodes=network.inlet_nodes,
                    outlet_nodes=network.outlet_nodes,
                    outlet_pressure=float(ts.TERMINAL_PRESSURE)
                    * ts.PA_TO_DYN_PER_CM2,
                    pressure_drop=(
                        float(ts.ROOT_PRESSURE) - float(ts.TERMINAL_PRESSURE)
                    )
                    * ts.PA_TO_DYN_PER_CM2,
                    solver=solver,
                )
            )
            _pressures = pressure_result.pressures
            flows = pressure_result.flows_cm3_s
            q_inlet_cm3_s = pressure_result.inlet_flow_cm3_s
            network.metadata["flow_ul_min"] = q_inlet_cm3_s * 60000.0
        else:
            _pressures, flows, *_ = ts.solve_kirchhoff(
                network.prox_ids,
                network.dist_ids,
                resistances,
                network.inlet_nodes,
                q_inlet_cm3_s,
                network.outlet_nodes,
            )
        cin, cout, _node_conc, _history = ts.solve_network_concentrations(
            network.starts,
            network.ends,
            network.radii,
            network.lengths,
            flows,
            network.inlet_nodes,
            network.outlet_nodes,
            inlet_conc,
            fluid=fluid,
            prox_ids=network.prox_ids,
            dist_ids=network.dist_ids,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            omega=float(raw.get("omega", getattr(ts, "OMEGA", 0.5))),
        )

    network.flows = np.asarray(flows, dtype=float)
    network.vessel_quadrature = None
    if str(getattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")).strip().lower() == "graetz":
        network_topology = {
            "prox_ids": np.asarray(network.prox_ids, dtype=np.int64),
            "dist_ids": np.asarray(network.dist_ids, dtype=np.int64),
            "inlet_nodes": tuple(int(node) for node in network.inlet_nodes),
            "outlet_nodes": tuple(int(node) for node in network.outlet_nodes),
        }
        context = _build_cext_geometry_context(
            network,
            network.flows,
            network.starts,
            network.ends,
            network.radii,
            network.lengths,
            inlet_concentration=inlet_conc,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            build_candidate_index=False,
            network_topology=network_topology,
        )
        chb_max = np.zeros_like(network.radii, dtype=float)
        if str(fluid).lower() == "blood":
            for segment_id, radius in enumerate(network.radii):
                ht = ts.tube_hematocrit(radius, hd=ts.HD_DISCHARGE)
                chb_max[segment_id] = ts.segment_O2_capacity_from_HT(ht)
        ext_state = {
            "c_ext_gl": np.zeros(
                (network.segment_count, int(np.asarray(context["gl_t"]).size)),
                dtype=np.float32,
            )
        }
        cin, cout, _civ, _backend, _transfer = _run_network_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=inlet_conc,
            vmax=vmax,
            km=km,
            chb_max=chb_max,
            fluid_mode=str(fluid).lower(),
        )
        network.vessel_quadrature = {
            "solver": "simple_channel_graetz",
            "gl_points_si": np.asarray(context["gl_points_si"], dtype=np.float32),
            "c_iv_gl": np.asarray(ext_state["c_iv_gl"], dtype=np.float32),
            "c_bulk_gl": np.asarray(ext_state["c_bulk_gl"], dtype=np.float32),
            "c_wall_gl": np.asarray(ext_state["c_wall_gl"], dtype=np.float32),
            "c_ext_gl": np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
        }
    if "_pressures" in locals():
        network.node_pressures = np.asarray(_pressures, dtype=float)
        network.pressures = network.node_pressures[network.prox_ids]
    network.cin = np.asarray(cin, dtype=float)
    network.cout = np.asarray(cout, dtype=float)
    network.data = _simple_tree_data(
        network.starts,
        network.ends,
        network.radii,
        network.lengths,
        network.flows,
    )
    network.parameters = SimpleNamespace(
        root_pressure=float(getattr(ts, "ROOT_PRESSURE", np.nan)),
        terminal_pressure=float(getattr(ts, "TERMINAL_PRESSURE", np.nan)),
        root_flow=(
            float(q_inlet_cm3_s)
            if pressure_pressure_mode
            else float(np.sum(np.abs(network.flows)))
        ),
        terminal_flow=(
            float(np.mean(np.abs(network.flows))) if network.flows.size else np.nan
        ),
    )
    network.metadata["concentration_inlet"] = inlet_conc
    return network


def simple_details(
    network: SimpleNetwork,
    sample_points: np.ndarray,
    ts,
    config: RunConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute tissue results and summaries for an already solved simple network."""
    raw = dict(config.network.simple or {})
    inlet_concentration = float(network.metadata["concentration_inlet"])
    if sample_points.size:
        mask, tissue_conc = ts.compute_tissue_samples_greens(
            sample_points,
            network.tissue_starts,
            network.tissue_ends,
            network.radii,
            network.cin,
            network.flows,
            diffusivity=float(raw.get("diffusivity", ts.SOLUTE_DIFFUSIVITY)),
            vmax=float(raw.get("vmax", ts.VMAX_MM)),
            km=float(raw.get("km", ts.K_M_MM)),
            window_factor=float(raw.get("window_factor", ts.WINDOW_FACTOR)),
            inlet_concentration=inlet_concentration,
            tissue_cache=None,
        )
        keep = np.asarray(mask, dtype=bool)
        tissue_points = sample_points[keep]
        tissue_values = np.asarray(tissue_conc, dtype=float)[keep]
    else:
        tissue_points = np.empty((0, 3), dtype=float)
        tissue_values = np.empty((0,), dtype=float)

    total_volume = float(
        np.nansum(np.pi * network.radii * network.radii * network.lengths)
    )
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    mean_tissue = float(np.nanmean(tissue_values)) if tissue_values.size else np.nan
    total_flow = (
        float(network.metadata["flow_ul_min"]) / 60000.0
        if _normalize_kirchhoff_bc_mode() == "pressure_pressure"
        else float(np.nansum(np.abs(network.flows)))
    )
    summary = {
        "target_terminals": int(network.n_terminals),
        "cube_side_length": float(config.domain.side_length),
        "distance_sample_count": int(sample_points.shape[0]),
        "concentration_solver": "simple_channel",
        "finite_radius_o2_terms": str(getattr(ts, "FINITE_RADIUS_O2_TERMS", "none")),
        "lumen_wall_closure": str(getattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")),
        "total_volume": total_volume,
        "total_flowrate": total_flow,
        "pressure_in_root": float(getattr(network.parameters, "root_pressure", np.nan)),
        "pressure_out_terminals": float(
            getattr(network.parameters, "terminal_pressure", np.nan)
        ),
        "avg_radius": float(np.nanmean(network.radii))
        if network.radii.size
        else np.nan,
        "avg_length": float(np.nanmean(network.lengths))
        if network.lengths.size
        else np.nan,
        "total_length": float(np.nansum(network.lengths)),
        "avg_distance_to_channel": np.nan,
        "terminal_segments": int(network.n_terminals),
        "total_segments": int(network.segment_count),
        "dlp_angle": 0.0,
        "Rnet": np.nan,
        "dRnet": np.nan,
        "Qmin_over_Qinlet": (
            float(np.nanmin(np.abs(network.flows)) / max(total_flow, 1.0e-30))
            if network.flows.size
            else np.nan
        ),
        "C_LQ_over_Cmax": (
            float(np.nanmean(network.cout) / conc_max)
            if np.isfinite(conc_max) and conc_max
            else np.nan
        ),
        "C_tiss_over_Cmax": (
            mean_tissue / conc_max
            if np.isfinite(mean_tissue) and np.isfinite(conc_max) and conc_max
            else np.nan
        ),
        "inlet_flow_ul_per_min": float(network.metadata["flow_ul_min"]),
    }
    details = {
        "starts": network.starts,
        "ends": network.ends,
        "radii": network.radii,
        "lengths": network.lengths,
        "flows": network.flows,
        "pressures": np.asarray(
            getattr(network, "pressures", np.full(network.segment_count, np.nan)),
            dtype=float,
        ),
        "cin": network.cin,
        "cout": network.cout,
        "tissue_points": tissue_points,
        "tissue_values": tissue_values,
        "inlet_concentration": inlet_concentration,
        "vessel_quadrature": getattr(network, "vessel_quadrature", None),
    }
    return summary, details


def _solve_independent_channels(
    ts,
    starts,
    radii,
    lengths,
    q_inlet_cm3_s,
    inlet_conc,
    fluid,
    *,
    diffusivity,
    vmax,
    km,
):
    flows = np.full((starts.shape[0],), float(q_inlet_cm3_s), dtype=float)
    cin = np.full((starts.shape[0],), float(inlet_conc), dtype=float)
    cout = np.empty_like(cin)
    for i in range(starts.shape[0]):
        if str(fluid).lower() == "blood":
            ht = ts.tube_hematocrit(radii[i], hd=ts.HD_DISCHARGE)
            ccap = ts.segment_O2_capacity_from_HT(ht)
            decay = ts._blood_greens_decay_factor(
                flows[i] * ts.CM3_TO_M3,
                radii[i] * ts.CM_TO_M,
                lengths[i] * ts.CM_TO_M,
                float(diffusivity) * ts.CM2_TO_M2,
                float(vmax),
                float(km),
                cin[i],
                ccap,
            )
        else:
            decay = ts._greens_decay_factor(
                flows[i] * ts.CM3_TO_M3,
                radii[i] * ts.CM_TO_M,
                lengths[i] * ts.CM_TO_M,
                float(diffusivity) * ts.CM2_TO_M2,
                float(vmax),
                float(km),
                cin[i],
            )
        cout[i] = cin[i] * decay
    return flows, cin, cout


def _fluid_mu_base(ts, fluid: str) -> float:
    fluid_mode = str(fluid or "").lower()
    if fluid_mode in {"water", "cell media", "media"}:
        rho = 0.99336
        nu = 0.6959 / 100.0
        return rho * nu
    if fluid_mode == "blood":
        return float(getattr(ts, "MU_PLASMA_CGS", 0.012))
    if fluid_mode == "custom":
        return float(getattr(ts, "CUSTOM_FLUID_DYNAMIC_VISCOSITY_CP", 1.0)) / 100.0
    raise ValueError(f"Unsupported simple-network fluid: {fluid!r}")


__all__ = ["simple_details", "solve_simple_network"]
