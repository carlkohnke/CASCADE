"""End-to-end tree hemodynamics, concentration, and tissue simulation.

Tree storage is converted to common geometry/topology arrays before flow,
vessel transport, Cext, and tissue solvers run in physical dependency order.
The returned details preserve the exact arrays used to calculate run metrics.
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, Optional

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.concentration.external_field.state import _clear_cext_runtime_state
from cascade.concentration.tissue.greens import (
    compute_tissue_samples_greens,
    compute_tissue_samples_greens_from_cext_state,
)
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.vessel.dispatch import (
    _resolve_concentration_solver,
    _solve_channel_concentrations,
)
from cascade.diagnostics.runtime import _resolve_tissue_accel_mode
from cascade.domain.sampling import _characteristic_length, compute_average_distance
from cascade.flow.hematocrit import (
    _hematocrit_context_for_tree,
    _normalize_hematocrit_model,
    _store_tree_hematocrit_cache,
    _tree_exact_connectivity,
    _tube_hematocrit_from_hd_radius,
    compute_tree_hematocrit,
)
from cascade.flow.kirchhoff import (
    _fixed_terminal_flows_numba,
    _pressures_from_tree_flows_numba,
    solve_kirchhoff,
)
from cascade.flow.rheology import segment_viscosity_from_radius_hd
from cascade.flow.topology import _normalize_kirchhoff_bc_mode

from cascade.flow.tree import assemble_tree_segments


def run_tree_simulation(
    tree: _state.Tree,
    sample_points: np.ndarray,
    target_terminals: int,
    *,
    side_length: Optional[float] = None,
    fluid: Optional[str] = None,
    inlet_flow_cm3_s: Optional[float] = None,
    inlet_concentration_override: Optional[float] = None,
    tissue_cache: dict | None = None,
    concentration_solver: str | None = None,
    return_details: bool = False,
) -> Dict[str, float] | tuple[Dict[str, float], dict]:
    """Run the complete flow, vessel-transport, and tissue pipeline for one tree.

    Geometry is normalized once, then the selected hematocrit/flow model is
    solved before concentration transport. Tissue oxygen is evaluated from the
    resulting vessel state, and the metrics describe that same solution.
    """

    hematocrit_model = _normalize_hematocrit_model()
    seg_count = int(getattr(tree, "segment_count", 0))
    if seg_count == 0:
        raise RuntimeError("Tree contains no segments; cannot summarize.")

    n_terminals = max(int(getattr(tree, "n_terminals", 0)) - 1, 0)
    flow_inlet = inlet_flow_cm3_s
    if flow_inlet is None or not np.isfinite(flow_inlet):
        q_scale = (
            (side_length**3)
            if (side_length is not None and _state.SCALE_Q_BY_VOLUME)
            else 1.0
        )
        flow_inlet = float(_state.QIN_TARGET) * 1e-3 / 60.0 * q_scale

    analysis_fluid = fluid or _state.ACTIVE_FLUID
    inlet_concentration = (
        float(inlet_concentration_override)
        if inlet_concentration_override is not None
        and np.isfinite(float(inlet_concentration_override))
        else get_concentration_inlet(analysis_fluid)
    )
    concentration_solver_mode = _resolve_concentration_solver(concentration_solver)

    # Normalize SVV storage into contiguous geometry and topology arrays shared
    # by every downstream solver backend.
    t_assemble = perf_counter()
    (
        starts_arr,
        ends_arr,
        radii_arr,
        lengths_arr,
        mu_arr,
        inlet_nodes,
        outlet_nodes,
        prox_ids,
        dist_ids,
    ) = assemble_tree_segments(tree, analysis_fluid)
    t_assemble = perf_counter() - t_assemble

    total_length = float(np.nansum(lengths_arr))
    total_volume = float(np.nansum(np.pi * radii_arr**2 * lengths_arr))
    avg_radius = float(np.nanmean(radii_arr)) if radii_arr.size else float("nan")
    avg_length = float(np.nanmean(lengths_arr)) if lengths_arr.size else float("nan")
    pressure_in = float(tree.parameters.root_pressure)
    pressure_out = float(tree.parameters.terminal_pressure)
    pressure_drop = pressure_in - pressure_out

    # Hemodynamics may be a single Kirchhoff solve or an outer hematocrit/
    # viscosity fixed point; both paths produce the same pressure/flow contract.
    t_kirchhoff = perf_counter()
    hd_flow_iter = np.empty((0,), dtype=float)
    t_hematocrit_flow_iter = 0.0
    hematocrit_iter_log: list[dict[str, float]] = []
    hematocrit_iterations_completed = 0
    if starts_arr.size == 0:
        empty = np.empty((0,), dtype=float)
        flows = empty
        pressures = empty
    else:
        radius_safe = np.maximum(radii_arr, 1e-12)
        fixed_flow_tree_shortcut = (
            _state._HAVE_NUMBA
            and str(_state.KIRCHHOFF_SOLVER).strip().lower()
            in ("tree", "tree_neumann", "tree-current-bc")
            and _normalize_kirchhoff_bc_mode() == "legacy_equal_terminal_flow"
            and inlet_nodes
            and flow_inlet is not None
            and np.isfinite(flow_inlet)
        )
        flows_fixed = None
        fixed_flow_s = 0.0
        if fixed_flow_tree_shortcut:
            t_fixed_flow = perf_counter()
            hct_context_fixed = _hematocrit_context_for_tree(tree)
            flows_fixed, _, fixed_term_count = _fixed_terminal_flows_numba(
                np.asarray(hct_context_fixed["order"], dtype=np.int64),
                np.asarray(hct_context_fixed["left_child"], dtype=np.int64),
                np.asarray(hct_context_fixed["right_child"], dtype=np.int64),
                float(flow_inlet),
            )
            flows_fixed = np.asarray(flows_fixed, dtype=float)
            fixed_flow_s = perf_counter() - t_fixed_flow
            if _state.KIRCHHOFF_DIAGNOSTICS:
                print(
                    "Kirchhoff diagnostics: solver_used=tree_fixed_equal_terminal_flow "
                    f"bc=legacy_equal_terminal_flow flow_time={fixed_flow_s:.3f}s terminals={int(fixed_term_count)}"
                )
        if (
            str(analysis_fluid).lower() == "blood"
            and hematocrit_model == "pries_secomb"
            and int(_state.HEMATOCRIT_FLOW_ITERATIONS) > 0
        ):
            t_hct_iter = perf_counter()
            hd_iter = np.full(
                starts_arr.shape[0], float(_state.HD_DISCHARGE), dtype=float
            )
            flows_old_iter = np.zeros(starts_arr.shape[0], dtype=float)
            rho = float(tree.parameters.fluid_density)
            nu = float(tree.parameters.kinematic_viscosity)
            mu_base = rho * nu
            iter_count = max(int(_state.HEMATOCRIT_FLOW_ITERATIONS), 0)
            relax = float(np.clip(_state.HEMATOCRIT_RELAXATION, 0.0, 1.0))
            qtol_cgs = float(_state.HEMATOCRIT_QTOL_NL_MIN) * 1.0e-6 / 60.0
            for h_iter in range(iter_count):
                if (h_iter + 1) % 5 == 0:
                    relax *= 0.8
                mu_iter = segment_viscosity_from_radius_hd(
                    radii_arr, mu_base, analysis_fluid, hd_iter
                )
                resistances_iter = (8.0 * mu_iter * lengths_arr) / (
                    np.pi * radius_safe**4
                )
                if flows_fixed is not None:
                    flows_iter = flows_fixed
                else:
                    pressures_iter, flows_iter, _, _, _ = solve_kirchhoff(
                        prox_ids,
                        dist_ids,
                        resistances_iter,
                        inlet_nodes,
                        flow_inlet,
                        outlet_nodes,
                    )
                hd_new, _ = compute_tree_hematocrit(
                    tree,
                    hd_root=_state.HD_DISCHARGE,
                    flows=flows_iter,
                    model=hematocrit_model,
                )
                delta_hd = (
                    float(np.nanmax(np.abs(hd_new - hd_iter))) if hd_new.size else 0.0
                )
                delta_q = (
                    float(np.nanmax(np.abs(flows_iter - flows_old_iter)))
                    if flows_iter.size
                    else 0.0
                )
                hd_iter = (1.0 - relax) * hd_iter + relax * hd_new
                flows_old_iter = np.asarray(flows_iter, dtype=float).copy()
                hematocrit_iterations_completed = h_iter + 1
                hematocrit_iter_log.append(
                    {
                        "iteration": float(h_iter + 1),
                        "relax": float(relax),
                        "max_q_delta_cm3_s": float(delta_q),
                        "max_hd_delta": float(delta_hd),
                    }
                )
                if _state.HEMATOCRIT_DIAGNOSTICS:
                    finite_hd = hd_iter[np.isfinite(hd_iter)]
                    h50 = (
                        float(np.nanmedian(finite_hd))
                        if finite_hd.size
                        else float("nan")
                    )
                    print(
                        f"  Hematocrit flow iteration {h_iter + 1}/{iter_count}: "
                        f"relax={relax:.3f} max_q_delta={delta_q:.3e} max_hd_delta={delta_hd:.3e} HD_median={h50:.3f}"
                    )
                if delta_q < qtol_cgs and delta_hd < float(_state.HEMATOCRIT_HDTOL):
                    if _state.HEMATOCRIT_DIAGNOSTICS:
                        print(
                            f"  Hematocrit flow iterations converged at {h_iter + 1}/{iter_count}: "
                            f"max_q_delta={delta_q:.3e} max_hd_delta={delta_hd:.3e}"
                        )
                    break
            mu_arr = segment_viscosity_from_radius_hd(
                radii_arr, mu_base, analysis_fluid, hd_iter
            )
            hd_flow_iter = hd_iter
            _store_tree_hematocrit_cache(
                tree,
                hd_iter,
                _tube_hematocrit_from_hd_radius(radii_arr, hd_iter),
                model=hematocrit_model,
                flows=flows_old_iter,
                fixed_flow_bc=(
                    _normalize_kirchhoff_bc_mode() == "legacy_equal_terminal_flow"
                ),
            )
            t_hematocrit_flow_iter = perf_counter() - t_hct_iter
        resistances = (8.0 * mu_arr * lengths_arr) / (np.pi * radius_safe**4)
        if flows_fixed is not None:
            flows = np.asarray(flows_fixed, dtype=float)
            num_nodes = (
                int(max(int(np.max(prox_ids)), int(np.max(dist_ids))) + 1)
                if prox_ids.size
                else 0
            )
            root_pressure_dyn = (
                float(tree.parameters.root_pressure) * _state.PA_TO_DYN_PER_CM2
            )
            t_pressure = perf_counter()
            pressures, assigned_edges = _pressures_from_tree_flows_numba(
                np.asarray(_hematocrit_context_for_tree(tree)["order"], dtype=np.int64),
                np.asarray(prox_ids, dtype=np.int64),
                np.asarray(dist_ids, dtype=np.int64),
                np.asarray(flows, dtype=np.float64),
                np.maximum(np.asarray(resistances, dtype=np.float64), 1.0e-30),
                int(inlet_nodes[0]),
                float(root_pressure_dyn),
                int(num_nodes),
            )
            pressure_s = perf_counter() - t_pressure
            if int(assigned_edges) != int(prox_ids.size) or not np.all(
                np.isfinite(pressures)
            ):
                if _state.KIRCHHOFF_DIAGNOSTICS:
                    print(
                        "Kirchhoff fixed-flow pressure reconstruction incomplete; "
                        f"assigned_edges={int(assigned_edges)}/{int(prox_ids.size)}. Falling back to tree solve."
                    )
                pressures, flows, _, _, _ = solve_kirchhoff(
                    prox_ids,
                    dist_ids,
                    resistances,
                    inlet_nodes,
                    flow_inlet,
                    outlet_nodes,
                )
                if pressures.size and inlet_nodes:
                    target = (
                        float(tree.parameters.root_pressure) * _state.PA_TO_DYN_PER_CM2
                    )
                    delta = target - float(pressures[inlet_nodes[0]])
                    pressures = pressures + delta
            elif _state.KIRCHHOFF_DIAGNOSTICS:
                print(
                    "Kirchhoff diagnostics: solver_used=tree_fixed_pressure_reconstruct "
                    f"bc=legacy_equal_terminal_flow solve_time={pressure_s:.3f}s true_rel_resid=0.000e+00"
                )
        else:
            pressures, flows, _, _, _ = solve_kirchhoff(
                prox_ids,
                dist_ids,
                resistances,
                inlet_nodes,
                flow_inlet,
                outlet_nodes,
            )
            if pressures.size and inlet_nodes:
                target = float(tree.parameters.root_pressure) * _state.PA_TO_DYN_PER_CM2
                delta = target - float(pressures[inlet_nodes[0]])
                pressures = pressures + delta
    t_kirchhoff = perf_counter() - t_kirchhoff
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
        pressure_in = p_in
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
        pressure_out = p_out
    pressure_drop = pressure_in - pressure_out

    # Transport dispatch selects network, top-down, Graetz, or Cext coupling
    # without changing the geometry and flow solution established above.
    t_conc = perf_counter()
    cin, cout = _solve_channel_concentrations(
        tree,
        flows,
        inlet_nodes,
        outlet_nodes,
        starts_arr,
        ends_arr,
        radii_arr,
        lengths_arr,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=_state.SOLUTE_DIFFUSIVITY,
        vmax=_state.VMAX_MM,
        km=_state.K_M_MM,
        fluid=analysis_fluid,
        solver=concentration_solver,
    )
    t_conc = perf_counter() - t_conc
    concentration_timing_details = dict(_state._LAST_CONCENTRATION_TIMINGS)

    q_inlet = float(np.abs(flows[0])) if flows.size else float("nan")
    q_inlet_ul_min = float(q_inlet * 60000.0) if np.isfinite(q_inlet) else float("nan")
    total_flow = float(q_inlet) if np.isfinite(q_inlet) else float("nan")
    positive_flows = np.abs(flows[flows != 0]) if flows.size else np.empty((0,))
    q_min = float(np.min(positive_flows)) if positive_flows.size else float("nan")

    q_ratio = float("nan")
    if np.isfinite(q_min) and np.isfinite(q_inlet) and q_inlet != 0.0:
        q_ratio = q_min / q_inlet

    dR_net = float("nan")
    r_net = float("nan")
    reference_flow = q_inlet if np.isfinite(q_inlet) else total_flow
    if (
        np.isfinite(pressure_drop)
        and np.isfinite(reference_flow)
        and reference_flow != 0.0
        and total_length > 0.0
    ):
        r_net = pressure_drop / reference_flow
        dR_net = r_net / total_length

    avg_distance = (
        compute_average_distance(sample_points, starts_arr, ends_arr)
        if _state.COMPUTE_AVG_DISTANCE_TO_CHANNEL
        else float("nan")
    )
    finite_radii = radii_arr[np.isfinite(radii_arr)]
    radius_min = float(np.nanmin(finite_radii)) if finite_radii.size else float("nan")
    radius_max = float(np.nanmax(finite_radii)) if finite_radii.size else float("nan")

    vessel_map = getattr(tree, "vessel_map", {}) or {}
    terminals = (
        [
            idx
            for idx in range(cout.size)
            if not vessel_map.get(idx, {}).get("downstream")
        ]
        if vessel_map
        else []
    )
    if not terminals:
        conn = _tree_exact_connectivity(tree, np.asarray(tree.data[:seg_count]))
        terminals = list(np.where((conn[:, 0] < 0) & (conn[:, 1] < 0))[0])
    target_concs = cout[terminals] if terminals else cout
    finite = target_concs[np.isfinite(target_concs)]
    c_lq = float(np.percentile(finite, 25.0)) if finite.size else float("nan")

    # Tissue evaluation consumes the converged vessel source state. Cext-aware
    # solvers reuse their quadrature state to avoid recomputing vessel exchange.
    t_tissue = perf_counter()
    cext_source_state = (
        _state._LAST_CEXT_SOURCE_STATE
        if concentration_solver_mode
        in {
            "network_ext",
            "topdown_ext",
            "topdown_ext_hybrid_bg",
            "network_ext_hybrid_bg",
            "topdown_ext_treecode",
        }
        else None
    )
    if isinstance(cext_source_state, dict):
        mask, tissue_conc = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            starts_arr,
            ends_arr,
            radii_arr,
            cext_source_state,
            tissue_cache=tissue_cache,
        )
    else:
        if _state.SOLVER_TIMING_DETAILS:
            print("  Tissue Greens solve: source_mode=reconstructed_vessel_flux")
        mask, tissue_conc = compute_tissue_samples_greens(
            sample_points,
            starts_arr,
            ends_arr,
            radii_arr,
            cin,
            flows,
            diffusivity=_state.SOLUTE_DIFFUSIVITY,
            vmax=_state.VMAX_MM,
            km=_state.K_M_MM,
            window_factor=_state.WINDOW_FACTOR,
            inlet_concentration=inlet_concentration,
            tissue_cache=tissue_cache,
        )
    t_tissue = perf_counter() - t_tissue
    tissue_timing_details = dict(_state._LAST_TISSUE_TIMINGS)
    tissue_backend = str(
        tissue_timing_details.get("backend", _resolve_tissue_accel_mode())
    )
    t_tissue_geometry = float(
        tissue_timing_details.get("t_tissue_geometry_s", 0.0) or 0.0
    )
    t_tissue_oxygen = float(
        tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0
    )
    tissue_vals = tissue_conc[mask]
    tissue_pts = sample_points[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")
    cext_concentration_mean = float("nan")
    cext_concentration_std = float("nan")
    cext_concentration_count = 0
    if isinstance(cext_source_state, dict) and "c_ext_gl" in cext_source_state:
        cext_vals = np.asarray(
            cext_source_state.get("c_ext_gl"), dtype=np.float32
        ).reshape(-1)
        cext_vals = cext_vals[np.isfinite(cext_vals)]
        cext_concentration_count = int(cext_vals.size)
        if cext_vals.size:
            cext_concentration_mean = float(np.nanmean(cext_vals))
            cext_concentration_std = (
                float(np.nanstd(cext_vals, ddof=1)) if cext_vals.size > 1 else 0.0
            )

    if (
        np.isfinite(_state.CONC_MAX_FOR_NORMALIZATION)
        and _state.CONC_MAX_FOR_NORMALIZATION != 0.0
    ):
        ratio_lq = (
            c_lq / _state.CONC_MAX_FOR_NORMALIZATION
            if np.isfinite(c_lq)
            else float("nan")
        )
        ratio_tiss = (
            tissue_avg / _state.CONC_MAX_FOR_NORMALIZATION
            if np.isfinite(tissue_avg)
            else float("nan")
        )
        fractions = {}
        if tissue_vals.size:
            normalized = tissue_vals / _state.CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                fractions[key] = float(np.mean(normalized >= threshold))
        else:
            fractions = {
                k: float("nan")
                for k in [
                    "FracAbove50pct",
                    "FracAbove25pct",
                    "FracAbove10pct",
                    "FracAbove5pct",
                    "FracAbove1pct",
                ]
            }
    else:
        ratio_lq = float("nan")
        ratio_tiss = float("nan")
        fractions = {
            k: float("nan")
            for k in [
                "FracAbove50pct",
                "FracAbove25pct",
                "FracAbove10pct",
                "FracAbove5pct",
                "FracAbove1pct",
            ]
        }

    area = np.pi * radii_arr * radii_arr
    velocities = np.divide(
        np.abs(flows),
        area,
        out=np.full_like(flows, np.nan),
        where=area > 0,
    )
    bbox_length = _characteristic_length(tree, fallback_side_length=side_length)
    radius_sq = radii_arr * radii_arr
    permeation_rates = np.divide(
        3.66 * _state.SOLUTE_DIFFUSIVITY,
        radius_sq,
        out=np.zeros_like(radius_sq),
        where=radius_sq > 0,
    )
    valid = (lengths_arr > 1e-5) & np.isfinite(velocities) & (velocities > 0)
    damkohler = np.full_like(lengths_arr, np.nan)
    damkohler[valid] = (permeation_rates[valid] * bbox_length) / velocities[valid]
    avg_damkohler = (
        float(np.nanmean(damkohler))
        if np.isnan(damkohler).sum() < damkohler.size
        else float("nan")
    )

    parms = getattr(tree, "parameters", None)
    dlp_enabled = (
        bool(getattr(parms, "dlp_enable", False)) if parms is not None else False
    )
    dlp_angle = float(getattr(parms, "dlp_min_angle_deg", 0.0)) if dlp_enabled else 0.0

    solved_pressure_scale = _state.DYN_PER_CM2_TO_PA if pressures.size else 1.0
    # Keep reporting and detail arrays tied to this exact solve so exporters do
    # not reconstruct scientific quantities independently.
    result = {
        "target_terminals": int(target_terminals),
        "cube_side_length": _characteristic_length(
            tree, fallback_side_length=side_length
        ),
        "distance_sample_count": int(_state.DISTANCE_SAMPLE_COUNT),
        "concentration_solver": concentration_solver_mode,
        "tissue_quadrature_order": int(_state.GL_ORDER),
        "external_field_quadrature_order": int(_state.GL_ORDER_CEXT),
        "finite_radius_o2_terms": str(_state.FINITE_RADIUS_O2_TERMS),
        "lumen_wall_closure": str(_state.LUMEN_WALL_CLOSURE),
        "graetz_n_radial": int(_state.GRAETZ_N_RADIAL),
        "graetz_n_modes": int(_state.GRAETZ_N_MODES),
        "total_volume": total_volume,
        "total_flowrate": reference_flow,
        # The Kirchhoff system is solved in dyn/cm^2. Public summary fields are Pa.
        "pressure_in_root": pressure_in * solved_pressure_scale,
        "pressure_out_terminals": pressure_out * solved_pressure_scale,
        "pressure_drop": pressure_drop * solved_pressure_scale,
        "avg_radius": avg_radius,
        "radius_min": radius_min,
        "radius_max": radius_max,
        "avg_length": avg_length,
        "total_length": total_length,
        "avg_distance_to_channel": avg_distance,
        "terminal_segments": n_terminals,
        "total_segments": seg_count,
        "dlp_angle": dlp_angle,
        "Rnet": r_net,
        "dRnet": dR_net,
        "inlet_flow_ul_per_min": q_inlet_ul_min,
        "Qmin_over_Qinlet": q_ratio,
        "C_LQ_over_Cmax": ratio_lq,
        "C_tiss_over_Cmax": ratio_tiss,
        "Damkohler": avg_damkohler,
        # Public timing categories:
        # - t_assembly_s includes segment assembly plus tissue geometry/query/refine/setup.
        # - t_tissue_s is only the final tissue Greens oxygen kernel.
        "t_load_s": 0.0,
        "t_assembly_s": t_assemble + t_tissue_geometry,
        "t_kirchhoff_s": t_kirchhoff,
        "t_hematocrit_flow_iteration_s": t_hematocrit_flow_iter,
        "t_concentration_s": t_conc,
        "t_cext_total_s": float(
            concentration_timing_details.get("t_cext_total_s", 0.0) or 0.0
        ),
        "t_cext_hct_setup_s": float(
            concentration_timing_details.get("t_cext_hct_setup_s", 0.0) or 0.0
        ),
        "t_cext_context_build_s": float(
            concentration_timing_details.get("t_cext_context_build_s", 0.0) or 0.0
        ),
        "t_cext_solver_setup_s": float(
            concentration_timing_details.get("t_cext_solver_setup_s", 0.0) or 0.0
        ),
        "t_cext_query_s": float(
            concentration_timing_details.get("t_cext_query_s", 0.0) or 0.0
        ),
        "t_cext_kernel_s": float(
            concentration_timing_details.get("t_cext_kernel_s", 0.0) or 0.0
        ),
        "t_cext_gpu_transfer_s": float(
            concentration_timing_details.get("t_cext_gpu_transfer_s", 0.0) or 0.0
        ),
        "t_cext_hybrid_deposit_s": float(
            concentration_timing_details.get("t_cext_hybrid_deposit_s", 0.0) or 0.0
        ),
        "t_cext_hybrid_fft_s": float(
            concentration_timing_details.get("t_cext_hybrid_fft_s", 0.0) or 0.0
        ),
        "t_cext_hybrid_o2_terms_s": float(
            concentration_timing_details.get("t_cext_hybrid_o2_terms_s", 0.0) or 0.0
        ),
        "t_cext_hybrid_self_subtract_s": float(
            concentration_timing_details.get("t_cext_hybrid_self_subtract_s", 0.0)
            or 0.0
        ),
        "t_cext_hybrid_local_corr_s": float(
            concentration_timing_details.get("t_cext_hybrid_local_corr_s", 0.0) or 0.0
        ),
        "t_cext_hybrid_sample_s": float(
            concentration_timing_details.get("t_cext_hybrid_sample_s", 0.0) or 0.0
        ),
        "t_cext_frozen_gpu_transfer_s": float(
            concentration_timing_details.get("t_cext_frozen_gpu_transfer_s", 0.0) or 0.0
        ),
        "t_cext_init_s": float(
            concentration_timing_details.get("t_cext_init_s", 0.0) or 0.0
        ),
        "t_cext_init_kernel_s": float(
            concentration_timing_details.get("t_cext_init_kernel_s", 0.0) or 0.0
        ),
        "t_cext_init_gpu_transfer_s": float(
            concentration_timing_details.get("t_cext_init_gpu_transfer_s", 0.0) or 0.0
        ),
        "t_cext_init_frozen_gpu_transfer_s": float(
            concentration_timing_details.get("t_cext_init_frozen_gpu_transfer_s", 0.0)
            or 0.0
        ),
        "t_cext_init_frozen_kernel_s": float(
            concentration_timing_details.get("t_cext_init_frozen_kernel_s", 0.0) or 0.0
        ),
        "t_cext_init_cache_s": float(
            concentration_timing_details.get("t_cext_init_cache_s", 0.0) or 0.0
        ),
        "t_cext_init_hybrid_call_s": float(
            concentration_timing_details.get("t_cext_init_hybrid_call_s", 0.0) or 0.0
        ),
        "t_cext_final_frozen_gpu_transfer_s": float(
            concentration_timing_details.get("t_cext_final_frozen_gpu_transfer_s", 0.0)
            or 0.0
        ),
        "t_cext_final_frozen_kernel_s": float(
            concentration_timing_details.get("t_cext_final_frozen_kernel_s", 0.0) or 0.0
        ),
        "t_cext_final_cache_s": float(
            concentration_timing_details.get("t_cext_final_cache_s", 0.0) or 0.0
        ),
        "t_cext_final_source_state_s": float(
            concentration_timing_details.get("t_cext_final_source_state_s", 0.0) or 0.0
        ),
        "t_cext_backend": str(concentration_timing_details.get("backend", "none")),
        "cext_init_mode": str(
            concentration_timing_details.get("cext_init_mode", "zero")
        ),
        "cext_init_performed": bool(
            concentration_timing_details.get("cext_init_performed", False)
        ),
        "t_tissue_s": t_tissue_oxygen,
        "t_tissue_geometry_s": t_tissue_geometry,
        "t_tissue_oxygen_s": t_tissue_oxygen,
        "t_tissue_total_s": t_tissue,
        "t_tissue_kdtree_query_s": float(
            tissue_timing_details.get("t_tissue_kdtree_query_s", float("nan"))
        ),
        "t_tissue_gpu_refine_s": float(
            tissue_timing_details.get("t_tissue_gpu_refine_s", float("nan"))
        ),
        "t_tissue_gpu_transfer_s": float(
            tissue_timing_details.get("t_tissue_gpu_transfer_s", float("nan"))
        ),
        "t_tissue_backend": tissue_backend,
        "hematocrit_model": hematocrit_model,
        "hematocrit_flow_iterations": int(_state.HEMATOCRIT_FLOW_ITERATIONS),
        "hematocrit_flow_iterations_completed": int(hematocrit_iterations_completed),
        "cext_outer_iterations_completed": int(
            concentration_timing_details.get("cext_outer_iterations_completed", 0) or 0
        ),
        "cext_total_iterations_effective": int(
            concentration_timing_details.get("cext_total_iterations_effective", 0) or 0
        ),
        "cext_accel_mode": str(
            concentration_timing_details.get("cext_accel_mode", "none")
        ),
        "cext_accel_step_last": str(
            concentration_timing_details.get("cext_accel_step_last", "picard")
        ),
        "cext_accel_rejections": int(
            concentration_timing_details.get("cext_accel_rejections", 0) or 0
        ),
        "cext_accel_restarts": int(
            concentration_timing_details.get("cext_accel_restarts", 0) or 0
        ),
        "cext_anderson_depth_used": int(
            concentration_timing_details.get("cext_anderson_depth_used", 0) or 0
        ),
        "cext_omega_last": float(
            concentration_timing_details.get(
                "cext_omega_last", _state.CEXT_VESS_COUPLING_OMEGA
            )
            or _state.CEXT_VESS_COUPLING_OMEGA
        ),
        "cext_rel_residual_last": float(
            concentration_timing_details.get("cext_rel_residual_last", 0.0) or 0.0
        ),
        "cext_max_delta_last": float(
            concentration_timing_details.get("cext_max_delta_last", 0.0) or 0.0
        ),
        "cext_active_source_count": int(
            concentration_timing_details.get("cext_active_source_count", 0) or 0
        ),
        "cext_frozen_source_count": int(
            concentration_timing_details.get("cext_frozen_source_count", 0) or 0
        ),
        "cext_active_target_count": int(
            concentration_timing_details.get("cext_active_target_count", 0) or 0
        ),
        "cext_frozen_target_count": int(
            concentration_timing_details.get("cext_frozen_target_count", 0) or 0
        ),
        "cext_active_component_count": int(
            concentration_timing_details.get("cext_active_component_count", 0) or 0
        ),
        "cext_largest_component_size": int(
            concentration_timing_details.get("cext_largest_component_size", 0) or 0
        ),
        "cext_target_freeze_events": int(
            concentration_timing_details.get("cext_target_freeze_events", 0) or 0
        ),
        "cext_target_reactivations": int(
            concentration_timing_details.get("cext_target_reactivations", 0) or 0
        ),
        "cext_hybrid_bg_grid": int(
            concentration_timing_details.get("cext_hybrid_bg_grid", 0) or 0
        ),
        "cext_hybrid_lambda_bins": int(
            concentration_timing_details.get("cext_hybrid_lambda_bins", 0) or 0
        ),
        "cext_hybrid_lambda_bin_policy": str(
            concentration_timing_details.get("cext_hybrid_lambda_bin_policy", "")
        ),
        "cext_hybrid_lambda_bin_edges_hash": str(
            concentration_timing_details.get("cext_hybrid_lambda_bin_edges_hash", "")
        ),
        "cext_hybrid_bg_solver": str(
            concentration_timing_details.get("cext_hybrid_bg_solver", "")
        ),
        "cext_concentration_mean": cext_concentration_mean,
        "cext_concentration_std": cext_concentration_std,
        "cext_concentration_count": int(cext_concentration_count),
        "concentration_inlet": inlet_concentration,
        **fractions,
    }
    vessel_quadrature = None
    if isinstance(cext_source_state, dict):
        vessel_quadrature = {
            "solver": str(cext_source_state.get("solver", concentration_solver_mode)),
            "gl_points_si": np.asarray(
                cext_source_state["gl_points_si"], dtype=np.float32
            ),
        }
        for key in ("c_iv_gl", "c_bulk_gl", "c_wall_gl", "c_ext_gl"):
            if key in cext_source_state:
                vessel_quadrature[key] = np.asarray(
                    cext_source_state[key], dtype=np.float32
                )
    _clear_cext_runtime_state(tree)
    if not return_details:
        return result
    details = {
        "starts": starts_arr,
        "ends": ends_arr,
        "radii": radii_arr,
        "lengths": lengths_arr,
        "mu": mu_arr,
        "pressures": pressures,
        "flows": flows,
        "cin": cin,
        "cout": cout,
        "concentration_timing_details": concentration_timing_details,
        "flow_iteration_hematocrit": hd_flow_iter,
        "hematocrit_iteration_log": hematocrit_iter_log,
        "tissue_points": tissue_pts,
        "tissue_values": tissue_vals,
        "inlet_concentration": inlet_concentration,
        "cext_source_state": (
            dict(cext_source_state) if isinstance(cext_source_state, dict) else None
        ),
        "vessel_quadrature": vessel_quadrature,
    }
    HD_detail = np.asarray(
        getattr(tree, "discharge_hematocrit", np.empty((0,), dtype=float)), dtype=float
    )
    HT_detail = np.asarray(
        getattr(tree, "tube_hematocrit", np.empty((0,), dtype=float)), dtype=float
    )
    if HD_detail.shape[0] != seg_count or HT_detail.shape[0] != seg_count:
        if str(analysis_fluid).lower() == "blood":
            HD_detail, HT_detail = compute_tree_hematocrit(
                tree,
                hd_root=_state.HD_DISCHARGE,
                flows=flows,
                model=hematocrit_model,
            )
        else:
            HD_detail = np.empty((0,), dtype=float)
            HT_detail = np.empty((0,), dtype=float)
    details["discharge_hematocrit"] = HD_detail
    details["tube_hematocrit"] = HT_detail
    return result, details


def summarize_tissue_only_from_details(
    base_metrics: Dict[str, float],
    details: dict,
    sample_points: np.ndarray,
    *,
    tissue_cache: dict | None,
    tissue_cache_build_elapsed: float,
    base_core_assembly_s: float,
) -> Dict[str, float]:
    """Reuse a solved tree/flow/concentration state and recompute only tissue metrics."""
    metrics = dict(base_metrics)
    cext_source_state = details.get("cext_source_state")
    t_tissue = perf_counter()
    if isinstance(cext_source_state, dict):
        mask, tissue_conc = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
            np.asarray(details["radii"], dtype=float),
            cext_source_state,
            tissue_cache=tissue_cache,
        )
    else:
        mask, tissue_conc = compute_tissue_samples_greens(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
            np.asarray(details["radii"], dtype=float),
            np.asarray(details["cin"], dtype=float),
            np.asarray(details["flows"], dtype=float),
            diffusivity=_state.SOLUTE_DIFFUSIVITY,
            vmax=_state.VMAX_MM,
            km=_state.K_M_MM,
            window_factor=_state.WINDOW_FACTOR,
            inlet_concentration=float(
                details.get("inlet_concentration", get_concentration_inlet())
            ),
            tissue_cache=tissue_cache,
        )
    t_tissue = perf_counter() - t_tissue
    tissue_timing_details = dict(_state._LAST_TISSUE_TIMINGS)
    tissue_vals = tissue_conc[mask]
    tissue_avg = float(np.nanmean(tissue_vals)) if tissue_vals.size else float("nan")
    if (
        np.isfinite(_state.CONC_MAX_FOR_NORMALIZATION)
        and _state.CONC_MAX_FOR_NORMALIZATION != 0.0
    ):
        metrics["C_tiss_over_Cmax"] = (
            tissue_avg / _state.CONC_MAX_FOR_NORMALIZATION
            if np.isfinite(tissue_avg)
            else float("nan")
        )
        if tissue_vals.size:
            normalized = tissue_vals / _state.CONC_MAX_FOR_NORMALIZATION
            for threshold, key in [
                (0.50, "FracAbove50pct"),
                (0.25, "FracAbove25pct"),
                (0.10, "FracAbove10pct"),
                (0.05, "FracAbove5pct"),
                (0.01, "FracAbove1pct"),
            ]:
                metrics[key] = float(np.mean(normalized >= threshold))
        else:
            for key in [
                "FracAbove50pct",
                "FracAbove25pct",
                "FracAbove10pct",
                "FracAbove5pct",
                "FracAbove1pct",
            ]:
                metrics[key] = float("nan")
    else:
        metrics["C_tiss_over_Cmax"] = float("nan")
        for key in [
            "FracAbove50pct",
            "FracAbove25pct",
            "FracAbove10pct",
            "FracAbove5pct",
            "FracAbove1pct",
        ]:
            metrics[key] = float("nan")

    metrics["distance_sample_count"] = int(sample_points.shape[0])
    t_tissue_geometry = float(
        tissue_timing_details.get("t_tissue_geometry_s", 0.0) or 0.0
    )
    metrics["t_assembly_s"] = (
        float(base_core_assembly_s)
        + float(tissue_cache_build_elapsed)
        + t_tissue_geometry
    )
    metrics["t_tissue_s"] = float(
        tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0
    )
    metrics["t_tissue_geometry_s"] = t_tissue_geometry
    metrics["t_tissue_oxygen_s"] = float(
        tissue_timing_details.get("t_tissue_oxygen_s", t_tissue) or 0.0
    )
    metrics["t_tissue_total_s"] = float(
        tissue_timing_details.get("t_tissue_total_s", t_tissue) or t_tissue
    )
    metrics["t_tissue_kdtree_query_s"] = float(
        tissue_timing_details.get("t_tissue_kdtree_query_s", float("nan"))
    )
    metrics["t_tissue_gpu_refine_s"] = float(
        tissue_timing_details.get("t_tissue_gpu_refine_s", float("nan"))
    )
    metrics["t_tissue_gpu_transfer_s"] = float(
        tissue_timing_details.get("t_tissue_gpu_transfer_s", float("nan"))
    )
    metrics["t_tissue_backend"] = str(
        tissue_timing_details.get("backend", _resolve_tissue_accel_mode())
    )
    if _state.COMPUTE_AVG_DISTANCE_TO_CHANNEL:
        metrics["avg_distance_to_channel"] = compute_average_distance(
            sample_points,
            np.asarray(details["starts"], dtype=float),
            np.asarray(details["ends"], dtype=float),
        )
    return metrics


# Public shorthand retained for scripts that prefer the summary-oriented name.
summarize_tree = run_tree_simulation

__all__ = [
    "run_tree_simulation",
    "summarize_tree",
    "summarize_tissue_only_from_details",
]
