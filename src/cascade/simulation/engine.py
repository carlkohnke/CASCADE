"""Coordinate domain samples, vascular solvers, tissue analysis, and unified run results."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any

import numpy as np

from cascade.configuration.schema import RunConfig
from cascade.configuration import solver_state as _state
from cascade.concentration.tissue.cache import build_tissue_cache_from_tree
from cascade.concentration.properties import get_concentration_inlet
from cascade.concentration.tissue.geometry import _prepare_tissue_geometry
from cascade.concentration.tissue.greens import compute_tissue_samples_greens
from cascade.concentration.tissue.greens import (
    compute_tissue_samples_greens_from_cext_state,
)
from cascade.concentration.external_field.multinetwork import (
    compact_external_field_state,
    solve_multinetwork_external_field,
)
from cascade.domain.sampling import (
    compute_distance_to_nearest_channel,
)
from cascade.domain.workflow import prepare_sample_points
from cascade.configuration.bridge import load_runtime_module
from cascade.vessels.conditions import (
    flow_for_tree,
    inlet_concentration_for_tree,
    sync_tree_parameters_for_run,
    terminal_flow_for_target,
)
from cascade.simulation.simple import simple_details, solve_simple_network
from cascade.flow.boundary_conditions import allocate_inlet_flows
from cascade.vessels.collections import VascularNetworkSet
from cascade.vessels.interventions import (
    InterventionState,
    Occlusion,
    apply_occlusion,
    zero_occluded_solution,
)
from cascade.simulation.network import run_tree_simulation
from cascade.simulation.results import combine_network_solutions

DYN_PER_CM2_TO_PA = 0.1


@dataclass
class TreeSimulation:
    tree_id: int
    tree: Any
    target_count: int
    summary: dict[str, Any]
    details: dict[str, Any]
    elapsed_s: float


@dataclass
class SimulationResult:
    tree_results: list[TreeSimulation]
    summary_rows: list[dict[str, Any]]
    segment_rows: list[dict[str, Any]]
    point_rows: list[dict[str, Any]]
    sample_points: np.ndarray
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    sample_meta: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    interventions: list[dict[str, Any]] = field(default_factory=list)
    external_field: dict[str, Any] | None = None


def run_simulation(
    domain,
    trees: list[Any],
    target_counts: list[int],
    config: RunConfig,
    *,
    sample_points: np.ndarray | None = None,
    tissue_caches: list[dict | None] | None = None,
    reuse_geometry: bool = False,
    materialize_segment_rows: bool = True,
) -> SimulationResult:
    """Execute every network in a run through one consistent simulation path.

    Simple graphs are solved first because they do not carry SVV tree methods.
    Tree and forest cases then use either independent external fields or one
    shared multi-network field. Results are normalized into common records for
    every supported geometry type.
    """

    ts = load_runtime_module()
    timings: dict[str, float] = {}

    t0 = perf_counter()
    sample_points = _sample_points(ts, domain, config, provided=sample_points)
    timings["sample_points_resolve_s"] = perf_counter() - t0

    for network in trees:
        if getattr(network, "_cascade_simple_network", False):
            solve_simple_network(network, ts, config)

    tree_results: list[TreeSimulation] = []
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    intervention_records: list[dict[str, Any]] = []
    global_segment_id = 0
    want_point_data = (
        (
            bool(config.outputs.write_points_csv)
            or (
                bool(config.outputs.write_paraview)
                and bool(config.outputs.write_tissue_vtp)
            )
        )
        and not bool(config.simulation.geometry_only)
        and not bool(config.simulation.skip_tissue_oxygen)
    )
    # CLI/GUI callers opt into array-native exports.  Keep materialized rows as
    # the API default for third-party callers which inspect SimulationResult.
    want_segment_rows = (
        (
            bool(materialize_segment_rows)
            and (
                bool(config.outputs.write_segments_csv)
                or (
                    bool(config.outputs.write_paraview)
                    and bool(config.outputs.write_vessels_vtp)
                )
            )
        )
        or (want_point_data and bool(config.outputs.include_tissue_nearest_fields))
    )

    external_field = config.simulation.external_field
    if (
        external_field.enabled
        and external_field.scope == "shared"
        and not config.simulation.geometry_only
    ):
        return _run_shared_external_field_case(
            ts,
            domain,
            trees,
            target_counts,
            config,
            sample_points,
            timings,
            want_segment_rows=want_segment_rows,
            want_point_data=want_point_data,
        )

    networks = VascularNetworkSet.from_networks(trees)
    for tree_id, (tree, target_count) in enumerate(zip(trees, target_counts)):
        t_tree = perf_counter()
        if getattr(tree, "_cascade_simple_network", False):
            summary, details = simple_details(tree, sample_points, ts, config)
            elapsed = perf_counter() - t_tree
            row = _summary_row(
                ts,
                summary,
                config,
                tree_id=tree_id,
                n_trees=len(trees),
                elapsed_s=elapsed,
            )
            summary_rows.append(row)
            if want_segment_rows:
                rows, global_segment_id = _segment_rows(
                    details, row, tree_id=tree_id, start_global_id=global_segment_id
                )
                segment_rows.extend(rows)
            else:
                global_segment_id += _detail_segment_count(details, tree)
            tree_results.append(
                TreeSimulation(
                    tree_id=tree_id,
                    tree=tree,
                    target_count=int(target_count),
                    summary=summary,
                    details=details,
                    elapsed_s=elapsed,
                )
            )
            continue
        qin_cm3_s = _inlet_flow_for_tree(config, tree, tree_id, trees)
        terminal_flow = terminal_flow_for_target(config, qin_cm3_s, int(target_count))
        if hasattr(ts, "_sync_loaded_tree_params"):
            sync_tree_parameters_for_run(ts, tree, config, tree_id, terminal_flow)
        with _occlusion_override(config, networks, tree_id) as intervention:
            if config.simulation.geometry_only:
                summary, details = _geometry_only_result(
                    ts, tree, target_count, config, qin_cm3_s
                )
            else:
                tissue_cache = (
                    tissue_caches[tree_id]
                    if tissue_caches is not None and tree_id < len(tissue_caches)
                    else None
                )
                if (
                    tissue_cache is None
                    and not config.simulation.skip_tissue_oxygen
                    and sample_points.size
                ):
                    t_cache = perf_counter()
                    tissue_cache = build_tissue_cache_from_tree(tree, sample_points)
                    timings[f"tree_{tree_id}_tissue_cache_s"] = perf_counter() - t_cache
                summary, details = run_tree_simulation(
                    tree,
                    sample_points,
                    int(target_count),
                    side_length=float(config.domain.side_length),
                    fluid=config.simulation.fluid,
                    inlet_flow_cm3_s=qin_cm3_s,
                    inlet_concentration_override=inlet_concentration_for_tree(
                        config, tree_id
                    ),
                    tissue_cache=tissue_cache,
                    concentration_solver=config.simulation.concentration_solver,
                    reuse_geometry=(
                        bool(reuse_geometry) and config.simulation.occlusion is None
                    ),
                    return_details=True,
                )
            zero_occluded_solution(details, intervention)
        intervention.restore_solution_geometry(details)
        if intervention.applied:
            intervention_records.append(intervention.metadata(networks))

        elapsed = perf_counter() - t_tree
        row = _summary_row(
            ts, summary, config, tree_id=tree_id, n_trees=len(trees), elapsed_s=elapsed
        )
        summary_rows.append(row)
        if want_segment_rows:
            rows, global_segment_id = _segment_rows(
                details, row, tree_id=tree_id, start_global_id=global_segment_id
            )
            segment_rows.extend(rows)
        else:
            global_segment_id += _detail_segment_count(details, tree)
        tree_results.append(
            TreeSimulation(
                tree_id=tree_id,
                tree=tree,
                target_count=int(target_count),
                summary=summary,
                details=details,
                elapsed_s=elapsed,
            )
        )

    t0 = perf_counter()
    point_data = (
        _point_data(ts, tree_results, segment_rows, sample_points, config)
        if want_point_data
        else {}
    )
    point_rows = (
        _point_rows_from_data(point_data) if config.outputs.write_points_csv else []
    )
    timings["points_s"] = perf_counter() - t0
    return SimulationResult(
        tree_results=tree_results,
        summary_rows=summary_rows,
        segment_rows=segment_rows,
        point_rows=point_rows,
        point_data=point_data,
        sample_points=sample_points,
        sample_meta=_sample_meta_for_config(config, sample_points),
        timings=timings,
        interventions=intervention_records,
    )


def _run_shared_external_field_case(
    runtime,
    domain,
    trees: list[Any],
    target_counts: list[int],
    config: RunConfig,
    sample_points: np.ndarray,
    timings: dict[str, float],
    *,
    want_segment_rows: bool,
    want_point_data: bool,
) -> SimulationResult:
    """Run the shared multi-network Cext path through the canonical engine."""
    networks = VascularNetworkSet.from_networks(trees)
    inlet_flows = [
        _inlet_flow_for_tree(config, tree, network_id, trees)
        for network_id, tree in enumerate(trees)
    ]
    configured = config.simulation.occlusion
    occlusion = (
        Occlusion(
            global_segment_id=configured.global_segment_id,
            fraction_blocked=configured.fraction_blocked,
            include_downstream_when_complete=(
                configured.include_downstream_when_complete
            ),
        )
        if configured is not None
        else None
    )
    t0 = perf_counter()
    with apply_occlusion(networks, occlusion) as intervention:
        solutions, external_metadata = solve_multinetwork_external_field(
            runtime,
            runtime,
            networks,
            inlet_flows,
            fluid=config.simulation.fluid,
            mode=config.simulation.external_field.mode,
        )
    timings["shared_external_field_s"] = perf_counter() - t0
    if intervention.applied and intervention.network_id is not None:
        affected = solutions[intervention.network_id]
        intervention.restore_solution_geometry(affected)
        zero_occluded_solution(affected, intervention)

    tree_results: list[TreeSimulation] = []
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    global_segment_id = 0
    for network_id, (tree, target_count, solution) in enumerate(
        zip(trees, target_counts, solutions)
    ):
        compact_state = compact_external_field_state(solution)
        details = dict(solution)
        details["cext_source_state"] = compact_state
        details["vessel_quadrature"] = {
            key: compact_state[key]
            for key in (
                "solver",
                "gl_points_si",
                "c_iv_gl",
                "c_bulk_gl",
                "c_wall_gl",
                "c_ext_gl",
            )
        }
        summary = _summary_from_solution(
            runtime,
            tree,
            solution,
            target_count=int(target_count),
            config=config,
            external_metadata=external_metadata,
        )
        row = _summary_row(
            runtime,
            summary,
            config,
            tree_id=network_id,
            n_trees=len(trees),
            elapsed_s=float(timings["shared_external_field_s"]),
        )
        summary_rows.append(row)
        if want_segment_rows:
            rows, global_segment_id = _segment_rows(
                details,
                row,
                tree_id=network_id,
                start_global_id=global_segment_id,
            )
            segment_rows.extend(rows)
        else:
            global_segment_id += int(np.asarray(solution["starts"]).shape[0])
        tree_results.append(
            TreeSimulation(
                tree_id=network_id,
                tree=tree,
                target_count=int(target_count),
                summary=summary,
                details=details,
                elapsed_s=float(timings["shared_external_field_s"]),
            )
        )

    point_data: dict[str, np.ndarray] = {}
    if want_point_data and sample_points.size:
        t0 = perf_counter()
        combined = combine_network_solutions(solutions)
        retained, concentrations = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            combined["starts"],
            combined["ends"],
            combined["radii"],
            combined["cext_state"],
            tissue_cache=None,
        )
        mask = np.asarray(retained, dtype=bool)
        point_data = _point_data_from_samples(
            runtime,
            sample_points[mask],
            np.asarray(concentrations)[mask],
            segment_rows,
            config,
        )
        timings["points_s"] = perf_counter() - t0
    point_rows = (
        _point_rows_from_data(point_data) if config.outputs.write_points_csv else []
    )
    intervention_records = (
        [intervention.metadata(networks)] if intervention.applied else []
    )
    return SimulationResult(
        tree_results=tree_results,
        summary_rows=summary_rows,
        segment_rows=segment_rows,
        point_rows=point_rows,
        point_data=point_data,
        sample_points=sample_points,
        sample_meta=_sample_meta_for_config(config, sample_points),
        timings=timings,
        interventions=intervention_records,
        external_field=dict(external_metadata),
    )


def _summary_from_solution(
    runtime,
    tree,
    solution: dict[str, Any],
    *,
    target_count: int,
    config: RunConfig,
    external_metadata: dict[str, Any],
) -> dict[str, Any]:
    radii = np.asarray(solution["radii"], dtype=float)
    lengths = np.asarray(solution["lengths"], dtype=float)
    flows = np.asarray(solution["flows"], dtype=float)
    cin = np.asarray(solution["cin"], dtype=float)
    cout = np.asarray(solution["cout"], dtype=float)
    return {
        "target_terminals": int(target_count),
        "cube_side_length": float(config.domain.side_length),
        "concentration_solver": config.simulation.concentration_solver,
        "total_volume": float(np.nansum(np.pi * radii**2 * lengths)),
        "total_flowrate": float(flows[0]) if flows.size else math.nan,
        "pressure_in_root": float(solution.get("p_in", np.nan)) * DYN_PER_CM2_TO_PA,
        "pressure_out_terminals": float(solution.get("p_out", np.nan))
        * DYN_PER_CM2_TO_PA,
        "avg_radius": float(np.nanmean(radii)) if radii.size else math.nan,
        "avg_length": float(np.nanmean(lengths)) if lengths.size else math.nan,
        "total_length": float(np.nansum(lengths)),
        "terminal_segments": max(int(getattr(tree, "n_terminals", 0)) - 1, 0),
        "total_segments": int(radii.size),
        "inlet_flow_ul_per_min": float(flows[0] * 60000.0) if flows.size else math.nan,
        "C_LQ_over_Cmax": (
            float(np.nanmean(cout) / runtime.CONC_MAX_FOR_NORMALIZATION)
            if cout.size and float(runtime.CONC_MAX_FOR_NORMALIZATION) > 0.0
            else math.nan
        ),
        "t_cext_total_s": float(external_metadata.get("total_s", 0.0)),
        "t_cext_backend": str(external_metadata.get("mode", "shared")),
        "concentration_inlet": float(solution.get("inlet_concentration", np.nan)),
        "concentration_outlet_mean": float(np.nanmean(cout)) if cout.size else math.nan,
        "concentration_inlet_mean": float(np.nanmean(cin)) if cin.size else math.nan,
    }


def _detail_segment_count(details: dict[str, Any], tree) -> int:
    starts = details.get("starts")
    if starts is not None:
        try:
            return int(np.asarray(starts).shape[0])
        except Exception:
            pass
    return int(getattr(tree, "segment_count", 0) or 0)


def _sample_points(
    ts, domain, config: RunConfig, *, provided: np.ndarray | None = None
) -> np.ndarray:
    if provided is not None:
        return np.asarray(provided, dtype=float)
    points, _ = prepare_sample_points(domain, config, ts=ts)
    return points


def _sample_meta_for_config(config: RunConfig, points: np.ndarray) -> dict[str, Any]:
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return {"sample_mode": "none", "points": 0}
    if config.simulation.sample_mode == "grid":
        meta = {"sample_mode": "grid", "points": int(np.asarray(points).shape[0])}
        meta.update(dict(config.simulation.tissue_grid or {}))
        return meta
    if config.simulation.sample_mode == "file":
        return {
            "sample_mode": "file",
            "points": int(np.asarray(points).shape[0]),
            "path": config.simulation.sample_points_path,
            "coordinate_units": "cm",
        }
    return {
        "sample_mode": "random",
        "requested_points": int(config.simulation.distance_sample_count),
        "points": int(np.asarray(points).shape[0]),
    }


def _inlet_flow_for_tree(
    config: RunConfig, tree, tree_id: int, trees: list[Any]
) -> float:
    source = str(config.simulation.flow_source).strip().lower().replace("_", "-")
    prescribed = [
        flow_for_tree(config, index, len(trees)) for index in range(len(trees))
    ]
    total_cm3_s = None
    if source == "total-qin-split":
        total_ul_min = (
            float(config.simulation.total_qin_ul_min)
            if config.simulation.total_qin_ul_min is not None
            else float(config.simulation.qin_target_ul_min)
        )
        total_cm3_s = (
            total_ul_min * 1.0e-3 / 60.0 * (float(config.domain.side_length) ** 3)
        )
    return allocate_inlet_flows(
        trees,
        source=source,
        prescribed_per_network=prescribed,
        total_qin_cm3_s=total_cm3_s,
        require_stored_root_flow=False,
    )[int(tree_id)]


@contextmanager
def _occlusion_override(
    config: RunConfig,
    networks: VascularNetworkSet,
    tree_id: int,
):
    configured = config.simulation.occlusion
    occlusion = (
        Occlusion(
            global_segment_id=configured.global_segment_id,
            fraction_blocked=configured.fraction_blocked,
            include_downstream_when_complete=(
                configured.include_downstream_when_complete
            ),
        )
        if configured is not None
        else None
    )
    if occlusion is None or config.simulation.geometry_only:
        yield InterventionState()
        return
    address = networks.address(occlusion.global_segment_id)
    if address.network_id != int(tree_id):
        yield InterventionState()
        return
    with apply_occlusion(networks, occlusion, geometry_only=False) as state:
        yield state


def _geometry_only_result(
    ts, tree, target_count: int, config: RunConfig, qin_cm3_s: float
) -> tuple[dict, dict]:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    data = (
        np.asarray(tree.data[:seg_count])
        if seg_count > 0
        else np.empty((0, 31), dtype=float)
    )
    starts = data[:, 0:3] if data.size else np.empty((0, 3), dtype=float)
    ends = data[:, 3:6] if data.size else np.empty((0, 3), dtype=float)
    radii = (
        data[:, 21] if data.size and data.shape[1] > 21 else np.empty((0,), dtype=float)
    )
    lengths = (
        data[:, 20] if data.size and data.shape[1] > 20 else np.empty((0,), dtype=float)
    )
    n = starts.shape[0]
    nan = np.full(n, np.nan, dtype=float)
    summary = {
        "target_terminals": int(target_count),
        "cube_side_length": float(config.domain.side_length),
        "distance_sample_count": 0,
        "concentration_solver": "geometry_only",
        "finite_radius_o2_terms": str(getattr(ts, "FINITE_RADIUS_O2_TERMS", "none")),
        "lumen_wall_closure": str(getattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")),
        "graetz_n_radial": int(getattr(ts, "GRAETZ_N_RADIAL", 0)),
        "graetz_n_modes": int(getattr(ts, "GRAETZ_N_MODES", 0)),
        "total_volume": float(np.nansum(np.pi * radii * radii * lengths))
        if n
        else math.nan,
        "total_flowrate": float(qin_cm3_s),
        "pressure_in_root": float(getattr(tree.parameters, "root_pressure", math.nan)),
        "pressure_out_terminals": float(
            getattr(tree.parameters, "terminal_pressure", math.nan)
        ),
        "avg_radius": float(np.nanmean(radii)) if n else math.nan,
        "avg_length": float(np.nanmean(lengths)) if n else math.nan,
        "total_length": float(np.nansum(lengths)) if n else math.nan,
        "avg_distance_to_channel": math.nan,
        "terminal_segments": max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 0),
        "total_segments": int(n),
        "dlp_angle": 0.0,
        "Rnet": math.nan,
        "dRnet": math.nan,
        "Qmin_over_Qinlet": math.nan,
        "C_LQ_over_Cmax": math.nan,
        "C_tiss_over_Cmax": math.nan,
        "Damkohler": math.nan,
        "inlet_flow_ul_per_min": float(qin_cm3_s) * 60000.0,
    }
    details = {
        "starts": starts,
        "ends": ends,
        "radii": radii,
        "lengths": lengths,
        "flows": nan.copy(),
        "cin": nan.copy(),
        "cout": nan.copy(),
        "tissue_points": np.empty((0, 3), dtype=float),
        "tissue_values": np.empty((0,), dtype=float),
        "inlet_concentration": math.nan,
    }
    return summary, details


def _summary_row(
    ts,
    metrics: dict[str, Any],
    config: RunConfig,
    *,
    tree_id: int,
    n_trees: int,
    elapsed_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tree_id": int(tree_id),
        "number_of_trees": int(n_trees),
        "fluid": config.simulation.fluid,
        "dlp_angle": 0.0,
        "elapsed_s": float(elapsed_s),
    }
    row.update(metrics)
    for metric in getattr(ts, "AGGREGATED_METRICS", ()):
        if metric in metrics:
            row[f"{metric}_mean"] = metrics.get(metric)
            row[f"{metric}_std"] = 0.0
    return row


def _segment_rows(
    details: dict[str, Any],
    summary_row: dict[str, Any],
    *,
    tree_id: int,
    start_global_id: int,
) -> tuple[list[dict[str, Any]], int]:
    starts = np.asarray(details.get("starts", np.empty((0, 3))), dtype=float)
    ends = np.asarray(details.get("ends", np.empty((0, 3))), dtype=float)
    radii = np.asarray(details.get("radii", np.empty((0,))), dtype=float)
    lengths = np.asarray(details.get("lengths", np.empty((0,))), dtype=float)
    flows = np.asarray(
        details.get("flows", np.full(starts.shape[0], np.nan)), dtype=float
    )
    pressures = np.asarray(
        details.get("pressures", np.full(starts.shape[0], np.nan)), dtype=float
    )
    cin = np.asarray(details.get("cin", np.full(starts.shape[0], np.nan)), dtype=float)
    cout = np.asarray(
        details.get("cout", np.full(starts.shape[0], np.nan)), dtype=float
    )
    hd = np.asarray(details.get("discharge_hematocrit", np.empty((0,))), dtype=float)
    ht = np.asarray(details.get("tube_hematocrit", np.empty((0,))), dtype=float)
    rows: list[dict[str, Any]] = []
    for local_id in range(starts.shape[0]):
        global_id = start_global_id + local_id
        row = {
            "global_segment_id": int(global_id),
            "tree_id": int(tree_id),
            "local_segment_id": int(local_id),
            "start_x": float(starts[local_id, 0]),
            "start_y": float(starts[local_id, 1]),
            "start_z": float(starts[local_id, 2]),
            "end_x": float(ends[local_id, 0]),
            "end_y": float(ends[local_id, 1]),
            "end_z": float(ends[local_id, 2]),
            "radius_cm": float(radii[local_id]) if local_id < radii.size else math.nan,
            "length_cm": float(lengths[local_id])
            if local_id < lengths.size
            else math.nan,
            "flow_cm3_s": float(flows[local_id]) if local_id < flows.size else math.nan,
            "flow_ul_min": float(flows[local_id] * 60000.0)
            if local_id < flows.size
            else math.nan,
            # Runtime Kirchhoff arrays use dyn/cm^2; exports promise pascals.
            "pressure_pa": (
                float(pressures[local_id]) * DYN_PER_CM2_TO_PA
                if local_id < pressures.size
                else math.nan
            ),
            "cin": float(cin[local_id]) if local_id < cin.size else math.nan,
            "cout": float(cout[local_id]) if local_id < cout.size else math.nan,
            "pressure_in_root": summary_row.get("pressure_in_root", math.nan),
            "pressure_out_terminals": summary_row.get(
                "pressure_out_terminals", math.nan
            ),
        }
        if local_id < hd.size:
            row["discharge_hematocrit"] = float(hd[local_id])
        if local_id < ht.size:
            row["tube_hematocrit"] = float(ht[local_id])
        rows.append(row)
    return rows, start_global_id + starts.shape[0]


def _point_data(
    ts,
    tree_results: list[TreeSimulation],
    segment_rows: list[dict[str, Any]],
    sample_points: np.ndarray,
    config: RunConfig,
) -> dict[str, np.ndarray]:
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return {}
    if len(tree_results) == 1:
        pts = np.asarray(
            tree_results[0].details.get("tissue_points", np.empty((0, 3))), dtype=float
        )
        vals = np.asarray(
            tree_results[0].details.get("tissue_values", np.empty((0,))), dtype=float
        )
    else:
        pts, vals = _combined_tissue_points(ts, tree_results, sample_points)
    n_points = min(int(pts.shape[0]), int(vals.shape[0]))
    if n_points <= 0:
        return {}
    return _point_data_from_samples(
        ts,
        pts[:n_points],
        vals[:n_points],
        segment_rows,
        config,
    )


def _point_data_from_samples(
    ts,
    pts: np.ndarray,
    vals: np.ndarray,
    segment_rows: list[dict[str, Any]],
    config: RunConfig,
) -> dict[str, np.ndarray]:
    """Classify and annotate already-computed tissue samples."""
    pts = np.asarray(pts, dtype=float)
    vals = np.asarray(vals, dtype=float).reshape(-1)
    n_points = min(int(pts.shape[0]), int(vals.shape[0]))
    if n_points <= 0:
        return {}
    pts = pts[:n_points]
    vals = vals[:n_points]
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    viability_threshold = config.simulation.viability_threshold
    if viability_threshold is None:
        viability_threshold = 0.01 * conc_max if np.isfinite(conc_max) else math.nan
    nearest = None
    nearest_fields: dict[str, np.ndarray] = {}
    if config.outputs.include_tissue_nearest_fields and pts.size and segment_rows:
        starts, ends, radii = _segment_geometry_from_rows(segment_rows)
        nearest = compute_distance_to_nearest_channel(pts, starts, ends, radii)
        nearest_fields = _nearest_segment_fields(
            ts, pts, starts, ends, radii, segment_rows
        )
    finite_concentration = np.isfinite(vals)
    normalized = np.full(n_points, np.nan, dtype=float)
    if np.isfinite(conc_max) and conc_max:
        normalized[finite_concentration] = vals[finite_concentration] / conc_max
    viable = np.zeros(n_points, dtype=np.int32)
    if np.isfinite(viability_threshold):
        viable[finite_concentration] = (
            vals[finite_concentration] >= viability_threshold
        ).astype(np.int32)
    data = {
        "point_id": np.arange(n_points, dtype=np.int32),
        "x": np.asarray(pts[:, 0], dtype=float),
        "y": np.asarray(pts[:, 1], dtype=float),
        "z": np.asarray(pts[:, 2], dtype=float),
        "inside_tissue": np.ones(n_points, dtype=np.int32),
        "local_concentration": np.asarray(vals, dtype=float),
        # Preserve the established output names while keeping the concise
        # local_concentration field used by the current GUI.
        "local_concentration_raw": np.asarray(vals, dtype=float),
        "local_concentration_norm": normalized,
        "viability": viable,
    }
    if nearest is not None:
        data["dnc_cm"] = np.asarray(nearest[:n_points], dtype=float)
    for key, raw_values in nearest_fields.items():
        values = np.asarray(raw_values)[:n_points]
        data[key] = values.astype(
            np.int32 if key.endswith("_id") else float, copy=False
        )
    return data


def _point_rows_from_data(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Materialize CSV dictionaries only when the user explicitly requests CSV."""
    if not data:
        return []
    count = min(len(values) for values in data.values())
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append({key: values[index].item() for key, values in data.items()})
    return rows


def _combined_tissue_points(
    ts, tree_results: list[TreeSimulation], sample_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    starts = []
    ends = []
    radii = []
    cin = []
    flows = []
    for result in tree_results:
        details = result.details
        starts.append(np.asarray(details.get("starts", np.empty((0, 3))), dtype=float))
        ends.append(np.asarray(details.get("ends", np.empty((0, 3))), dtype=float))
        radii.append(np.asarray(details.get("radii", np.empty((0,))), dtype=float))
        cin.append(np.asarray(details.get("cin", np.empty((0,))), dtype=float))
        flows.append(np.asarray(details.get("flows", np.empty((0,))), dtype=float))
    if not starts or sample_points.size == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    starts_arr = np.concatenate(starts, axis=0)
    ends_arr = np.concatenate(ends, axis=0)
    radii_arr = np.concatenate(radii, axis=0)
    cext_states = [result.details.get("cext_source_state") for result in tree_results]
    if cext_states and all(isinstance(state, dict) for state in cext_states):
        combined = combine_network_solutions(
            [
                {
                    "starts": result.details["starts"],
                    "ends": result.details["ends"],
                    "radii": result.details["radii"],
                    "lengths": result.details["lengths"],
                    "flows": result.details["flows"],
                    "cin": result.details["cin"],
                    "cout": result.details["cout"],
                    "cext_state": result.details["cext_source_state"],
                }
                for result in tree_results
            ]
        )
        mask, conc = compute_tissue_samples_greens_from_cext_state(
            sample_points,
            starts_arr,
            ends_arr,
            radii_arr,
            combined["cext_state"],
            tissue_cache=None,
        )
        keep = np.asarray(mask, dtype=bool)
        return sample_points[keep], np.asarray(conc, dtype=float)[keep]
    cin_arr = np.concatenate(cin, axis=0)
    flows_arr = np.concatenate(flows, axis=0)
    mask, conc = compute_tissue_samples_greens(
        sample_points,
        starts_arr,
        ends_arr,
        radii_arr,
        cin_arr,
        flows_arr,
        diffusivity=float(_state.SOLUTE_DIFFUSIVITY),
        vmax=float(_state.VMAX_MM),
        km=float(_state.K_M_MM),
        window_factor=float(_state.WINDOW_FACTOR),
        inlet_concentration=float(get_concentration_inlet()),
        tissue_cache=None,
    )
    return sample_points[np.asarray(mask, dtype=bool)], np.asarray(conc, dtype=float)[
        np.asarray(mask, dtype=bool)
    ]


def _segment_geometry_from_rows(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.array(
        [[r["start_x"], r["start_y"], r["start_z"]] for r in rows], dtype=float
    )
    ends = np.array([[r["end_x"], r["end_y"], r["end_z"]] for r in rows], dtype=float)
    radii = np.array([r["radius_cm"] for r in rows], dtype=float)
    return starts, ends, radii


def _nearest_segment_fields(
    ts,
    pts: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    segment_rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    if not hasattr(ts, "_prepare_tissue_geometry"):
        return {}
    try:
        max_nearby = min(
            int(getattr(ts, "NEAREST_TISSUE_VESSELS", 250)), int(starts.shape[0])
        )
        cache = _prepare_tissue_geometry(
            pts, starts, ends, radii, max_nearby=max_nearby
        )
        nearest_idx = np.asarray(cache["nearest_idx"])
        d_center = np.asarray(cache["d_center"], dtype=np.float64)
        valid = np.asarray(cache["valid_mask"], dtype=bool)
        valid_global = np.flatnonzero(valid).astype(np.int64)
        if not nearest_idx.size or not d_center.size or not valid_global.size:
            return {}
        local = np.argmin(d_center, axis=1)
        row = np.arange(nearest_idx.shape[0])
        seg_valid = nearest_idx[row, local].astype(np.int64)
        seg_valid = np.clip(seg_valid, 0, max(valid_global.size - 1, 0))
        closest_global = valid_global[seg_valid]
        tree_ids = np.asarray(
            [r.get("tree_id", -1) for r in segment_rows], dtype=np.int64
        )
        flows = np.asarray(
            [r.get("flow_ul_min", np.nan) for r in segment_rows], dtype=float
        )
        return {
            "closest_segment_id": closest_global.astype(np.int64),
            "closest_tree_id": tree_ids[closest_global].astype(np.int64),
            "closest_flow_ul_min": flows[closest_global].astype(float),
        }
    except Exception as exc:
        print(f"Warning: failed to compute nearest tissue fields ({exc}).", flush=True)
        return {}
