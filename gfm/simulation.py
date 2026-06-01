from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any

import numpy as np

from .config import RunConfig
from .connectivity import collect_downstream_segment_ids
from .grid import sample_grid_points
from .growth import flow_for_tree, load_tissuesim_module, terminal_flow_for_target
from .simple import simple_details


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
    sample_meta: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class InfarctionState:
    applied: bool = False
    fraction_blocked: float = 0.0
    global_segment_id: int | None = None
    tree_id: int | None = None
    local_segment_id: int | None = None
    blocked_local_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64))


def run_simulation(
    domain,
    trees: list[Any],
    target_counts: list[int],
    config: RunConfig,
    *,
    sample_points: np.ndarray | None = None,
) -> SimulationResult:
    ts = load_tissuesim_module()
    timings: dict[str, float] = {}

    t0 = perf_counter()
    sample_points = _sample_points(ts, domain, config, provided=sample_points)
    timings["sample_points_resolve_s"] = perf_counter() - t0

    tree_results: list[TreeSimulation] = []
    summary_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    global_segment_id = 0
    want_point_rows = (
        (bool(config.outputs.write_points_csv) or bool(config.outputs.write_paraview))
        and not bool(config.simulation.geometry_only)
        and not bool(config.simulation.skip_tissue_oxygen)
    )
    want_segment_rows = (
        bool(config.outputs.write_segments_csv)
        or bool(config.outputs.write_paraview)
        or (want_point_rows and bool(config.outputs.include_tissue_nearest_fields))
    )

    for tree_id, (tree, target_count) in enumerate(zip(trees, target_counts)):
        t_tree = perf_counter()
        if getattr(tree, "_gfm_simple_network", False):
            summary, details = simple_details(tree, sample_points, ts, config)
            elapsed = perf_counter() - t_tree
            row = _summary_row(ts, summary, config, tree_id=tree_id, n_trees=len(trees), elapsed_s=elapsed)
            summary_rows.append(row)
            if want_segment_rows:
                rows, global_segment_id = _segment_rows(details, row, tree_id=tree_id, start_global_id=global_segment_id)
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
            ts._sync_loaded_tree_params(
                tree,
                side_length=float(config.domain.side_length),
                terminal_flow_override=terminal_flow,
            )
        with _infarction_override(config, tree, tree_id, global_segment_id) as infarction:
            if config.simulation.geometry_only:
                summary, details = _geometry_only_result(ts, tree, target_count, config, qin_cm3_s)
            else:
                tissue_cache = None
                if not config.simulation.skip_tissue_oxygen and sample_points.size:
                    t_cache = perf_counter()
                    tissue_cache = ts.build_tissue_cache_from_tree(tree, sample_points)
                    timings[f"tree_{tree_id}_tissue_cache_s"] = perf_counter() - t_cache
                summary, details = ts.summarize_tree(
                    tree,
                    sample_points,
                    int(target_count),
                    side_length=float(config.domain.side_length),
                    fluid=config.simulation.fluid,
                    inlet_flow_cm3_s=qin_cm3_s,
                    tissue_cache=tissue_cache,
                    concentration_solver=config.simulation.concentration_solver,
                    return_details=True,
                )
            if infarction.blocked_local_ids.size:
                _zero_blocked_solution(details, infarction.blocked_local_ids)

        elapsed = perf_counter() - t_tree
        row = _summary_row(ts, summary, config, tree_id=tree_id, n_trees=len(trees), elapsed_s=elapsed)
        summary_rows.append(row)
        if want_segment_rows:
            rows, global_segment_id = _segment_rows(details, row, tree_id=tree_id, start_global_id=global_segment_id)
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
    point_rows = _point_rows(ts, tree_results, segment_rows, sample_points, config) if want_point_rows else []
    timings["points_s"] = perf_counter() - t0
    return SimulationResult(
        tree_results=tree_results,
        summary_rows=summary_rows,
        segment_rows=segment_rows,
        point_rows=point_rows,
        sample_points=sample_points,
        sample_meta=_sample_meta_for_config(config, sample_points),
        timings=timings,
    )


def _detail_segment_count(details: dict[str, Any], tree) -> int:
    starts = details.get("starts")
    if starts is not None:
        try:
            return int(np.asarray(starts).shape[0])
        except Exception:
            pass
    return int(getattr(tree, "segment_count", 0) or 0)


def _sample_points(ts, domain, config: RunConfig, *, provided: np.ndarray | None = None) -> np.ndarray:
    if provided is not None:
        return np.asarray(provided, dtype=float)
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return np.empty((0, 3), dtype=float)
    if config.simulation.sample_mode == "grid":
        points, _ = sample_grid_points(domain, config.simulation.tissue_grid)
        return points
    n = int(config.simulation.distance_sample_count)
    if n <= 0:
        return np.empty((0, 3), dtype=float)
    return np.asarray(ts.sample_domain_points(domain, n), dtype=float)


def _sample_meta_for_config(config: RunConfig, points: np.ndarray) -> dict[str, Any]:
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return {"sample_mode": "none", "points": 0}
    if config.simulation.sample_mode == "grid":
        meta = {"sample_mode": "grid", "points": int(np.asarray(points).shape[0])}
        meta.update(dict(config.simulation.tissue_grid or {}))
        return meta
    return {
        "sample_mode": "random",
        "requested_points": int(config.simulation.distance_sample_count),
        "points": int(np.asarray(points).shape[0]),
    }


def _tree_root_flow_cm3_s(tree) -> float:
    params = getattr(tree, "parameters", None)
    if params is not None:
        value = getattr(params, "root_flow", None)
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        value = float(data[0, 22])
        if np.isfinite(value) and value > 0.0:
            return value
    return math.nan


def _inlet_flow_for_tree(config: RunConfig, tree, tree_id: int, trees: list[Any]) -> float:
    source = str(config.simulation.flow_source).strip().lower().replace("_", "-")
    if source == "tree-root-flow":
        root_flow = _tree_root_flow_cm3_s(tree)
        if np.isfinite(root_flow) and root_flow > 0.0:
            return float(root_flow)
        return flow_for_tree(config, tree_id, len(trees))
    if source == "total-qin-split":
        total_ul_min = (
            float(config.simulation.total_qin_ul_min)
            if config.simulation.total_qin_ul_min is not None
            else float(config.simulation.qin_target_ul_min)
        )
        total_cm3_s = total_ul_min * 1.0e-3 / 60.0 * (float(config.domain.side_length) ** 3)
        root_flows = np.array([_tree_root_flow_cm3_s(t) for t in trees], dtype=float)
        weights = np.where(np.isfinite(root_flows) & (root_flows > 0.0), root_flows, 0.0)
        if not np.any(weights > 0.0):
            weights = np.ones(len(trees), dtype=float)
        return float(total_cm3_s * weights[int(tree_id)] / float(np.sum(weights)))
    return flow_for_tree(config, tree_id, len(trees))


@contextmanager
def _infarction_override(config: RunConfig, tree, tree_id: int, tree_global_offset: int):
    state = _infarction_state_for_tree(config, tree, tree_id, tree_global_offset)
    if not state.applied:
        yield state
        return
    ids = state.blocked_local_ids
    data = np.asarray(tree.data)
    original = np.asarray(data[ids, 21], dtype=float).copy() if ids.size else np.empty((0,), dtype=float)
    try:
        if ids.size:
            if state.fraction_blocked >= 1.0:
                data[ids, 21] = 0.0
            else:
                data[int(state.local_segment_id), 21] = original[0] * (1.0 - float(state.fraction_blocked))
        yield state
    finally:
        if ids.size:
            data[ids, 21] = original


def _infarction_state_for_tree(config: RunConfig, tree, tree_id: int, tree_global_offset: int) -> InfarctionState:
    raw = dict(config.simulation.infarction or {})
    fraction = float(raw.get("fraction_blocked", raw.get("fraction_blocked_infarction", 0.0)) or 0.0)
    if not np.isfinite(fraction) or fraction <= 0.0:
        return InfarctionState()
    if fraction > 1.0:
        raise ValueError("simulation.infarction.fraction_blocked must be in [0, 1].")
    if config.simulation.geometry_only:
        return InfarctionState()
    target_global = raw.get("global_segment_id", raw.get("infarction_global_segment_id"))
    if target_global is None:
        raise ValueError("simulation.infarction.global_segment_id is required when fraction_blocked > 0.")
    target_global = int(target_global)
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    local = target_global - int(tree_global_offset)
    if local < 0 or local >= seg_count:
        return InfarctionState()
    blocked = np.array([local], dtype=np.int64)
    if fraction >= 1.0:
        blocked = collect_downstream_segment_ids(tree, int(local))
    return InfarctionState(
        applied=True,
        fraction_blocked=float(fraction),
        global_segment_id=int(target_global),
        tree_id=int(tree_id),
        local_segment_id=int(local),
        blocked_local_ids=np.asarray(blocked, dtype=np.int64),
    )


def _zero_blocked_solution(details: dict[str, Any], blocked_local_ids: np.ndarray) -> None:
    ids = np.asarray(blocked_local_ids, dtype=np.int64)
    if ids.size == 0:
        return
    for key in ("flows", "cin", "cout"):
        arr = details.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr)
        valid = ids[ids < arr.shape[0]]
        if valid.size:
            arr[valid] = 0.0


def _geometry_only_result(ts, tree, target_count: int, config: RunConfig, qin_cm3_s: float) -> tuple[dict, dict]:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:seg_count]) if seg_count > 0 else np.empty((0, 31), dtype=float)
    starts = data[:, 0:3] if data.size else np.empty((0, 3), dtype=float)
    ends = data[:, 3:6] if data.size else np.empty((0, 3), dtype=float)
    radii = data[:, 21] if data.size and data.shape[1] > 21 else np.empty((0,), dtype=float)
    lengths = data[:, 20] if data.size and data.shape[1] > 20 else np.empty((0,), dtype=float)
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
        "total_volume": float(np.nansum(np.pi * radii * radii * lengths)) if n else math.nan,
        "total_flowrate": float(qin_cm3_s),
        "pressure_in_root": float(getattr(tree.parameters, "root_pressure", math.nan)),
        "pressure_out_terminals": float(getattr(tree.parameters, "terminal_pressure", math.nan)),
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


def _summary_row(ts, metrics: dict[str, Any], config: RunConfig, *, tree_id: int, n_trees: int, elapsed_s: float) -> dict[str, Any]:
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
    flows = np.asarray(details.get("flows", np.full(starts.shape[0], np.nan)), dtype=float)
    cin = np.asarray(details.get("cin", np.full(starts.shape[0], np.nan)), dtype=float)
    cout = np.asarray(details.get("cout", np.full(starts.shape[0], np.nan)), dtype=float)
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
            "length_cm": float(lengths[local_id]) if local_id < lengths.size else math.nan,
            "flow_cm3_s": float(flows[local_id]) if local_id < flows.size else math.nan,
            "flow_ul_min": float(flows[local_id] * 60000.0) if local_id < flows.size else math.nan,
            "cin": float(cin[local_id]) if local_id < cin.size else math.nan,
            "cout": float(cout[local_id]) if local_id < cout.size else math.nan,
            "pressure_in_root": summary_row.get("pressure_in_root", math.nan),
            "pressure_out_terminals": summary_row.get("pressure_out_terminals", math.nan),
        }
        if local_id < hd.size:
            row["discharge_hematocrit"] = float(hd[local_id])
        if local_id < ht.size:
            row["tube_hematocrit"] = float(ht[local_id])
        rows.append(row)
    return rows, start_global_id + starts.shape[0]


def _point_rows(
    ts,
    tree_results: list[TreeSimulation],
    segment_rows: list[dict[str, Any]],
    sample_points: np.ndarray,
    config: RunConfig,
) -> list[dict[str, Any]]:
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return []
    if len(tree_results) == 1:
        pts = np.asarray(tree_results[0].details.get("tissue_points", np.empty((0, 3))), dtype=float)
        vals = np.asarray(tree_results[0].details.get("tissue_values", np.empty((0,))), dtype=float)
    else:
        pts, vals = _combined_tissue_points(ts, tree_results, sample_points)
    rows = []
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    nearest = None
    nearest_fields: dict[str, np.ndarray] = {}
    if config.outputs.include_tissue_nearest_fields and pts.size and segment_rows:
        starts, ends, radii = _segment_geometry_from_rows(segment_rows)
        nearest = ts.compute_distance_to_nearest_channel(pts, starts, ends, radii)
        nearest_fields = _nearest_segment_fields(ts, pts, starts, ends, radii, segment_rows)
    for i in range(pts.shape[0]):
        c = float(vals[i]) if i < vals.size else math.nan
        row = {
            "point_id": int(i),
            "x": float(pts[i, 0]),
            "y": float(pts[i, 1]),
            "z": float(pts[i, 2]),
            "inside_tissue": 1,
            "local_concentration": c,
            "local_concentration_raw": c,
            "local_concentration_norm": c / conc_max if np.isfinite(c) and np.isfinite(conc_max) and conc_max else math.nan,
            "viability": int(c >= 0.01 * conc_max) if np.isfinite(c) and np.isfinite(conc_max) and conc_max else 0,
        }
        if nearest is not None:
            row["dnc_cm"] = float(nearest[i])
        for key, values in nearest_fields.items():
            if i < values.size:
                value = values[i]
                row[key] = int(value) if key.endswith("_id") else float(value)
        rows.append(row)
    return rows


def _combined_tissue_points(ts, tree_results: list[TreeSimulation], sample_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    cin_arr = np.concatenate(cin, axis=0)
    flows_arr = np.concatenate(flows, axis=0)
    mask, conc = ts.compute_tissue_samples_greens(
        sample_points,
        starts_arr,
        ends_arr,
        radii_arr,
        cin_arr,
        flows_arr,
        diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
        window_factor=float(ts.WINDOW_FACTOR),
        inlet_concentration=float(ts.get_concentration_inlet()),
        tissue_cache=None,
    )
    return sample_points[np.asarray(mask, dtype=bool)], np.asarray(conc, dtype=float)[np.asarray(mask, dtype=bool)]


def _segment_geometry_from_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.array([[r["start_x"], r["start_y"], r["start_z"]] for r in rows], dtype=float)
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
        max_nearby = min(int(getattr(ts, "NEAREST_TISSUE_VESSELS", 250)), int(starts.shape[0]))
        cache = ts._prepare_tissue_geometry(pts, starts, ends, radii, max_nearby=max_nearby)
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
        tree_ids = np.asarray([r.get("tree_id", -1) for r in segment_rows], dtype=np.int64)
        flows = np.asarray([r.get("flow_ul_min", np.nan) for r in segment_rows], dtype=float)
        return {
            "closest_segment_id": closest_global.astype(np.int64),
            "closest_tree_id": tree_ids[closest_global].astype(np.int64),
            "closest_flow_ul_min": flows[closest_global].astype(float),
        }
    except Exception as exc:
        print(f"Warning: failed to compute nearest tissue fields ({exc}).", flush=True)
        return {}
