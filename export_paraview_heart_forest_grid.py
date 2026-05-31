#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import sys
import types
from pathlib import Path
from time import perf_counter

import numpy as np
import pyvista as pv
from tqdm import tqdm

from svv.domain.domain import Domain
from svv.forest.forest import Forest


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FOREST_DIR = SCRIPT_DIR / "FORESTS"
DEFAULT_FOREST = DEFAULT_FOREST_DIR / "heart_seed2_grown25M.forest"
DEFAULT_OUT_DIR = SCRIPT_DIR / "vtks_forest"
DEFAULT_DOMAIN = DEFAULT_FOREST_DIR / "tree_87b94fb1cc22bdd55af637e8910d3fa5_t1001.f64.tree.dmn"
UNC_PREFIX = "\\\\wsl.localhost\\Ubuntu\\"

# Oxygen/concentration defaults. These intentionally mirror TissueSim_heart_accel.py
# so this exporter can be configured from the top of the script or from CLI.
TISSUE_ACCEL_MODE_DEFAULT = "gpu"
TISSUE_GPU_CHUNK_POINTS_DEFAULT = 8192
TISSUE_GPU_VALIDATE_POINTS_DEFAULT = 0
SOLUTE_DIFFUSIVITY_DEFAULT = 2.41e-5
VMAX_MM_DEFAULT = 2e-16 * 2.2e13
# VMAX_MM_DEFAULT = 0.04
K_M_MM_DEFAULT = 0.0069
NEAREST_TISSUE_VESSELS_DEFAULT = 128
WINDOW_FACTOR_DEFAULT = 4.0
GL_ORDER_DEFAULT = 5
TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT = 6
AXIAL_BLOOD_STEPS_DEFAULT = 5
CONC_MAX_FOR_NORMALIZATION_DEFAULT = 0.14
HEMATOCRIT_MODEL_DEFAULT = "pries_secomb"
HEMATOCRIT_FLOW_ITERATIONS_DEFAULT = 10
HEMATOCRIT_RELAXATION_DEFAULT = 1.0
HEMATOCRIT_QTOL_NL_MIN_DEFAULT = 1.0e-3
HEMATOCRIT_HDTOL_DEFAULT = 1.0e-3
EXPORT_FLOAT_DTYPE_DEFAULT = "float32"
EXPORT_INDEX_DTYPE_DEFAULT = "int64"


def _normalize_path(value: str | Path) -> Path:
    s = str(value).strip()
    if s.startswith(UNC_PREFIX):
        s = "/" + s[len(UNC_PREFIX):].replace("\\", "/")
    return Path(s).expanduser()


def _float_dtype_from_name(value: str) -> np.dtype:
    text = str(value or "float32").strip().lower()
    if text in {"float32", "f32", "32"}:
        return np.dtype(np.float32)
    if text in {"float64", "f64", "64"}:
        return np.dtype(np.float64)
    raise ValueError(f"Unknown export float dtype: {value!r}")


def _int_dtype_from_name(value: str) -> np.dtype:
    text = str(value or "int32").strip().lower()
    if text in {"int32", "i32", "32"}:
        return np.dtype(np.int32)
    if text in {"int64", "i64", "64"}:
        return np.dtype(np.int64)
    raise ValueError(f"Unknown export index dtype: {value!r}")


def _resolve_forest_path(value: str | Path) -> Path:
    raw = _normalize_path(value)
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw
    return DEFAULT_FOREST_DIR / raw.name


def _default_simulation_cache_path(forest_path: Path) -> Path:
    return forest_path.with_name(forest_path.name + ".simcache")


def _should_use_simulation_cache(source_path: Path, cache_path: Path) -> bool:
    return (
        cache_path.exists()
        and Forest._is_simulation_cache(str(cache_path))
        and cache_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    )


def _load_tissuesim():
    try:
        import pymeshfix  # noqa: F401
    except Exception:
        class _DummyMeshFix:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("pymeshfix is not installed; mesh repair is unavailable.")

        sys.modules.setdefault("pymeshfix", types.SimpleNamespace(MeshFix=_DummyMeshFix))

    import TissueSim_heart_accel as ts

    return ts


def _apply_tissuesim_overrides(ts, args) -> dict:
    overrides = {
        "TISSUE_ACCEL_MODE": str(args.tissue_accel),
        "TISSUE_GPU_CHUNK_POINTS": int(args.tissue_gpu_chunk_points),
        "TISSUE_GPU_VALIDATE_POINTS": int(args.tissue_gpu_validate_points),
        "SOLUTE_DIFFUSIVITY": float(args.solute_diffusivity),
        "VMAX_MM": float(args.vmax_mm),
        "K_M_MM": float(args.km_mm),
        "NEAREST_TISSUE_VESSELS": int(args.nearest_tissue_vessels),
        "WINDOW_FACTOR": float(args.window_factor),
        "GL_ORDER": int(args.gl_order),
        "TISSUE_KDTREE_CANDIDATE_MULT": int(args.tissue_kdtree_candidate_mult),
        "AXIAL_BLOOD_STEPS": int(args.axial_blood_steps),
        "CONC_MAX_FOR_NORMALIZATION": float(args.conc_max_for_normalization),
        "HEMATOCRIT_MODEL": str(args.hematocrit_model),
        "HEMATOCRIT_FLOW_ITERATIONS": int(args.hematocrit_flow_iterations),
        "HEMATOCRIT_RELAXATION": float(args.hematocrit_relaxation),
        "HEMATOCRIT_QTOL_NL_MIN": float(args.hematocrit_qtol_nl_min),
        "HEMATOCRIT_HDTOL": float(args.hematocrit_hdtol),
    }
    if args.kirchhoff_bc_mode is not None:
        overrides["KIRCHHOFF_BC_MODE"] = str(args.kirchhoff_bc_mode)
    for name, value in overrides.items():
        if hasattr(ts, name):
            setattr(ts, name, value)
    if hasattr(ts, "CONCENTRATION_SOLVER"):
        ts.CONCENTRATION_SOLVER = str(args.concentration_solver)
    return {name: getattr(ts, name, value) for name, value in overrides.items()}


def _log(message: str) -> None:
    print(message, flush=True)


def _build_domain(ts, domain_path: Path, side_length: float):
    suffix = str(domain_path.suffix).strip().lower()
    if suffix == ".dmn":
        return Domain.load(str(domain_path))
    if hasattr(ts, "DEFAULT_STL"):
        ts.DEFAULT_STL = str(domain_path)
    if hasattr(ts, "DOMAIN_CACHE_PATH"):
        # The heart forest can use arbitrary imported root geometry; avoid accidentally
        # reusing an incompatible single-tree domain cache.
        ts.DOMAIN_CACHE_PATH = None
    return ts.build_domain(float(side_length))


def _attach_domain(forest: Forest, domain) -> None:
    try:
        forest.attach_domain_for_simulation(domain)
    except Exception:
        forest.domain = domain
        forest.geodesic = None
        for tree in forest.networks[0]:
            tree.set_domain(domain)
            tree.domain = domain


def _normalize_forest_dtype_attrs(forest: Forest) -> list[dict]:
    summaries: list[dict] = []
    for tree_id, tree in enumerate(forest.networks[0]):
        data = np.asarray(tree.data)
        conn = getattr(tree, "connectivity", None)
        conn_arr = np.asarray(conn) if conn is not None else np.empty((0, 3), dtype=np.int64)
        data_dtype = data.dtype if data.dtype.kind == "f" else np.dtype(np.float64)
        index_dtype = conn_arr.dtype if conn is not None and conn_arr.dtype.kind == "i" else np.dtype(getattr(tree, "index_dtype", np.int64))
        tree.data_dtype = np.dtype(data_dtype)
        tree.index_dtype = np.dtype(index_dtype)
        summaries.append(
            {
                "tree_id": int(tree_id),
                "segments": int(getattr(tree, "segment_count", 0) or 0),
                "terminals": int(getattr(tree, "n_terminals", 0) or 0),
                "data_dtype": str(data.dtype),
                "preallocate_dtype": str(np.asarray(getattr(tree, "preallocate", [])).dtype),
                "connectivity_dtype": str(conn_arr.dtype),
                "data_dtype_attr": str(tree.data_dtype),
                "index_dtype_attr": str(tree.index_dtype),
            }
        )
    return summaries


def _make_tree_analysis_only(tree) -> None:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:seg_count])
    conn = getattr(tree, "connectivity", None)
    if conn is not None:
        conn_arr = np.asarray(conn)
        if conn_arr.dtype.kind == "i":
            tree.index_dtype = conn_arr.dtype
    tree.preallocate = tree.data
    tree.preallocation_step = int(seg_count)
    tree.preallocate_midpoints = np.empty((0, 3), dtype=getattr(tree, "data_dtype", data.dtype))
    tree.midpoints = tree.preallocate_midpoints
    tree.vessel_map = {}
    tree.kdtm = None
    tree.hnsw_tree = None
    tree.hnsw_tree_id = None
    tree.connectivity = None
    tree._analysis_only_load = True


def _make_forest_analysis_only(forest: Forest) -> None:
    for tree in forest.networks[0]:
        _make_tree_analysis_only(tree)


def _repair_tree_parent_columns_from_children(tree) -> int:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if seg_count <= 0:
        return 0
    data = np.asarray(tree.data[:seg_count])
    index_dtype = getattr(tree, "index_dtype", np.int64)
    conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)
    # Some older forest-export repairs wrote -1 into tree.data child slots.
    # SVV's solver code treats NaN child slots as terminal leaves, so keep -1
    # only in tree.connectivity and preserve/restore NaNs in tree.data.
    child_conn = conn[:, 0:2]
    child_conn[data[:, 15:17] < 0] = -1
    conn[:, 0:2] = child_conn

    rows = np.arange(seg_count, dtype=np.int64)
    parent = np.full(seg_count, -1, dtype=np.int64)
    child_counts = np.zeros(seg_count, dtype=np.uint8)
    for col in (0, 1):
        children = conn[:, col].astype(np.int64, copy=False)
        valid = children >= 0
        if not np.any(valid):
            continue
        valid_children = children[valid]
        max_child = int(valid_children.max())
        if max_child >= seg_count:
            row = int(rows[valid][int(np.argmax(valid_children))])
            raise RuntimeError(f"Connectivity child index out of range: row={row} child={max_child}")
        np.add.at(child_counts, valid_children, 1)
        parent[valid_children] = rows[valid]
    duplicate_children = np.flatnonzero(child_counts > 1)
    if duplicate_children.size:
        raise RuntimeError(f"Connectivity has duplicate child references: {duplicate_children[:5].tolist()}")

    changed = conn[:, 2].astype(np.int64, copy=False) != parent
    n_changed = int(np.count_nonzero(changed))
    conn[:, 2] = parent.astype(conn.dtype, copy=False)
    data_conn = conn.astype(float)
    data_conn[data_conn < 0] = np.nan
    tree.data[:seg_count, 15:18] = data_conn.astype(np.asarray(tree.data).dtype, copy=False)
    try:
        tree.preallocate[:seg_count, 15:18] = data_conn.astype(np.asarray(tree.preallocate).dtype, copy=False)
    except Exception:
        pass
    tree.connectivity = conn.astype(index_dtype, copy=False)
    return n_changed


def _repair_forest_connectivity(forest: Forest) -> list[int]:
    repairs = []
    for tree in forest.networks[0]:
        repairs.append(_repair_tree_parent_columns_from_children(tree))
    return repairs


def _connectivity_report(tree, *, geometry_atol: float = 1e-6) -> dict:
    n = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:n])
    if n == 0:
        return {"segments": 0, "roots": [], "reachable": 0, "bad_parent": 0, "bad_child": 0, "bad_geom": 0}
    conn = np.nan_to_num(data[:, 15:18], nan=-1).astype(int)
    roots = np.flatnonzero(conn[:, 2] < 0)
    row_ids = np.arange(n, dtype=np.int64)

    parent = conn[:, 2].astype(np.int64, copy=False)
    valid_parent = parent >= 0
    bad_parent_range = valid_parent & (parent >= n)
    good_parent = valid_parent & (parent < n)
    bad_parent = int(np.count_nonzero(bad_parent_range))
    bad_geom = 0
    if np.any(good_parent):
        rows = row_ids[good_parent]
        parents = parent[good_parent]
        parent_children = conn[parents, 0:2]
        parent_links_back = np.any(parent_children == rows[:, None], axis=1)
        bad_parent += int(np.count_nonzero(~parent_links_back))
        geom_ok = np.all(
            np.abs(data[parents, 3:6] - data[rows, 0:3]) <= float(geometry_atol),
            axis=1,
        )
        bad_geom += int(np.count_nonzero(parent_links_back & ~geom_ok))

    child_conn = conn[:, 0:2].astype(np.int64, copy=False)
    valid_child = child_conn >= 0
    bad_child_range = valid_child & (child_conn >= n)
    bad_child = int(np.count_nonzero(bad_child_range))
    good_child = valid_child & (child_conn < n)
    if np.any(good_child):
        parent_rows, child_cols = np.nonzero(good_child)
        child_ids = child_conn[parent_rows, child_cols]
        child_links_back = conn[child_ids, 2] == parent_rows
        bad_child += int(np.count_nonzero(~child_links_back))
        geom_ok = np.all(
            np.abs(data[parent_rows, 3:6] - data[child_ids, 0:3]) <= float(geometry_atol),
            axis=1,
        )
        bad_geom += int(np.count_nonzero(child_links_back & ~geom_ok))

    # For a repaired rooted tree, one root plus valid reciprocal parent/child
    # references is enough for the exporter. Avoid a Python DFS over tens of
    # millions of rows; disconnected cycles would already be non-tree input.
    reachable = n if len(roots) == 1 and bad_parent == 0 and bad_child == 0 else 0
    return {
        "segments": n,
        "roots": roots.tolist(),
        "reachable": int(reachable),
        "bad_parent": int(bad_parent),
        "bad_child": int(bad_child),
        "bad_geom": int(bad_geom),
    }


def _validate_forest_connectivity(forest: Forest, *, fail: bool, geometry_atol: float) -> list[dict]:
    reports = []
    for tree_id, tree in enumerate(forest.networks[0]):
        report = _connectivity_report(tree, geometry_atol=float(geometry_atol))
        report["tree_id"] = int(tree_id)
        reports.append(report)
        _log(
            "Connectivity tree={tree_id}: segments={segments} roots={roots} "
            "reachable={reachable}/{segments} bad_parent={bad_parent} "
            "bad_child={bad_child} bad_geom={bad_geom}".format(**report)
        )
        bad = (
            len(report["roots"]) != 1
            or int(report["reachable"]) != int(report["segments"])
            or int(report["bad_parent"]) > 0
            or int(report["bad_child"]) > 0
            or int(report["bad_geom"]) > 0
        )
        if fail and bad:
            raise RuntimeError(f"Connectivity validation failed for tree {tree_id}: {report}")
    return reports


def _tree_root_flow_cm3_s(tree) -> float:
    params = getattr(tree, "parameters", None)
    if params is not None:
        val = getattr(params, "root_flow", None)
        if val is not None and np.isfinite(float(val)) and float(val) > 0.0:
            return float(val)
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        val = float(data[0, 22])
        if np.isfinite(val) and val > 0.0:
            return val
    return 1.0


def _flow_inputs(args, forest: Forest) -> list[float]:
    trees = list(forest.networks[0])
    root_flows = np.array([_tree_root_flow_cm3_s(tree) for tree in trees], dtype=float)
    if str(args.flow_source).lower() == "tree-root-flow":
        return [float(v) for v in root_flows]
    if args.total_qin_ul_min is None:
        raise ValueError("--total-qin-ul-min is required when --flow-source total-qin-split")
    total_qin_cm3_s = float(args.total_qin_ul_min) * 1.0e-3 / 60.0
    weights = np.maximum(root_flows, 0.0)
    if not np.any(weights > 0.0):
        weights = np.ones(len(trees), dtype=float)
    flows = total_qin_cm3_s * weights / float(np.sum(weights))
    return [float(v) for v in flows]


def _safe_index_array(values, dtype: np.dtype) -> np.ndarray:
    arr64 = np.asarray(values, dtype=np.int64)
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int32):
        info = np.iinfo(np.int32)
        if arr64.size and (int(np.nanmax(arr64)) > info.max or int(np.nanmin(arr64)) < info.min):
            raise OverflowError("Index array cannot be represented as int32")
    return arr64.astype(dtype, copy=False)


def _collect_downstream_segment_ids(tree, segment_id: int, seg_count: int) -> np.ndarray:
    if seg_count <= 0 or segment_id < 0 or segment_id >= seg_count:
        return np.empty((0,), dtype=np.int64)
    conn = getattr(tree, "connectivity", None)
    if conn is None or np.asarray(conn).size == 0:
        data = np.asarray(tree.data[:seg_count])
        conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(np.int64)
    else:
        conn = np.asarray(conn[:seg_count], dtype=np.int64)
    seen = np.zeros(seg_count, dtype=bool)
    stack = [int(segment_id)]
    ordered: list[int] = []
    while stack:
        current = int(stack.pop())
        if current < 0 or current >= seg_count or seen[current]:
            continue
        seen[current] = True
        ordered.append(current)
        children = np.asarray(conn[current, 0:2], dtype=np.int64).reshape(-1)
        for child in children[::-1]:
            if child >= 0:
                stack.append(int(child))
    return np.asarray(ordered, dtype=np.int64)


def _resolve_global_segment_id(forest: Forest, global_segment_id: int) -> tuple[int, int]:
    offset = 0
    target = int(global_segment_id)
    for tree_id, tree in enumerate(forest.networks[0]):
        seg_count = int(getattr(tree, "segment_count", 0) or 0)
        if target < offset + seg_count:
            return int(tree_id), int(target - offset)
        offset += seg_count
    raise ValueError(f"Global segment id {global_segment_id} is out of range for {offset} total segments")


def _solve_tree(ts, tree, inlet_flow_cm3_s: float, *, fluid: str, concentration_solver: str | None) -> dict:
    inlet_concentration = float(ts.get_concentration_inlet(fluid))
    _log(
        f"  Solving tree: segments={int(tree.segment_count)} terminals={int(tree.n_terminals)} "
        f"qin={inlet_flow_cm3_s:.9g} cm3/s"
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
        prox_ids,
        dist_ids,
    ) = ts.recompute_tree_flows(tree, inlet_flow_cm3_s, fluid=fluid)
    _log(f"    flow recompute completed in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    cin, cout = ts._solve_channel_concentrations(
        tree,
        flows,
        inlet_nodes,
        outlet_nodes,
        starts,
        ends,
        radii,
        lengths,
        prox_ids=prox_ids,
        dist_ids=dist_ids,
        inlet_concentration=inlet_concentration,
        diffusivity=float(getattr(ts, "SOLUTE_DIFFUSIVITY", 3e-5)),
        vmax=float(getattr(ts, "VMAX_MM", 1.0)),
        km=float(getattr(ts, "K_M_MM", 1.0)),
        fluid=fluid,
        solver=concentration_solver,
    )
    _log(f"    concentration solve completed in {perf_counter() - t0:.2f}s")
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    return {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(cin),
        "cout": np.asarray(cout),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
    }


def _concat_tree_solutions(solutions: list[dict], *, index_dtype: np.dtype) -> dict:
    out = {}
    for key in ("starts", "ends", "radii", "lengths", "flows", "cin", "cout"):
        out[key] = np.concatenate([s[key] for s in solutions], axis=0) if solutions else np.empty((0,))
    tree_ids = []
    local_ids = []
    for tree_id, sol in enumerate(solutions):
        n = int(sol["starts"].shape[0])
        tree_ids.append(np.full(n, tree_id, dtype=np.int16))
        local_ids.append(_safe_index_array(np.arange(n, dtype=np.int64), index_dtype))
    out["tree_id"] = np.concatenate(tree_ids) if tree_ids else np.empty((0,), dtype=np.int16)
    out["local_segment_id"] = np.concatenate(local_ids) if local_ids else np.empty((0,), dtype=index_dtype)
    out["global_segment_id"] = _safe_index_array(np.arange(out["tree_id"].shape[0], dtype=np.int64), index_dtype)
    return out


def _build_vessel_polydata(combo: dict, *, resolution: int, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    starts = np.asarray(combo["starts"], dtype=float_dtype)
    ends = np.asarray(combo["ends"], dtype=float_dtype)
    nseg = int(starts.shape[0])
    if nseg == 0:
        return pv.PolyData()
    res = max(int(resolution), 2)
    t = np.linspace(0.0, 1.0, res, dtype=float_dtype)
    _log(f"Building vessel VTP arrays: segments={nseg} resolution={res}")
    points = (starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]).reshape(nseg * res, 3)
    line_ids = np.arange(nseg * res, dtype=np.int64).reshape(nseg, res)
    lines = np.empty((nseg, res + 1), dtype=np.int64)
    lines[:, 0] = res
    lines[:, 1:] = line_ids
    poly = pv.PolyData(points, lines=lines.reshape(-1))
    cin = np.asarray(combo["cin"], dtype=float_dtype)
    cout = np.asarray(combo["cout"], dtype=float_dtype)
    flow = np.asarray(combo["flows"], dtype=float_dtype)
    radii = np.asarray(combo["radii"], dtype=float_dtype)
    lengths = np.asarray(combo["lengths"], dtype=float_dtype)
    tree_ids = np.asarray(combo["tree_id"], dtype=np.int16)
    local_ids = np.asarray(combo["local_segment_id"], dtype=index_dtype)
    global_ids = np.asarray(combo["global_segment_id"], dtype=index_dtype)
    conc_samples = (cin[:, None] + t[None, :] * (cout - cin)[:, None]).reshape(nseg * res)
    poly.point_data["concentration"] = conc_samples
    poly.point_data["flow_cm3_s"] = np.repeat(flow, res)
    poly.point_data["flow_ul_min"] = np.repeat(flow * 60000.0, res).astype(float_dtype, copy=False)
    poly.point_data["radius"] = np.repeat(radii, res)
    poly.point_data["length"] = np.repeat(lengths, res)
    poly.point_data["tree_id"] = np.repeat(tree_ids, res)
    poly.point_data["local_segment_id"] = np.repeat(local_ids, res)
    poly.point_data["global_segment_id"] = np.repeat(global_ids, res)
    return poly


def _build_points_polydata(points: np.ndarray, metrics: dict[str, np.ndarray], *, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    points = np.asarray(points, dtype=float_dtype)
    if points.size == 0:
        return pv.PolyData()
    n = points.shape[0]
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).reshape(-1)
    poly = pv.PolyData(points, verts=verts)
    poly.point_data["point_id"] = _safe_index_array(np.arange(n, dtype=np.int64), index_dtype)
    poly.point_data["x"] = points[:, 0]
    poly.point_data["y"] = points[:, 1]
    poly.point_data["z"] = points[:, 2]
    for key, values in metrics.items():
        arr = np.asarray(values)
        if arr.shape[0] != n:
            raise ValueError(f"Metric {key} length {arr.shape[0]} != point count {n}")
        if arr.dtype.kind == "f":
            arr = arr.astype(float_dtype, copy=False)
        elif key in {"tree_id", "closest_tree_id"}:
            arr = arr.astype(np.int16, copy=False)
        elif key.endswith("_id") or key in {"point_id", "local_segment_id", "global_segment_id"}:
            arr = _safe_index_array(arr.astype(np.int64, copy=False), index_dtype)
        poly.point_data[key] = arr
    return poly


def _filter_export_points(points: np.ndarray, metrics: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if points.shape[0] != keep.shape[0]:
        raise ValueError(f"Export mask length {keep.shape[0]} != point count {points.shape[0]}")
    filtered_metrics: dict[str, np.ndarray] = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        if arr.shape[0] != keep.shape[0]:
            raise ValueError(f"Metric {key} length {arr.shape[0]} != export mask length {keep.shape[0]}")
        filtered_metrics[key] = arr[keep]
    return points[keep], filtered_metrics


def _get_boundary(domain, boundary_resolution: int) -> pv.PolyData:
    boundary = getattr(domain, "boundary", None)
    if boundary is None and getattr(domain, "mesh", None) is not None:
        boundary = domain.mesh.extract_surface()
    if boundary is None:
        domain.build(resolution=int(boundary_resolution), skip_boundary=False)
        boundary = domain.boundary
    if boundary is None:
        raise RuntimeError("Could not obtain domain boundary.")
    if not boundary.is_all_triangles:
        boundary = boundary.triangulate()
    return boundary.clean()


def _grid_axes(boundary: pv.PolyData, nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bmin = np.min(boundary.points, axis=0).astype(np.float64)
    bmax = np.max(boundary.points, axis=0).astype(np.float64)
    x = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    y = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    z = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    return x, y, z


def _grid_points_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _grid_points(boundary: pv.PolyData, nx: int, ny: int, nz: int) -> np.ndarray:
    return _grid_points_from_axes(*_grid_axes(boundary, nx, ny, nz))


def _inside_mask(domain, boundary: pv.PolyData, points: np.ndarray, args) -> np.ndarray:
    implicit = np.asarray(domain(points)).reshape(-1) <= -float(args.implicit_margin)
    if bool(args.disable_enclosed_check):
        return implicit
    try:
        cloud = pv.PolyData(points.astype(np.float64))
        selected = cloud.select_enclosed_points(
            boundary,
            tolerance=float(args.enclosed_tolerance),
            check_surface=False,
        )
        enclosed = np.asarray(selected.point_data["SelectedPoints"]).astype(bool).reshape(-1)
        if args.inside_combine_mode == "or":
            return np.logical_or(implicit, enclosed)
        return np.logical_and(implicit, enclosed)
    except Exception as exc:
        _log(f"Warning: enclosed-point check failed ({exc}); using implicit-only mask.")
        return implicit


def _inside_grid_points_chunked(domain, boundary: pv.PolyData, args) -> np.ndarray:
    x, y, z = _grid_axes(boundary, int(args.nx), int(args.ny), int(args.nz))
    total_points = int(x.size * y.size * z.size)
    chunk_points = max(int(getattr(args, "tissue_grid_chunk_points", 250_000)), 1)
    yz_count = max(int(y.size * z.size), 1)
    x_step = max(1, min(int(x.size), chunk_points // yz_count))
    chunks = []
    inside_total = 0
    with tqdm(total=total_points, desc="Filtering tissue grid", unit="pt") as progress:
        for start in range(0, int(x.size), x_step):
            stop = min(start + x_step, int(x.size))
            chunk = _grid_points_from_axes(x[start:stop], y, z)
            implicit = np.asarray(domain(chunk)).reshape(-1) <= -float(args.implicit_margin)
            if np.any(implicit):
                inside_chunk = chunk[implicit]
                chunks.append(inside_chunk)
                inside_total += int(inside_chunk.shape[0])
            progress.update(int(chunk.shape[0]))
            del chunk, implicit
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    _log(f"Chunked inside-domain filter retained {inside_total} / {total_points} points")
    return np.concatenate(chunks, axis=0)


def _compute_tissue(
    ts,
    combo: dict,
    domain,
    args,
    *,
    float_dtype: np.dtype,
    index_dtype: np.dtype,
) -> tuple[pv.PolyData, dict]:
    boundary = _get_boundary(domain, int(args.boundary_resolution))
    total_grid_points = int(args.nx) * int(args.ny) * int(args.nz)
    _log(f"Tissue grid: {args.nx} x {args.ny} x {args.nz} = {total_grid_points} points")
    if bool(args.disable_enclosed_check):
        points = _inside_grid_points_chunked(domain, boundary, args)
    else:
        _log("Using PyVista enclosed-point check; this is CPU-heavy for large grids.")
        grid = _grid_points(boundary, int(args.nx), int(args.ny), int(args.nz))
        inside = _inside_mask(domain, boundary, grid, args)
        points = grid[inside]
        del grid, inside
    _log(f"Inside-domain tissue points: {points.shape[0]}")
    gc.collect()

    tissue_cache = None
    if bool(args.include_tissue_nearest_fields):
        max_nearby = min(int(getattr(ts, "NEAREST_TISSUE_VESSELS", 250)), int(combo["starts"].shape[0]))
        _log(f"Building tissue geometry cache for nearest fields (max_nearby={max_nearby})...")
        tissue_cache = ts._prepare_tissue_geometry(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
            max_nearby=max_nearby,
        )

    _log("Computing tissue oxygen field...")
    keep_mask, tissue_conc = ts.compute_tissue_samples_greens(
        points,
        combo["starts"],
        combo["ends"],
        combo["radii"],
        combo["cin"],
        combo["flows"],
        diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
        window_factor=float(ts.WINDOW_FACTOR),
        inlet_concentration=float(ts.get_concentration_inlet(args.fluid)),
        tissue_cache=tissue_cache,
    )
    conc = np.asarray(tissue_conc, dtype=float_dtype)
    keep = np.asarray(keep_mask, dtype=bool)
    conc_masked = conc.copy()
    conc_masked[~keep] = np.nan
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    if np.isfinite(conc_max) and conc_max > 0.0:
        conc_norm = (conc_masked / conc_max).astype(float_dtype, copy=False)
        viability = (conc >= 0.01 * conc_max).astype(np.int8)
    else:
        conc_norm = np.full(conc_masked.shape, np.nan, dtype=float_dtype)
        viability = np.zeros(conc_masked.shape, dtype=np.int8)

    metrics: dict[str, np.ndarray] = {
        "inside_tissue": keep.astype(np.uint8),
        "local_concentration": conc_masked,
        "local_concentration_raw": conc,
        "local_concentration_norm": conc_norm,
        "viability": viability,
    }

    if tissue_cache is not None:
        nearest_idx = np.asarray(tissue_cache["nearest_idx"])
        d_center = np.asarray(tissue_cache["d_center"], dtype=np.float64)
        valid = np.asarray(tissue_cache["valid_mask"], dtype=bool)
        valid_global = np.flatnonzero(valid).astype(np.int64)
        if nearest_idx.size and d_center.size:
            local = np.argmin(d_center, axis=1)
            row = np.arange(nearest_idx.shape[0])
            seg_valid = nearest_idx[row, local].astype(np.int64)
            seg_valid = np.clip(seg_valid, 0, max(valid_global.size - 1, 0))
            closest_global = valid_global[seg_valid]
            metrics["closest_segment_id"] = _safe_index_array(closest_global, index_dtype)
            metrics["closest_tree_id"] = np.asarray(combo["tree_id"], dtype=np.int16)[closest_global]
            metrics["closest_flow_ul_min"] = (
                np.asarray(combo["flows"], dtype=np.float64)[closest_global] * 60000.0
            ).astype(float_dtype)
        _log("Computing DNC field...")
        metrics["dnc_cm"] = ts.compute_distance_to_nearest_channel(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
        ).astype(float_dtype)

    export_keep = keep & np.all(np.isfinite(points), axis=1)
    dropped = int(points.shape[0] - np.count_nonzero(export_keep))
    if dropped:
        _log(f"Dropping {dropped} tissue export points with invalid samples or non-finite coordinates.")
    export_points, export_metrics = _filter_export_points(points, metrics, export_keep)

    points_poly = _build_points_polydata(export_points, export_metrics, float_dtype=float_dtype, index_dtype=index_dtype)
    meta = {
        "n_grid_points_inside": int(points.shape[0]),
        "n_tissue_points": int(np.sum(keep)),
        "n_tissue_points_exported": int(export_points.shape[0]),
        "conc_max_for_normalization": conc_max,
        "tissue_timings": dict(getattr(ts, "_LAST_TISSUE_TIMINGS", {}) or {}),
    }
    return points_poly, meta


def _save_domain_outputs(domain, out_dir: Path, prefix: str) -> dict:
    outputs = {}
    boundary = getattr(domain, "boundary", None)
    if boundary is None and getattr(domain, "mesh", None) is not None:
        boundary = domain.mesh.extract_surface()
    if boundary is not None:
        path = out_dir / f"{prefix}_domain_boundary.vtp"
        boundary.save(str(path))
        outputs["domain_boundary"] = str(path)
    mesh = getattr(domain, "mesh", None)
    if mesh is not None:
        path = out_dir / f"{prefix}_domain_mesh.vtu"
        mesh.save(str(path))
        outputs["domain_mesh"] = str(path)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a two-tree heart .forest plus tissue oxygen grid for ParaView/ParaFlow.")
    parser.add_argument("--forest", default=str(DEFAULT_FOREST), help="Input .forest path.")
    parser.add_argument("--no-simulation-cache", action="store_true", help="Disable automatic simulation-cache read/write.")
    parser.add_argument("--domain", dest="domain_path", default=DEFAULT_DOMAIN, help="Heart domain path (.stl or .dmn).")
    parser.add_argument("--domain-stl", dest="domain_path", help="Deprecated alias for --domain.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--prefix", default=None, help="Output filename prefix. Defaults to forest stem.")
    parser.add_argument("--side-length", type=float, default=1.0)
    parser.add_argument("--fluid", default="blood", choices=("blood", "water", "media", "cell media"))
    parser.add_argument("--concentration-solver", default="topdown", choices=("topdown", "network"))
    parser.add_argument(
        "--kirchhoff-bc-mode",
        default=None,
        choices=("legacy_equal_terminal_flow", "terminal_pressure"),
        help=(
            "Flow boundary condition mode forwarded to TissueSim_heart_accel. "
            "legacy_equal_terminal_flow prescribes total inlet flow and equal terminal sinks; "
            "terminal_pressure prescribes inlet flow with terminal pressures."
        ),
    )
    parser.add_argument(
        "--flow-source",
        default="tree-root-flow",
        choices=("tree-root-flow", "total-qin-split"),
        help=(
            "'tree-root-flow' uses the root_flow saved in each forest tree. "
            "'total-qin-split' overrides those magnitudes by splitting --total-qin-ul-min "
            "across trees in proportion to their saved root_flow."
        ),
    )
    parser.add_argument(
        "--total-qin-ul-min",
        type=float,
        default=None,
        help="Total inlet flow in uL/min, only used with --flow-source total-qin-split.",
    )
    parser.add_argument("--tissue-accel", default=TISSUE_ACCEL_MODE_DEFAULT, choices=("cpu", "gpu", "auto"))
    parser.add_argument("--tissue-gpu-chunk-points", type=int, default=TISSUE_GPU_CHUNK_POINTS_DEFAULT)
    parser.add_argument("--tissue-gpu-validate-points", type=int, default=TISSUE_GPU_VALIDATE_POINTS_DEFAULT)
    parser.add_argument("--solute-diffusivity", type=float, default=SOLUTE_DIFFUSIVITY_DEFAULT)
    parser.add_argument("--vmax-mm", type=float, default=VMAX_MM_DEFAULT)
    parser.add_argument("--km-mm", type=float, default=K_M_MM_DEFAULT)
    parser.add_argument("--nearest-tissue-vessels", type=int, default=NEAREST_TISSUE_VESSELS_DEFAULT)
    parser.add_argument("--window-factor", type=float, default=WINDOW_FACTOR_DEFAULT)
    parser.add_argument("--gl-order", type=int, default=GL_ORDER_DEFAULT, choices=(5, 9, 20))
    parser.add_argument("--tissue-kdtree-candidate-mult", type=int, default=TISSUE_KDTREE_CANDIDATE_MULT_DEFAULT)
    parser.add_argument("--axial-blood-steps", type=int, default=AXIAL_BLOOD_STEPS_DEFAULT)
    parser.add_argument("--conc-max-for-normalization", type=float, default=CONC_MAX_FOR_NORMALIZATION_DEFAULT)
    parser.add_argument(
        "--hematocrit-model",
        default=HEMATOCRIT_MODEL_DEFAULT,
        choices=("uniform_tube", "pries_secomb"),
    )
    parser.add_argument("--hematocrit-flow-iterations", type=int, default=HEMATOCRIT_FLOW_ITERATIONS_DEFAULT)
    parser.add_argument("--hematocrit-relaxation", type=float, default=HEMATOCRIT_RELAXATION_DEFAULT)
    parser.add_argument("--hematocrit-qtol-nl-min", type=float, default=HEMATOCRIT_QTOL_NL_MIN_DEFAULT)
    parser.add_argument("--hematocrit-hdtol", type=float, default=HEMATOCRIT_HDTOL_DEFAULT)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument("--boundary-resolution", type=int, default=28)
    parser.add_argument("--implicit-margin", type=float, default=0.0)
    parser.add_argument("--disable-enclosed-check", action="store_true", default=True)
    parser.add_argument("--enable-enclosed-check", dest="disable_enclosed_check", action="store_false")
    parser.add_argument("--tissue-grid-chunk-points", type=int, default=250_000)
    parser.add_argument("--enclosed-tolerance", type=float, default=1e-6)
    parser.add_argument("--inside-combine-mode", default="and", choices=("and", "or"))
    parser.add_argument("--vessel-resolution", type=int, default=2)
    parser.add_argument(
        "--fraction-blocked-infarction",
        type=float,
        default=0.0,
        help=(
            "Solve-time infarction severity for the selected global vessel segment. "
            "0.0 = none, 0.5 = 50%% radius reduction, 1.0 = full occlusion "
            "(segment + downstream subtree forced to zero flow/concentration outputs)."
        ),
    )
    parser.add_argument(
        "--infarction-global-segment-id",
        type=int,
        default=17,
        help="Exporter global_segment_id to target for solve-time infarction.",
    )
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--skip-tissue-oxygen", action="store_true")
    parser.add_argument("--include-tissue-nearest-fields", action="store_true")
    parser.add_argument("--validate-connectivity", action="store_true", default=False)
    parser.add_argument("--no-validate-connectivity", dest="validate_connectivity", action="store_false")
    parser.add_argument("--no-repair-connectivity", action="store_true", help="Do not repair stale parent columns before validation/solve.")
    parser.add_argument("--connectivity-geometry-atol", type=float, default=1e-6)
    parser.add_argument("--no-fail-connectivity", action="store_true")
    parser.add_argument(
        "--export-float-dtype",
        default=EXPORT_FLOAT_DTYPE_DEFAULT,
        choices=("float32", "float64"),
        help="Floating-point dtype for user-facing exported VTP arrays.",
    )
    parser.add_argument(
        "--export-index-dtype",
        default=EXPORT_INDEX_DTYPE_DEFAULT,
        choices=("int32", "int64"),
        help="Integer dtype for user-facing exported ID arrays. Default keeps vessel IDs as int64.",
    )
    args = parser.parse_args()
    export_float_dtype = _float_dtype_from_name(args.export_float_dtype)
    export_index_dtype = _int_dtype_from_name(args.export_index_dtype)
    pct_blocked = float(args.fraction_blocked_infarction)
    if not np.isfinite(pct_blocked) or pct_blocked < 0.0 or pct_blocked > 1.0:
        raise ValueError("--fraction-blocked-infarction must be in [0, 1].")
    if pct_blocked > 0.0 and args.infarction_global_segment_id is None:
        raise ValueError("--infarction-global-segment-id is required when --fraction-blocked-infarction > 0.")

    t_start = perf_counter()
    forest_path = _resolve_forest_path(args.forest)
    out_dir = _normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or forest_path.stem

    _log(f"Loading TissueSim_heart_accel...")
    ts = _load_tissuesim()
    solver_parameters = _apply_tissuesim_overrides(ts, args)
    _log(
        "Oxygen parameters: "
        f"D={ts.SOLUTE_DIFFUSIVITY:g} vmax={ts.VMAX_MM:g} km={ts.K_M_MM:g} "
        f"nearby={ts.NEAREST_TISSUE_VESSELS} window={ts.WINDOW_FACTOR:g} gl={ts.GL_ORDER} "
        f"hematocrit={ts.HEMATOCRIT_MODEL}/{ts.HEMATOCRIT_FLOW_ITERATIONS}"
    )
    _log(f"Tissue accel: mode={ts.TISSUE_ACCEL_MODE} gpu_chunk_points={ts.TISSUE_GPU_CHUNK_POINTS}")
    if args.kirchhoff_bc_mode is not None:
        _log(f"Kirchhoff BC mode: {ts.KIRCHHOFF_BC_MODE}")

    _log(f"Loading forest: {forest_path}")
    cache_path = _default_simulation_cache_path(forest_path)
    cache_enabled = not bool(args.no_simulation_cache)
    load_source = forest_path
    if cache_enabled and _should_use_simulation_cache(forest_path, cache_path):
        load_source = cache_path
        _log(f"Using simulation cache: {cache_path}")
    elif cache_enabled:
        try:
            _log(f"Building simulation cache without vessel maps: {cache_path}")
            Forest.build_simulation_cache_from_legacy(str(forest_path), str(cache_path), show_progress=True)
            load_source = cache_path
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            try:
                cache_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(
                f"Fast simulation-cache build failed ({exc}). "
                "Aborting instead of falling back to the legacy forest loader."
            ) from exc
    forest = Forest.load(str(load_source), mode="simulation")
    _make_forest_analysis_only(forest)
    _log("Forest load mode: analysis-only (skipping build-time spatial indices)")
    if cache_enabled and load_source == forest_path:
        try:
            forest.save_simulation_cache(str(cache_path))
            _log(f"Wrote simulation cache: {cache_path}")
        except Exception as exc:
            _log(f"Warning: failed to write simulation cache ({exc}).")
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest loaded: n_networks={forest.n_networks} n_trees_per_network={forest.n_trees_per_network}")

    _log(f"Building/loading domain: {args.domain_path}")
    t0 = perf_counter()
    domain = _build_domain(ts, _normalize_path(args.domain_path), float(args.side_length))
    _log(f"Domain loaded in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    _attach_domain(forest, domain)
    _log(f"Domain attached in {perf_counter() - t0:.2f}s")
    t0 = perf_counter()
    input_dtype_summaries = _normalize_forest_dtype_attrs(forest)
    _log(f"Forest dtype attrs normalized in {perf_counter() - t0:.2f}s")

    parent_repairs = []
    if not bool(args.no_repair_connectivity):
        t0 = perf_counter()
        _log("Repairing connectivity parent columns...")
        parent_repairs = _repair_forest_connectivity(forest)
        _log(f"Connectivity repair completed in {perf_counter() - t0:.2f}s")
        if any(parent_repairs):
            _log(f"Connectivity repair: stale parent refs per tree={parent_repairs}")

    connectivity_reports = []
    if args.validate_connectivity:
        t0 = perf_counter()
        _log("Validating connectivity...")
        connectivity_reports = _validate_forest_connectivity(
            forest,
            fail=not bool(args.no_fail_connectivity),
            geometry_atol=float(args.connectivity_geometry_atol),
        )
        _log(f"Connectivity validation completed in {perf_counter() - t0:.2f}s")

    outputs: dict[str, str] = {}
    meta: dict = {
        "forest": str(forest_path),
        "domain_path": str(args.domain_path),
        "fluid": str(args.fluid),
        "flow_source": str(args.flow_source),
        "connectivity": connectivity_reports,
        "parent_connectivity_repairs": [int(v) for v in parent_repairs],
        "solver_parameters": solver_parameters,
        "input_tree_dtypes": input_dtype_summaries,
        "export_float_dtype": str(export_float_dtype),
        "export_index_dtype": str(export_index_dtype),
        "tree_summaries": [],
    }
    infarction_meta = {
        "target_global_segment_id": None if args.infarction_global_segment_id is None else int(args.infarction_global_segment_id),
        "target_tree_id": None,
        "target_local_segment_id": None,
        "fraction_blocked_infarction": pct_blocked,
        "applied": False,
        "mode": "none",
        "geometry_only_ignored": bool(args.geometry_only and pct_blocked > 0.0),
        "target_segment_original_radius": None,
        "target_segment_effective_radius": None,
        "blocked_subtree_count": 0,
        "blocked_subtree_preview_global": [],
        "blocked_subtree_preview_local": [],
    }
    if args.geometry_only and pct_blocked > 0.0:
        _log("Infarction requested, but --geometry-only is enabled; ignoring infarction for this run.")
    elif pct_blocked > 0.0:
        target_tree_id, target_local_segment_id = _resolve_global_segment_id(forest, int(args.infarction_global_segment_id))
        infarction_meta["target_tree_id"] = int(target_tree_id)
        infarction_meta["target_local_segment_id"] = int(target_local_segment_id)
        _log(
            f"Infarction target: global_segment_id={args.infarction_global_segment_id} "
            f"-> tree_id={target_tree_id} local_segment_id={target_local_segment_id}"
        )

    tree_solutions: list[dict] = []
    if args.geometry_only:
        _log("Geometry-only mode: skipping flow and concentration solves.")
        for tree_id, tree in enumerate(forest.networks[0]):
            data = np.asarray(tree.data[: int(tree.segment_count)])
            sol = {
                "starts": data[:, 0:3],
                "ends": data[:, 3:6],
                "radii": data[:, 21],
                "lengths": data[:, 20],
                "flows": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cin": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "cout": np.full(data.shape[0], np.nan, dtype=export_float_dtype),
                "p_in": float("nan"),
                "p_out": float("nan"),
                "inlet_concentration": float("nan"),
            }
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": tree_id,
                    "segments": int(data.shape[0]),
                    "geometry_only": True,
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )
    else:
        t0 = perf_counter()
        inlet_flows = _flow_inputs(args, forest)
        _log(f"Flow inputs prepared in {perf_counter() - t0:.2f}s")
        tree_offsets: list[int] = []
        offset = 0
        for tree in forest.networks[0]:
            tree_offsets.append(offset)
            offset += int(getattr(tree, "segment_count", 0) or 0)
        for tree_id, (tree, inlet_flow) in enumerate(zip(forest.networks[0], inlet_flows)):
            t_tree = perf_counter()
            _log(f"Solving tree {tree_id}...")
            blocked_subtree_ids = np.empty((0,), dtype=np.int64)
            restore_radii_ids = np.empty((0,), dtype=np.int64)
            restore_radii_values = np.empty((0,), dtype=float)
            if pct_blocked > 0.0 and int(infarction_meta["target_tree_id"]) == int(tree_id):
                target_id = int(infarction_meta["target_local_segment_id"])
                target_radius = float(tree.data[target_id, 21])
                infarction_meta["target_segment_original_radius"] = target_radius
                if pct_blocked < 1.0:
                    effective_radius = target_radius * (1.0 - pct_blocked)
                    infarction_meta["target_segment_effective_radius"] = float(effective_radius)
                    infarction_meta["mode"] = "radius_reduction"
                    infarction_meta["applied"] = True
                    restore_radii_ids = np.array([target_id], dtype=np.int64)
                    restore_radii_values = np.array([target_radius], dtype=float)
                    tree.data[target_id, 21] = effective_radius
                    _log(
                        f"Infarction solve override: tree {tree_id} segment {target_id} radius "
                        f"{target_radius:.9g} -> {effective_radius:.9g}"
                    )
                else:
                    seg_count = int(getattr(tree, "segment_count", 0) or 0)
                    blocked_subtree_ids = _collect_downstream_segment_ids(tree, target_id, seg_count)
                    blocked_count = int(blocked_subtree_ids.size)
                    infarction_meta["mode"] = "full_occlusion_subtree"
                    infarction_meta["applied"] = True
                    infarction_meta["blocked_subtree_count"] = blocked_count
                    infarction_meta["blocked_subtree_preview_local"] = blocked_subtree_ids[:20].tolist()
                    infarction_meta["blocked_subtree_preview_global"] = (blocked_subtree_ids[:20] + int(tree_offsets[tree_id])).tolist()
                    infarction_meta["target_segment_effective_radius"] = 0.0
                    if blocked_count > 0:
                        restore_radii_ids = blocked_subtree_ids
                        restore_radii_values = np.asarray(tree.data[restore_radii_ids, 21], dtype=float)
                        tree.data[restore_radii_ids, 21] = 0.0
                    _log(
                        f"Infarction full occlusion: tree {tree_id} segment {target_id} "
                        f"subtree size {blocked_count} set to zero effective radius for solves."
                    )
            try:
                sol = _solve_tree(ts, tree, inlet_flow, fluid=str(args.fluid), concentration_solver=args.concentration_solver)
            finally:
                if restore_radii_ids.size > 0:
                    tree.data[restore_radii_ids, 21] = restore_radii_values
            if pct_blocked >= 1.0 and blocked_subtree_ids.size > 0:
                valid_blocked = blocked_subtree_ids[blocked_subtree_ids < sol["flows"].shape[0]]
                if valid_blocked.size > 0:
                    sol["flows"][valid_blocked] = 0.0
                    sol["cin"][valid_blocked] = 0.0
                    sol["cout"][valid_blocked] = 0.0
                    _log(
                        f"Infarction post-process: forced zero flow/cin/cout on {valid_blocked.size} "
                        f"blocked subtree segments in tree {tree_id}."
                    )
            _log(f"Tree {tree_id} solved in {perf_counter() - t_tree:.2f}s")
            tree_solutions.append(sol)
            meta["tree_summaries"].append(
                {
                    "tree_id": int(tree_id),
                    "segments": int(sol["starts"].shape[0]),
                    "terminals": int(getattr(tree, "n_terminals", 0)),
                    "inlet_flow_cm3_s": float(inlet_flow),
                    "inlet_flow_ul_min": float(inlet_flow * 60000.0),
                    "pressure_in": float(sol["p_in"]),
                    "pressure_out": float(sol["p_out"]),
                    "data_dtype": str(np.asarray(getattr(tree, "data", [])).dtype),
                    "connectivity_dtype": str(np.asarray(getattr(tree, "connectivity", [])).dtype),
                }
            )

    t0 = perf_counter()
    combo = _concat_tree_solutions(tree_solutions, index_dtype=export_index_dtype)
    _log(f"Combined tree solution arrays in {perf_counter() - t0:.2f}s")
    _log(f"Combined vessel segments: {combo['starts'].shape[0]}")

    t0 = perf_counter()
    vessels = _build_vessel_polydata(
        combo,
        resolution=int(args.vessel_resolution),
        float_dtype=export_float_dtype,
        index_dtype=export_index_dtype,
    )
    _log(f"Built vessel VTP in {perf_counter() - t0:.2f}s")
    vessels_path = out_dir / f"{prefix}_forest_vessels.vtp"
    _log(f"Saving vessel VTP: {vessels_path}")
    t0 = perf_counter()
    vessels.save(str(vessels_path))
    _log(f"Saved vessel VTP in {perf_counter() - t0:.2f}s")
    outputs["vessels"] = str(vessels_path)
    del vessels
    gc.collect()

    if not args.geometry_only and not args.skip_tissue_oxygen:
        t0 = perf_counter()
        points_poly, tissue_meta = _compute_tissue(
            ts,
            combo,
            domain,
            args,
            float_dtype=export_float_dtype,
            index_dtype=export_index_dtype,
        )
        _log(f"Computed tissue oxygen in {perf_counter() - t0:.2f}s")
        points_path = out_dir / f"{prefix}_forest_oxygen_points.vtp"
        _log(f"Saving tissue oxygen VTP: {points_path}")
        t0 = perf_counter()
        points_poly.save(str(points_path))
        _log(f"Saved tissue oxygen VTP in {perf_counter() - t0:.2f}s")
        outputs["oxygen_points"] = str(points_path)
        meta["tissue"] = tissue_meta
        del points_poly
        gc.collect()
    else:
        _log("Skipping tissue oxygen output.")

    outputs.update(_save_domain_outputs(domain, out_dir, prefix))
    meta["infarction"] = infarction_meta
    meta["outputs"] = outputs
    meta["elapsed_s"] = perf_counter() - t_start
    meta["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    meta_path = out_dir / f"{prefix}_forest_paraview_export.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"Metadata: {meta_path}")
    _log(f"Done in {meta['elapsed_s']:.2f}s")
    print("Wrote:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")
    print(f"  meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
