"""Per-tree flow and concentration preparation."""

from __future__ import annotations

from .heart_support import (
    Forest,
    np,
    perf_counter,
)

from .heart_domain import (
    _log,
)

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


def _restore_solution_radii(
    solution: dict,
    segment_ids: np.ndarray,
    original_radii: np.ndarray,
) -> None:
    radii = np.asarray(solution["radii"])
    ids = np.asarray(segment_ids, dtype=np.int64)
    values = np.asarray(original_radii)
    valid = (ids >= 0) & (ids < radii.shape[0])
    if np.any(valid):
        radii[ids[valid]] = values[valid]


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


def _cext_chb_max_for_tree(cext_ts, tree, nseg: int, flows: np.ndarray, *, fluid: str) -> np.ndarray:
    if str(fluid).lower() != "blood":
        return np.zeros((int(nseg),), dtype=np.float32)
    cached = cext_ts._get_tree_hematocrit_cache(
        tree,
        int(nseg),
        model=str(cext_ts.HEMATOCRIT_MODEL),
        flows=flows,
    )
    if cached is None:
        hct_context = cext_ts._hematocrit_context_for_tree(tree)
        HD, HT = cext_ts.compute_tree_hematocrit(
            tree,
            hd_root=float(cext_ts.HD_DISCHARGE),
            flows=flows,
            model=str(cext_ts.HEMATOCRIT_MODEL),
            order=np.asarray(hct_context["order"], dtype=np.int64),
        )
        cext_ts._store_tree_hematocrit_cache(
            tree,
            HD,
            HT,
            model=str(cext_ts.HEMATOCRIT_MODEL),
            flows=flows,
        )
    else:
        _, HT = cached
    return np.asarray(HT, dtype=np.float32) * np.float32(float(cext_ts.O2_CAP_PER_HCT))


def _run_cext_frozen_step(cext_ts, sol: dict, *, fluid: str) -> None:
    ext_state = sol["cext_state"]
    context = sol["cext_context"]
    frozen_backend = "gpu"
    if hasattr(cext_ts, "_resolve_cext_frozen_accel_mode"):
        frozen_backend = str(cext_ts._resolve_cext_frozen_accel_mode())
    cin, cout, c_iv, backend, transfer = cext_ts._run_topdown_ext_frozen_step(
        context,
        ext_state,
        inlet_concentration=float(sol["inlet_concentration"]),
        vmax=float(cext_ts.VMAX_MM),
        km=float(cext_ts.K_M_MM),
        chb_max=np.asarray(sol["chb_max"], dtype=np.float32),
        fluid_mode=str(fluid),
        frozen_backend=frozen_backend,
    )
    ext_state["cin_seg"] = np.asarray(cin, dtype=np.float32)
    ext_state["cout_seg"] = np.asarray(cout, dtype=np.float32)
    ext_state["c_iv_gl"] = np.asarray(c_iv, dtype=np.float32)
    cext_ts._build_cext_iteration_cache(context, ext_state)
    sol["cin"] = ext_state["cin_seg"]
    sol["cout"] = ext_state["cout_seg"]
    sol["cext_frozen_backend"] = str(backend)
    sol["cext_frozen_transfer_s"] = float(transfer)


def _solve_tree_cext_prepare(cext_ts, ts, tree, inlet_flow_cm3_s: float, *, fluid: str) -> dict:
    inlet_concentration = float(ts.get_concentration_inlet(fluid))
    _log(
        f"  Cext tree prep: segments={int(tree.segment_count)} terminals={int(tree.n_terminals)} "
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
    nseg = int(np.asarray(starts).shape[0])
    t0 = perf_counter()
    context = cext_ts._build_cext_geometry_context(
        tree,
        np.asarray(flows),
        np.asarray(starts),
        np.asarray(ends),
        np.asarray(radii),
        np.asarray(lengths),
        inlet_concentration=inlet_concentration,
        diffusivity=float(ts.SOLUTE_DIFFUSIVITY),
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
        build_candidate_index=False,
    )
    cache_key = (
        str(fluid),
        int(cext_ts.GL_ORDER_CEXT),
        float(inlet_concentration),
        float(ts.SOLUTE_DIFFUSIVITY),
        float(ts.VMAX_MM),
        float(ts.K_M_MM),
    )
    ext_state = cext_ts._initialize_cext_state(
        tree,
        cache_key=cache_key,
        nseg=nseg,
        inlet_concentration=inlet_concentration,
        vmax=float(ts.VMAX_MM),
        km=float(ts.K_M_MM),
    )
    chb_max = _cext_chb_max_for_tree(cext_ts, tree, nseg, np.asarray(flows), fluid=fluid)
    p_in = float("nan")
    p_out = float("nan")
    if pressures.size and inlet_nodes:
        p_in = float(pressures[inlet_nodes[0]])
    if pressures.size and outlet_nodes:
        p_out = float(np.mean(pressures[outlet_nodes]))
    sol = {
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "radii": np.asarray(radii),
        "lengths": np.asarray(lengths),
        "flows": np.asarray(flows),
        "cin": np.asarray(ext_state["cin_seg"]),
        "cout": np.asarray(ext_state["cout_seg"]),
        "pressures": np.asarray(pressures),
        "p_in": p_in,
        "p_out": p_out,
        "inlet_concentration": inlet_concentration,
        "cext_context": context,
        "cext_state": ext_state,
        "chb_max": chb_max,
    }
    _run_cext_frozen_step(cext_ts, sol, fluid=fluid)
    _log(f"    initial frozen topdown and Cext geometry completed in {perf_counter() - t0:.2f}s")
    return sol




__all__ = ('_tree_root_flow_cm3_s', '_flow_inputs', '_safe_index_array', '_collect_downstream_segment_ids', '_restore_solution_radii', '_resolve_global_segment_id', '_solve_tree', '_cext_chb_max_for_tree', '_run_cext_frozen_step', '_solve_tree_cext_prepare')
