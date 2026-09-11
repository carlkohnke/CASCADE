"""Tissue evaluation and heart output construction."""

from __future__ import annotations

from .heart_cext import (
    _compact_cext_state_for_concat,
)

from .heart_support import (
    Path,
    _normalize_path,
    gc,
    hashlib,
    np,
    pv,
    tqdm,
)

from .heart_flow import (
    _safe_index_array,
)

from .heart_domain import (
    _log,
)

def _concat_tree_solutions(solutions: list[dict], *, index_dtype: np.dtype) -> dict:
    out = {}
    for key in ("starts", "ends", "radii", "lengths", "flows", "cin", "cout"):
        out[key] = np.concatenate([s[key] for s in solutions], axis=0) if solutions else np.empty((0,))
    if solutions and all(isinstance(s.get("cext_state"), dict) for s in solutions):
        compact = [_compact_cext_state_for_concat(s) for s in solutions]
        out["cext_state"] = {
            "solver": str(compact[0]["solver"]),
            "backend": str(compact[0]["backend"]),
            "gl_points_si": np.concatenate([s["gl_points_si"] for s in compact], axis=0),
            "diffusivity_si": float(compact[0]["diffusivity_si"]),
            "window_factor": float(compact[0]["window_factor"]),
            "c_iv_gl": np.concatenate([s["c_iv_gl"] for s in compact], axis=0),
            "c_bulk_gl": np.concatenate([s["c_bulk_gl"] for s in compact], axis=0),
            "c_wall_gl": np.concatenate([s["c_wall_gl"] for s in compact], axis=0),
            "c_ext_gl": np.concatenate([s["c_ext_gl"] for s in compact], axis=0),
            "lambda_iv_gl": np.concatenate([s["lambda_iv_gl"] for s in compact], axis=0),
            "k_if_gl": np.concatenate([s["k_if_gl"] for s in compact], axis=0),
            "q_line_gl": np.concatenate([s["q_line_gl"] for s in compact], axis=0),
            "q_weighted_gl": np.concatenate([s["q_weighted_gl"] for s in compact], axis=0),
            "mono2_weight_gl": np.concatenate([s["mono2_weight_gl"] for s in compact], axis=0),
            "dipole2_weight_gl": np.concatenate([s["dipole2_weight_gl"] for s in compact], axis=0),
            "seg_cap_gl": np.concatenate([s["seg_cap_gl"] for s in compact], axis=0),
            "segment_vectors": np.concatenate([s["segment_vectors"] for s in compact], axis=0),
            "export_slim_state": True,
        }
        out["cext_mean"] = np.concatenate(
            [np.mean(np.asarray(s["c_ext_gl"], dtype=np.float32), axis=1) for s in compact],
            axis=0,
        )
        out["c_iv_minus_cext_mean"] = np.concatenate(
            [
                np.mean(np.asarray(s["c_iv_gl"], dtype=np.float32) - np.asarray(s["c_ext_gl"], dtype=np.float32), axis=1)
                for s in compact
            ],
            axis=0,
        )
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
    if "cext_mean" in combo:
        cext_mean = np.asarray(combo["cext_mean"], dtype=float_dtype)
        poly.point_data["cext_mean"] = np.repeat(cext_mean, res)
    if "c_iv_minus_cext_mean" in combo:
        delta_mean = np.asarray(combo["c_iv_minus_cext_mean"], dtype=float_dtype)
        poly.point_data["c_iv_minus_cext_mean"] = np.repeat(delta_mean, res)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tissue_points(path_value: str | Path) -> tuple[np.ndarray, dict]:
    path = _normalize_path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tissue point file does not exist: {path}")
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Tissue point array must have shape (N, 3); got {points.shape} from {path}"
        )
    if points.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"Tissue point array must be numeric; got dtype {points.dtype} from {path}")
    points = np.asarray(points, dtype=np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"Tissue point array contains NaN or infinite coordinates: {path}")
    return points, {
        "mode": "explicit_npy",
        "path": str(path),
        "sha256": _sha256_file(path),
        "count": int(points.shape[0]),
        "coordinate_units": "cm",
    }


def _compute_tissue(
    ts,
    cext_ts,
    combo: dict,
    domain,
    args,
    *,
    float_dtype: np.dtype,
    index_dtype: np.dtype,
) -> tuple[pv.PolyData, dict]:
    if args.tissue_points is not None:
        points, point_source = _load_tissue_points(args.tissue_points)
        total_grid_points = int(points.shape[0])
        _log(
            f"Explicit tissue points: {total_grid_points} from {point_source['path']} "
            f"(sha256={point_source['sha256']})"
        )
    else:
        boundary = _get_boundary(domain, int(args.boundary_resolution))
        total_grid_points = int(args.nx) * int(args.ny) * int(args.nz)
        point_source = {
            "mode": "generated_grid",
            "shape": [int(args.nx), int(args.ny), int(args.nz)],
            "candidate_count": total_grid_points,
        }
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

    if "cext_state" in combo:
        if cext_ts is None:
            raise RuntimeError("Combined solution contains Cext state, but the Cext module was not loaded.")
        _log("Computing tissue oxygen field from Cext source state...")
        keep_mask, tissue_conc = cext_ts.compute_tissue_samples_greens_from_cext_state(
            points,
            combo["starts"],
            combo["ends"],
            combo["radii"],
            combo["cext_state"],
            tissue_cache=tissue_cache,
        )
        tissue_timings = dict(getattr(cext_ts, "_LAST_TISSUE_TIMINGS", {}) or {})
    else:
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
        tissue_timings = dict(getattr(ts, "_LAST_TISSUE_TIMINGS", {}) or {})
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
        "source_mode": "cext_state" if "cext_state" in combo else "legacy_vessel_cin_flow",
        "point_source": point_source,
        "tissue_timings": tissue_timings,
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




__all__ = ('_concat_tree_solutions', '_build_vessel_polydata', '_build_points_polydata', '_filter_export_points', '_get_boundary', '_grid_axes', '_grid_points_from_axes', '_grid_points', '_inside_mask', '_inside_grid_points_chunked', '_sha256_file', '_load_tissue_points', '_compute_tissue', '_save_domain_outputs')
