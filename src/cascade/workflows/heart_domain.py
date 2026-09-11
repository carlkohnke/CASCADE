"""Heart-domain loading, attachment, and connectivity validation."""

from __future__ import annotations

from .heart_support import (
    Domain,
    Forest,
    Path,
    hashlib,
    importlib,
    np,
    os,
    pv,
)

def _log(message: str) -> None:
    print(message, flush=True)


def _domain_cache_dir() -> Path:
    override = os.environ.get("CASCADE_DOMAIN_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (root / "cascade" / "domains").resolve()


def _domain_cache_path(domain_path: Path, cache_dir: Path) -> Path:
    digest = hashlib.sha256()
    with domain_path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    try:
        svv_version = importlib.metadata.version("svv")
    except importlib.metadata.PackageNotFoundError:
        svv_version = "unknown"
    safe_version = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in svv_version)
    return cache_dir / f"surface-v1-{digest.hexdigest()}-svv-{safe_version}.dmn"


def _build_domain(
    ts,
    domain_path: Path,
    side_length: float,
    *,
    cache_dir: Path | None = None,
):
    suffix = str(domain_path.suffix).strip().lower()
    if suffix == ".dmn":
        return Domain.load(str(domain_path))
    cache_path = None if cache_dir is None else _domain_cache_path(domain_path, cache_dir)
    if cache_path is not None and cache_path.is_file():
        try:
            domain = Domain.load(str(cache_path))
            _log(f"Using CASCADE domain cache: {cache_path}")
            return domain
        except Exception as exc:
            _log(f"Warning: could not load CASCADE domain cache {cache_path} ({exc}); rebuilding.")
    if hasattr(ts, "build_domain_from_pyvista"):
        mesh = pv.read(str(domain_path))
        if not isinstance(mesh, pv.PolyData):
            mesh = mesh.extract_surface()
        domain = ts.build_domain_from_pyvista(mesh)
    else:
        if hasattr(ts, "DEFAULT_STL"):
            ts.DEFAULT_STL = str(domain_path)
        if hasattr(ts, "DOMAIN_CACHE_PATH"):
            # The heart forest can use arbitrary imported root geometry; avoid accidentally
            # reusing an incompatible single-tree domain cache.
            ts.DOMAIN_CACHE_PATH = None
        domain = ts.build_domain(float(side_length))
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            domain.save(
                str(cache_path),
                include_boundary=True,
                include_mesh=True,
                include_patch_normals=False,
            )
            _log(f"Saved CASCADE domain cache: {cache_path}")
        except Exception as exc:
            _log(f"Warning: could not save CASCADE domain cache {cache_path} ({exc}).")
    return domain


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
    # Keep compact exact integer topology from a simulation cache. It is the
    # authority when float32 physical tables contain IDs above 2**24.
    if conn is not None:
        tree.connectivity = conn
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
    existing_conn = getattr(tree, "connectivity", None)
    if existing_conn is not None and np.asarray(existing_conn).shape == (seg_count, 3):
        conn = np.asarray(existing_conn, dtype=index_dtype).copy()
    else:
        conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)
    # Some older forest-export repairs wrote -1 into tree.data child slots.
    # SVV's solver code treats NaN child slots as terminal leaves, so keep -1
    # only in tree.connectivity and preserve/restore NaNs in tree.data.
    child_conn = conn[:, 0:2]
    child_conn[child_conn < 0] = -1
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
    data_dtype = np.asarray(tree.data).dtype
    can_embed_exactly = data_dtype == np.dtype(np.float64) or seg_count <= 2**24
    if can_embed_exactly:
        tree.data[:seg_count, 15:18] = data_conn.astype(data_dtype, copy=False)
        try:
            tree.preallocate[:seg_count, 15:18] = data_conn.astype(
                np.asarray(tree.preallocate).dtype, copy=False
            )
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
    exact_conn = getattr(tree, "connectivity", None)
    if exact_conn is not None and np.asarray(exact_conn).shape == (n, 3):
        conn = np.asarray(exact_conn, dtype=np.int64)
    else:
        conn = np.nan_to_num(data[:, 15:18], nan=-1).astype(np.int64)
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




__all__ = ('_log', '_domain_cache_dir', '_domain_cache_path', '_build_domain', '_attach_domain', '_normalize_forest_dtype_attrs', '_make_tree_analysis_only', '_make_forest_analysis_only', '_repair_tree_parent_columns_from_children', '_repair_forest_connectivity', '_connectivity_report', '_validate_forest_connectivity')
