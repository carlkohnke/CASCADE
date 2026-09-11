"""Repair and validate parent/child connectivity in vascular trees."""

from __future__ import annotations

from typing import Any

import numpy as np


def repair_tree_parent_columns(tree: Any) -> int:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if seg_count <= 0:
        return 0
    data = np.asarray(tree.data[:seg_count])
    index_dtype = getattr(tree, "index_dtype", np.int64)
    exact_connectivity = getattr(tree, "connectivity", None)
    if (
        exact_connectivity is not None
        and np.asarray(exact_connectivity).shape == (seg_count, 3)
    ):
        conn = np.asarray(exact_connectivity, dtype=index_dtype).copy()
    else:
        conn = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)

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
            raise RuntimeError(
                f"Connectivity child index out of range: row={row} child={max_child}"
            )
        np.add.at(child_counts, valid_children, 1)
        parent[valid_children] = rows[valid]
    duplicate_children = np.flatnonzero(child_counts > 1)
    if duplicate_children.size:
        raise RuntimeError(
            f"Connectivity has duplicate child references: {duplicate_children[:5].tolist()}"
        )

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


def repair_trees(trees: list[Any]) -> list[int]:
    return [repair_tree_parent_columns(tree) for tree in trees]


def connectivity_report(tree: Any, *, geometry_atol: float = 1.0e-6) -> dict[str, Any]:
    n = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:n])
    if n == 0:
        return {
            "segments": 0,
            "roots": [],
            "reachable": 0,
            "bad_parent": 0,
            "bad_child": 0,
            "bad_geom": 0,
        }
    exact_connectivity = getattr(tree, "connectivity", None)
    if (
        exact_connectivity is not None
        and np.asarray(exact_connectivity).shape == (n, 3)
    ):
        conn = np.asarray(exact_connectivity, dtype=np.int64)
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
            np.abs(data[parent_rows, 3:6] - data[child_ids, 0:3])
            <= float(geometry_atol),
            axis=1,
        )
        bad_geom += int(np.count_nonzero(child_links_back & ~geom_ok))

    reachable = n if len(roots) == 1 and bad_parent == 0 and bad_child == 0 else 0
    return {
        "segments": int(n),
        "roots": roots.tolist(),
        "reachable": int(reachable),
        "bad_parent": int(bad_parent),
        "bad_child": int(bad_child),
        "bad_geom": int(bad_geom),
    }


def validate_trees(
    trees: list[Any], *, fail: bool, geometry_atol: float
) -> list[dict[str, Any]]:
    reports = []
    for tree_id, tree in enumerate(trees):
        report = connectivity_report(tree, geometry_atol=float(geometry_atol))
        report["tree_id"] = int(tree_id)
        reports.append(report)
        bad = (
            len(report["roots"]) != 1
            or int(report["reachable"]) != int(report["segments"])
            or int(report["bad_parent"]) > 0
            or int(report["bad_child"]) > 0
            or int(report["bad_geom"]) > 0
        )
        if fail and bad:
            raise RuntimeError(
                f"Connectivity validation failed for tree {tree_id}: {report}"
            )
    return reports


def collect_downstream_segment_ids(tree: Any, segment_id: int) -> np.ndarray:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
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
