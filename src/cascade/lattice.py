"""Paper lattice generators used by the GUI and simple-network backend.

The arrays returned here are graph arrays, not a vascular ``Tree``. Lattices
contain loops and high-degree nodes, so callers must use the sparse Kirchhoff
and network concentration solvers.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


LATTICE_TYPES = ("cubic", "octet", "bcc", "diamond")


def channel_count(cells: int, lattice_type: str) -> int:
    n = int(cells)
    mode = _mode(lattice_type)
    if n < 1:
        raise ValueError("lattice cells must be at least 1")
    if mode == "cubic":
        return 3 * n * (n + 1) * (n + 1)
    if mode == "octet":
        if n % 2:
            raise ValueError("octet lattices require an even cell count")
        return 3 * n * n * (n + 1)
    if mode == "bcc":
        return 8 * n**3
    return 16 * n**3 + 1


def generate_lattice(
    cells: int,
    dimensions_cm: tuple[float, float, float],
    radius_cm: float,
    *,
    lattice_type: str = "cubic",
    inlet_points_cm: list[list[float]] | None = None,
    outlet_points_cm: list[list[float]] | None = None,
    radius_expression: str | None = None,
    subdivisions: int = 1,
    center_cm: tuple[float, float, float] | None = None,
    node_inside: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Build one of the four lattice families used in the CASCADE paper.

    The lattice is generated over a bounding box centered at ``center_cm``. When
    ``node_inside`` is supplied, outside nodes and all edges incident to them are
    removed before boundary points are snapped. ``radius_expression`` is evaluated
    at segment midpoints with numexpr and may use x, y, z, r0 and L. Subdivision
    inserts graph nodes before clipping and radius evaluation.
    """
    mode = _mode(lattice_type)
    n = int(cells)
    dims = np.asarray(dimensions_cm, dtype=float)
    center = np.asarray(center_cm or (0.0, 0.0, 0.0), dtype=float)
    r0 = float(radius_cm)
    if n < 1 or np.any(~np.isfinite(dims)) or np.any(dims <= 0.0):
        raise ValueError("lattice cells and all domain dimensions must be positive")
    if r0 <= 0.0 or not np.isfinite(r0):
        raise ValueError("lattice radius must be positive and finite")
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        raise ValueError("lattice center must contain three finite coordinates")
    if mode == "octet" and n % 2:
        raise ValueError("octet lattices require an even cell count")

    nodes, keys, key_to_id = _nodes(n, dims, mode, center)
    edges = _edges(n, mode, key_to_id)
    if mode == "diamond":
        nodes, keys, edges, key_to_id = _keep_inlet_component_and_cap(
            n, nodes, keys, edges
        )
    edge_nodes = np.asarray(edges, dtype=np.int64)
    expected = channel_count(n, mode)
    if edge_nodes.shape[0] != expected:
        raise RuntimeError(
            f"{mode} lattice produced {edge_nodes.shape[0]} struts; expected {expected}"
        )

    if int(subdivisions) > 1:
        nodes, edge_nodes = _subdivide(nodes, edge_nodes, int(subdivisions))

    original_node_count = int(nodes.shape[0])
    original_edge_count = int(edge_nodes.shape[0])
    if node_inside is not None:
        nodes, edge_nodes = _clip_to_domain(nodes, edge_nodes, node_inside)
    domain_node_count = int(nodes.shape[0])
    domain_edge_count = int(edge_nodes.shape[0])

    lower_corner = center - 0.5 * dims
    upper_corner = center + 0.5 * dims
    base_node_count = int(nodes.shape[0])
    if inlet_points_cm:
        nodes, edge_nodes, inlet_nodes, inlet_anchors, inlet_connections = (
            _attach_boundary_points(
                nodes,
                edge_nodes,
                inlet_points_cm,
                candidate_count=base_node_count,
                boundary_first=True,
            )
        )
    else:
        inlet_nodes = _snap_nodes(nodes, [lower_corner.tolist()])
        inlet_anchors = list(inlet_nodes)
        inlet_connections = 0
    if outlet_points_cm:
        nodes, edge_nodes, outlet_nodes, outlet_anchors, outlet_connections = (
            _attach_boundary_points(
                nodes,
                edge_nodes,
                outlet_points_cm,
                candidate_count=base_node_count,
                boundary_first=False,
            )
        )
    else:
        outlet_nodes = _snap_nodes(nodes[:base_node_count], [upper_corner.tolist()])
        outlet_anchors = list(outlet_nodes)
        outlet_connections = 0
    if set(inlet_anchors) & set(outlet_anchors):
        raise ValueError("an inlet and outlet snapped to the same retained lattice node")
    nodes, edge_nodes, inlet_nodes, outlet_nodes = _keep_flow_components(
        nodes, edge_nodes, inlet_nodes, outlet_nodes
    )
    starts = nodes[edge_nodes[:, 0]]
    ends = nodes[edge_nodes[:, 1]]
    lengths = np.linalg.norm(ends - starts, axis=1)
    radii = _radii(starts, ends, r0, dims, radius_expression)
    return {
        "lattice_type": mode,
        "cells": n,
        "node_coords_cm": nodes,
        "edge_nodes": edge_nodes,
        "segment_starts_cm": starts,
        "segment_ends_cm": ends,
        "segment_lengths_cm": lengths,
        "segment_radii_cm": radii,
        "inlet_nodes": inlet_nodes,
        "outlet_nodes": outlet_nodes,
        "inlet_points_cm": nodes[np.asarray(inlet_nodes, dtype=np.int64)],
        "outlet_points_cm": nodes[np.asarray(outlet_nodes, dtype=np.int64)],
        "subdivisions": int(subdivisions),
        "radius_expression": radius_expression or "",
        "bounding_box_center_cm": center,
        "inlet_connection_count": int(inlet_connections),
        "outlet_connection_count": int(outlet_connections),
        "nodes_removed_by_domain": original_node_count - domain_node_count,
        "edges_removed_by_domain": original_edge_count - domain_edge_count,
    }


def _mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "simple_cubic": "cubic",
        "tetrahedral": "diamond",
        "diamond_cubic": "diamond",
    }
    mode = aliases.get(mode, mode)
    if mode not in LATTICE_TYPES:
        raise ValueError(f"lattice type must be one of {LATTICE_TYPES}")
    return mode


def _nodes(n: int, dims: np.ndarray, mode: str, center: np.ndarray):
    scale = 4 if mode == "diamond" else (2 if mode == "bcc" else 1)
    coords = [
        np.linspace(
            center[a] - 0.5 * dims[a],
            center[a] + 0.5 * dims[a],
            scale * n + 1,
        )
        for a in range(3)
    ]
    keys: list[tuple[int, int, int]] = []
    if mode in {"cubic", "octet"}:
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if mode == "octet" and (i + j + k) % 2:
                        continue
                    keys.append((i, j, k))
    elif mode == "bcc":
        keys.extend(
            (2 * i, 2 * j, 2 * k)
            for i in range(n + 1)
            for j in range(n + 1)
            for k in range(n + 1)
        )
        keys.extend(
            (2 * i + 1, 2 * j + 1, 2 * k + 1)
            for i in range(n)
            for j in range(n)
            for k in range(n)
        )
    else:
        fcc = {(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)}
        shifted = {(1, 1, 1), (1, 3, 3), (3, 1, 3), (3, 3, 1)}
        for i in range(4 * n + 1):
            for j in range(4 * n + 1):
                for k in range(4 * n + 1):
                    residue = (i % 4, j % 4, k % 4)
                    if residue in fcc or residue in shifted:
                        keys.append((i, j, k))
    lookup = {key: idx for idx, key in enumerate(keys)}
    nodes = np.asarray(
        [(coords[0][i], coords[1][j], coords[2][k]) for i, j, k in keys],
        dtype=np.float64,
    )
    return nodes, keys, lookup


def _edges(n: int, mode: str, ids: dict[tuple[int, int, int], int]):
    edges: list[tuple[int, int]] = []
    if mode == "cubic":
        edges.extend(
            (ids[(i, j, k)], ids[(i + 1, j, k)])
            for i in range(n)
            for j in range(n + 1)
            for k in range(n + 1)
        )
        edges.extend(
            (ids[(i, j, k)], ids[(i, j + 1, k)])
            for i in range(n + 1)
            for j in range(n)
            for k in range(n + 1)
        )
        edges.extend(
            (ids[(i, j, k)], ids[(i, j, k + 1)])
            for i in range(n + 1)
            for j in range(n + 1)
            for k in range(n)
        )
    elif mode == "octet":
        directions = (
            (1, 1, 0),
            (1, -1, 0),
            (1, 0, 1),
            (1, 0, -1),
            (0, 1, 1),
            (0, 1, -1),
        )
        for key, start in ids.items():
            for delta in directions:
                end = ids.get(tuple(key[a] + delta[a] for a in range(3)))
                if end is not None:
                    edges.append((start, end))
    elif mode == "bcc":
        offsets = tuple((a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    center = (2 * i + 1, 2 * j + 1, 2 * k + 1)
                    edges.extend(
                        (ids[center], ids[tuple(center[a] + d[a] for a in range(3))])
                        for d in offsets
                    )
    else:
        fcc = {(0, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)}
        offsets = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
        for key, start in ids.items():
            if tuple(v % 4 for v in key) not in fcc:
                continue
            for d in offsets:
                end = ids.get(tuple(key[a] + d[a] for a in range(3)))
                if end is not None:
                    edges.append((start, end))
    return edges


def _keep_inlet_component_and_cap(n, nodes, keys, edges):
    ids = {key: idx for idx, key in enumerate(keys)}
    inlet, outlet = ids[(0, 0, 0)], ids[(4 * n, 4 * n, 4 * n)]
    parent = np.arange(nodes.shape[0], dtype=np.int64)

    def find(i):
        while int(parent[i]) != i:
            parent[i] = parent[int(parent[i])]
            i = int(parent[i])
        return i

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    if find(inlet) != find(outlet):
        connected = np.asarray(
            [i for i in range(nodes.shape[0]) if find(i) == find(inlet) and i != outlet]
        )
        near = int(
            connected[
                np.argmin(np.sum((nodes[connected] - nodes[outlet]) ** 2, axis=1))
            ]
        )
        edges.append((near, outlet))
        parent[find(outlet)] = find(near)
    keep = [i for i in range(nodes.shape[0]) if find(i) == find(inlet)]
    remap = {old: new for new, old in enumerate(keep)}
    new_edges = [(remap[a], remap[b]) for a, b in edges if a in remap and b in remap]
    new_keys = [keys[i] for i in keep]
    return (
        nodes[np.asarray(keep)],
        new_keys,
        new_edges,
        {key: i for i, key in enumerate(new_keys)},
    )


def _snap_nodes(nodes: np.ndarray, points: list[list[float]]) -> list[int]:
    selected: list[int] = []
    for point in points:
        p = np.asarray(point, dtype=float).reshape(3)
        idx = int(np.argmin(np.sum((nodes - p) ** 2, axis=1)))
        if idx not in selected:
            selected.append(idx)
    if not selected:
        raise ValueError("at least one boundary point is required")
    return selected


def _attach_boundary_points(
    nodes: np.ndarray,
    edges: np.ndarray,
    points: list[list[float]],
    *,
    candidate_count: int,
    boundary_first: bool,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], int]:
    """Add prescribed boundary locations and connect each to its nearest lattice node."""
    base = np.asarray(nodes[: int(candidate_count)], dtype=float)
    if not len(base):
        raise ValueError("cannot attach a boundary point to an empty lattice")
    node_list = [row.copy() for row in np.asarray(nodes, dtype=float)]
    edge_list = [tuple(map(int, edge)) for edge in np.asarray(edges, dtype=np.int64)]
    boundary_nodes: list[int] = []
    anchors: list[int] = []
    added = 0
    tolerance = max(float(np.ptp(base, axis=0).max()), 1.0) * 1e-12
    for raw in points:
        point = np.asarray(raw, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("each inlet/outlet location must contain three finite coordinates")
        anchor = int(np.argmin(np.sum((base - point) ** 2, axis=1)))
        if float(np.linalg.norm(base[anchor] - point)) <= tolerance:
            boundary = anchor
        else:
            duplicate = next(
                (
                    index
                    for index in boundary_nodes
                    if float(np.linalg.norm(np.asarray(node_list[index]) - point))
                    <= tolerance
                ),
                None,
            )
            if duplicate is not None:
                boundary = int(duplicate)
            else:
                boundary = len(node_list)
                node_list.append(point.copy())
                edge = (
                    (boundary, anchor) if boundary_first else (anchor, boundary)
                )
                edge_list.append(edge)
                added += 1
        if boundary not in boundary_nodes:
            boundary_nodes.append(boundary)
            anchors.append(anchor)
    if not boundary_nodes:
        raise ValueError("at least one inlet/outlet location is required")
    return (
        np.asarray(node_list, dtype=float),
        np.asarray(edge_list, dtype=np.int64),
        boundary_nodes,
        anchors,
        added,
    )


def _subdivide(nodes, edges, subdivisions):
    node_list = [row.copy() for row in np.asarray(nodes)]
    new_edges: list[tuple[int, int]] = []
    for a, b in np.asarray(edges, dtype=np.int64):
        chain = [int(a)]
        for part in range(1, subdivisions):
            t = part / subdivisions
            node_list.append((1.0 - t) * nodes[a] + t * nodes[b])
            chain.append(len(node_list) - 1)
        chain.append(int(b))
        new_edges.extend(zip(chain[:-1], chain[1:]))
    return np.asarray(node_list), np.asarray(new_edges, dtype=np.int64)


def _clip_to_domain(nodes, edges, node_inside):
    try:
        inside = np.asarray(node_inside(np.asarray(nodes, dtype=float)), dtype=bool).reshape(-1)
    except Exception as exc:
        raise ValueError(f"could not evaluate lattice nodes against the domain: {exc}") from exc
    if inside.size != nodes.shape[0]:
        raise ValueError(
            f"domain containment returned {inside.size} values for {nodes.shape[0]} lattice nodes"
        )
    keep_edges = inside[edges[:, 0]] & inside[edges[:, 1]]
    clipped_edges = np.asarray(edges[keep_edges], dtype=np.int64)
    if clipped_edges.size == 0:
        raise ValueError(
            "domain clipping removed every lattice edge; increase cells per axis or verify the domain scale"
        )
    used_nodes = np.unique(clipped_edges.reshape(-1))
    remap = np.full(nodes.shape[0], -1, dtype=np.int64)
    remap[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    return np.asarray(nodes[used_nodes], dtype=float), remap[clipped_edges]


def _keep_flow_components(nodes, edges, inlet_nodes, outlet_nodes):
    """Discard clipped graph islands that have no inlet-to-outlet flow path."""
    parent = np.arange(nodes.shape[0], dtype=np.int64)

    def find(index):
        index = int(index)
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    for a, b in np.asarray(edges, dtype=np.int64):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
    labels = np.asarray([find(i) for i in range(nodes.shape[0])], dtype=np.int64)
    inlet_components = {int(labels[i]) for i in inlet_nodes}
    outlet_components = {int(labels[i]) for i in outlet_nodes}
    flow_components = inlet_components & outlet_components
    if not flow_components:
        raise ValueError(
            "no connected lattice path remains between the selected inlet and outlet locations after domain clipping"
        )
    keep = np.flatnonzero(np.isin(labels, list(flow_components)))
    remap = np.full(nodes.shape[0], -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    edge_mask = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    kept_edges = remap[np.asarray(edges[edge_mask], dtype=np.int64)]
    kept_inlets = [int(remap[i]) for i in inlet_nodes if remap[i] >= 0]
    kept_outlets = [int(remap[i]) for i in outlet_nodes if remap[i] >= 0]
    return np.asarray(nodes[keep], dtype=float), kept_edges, kept_inlets, kept_outlets


def _radii(starts, ends, r0, dims, expression):
    if not str(expression or "").strip():
        return np.full(starts.shape[0], r0, dtype=np.float64)
    try:
        import numexpr as ne

        mid = 0.5 * (starts + ends)
        normalized_expression = str(expression).replace("^", "**")
        values = ne.evaluate(
            normalized_expression,
            local_dict={
                "x": mid[:, 0],
                "y": mid[:, 1],
                "z": mid[:, 2],
                "r0": r0,
                "L": float(np.max(dims)),
            },
            global_dict={},
        )
    except Exception as exc:
        raise ValueError(f"invalid radius expression: {exc}") from exc
    radii = np.broadcast_to(np.asarray(values, dtype=float), (starts.shape[0],)).copy()
    if np.any(~np.isfinite(radii)):
        raise ValueError("radius expression must be finite on every segment")
    if np.any(radii <= 0.0):
        raise ValueError(
            "radius expression must be positive on every segment. abs(...) is supported, "
            "but abs(0) is still zero; add a small positive floor such as + 1e-8."
        )
    return radii
