"""Domain, vessel, and tissue preview geometry construction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from cascade.utils.resources import resolve_domain_path
from cascade.vessels.lattice import (
    channel_count,
    generate_lattice,
    resolve_lattice_layout,
)

from cascade.gui.visualization.fields import (
    _mesh_line_segments,
    _point_array,
    _polyline_data,
    _values,
)


# Display-only rotation that brings the packaged bivent3 coordinate frame into
# CASCADE Studio's ordinary home camera.  It never changes solver coordinates.
_BIVENT3_DISPLAY_ROTATION = np.asarray(
    (
        (-0.961013694169, 0.242651846526, -0.132562291004),
        (0.023633505460, 0.549759394977, 0.834988661632),
        (0.275488905473, 0.799302626677, -0.534061020811),
    ),
    dtype=float,
)

_MAX_EXACT_LATTICE_PREVIEW_STRUTS = 1_000_000


def domain_display_rotation(domain: dict[str, Any]) -> np.ndarray:
    """Return the preview-only model rotation for a configured domain."""
    kind = str(domain.get("type", domain.get("kind", ""))).lower()
    path_name = Path(str(domain.get("path", ""))).name.lower()
    if kind == "bivent3" or (kind == "file" and path_name == "bivent3.stl"):
        return _BIVENT3_DISPLAY_ROTATION.copy()
    return np.eye(3, dtype=float)


def domain_wireframe(domain: dict[str, Any]) -> np.ndarray:
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    if kind == "sphere":
        radius = float(domain.get("radius", 0.5))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        lines = []
        theta = np.linspace(0, 2 * np.pi, 49)
        for phi in np.linspace(-np.pi / 2, np.pi / 2, 7)[1:-1]:
            ring = center + radius * np.column_stack(
                (
                    np.cos(phi) * np.cos(theta),
                    np.cos(phi) * np.sin(theta),
                    np.full_like(theta, np.sin(phi)),
                )
            )
            lines.extend(np.stack((ring[:-1], ring[1:]), axis=1))
        phi = np.linspace(-np.pi / 2, np.pi / 2, 25)
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            arc = center + radius * np.column_stack(
                (np.cos(phi) * np.cos(angle), np.cos(phi) * np.sin(angle), np.sin(phi))
            )
            lines.extend(np.stack((arc[:-1], arc[1:]), axis=1))
        return np.asarray(lines, dtype=np.float32)
    if kind in {"cylinder", "disk"}:
        radius = float(domain.get("radius", 0.5))
        height = float(domain.get("height", domain.get("z_length", 1.0)))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        theta = np.linspace(0.0, 2.0 * np.pi, 49)
        rings = []
        for z in (-0.5 * height, 0.5 * height):
            ring = center + np.column_stack(
                (radius * np.cos(theta), radius * np.sin(theta), np.full_like(theta, z))
            )
            rings.extend(np.stack((ring[:-1], ring[1:]), axis=1))
        for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            bottom = center + [radius * np.cos(angle), radius * np.sin(angle), -0.5 * height]
            top = center + [radius * np.cos(angle), radius * np.sin(angle), 0.5 * height]
            rings.append(np.stack((bottom, top)))
        return np.asarray(rings, dtype=np.float32)
    if kind == "file" and domain.get("path"):
        try:
            path = resolve_domain_path(domain["path"])
            if path is None:
                raise FileNotFoundError(domain["path"])
            stat = path.stat()
            return _cached_domain_wireframe(
                str(path), int(stat.st_mtime_ns), int(stat.st_size)
            ).copy()
        except Exception:
            return np.empty((0, 2, 3), dtype=np.float32)
    dims = np.array(
        [
            domain.get("x_length", domain.get("side_length", 1.0)),
            domain.get("y_length", domain.get("side_length", 1.0)),
            domain.get("z_length", domain.get("side_length", 1.0)),
        ],
        dtype=float,
    )
    center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
    corners = (
        np.array(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
            dtype=float,
        )
        * dims
        + center
    )
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.sum(corners[i] != corners[j]) == 1
    ]
    return np.asarray([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)


def domain_surface_triangles(domain: dict[str, Any]) -> np.ndarray:
    """Return a bounded triangle surface suitable for the lightweight canvas."""
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    if kind == "file" and domain.get("path"):
        try:
            path = resolve_domain_path(domain["path"])
            if path is None:
                raise FileNotFoundError(domain["path"])
            stat = path.stat()
            return _cached_domain_triangles(
                str(path), int(stat.st_mtime_ns), int(stat.st_size)
            ).copy()
        except Exception:
            return np.empty((0, 3, 3), dtype=np.float32)
    if kind == "sphere":
        radius = float(domain.get("radius", 0.5))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        longitude = np.linspace(0.0, 2.0 * np.pi, 33)
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 19)
        points = center + radius * np.asarray(
            [
                [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]
                for phi in latitude
                for theta in longitude
            ],
            dtype=float,
        )
        triangles = []
        width = len(longitude)
        for row in range(len(latitude) - 1):
            for column in range(len(longitude) - 1):
                a = row * width + column
                b = a + 1
                c = a + width
                d = c + 1
                triangles.extend((points[[a, c, b]], points[[b, c, d]]))
        return np.asarray(triangles, dtype=np.float32)
    if kind in {"cylinder", "disk"}:
        radius = float(domain.get("radius", 0.5))
        height = float(domain.get("height", domain.get("z_length", 1.0)))
        center = np.asarray(domain.get("center", [0.0, 0.0, 0.0]), dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, 49)
        bottom_center = center + [0.0, 0.0, -0.5 * height]
        top_center = center + [0.0, 0.0, 0.5 * height]
        bottom = center + np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, -0.5 * height))
        )
        top = center + np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, 0.5 * height))
        )
        triangles = []
        for index in range(len(angles) - 1):
            triangles.extend(
                (
                    np.asarray([bottom[index], top[index], bottom[index + 1]]),
                    np.asarray([bottom[index + 1], top[index], top[index + 1]]),
                    np.asarray([bottom_center, bottom[index + 1], bottom[index]]),
                    np.asarray([top_center, top[index], top[index + 1]]),
                )
            )
        return np.asarray(triangles, dtype=np.float32)

    dims, center = _domain_dimensions(domain)
    corners = (
        np.asarray(
            [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
            dtype=float,
        )
        * dims
        + center
    )
    quads = (
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    )
    triangles = []
    for a, b, c, d in quads:
        triangles.extend((corners[[a, b, c]], corners[[a, c, d]]))
    return np.asarray(triangles, dtype=np.float32)


def network_geometry(config: dict[str, Any]):
    network = config.get("network", {})
    source = config.get("gui", {}).get("network_source", "svv_generated")
    simple = network.get("simple", {})
    if source == "custom" or simple.get("mode") == "custom":
        from cascade.vessels.simple import _load_custom_geometry

        raw_path = simple.get("path", simple.get("geometry_path"))
        if not raw_path:
            raise ValueError("Choose a CSV or NPZ vessel network to preview")
        starts, ends, radii, _lengths, inlets, outlets, prox, dist = (
            _load_custom_geometry(
                Path(str(raw_path)).expanduser(),
                default_radius_cm=float(simple.get("radius_cm", 0.015)),
                inlet_nodes=simple.get("inlet_nodes"),
                outlet_nodes=simple.get("outlet_nodes"),
            )
        )
        inlet_mask = np.isin(prox, np.asarray(inlets, dtype=np.int64))
        outlet_mask = np.isin(dist, np.asarray(outlets, dtype=np.int64))
        return (
            starts,
            ends,
            radii,
            starts[inlet_mask],
            ends[outlet_mask],
            None,
            f"{len(starts):,} imported vessels shown",
        )
    if source == "lattice" or simple.get("mode") == "lattice":
        lattice_type = str(simple.get("lattice_type", "cubic"))
        subdivisions = max(int(simple.get("subdivisions", 1)), 1)
        dims, center = _domain_dimensions(config.get("domain", {}))
        layout = resolve_lattice_layout(
            tuple(float(value) for value in dims),
            cells=int(simple.get("cells", simple.get("cells_per_axis", 4))),
            sizing_mode=str(simple.get("sizing_mode", "cells")),
            cell_spacing_cm=simple.get("cell_spacing_cm"),
            anisotropy_yx=float(simple.get("anisotropy_yx", 1.0)),
            anisotropy_zx=float(simple.get("anisotropy_zx", 1.0)),
            lattice_type=lattice_type,
        )
        estimated = (
            channel_count(int(layout["generator_cells"]), lattice_type) * subdivisions
        )
        if estimated > _MAX_EXACT_LATTICE_PREVIEW_STRUTS:
            raise ValueError(
                f"exact lattice preview requires {estimated:,} struts, above the "
                f"{_MAX_EXACT_LATTICE_PREVIEW_STRUTS:,}-strut interactive limit; "
                "no lower-resolution substitute is shown"
            )
        inside = _analytic_inside(config.get("domain", {}))
        lattice = generate_lattice(
            int(layout["generator_cells"]),
            tuple(float(value) for value in layout["generator_dimensions_cm"]),
            float(simple.get("radius_cm", 0.0005)),
            lattice_type=lattice_type,
            inlet_points_cm=simple.get("inlet_points_cm"),
            outlet_points_cm=simple.get("outlet_points_cm"),
            radius_expression=simple.get("radius_expression"),
            subdivisions=subdivisions,
            center_cm=tuple(
                float(center[axis] + layout["generator_center_offset_cm"][axis])
                for axis in range(3)
            ),
            node_inside=inside,
            radius_reference_dimensions_cm=tuple(float(value) for value in dims),
        )
        inlet_connections = int(lattice.get("inlet_connection_count", 0))
        outlet_connections = int(lattice.get("outlet_connection_count", 0))
        detail = (
            f"{len(lattice['segment_starts_cm']):,} vessels  │  "
            f"{len(lattice['inlet_nodes'])} inlet{'s' if len(lattice['inlet_nodes']) != 1 else ''}  │  "
            f"{len(lattice['outlet_nodes'])} outlet{'s' if len(lattice['outlet_nodes']) != 1 else ''}"
        )
        if inlet_connections or outlet_connections:
            detail += (
                f"  │  {inlet_connections + outlet_connections} boundary connections"
            )
        return (
            lattice["segment_starts_cm"],
            lattice["segment_ends_cm"],
            lattice["segment_radii_cm"],
            lattice["inlet_points_cm"],
            lattice["outlet_points_cm"],
            None,
            detail,
        )
    if source == "simple" or network.get("mode") == "simple":
        starts, ends = _simple_geometry(config)
        radii = np.full(len(starts), float(simple.get("radius_cm", 0.015)))
        return (
            starts,
            ends,
            radii,
            starts[:1],
            ends[-1:],
            None,
            f"{len(starts):,} channels shown",
        )
    if source == "uploaded" and network.get("input_path"):
        starts, ends = _load_uploaded_geometry(Path(str(network["input_path"])))
        return (
            starts,
            ends,
            None,
            starts[:1],
            ends[-1:],
            None,
            f"{len(starts):,} uploaded vessels",
        )
    roots = network.get("roots") or ([network["root"]] if network.get("root") else [])
    starts, ends, alpha, inlets = _svv_placeholder_geometry(config, roots)
    return (
        starts,
        ends,
        None,
        inlets,
        None,
        alpha,
        "Preparing the exact hydraulic SVV seed…",
    )


def _svv_placeholder_geometry(config, roots):
    """Immediate low-cost branching cue while the exact seed worker runs."""
    side = float(config.get("domain", {}).get("side_length", 1.0))
    all_starts, all_ends, all_alpha, inlets = [], [], [], []
    for root_index, root in enumerate(roots or [{}]):
        start = np.asarray(root.get("start", [0.0, 0.0, 0.0]), dtype=float)
        direction = np.asarray(root.get("direction", [1.0, 0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm else np.array([1.0, 0.0, 0.0])
        inlets.append(start)
        frontier = [(start, direction)]
        length = 0.18 * side
        for depth in range(5):
            next_frontier = []
            for branch_index, (origin, heading) in enumerate(frontier):
                endpoint = origin + heading * length
                all_starts.append(origin)
                all_ends.append(endpoint)
                all_alpha.append(max(0.22, 1.0 - 0.18 * depth))
                if depth == 4:
                    continue
                axis = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(heading, axis))) > 0.82:
                    axis = np.array([0.0, 1.0, 0.0])
                lateral = np.cross(heading, axis)
                lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
                twist = np.cross(heading, lateral)
                sign = -1.0 if (branch_index + root_index + depth) % 2 else 1.0
                for branch_sign in (-1.0, 1.0):
                    child = (
                        0.84 * heading
                        + 0.48 * branch_sign * lateral
                        + 0.13 * sign * twist
                    )
                    child /= max(float(np.linalg.norm(child)), 1e-12)
                    next_frontier.append((endpoint, child))
            frontier = next_frontier
            length *= 0.60
    return (
        np.asarray(all_starts, dtype=np.float32),
        np.asarray(all_ends, dtype=np.float32),
        np.asarray(all_alpha, dtype=np.float32),
        np.asarray(inlets, dtype=np.float32),
    )


def limit_near_inlets(
    starts: np.ndarray,
    ends: np.ndarray,
    inlet_points: np.ndarray | list,
    *,
    limit: int,
    values: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
):
    starts, ends = _point_array(starts), _point_array(ends)
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    vals = _values(values, n)
    alphas = _values(alpha, n, fill=1.0)
    if n <= max(int(limit), 0):
        return starts, ends, vals if values is not None else alphas, alphas
    inlets = _point_array(inlet_points)
    if not len(inlets):
        inlets = starts[:1]
    midpoint = 0.5 * (starts + ends)
    distances = np.linalg.norm(midpoint[:, None, :] - inlets[None, :, :], axis=2)
    owner = np.argmin(distances, axis=1)
    chosen: list[int] = []
    quota = max(int(limit) // len(inlets), 1)
    for inlet in range(len(inlets)):
        ids = np.flatnonzero(owner == inlet)
        ids = ids[np.argsort(distances[ids, inlet], kind="stable")]
        chosen.extend(ids[:quota].tolist())
    if len(chosen) < int(limit):
        remaining = np.setdiff1d(
            np.arange(n), np.asarray(chosen, dtype=int), assume_unique=False
        )
        nearest = np.min(distances[remaining], axis=1)
        chosen.extend(
            remaining[
                np.argsort(nearest, kind="stable")[: int(limit) - len(chosen)]
            ].tolist()
        )
    ids = np.asarray(chosen[: int(limit)], dtype=int)
    third = vals[ids] if values is not None else alphas[ids]
    return starts[ids], ends[ids], third, alphas[ids]


def select_vessel_indices(
    starts: np.ndarray,
    ends: np.ndarray,
    inlet_points: np.ndarray | list,
    *,
    mode: str,
    limit: int,
) -> np.ndarray:
    """Select logical vessels deterministically without copying geometry."""
    starts, ends = _point_array(starts), _point_array(ends)
    count = min(len(starts), len(ends))
    mode = str(mode or "near").lower()
    limit = min(max(int(limit), 0), count)
    if count == 0 or mode == "none" or limit == 0:
        return np.empty((0,), dtype=int)
    if mode == "all" or count <= limit:
        return np.arange(count, dtype=int)
    if mode == "random":
        ids = np.random.default_rng(42).choice(count, size=limit, replace=False)
        ids.sort()
        return ids

    inlets = _point_array(inlet_points)
    if not len(inlets):
        inlets = starts[:1]
    midpoint = 0.5 * (starts[:count] + ends[:count])
    distances = np.linalg.norm(midpoint[:, None, :] - inlets[None, :, :], axis=2)
    owner = np.argmin(distances, axis=1)
    chosen: list[int] = []
    quota = max(limit // len(inlets), 1)
    for inlet in range(len(inlets)):
        ids = np.flatnonzero(owner == inlet)
        ids = ids[np.argsort(distances[ids, inlet], kind="stable")]
        chosen.extend(ids[:quota].tolist())
    if len(chosen) < limit:
        remaining = np.setdiff1d(
            np.arange(count), np.asarray(chosen, dtype=int), assume_unique=False
        )
        nearest = np.min(distances[remaining], axis=1)
        chosen.extend(
            remaining[
                np.argsort(nearest, kind="stable")[: limit - len(chosen)]
            ].tolist()
        )
    return np.asarray(chosen[:limit], dtype=int)


def select_tissue_points(
    points: np.ndarray,
    inlet_points: np.ndarray | list,
    *,
    mode: str,
    limit: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose result/preview points without changing the underlying dataset."""
    points = _point_array(points)
    count = len(points)
    if count == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=np.float32)
    mode = str(mode or "near").lower()
    limit = max(int(limit), 0)
    if mode == "none":
        return np.empty((0,), dtype=int), np.empty((0,), dtype=np.float32)
    if mode == "all" or count <= limit:
        return np.arange(count, dtype=int), np.ones(count, dtype=np.float32)
    if mode == "random":
        ids = np.random.default_rng(42).choice(count, size=limit, replace=False)
        ids.sort()
        return ids, np.ones(len(ids), dtype=np.float32)

    inlets = _point_array(inlet_points)
    if not len(inlets):
        inlets = np.mean(points, axis=0, keepdims=True)
    nearest = np.full(count, np.inf, dtype=float)
    for inlet in inlets:
        nearest = np.minimum(nearest, np.linalg.norm(points - inlet, axis=1))
    ids = np.argpartition(nearest, limit - 1)[:limit]
    ids = ids[np.argsort(nearest[ids], kind="stable")]
    selected_distance = nearest[ids]
    span = float(np.ptp(selected_distance))
    if span <= 1.0e-12:
        alpha = np.ones(len(ids), dtype=np.float32)
    else:
        fade = 1.0 - (selected_distance - selected_distance.min()) / span
        alpha = np.asarray(np.sqrt(np.clip(fade, 0.0, 1.0)), dtype=np.float32)
    return ids, alpha


def _requested_tissue_count(config: dict[str, Any]) -> int:
    simulation = config.get("simulation", {})
    if str(simulation.get("sample_mode", "random")) == "grid":
        grid = simulation.get("tissue_grid", {})
        fallback = list(grid.get("shape", grid.get("dimensions", [20, 20, 20])))
        fallback = (fallback + [20, 20, 20])[:3]
        shape = np.maximum(
            np.asarray(
                [
                    grid.get("nx", fallback[0]),
                    grid.get("ny", fallback[1]),
                    grid.get("nz", fallback[2]),
                ],
                dtype=int,
            ),
            2,
        )
        return int(np.prod(shape, dtype=np.int64))
    return max(int(simulation.get("distance_sample_count", 10_000)), 0)


def preview_tissue_geometry(
    config: dict[str, Any],
    inlet_points: np.ndarray | list | None,
    *,
    mode: str = "near",
    limit: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a bounded visual sample of the requested tissue-point layout."""
    simulation = config.get("simulation", {})
    layout_mode = str(simulation.get("sample_mode", "random"))
    if layout_mode == "grid":
        grid = simulation.get("tissue_grid", {})
        fallback = list(grid.get("shape", grid.get("dimensions", [20, 20, 20])))
        fallback = (fallback + [20, 20, 20])[:3]
        raw_shape = [
            grid.get("nx", fallback[0]),
            grid.get("ny", fallback[1]),
            grid.get("nz", fallback[2]),
        ]
        shape = np.maximum(np.asarray(raw_shape, dtype=int), 2)
        requested = int(np.prod(shape, dtype=np.int64))
    else:
        requested = max(int(simulation.get("distance_sample_count", 10_000)), 0)
        shape = None
    if requested == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            0,
        )

    selection_mode = str(mode or "near").lower()
    if selection_mode == "none" or limit <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            requested,
        )

    # Evaluate enough candidates to make "nearest" representative while
    # keeping setup previews small even if a production grid has billions of points.
    multiplier = 6 if selection_mode == "near" else 1
    candidate_count = min(requested, max(limit * multiplier, limit))
    points = _sample_preview_domain_points(
        config.get("domain", {}),
        mode=layout_mode,
        count=candidate_count,
        shape=shape,
        random_seed=int(config.get("domain", {}).get("random_seed", 42)),
    )
    if requested <= limit:
        selection_mode = "all"
    ids, alpha = select_tissue_points(
        points,
        inlet_points if inlet_points is not None else [],
        mode=selection_mode,
        limit=limit,
    )
    return points[ids], alpha, requested


def _sample_preview_domain_points(
    domain: dict[str, Any],
    *,
    mode: str,
    count: int,
    shape: np.ndarray | None,
    random_seed: int,
) -> np.ndarray:
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    dims, center = _domain_dimensions(domain)
    surface = None
    if kind == "file" and domain.get("path"):
        path = resolve_domain_path(domain["path"])
        if path is None:
            raise FileNotFoundError(domain["path"])
        stat = path.stat()
        surface = _cached_domain_surface(
            str(path), int(stat.st_mtime_ns), int(stat.st_size)
        )
        bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
        center = bounds.mean(axis=1)
        dims = bounds[:, 1] - bounds[:, 0]

    already_inside = False
    if mode == "grid":
        grid_shape = _bounded_grid_shape(shape, count)
        axes = [
            np.linspace(
                center[axis] - 0.5 * dims[axis],
                center[axis] + 0.5 * dims[axis],
                int(grid_shape[axis]),
            )
            for axis in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack([values.reshape(-1) for values in mesh])
    elif kind == "sphere":
        rng = np.random.default_rng(random_seed)
        direction = rng.normal(size=(count, 3))
        direction /= np.maximum(
            np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-12
        )
        radius = float(domain.get("radius", dims[0] / 2.0))
        points = (
            center + direction * (rng.random(count) ** (1.0 / 3.0) * radius)[:, None]
        )
    elif surface is not None:
        rng = np.random.default_rng(random_seed)
        accepted = []
        remaining = count
        for _attempt in range(8):
            if remaining <= 0:
                break
            batch_count = min(max(remaining * 6, 10_000), 500_000)
            candidates = center + (rng.random((batch_count, 3)) - 0.5) * dims
            enclosed = _points_inside_surface(candidates, surface)
            if len(enclosed):
                accepted.append(enclosed[:remaining])
                remaining -= min(len(enclosed), remaining)
        points = (
            np.concatenate(accepted, axis=0)
            if accepted
            else np.empty((0, 3), dtype=float)
        )
        already_inside = True
    else:
        rng = np.random.default_rng(random_seed)
        points = center + (rng.random((count, 3)) - 0.5) * dims

    inside = _analytic_inside(domain)
    if inside is not None:
        points = points[np.asarray(inside(points), dtype=bool)]
    elif surface is not None and len(points) and not already_inside:
        points = _points_inside_surface(points, surface)
    return np.asarray(points, dtype=np.float32)


def _points_inside_surface(points: np.ndarray, surface) -> np.ndarray:
    mask = _surface_inside_mask(points, surface)
    return np.asarray(points)[mask]


def _surface_inside_mask(points: np.ndarray, surface) -> np.ndarray:
    """Return one containment flag per point for a closed mesh surface."""
    import pyvista as pv

    selected = pv.PolyData(points).select_enclosed_points(
        surface, tolerance=1.0e-6, check_surface=False
    )
    return np.asarray(selected["SelectedPoints"], dtype=bool)


def _bounded_grid_shape(shape: np.ndarray | None, maximum: int) -> np.ndarray:
    result = np.maximum(
        np.asarray(shape if shape is not None else [2, 2, 2], dtype=int), 2
    )
    product = int(np.prod(result, dtype=np.int64))
    if product <= maximum:
        return result
    scale = (float(maximum) / float(product)) ** (1.0 / 3.0)
    result = np.maximum(np.floor(result * scale).astype(int), 2)
    while int(np.prod(result, dtype=np.int64)) > maximum:
        axis = int(np.argmax(result))
        result[axis] = max(int(result[axis]) - 1, 2)
    return result


def _domain_dimensions(domain):
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    if kind == "file" and domain.get("path"):
        surface = _file_domain_surface(domain)
        bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
        return bounds[:, 1] - bounds[:, 0], bounds.mean(axis=1)

    side = float(domain.get("side_length", 1.0))
    if kind == "sphere":
        side = 2.0 * float(domain.get("radius", side / 2.0))
    elif kind in {"cylinder", "disk"}:
        radius = float(domain.get("radius", side / 2.0))
        height = float(domain.get("height", domain.get("z_length", side)))
        return np.asarray([2.0 * radius, 2.0 * radius, height]), np.asarray(
            domain.get("center", [0, 0, 0]), dtype=float
        )
    dims = np.asarray(
        [
            domain.get("x_length", side),
            domain.get("y_length", side),
            domain.get("z_length", side),
        ],
        dtype=float,
    )
    center = np.asarray(domain.get("center", [0, 0, 0]), dtype=float)
    return dims, center


def _analytic_inside(domain):
    kind = str(domain.get("type", domain.get("kind", "cube"))).lower()
    dims, center = _domain_dimensions(domain)
    if kind == "sphere":
        radius = float(domain.get("radius", dims[0] / 2.0))
        return lambda points: (
            np.linalg.norm(np.asarray(points) - center, axis=1)
            <= radius * (1.0 + 1e-10)
        )
    if kind in {"cylinder", "disk"}:
        radius = float(domain.get("radius", dims[0] / 2.0))
        half_height = 0.5 * float(domain.get("height", dims[2]))
        return lambda points: (
            np.sum((np.asarray(points)[:, :2] - center[:2]) ** 2, axis=1)
            <= radius * radius * (1.0 + 1e-10)
        ) & (
            np.abs(np.asarray(points)[:, 2] - center[2])
            <= half_height * (1.0 + 1e-10)
        )
    if kind in {"cube", "box"}:
        return lambda points: np.all(
            np.abs(np.asarray(points) - center) <= 0.5 * dims + 1e-10, axis=1
        )
    if kind == "file" and domain.get("path"):
        surface = _file_domain_surface(domain)
        return lambda points: _surface_inside_mask(points, surface)
    return None


def _file_domain_surface(domain: dict[str, Any]):
    """Resolve and cache the triangulated surface for a file-backed domain."""
    path = resolve_domain_path(domain.get("path"))
    if path is None or not path.exists():
        raise FileNotFoundError(domain.get("path"))
    stat = path.stat()
    return _cached_domain_surface(str(path), int(stat.st_mtime_ns), int(stat.st_size))


def _simple_geometry(config):
    simple = config.get("network", {}).get("simple", {})
    dims, center = _domain_dimensions(config.get("domain", {}))
    mode = str(simple.get("mode", "onechannel"))
    axis = 0 if str(simple.get("axis", "x")) == "x" else 1
    lo, hi = center - 0.48 * dims, center + 0.48 * dims
    if mode == "multichannel":
        offsets = [float(v) for v in simple.get("y_offsets_cm", [-0.2, 0.0, 0.2])]
        starts, ends = [], []
        for offset in offsets:
            a, b = center.copy(), center.copy()
            a[axis], b[axis] = lo[axis], hi[axis]
            a[1 - axis] += offset
            b[1 - axis] += offset
            starts.append(a)
            ends.append(b)
        return np.asarray(starts), np.asarray(ends)
    if mode == "snake":
        # Use the solver's channel generator rather than a decorative sine
        # wave.  The setup preview must be the exact path that is exported.
        from cascade.vessels.simple import _snake_channel_points

        z = -dims[2] / 2.0 + float(simple.get("z_from_bottom_cm", dims[2] * 0.5))
        points = _snake_channel_points(
            z,
            int(simple.get("snake_arc_segments", 5)),
            int(simple.get("snake_straight_segments", 5)),
        )
        return points[:-1], points[1:]
    a, b = center.copy(), center.copy()
    a[axis], b[axis] = lo[axis], hi[axis]
    return a.reshape(1, 3), b.reshape(1, 3)


def _load_uploaded_geometry(path: Path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "starts" in data and "ends" in data:
                return np.asarray(data["starts"]), np.asarray(data["ends"])
            if "data" in data:
                raw = np.asarray(data["data"])
                return raw[:, :3], raw[:, 3:6]
    if path.suffix.lower() in {".vtk", ".vtp", ".vtu", ".ply"}:
        import pyvista as pv

        starts, ends, _, _ = _polyline_data(pv.read(path))
        return starts, ends
    raise ValueError("Build the preview seed to inspect this network format")


def _mesh_wireframe(mesh) -> np.ndarray:
    if mesh is None:
        return np.empty((0, 2, 3), dtype=np.float32)
    try:
        surface = mesh.extract_surface().triangulate().clean()
        if surface.n_cells > 5_000:
            reduction = 1.0 - 4_000.0 / float(surface.n_cells)
            surface = (
                surface.decimate_pro(reduction, preserve_topology=True)
                .triangulate()
                .clean()
            )
        bounds = np.asarray(surface.bounds, dtype=float).reshape(3, 2)
        center = bounds.mean(axis=1)
        contour_segments = []
        for axis in range(3):
            normal = np.zeros(3, dtype=float)
            normal[axis] = 1.0
            # Nine coherent orthogonal contours show the full domain extent
            # without reverting to the old every-triangle visual noise.
            for fraction in (0.20, 0.50, 0.80):
                origin = center.copy()
                origin[axis] = bounds[axis, 0] + fraction * (
                    bounds[axis, 1] - bounds[axis, 0]
                )
                sliced = surface.slice(normal=normal, origin=origin)
                segments = _mesh_line_segments(sliced)
                if len(segments):
                    contour_segments.append(segments)

        feature_mesh = surface.extract_feature_edges(
            feature_angle=85.0,
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=True,
            manifold_edges=False,
        )
        features = _mesh_line_segments(feature_mesh)
        if len(features) > 600:
            # Sampling individual edge fragments creates broken, dotted
            # contours.  Prefer the coherent section curves on very intricate
            # surfaces instead.
            features = np.empty((0, 2, 3), dtype=np.float32)
        if len(features):
            contour_segments.append(features)
        if not contour_segments:
            return np.empty((0, 2, 3), dtype=np.float32)
        lines = np.concatenate(contour_segments, axis=0)
        return np.asarray(lines, dtype=np.float32)
    except Exception:
        return np.empty((0, 2, 3), dtype=np.float32)


def _mesh_triangles(mesh, *, maximum: int = 1_800) -> np.ndarray:
    if mesh is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    try:
        surface = mesh.extract_surface().triangulate().clean()
        if surface.n_cells > maximum:
            reduction = 1.0 - float(maximum) / float(surface.n_cells)
            surface = (
                surface.decimate_pro(reduction, preserve_topology=True)
                .triangulate()
                .clean()
            )
        faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
        faces = faces[faces[:, 0] == 3, 1:4]
        if len(faces) > maximum:
            faces = faces[:maximum]
        return np.asarray(surface.points[faces], dtype=np.float32)
    except Exception:
        return np.empty((0, 3, 3), dtype=np.float32)


@lru_cache(maxsize=6)
def _cached_domain_wireframe(
    path: str, _modified_ns: int, _file_size: int
) -> np.ndarray:
    return _mesh_wireframe(_cached_domain_surface(path, _modified_ns, _file_size))


@lru_cache(maxsize=6)
def _cached_domain_triangles(
    path: str, _modified_ns: int, _file_size: int
) -> np.ndarray:
    return _mesh_triangles(_cached_domain_surface(path, _modified_ns, _file_size))


@lru_cache(maxsize=6)
def _cached_domain_surface(path: str, _modified_ns: int, _file_size: int):
    import pyvista as pv

    return pv.read(path).extract_surface().triangulate().clean()


__all__ = (
    "domain_display_rotation",
    "domain_wireframe",
    "domain_surface_triangles",
    "network_geometry",
    "_svv_placeholder_geometry",
    "limit_near_inlets",
    "select_vessel_indices",
    "select_tissue_points",
    "_requested_tissue_count",
    "preview_tissue_geometry",
    "_sample_preview_domain_points",
    "_points_inside_surface",
    "_bounded_grid_shape",
    "_domain_dimensions",
    "_analytic_inside",
    "_simple_geometry",
    "_load_uploaded_geometry",
    "_mesh_wireframe",
    "_mesh_triangles",
    "_cached_domain_wireframe",
    "_cached_domain_triangles",
    "_cached_domain_surface",
)
