"""Construct regular tissue grids and retain points that lie inside a domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv


@dataclass
class GridSpec:
    nx: int = 200
    ny: int = 200
    nz: int = 200
    boundary_resolution: int = 28
    implicit_margin: float = 0.0
    disable_enclosed_check: bool = True
    enclosed_tolerance: float = 1.0e-6
    inside_combine_mode: str = "and"
    chunk_points: int = 250_000


def grid_spec_from_dict(raw: dict[str, Any] | None) -> GridSpec:
    data = dict(raw or {})
    return GridSpec(
        nx=max(int(data.get("nx", 200)), 1),
        ny=max(int(data.get("ny", 200)), 1),
        nz=max(int(data.get("nz", 200)), 1),
        boundary_resolution=max(int(data.get("boundary_resolution", 28)), 1),
        implicit_margin=float(data.get("implicit_margin", 0.0)),
        disable_enclosed_check=_as_bool(data.get("disable_enclosed_check"), True),
        enclosed_tolerance=float(data.get("enclosed_tolerance", 1.0e-6)),
        inside_combine_mode=str(data.get("inside_combine_mode", "and")).strip().lower(),
        chunk_points=max(
            int(
                data.get("chunk_points", data.get("tissue_grid_chunk_points", 250_000))
            ),
            1,
        ),
    )


def sample_grid_points(
    domain: Any, raw_spec: dict[str, Any] | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = grid_spec_from_dict(raw_spec)
    boundary = get_boundary(domain, spec.boundary_resolution)
    total_grid_points = int(spec.nx) * int(spec.ny) * int(spec.nz)
    if spec.disable_enclosed_check:
        points = inside_grid_points_chunked(domain, boundary, spec)
    else:
        grid = grid_points(boundary, spec.nx, spec.ny, spec.nz)
        inside = inside_mask(domain, boundary, grid, spec)
        points = grid[inside]
    meta = {
        "sample_mode": "grid",
        "nx": int(spec.nx),
        "ny": int(spec.ny),
        "nz": int(spec.nz),
        "total_grid_points": int(total_grid_points),
        "inside_grid_points": int(points.shape[0]),
        "implicit_margin": float(spec.implicit_margin),
        "disable_enclosed_check": bool(spec.disable_enclosed_check),
        "inside_combine_mode": str(spec.inside_combine_mode),
    }
    return np.asarray(points, dtype=float), meta


def get_boundary(domain: Any, boundary_resolution: int) -> pv.PolyData:
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


def grid_axes(
    boundary: pv.PolyData, nx: int, ny: int, nz: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bmin = np.min(boundary.points, axis=0).astype(np.float64)
    bmax = np.max(boundary.points, axis=0).astype(np.float64)
    x = np.linspace(bmin[0], bmax[0], int(nx), dtype=np.float64)
    y = np.linspace(bmin[1], bmax[1], int(ny), dtype=np.float64)
    z = np.linspace(bmin[2], bmax[2], int(nz), dtype=np.float64)
    return x, y, z


def grid_points_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def grid_points(boundary: pv.PolyData, nx: int, ny: int, nz: int) -> np.ndarray:
    return grid_points_from_axes(*grid_axes(boundary, nx, ny, nz))


def inside_mask(
    domain: Any, boundary: pv.PolyData, points: np.ndarray, spec: GridSpec
) -> np.ndarray:
    implicit = np.asarray(domain(points)).reshape(-1) <= -float(spec.implicit_margin)
    if spec.disable_enclosed_check:
        return implicit
    try:
        cloud = pv.PolyData(points.astype(np.float64))
        selected = cloud.select_enclosed_points(
            boundary,
            tolerance=float(spec.enclosed_tolerance),
            check_surface=False,
        )
        enclosed = (
            np.asarray(selected.point_data["SelectedPoints"]).astype(bool).reshape(-1)
        )
        if spec.inside_combine_mode == "or":
            return np.logical_or(implicit, enclosed)
        return np.logical_and(implicit, enclosed)
    except Exception as exc:
        print(
            f"Warning: enclosed-point check failed ({exc}); using implicit-only mask.",
            flush=True,
        )
        return implicit


def inside_grid_points_chunked(
    domain: Any, boundary: pv.PolyData, spec: GridSpec
) -> np.ndarray:
    x, y, z = grid_axes(boundary, spec.nx, spec.ny, spec.nz)
    chunk_points = max(int(spec.chunk_points), 1)
    yz_count = max(int(y.size * z.size), 1)
    x_step = max(1, min(int(x.size), chunk_points // yz_count))
    chunks = []
    for start in range(0, int(x.size), x_step):
        stop = min(start + x_step, int(x.size))
        chunk = grid_points_from_axes(x[start:stop], y, z)
        implicit = np.asarray(domain(chunk)).reshape(-1) <= -float(spec.implicit_margin)
        if np.any(implicit):
            chunks.append(chunk[implicit])
    if not chunks:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)
