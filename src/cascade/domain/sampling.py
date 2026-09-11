"""Domain sampling and vessel-distance calculations.

The distance paths select exact or KD-tree calculations from explicit inputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from cascade.configuration import _legacy_state as _state
from cascade.utils.hashing import file_sha256

try:
    from scipy.spatial import cKDTree as _cKDTree
except ImportError:  # pragma: no cover - the exact path remains available
    _cKDTree = None

def sample_domain_points(domain: _state.Domain, n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.empty((0, domain.points.shape[1]))
    pts, _ = domain.get_interior_points(n_points, method="implicit_only", convex=True)
    return np.asarray(pts, dtype=float)


def load_sample_points(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a finite N-by-3 coordinate fixture and its provenance metadata."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Tissue sample file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        points = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as payload:
            key = next(
                (name for name in ("points", "sample_points") if name in payload),
                None,
            )
            if key is None:
                raise ValueError(
                    "NPZ tissue sample file must contain 'points' or 'sample_points'."
                )
            points = np.asarray(payload[key])
    elif suffix == ".csv":
        table = np.genfromtxt(
            source,
            delimiter=",",
            names=True,
            dtype=float,
            encoding="utf-8-sig",
        )
        if table.dtype.names is None:
            raise ValueError("CSV tissue sample file must have x,y,z header columns.")
        names = {name.strip().lower(): name for name in table.dtype.names}
        if not all(axis in names for axis in ("x", "y", "z")):
            raise ValueError(
                "CSV tissue sample file must have x,y,z header columns in centimetres."
            )
        table = np.atleast_1d(table)
        points = np.column_stack([table[names[axis]] for axis in ("x", "y", "z")])
    else:
        raise ValueError("Tissue sample file must be CSV, NPY, or NPZ.")

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Tissue sample coordinates must have shape (N, 3); got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Tissue sample coordinates contain NaN or infinite values.")
    metadata = {
        "sample_mode": "file",
        "points": int(points.shape[0]),
        "path": str(source),
        "sha256": file_sha256(source),
        "coordinate_units": "cm",
    }
    return points, metadata


def has_dlp_feasible_parent(tree: _state.Tree, point: np.ndarray, k: int = 10) -> bool:
    """
    Fast check: is there at least one nearby parent vessel that could satisfy DLP constraints
    to grow toward the given point?
    """
    if tree is None or tree.segment_count <= 0:
        return False
    parms = getattr(tree, "parameters", None)
    if not (parms and getattr(parms, "dlp_enable", False) and getattr(parms, "dlp_build_dir", None) is not None):
        return True
    bdir = np.asarray(getattr(parms, "dlp_build_dir", None), dtype=float)
    n = float(np.linalg.norm(bdir))
    if n <= 0.0:
        return False
    b_hat = bdir / n
    s_min = float(getattr(parms, "dlp_min_adv", 0.0))
    sin_theta_min = math.sin(math.radians(float(getattr(parms, "dlp_min_angle_deg", 0.0))))
    data = tree.data[:tree.segment_count, :]
    query_k = max(1, min(int(k), data.shape[0]))
    try:
        _, idx = tree.hnsw_tree.query(np.asarray(point, dtype=float).reshape(1, 3), k=query_k)
    except Exception:
        return False
    idx = np.atleast_1d(idx).reshape(-1)
    for ii in idx:
        parent_prox = data[int(ii), 0:3]
        u = np.asarray(point, dtype=float) - parent_prox
        adv = float(np.dot(u, b_hat))
        if adv < s_min:
            continue
        L = float(np.linalg.norm(u))
        if L <= 1e-12:
            continue
        if abs(adv) / L < sin_theta_min:
            continue
        return True
    return False


def _characteristic_length(tree: _state.Tree, fallback_side_length: float | None = None) -> float:
    domain = getattr(tree, "domain", None)
    if domain is not None:
        char_len = getattr(domain, "characteristic_length", None)
        if char_len is not None and np.isfinite(char_len):
            return float(char_len)
        bounds = getattr(domain, "bounds", None) or getattr(domain, "bounding_box", None)
        if bounds is not None and len(bounds) >= 6:
            span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            return float(span)
    if fallback_side_length is not None and np.isfinite(fallback_side_length):
        return float(fallback_side_length)
    return 1.0


def compute_average_distance(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
    if points.size == 0 or starts.size == 0:
        return float("nan")

    points = np.asarray(points, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)

    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    seg_len_sq[seg_len_sq <= 0] = 1e-12

    use_kdtree = (
        _state._HAVE_SCIPY_SPATIAL
        and _state.DISTANCE_USE_KDTREE_FOR_LARGE_TREES
        and starts.shape[0] > _state.DISTANCE_BRUTE_FORCE_MAX_SEGMENTS
    )

    min_dists: list[np.ndarray] = []
    if use_kdtree:
        midpoints = 0.5 * (starts + ends)
        kdtree = _cKDTree(midpoints)
        candidate_k = min(
            starts.shape[0],
            max(
                int(_state.DISTANCE_KDTREE_MIN_CANDIDATES),
                min(
                    int(_state.DISTANCE_KDTREE_MAX_CANDIDATES),
                    int(_state.DISTANCE_KDTREE_MIN_CANDIDATES * _state.DISTANCE_KDTREE_CANDIDATE_MULT),
                ),
            ),
        )
        for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + _state.DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            _, cand = kdtree.query(chunk, k=candidate_k)
            if cand.ndim == 1:
                cand = cand[:, None]
            starts_c = starts[cand]
            seg_c = segment_vectors[cand]
            seg_len_sq_c = seg_len_sq[cand]
            diff = chunk[:, None, :] - starts_c
            proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts_c + proj[:, :, None] * seg_c
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            min_dists.append(np.min(distances, axis=1))
    else:
        for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + _state.DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            diff = chunk[:, None, :] - starts[None, :, :]
            proj = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            min_dists.append(np.min(distances, axis=1))

    if not min_dists:
        return float("nan")
    all_dists = np.concatenate(min_dists)
    return float(np.mean(all_dists))


def compute_distance_to_nearest_channel(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray | None = None,
) -> np.ndarray:
    if points.size == 0 or starts.size == 0:
        return np.empty((0,), dtype=float)

    points = np.asarray(points, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if radii is None:
        radii = np.zeros((starts.shape[0],), dtype=float)
    radii = np.asarray(radii, dtype=float).reshape(-1)
    if radii.size != starts.shape[0]:
        raise ValueError("radii must have the same length as starts/ends")
    radii = np.maximum(radii, 0.0)

    segment_vectors = ends - starts
    seg_len_sq = np.sum(segment_vectors**2, axis=1)
    seg_len_sq[seg_len_sq <= 0] = 1e-12

    use_kdtree = (
        _state._HAVE_SCIPY_SPATIAL
        and _state.DISTANCE_USE_KDTREE_FOR_LARGE_TREES
        and starts.shape[0] > _state.DISTANCE_BRUTE_FORCE_MAX_SEGMENTS
    )

    min_dists: list[np.ndarray] = []
    if use_kdtree:
        midpoints = 0.5 * (starts + ends)
        kdtree = _cKDTree(midpoints)
        candidate_k = min(
            starts.shape[0],
            max(
                int(_state.DISTANCE_KDTREE_MIN_CANDIDATES),
                min(
                    int(_state.DISTANCE_KDTREE_MAX_CANDIDATES),
                    int(_state.DISTANCE_KDTREE_MIN_CANDIDATES * _state.DISTANCE_KDTREE_CANDIDATE_MULT),
                ),
            ),
        )
        for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + _state.DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            _, cand = kdtree.query(chunk, k=candidate_k)
            if cand.ndim == 1:
                cand = cand[:, None]
            starts_c = starts[cand]
            seg_c = segment_vectors[cand]
            seg_len_sq_c = seg_len_sq[cand]
            radii_c = radii[cand]
            diff = chunk[:, None, :] - starts_c
            proj = np.sum(diff * seg_c, axis=2) / seg_len_sq_c
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts_c + proj[:, :, None] * seg_c
            d_center = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            d_wall = np.maximum(d_center - radii_c, 0.0)
            min_dists.append(np.min(d_wall, axis=1))
    else:
        for idx in range(0, len(points), _state.DISTANCE_CHUNK_SIZE):
            chunk = points[idx: idx + _state.DISTANCE_CHUNK_SIZE]
            if chunk.size == 0:
                continue
            diff = chunk[:, None, :] - starts[None, :, :]
            proj = np.sum(diff * segment_vectors[None, :, :], axis=2) / seg_len_sq[None, :]
            proj = np.clip(proj, 0.0, 1.0)
            closest = starts[None, :, :] + proj[:, :, None] * segment_vectors[None, :, :]
            distances = np.linalg.norm(chunk[:, None, :] - closest, axis=2)
            nearest_idx = np.argmin(distances, axis=1)
            row_idx = np.arange(distances.shape[0], dtype=int)
            d_center = distances[row_idx, nearest_idx]
            r_local = radii[nearest_idx]
            d_wall = np.maximum(d_center - r_local, 0.0)
            min_dists.append(d_wall)

    if not min_dists:
        return np.empty((0,), dtype=float)
    return np.concatenate(min_dists)


__all__ = [
    "sample_domain_points",
    "load_sample_points",
    "has_dlp_feasible_parent",
    "_characteristic_length",
    "compute_average_distance",
    "compute_distance_to_nearest_channel",
]
