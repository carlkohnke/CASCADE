"""Build bounded vessel-preview data in an isolated Studio subprocess.

CASCADE Studio invokes this internal worker with ``--request`` and ``--output``
paths. It constructs only the exact reusable seed needed for the active display
limit, or loads an uploaded network, then returns compact geometry to the GUI.
It is not intended as a user-facing command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from cascade.configuration.schema import load_config
from cascade.utils.resources import resolve_domain_path
from cascade.vessels.build import build_or_load_network
from cascade.vessels.cache import save_network_if_requested
from cascade.domain.svv import Domain
from cascade.domain.svv.routines.tetrahedralize import tetrahedralize


PREVIEW_VESSEL_LIMIT = 5_000
MAX_PREVIEW_VESSEL_LIMIT = 250_000


def _bounded_preview_target(
    target: int,
    tree_count: int = 1,
    vessel_limit: int = PREVIEW_VESSEL_LIMIT,
) -> int:
    """Build enough equal-sized trees to reach the requested display limit."""
    trees = max(int(tree_count), 1)
    displayed = min(max(int(vessel_limit), 1), MAX_PREVIEW_VESSEL_LIMIT)
    # Each binary tree contains approximately 2*T+1 segments after T terminal
    # additions.  Round up so the combined trees contain at least ``displayed``.
    per_tree_limit = max(math.ceil((displayed - trees) / (2 * trees)), 1)
    return min(max(int(target), 1), per_tree_limit)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Studio project JSON path.")
    parser.add_argument(
        "--output", required=True, help="Directory for preview geometry and metadata."
    )
    args = parser.parse_args(argv)
    request = Path(args.request).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = json.loads(request.read_text(encoding="utf-8"))
    network = raw.setdefault("network", {})
    source = str(raw.get("gui", {}).get("network_source", "svv_generated"))
    viewer = raw.get("gui", {}).get("viewer", {})
    vessel_mode = str(viewer.get("vessel_mode", "near")).lower()
    renderer_limit = min(
        max(
            int(viewer.get("renderer_vessel_limit", MAX_PREVIEW_VESSEL_LIMIT)),
            1,
        ),
        MAX_PREVIEW_VESSEL_LIMIT,
    )
    requested_vessel_limit = (
        renderer_limit
        if vessel_mode == "all"
        else int(viewer.get("vessel_limit", PREVIEW_VESSEL_LIMIT))
    )
    preview_vessel_limit = min(max(requested_vessel_limit, 1), renderer_limit)
    # The production default reserves four million rows and needlessly costs
    # ~1 GB here. Size this worker for the effective cross-tab display limit
    # plus the two rows touched while inserting its final bifurcation.
    os.environ.setdefault("SVV_TREE_PREALLOCATION_STEP", str(preview_vessel_limit + 2))
    if source == "svv_generated":
        roots = network.get("roots") or (
            [network["root"]] if network.get("root") else []
        )
        n_trees = max(len(roots), 1)
        final_target = int(network.get("target_terminal_count") or 1)
        sweep_targets = [
            int(value)
            for sweep in raw.get("gui", {}).get("sweeps", [])
            if sweep.get("enabled", True)
            and sweep.get("path") == "network.target_terminal_count"
            for value in sweep.get("values", [])
            if isinstance(value, (int, float)) and int(value) > 0
        ]
        if sweep_targets:
            final_target = min([final_target, *sweep_targets])
        # A binary SVV tree has approximately 2*T+1 segments for T terminal adds.
        # Construct enough of the reusable seed for the configured viewer limit;
        # the final simulation resumes growth from this same seed when necessary.
        preview_target = _bounded_preview_target(
            final_target,
            n_trees,
            preview_vessel_limit,
        )
        network["target_terminal_counts"] = [preview_target] * n_trees
        network.pop("target_total_terminal_count", None)
        network.pop("target_terminal_count", None)
    else:
        raw.setdefault("growth", {})["enabled"] = False
    raw.setdefault("simulation", {}).update(
        {"geometry_only": True, "skip_tissue_oxygen": True, "distance_sample_count": 0}
    )
    raw.setdefault("outputs", {}).update(
        {
            "out_dir": str(output),
            "prefix": "preview_seed",
            "save_network": source == "svv_generated",
            "write_paraview": False,
            "write_summary_csv": False,
            "write_segments_csv": False,
            "write_points_csv": False,
            "use_cache": False,
        }
    )
    prepared = output / "preview.settings.json"
    prepared.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    config = load_config(prepared)
    preview_domain = _fast_file_preview_domain(config)
    build = build_or_load_network(config, domain_override=preview_domain)
    if source == "svv_generated":
        seed_path = save_network_if_requested(build, config)
    elif source == "custom":
        seed_path = Path(
            str(network.get("simple", {}).get("path", ""))
        ).expanduser().resolve()
    else:
        seed_path = Path(str(network.get("input_path", ""))).expanduser().resolve()

    starts, ends, radii, alpha, tree_ids = [], [], [], [], []
    root_radii = []
    totals = [int(getattr(tree, "segment_count", 0) or 0) for tree in build.trees]
    allocations = _fair_allocations(totals, preview_vessel_limit)
    for tree_id, (tree, take) in enumerate(
        zip(build.trees, allocations, strict=True)
    ):
        total = int(getattr(tree, "segment_count", 0) or 0)
        n = min(total, int(take))
        data = np.asarray(tree.data[:n], dtype=float)
        if not n:
            continue
        starts.append(data[:, 0:3])
        ends.append(data[:, 3:6])
        radii.append(data[:, 21])
        root_radii.append(float(data[0, 21]))
        preview_truncated = n < total or (
            source == "svv_generated" and preview_target < final_target
        )
        if preview_truncated:
            parents = np.nan_to_num(data[:, 17], nan=-1.0).astype(np.int64)
            depth = np.zeros(n, dtype=float)
            for segment_id in range(n):
                parent_id = int(parents[segment_id])
                if 0 <= parent_id < segment_id:
                    depth[segment_id] = depth[parent_id] + 1.0
            depth /= max(float(np.max(depth)), 1.0)
            # A fading distal edge indicates that the preview ends before the
            # complete tree; fully represented trees remain uniformly opaque.
            alpha.append(np.clip(1.0 - 0.65 * depth**1.8, 0.35, 1.0))
        else:
            alpha.append(np.ones(n, dtype=float))
        tree_ids.append(np.full(n, tree_id, dtype=np.int32))
    geometry_path = output / "preview.geometry.npz"
    np.savez_compressed(
        geometry_path,
        starts=np.concatenate(starts, axis=0) if starts else np.empty((0, 3)),
        ends=np.concatenate(ends, axis=0) if ends else np.empty((0, 3)),
        radii=np.concatenate(radii) if radii else np.empty((0,)),
        alpha=np.concatenate(alpha) if alpha else np.empty((0,)),
        tree_ids=np.concatenate(tree_ids)
        if tree_ids
        else np.empty((0,), dtype=np.int32),
    )
    response = {
        "seed_path": str(seed_path) if seed_path else "",
        "geometry_path": str(geometry_path),
        "source": source,
        "trees": len(build.trees),
        "segments": totals,
        "shown_segments": allocations,
        "requested_segments": (
            [max(2 * final_target + 1, 0)] * n_trees
            if source == "svv_generated"
            else totals
        ),
        "requested_growth_count": final_target
        if source == "svv_generated"
        else None,
        "preview_growth_count": preview_target
        if source == "svv_generated"
        else None,
        "requested_terminal_leaves": final_target + 1
        if source == "svv_generated"
        else None,
        "preview_terminal_leaves": preview_target + 1
        if source == "svv_generated"
        else None,
        # Retain the original response keys for older Studio consumers. Their
        # values are growth counts, as the explicit keys above now make clear.
        "requested_terminal_target": final_target
        if source == "svv_generated"
        else None,
        "preview_terminal_target": preview_target
        if source == "svv_generated"
        else None,
        "preview_vessel_limit": preview_vessel_limit
        if source == "svv_generated"
        else None,
        "preview_limited": bool(
            source == "svv_generated" and preview_target < final_target
        ),
        "root_radii_cm": root_radii,
        "domain_surface_repaired": bool(
            getattr(build.domain, "surface_repaired_for_tetgen", False)
        ),
    }
    (output / "response.json").write_text(
        json.dumps(response, indent=2), encoding="utf-8"
    )
    print(json.dumps(response), flush=True)
    return 0


class _MeshPreviewDomain(Domain):
    """Exact mesh-backed membership test without fitting thousands of RBF patches."""

    def __call__(self, points, **_kwargs):
        values = np.asarray(points, dtype=float)
        original_shape = values.shape[:-1]
        values = values.reshape(-1, values.shape[-1])
        cells = np.asarray(self.mesh.find_containing_cell(values), dtype=np.int64)
        # SVV expects a negative implicit value inside and a positive value outside.
        return np.where(cells >= 0, -0.5, 0.5).reshape(original_shape)


def _fast_file_preview_domain(config):
    """Prepare uploaded surfaces for preview growth in seconds, not minutes."""
    domain_config = config.domain
    kind = str(domain_config.kind).strip().lower()
    if kind in {
        "cube",
        "box",
        "rectangular",
        "rectangular_box",
        "sphere",
        "pv.sphere",
        "pyvista_sphere",
        "cylinder",
        "disk",
        "pv.cylinder",
        "pyvista_cylinder",
    }:
        return None
    path = resolve_domain_path(
        domain_config.path,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if path is None or path.suffix.lower() == ".dmn":
        return None

    import pymeshfix

    surface = pv.read(str(path)).extract_surface().triangulate().clean()
    fixer = pymeshfix.MeshFix(surface)
    fixer.repair(
        verbose=False,
        joincomp=True,
        remove_smallest_components=False,
    )
    surface = fixer.mesh.extract_surface().triangulate().clean()
    mesh, nodes, vertices = tetrahedralize(
        surface,
        order=1,
        nobisect=False,
        verbose=False,
    )
    mesh = mesh.compute_cell_sizes()
    volume = np.asarray(mesh.cell_data["Volume"], dtype=float)
    total_volume = float(np.sum(volume))
    if total_volume <= 0.0:
        raise ValueError("The uploaded domain has no positive tetrahedral volume.")
    probability = volume / total_volume
    mesh.cell_data["Normalized_Volume"] = probability
    mesh.cell_data["probability"] = probability.copy()

    boundary = surface.compute_cell_sizes()
    area = np.asarray(boundary.cell_data["Area"], dtype=float)
    total_area = float(np.sum(area))
    if total_area > 0.0:
        boundary.cell_data["Normalized_Area"] = area / total_area

    domain = _MeshPreviewDomain(np.asarray(surface.points, dtype=np.float64))
    domain.original_boundary = surface
    domain.boundary = boundary
    domain.boundary_nodes = np.asarray(boundary.points, dtype=np.float64)
    domain.boundary_vertices = np.asarray(boundary.faces, dtype=np.int64).reshape(
        -1, 4
    )[:, 1:]
    domain.mesh = mesh
    domain.mesh_nodes = np.asarray(nodes, dtype=np.float64)
    domain.mesh_vertices = np.asarray(vertices, dtype=np.int64)
    centers = mesh.cell_centers().points[:, :3]
    domain.mesh_tree = cKDTree(centers, leafsize=4)
    domain.mesh_tree_2 = BallTree(centers)
    domain.all_mesh_cells = np.arange(mesh.n_cells, dtype=np.int64)
    domain.cumulative_probability = np.cumsum(probability)
    domain.characteristic_length = total_volume ** (1.0 / 3.0)
    domain.area = float(boundary.area)
    domain.volume = total_volume
    # Uploaded anatomical surfaces should use SVV's non-convex growth path.
    domain.convexity = 0.0
    domain.random_seed = int(domain_config.random_seed)
    domain.set_random_generator()
    domain.surface_repaired_for_tetgen = True
    print(
        f"Preview domain ready: direct mesh path · {mesh.n_cells:,} tetrahedra",
        flush=True,
    )
    return domain


def _fair_allocations(counts: list[int], limit: int) -> list[int]:
    if sum(counts) <= limit:
        return list(counts)
    n = max(len(counts), 1)
    allocations = [min(count, limit // n) for count in counts]
    remaining = limit - sum(allocations)
    while remaining > 0:
        progressed = False
        for index, count in enumerate(counts):
            if allocations[index] < count and remaining > 0:
                allocations[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocations


if __name__ == "__main__":
    raise SystemExit(main())
