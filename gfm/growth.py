from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pyvista as pv
from svv.tree.collision.tree_collision import tree_collision
from svv.tree.data.data import TreeMap
from svv.tree.utils.TreeManager import KDTreeManager, USearchTree

from .connectivity import repair_trees, validate_trees
from .config import RunConfig, RootConfig
from . import _svv_branch_bifurcation as gfm_bifurcation
from .grid import sample_grid_points
from .simple import build_simple_network
from .svv_adapter import Domain, Forest, Tree


GFM_ROOT = Path(__file__).resolve().parents[1]
if str(GFM_ROOT) not in sys.path:
    sys.path.insert(0, str(GFM_ROOT))


@dataclass
class NetworkBuildResult:
    domain: Any
    trees: list[Any]
    forest: Any | None
    target_counts: list[int]
    build_timings: dict[str, float]
    sample_points: np.ndarray | None = None
    sample_meta: dict[str, Any] | None = None
    connectivity_repairs: list[int] | None = None
    connectivity_reports: list[dict[str, Any]] | None = None
    network_path: Path | None = None
    cache_path: Path | None = None
    load_source: Path | None = None


def load_tissuesim_module():
    import TissueSim_cube_local as ts

    return ts


def resolve_path(value: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def apply_tissuesim_settings(ts, config: RunConfig) -> None:
    sim = config.simulation
    growth = config.growth
    ts.FLUID = sim.fluid
    ts.ACTIVE_FLUID = sim.fluid
    ts.BUILD_FLUID = sim.build_fluid
    ts.QIN_TARGET = float(sim.qin_target_ul_min)
    ts.DISTANCE_SAMPLE_COUNT = int(sim.distance_sample_count)
    ts.COMPUTE_AVG_DISTANCE_TO_CHANNEL = bool(sim.compute_avg_distance_to_channel)
    ts.CONCENTRATION_SOLVER = str(sim.concentration_solver)
    ts.N_EQUAL_BIFURCATIONS = growth.n_equal_bifurcations
    ts.TREE_DATA_DTYPE = np.float64
    ts.TREE_INDEX_DTYPE = np.int64

    if sim.tissue_accel is not None and hasattr(ts, "TISSUE_ACCEL_MODE"):
        ts.TISSUE_ACCEL_MODE = str(sim.tissue_accel).strip().lower()
    if sim.tissue_gpu_validate_points is not None and hasattr(ts, "TISSUE_GPU_VALIDATE_POINTS"):
        ts.TISSUE_GPU_VALIDATE_POINTS = int(sim.tissue_gpu_validate_points)

    for key, value in sim.tissuesim.items():
        const_name = str(key).strip().upper()
        if hasattr(ts, const_name):
            setattr(ts, const_name, value)

    for key, value in growth.equal_terminal.items():
        const_name = str(key).strip().upper()
        if not const_name.startswith("EQUAL_"):
            const_name = f"EQUAL_TERMINAL_{const_name}"
        if hasattr(ts, const_name):
            setattr(ts, const_name, value)

    for key, value in sim.cext.items():
        const_name = str(key).strip().upper()
        if not const_name.startswith("CEXT_"):
            const_name = f"CEXT_{const_name}"
        if hasattr(ts, const_name):
            setattr(ts, const_name, value)


def build_domain(config: RunConfig, ts=None):
    ts = ts or load_tissuesim_module()
    domain_cfg = config.domain
    np.random.seed(int(domain_cfg.random_seed))
    kind = str(domain_cfg.kind).strip().lower()
    if kind == "cube":
        domain = ts.build_domain(float(domain_cfg.side_length))
        domain.random_seed = int(domain_cfg.random_seed)
        return domain
    if kind in {"box", "rectangular", "rectangular_box"}:
        x_len = float(domain_cfg.x_length if domain_cfg.x_length is not None else domain_cfg.side_length)
        y_len = float(domain_cfg.y_length if domain_cfg.y_length is not None else domain_cfg.side_length)
        z_len = float(domain_cfg.z_length if domain_cfg.z_length is not None else domain_cfg.side_length)
        domain = Domain(pv.Cube(x_length=x_len, y_length=y_len, z_length=z_len))
        domain.random_seed = int(domain_cfg.random_seed)
        domain.create()
        domain.solve()
        domain.build()
        domain.set_random_generator()
        return domain

    domain_path = resolve_path(domain_cfg.path, base_dir=config.settings_path.parent if config.settings_path else None)
    if domain_path is None:
        raise ValueError("domain.path is required for non-cube domains.")
    suffix = domain_path.suffix.lower()
    if suffix == ".dmn":
        domain = Domain.load(str(domain_path))
        domain.random_seed = int(domain_cfg.random_seed)
        return domain

    mesh = pv.read(str(domain_path))
    domain = Domain(mesh)
    domain.random_seed = int(domain_cfg.random_seed)
    domain.create()
    domain.solve()
    domain.build()
    domain.set_random_generator()
    return domain


def build_or_load_network(config: RunConfig) -> NetworkBuildResult:
    ts = load_tissuesim_module()
    apply_tissuesim_settings(ts, config)
    timings: dict[str, float] = {}

    t0 = perf_counter()
    domain = build_domain(config, ts=ts)
    timings["domain_s"] = perf_counter() - t0

    t0 = perf_counter()
    sample_points, sample_meta = _pre_sample_points(ts, domain, config)
    timings["sample_points_s"] = perf_counter() - t0

    input_path = resolve_path(
        config.network.input_path,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if input_path is not None:
        checkpoint_path = resolve_path(
            config.growth.checkpoint_path,
            base_dir=config.settings_path.parent if config.settings_path else None,
        )
        if bool(config.growth.resume_from_checkpoint) and checkpoint_path is not None and checkpoint_path.exists():
            input_path = checkpoint_path
        t0 = perf_counter()
        trees, forest = _load_existing_network(input_path, domain, config)
        timings["network_load_s"] = perf_counter() - t0
        target_counts = _target_counts_for_config(config, len(trees), trees=trees)
        if config.growth.enabled:
            t0 = perf_counter()
            _extend_trees_to_targets(ts, trees, domain, config, target_counts, forest=forest)
            timings["growth_s"] = perf_counter() - t0
        repairs, reports = _repair_and_validate_if_requested(trees, config)
        return NetworkBuildResult(
            domain=domain,
            trees=trees,
            forest=forest,
            target_counts=target_counts,
            build_timings=timings,
            sample_points=sample_points,
            sample_meta=sample_meta,
            connectivity_repairs=repairs,
            connectivity_reports=reports,
            load_source=input_path,
        )

    if config.network_mode == "simple":
        t0 = perf_counter()
        simple_network = build_simple_network(ts, domain, config)
        timings["growth_s"] = perf_counter() - t0
        return NetworkBuildResult(
            domain=domain,
            trees=[simple_network],
            forest=None,
            target_counts=[int(getattr(simple_network, "n_terminals", 1) or 1)],
            build_timings=timings,
            sample_points=sample_points,
            sample_meta=sample_meta,
        )

    t0 = perf_counter()
    trees = _build_configured_trees(ts, domain, config)
    timings["growth_s"] = perf_counter() - t0
    forest = _make_forest(config, domain, trees) if config.network_mode == "forest" else None
    target_counts = _target_counts_for_config(config, len(trees), trees=trees)
    repairs, reports = _repair_and_validate_if_requested(trees, config)
    return NetworkBuildResult(
        domain=domain,
        trees=trees,
        forest=forest,
        target_counts=target_counts,
        build_timings=timings,
        sample_points=sample_points,
        sample_meta=sample_meta,
        connectivity_repairs=repairs,
        connectivity_reports=reports,
    )


def save_network_if_requested(result: NetworkBuildResult, config: RunConfig) -> Path | None:
    if not config.outputs.save_network:
        return None
    out_dir = resolve_path(config.outputs.out_dir, base_dir=config.settings_path.parent if config.settings_path else None)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = resolve_path(config.network.save_path, base_dir=out_dir) if config.network.save_path else None
    if result.forest is not None:
        path = save_path or (out_dir / f"{config.prefix}.forest")
        saved = Path(result.forest.save(str(path))).resolve()
        if config.outputs.use_cache:
            cache_path = _forest_cache_path(saved, config)
            result.forest.save_simulation_cache(str(cache_path))
            result.cache_path = cache_path.resolve()
    elif result.trees and getattr(result.trees[0], "_gfm_simple_network", False):
        path = save_path or (out_dir / f"{config.prefix}.simple.npz")
        simple = result.trees[0]
        np.savez_compressed(
            path,
            starts=np.asarray(simple.starts, dtype=np.float64),
            ends=np.asarray(simple.ends, dtype=np.float64),
            radii=np.asarray(simple.radii, dtype=np.float64),
            lengths=np.asarray(simple.lengths, dtype=np.float64),
            flows=np.asarray(simple.flows, dtype=np.float64),
            cin=np.asarray(simple.cin, dtype=np.float64),
            cout=np.asarray(simple.cout, dtype=np.float64),
            metadata=json.dumps(simple.metadata),
        )
        saved = path.resolve()
    else:
        path = save_path or (out_dir / f"{config.prefix}.tree.npz")
        saved = Path(result.trees[0].save(str(path), include_domain=False)).resolve()
    result.network_path = saved
    return saved


def _load_existing_network(path: Path, domain, config: RunConfig) -> tuple[list[Any], Any | None]:
    suffix = path.suffix.lower()
    if Forest._is_simulation_cache(str(path)):
        if config.growth.enabled:
            raise ValueError("Direct .forest simulation-cache inputs require growth.enabled=false.")
        forest = Forest.load(str(path), mode="simulation")
        _attach_domain(forest, domain, simulation_only=True)
        trees = [tree for network in forest.networks for tree in network]
        return trees, forest

    if suffix == ".forest" or path.name.endswith(".forest"):
        mode = "simulation" if not config.growth.enabled else "growth"
        load_path = _forest_load_path(path, config, mode=mode)
        forest = Forest.load(str(load_path), mode=mode)
        _attach_domain(forest, domain, simulation_only=(mode == "simulation"))
        trees = [tree for network in forest.networks for tree in network]
        return trees, forest

    tree = Tree.load(str(path), domain=domain, data_dtype=np.float64, index_dtype=np.int64)
    _attach_tree_domain(tree, domain)
    return [tree], None


def _attach_domain(forest, domain, *, simulation_only: bool) -> None:
    if simulation_only and hasattr(forest, "attach_domain_for_simulation"):
        forest.attach_domain_for_simulation(domain)
        return
    try:
        forest.set_domain(domain)
    except Exception:
        forest.domain = domain
        forest.geodesic = None
        for network in forest.networks:
            for tree in network:
                _attach_tree_domain(tree, domain)


def _attach_tree_domain(tree, domain) -> None:
    try:
        tree.set_domain(domain)
    except Exception:
        tree.domain = domain


def _forest_cache_path(path: Path, config: RunConfig) -> Path:
    explicit = resolve_path(
        config.outputs.cache_path,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if explicit is not None:
        return explicit
    return path.with_name(path.name + ".simcache")


def _forest_load_path(path: Path, config: RunConfig, *, mode: str) -> Path:
    if mode != "simulation" or not config.outputs.use_cache:
        return path
    if Forest._is_simulation_cache(str(path)):
        return path
    cache_path = _forest_cache_path(path, config)
    if (
        cache_path.exists()
        and Forest._is_simulation_cache(str(cache_path))
        and cache_path.stat().st_mtime_ns >= path.stat().st_mtime_ns
    ):
        return cache_path
    try:
        Forest.build_simulation_cache_from_legacy(str(path), str(cache_path), show_progress=True)
        return cache_path
    except Exception as exc:
        print(f"Warning: failed to build forest simulation cache ({exc}); loading forest directly.", flush=True)
        return path


def _build_configured_trees(ts, domain, config: RunConfig) -> list[Any]:
    targets = _target_counts_for_config(config, len(config.network.roots))
    trees = []
    for idx, (root, target) in enumerate(zip(config.network.roots, targets)):
        _set_tissuesim_root(ts, root, side_length=float(config.domain.side_length))
        qin_cm3_s = _flow_for_tree(config, idx, len(targets))
        terminal_flow = _terminal_flow_for_target(config, qin_cm3_s, int(target))
        tree = ts.grow_tree(
            domain,
            max(int(target), 1),
            dlp_enable=False,
            min_theta=0.0,
            side_length=float(config.domain.side_length),
            terminal_flow_override=terminal_flow,
            fluid=config.simulation.build_fluid,
            n_closest_vessels=int(config.growth.n_closest_vessels),
            n_points=int(config.growth.n_points),
            weighted_sampling=bool(config.growth.weighted_sampling),
            ignore_collisions=bool(config.growth.ignore_collisions),
            allow_inside_vessels=bool(config.growth.allow_inside_vessels),
        )
        trees.append(tree)
    return trees


def _pre_sample_points(ts, domain, config: RunConfig) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if config.simulation.geometry_only or config.simulation.skip_tissue_oxygen:
        return np.empty((0, 3), dtype=float), {"sample_mode": "none"}
    if config.simulation.sample_mode == "grid":
        return sample_grid_points(domain, config.simulation.tissue_grid)
    n_points = int(config.simulation.distance_sample_count)
    if n_points <= 0:
        return np.empty((0, 3), dtype=float), {"sample_mode": "random", "requested_points": 0}
    return (
        np.asarray(ts.sample_domain_points(domain, n_points), dtype=float),
        {"sample_mode": "random", "requested_points": int(n_points)},
    )


def _extend_trees_to_targets(ts, trees: list[Any], domain, config: RunConfig, targets: list[int], *, forest=None) -> None:
    if config.growth.assignment == "nearest-tree":
        _extend_trees_nearest(ts, trees, domain, config, targets, forest=forest)
        return
    if config.growth.assignment == "scheduled" or config.growth.bulk_growth_mode == "never":
        _extend_trees_scheduled(ts, trees, domain, config, targets, forest=forest)
        return
    for idx, (tree, target) in enumerate(zip(trees, targets)):
        _attach_tree_domain(tree, domain)
        current = max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 0)
        target = max(int(target), 1)
        qin_cm3_s = _flow_for_tree(config, idx, len(trees))
        terminal_flow = _terminal_flow_for_target(config, qin_cm3_s, target)
        ts._sync_loaded_tree_params(
            tree,
            side_length=float(config.domain.side_length),
            terminal_flow_override=terminal_flow,
        )
        if target <= current:
            continue
        tree.n_add(
            target - current,
            n_closest_vessels=int(config.growth.n_closest_vessels),
            n_points=int(config.growth.n_points),
            use_random_int=not bool(config.growth.weighted_sampling),
            ignore_collisions=bool(config.growth.ignore_collisions),
            allow_inside_vessels=bool(config.growth.allow_inside_vessels),
            debug_add_vessel=getattr(ts, "DEBUG_ADD_VESSEL", False),
        )


def _extend_trees_scheduled(ts, trees: list[Any], domain, config: RunConfig, targets: list[int], *, forest=None) -> None:
    remaining = [max(0, int(target) - max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 0)) for tree, target in zip(trees, targets)]
    total_add = int(sum(remaining))
    if total_add <= 0:
        return
    checkpoint_every = max(int(config.growth.checkpoint_every_adds), 0)
    completed = 0
    saved_targets: set[int] = set()
    for tree in trees:
        _attach_tree_domain(tree, domain)
    while any(rem > 0 for rem in remaining):
        for idx, tree in enumerate(trees):
            if remaining[idx] <= 0:
                continue
            target = max(int(targets[idx]), 1)
            qin_cm3_s = _flow_for_tree(config, idx, len(trees))
            terminal_flow = _terminal_flow_for_target(config, qin_cm3_s, target)
            ts._sync_loaded_tree_params(
                tree,
                side_length=float(config.domain.side_length),
                terminal_flow_override=terminal_flow,
            )
            tree.n_add(
                1,
                n_closest_vessels=int(config.growth.n_closest_vessels),
                n_points=int(config.growth.n_points),
                use_random_int=not bool(config.growth.weighted_sampling),
                ignore_collisions=bool(config.growth.ignore_collisions),
                allow_inside_vessels=bool(config.growth.allow_inside_vessels),
                debug_add_vessel=getattr(ts, "DEBUG_ADD_VESSEL", False),
            )
            remaining[idx] -= 1
            completed += 1
            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                _save_growth_checkpoint(config, trees, forest, completed=completed, total_add=total_add, targets=targets)
            saved_targets = _save_reached_targets(config, trees, forest, saved_targets, targets)
            if not any(rem > 0 for rem in remaining):
                break
    _save_growth_checkpoint(config, trees, forest, completed=completed, total_add=total_add, targets=targets, final=True)


def _extend_trees_nearest(ts, trees: list[Any], domain, config: RunConfig, targets: list[int], *, forest=None) -> None:
    if len(trees) <= 1:
        _extend_trees_scheduled(ts, trees, domain, config, targets, forest=forest)
        return

    remaining = [max(0, int(target) - max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 0)) for tree, target in zip(trees, targets)]
    total_add = int(sum(remaining))
    if total_add <= 0:
        return

    for tree in trees:
        _attach_tree_domain(tree, domain)
        _prepare_loaded_tree_for_incremental_growth(tree)

    completed = 0
    saved_targets: set[int] = set()
    checkpoint_every = max(int(config.growth.checkpoint_every_adds), 0)

    while any(rem > 0 for rem in remaining):
        ignore_now = _ignore_intertree_collisions_now(config, trees)
        if _nearest_bulk_growth_allowed(config, trees, remaining, ignore_now=ignore_now):
            _save_growth_checkpoint(
                config,
                trees,
                forest,
                completed=completed,
                total_add=total_add,
                targets=targets,
            )
            added = _grow_remaining_bulk_no_collision(ts, trees, domain, config, targets, remaining)
            completed += int(added)
            remaining = [0 for _ in remaining]
            saved_targets = _save_reached_targets(config, trees, forest, saved_targets, targets)
            break

        try:
            tree_idx, elapsed = _grow_one_nearest_tree(
                ts,
                trees,
                domain,
                config,
                targets,
                remaining,
                ignore_intertree_collisions=ignore_now,
            )
        except RuntimeError:
            if str(config.growth.collision_failure_mode).strip().lower() != "switch":
                _save_growth_checkpoint(
                    config,
                    trees,
                    forest,
                    completed=completed,
                    total_add=total_add,
                    targets=targets,
                    final=True,
                )
                raise
            live_remaining = [
                max(0, int(target) - max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 0))
                for tree, target in zip(trees, targets)
            ]
            added = _grow_remaining_bulk_no_collision(ts, trees, domain, config, targets, live_remaining)
            completed += int(added)
            remaining = [0 for _ in remaining]
            break

        remaining[int(tree_idx)] = max(0, int(remaining[int(tree_idx)]) - 1)
        completed += 1
        report_every = int(config.growth.growth_report_every)
        if report_every > 0 and (completed == 1 or completed % report_every == 0):
            print(
                "Nearest-tree growth: "
                f"add={completed}/{total_add} tree={tree_idx} remaining={remaining} last_s={elapsed:.3f}",
                flush=True,
            )
        if checkpoint_every > 0 and completed % checkpoint_every == 0:
            _save_growth_checkpoint(config, trees, forest, completed=completed, total_add=total_add, targets=targets)
        saved_targets = _save_reached_targets(config, trees, forest, saved_targets, targets)

    _save_growth_checkpoint(config, trees, forest, completed=completed, total_add=total_add, targets=targets, final=True)


def _grow_remaining_bulk_no_collision(
    ts,
    trees: list[Any],
    domain,
    config: RunConfig,
    targets: list[int],
    remaining: list[int],
) -> int:
    completed = 0
    for idx, (tree, add_n) in enumerate(zip(trees, remaining)):
        add_n = int(add_n)
        if add_n <= 0:
            continue
        _attach_tree_domain(tree, domain)
        qin_cm3_s = _flow_for_tree(config, idx, len(trees))
        terminal_flow = _terminal_flow_for_target(config, qin_cm3_s, int(targets[idx]))
        ts._sync_loaded_tree_params(
            tree,
            side_length=float(config.domain.side_length),
            terminal_flow_override=terminal_flow,
        )
        _prepare_loaded_tree_for_incremental_growth(tree)
        tree.n_add(
            add_n,
            n_closest_vessels=int(config.growth.n_closest_vessels),
            n_points=int(config.growth.n_points),
            use_random_int=not bool(config.growth.weighted_sampling),
            ignore_collisions=True,
            allow_inside_vessels=bool(config.growth.allow_inside_vessels),
            debug_add_vessel=getattr(ts, "DEBUG_ADD_VESSEL", False),
        )
        completed += add_n
    return completed


def _grow_one_nearest_tree(
    ts,
    trees: list[Any],
    domain,
    config: RunConfig,
    targets: list[int],
    remaining: list[int],
    *,
    ignore_intertree_collisions: bool,
) -> tuple[int, float]:
    max_attempts = max(1, int(config.growth.collision_retry_limit))
    last_error: Exception | None = None
    for _attempt in range(max_attempts):
        tree_idx, point, mesh_cell = _sample_nearest_tree_candidate(trees, domain, config, remaining)
        tree = trees[int(tree_idx)]
        qin_cm3_s = _flow_for_tree(config, int(tree_idx), len(trees))
        terminal_flow = _terminal_flow_for_target(config, qin_cm3_s, int(targets[int(tree_idx)]))
        ts._sync_loaded_tree_params(
            tree,
            side_length=float(config.domain.side_length),
            terminal_flow_override=terminal_flow,
        )
        t0 = perf_counter()
        try:
            with _fixed_growth_points(point, mesh_cell):
                result = tree.add(
                    inplace=False,
                    n_closest_vessels=int(config.growth.n_closest_vessels),
                    n_points=1,
                    use_random_int=not bool(config.growth.weighted_sampling),
                    ignore_collisions=bool(ignore_intertree_collisions),
                    allow_inside_vessels=bool(config.growth.allow_inside_vessels),
                    debug_add_vessel=getattr(ts, "DEBUG_ADD_VESSEL", False),
                )
        except Exception as exc:
            last_error = exc
            continue
        if _candidate_leaves_domain(config, tree, result[-1]):
            continue
        if (not ignore_intertree_collisions) and _candidate_hits_other_tree(
            trees,
            int(tree_idx),
            result[-1],
            clearance=float(config.network.physical_clearance),
        ):
            continue
        _commit_tree_add_result(tree, result)
        return int(tree_idx), float(perf_counter() - t0)
    raise RuntimeError(
        f"Could not add a nearest-tree vessel after {max_attempts} attempts."
    ) from last_error


def _sample_nearest_tree_candidate(
    trees: list[Any],
    domain,
    config: RunConfig,
    remaining: list[int],
) -> tuple[int, np.ndarray, int]:
    active = [idx for idx, rem in enumerate(remaining) if int(rem) > 0]
    if not active:
        raise RuntimeError("No active trees remain for nearest-tree growth.")
    points, cells = _domain_interior_points(domain, int(config.growth.nearest_tree_batch_points))
    finite = np.isfinite(points).all(axis=1)
    points = points[finite, :]
    cells = cells[finite]
    if points.shape[0] == 0:
        raise RuntimeError("Domain returned no finite nearest-tree candidate points.")

    nearest_by_tree = []
    for idx in active:
        data = np.asarray(trees[int(idx)].data[: int(getattr(trees[int(idx)], "segment_count", 0) or 0), :])
        nearest_by_tree.append(_min_point_segment_wall_distance(data, points))
    nearest_by_tree_arr = np.vstack(nearest_by_tree)
    assigned_pos = np.argmin(nearest_by_tree_arr, axis=0)
    counts = np.bincount(assigned_pos, minlength=len(active))
    best_pos = int(np.argmax(counts))
    point_candidates = np.flatnonzero(assigned_pos == best_pos)
    if point_candidates.size == 0:
        point_i = 0
        best_pos = int(assigned_pos[0])
    else:
        local_dist = nearest_by_tree_arr[best_pos, point_candidates]
        point_i = int(point_candidates[int(np.argmax(local_dist))])
    return int(active[best_pos]), points[point_i, :].copy(), int(cells[point_i])


def _domain_interior_points(domain, count: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        points, cells = domain.get_interior_points(int(count))
    except TypeError:
        points = domain.get_interior_points(int(count))
        cells = np.full((np.asarray(points).reshape(-1, 3).shape[0],), -1, dtype=np.int64)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells, dtype=np.int64).reshape(-1)
    if cells.size != points.shape[0]:
        cells = np.full((points.shape[0],), -1, dtype=np.int64)
    return points, cells


@contextmanager
def _fixed_growth_points(point: np.ndarray, mesh_cell: int):
    original = gfm_bifurcation.add_vessel.__globals__["get_points"]
    fixed_point = np.asarray(point, dtype=np.float64).reshape(1, 3)
    fixed_cell = int(mesh_cell)
    used = False

    def _get_points(tree, n_points, **kwargs):
        nonlocal used
        if used:
            raise RuntimeError("Fixed nearest-tree growth point was rejected by CCO.")
        used = True
        return _fixed_point_candidates(tree, fixed_point, fixed_cell, kwargs)

    gfm_bifurcation.add_vessel.__globals__["get_points"] = _get_points
    try:
        yield
    finally:
        gfm_bifurcation.add_vessel.__globals__["get_points"] = original


def _fixed_point_candidates(tree, points: np.ndarray, mesh_cell: int, kwargs: dict[str, Any]):
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0), :], dtype=np.float64)
    if data.shape[0] <= 0:
        raise RuntimeError("Cannot grow from a fixed point on an empty tree.")
    n_points = int(points.shape[0])
    n_vessels_native = max(1, min(int(kwargs.get("n_vessels", min(data.shape[0], 10))), int(data.shape[0])))
    n_heuristic = int(kwargs.get("n_heuristic", 500))
    threshold_cutoff = int(kwargs.get("n_random_int", 10000))
    threshold = float(kwargs.get("threshold", getattr(tree, "physical_clearance", 0.0) or 0.0))
    if int(getattr(tree, "n_terminals", 0) or 0) >= threshold_cutoff:
        threshold = 0.0
    use_all_candidates = int(getattr(tree, "n_terminals", 0) or 0) < n_heuristic
    keep_k = int(data.shape[0]) if use_all_candidates else n_vessels_native
    distances = _point_segment_distance_matrix(data, points, subtract_radius=not use_all_candidates)
    order = np.argsort(distances, axis=1)[:, :keep_k]
    ordered_distances = np.take_along_axis(distances, order, axis=1)
    min_dist = np.min(ordered_distances, axis=1)

    out_points = np.full((n_points, 3), np.nan, dtype=np.float64)
    out_dist = np.full((keep_k, n_points), np.nan, dtype=np.float64)
    out_idx = np.zeros((keep_k, n_points), dtype=np.int64)
    out_cells = np.full((n_points,), -1, dtype=np.int64)
    keep = min_dist > threshold
    if np.any(keep):
        out_points[keep, :] = points[keep, :]
        out_dist[:, keep] = ordered_distances[keep, :].T
        out_idx[:, keep] = order[keep, :].T
        out_cells[keep] = int(mesh_cell)
    return out_points, out_dist, out_idx, out_cells


def _point_segment_distance_matrix(data: np.ndarray, points: np.ndarray, *, subtract_radius: bool) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    ab = data[:, 3:6] - data[:, 0:3]
    ap = points[:, None, :] - data[None, :, 0:3]
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom > 0.0, denom, 1.0e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.sum(ap * ab[None, :, :], axis=2) / denom[None, :], 0.0, 1.0)
    closest = data[None, :, 0:3] + t[:, :, None] * ab[None, :, :]
    distances = np.linalg.norm(points[:, None, :] - closest, axis=2)
    if subtract_radius and data.shape[1] > 21:
        distances = distances - np.maximum(data[None, :, 21], 0.0)
    return distances


def _min_point_segment_wall_distance(data: np.ndarray, points: np.ndarray) -> np.ndarray:
    if data.shape[0] == 0:
        return np.full((np.asarray(points).reshape(-1, 3).shape[0],), np.inf, dtype=float)
    return np.min(_point_segment_distance_matrix(data, points, subtract_radius=True), axis=1)


def _ignore_intertree_collisions_now(config: RunConfig, trees: list[Any]) -> bool:
    if bool(config.growth.ignore_collisions):
        return True
    threshold = config.growth.n_ignore_collisions
    if threshold is None:
        return False
    threshold = int(threshold)
    if threshold < 0:
        return False
    if threshold == 0:
        return True
    total_segments = sum(int(getattr(tree, "segment_count", 0) or 0) for tree in trees)
    return int(total_segments) >= threshold


def _nearest_bulk_growth_allowed(config: RunConfig, trees: list[Any], remaining: list[int], *, ignore_now: bool) -> bool:
    if not bool(ignore_now):
        return False
    mode = str(config.growth.bulk_growth_mode or "never").strip().lower().replace("_", "-")
    if mode == "never":
        return False
    if mode == "always":
        return True
    all_equal = all(int(rem) <= 0 or _tree_in_equal_bifurcation_mode(config, tree) for rem, tree in zip(remaining, trees))
    if mode == "equal-bifurcation":
        return all_equal
    if mode == "after-collision-threshold":
        return True
    return False


def _tree_in_equal_bifurcation_mode(config: RunConfig, tree: Any) -> bool:
    params = getattr(tree, "parameters", None)
    n_equal = getattr(params, "n_equal_bifurcations", config.growth.n_equal_bifurcations) if params is not None else config.growth.n_equal_bifurcations
    if n_equal is None:
        return False
    try:
        n_equal_int = int(n_equal)
    except Exception:
        return False
    return n_equal_int >= 0 and int(getattr(tree, "n_terminals", 0) or 0) >= n_equal_int


def _candidate_hits_other_tree(trees: list[Any], tree_idx: int, added_vessels: Any, *, clearance: float) -> bool:
    for other_idx, other_tree in enumerate(trees):
        if int(other_idx) == int(tree_idx):
            continue
        other_count = int(getattr(other_tree, "segment_count", 0) or 0)
        if other_count <= 0:
            continue
        other_data = np.asarray(other_tree.data[:other_count], dtype=np.float64)
        for vessel in np.asarray(added_vessels, dtype=np.float64).reshape(-1, np.asarray(added_vessels).shape[-1]):
            if tree_collision(other_data, vessel.reshape(1, -1), clearance=float(clearance)):
                return True
    return False


def _candidate_leaves_domain(config: RunConfig, tree: Any, added_vessels: Any) -> bool:
    if not bool(config.growth.strict_domain_segments):
        return False
    max_terminals = int(config.growth.strict_domain_max_terminals)
    if max_terminals >= 0 and int(getattr(tree, "n_terminals", 0) or 0) > max_terminals:
        return False
    domain = getattr(tree, "domain", None)
    if domain is None:
        return False
    n_samples = max(2, int(config.growth.domain_line_samples))
    tolerance = float(config.growth.domain_line_tolerance)
    t = np.linspace(0.0, 1.0, n_samples, dtype=float).reshape(-1, 1)
    for vessel in np.asarray(added_vessels, dtype=float).reshape(-1, np.asarray(added_vessels).shape[-1]):
        start = vessel[0:3]
        end = vessel[3:6]
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            return True
        line = start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t
        values = np.asarray(domain(line), dtype=float).reshape(-1)
        if values.size == 0 or np.any(~np.isfinite(values)) or np.any(values > tolerance):
            return True
    return False


def _prepare_loaded_tree_for_incremental_growth(tree: Any) -> None:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if seg_count <= 0:
        return
    repair_trees([tree])
    if getattr(tree, "vessel_map", None) is None or len(getattr(tree, "vessel_map", {})) < seg_count:
        _repair_loaded_tree_vessel_map(tree)
    _rebuild_tree_spatial_indices(tree)


def _repair_loaded_tree_vessel_map(tree: Any) -> None:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if seg_count <= 0:
        return
    data = np.asarray(tree.data[:seg_count])
    index_dtype = getattr(tree, "index_dtype", np.int64)
    connectivity = np.nan_to_num(data[:, 15:18], nan=-1.0).astype(index_dtype)
    tree.connectivity = connectivity

    ancestor_cache: dict[int, list[int]] = {}
    descendant_cache: dict[int, list[int]] = {}

    def ancestors(row: int) -> list[int]:
        row = int(row)
        if row in ancestor_cache:
            return list(ancestor_cache[row])
        parent = int(connectivity[row, 2])
        out = [] if parent < 0 else ancestors(parent) + [parent]
        ancestor_cache[row] = list(out)
        return out

    def descendants(row: int) -> list[int]:
        row = int(row)
        if row in descendant_cache:
            return list(descendant_cache[row])
        out: list[int] = []
        for child in [int(x) for x in connectivity[row, 0:2] if int(x) >= 0]:
            out.append(child)
            out.extend(descendants(child))
        descendant_cache[row] = list(out)
        return out

    vessel_map = TreeMap()
    for row in range(seg_count):
        vessel_map[int(row)] = {"upstream": ancestors(row), "downstream": descendants(row)}
    tree.vessel_map = vessel_map


def _commit_tree_add_result(tree: Any, result: tuple[Any, ...], *, decay_probability: float = 0.9) -> None:
    change_i, change_j, new_tmp_data, _old_tmp_data, new_vessel_map, connectivity, new_inds, mesh_cell, added_vessels = result
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    if hasattr(tree, "ensure_preallocation"):
        tree.ensure_preallocation(seg_count + 2)
    tree.preallocate[seg_count, :] = added_vessels[0]
    tree.preallocate[seg_count + 1, :] = added_vessels[1]
    change_i = np.asarray(change_i, dtype=int)
    change_j = np.asarray(change_j, dtype=int)
    tree.preallocate[change_i, change_j] = np.asarray(new_tmp_data)
    tree.segment_count = seg_count + 2
    tree.n_terminals = int(getattr(tree, "n_terminals", 0) or 0) + 1
    tree.data = tree.preallocate[: tree.segment_count, :]
    if getattr(tree, "vessel_map", None) is None:
        tree.vessel_map = TreeMap()
    for key, value in new_vessel_map.items():
        if key in tree.vessel_map:
            tree.vessel_map[key]["upstream"].extend(value.get("upstream", []))
            tree.vessel_map[key]["downstream"].extend(value.get("downstream", []))
        else:
            tree.vessel_map[key] = {
                "upstream": list(value.get("upstream", [])),
                "downstream": list(value.get("downstream", [])),
            }
    tree.connectivity = connectivity
    tree.max_distal_node = int(getattr(tree, "max_distal_node", 0) or 0) + 2
    try:
        mesh_cell_i = int(mesh_cell)
    except Exception:
        mesh_cell_i = -1
    if mesh_cell_i >= 0 and getattr(tree, "probability", None) is not None:
        try:
            tree.probability[mesh_cell_i] *= float(decay_probability)
            total = float(np.sum(tree.probability))
            if total > 0.0:
                tree.probability = tree.probability / total
                tree.domain.cumulative_probability = np.cumsum(tree.probability)
        except Exception:
            pass
    if hasattr(tree, "new_tree_scale"):
        tree.tree_scale = tree.new_tree_scale
    _update_tree_spatial_indices(tree, np.asarray(new_inds, dtype=int).reshape(-1), old_segment_count=seg_count)
    repair_trees([tree])


def _update_tree_spatial_indices(tree: Any, rows: np.ndarray, *, old_segment_count: int) -> None:
    rows = np.asarray(rows, dtype=int).reshape(-1)
    rows = np.unique(rows[(rows >= 0) & (rows < int(getattr(tree, "segment_count", 0) or 0))])
    if rows.size == 0:
        return
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    mid = (data[rows, 0:3] + data[rows, 3:6]) * 0.5
    tree.preallocate_midpoints[rows, :] = mid
    tree.midpoints = tree.preallocate_midpoints[: int(getattr(tree, "segment_count", 0) or 0), :]
    hnsw = getattr(tree, "hnsw_tree", None)
    try:
        if hnsw is None:
            raise RuntimeError("Missing HNSW tree.")
        replace_mask = rows < int(old_segment_count)
        if np.any(replace_mask):
            hnsw.replace(mid[replace_mask].astype(np.float32), rows[replace_mask].astype(int))
        if np.any(~replace_mask):
            hnsw.add_items(mid[~replace_mask].astype(np.float32), rows[~replace_mask].astype(int))
        tree.hnsw_tree_id = id(hnsw)
    except Exception:
        _rebuild_tree_spatial_indices(tree)


def _rebuild_tree_spatial_indices(tree: Any) -> None:
    seg_count = int(getattr(tree, "segment_count", 0) or 0)
    data = np.asarray(tree.data[:seg_count])
    mid = (data[:, 0:3] + data[:, 3:6]) * 0.5 if seg_count else np.empty((0, 3), dtype=float)
    if getattr(tree, "preallocate_midpoints", None) is None or int(tree.preallocate_midpoints.shape[0]) < seg_count:
        tree.preallocate_midpoints = np.zeros((max(seg_count, 1), 3), dtype=getattr(tree, "data_dtype", np.float64))
    tree.preallocate_midpoints[:seg_count, :] = mid
    tree.midpoints = tree.preallocate_midpoints[:seg_count, :]
    tree.kdtm = KDTreeManager(mid)
    tree.hnsw_tree = USearchTree(mid.astype(np.float32))
    tree.hnsw_tree_id = id(tree.hnsw_tree)


def _save_growth_checkpoint(
    config: RunConfig,
    trees: list[Any],
    forest,
    *,
    completed: int,
    total_add: int,
    targets: list[int],
    final: bool = False,
) -> None:
    checkpoint = resolve_path(
        config.growth.checkpoint_path,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if checkpoint is None:
        return
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if forest is not None:
        forest.save(str(checkpoint))
    elif trees:
        trees[0].save(str(checkpoint), include_domain=False)
    meta = {
        "reason": "final" if final else "periodic",
        "completed_adds": int(completed),
        "total_adds": int(total_add),
        "targets": [int(v) for v in targets],
        "terminal_segments": [max(int(getattr(t, "n_terminals", 0) or 0) - 1, 0) for t in trees],
    }
    checkpoint.with_name(checkpoint.name + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _save_reached_targets(
    config: RunConfig,
    trees: list[Any],
    forest,
    saved: set[int],
    final_targets: list[int],
) -> set[int]:
    if not config.growth.save_target_counts:
        return saved
    current_total = sum(max(int(getattr(t, "n_terminals", 0) or 0) - 1, 0) for t in trees)
    out_dir = resolve_path(config.outputs.out_dir, base_dir=config.settings_path.parent if config.settings_path else None)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    for target in config.growth.save_target_counts:
        target = int(target)
        if target in saved or current_total < target:
            continue
        if forest is not None:
            path = out_dir / f"{config.prefix}_t{target}.forest"
            forest.save(str(path))
        elif trees:
            path = out_dir / f"{config.prefix}_t{target}.tree.npz"
            trees[0].save(str(path), include_domain=False)
        meta = {
            "reason": "target_save",
            "total_terminal_segments": int(current_total),
            "target_total": int(target),
            "final_targets": [int(v) for v in final_targets],
        }
        path.with_name(path.name + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        saved.add(target)
    return saved


def _make_forest(config: RunConfig, domain, trees: list[Any]):
    starts = [[np.asarray(root.start, dtype=float).reshape(1, 3) for root in config.network.roots]]
    directions = [[None if root.direction is None else np.asarray(root.direction, dtype=float).reshape(1, 3) for root in config.network.roots]]
    forest = Forest(
        domain=domain,
        n_networks=1,
        n_trees_per_network=[len(trees)],
        start_points=starts,
        directions=directions,
        physical_clearance=float(config.network.physical_clearance),
        compete=False,
        preallocation_step=1,
    )
    forest.networks = [list(trees)]
    forest.domain = domain
    forest.geodesic = None
    return forest


def _target_counts_for_config(config: RunConfig, n_trees: int, *, trees: list[Any] | None = None) -> list[int]:
    if trees and config.growth.add_per_tree:
        adds = [max(int(v), 0) for v in config.growth.add_per_tree]
        if len(adds) == 1 and n_trees > 1:
            adds = adds * n_trees
        if len(adds) != n_trees:
            raise ValueError(f"Expected {n_trees} growth.add_per_tree values, got {len(adds)}.")
        current = [_tree_terminal_segments(tree) for tree in trees]
        counts = [max(cur + add, 1) for cur, add in zip(current, adds)]
    elif trees and config.growth.add_total is not None:
        current = [_tree_terminal_segments(tree) for tree in trees]
        adds = _split_total_adds(int(config.growth.add_total), trees, mode=config.growth.add_split_mode)
        counts = [max(cur + add, 1) for cur, add in zip(current, adds)]
    elif config.network.target_terminal_counts:
        counts = [max(int(v), 1) for v in config.network.target_terminal_counts]
    elif config.network.target_total_terminal_count is not None:
        counts = _split_total_target(int(config.network.target_total_terminal_count), n_trees, trees=trees)
    elif config.network.target_terminal_count is not None:
        counts = [max(int(config.network.target_terminal_count), 1)]
    elif trees:
        counts = [max(int(getattr(tree, "n_terminals", 1) or 1) - 1, 1) for tree in trees]
    else:
        counts = [1]

    if len(counts) == 1 and n_trees > 1:
        counts = counts * n_trees
    if len(counts) != n_trees:
        raise ValueError(f"Expected {n_trees} target counts, got {len(counts)}.")
    return counts


def _tree_terminal_segments(tree: Any) -> int:
    return max(int(getattr(tree, "n_terminals", 0) or 0) - 1, 1)


def _tree_root_flow_for_weight(tree: Any) -> float:
    params = getattr(tree, "parameters", None)
    if params is not None:
        value = getattr(params, "root_flow", None)
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        value = float(data[0, 22])
        if np.isfinite(value) and value > 0.0:
            return value
    return 1.0


def _split_total_adds(total: int, trees: list[Any], *, mode: str) -> list[int]:
    total = max(int(total), 0)
    n_trees = len(trees)
    if n_trees <= 0:
        return []
    if str(mode).strip().lower() == "flow-proportional":
        weights = np.array([_tree_root_flow_for_weight(tree) for tree in trees], dtype=float)
    else:
        weights = np.ones(n_trees, dtype=float)
    return _integer_split(total, weights)


def _split_total_target(total: int, n_trees: int, *, trees: list[Any] | None) -> list[int]:
    total = max(int(total), n_trees)
    if trees:
        weights = np.array([_tree_root_flow_for_weight(tree) for tree in trees], dtype=float)
    else:
        weights = np.ones(n_trees, dtype=float)
    return [max(v, 1) for v in _integer_split(total, weights)]


def _integer_split(total: int, weights: np.ndarray) -> list[int]:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        return []
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if not np.any(weights > 0.0):
        weights = np.ones_like(weights)
    raw = float(total) * weights / float(np.sum(weights))
    base = np.floor(raw).astype(int)
    remainder = int(total - int(np.sum(base)))
    if remainder > 0:
        order = np.argsort(-(raw - base))
        for idx in order[:remainder]:
            base[int(idx)] += 1
    return [int(v) for v in base]


def _repair_and_validate_if_requested(trees: list[Any], config: RunConfig) -> tuple[list[int], list[dict[str, Any]]]:
    repairs: list[int] = []
    reports: list[dict[str, Any]] = []
    if bool(config.network.repair_connectivity):
        repairs = repair_trees(trees)
    if bool(config.network.validate_connectivity):
        reports = validate_trees(
            trees,
            fail=bool(config.network.fail_connectivity),
            geometry_atol=float(config.network.connectivity_geometry_atol),
        )
    return repairs, reports


def _set_tissuesim_root(ts, root: RootConfig, *, side_length: float) -> None:
    scale = float(side_length) if side_length else 1.0
    start = np.asarray(root.start, dtype=float).reshape(1, 3) / scale
    ts.ROOT_LOCATION = start
    ts.ROOT_DIR = None if root.direction is None else np.asarray(root.direction, dtype=float).reshape(1, 3)


def _flow_for_tree(config: RunConfig, tree_id: int, n_trees: int) -> float:
    source = str(config.simulation.flow_source).strip().lower().replace("_", "-")
    qin_ul_min = (
        float(config.simulation.total_qin_ul_min)
        if source == "total-qin-split" and config.simulation.total_qin_ul_min is not None
        else float(config.simulation.qin_target_ul_min)
    )
    if source in {"total-split", "total-qin-split"} and n_trees > 0:
        qin_ul_min = qin_ul_min / float(n_trees)
    q_scale = float(config.domain.side_length) ** 3
    return qin_ul_min * 1.0e-3 / 60.0 * q_scale


def _terminal_flow_for_target(config: RunConfig, qin_cm3_s: float, target_count: int) -> float | None:
    # Match TissueSim_cube_local's scaled-flow convention: target_count is the
    # number passed to n_add, so terminal sinks are target_count + 1.
    if not getattr(load_tissuesim_module(), "SCALE_Q_BY_VOLUME", True):
        return None
    return float(qin_cm3_s) / float(max(int(target_count), 1) + 1)


def flow_for_tree(config: RunConfig, tree_id: int, n_trees: int) -> float:
    return _flow_for_tree(config, tree_id, n_trees)


def terminal_flow_for_target(config: RunConfig, qin_cm3_s: float, target_count: int) -> float | None:
    return _terminal_flow_for_target(config, qin_cm3_s, target_count)
