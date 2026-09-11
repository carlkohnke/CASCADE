"""Public vessel construction orchestration."""

from __future__ import annotations

from cascade.vessels.results import (
    NetworkBuildResult,
)

from cascade.vessels._build_common import (
    Domain,
    RunConfig,
    build_simple_network,
    perf_counter,
    resolve_path,
)

from cascade.vessels.conditions import (
    _flow_for_tree,
    _inlet_condition_for_tree,
    _integer_split,
    _make_forest,
    _repair_and_validate_if_requested,
    _set_runtime_root,
    _set_runtime_tree_conditions,
    _split_total_adds,
    _split_total_target,
    _target_counts_for_config,
    _terminal_flow_for_target,
    _tree_root_flow_for_weight,
    _tree_terminal_segments,
    flow_for_tree,
    inlet_concentration_for_tree,
    pressures_for_tree,
    sync_tree_parameters_for_run,
    terminal_flow_for_target,
)

from cascade.vessels.cache import (
    _attach_domain,
    _attach_tree_domain,
    _forest_cache_path,
    _forest_load_path,
    _load_existing_network,
    _load_shared_geometry_cache,
    _published_archive_path,
    _save_shared_geometry_cache,
    _shared_geometry_cache_dir,
    save_network_if_requested,
)

from cascade.vessels.growth import (
    _build_configured_trees,
    _candidate_hits_other_tree,
    _candidate_leaves_domain,
    _commit_tree_add_result,
    _domain_interior_points,
    _extend_trees_nearest,
    _extend_trees_scheduled,
    _extend_trees_to_targets,
    _file_sha256,
    _fixed_growth_points,
    _fixed_point_candidates,
    _grow_one_nearest_tree,
    _grow_remaining_bulk_no_collision,
    _ignore_intertree_collisions_now,
    _load_sample_points,
    _min_point_segment_wall_distance,
    _nearest_bulk_growth_allowed,
    _point_segment_distance_matrix,
    _pre_sample_points,
    _prepare_loaded_tree_for_incremental_growth,
    _rebuild_tree_spatial_indices,
    _repair_loaded_tree_vessel_map,
    _sample_nearest_tree_candidate,
    _save_growth_checkpoint,
    _save_reached_targets,
    _tree_in_equal_bifurcation_mode,
    _update_tree_spatial_indices,
)

from cascade.configuration.bridge import (
    apply_runtime_settings,
    load_runtime_module,
)

from cascade.domain.workflow import (
    build_domain,
)

def build_or_load_network(
    config: RunConfig, *, domain_override=None
) -> NetworkBuildResult:
    ts = load_runtime_module()
    apply_runtime_settings(ts, config)
    timings: dict[str, float] = {}

    cached = None if domain_override is not None else _load_shared_geometry_cache(config)
    if cached is not None:
        domain, trees, forest, load_source = cached
        timings["geometry_cache_load_s"] = float(load_source[1])
        t0 = perf_counter()
        sample_points, sample_meta = _pre_sample_points(ts, domain, config)
        timings["sample_points_s"] = perf_counter() - t0
        if trees is not None:
            target_counts = _target_counts_for_config(config, len(trees), trees=trees)
            print(f"Reusing sweep geometry cache: {load_source[0]}", flush=True)
            return NetworkBuildResult(
                domain=domain,
                trees=trees,
                forest=forest,
                target_counts=target_counts,
                build_timings=timings,
                sample_points=sample_points,
                sample_meta=sample_meta,
                load_source=load_source[0],
            )
        print(f"Reusing sweep domain cache: {load_source[0]}", flush=True)
    else:
        t0 = perf_counter()
        domain = domain_override if domain_override is not None else build_domain(config, ts=ts)
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
        trees, forest = _load_existing_network(input_path, domain, config, ts=ts)
        timings["network_load_s"] = perf_counter() - t0
        target_counts = _target_counts_for_config(config, len(trees), trees=trees)
        if config.growth.enabled:
            t0 = perf_counter()
            _extend_trees_to_targets(ts, trees, domain, config, target_counts, forest=forest)
            timings["growth_s"] = perf_counter() - t0
        repairs, reports = _repair_and_validate_if_requested(trees, config)
        return _save_shared_geometry_cache(NetworkBuildResult(
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
        ), config)

    if config.network_mode == "simple":
        t0 = perf_counter()
        simple_network = build_simple_network(ts, domain, config)
        timings["growth_s"] = perf_counter() - t0
        return _save_shared_geometry_cache(NetworkBuildResult(
            domain=domain,
            trees=[simple_network],
            forest=None,
            target_counts=[int(getattr(simple_network, "n_terminals", 1) or 1)],
            build_timings=timings,
            sample_points=sample_points,
            sample_meta=sample_meta,
        ), config)

    t0 = perf_counter()
    trees = _build_configured_trees(ts, domain, config)
    timings["growth_s"] = perf_counter() - t0
    forest = _make_forest(config, domain, trees) if config.network_mode == "forest" else None
    target_counts = _target_counts_for_config(config, len(trees), trees=trees)
    repairs, reports = _repair_and_validate_if_requested(trees, config)
    return _save_shared_geometry_cache(NetworkBuildResult(
        domain=domain,
        trees=trees,
        forest=forest,
        target_counts=target_counts,
        build_timings=timings,
        sample_points=sample_points,
        sample_meta=sample_meta,
        connectivity_repairs=repairs,
        connectivity_reports=reports,
    ), config)




__all__ = ('Domain', 'NetworkBuildResult', 'load_runtime_module', 'apply_runtime_settings', 'build_domain', 'build_or_load_network', '_shared_geometry_cache_dir', '_load_shared_geometry_cache', '_save_shared_geometry_cache', '_published_archive_path', 'save_network_if_requested', '_load_existing_network', '_attach_domain', '_attach_tree_domain', '_forest_cache_path', '_forest_load_path', '_build_configured_trees', '_pre_sample_points', '_load_sample_points', '_file_sha256', '_extend_trees_to_targets', '_extend_trees_scheduled', '_extend_trees_nearest', '_grow_remaining_bulk_no_collision', '_grow_one_nearest_tree', '_sample_nearest_tree_candidate', '_domain_interior_points', '_fixed_growth_points', '_fixed_point_candidates', '_point_segment_distance_matrix', '_min_point_segment_wall_distance', '_ignore_intertree_collisions_now', '_nearest_bulk_growth_allowed', '_tree_in_equal_bifurcation_mode', '_candidate_hits_other_tree', '_candidate_leaves_domain', '_prepare_loaded_tree_for_incremental_growth', '_repair_loaded_tree_vessel_map', '_commit_tree_add_result', '_update_tree_spatial_indices', '_rebuild_tree_spatial_indices', '_save_growth_checkpoint', '_save_reached_targets', '_make_forest', '_target_counts_for_config', '_tree_terminal_segments', '_tree_root_flow_for_weight', '_split_total_adds', '_split_total_target', '_integer_split', '_repair_and_validate_if_requested', '_set_runtime_root', '_flow_for_tree', '_terminal_flow_for_target', 'flow_for_tree', '_inlet_condition_for_tree', 'pressures_for_tree', 'inlet_concentration_for_tree', '_set_runtime_tree_conditions', 'sync_tree_parameters_for_run', 'terminal_flow_for_target')
