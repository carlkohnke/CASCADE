"""Public vessel construction orchestration."""

from __future__ import annotations

from cascade.vessels.results import (
    NetworkBuildResult,
)

from cascade.vessels._build_common import (
    RunConfig,
    build_simple_network,
    perf_counter,
    resolve_path,
)

from cascade.vessels.conditions import (
    _make_forest,
    _repair_and_validate_if_requested,
    _target_counts_for_config,
)

from cascade.vessels.cache import (
    _load_existing_network,
    _load_shared_geometry_cache,
    _save_shared_geometry_cache,
)

from cascade.vessels.growth import (
    _build_configured_trees,
    _extend_trees_to_targets,
    _pre_sample_points,
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




__all__ = ["NetworkBuildResult", "build_or_load_network"]
