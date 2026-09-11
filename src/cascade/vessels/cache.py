"""Vessel archive and geometry cache handling."""

from __future__ import annotations

from cascade.vessels.results import (
    NetworkBuildResult,
)

from cascade.vessels._build_common import (
    Any,
    Domain,
    Forest,
    Path,
    RunConfig,
    Tree,
    json,
    np,
    perf_counter,
    resolve_path,
)

def _shared_geometry_cache_dir(config: RunConfig) -> Path | None:
    gui = config.raw.get("gui", {}) if isinstance(config.raw, dict) else {}
    value = gui.get("shared_geometry_cache_dir") if isinstance(gui, dict) else None
    if not value:
        return None
    return resolve_path(value, base_dir=config.settings_path.parent if config.settings_path else None)


def _load_shared_geometry_cache(
    config: RunConfig,
) -> tuple[Any, list[Any] | None, Any | None, tuple[Path, float]] | None:
    """Load a completed GUI-sweep geometry bundle, if one is available."""
    cache_dir = _shared_geometry_cache_dir(config)
    if cache_dir is None:
        return None
    manifest_path = cache_dir / "ready.json"
    if not manifest_path.is_file():
        return None
    t0 = perf_counter()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("format", 0)) != 1:
            raise ValueError("unsupported cache format")
        domain_path = cache_dir / str(manifest["domain"])
        domain = Domain.load(str(domain_path))
        domain.random_seed = int(config.domain.random_seed)
        domain.set_random_generator()
        network_name = manifest.get("network")
        if not network_name:
            return domain, None, None, (cache_dir, perf_counter() - t0)

        # Cached networks are final geometry.  Load them in simulation mode so
        # the normal input-path code cannot extend/regrow them for this run.
        growth_enabled = bool(config.growth.enabled)
        use_cache = bool(config.outputs.use_cache)
        config.growth.enabled = False
        config.outputs.use_cache = False
        try:
            trees, forest = _load_existing_network(cache_dir / str(network_name), domain, config)
        finally:
            config.growth.enabled = growth_enabled
            config.outputs.use_cache = use_cache
        return domain, trees, forest, (cache_dir, perf_counter() - t0)
    except Exception as exc:
        print(f"Warning: could not reuse sweep geometry cache at {cache_dir}: {exc}", flush=True)
        return None


def _save_shared_geometry_cache(result: NetworkBuildResult, config: RunConfig) -> NetworkBuildResult:
    """Publish a geometry bundle after a successful first run in a GUI sweep."""
    cache_dir = _shared_geometry_cache_dir(config)
    if cache_dir is None:
        return result
    manifest_path = cache_dir / "ready.json"
    if manifest_path.is_file():
        return result
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        domain_saved = Path(
            result.domain.save(
                str(cache_dir / "domain.dmn"),
                include_boundary=True,
                include_mesh=True,
                include_patch_normals=True,
            )
        )
        network_saved: Path | None = None
        network_kind = "domain"
        if result.forest is not None:
            network_saved = _published_archive_path(
                result.forest.save(str(cache_dir / "network.forest"))
            )
            network_kind = "forest"
        elif result.trees and not getattr(result.trees[0], "_cascade_simple_network", False):
            network_saved = _published_archive_path(
                result.trees[0].save(str(cache_dir / "network.tree.npz"))
            )
            network_kind = "tree"

        manifest = {
            "format": 1,
            "kind": network_kind,
            "domain": domain_saved.name,
            "network": None if network_saved is None else network_saved.name,
        }
        temporary = cache_dir / "ready.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
        print(f"Saved sweep geometry cache: {cache_dir}", flush=True)
    except Exception as exc:
        # Caching is an optimization: never discard a completed simulation
        # setup merely because its reusable copy could not be written.
        print(f"Warning: could not save sweep geometry cache at {cache_dir}: {exc}", flush=True)
    return result


def _published_archive_path(reported_path: str | Path) -> Path:
    """Resolve NumPy's implicit .npz suffix and publish the reported filename."""
    path = Path(reported_path)
    if path.is_file():
        return path
    appended = Path(str(path) + ".npz")
    if appended.is_file():
        appended.replace(path)
        return path
    raise FileNotFoundError(f"Network save did not create {path} or {appended}.")


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
    elif result.trees and getattr(result.trees[0], "_cascade_simple_network", False):
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


def _load_existing_network(path: Path, domain, config: RunConfig, *, ts) -> tuple[list[Any], Any | None]:
    suffix = path.suffix.lower()
    if Forest._is_simulation_cache(str(path)):
        if config.growth.enabled:
            raise ValueError("Direct .forest simulation-cache inputs require growth.enabled=false.")
        forest = Forest.load(
            str(path),
            mode="simulation",
            data_dtype=ts.TREE_DATA_DTYPE,
            index_dtype=ts.TREE_INDEX_DTYPE,
        )
        _attach_domain(forest, domain, simulation_only=True)
        trees = [tree for network in forest.networks for tree in network]
        return trees, forest

    if suffix == ".forest" or path.name.endswith(".forest"):
        mode = "simulation" if not config.growth.enabled else "growth"
        load_path = _forest_load_path(path, config, mode=mode)
        forest = Forest.load(
            str(load_path),
            mode=mode,
            data_dtype=ts.TREE_DATA_DTYPE if mode == "simulation" else None,
            index_dtype=ts.TREE_INDEX_DTYPE,
        )
        _attach_domain(forest, domain, simulation_only=(mode == "simulation"))
        trees = [tree for network in forest.networks for tree in network]
        return trees, forest

    tree = Tree.load(
        str(path),
        domain=domain,
        data_dtype=ts.TREE_DATA_DTYPE,
        index_dtype=ts.TREE_INDEX_DTYPE,
        analysis_only=not config.growth.enabled,
    )
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




__all__ = ('_shared_geometry_cache_dir', '_load_shared_geometry_cache', '_save_shared_geometry_cache', '_published_archive_path', 'save_network_if_requested', '_load_existing_network', '_attach_domain', '_attach_tree_domain', '_forest_cache_path', '_forest_load_path')
