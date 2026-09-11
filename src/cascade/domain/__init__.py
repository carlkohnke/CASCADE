"""Domain construction, geometry, grid generation, and sampling."""

from cascade.utils.lazy import resolve_export

_EXPORTS = {
    "Domain": ("cascade.domain.svv", "Domain"),
    "build_domain": ("cascade.domain.builders", "build_domain"),
    "build_domain_from_pyvista": (
        "cascade.domain.builders",
        "build_domain_from_pyvista",
    ),
    "build_domain_from_config": ("cascade.domain.workflow", "build_domain"),
    "compute_average_distance": ("cascade.domain.sampling", "compute_average_distance"),
    "compute_distance_to_nearest_channel": (
        "cascade.domain.sampling",
        "compute_distance_to_nearest_channel",
    ),
    "default_domain_cache_dir": ("cascade.domain.cache", "default_domain_cache_dir"),
    "domain_cache_path": ("cascade.domain.cache", "domain_cache_path"),
    "load_sample_points": ("cascade.domain.sampling", "load_sample_points"),
    "sample_domain_points": ("cascade.domain.sampling", "sample_domain_points"),
    "sample_grid_points": ("cascade.domain.grid", "sample_grid_points"),
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), _EXPORTS, name)
