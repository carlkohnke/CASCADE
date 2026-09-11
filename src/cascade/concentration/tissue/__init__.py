"""Green's Function Method tissue oxygen analysis."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TissueOxygenProblem": (".api", "TissueOxygenProblem"),
    "TissueOxygenResult": (".api", "TissueOxygenResult"),
    "solve_tissue_oxygen": (".api", "solve_tissue_oxygen"),
    "build_tissue_cache_from_tree": (".cache", "build_tissue_cache_from_tree"),
    "compute_tissue_samples_greens": (".greens", "compute_tissue_samples_greens"),
    "compute_tissue_samples_greens_from_cext_state": (
        ".greens",
        "compute_tissue_samples_greens_from_cext_state",
    ),
    "estimate_bulk_tissue_concentration": (
        ".greens",
        "estimate_bulk_tissue_concentration",
    ),
    "compute_concentration_metrics": (".metrics", "compute_concentration_metrics"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
