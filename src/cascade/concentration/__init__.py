"""Vessel, external-field, and tissue concentration solvers."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "TissueOxygenProblem": (".tissue", "TissueOxygenProblem"),
    "TissueOxygenResult": (".tissue", "TissueOxygenResult"),
    "solve_tissue_oxygen": (".tissue", "solve_tissue_oxygen"),
    "VesselConcentrationProblem": (".vessel", "VesselConcentrationProblem"),
    "VesselConcentrationResult": (".vessel", "VesselConcentrationResult"),
    "solve_vessel_concentration": (".vessel", "solve_vessel_concentration"),
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
