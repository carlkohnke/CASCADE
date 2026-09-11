"""Intravascular concentration transport."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "VesselConcentrationProblem": (".api", "VesselConcentrationProblem"),
    "VesselConcentrationResult": (".api", "VesselConcentrationResult"),
    "solve_vessel_concentration": (".api", "solve_vessel_concentration"),
    "buffer_factor_B": (".greens", "buffer_factor_B"),
    "segment_O2_capacity": (".greens", "segment_O2_capacity"),
    "segment_O2_capacity_from_HT": (".greens", "segment_O2_capacity_from_HT"),
    "severinghaus_dSdP": (".greens", "severinghaus_dSdP"),
    "severinghaus_saturation": (".greens", "severinghaus_saturation"),
    "solve_network_concentrations": (".network", "solve_network_concentrations"),
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
