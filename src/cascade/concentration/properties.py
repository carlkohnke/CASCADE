"""Shared fluid and inlet-concentration properties for oxygen transport."""

from __future__ import annotations

from cascade.configuration import solver_state as _state


def get_analysis_fluids() -> tuple[str, ...]:
    mode = str(_state.FLUID).lower()
    if mode == "both":
        return ("water", "blood")
    if mode in {"water", "blood"}:
        return (mode,)
    raise ValueError('FLUID must be "water", "blood", or "both".')


def get_concentration_inlet(fluid: str | None = None) -> float:
    mode = str(fluid or _state.ACTIVE_FLUID).lower()
    if mode == "both":
        raise ValueError("Active fluid must be 'water' or 'blood'.")
    try:
        return float(_state.CONCENTRATION_INLET_BY_FLUID[mode])
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown fluid mode: {mode}") from exc


__all__ = ["get_analysis_fluids", "get_concentration_inlet"]
