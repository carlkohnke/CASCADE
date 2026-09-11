"""Shared, dependency-light contracts used across CASCADE subsystems."""

from .contracts import (
    FlowProblem,
    FlowResult,
    TissueOxygenProblem,
    TissueOxygenResult,
    VesselConcentrationProblem,
    VesselConcentrationResult,
)

__all__ = [
    "FlowProblem",
    "FlowResult",
    "TissueOxygenProblem",
    "TissueOxygenResult",
    "VesselConcentrationProblem",
    "VesselConcentrationResult",
]
