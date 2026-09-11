"""Shared, dependency-light contracts used across CASCADE subsystems."""

from .contracts import (
    ExternalFieldResult,
    FlowProblem,
    FlowResult,
    TissueOxygenProblem,
    TissueOxygenResult,
    VesselConcentrationProblem,
    VesselConcentrationResult,
)

__all__ = [
    "ExternalFieldResult",
    "FlowProblem",
    "FlowResult",
    "TissueOxygenProblem",
    "TissueOxygenResult",
    "VesselConcentrationProblem",
    "VesselConcentrationResult",
]
