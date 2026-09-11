"""Stable tabular output schemas and aggregate metric definitions.

The exporting layer copies schema values from the central solver state and
never mutates run configuration merely by being imported.
"""

from __future__ import annotations

from cascade.configuration import solver_state as _state

CSV_FIELDNAMES = tuple(_state.CSV_FIELDNAMES)
AGGREGATED_METRICS = tuple(_state.AGGREGATED_METRICS)
DNC_FIT_FIELDNAMES = tuple(_state.DNC_FIT_FIELDNAMES)
NONDIMENSIONAL_FIELDNAMES = tuple(_state.NONDIMENSIONAL_FIELDNAMES)

__all__ = [
    "AGGREGATED_METRICS",
    "CSV_FIELDNAMES",
    "DNC_FIT_FIELDNAMES",
    "NONDIMENSIONAL_FIELDNAMES",
]
