"""Shared extraction of prescribed root flow from vascular tree objects."""

from __future__ import annotations

import math

import numpy as np


def tree_root_flow_cm3_s(tree) -> float:
    """Return a valid stored root flow, or NaN when the tree has none."""
    parameters = getattr(tree, "parameters", None)
    if parameters is not None:
        value = getattr(parameters, "root_flow", None)
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)

    data = np.asarray(tree.data[: int(getattr(tree, "segment_count", 0) or 0)])
    if data.size:
        value = float(data[0, 22])
        if np.isfinite(value) and value > 0.0:
            return value
    return math.nan


__all__ = ["tree_root_flow_cm3_s"]
