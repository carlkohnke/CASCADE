"""Flow-boundary resolution shared by every simulation shape."""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


def tree_root_flow_cm3_s(tree: Any) -> float:
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


def allocate_inlet_flows(
    networks: Iterable[Any],
    *,
    source: str,
    prescribed_per_network: Sequence[float] | None = None,
    total_qin_cm3_s: float | None = None,
    require_stored_root_flow: bool = False,
) -> list[float]:
    """Resolve inlet flows for an arbitrary one- or multi-network case."""
    items = tuple(networks)
    if not items:
        return []
    mode = str(source).strip().lower().replace("_", "-")
    stored = np.asarray([tree_root_flow_cm3_s(tree) for tree in items], dtype=float)
    if mode == "tree-root-flow":
        missing = np.flatnonzero(~np.isfinite(stored) | (stored <= 0.0))
        if missing.size and require_stored_root_flow:
            indices = ", ".join(str(int(index)) for index in missing[:10])
            raise ValueError(
                "Tree-root flow was requested, but no positive stored root flow "
                f"exists for network index(es): {indices}."
            )
        if not missing.size:
            return stored.astype(float).tolist()
        mode = "per-tree"
    if mode == "total-qin-split":
        if total_qin_cm3_s is None or not np.isfinite(float(total_qin_cm3_s)):
            raise ValueError("total_qin_cm3_s is required for total-qin-split")
        weights = np.where(np.isfinite(stored) & (stored > 0.0), stored, 0.0)
        if not np.any(weights > 0.0):
            weights = np.ones(len(items), dtype=float)
        return (float(total_qin_cm3_s) * weights / float(np.sum(weights))).tolist()
    values = list(prescribed_per_network or ())
    if len(values) != len(items):
        raise ValueError(
            f"Expected {len(items)} prescribed inlet flows; received {len(values)}"
        )
    resolved = [float(value) for value in values]
    if not all(np.isfinite(value) and value > 0.0 for value in resolved):
        raise ValueError("Prescribed inlet flows must be finite and positive")
    return resolved


__all__ = ["allocate_inlet_flows", "tree_root_flow_cm3_s"]
