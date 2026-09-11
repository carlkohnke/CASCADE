"""Vessel construction result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class NetworkBuildResult:
    domain: Any
    trees: list[Any]
    forest: Any | None
    target_counts: list[int]
    build_timings: dict[str, float]
    sample_points: np.ndarray | None = None
    sample_meta: dict[str, Any] | None = None
    connectivity_repairs: list[int] | None = None
    connectivity_reports: list[dict[str, Any]] | None = None
    network_path: Path | None = None
    cache_path: Path | None = None
    load_source: Path | None = None


__all__ = ("NetworkBuildResult",)
