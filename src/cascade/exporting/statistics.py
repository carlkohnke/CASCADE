"""Trial aggregation and statistical summaries.

Statistical helpers are pure calculations over result rows.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from cascade.configuration import solver_state as _state


def _linear_fit_slope_r2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0.0:
        return float(slope), float(intercept), float("nan")
    r2 = 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def _constant_value(rows: list[Dict[str, float]], key: str):
    values = {row[key] for row in rows}
    if not values:
        return float("nan")
    if len(values) > 1:
        raise ValueError(f"Inconsistent values for '{key}': {values}")
    return values.pop()


def _mean_std(rows: list[Dict[str, float]], key: str) -> tuple[float, float]:
    data = np.asarray([row.get(key, float("nan")) for row in rows], dtype=float)
    return float(np.nanmean(data)), float(np.nanstd(data, ddof=1))


def _pooled_mean_std_from_summaries(
    rows: list[Dict[str, float]],
    *,
    mean_key: str,
    std_key: str,
    count_key: str,
) -> tuple[float, float]:
    summaries = []
    for row in rows:
        count = float(row.get(count_key, float("nan")))
        mean = float(row.get(mean_key, float("nan")))
        std = float(row.get(std_key, float("nan")))
        if np.isfinite(count) and count > 0.0 and np.isfinite(mean):
            summaries.append((count, mean, std if np.isfinite(std) else 0.0))
    if not summaries:
        return float("nan"), float("nan")
    total = float(sum(count for count, _, _ in summaries))
    if total <= 0.0:
        return float("nan"), float("nan")
    pooled_mean = float(sum(count * mean for count, mean, _ in summaries) / total)
    if total <= 1.0:
        return pooled_mean, 0.0
    ss = 0.0
    for count, mean, std in summaries:
        ss += max(count - 1.0, 0.0) * float(std) ** 2
        ss += count * (float(mean) - pooled_mean) ** 2
    return pooled_mean, float(math.sqrt(max(ss, 0.0) / max(total - 1.0, 1.0)))


def aggregate_trials(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("No trial rows provided for aggregation.")
    result: Dict[str, float] = {}
    result["number_of_trees"] = len(rows)
    result["target_terminals"] = _constant_value(rows, "target_terminals")
    result["dlp_angle"] = _constant_value(rows, "dlp_angle")
    result["cube_side_length"] = _constant_value(rows, "cube_side_length")
    result["concentration_solver"] = _constant_value(rows, "concentration_solver")
    result["cext_accel_mode"] = _constant_value(rows, "cext_accel_mode")
    result["pressure_in_root"] = _constant_value(rows, "pressure_in_root")
    result["pressure_out_terminals"] = _constant_value(rows, "pressure_out_terminals")
    result["concentration_inlet"] = _constant_value(rows, "concentration_inlet")
    result["extravascular_concentration"] = _state.EXTRAVASCULAR_CONCENTRATION
    result["solute_diffusivity"] = _state.SOLUTE_DIFFUSIVITY
    # result["tissue_decay_length"] = TISSUE_DECAY_LENGTH
    result["qin_target_uL_per_min"] = _mean_std(rows, "inlet_flow_ul_per_min")[0]
    result["distance_sample_count"] = _constant_value(rows, "distance_sample_count")
    for metric in _state.AGGREGATED_METRICS:
        mean, std = _mean_std(rows, metric)
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std
    cext_mean, cext_std = _pooled_mean_std_from_summaries(
        rows,
        mean_key="cext_concentration_mean",
        std_key="cext_concentration_std",
        count_key="cext_concentration_count",
    )
    result["cext_concentration_mean"] = cext_mean
    result["cext_concentration_std"] = cext_std
    return result


__all__ = [
    "_linear_fit_slope_r2",
    "_constant_value",
    "_mean_std",
    "_pooled_mean_std_from_summaries",
    "aggregate_trials",
]
