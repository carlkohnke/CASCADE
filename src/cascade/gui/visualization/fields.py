"""Result mesh extraction, scaling, legends, and color mapping."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PySide6.QtGui import QColor

from cascade.gui.visualization.palette import _MAP_STOPS


def _mesh_line_segments(mesh) -> np.ndarray:
    if mesh is None or not getattr(mesh, "n_points", 0):
        return np.empty((0, 2, 3), dtype=np.float32)
    points = np.asarray(mesh.points, dtype=np.float32)
    raw = np.asarray(getattr(mesh, "lines", []), dtype=np.int64)
    segments = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        ids = raw[cursor + 1 : cursor + 1 + count]
        if len(ids) >= 2:
            segments.extend((int(a), int(b)) for a, b in zip(ids[:-1], ids[1:]))
        cursor += count + 1
    if not segments:
        return np.empty((0, 2, 3), dtype=np.float32)
    edge_ids = np.asarray(segments, dtype=np.int64)
    return points[edge_ids]


def _polyline_data(mesh):
    if mesh is None or not getattr(mesh, "n_points", 0):
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty.copy(), {}, empty.copy()
    points = np.asarray(mesh.points, dtype=np.float32)
    raw = np.asarray(getattr(mesh, "lines", []), dtype=np.int64)
    first, last = [], []
    cell_first = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        ids = raw[cursor + 1 : cursor + 1 + count]
        if len(ids) >= 2:
            cell_first.append(int(ids[0]))
            for left, right in zip(ids[:-1], ids[1:]):
                first.append(int(left))
                last.append(int(right))
        cursor += count + 1
    if not first:
        n = len(points) // 2
        first, last = list(range(0, 2 * n, 2)), list(range(1, 2 * n, 2))
    first_arr, last_arr = np.asarray(first), np.asarray(last)
    arrays = {}
    for name, raw_values in getattr(mesh, "point_data", {}).items():
        values = np.asarray(raw_values)
        if (
            values.ndim == 1
            and np.issubdtype(values.dtype, np.number)
            and len(values) == len(points)
        ):
            arrays[str(name)] = (
                values[first_arr].astype(float) + values[last_arr].astype(float)
            ) * 0.5
    raw_local_ids = getattr(mesh, "point_data", {}).get("local_segment_id")
    if raw_local_ids is not None and cell_first:
        raw_local_ids = np.asarray(raw_local_ids)
        inlet_ids = [
            point_id for point_id in cell_first if raw_local_ids[point_id] == 0
        ]
        inlets = (
            points[np.asarray(inlet_ids, dtype=int)]
            if inlet_ids
            else points[np.asarray(cell_first[:1], dtype=int)]
        )
    else:
        inlets = (
            points[np.asarray(cell_first[:1], dtype=int)]
            if cell_first
            else points[first_arr[:1]]
        )
    return points[first_arr], points[last_arr], arrays, inlets


def _numeric_point_arrays(mesh, n):
    arrays = {}
    if mesh is None:
        return arrays
    for name, raw in getattr(mesh, "point_data", {}).items():
        values = np.asarray(raw)
        if (
            values.ndim == 1
            and len(values) == n
            and np.issubdtype(values.dtype, np.number)
        ):
            arrays[str(name)] = values.astype(float, copy=False)
    return arrays


def _line_array(value):
    if value is None:
        return np.empty((0, 2, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return (
        array.reshape(-1, 2, 3) if array.size else np.empty((0, 2, 3), dtype=np.float32)
    )


def _triangle_array(value):
    if value is None:
        return np.empty((0, 3, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return (
        array.reshape(-1, 3, 3) if array.size else np.empty((0, 3, 3), dtype=np.float32)
    )


def _point_array(value):
    if value is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1, 3) if array.size else np.empty((0, 3), dtype=np.float32)


def _values(value, n, fill=np.nan):
    if value is None:
        return None if np.isnan(fill) else np.full(n, fill, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(array) >= n:
        return array[:n]
    return np.pad(array, (0, n - len(array)), constant_values=fill)


def _normalize_with_scale(
    values: np.ndarray | None,
    requested_min: float | None = None,
    requested_max: float | None = None,
    scale: str = "linear",
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """Normalize a scalar field while keeping its displayed limits explicit."""
    if values is None or not len(values):
        return None, None
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 0.5), None
    log_scale = str(scale).lower() == "log"
    valid = finite & (values > 0.0) if log_scale else finite
    if not valid.any():
        return np.full(len(values), 0.5), None
    source = values[valid]
    auto_lo, auto_hi = np.nanpercentile(source, [2, 98])
    lo = (
        float(requested_min)
        if requested_min is not None and math.isfinite(requested_min)
        else float(auto_lo)
    )
    hi = (
        float(requested_max)
        if requested_max is not None and math.isfinite(requested_max)
        else float(auto_hi)
    )
    if log_scale and lo <= 0.0:
        # A zero linear bound cannot be represented on a log legend.  Use the
        # smallest available positive scalar instead of producing an invalid
        # scale or silently hiding the field.
        lo = float(np.nanmin(source))
    if hi <= lo:
        hi = float(np.nanmax(source))
    if hi <= lo:
        return np.full(len(values), 0.5), (lo, hi)
    if log_scale:
        logged = np.zeros(len(values), dtype=float)
        logged[valid] = np.log10(values[valid])
        norm = np.zeros(len(values), dtype=float)
        norm[valid] = (logged[valid] - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo)
        )
    else:
        norm = (values - lo) / (hi - lo)
    return np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0), (lo, hi)


def _scientific(value: float) -> str:
    return f"{float(value):.2e}"


def _display_field_values(
    kind: str, field: str, cache: dict[str, Any], options: dict[str, Any]
) -> tuple[np.ndarray | None, str]:
    """Return user-facing scalar values; never expose VTK/solver implementation names."""
    arrays = cache["vessel_arrays"] if kind == "vessel" else cache["tissue_arrays"]
    source_map = {
        "flow": "flow_ul_min",
        "pressure": "pressure_pa",
        "bulk_concentration": "concentration",
        "wall_concentration": "wall_oxygen",
        "radius": "radius_cm",
        "length": "length_cm",
        "hematocrit": "discharge_hematocrit",
        "tissue_concentration": "local_concentration",
        "viability": "viability",
        "distance": "dnc_cm",
    }
    labels = {
        "flow": "Flow Rate",
        "pressure": "Fluid pressure",
        "bulk_concentration": "Bulk concentration",
        "wall_concentration": "Wall concentration",
        "radius": "Radius",
        "length": "Length",
        "hematocrit": "Discharge hematocrit",
        "tissue_concentration": "Tissue concentration",
        "viability": "Viable tissue",
        "distance": "Distance to nearest vessel",
    }
    raw = arrays.get(source_map.get(field, field))
    if raw is None:
        return None, labels.get(field, str(field).replace("_", " ").title())
    values = np.asarray(raw, dtype=float).reshape(-1)
    if field == "flow":
        if options.get("flow_unit") == "cm3_s":
            values = values / 60_000.0
            unit = "cm³/s"
        else:
            unit = "μL/min"
    elif field == "pressure":
        values = values / 133.322387415
        unit = "mmHg"
    elif field in {"radius", "length", "distance"}:
        values = values * 10_000.0
        unit = "μm"
    elif field in {"bulk_concentration", "wall_concentration", "tissue_concentration"}:
        if options.get("concentration_unit") == "mmhg":
            values = values / 0.001408
            unit = "mmHg equivalent"
        else:
            unit = "mol/m³"
    else:
        unit = ""
    if options.get("normalize_fields") and field in {
        "flow",
        "pressure",
        "bulk_concentration",
        "wall_concentration",
        "tissue_concentration",
    }:
        settings = cache.get("settings", {})
        if field == "flow":
            denominator = float(
                settings.get("simulation", {}).get("qin_target_ul_min", 1.0)
            )
            if options.get("flow_unit") == "cm3_s":
                denominator /= 60_000.0
        elif field == "pressure":
            denominator = (
                float(
                    settings.get("settings", {})
                    .get("hemodynamics", {})
                    .get("root_pressure", 1.0)
                )
                / 133.322387415
            )
        else:
            denominator = float(
                settings.get("settings", {})
                .get("oxygen", {})
                .get("conc_max_for_normalization", 1.0)
            )
            if options.get("concentration_unit") == "mmhg":
                denominator /= 0.001408
        if math.isfinite(denominator) and denominator != 0.0:
            values = values / denominator
        unit = "normalized"
    label = labels.get(field, str(field).replace("_", " ").title())
    return values, f"{label} ({unit})" if unit else label


def _layer_field_label(layer: str, field: str) -> str:
    return f"{layer}: {field}" if field else layer


def _legend_tick_values(
    limits: tuple[float, float], scale: str, count: int = 5
) -> np.ndarray:
    """Return evenly spaced legend labels in the active display space."""
    lo, hi = (float(limits[0]), float(limits[1]))
    if count < 2 or not math.isfinite(lo) or not math.isfinite(hi):
        return np.asarray([lo, hi], dtype=float)
    if str(scale).lower() == "log" and lo > 0.0 and hi > 0.0:
        return np.geomspace(lo, hi, count)
    return np.linspace(lo, hi, count)


def _map_color(name, value):
    stops = _MAP_STOPS.get(name, _MAP_STOPS["viridis"])
    value = float(np.clip(value, 0.0, 1.0))
    scaled = value * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    frac = scaled - index
    rgb = tuple(
        int(round(stops[index][i] * (1 - frac) + stops[index + 1][i] * frac))
        for i in range(3)
    )
    return QColor(*rgb)


__all__ = (
    "_mesh_line_segments",
    "_polyline_data",
    "_numeric_point_arrays",
    "_line_array",
    "_triangle_array",
    "_point_array",
    "_values",
    "_normalize_with_scale",
    "_scientific",
    "_display_field_values",
    "_layer_field_label",
    "_legend_tick_values",
    "_map_color",
)
