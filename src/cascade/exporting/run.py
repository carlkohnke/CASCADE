"""Write simulation tables, meshes, metadata, and manifests to a run directory."""

from __future__ import annotations

import csv
import datetime as dt
from functools import lru_cache
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

from cascade import __version__
from cascade.configuration.schema import RunConfig
from cascade.utils.resources import resolve_path
from cascade.utils.hashing import file_sha256
from cascade.vessels.cache import save_network_if_requested
from cascade.vessels.results import NetworkBuildResult
from cascade.exporting.schema import CSV_FIELDNAMES
from cascade.exporting.domain import save_domain_geometry

if TYPE_CHECKING:
    from cascade.simulation.engine import SimulationResult


def export_run(
    config: RunConfig, build: NetworkBuildResult, simulation: SimulationResult
) -> dict[str, str]:
    """Write the output formats enabled for a completed simulation.

    All files are derived from the same ``SimulationResult`` so CSV, NumPy,
    VTK, and manifest outputs retain consistent segment and point identifiers.
    The returned mapping records only files that were actually written.
    """

    out_dir = resolve_path(
        config.outputs.out_dir,
        base_dir=config.settings_path.parent if config.settings_path else None,
    )
    if out_dir is None:
        raise ValueError("outputs.out_dir must identify an output directory.")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    network_path = save_network_if_requested(build, config)
    if network_path is not None:
        outputs["network"] = str(network_path)
    if build.cache_path is not None:
        outputs["simulation_cache"] = str(build.cache_path)

    if config.outputs.write_summary_csv:
        path = out_dir / "summary.csv"
        _write_csv(
            path, simulation.summary_rows, preferred_fields=_summary_fieldnames()
        )
        outputs["summary_csv"] = str(path)

    if config.outputs.write_segments_csv:
        path = out_dir / "segments.csv"
        if simulation.segment_rows:
            _write_csv(path, simulation.segment_rows)
        else:
            _write_segment_csv_from_results(path, simulation.tree_results)
        outputs["segments_csv"] = str(path)

    if config.outputs.write_points_csv and simulation.point_rows:
        path = out_dir / "points.csv"
        _write_csv(path, simulation.point_rows)
        outputs["points_csv"] = str(path)

    if config.outputs.write_paraview:
        if (
            simulation.segment_rows or simulation.tree_results
        ) and config.outputs.write_vessels_vtp:
            path = out_dir / "vessels.vtp"
            float_dtype = _float_dtype(config.outputs.export_float_dtype)
            index_dtype = _index_dtype(config.outputs.export_index_dtype)
            poly = (
                _segment_polydata(
                    simulation.segment_rows,
                    resolution=max(int(config.outputs.vessel_resolution), 2),
                    float_dtype=float_dtype,
                    index_dtype=index_dtype,
                    tree_results=simulation.tree_results,
                )
                if simulation.segment_rows
                else _segment_polydata_from_results(
                    simulation.tree_results,
                    resolution=max(int(config.outputs.vessel_resolution), 2),
                    float_dtype=float_dtype,
                    index_dtype=index_dtype,
                )
            )
            poly.save(str(path))
            outputs["vessels_vtp"] = str(path)
        if (
            simulation.point_data or simulation.point_rows
        ) and config.outputs.write_tissue_vtp:
            path = out_dir / "oxygen_points.vtp"
            if simulation.point_data:
                poly = _points_polydata_from_data(
                    simulation.point_data,
                    float_dtype=_float_dtype(config.outputs.export_float_dtype),
                    index_dtype=_index_dtype(config.outputs.export_index_dtype),
                )
            else:
                poly = _points_polydata(
                    simulation.point_rows,
                    float_dtype=_float_dtype(config.outputs.export_float_dtype),
                    index_dtype=_index_dtype(config.outputs.export_index_dtype),
                )
            poly.save(str(path))
            outputs["oxygen_points_vtp"] = str(path)
        outputs.update(_save_domain_outputs(build.domain, out_dir))

    manifest_path = out_dir / "manifest.json"
    manifest = _manifest(config, build, simulation, outputs)
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, allow_nan=True), encoding="utf-8"
    )
    outputs["manifest_json"] = str(manifest_path)
    return outputs


def _write_csv(
    path: Path, rows: list[dict[str, Any]], *, preferred_fields: list[str] | None = None
) -> None:
    fields: list[str] = []
    for name in preferred_fields or []:
        if name not in fields:
            fields.append(name)
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_fieldnames() -> list[str]:
    return list(CSV_FIELDNAMES)


def _write_segment_csv_from_results(path: Path, tree_results: list[Any]) -> None:
    """Stream per-vessel CSV rows without retaining one Python dict per vessel."""
    fields = [
        "global_segment_id",
        "tree_id",
        "local_segment_id",
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
        "radius_cm",
        "length_cm",
        "flow_cm3_s",
        "flow_ul_min",
        "pressure_pa",
        "cin",
        "cout",
        "pressure_in_root",
        "pressure_out_terminals",
    ]
    optional = [
        name
        for name in ("discharge_hematocrit", "tube_hematocrit")
        if any(name in (getattr(item, "details", {}) or {}) for item in tree_results)
    ]
    fields.extend(optional)
    global_id = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for result in tree_results:
            details = getattr(result, "details", {}) or {}
            starts = np.asarray(details.get("starts", np.empty((0, 3))), dtype=float)
            ends = np.asarray(details.get("ends", np.empty((0, 3))), dtype=float)
            count = int(starts.shape[0])
            arrays = {
                name: _detail_array(details, source, count)
                for name, source in (
                    ("radius_cm", "radii"),
                    ("length_cm", "lengths"),
                    ("flow_cm3_s", "flows"),
                    ("pressure_pa", "pressures"),
                    ("cin", "cin"),
                    ("cout", "cout"),
                    *[(name, name) for name in optional],
                )
            }
            summary = getattr(result, "summary", {}) or {}
            root_pressure = summary.get("pressure_in_root", np.nan)
            outlet_pressure = summary.get("pressure_out_terminals", np.nan)
            tree_id = int(getattr(result, "tree_id", 0))
            for local_id in range(count):
                flow = arrays["flow_cm3_s"][local_id]
                row = [
                    global_id,
                    tree_id,
                    local_id,
                    *starts[local_id],
                    *ends[local_id],
                    arrays["radius_cm"][local_id],
                    arrays["length_cm"][local_id],
                    flow,
                    flow * 60000.0,
                    arrays["pressure_pa"][local_id] * 0.1,
                    arrays["cin"][local_id],
                    arrays["cout"][local_id],
                    root_pressure,
                    outlet_pressure,
                ]
                row.extend(arrays[name][local_id] for name in optional)
                writer.writerow(row)
                global_id += 1


def _segment_polydata(
    rows: list[dict[str, Any]],
    *,
    resolution: int,
    float_dtype: np.dtype,
    index_dtype: np.dtype,
    tree_results: list[Any] | None = None,
) -> pv.PolyData:
    starts = np.array(
        [[r["start_x"], r["start_y"], r["start_z"]] for r in rows], dtype=float_dtype
    )
    ends = np.array(
        [[r["end_x"], r["end_y"], r["end_z"]] for r in rows], dtype=float_dtype
    )
    nseg = int(starts.shape[0])
    if nseg == 0:
        return pv.PolyData()

    def arr(name: str, default=np.nan, dtype=float_dtype):
        return np.asarray([r.get(name, default) for r in rows], dtype=dtype)

    profiles: dict[int, dict[str, Any]] = {}
    for result in tree_results or []:
        details = getattr(result, "details", {}) or {}
        profile = details.get("vessel_quadrature")
        if isinstance(profile, dict) and "gl_points_si" in profile:
            profiles[int(getattr(result, "tree_id", len(profiles)))] = profile

    cin = arr("cin")
    cout = arr("cout")
    point_chunks: list[np.ndarray] = []
    line_chunks: list[np.ndarray] = []
    concentration_chunks: list[np.ndarray] = []
    field_chunks: dict[str, list[np.ndarray]] = {
        "intralumen_oxygen": [],
        "bulk_oxygen": [],
        "wall_oxygen": [],
        "external_oxygen": [],
    }
    quadrature_node_chunks: list[np.ndarray] = []
    quadrature_order_chunks: list[np.ndarray] = []
    counts: list[int] = []
    offset = 0
    minimum_resolution = max(int(resolution), 2)
    profile_names = {
        "c_iv_gl": "intralumen_oxygen",
        "c_bulk_gl": "bulk_oxygen",
        "c_wall_gl": "wall_oxygen",
        "c_ext_gl": "external_oxygen",
    }
    for segment_id, row in enumerate(rows):
        start = np.asarray(starts[segment_id], dtype=float)
        end = np.asarray(ends[segment_id], dtype=float)
        vector = end - start
        length_sq = float(np.dot(vector, vector))
        tree_id = int(row.get("tree_id", -1))
        local_id = int(row.get("local_segment_id", -1))
        profile = profiles.get(tree_id)
        source_t = np.empty((0,), dtype=float)
        if profile is not None:
            gl_points = np.asarray(profile.get("gl_points_si"), dtype=float)
            if gl_points.ndim == 3 and 0 <= local_id < gl_points.shape[0]:
                points_cm = gl_points[local_id] * 100.0
                if length_sq > 0.0:
                    source_t = ((points_cm - start) @ vector) / length_sq
                    source_t = np.clip(source_t, 0.0, 1.0)

        base_t = np.linspace(0.0, 1.0, minimum_resolution, dtype=float)
        target_t = np.unique(np.concatenate((base_t, source_t)))
        target_points = start[None, :] + target_t[:, None] * vector[None, :]
        count = int(target_t.size)
        counts.append(count)
        point_chunks.append(target_points.astype(float_dtype, copy=False))
        line_chunks.append(
            np.concatenate(
                (
                    np.asarray([count], dtype=np.int64),
                    np.arange(offset, offset + count, dtype=np.int64),
                )
            )
        )
        offset += count

        order = int(source_t.size)
        quadrature_order_chunks.append(np.full(count, order, dtype=index_dtype))
        quadrature_node_chunks.append(
            np.any(
                np.isclose(target_t[:, None], source_t[None, :], rtol=1e-6, atol=1e-7),
                axis=1,
            ).astype(index_dtype)
            if order
            else np.zeros(count, dtype=index_dtype)
        )

        values_for_field: dict[str, np.ndarray] = {}
        if profile is not None and order:
            order_idx = np.argsort(source_t, kind="stable")
            sorted_t = source_t[order_idx]
            for source_name, output_name in profile_names.items():
                values = np.asarray(profile.get(source_name, ()), dtype=float)
                if (
                    values.ndim == 2
                    and 0 <= local_id < values.shape[0]
                    and values.shape[1] == order
                ):
                    sorted_values = values[local_id][order_idx]
                    values_for_field[output_name] = np.interp(
                        target_t,
                        sorted_t,
                        sorted_values,
                        left=float(sorted_values[0]),
                        right=float(sorted_values[-1]),
                    )
        fallback = np.linspace(float(cin[segment_id]), float(cout[segment_id]), count)
        concentration_chunks.append(
            values_for_field.get(
                "bulk_oxygen",
                values_for_field.get("intralumen_oxygen", fallback),
            )
        )
        for output_name in field_chunks:
            field_chunks[output_name].append(
                values_for_field.get(output_name, np.full(count, np.nan, dtype=float))
            )

    points = np.concatenate(point_chunks, axis=0)
    lines = np.concatenate(line_chunks, axis=0)
    poly = pv.PolyData(points, lines=lines)
    poly.point_data["concentration"] = np.concatenate(concentration_chunks).astype(
        float_dtype, copy=False
    )
    for name, chunks in field_chunks.items():
        values = np.concatenate(chunks).astype(float_dtype, copy=False)
        if np.any(np.isfinite(values)):
            poly.point_data[name] = values
    # Retain quadrature provenance in exported VTK for reproducibility; it is
    # intentionally not exposed in the Results field picker.
    poly.point_data["solver_quadrature_node"] = np.concatenate(
        quadrature_node_chunks
    ).astype(index_dtype, copy=False)
    poly.point_data["solver_quadrature_order"] = np.concatenate(
        quadrature_order_chunks
    ).astype(index_dtype, copy=False)
    for key in ("flow_cm3_s", "flow_ul_min", "pressure_pa", "radius_cm", "length_cm"):
        poly.point_data[key] = np.repeat(arr(key), counts).astype(
            float_dtype, copy=False
        )
    for key in ("tree_id", "local_segment_id", "global_segment_id"):
        poly.point_data[key] = np.repeat(
            arr(key, default=-1, dtype=index_dtype), counts
        ).astype(index_dtype, copy=False)
    if any("discharge_hematocrit" in r for r in rows):
        poly.point_data["discharge_hematocrit"] = np.repeat(
            arr("discharge_hematocrit"), counts
        )
    if any("tube_hematocrit" in r for r in rows):
        poly.point_data["tube_hematocrit"] = np.repeat(arr("tube_hematocrit"), counts)
    return poly


def _segment_polydata_from_results(
    tree_results: list[Any],
    *,
    resolution: int,
    float_dtype: np.dtype,
    index_dtype: np.dtype,
) -> pv.PolyData:
    """Vectorize VTK vessel construction without per-segment Python dicts."""
    point_chunks: list[np.ndarray] = []
    line_chunks: list[np.ndarray] = []
    fields: dict[str, list[np.ndarray]] = {}
    global_segment_offset = 0
    point_offset = 0
    minimum_resolution = max(int(resolution), 2)
    profile_names = {
        "c_iv_gl": "intralumen_oxygen",
        "c_bulk_gl": "bulk_oxygen",
        "c_wall_gl": "wall_oxygen",
        "c_ext_gl": "external_oxygen",
    }
    optional_fields = {
        name
        for name in ("discharge_hematocrit", "tube_hematocrit")
        if any(name in (getattr(item, "details", {}) or {}) for item in tree_results)
    }

    def add(name: str, values: np.ndarray) -> None:
        fields.setdefault(name, []).append(np.asarray(values).reshape(-1))

    for result in tree_results:
        details = getattr(result, "details", {}) or {}
        starts = np.asarray(details.get("starts", np.empty((0, 3))), dtype=float)
        ends = np.asarray(details.get("ends", np.empty((0, 3))), dtype=float)
        nseg = int(starts.shape[0])
        if nseg == 0:
            continue
        vectors = ends - starts
        profile = details.get("vessel_quadrature")
        gl_points = (
            np.asarray(profile.get("gl_points_si"), dtype=float)
            if isinstance(profile, dict)
            else np.empty((0, 0, 3), dtype=float)
        )
        order = int(gl_points.shape[1]) if gl_points.ndim == 3 else 0
        source_t = np.empty((0,), dtype=float)
        if order and gl_points.shape[0] >= nseg:
            length_sq = np.einsum("ij,ij->i", vectors, vectors)
            valid = np.flatnonzero(length_sq > 0.0)
            if valid.size:
                sample = int(valid[0])
                source_t = (
                    (gl_points[sample] * 100.0 - starts[sample])
                    @ vectors[sample]
                ) / length_sq[sample]
                source_t = np.clip(source_t, 0.0, 1.0)
        base_t = np.linspace(0.0, 1.0, minimum_resolution, dtype=float)
        target_t = np.unique(np.concatenate((base_t, source_t)))
        count = int(target_t.size)
        points = starts[:, None, :] + target_t[None, :, None] * vectors[:, None, :]
        point_chunks.append(points.reshape(-1, 3).astype(float_dtype, copy=False))
        ids = np.arange(point_offset, point_offset + nseg * count, dtype=np.int64)
        lines = np.column_stack(
            (np.full(nseg, count, dtype=np.int64), ids.reshape(nseg, count))
        )
        line_chunks.append(lines.reshape(-1))
        point_offset += nseg * count

        cin = _detail_array(details, "cin", nseg)
        cout = _detail_array(details, "cout", nseg)
        fallback = cin[:, None] + (cout - cin)[:, None] * target_t[None, :]
        interpolated: dict[str, np.ndarray] = {}
        if source_t.size:
            order_idx = np.argsort(source_t, kind="stable")
            sorted_t = source_t[order_idx]
            for source_name, output_name in profile_names.items():
                raw = np.asarray(profile.get(source_name, ()), dtype=float)
                if raw.ndim == 2 and raw.shape[0] >= nseg and raw.shape[1] == order:
                    interpolated[output_name] = _interp_rows(
                        sorted_t, raw[:nseg, order_idx], target_t
                    )
        add(
            "concentration",
            interpolated.get(
                "bulk_oxygen", interpolated.get("intralumen_oxygen", fallback)
            ),
        )
        for name in profile_names.values():
            add(name, interpolated.get(name, np.full((nseg, count), np.nan)))
        is_node = (
            np.any(
                np.isclose(
                    target_t[:, None], source_t[None, :], rtol=1e-6, atol=1e-7
                ),
                axis=1,
            )
            if source_t.size
            else np.zeros(count, dtype=bool)
        )
        add("solver_quadrature_node", np.tile(is_node, nseg).astype(index_dtype))
        add(
            "solver_quadrature_order",
            np.full(nseg * count, order, dtype=index_dtype),
        )
        scalar_fields = {
            "flow_cm3_s": _detail_array(details, "flows", nseg),
            "flow_ul_min": _detail_array(details, "flows", nseg) * 60000.0,
            "pressure_pa": _detail_array(details, "pressures", nseg) * 0.1,
            "radius_cm": _detail_array(details, "radii", nseg),
            "length_cm": _detail_array(details, "lengths", nseg),
        }
        for name, values in scalar_fields.items():
            add(name, np.repeat(values, count))
        add(
            "tree_id",
            np.full(nseg * count, int(getattr(result, "tree_id", 0)), dtype=index_dtype),
        )
        add("local_segment_id", np.repeat(np.arange(nseg), count))
        add(
            "global_segment_id",
            np.repeat(np.arange(global_segment_offset, global_segment_offset + nseg), count),
        )
        for source_name in optional_fields:
            add(
                source_name,
                np.repeat(_detail_array(details, source_name, nseg), count),
            )
        global_segment_offset += nseg

    if not point_chunks:
        return pv.PolyData()
    poly = pv.PolyData(np.concatenate(point_chunks), lines=np.concatenate(line_chunks))
    integer_fields = {
        "solver_quadrature_node",
        "solver_quadrature_order",
        "tree_id",
        "local_segment_id",
        "global_segment_id",
    }
    for name, chunks in fields.items():
        values = np.concatenate(chunks)
        if name in integer_fields:
            poly.point_data[name] = values.astype(index_dtype, copy=False)
        elif name not in profile_names.values() or np.any(np.isfinite(values)):
            poly.point_data[name] = values.astype(float_dtype, copy=False)
    return poly


def _detail_array(details: dict[str, Any], name: str, count: int) -> np.ndarray:
    values = np.asarray(details.get(name, ()), dtype=float).reshape(-1)
    if values.size >= count:
        return values[:count]
    out = np.full(count, np.nan, dtype=float)
    out[: values.size] = values
    return out


def _interp_rows(
    source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of one ``np.interp`` call per segment."""
    if source_t.size == 1:
        return np.repeat(values[:, :1], target_t.size, axis=1)
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 1, source_t.size - 1)
    left = right - 1
    denominator = source_t[right] - source_t[left]
    weight = np.divide(
        target_t - source_t[left],
        denominator,
        out=np.zeros_like(target_t),
        where=denominator != 0.0,
    )
    result = values[:, left] * (1.0 - weight) + values[:, right] * weight
    result[:, target_t <= source_t[0]] = values[:, :1]
    result[:, target_t >= source_t[-1]] = values[:, -1:]
    return result


def _points_polydata(
    rows: list[dict[str, Any]], *, float_dtype: np.dtype, index_dtype: np.dtype
) -> pv.PolyData:
    points = np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=float_dtype)
    n = int(points.shape[0])
    verts = np.column_stack(
        [np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]
    ).reshape(-1)
    poly = pv.PolyData(points, verts=verts)
    for key in rows[0].keys():
        if key in {"x", "y", "z"}:
            continue
        values = [r.get(key) for r in rows]
        if key.endswith("_id") or key in {"point_id", "viability", "inside_tissue"}:
            poly.point_data[key] = np.asarray(values, dtype=index_dtype)
        else:
            poly.point_data[key] = np.asarray(values, dtype=float_dtype)
    return poly


def _points_polydata_from_data(
    data: dict[str, np.ndarray], *, float_dtype: np.dtype, index_dtype: np.dtype
) -> pv.PolyData:
    """Write VTK point fields directly from vectorized simulation arrays."""
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(
        float_dtype, copy=False
    )
    n = int(points.shape[0])
    verts = np.column_stack(
        [np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]
    ).reshape(-1)
    poly = pv.PolyData(points, verts=verts)
    # Keep coordinates as explicit point fields as well as mesh geometry.  This
    # is part of the established tissue-output schema used by validation tools.
    for coordinate_index, coordinate_name in enumerate(("x", "y", "z")):
        poly.point_data[coordinate_name] = points[:, coordinate_index]
    for key, raw_values in data.items():
        if key in {"x", "y", "z"}:
            continue
        values = np.asarray(raw_values)[:n]
        if key.endswith("_id") or key in {"point_id", "viability", "inside_tissue"}:
            poly.point_data[key] = values.astype(index_dtype, copy=False)
        else:
            poly.point_data[key] = values.astype(float_dtype, copy=False)
    return poly


def _save_domain_outputs(domain, out_dir: Path) -> dict[str, str]:
    saved = save_domain_geometry(
        domain,
        out_dir,
        boundary_filename="domain_boundary.vtp",
        mesh_filename="domain_mesh.vtu",
    )
    outputs: dict[str, str] = {}
    if "boundary" in saved:
        outputs["domain_boundary_vtp"] = saved["boundary"]
    if "mesh" in saved:
        outputs["domain_mesh_vtu"] = saved["mesh"]
    return outputs


def _manifest(
    config: RunConfig,
    build: NetworkBuildResult,
    simulation: SimulationResult,
    outputs: dict[str, str],
) -> dict[str, Any]:
    try:
        import svv

        svv_version = getattr(svv, "__version__", "unknown")
    except Exception:
        svv_version = "unknown"
    return {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cascade": {
            "version": __version__,
            "source": _git_metadata(),
        },
        "settings_path": str(config.settings_path) if config.settings_path else None,
        "inputs": {
            "settings_sha256": _file_sha256(config.settings_path),
            "network_path": str(build.load_source) if build.load_source else None,
            "network_sha256": _file_sha256(build.load_source),
            "domain_path": config.domain.path,
            "domain_sha256": _file_sha256(
                resolve_path(
                    config.domain.path,
                    base_dir=config.settings_path.parent
                    if config.settings_path
                    else None,
                )
                if config.domain.path
                else None
            ),
        },
        "settings": config.raw,
        "resolved": {
            "network_mode": config.network_mode,
            "prefix": config.prefix,
            "target_counts": build.target_counts,
            "compute_float_policy": "float64",
            "export_float_dtype": config.outputs.export_float_dtype,
            "export_index_dtype": config.outputs.export_index_dtype,
            "simple_mode": "enabled"
            if config.network_mode == "simple"
            else "not_requested",
            "tissue_sample": build.sample_meta or simulation.sample_meta,
            "runtime_settings": config.runtime_setting_overrides,
            "runtime_setting_warnings": config.runtime_setting_warnings,
        },
        "dependencies": {
            "svv_version": svv_version,
            **_dependency_versions(),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_path": os.environ.get("CUDA_PATH"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
            "random_seed": int(config.domain.random_seed),
        },
        "timings": {
            **build.build_timings,
            **simulation.timings,
            "tree_solve_s": [r.elapsed_s for r in simulation.tree_results],
        },
        "network": {
            "load_source": str(build.load_source) if build.load_source else None,
            "saved_path": str(build.network_path) if build.network_path else None,
            "cache_path": str(build.cache_path) if build.cache_path else None,
            "tree_count": len(build.trees),
            "segments": [
                int(getattr(tree, "segment_count", 0) or 0) for tree in build.trees
            ],
            "terminals": [
                int(getattr(tree, "n_terminals", 0) or 0) for tree in build.trees
            ],
            "connectivity_repairs": build.connectivity_repairs or [],
            "connectivity_reports": build.connectivity_reports or [],
        },
        "simulation": {
            "external_field": simulation.external_field,
            "interventions": simulation.interventions,
        },
        "outputs": outputs,
        "geometry_only": bool(config.simulation.geometry_only),
        "skip_tissue_oxygen": bool(config.simulation.skip_tissue_oxygen),
    }


@lru_cache(maxsize=1)
def _dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in (
        "numpy",
        "scipy",
        "numba",
        "pyvista",
        "vtk",
        "cupy-cuda13x",
        "cupy-cuda12x",
        "cupy-cuda11x",
    ):
        key = distribution.replace("-", "_") + "_version"
        try:
            result[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[key] = None
    return result


@lru_cache(maxsize=1)
def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return None
    stat = candidate.stat()
    return _file_sha256_for_signature(
        str(candidate), int(stat.st_size), int(stat.st_mtime_ns)
    )


@lru_cache(maxsize=16)
def _file_sha256_for_signature(path: str, _size: int, _mtime_ns: int) -> str:
    """Avoid re-reading unchanged large inputs for every serial manifest."""
    return file_sha256(Path(path))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float_dtype(name: str) -> np.dtype:
    return np.dtype(np.float32 if str(name).lower() == "float32" else np.float64)


def _index_dtype(name: str) -> np.dtype:
    return np.dtype(np.int32 if str(name).lower() == "int32" else np.int64)
