from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from .config import RunConfig
from .growth import NetworkBuildResult, resolve_path, save_network_if_requested
from .simulation import SimulationResult


def export_run(config: RunConfig, build: NetworkBuildResult, simulation: SimulationResult) -> dict[str, str]:
    out_dir = resolve_path(config.outputs.out_dir, base_dir=config.settings_path.parent if config.settings_path else None)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    network_path = save_network_if_requested(build, config)
    if network_path is not None:
        outputs["network"] = str(network_path)
    if build.cache_path is not None:
        outputs["simulation_cache"] = str(build.cache_path)

    if config.outputs.write_summary_csv:
        path = out_dir / "summary.csv"
        _write_csv(path, simulation.summary_rows, preferred_fields=_summary_fieldnames())
        outputs["summary_csv"] = str(path)

    if config.outputs.write_segments_csv:
        path = out_dir / "segments.csv"
        _write_csv(path, simulation.segment_rows)
        outputs["segments_csv"] = str(path)

    if config.outputs.write_points_csv and simulation.point_rows:
        path = out_dir / "points.csv"
        _write_csv(path, simulation.point_rows)
        outputs["points_csv"] = str(path)

    if config.outputs.write_paraview:
        if simulation.segment_rows:
            path = out_dir / "vessels.vtp"
            _segment_polydata(
                simulation.segment_rows,
                resolution=max(int(config.outputs.vessel_resolution), 2),
                float_dtype=_float_dtype(config.outputs.export_float_dtype),
                index_dtype=_index_dtype(config.outputs.export_index_dtype),
            ).save(str(path))
            outputs["vessels_vtp"] = str(path)
        if simulation.point_rows:
            path = out_dir / "oxygen_points.vtp"
            _points_polydata(
                simulation.point_rows,
                float_dtype=_float_dtype(config.outputs.export_float_dtype),
                index_dtype=_index_dtype(config.outputs.export_index_dtype),
            ).save(str(path))
            outputs["oxygen_points_vtp"] = str(path)
        outputs.update(_save_domain_outputs(build.domain, out_dir))

    manifest_path = out_dir / "manifest.json"
    manifest = _manifest(config, build, simulation, outputs)
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, allow_nan=True), encoding="utf-8")
    outputs["manifest_json"] = str(manifest_path)
    return outputs


def _write_csv(path: Path, rows: list[dict[str, Any]], *, preferred_fields: list[str] | None = None) -> None:
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
    try:
        import TissueSim_cube_local as ts

        return list(getattr(ts, "CSV_FIELDNAMES", ()))
    except Exception:
        return []


def _segment_polydata(rows: list[dict[str, Any]], *, resolution: int, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    starts = np.array([[r["start_x"], r["start_y"], r["start_z"]] for r in rows], dtype=float_dtype)
    ends = np.array([[r["end_x"], r["end_y"], r["end_z"]] for r in rows], dtype=float_dtype)
    nseg = int(starts.shape[0])
    if nseg == 0:
        return pv.PolyData()
    t = np.linspace(0.0, 1.0, int(resolution), dtype=float_dtype)
    points = (starts[:, None, :] + t[None, :, None] * (ends - starts)[:, None, :]).reshape(nseg * int(resolution), 3)
    line_ids = np.arange(nseg * int(resolution), dtype=np.int64).reshape(nseg, int(resolution))
    lines = np.empty((nseg, int(resolution) + 1), dtype=np.int64)
    lines[:, 0] = int(resolution)
    lines[:, 1:] = line_ids
    poly = pv.PolyData(points, lines=lines.reshape(-1))

    def arr(name: str, default=np.nan, dtype=float_dtype):
        return np.asarray([r.get(name, default) for r in rows], dtype=dtype)

    cin = arr("cin")
    cout = arr("cout")
    conc = (cin[:, None] + t[None, :] * (cout - cin)[:, None]).reshape(nseg * int(resolution))
    poly.point_data["concentration"] = conc.astype(float_dtype, copy=False)
    for key in ("flow_cm3_s", "flow_ul_min", "radius_cm", "length_cm"):
        poly.point_data[key] = np.repeat(arr(key), int(resolution)).astype(float_dtype, copy=False)
    for key in ("tree_id", "local_segment_id", "global_segment_id"):
        poly.point_data[key] = np.repeat(arr(key, default=-1, dtype=index_dtype), int(resolution)).astype(index_dtype, copy=False)
    if any("discharge_hematocrit" in r for r in rows):
        poly.point_data["discharge_hematocrit"] = np.repeat(arr("discharge_hematocrit"), int(resolution))
    if any("tube_hematocrit" in r for r in rows):
        poly.point_data["tube_hematocrit"] = np.repeat(arr("tube_hematocrit"), int(resolution))
    return poly


def _points_polydata(rows: list[dict[str, Any]], *, float_dtype: np.dtype, index_dtype: np.dtype) -> pv.PolyData:
    points = np.array([[r["x"], r["y"], r["z"]] for r in rows], dtype=float_dtype)
    n = int(points.shape[0])
    verts = np.column_stack([np.ones(n, dtype=np.int64), np.arange(n, dtype=np.int64)]).reshape(-1)
    poly = pv.PolyData(points, verts=verts)
    for key in rows[0].keys():
        if key in {"x", "y", "z"}:
            continue
        values = [r.get(key) for r in rows]
        if key.endswith("_id") or key in {"point_id", "viability"}:
            poly.point_data[key] = np.asarray(values, dtype=index_dtype)
        else:
            poly.point_data[key] = np.asarray(values, dtype=float_dtype)
    return poly


def _save_domain_outputs(domain, out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    boundary = getattr(domain, "boundary", None)
    if boundary is None and getattr(domain, "mesh", None) is not None:
        try:
            boundary = domain.mesh.extract_surface()
        except Exception:
            boundary = None
    if boundary is None:
        try:
            domain.get_boundary()
            boundary = getattr(domain, "boundary", None)
        except Exception:
            boundary = None
    if boundary is not None:
        path = out_dir / "domain_boundary.vtp"
        boundary.save(str(path))
        outputs["domain_boundary_vtp"] = str(path)
    mesh = getattr(domain, "mesh", None)
    if mesh is not None:
        path = out_dir / "domain_mesh.vtu"
        mesh.save(str(path))
        outputs["domain_mesh_vtu"] = str(path)
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
        "settings_path": str(config.settings_path) if config.settings_path else None,
        "settings": config.raw,
        "resolved": {
            "network_mode": config.network_mode,
            "prefix": config.prefix,
            "target_counts": build.target_counts,
            "compute_float_policy": "float64",
            "export_float_dtype": config.outputs.export_float_dtype,
            "export_index_dtype": config.outputs.export_index_dtype,
            "simple_mode": "enabled" if config.network_mode == "simple" else "not_requested",
            "tissue_sample": build.sample_meta or simulation.sample_meta,
        },
        "dependencies": {
            "svv_version": svv_version,
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
            "segments": [int(getattr(tree, "segment_count", 0) or 0) for tree in build.trees],
            "terminals": [int(getattr(tree, "n_terminals", 0) or 0) for tree in build.trees],
            "connectivity_repairs": build.connectivity_repairs or [],
            "connectivity_reports": build.connectivity_reports or [],
        },
        "outputs": outputs,
        "geometry_only": bool(config.simulation.geometry_only),
        "skip_tissue_oxygen": bool(config.simulation.skip_tissue_oxygen),
    }


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
