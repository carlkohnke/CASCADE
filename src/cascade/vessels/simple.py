from __future__ import annotations

from dataclasses import dataclass, field
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from cascade.configuration.models import RunConfig, _reject_unknown
from .lattice import LATTICE_TYPES, generate_lattice


@dataclass
class SimpleNetwork:
    starts: np.ndarray
    ends: np.ndarray
    radii: np.ndarray
    lengths: np.ndarray
    flows: np.ndarray
    cin: np.ndarray
    cout: np.ndarray
    inlet_nodes: list[int]
    outlet_nodes: list[int]
    prox_ids: np.ndarray
    dist_ids: np.ndarray
    data: np.ndarray
    parameters: Any
    segment_count: int
    n_terminals: int
    mode: str
    decay_starts: np.ndarray
    decay_ends: np.ndarray
    tissue_starts: np.ndarray
    tissue_ends: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    _cascade_simple_network: bool = True


def build_simple_network(ts, domain, config: RunConfig) -> SimpleNetwork:
    raw = dict(config.network.simple or {})
    _reject_unknown(
        raw,
        {
            "mode", "channel_mode", "axis", "channel_axis", "radius_cm", "channel_radius_cm",
            "z_from_bottom_cm", "channel_z_from_bottom_cm", "flow_ul_min", "qin_ul_min",
            "concentration_inlet", "conc_inlet", "solve_channels_separately", "edge_extension_frac",
            "edge_channel_extension_frac", "y_offsets_cm", "channel_y_offsets_cm", "snake_arc_segments",
            "snake_straight_segments", "lattice_type", "cells", "cells_per_axis", "inlet_points_cm",
            "outlet_points_cm", "radius_expression", "subdivisions", "segment_subdivisions",
            "diffusivity", "vmax", "km", "omega", "window_factor",
            "path", "geometry_path", "inlet_nodes", "outlet_nodes",
        },
        "network.simple",
    )
    bounds_min, bounds_max = _domain_bounds(domain, config)
    dims = tuple(float(v) for v in (bounds_max - bounds_min))
    domain_center = tuple(float(v) for v in (0.5 * (bounds_min + bounds_max)))
    mode = str(raw.get("mode", raw.get("channel_mode", "onechannel"))).strip().lower()
    axis = str(raw.get("axis", raw.get("channel_axis", "x"))).strip().lower()
    radius = float(raw.get("radius_cm", raw.get("channel_radius_cm", 0.015)))
    z_from_bottom = float(raw.get("z_from_bottom_cm", raw.get("channel_z_from_bottom_cm", dims[2] * 0.5)))
    flow_ul_min = float(raw.get("flow_ul_min", raw.get("qin_ul_min", config.simulation.qin_target_ul_min)))
    inlet_conc = float(raw.get("concentration_inlet", raw.get("conc_inlet", ts.get_concentration_inlet(config.simulation.fluid))))
    solve_separate = _as_bool(raw.get("solve_channels_separately"), True)
    edge_extension_frac = float(raw.get("edge_extension_frac", raw.get("edge_channel_extension_frac", 0.0)))
    y_offsets = raw.get("y_offsets_cm", raw.get("channel_y_offsets_cm", [-0.2, 0.0, 0.2]))
    snake_arc_segments = int(raw.get("snake_arc_segments", 5))
    snake_straight_segments = int(raw.get("snake_straight_segments", 5))

    lattice_type = str(raw.get("lattice_type", mode)).strip().lower()
    is_lattice = mode == "lattice" or lattice_type in LATTICE_TYPES
    lattice_meta: dict[str, Any] = {}
    if mode == "custom":
        geometry_path = raw.get("path", raw.get("geometry_path"))
        if not geometry_path:
            raise ValueError("network.simple.path is required for custom geometry mode.")
        path = Path(str(geometry_path)).expanduser()
        if not path.is_absolute() and config.settings_path is not None:
            path = config.settings_path.parent / path
        starts, ends, radii, lengths, inlet_nodes, outlet_nodes, prox_ids, dist_ids = _load_custom_geometry(
            path.resolve(),
            default_radius_cm=radius,
            inlet_nodes=raw.get("inlet_nodes"),
            outlet_nodes=raw.get("outlet_nodes"),
        )
        solve_separate = False
    elif is_lattice:
        lattice_type = "cubic" if lattice_type == "lattice" else lattice_type
        lattice = generate_lattice(
            int(raw.get("cells", raw.get("cells_per_axis", 4))),
            dims,
            radius,
            lattice_type=lattice_type,
            inlet_points_cm=raw.get("inlet_points_cm"),
            outlet_points_cm=raw.get("outlet_points_cm"),
            radius_expression=raw.get("radius_expression"),
            subdivisions=int(raw.get("subdivisions", raw.get("segment_subdivisions", 1))),
            center_cm=domain_center,
            node_inside=domain.within,
        )
        starts = np.asarray(lattice["segment_starts_cm"], dtype=float)
        ends = np.asarray(lattice["segment_ends_cm"], dtype=float)
        radii = np.asarray(lattice["segment_radii_cm"], dtype=float)
        lengths = np.asarray(lattice["segment_lengths_cm"], dtype=float)
        edge_nodes = np.asarray(lattice["edge_nodes"], dtype=np.int64)
        prox_ids, dist_ids = edge_nodes[:, 0], edge_nodes[:, 1]
        inlet_nodes = [int(v) for v in lattice["inlet_nodes"]]
        outlet_nodes = [int(v) for v in lattice["outlet_nodes"]]
        mode = "lattice"
        solve_separate = False
        lattice_meta = {
            "lattice_type": lattice_type,
            "cells": int(lattice["cells"]),
            "subdivisions": int(lattice["subdivisions"]),
            "radius_expression": str(lattice["radius_expression"]),
            "inlet_points_cm": np.asarray(lattice["inlet_points_cm"]).tolist(),
            "outlet_points_cm": np.asarray(lattice["outlet_points_cm"]).tolist(),
            "inlet_connection_count": int(lattice["inlet_connection_count"]),
            "outlet_connection_count": int(lattice["outlet_connection_count"]),
            "bounding_box_center_cm": np.asarray(lattice["bounding_box_center_cm"]).tolist(),
            "nodes_removed_by_domain": int(lattice["nodes_removed_by_domain"]),
            "edges_removed_by_domain": int(lattice["edges_removed_by_domain"]),
        }
        if str(getattr(ts, "KIRCHHOFF_SOLVER", "tree")).strip().lower().startswith("tree"):
            print("Lattice graph selected: switching Kirchhoff solver from tree to spsolve.", flush=True)
            ts.KIRCHHOFF_SOLVER = "spsolve"
    else:
        starts, ends, radii, lengths, inlet_nodes, outlet_nodes, prox_ids, dist_ids = _build_channels(
            ts,
            dims=dims,
            mode=mode,
            axis=axis,
            radius_cm=radius,
            z_from_bottom_cm=z_from_bottom,
            y_offsets_cm=y_offsets,
            snake_arc_segments=snake_arc_segments,
            snake_straight_segments=snake_straight_segments,
        )
    tissue_starts, tissue_ends, decay_starts, decay_ends = _extended_tissue_geometry(
        starts,
        ends,
        dims=dims,
        extension_frac=edge_extension_frac,
    )

    fluid = config.simulation.fluid
    q_inlet_cm3_s = flow_ul_min * 1.0e-3 / 60.0
    mu_base = _fluid_mu_base(ts, fluid)
    mu_vals = ts.segment_viscosity_from_radius(radii, mu_base, fluid)
    resistances = (8.0 * mu_vals * lengths) / (np.pi * np.maximum(radii, 1.0e-12) ** 4)
    diffusivity = float(raw.get("diffusivity", ts.SOLUTE_DIFFUSIVITY))
    vmax = float(raw.get("vmax", ts.VMAX_MM))
    km = float(raw.get("km", ts.K_M_MM))
    if solve_separate and starts.shape[0] > 1 and mode == "multichannel":
        flows, cin, cout = _solve_independent_channels(
            ts,
            starts,
            radii,
            lengths,
            q_inlet_cm3_s,
            inlet_conc,
            fluid,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
        )
    else:
        _pressures, flows, *_ = ts.solve_kirchhoff(
            prox_ids,
            dist_ids,
            resistances,
            inlet_nodes,
            q_inlet_cm3_s,
            outlet_nodes,
        )
        cin, cout, _node_conc, _history = ts.solve_network_concentrations(
            starts,
            ends,
            radii,
            lengths,
            flows,
            inlet_nodes,
            outlet_nodes,
            inlet_conc,
            fluid=fluid,
            prox_ids=prox_ids,
            dist_ids=dist_ids,
            diffusivity=diffusivity,
            vmax=vmax,
            km=km,
            omega=float(raw.get("omega", getattr(ts, "OMEGA", 0.5))),
        )

    data = _simple_tree_data(starts, ends, radii, lengths, flows)
    params = SimpleNamespace(
        root_pressure=float(getattr(ts, "ROOT_PRESSURE", np.nan)),
        terminal_pressure=float(getattr(ts, "TERMINAL_PRESSURE", np.nan)),
        root_flow=float(np.sum(np.abs(flows))),
        terminal_flow=float(np.mean(np.abs(flows))) if flows.size else np.nan,
    )
    return SimpleNetwork(
        starts=starts,
        ends=ends,
        radii=radii,
        lengths=lengths,
        flows=np.asarray(flows, dtype=float),
        cin=np.asarray(cin, dtype=float),
        cout=np.asarray(cout, dtype=float),
        inlet_nodes=[int(v) for v in inlet_nodes],
        outlet_nodes=[int(v) for v in outlet_nodes],
        prox_ids=np.asarray(prox_ids, dtype=np.int64),
        dist_ids=np.asarray(dist_ids, dtype=np.int64),
        data=data,
        parameters=params,
        segment_count=int(starts.shape[0]),
        n_terminals=max(1, len(outlet_nodes)),
        mode=mode,
        decay_starts=decay_starts,
        decay_ends=decay_ends,
        tissue_starts=tissue_starts,
        tissue_ends=tissue_ends,
        metadata={
            "mode": mode,
            "axis": axis,
            "box_dimensions_cm": [float(v) for v in dims],
            "radius_cm": float(radius),
            "flow_ul_min": float(flow_ul_min),
            "concentration_inlet": float(inlet_conc),
            "edge_extension_frac": float(edge_extension_frac),
            **lattice_meta,
        },
    )


def simple_details(network: SimpleNetwork, sample_points: np.ndarray, ts, config: RunConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(config.network.simple or {})
    if sample_points.size:
        mask, tissue_conc = ts.compute_tissue_samples_greens(
            sample_points,
            network.tissue_starts,
            network.tissue_ends,
            network.radii,
            network.cin,
            network.flows,
            diffusivity=float(raw.get("diffusivity", ts.SOLUTE_DIFFUSIVITY)),
            vmax=float(raw.get("vmax", ts.VMAX_MM)),
            km=float(raw.get("km", ts.K_M_MM)),
            window_factor=float(raw.get("window_factor", ts.WINDOW_FACTOR)),
            inlet_concentration=float(network.metadata["concentration_inlet"]),
            tissue_cache=None,
        )
        keep = np.asarray(mask, dtype=bool)
        tissue_points = sample_points[keep]
        tissue_values = np.asarray(tissue_conc, dtype=float)[keep]
    else:
        tissue_points = np.empty((0, 3), dtype=float)
        tissue_values = np.empty((0,), dtype=float)

    total_volume = float(np.nansum(np.pi * network.radii * network.radii * network.lengths))
    conc_max = float(getattr(ts, "CONC_MAX_FOR_NORMALIZATION", np.nan))
    mean_tissue = float(np.nanmean(tissue_values)) if tissue_values.size else np.nan
    summary = {
        "target_terminals": int(network.n_terminals),
        "cube_side_length": float(config.domain.side_length),
        "distance_sample_count": int(sample_points.shape[0]),
        "concentration_solver": "simple_channel",
        "finite_radius_o2_terms": str(getattr(ts, "FINITE_RADIUS_O2_TERMS", "none")),
        "lumen_wall_closure": str(getattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")),
        "total_volume": total_volume,
        "total_flowrate": float(np.nansum(np.abs(network.flows))),
        "pressure_in_root": float(getattr(network.parameters, "root_pressure", np.nan)),
        "pressure_out_terminals": float(getattr(network.parameters, "terminal_pressure", np.nan)),
        "avg_radius": float(np.nanmean(network.radii)) if network.radii.size else np.nan,
        "avg_length": float(np.nanmean(network.lengths)) if network.lengths.size else np.nan,
        "total_length": float(np.nansum(network.lengths)),
        "avg_distance_to_channel": np.nan,
        "terminal_segments": int(network.n_terminals),
        "total_segments": int(network.segment_count),
        "dlp_angle": 0.0,
        "Rnet": np.nan,
        "dRnet": np.nan,
        "Qmin_over_Qinlet": float(np.nanmin(np.abs(network.flows)) / max(np.nansum(np.abs(network.flows)), 1.0e-30)) if network.flows.size else np.nan,
        "C_LQ_over_Cmax": float(np.nanmean(network.cout) / conc_max) if np.isfinite(conc_max) and conc_max else np.nan,
        "C_tiss_over_Cmax": mean_tissue / conc_max if np.isfinite(mean_tissue) and np.isfinite(conc_max) and conc_max else np.nan,
        "inlet_flow_ul_per_min": float(network.metadata["flow_ul_min"]),
    }
    details = {
        "starts": network.starts,
        "ends": network.ends,
        "radii": network.radii,
        "lengths": network.lengths,
        "flows": network.flows,
        "cin": network.cin,
        "cout": network.cout,
        "tissue_points": tissue_points,
        "tissue_values": tissue_values,
        "inlet_concentration": float(network.metadata["concentration_inlet"]),
    }
    return summary, details


def _domain_dimensions(config: RunConfig) -> tuple[float, float, float]:
    side = float(config.domain.side_length)
    return (
        float(config.domain.x_length if config.domain.x_length is not None else side),
        float(config.domain.y_length if config.domain.y_length is not None else side),
        float(config.domain.z_length if config.domain.z_length is not None else side),
    )


def _domain_bounds(domain, config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the built domain's actual axis-aligned bounds in centimeters."""
    for candidate in (
        getattr(domain, "boundary", None),
        getattr(domain, "original_boundary", None),
        getattr(domain, "mesh", None),
    ):
        bounds = getattr(candidate, "bounds", None)
        if bounds is not None and len(bounds) == 6:
            lower = np.asarray([bounds[0], bounds[2], bounds[4]], dtype=float)
            upper = np.asarray([bounds[1], bounds[3], bounds[5]], dtype=float)
            if np.all(np.isfinite(lower)) and np.all(upper > lower):
                return lower, upper
    points = np.asarray(getattr(domain, "points", np.empty((0, 3))), dtype=float)
    if points.ndim == 2 and points.shape[0] and points.shape[1] == 3:
        lower, upper = np.min(points, axis=0), np.max(points, axis=0)
        if np.all(np.isfinite(lower)) and np.all(upper > lower):
            return lower, upper

    dims = np.asarray(_domain_dimensions(config), dtype=float)
    if str(config.domain.kind).strip().lower() in {"sphere", "pv.sphere", "pyvista_sphere"}:
        radius = float(
            config.domain.radius
            if config.domain.radius is not None
            else config.domain.side_length / 2.0
        )
        dims[:] = 2.0 * radius
        center = np.asarray(config.domain.center or [0.0, 0.0, 0.0], dtype=float)
    else:
        center = np.zeros(3, dtype=float)
    return center - 0.5 * dims, center + 0.5 * dims


def _build_channels(
    ts,
    *,
    dims: tuple[float, float, float],
    mode: str,
    axis: str,
    radius_cm: float,
    z_from_bottom_cm: float,
    y_offsets_cm: Any,
    snake_arc_segments: int,
    snake_straight_segments: int,
):
    x_len, y_len, z_len = dims
    x_half = x_len / 2.0
    y_half = y_len / 2.0
    z = -z_len / 2.0 + float(z_from_bottom_cm)
    if mode not in {"onechannel", "single", "multichannel", "snake"}:
        raise ValueError("simple mode must be 'onechannel', 'multichannel', or 'snake'.")
    if mode == "single":
        mode = "onechannel"
    if mode == "snake":
        points = _snake_channel_points(z, snake_arc_segments, snake_straight_segments)
        starts = points[:-1].copy()
        ends = points[1:].copy()
    else:
        starts_list = []
        ends_list = []
        offsets = [0.0] if mode == "onechannel" else [float(v) for v in y_offsets_cm]
        for offset in offsets:
            if axis == "x":
                starts_list.append(np.array([-x_half, offset, z], dtype=float))
                ends_list.append(np.array([x_half, offset, z], dtype=float))
            elif axis == "y":
                starts_list.append(np.array([offset, -y_half, z], dtype=float))
                ends_list.append(np.array([offset, y_half, z], dtype=float))
            else:
                raise ValueError("simple axis must be 'x' or 'y'.")
        starts = np.asarray(starts_list, dtype=float)
        ends = np.asarray(ends_list, dtype=float)
    radii = np.full((starts.shape[0],), float(radius_cm), dtype=float)
    lengths = np.linalg.norm(ends - starts, axis=1)
    geom = np.zeros((starts.shape[0], 6), dtype=float)
    geom[:, 0:3] = starts
    geom[:, 3:6] = ends
    prox_ids, dist_ids, _nodes = ts._build_node_indices(geom)
    if mode == "snake":
        inlet_nodes = [int(prox_ids[0])]
        outlet_nodes = [int(dist_ids[-1])]
    else:
        inlet_nodes = [int(v) for v in prox_ids]
        outlet_nodes = [int(v) for v in dist_ids]
    return starts, ends, radii, lengths, inlet_nodes, outlet_nodes, prox_ids, dist_ids


def _load_custom_geometry(
    path: Path,
    *,
    default_radius_cm: float,
    inlet_nodes: Any = None,
    outlet_nodes: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int], np.ndarray, np.ndarray]:
    """Load an explicit segment network from CSV or NPZ.

    CSV requires start_x/start_y/start_z and end_x/end_y/end_z in centimetres.
    radius_cm and prox_id/dist_id are optional. NPZ uses starts, ends, radii,
    prox_ids and dist_ids arrays. When node IDs are omitted they are inferred
    by matching segment endpoints exactly after rounding to 12 decimal places.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Custom geometry file not found: {path}")

    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Custom geometry CSV is empty: {path}")
        coordinate_fields = ("start_x", "start_y", "start_z", "end_x", "end_y", "end_z")
        missing = [name for name in coordinate_fields if name not in rows[0]]
        if missing:
            raise ValueError(f"Custom geometry CSV is missing columns: {', '.join(missing)}")
        starts = np.asarray([[float(row[name]) for name in coordinate_fields[:3]] for row in rows], dtype=float)
        ends = np.asarray([[float(row[name]) for name in coordinate_fields[3:]] for row in rows], dtype=float)
        radii = np.asarray(
            [float(row.get("radius_cm") or default_radius_cm) for row in rows], dtype=float
        )
        if {"prox_id", "dist_id"}.issubset(rows[0]):
            prox_ids = np.asarray([int(row["prox_id"]) for row in rows], dtype=np.int64)
            dist_ids = np.asarray([int(row["dist_id"]) for row in rows], dtype=np.int64)
        else:
            prox_ids, dist_ids = _infer_node_ids(starts, ends)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            starts = np.asarray(data["starts"], dtype=float)
            ends = np.asarray(data["ends"], dtype=float)
            radii = np.asarray(data["radii"], dtype=float) if "radii" in data else np.full(starts.shape[0], default_radius_cm)
            if "prox_ids" in data and "dist_ids" in data:
                prox_ids = np.asarray(data["prox_ids"], dtype=np.int64)
                dist_ids = np.asarray(data["dist_ids"], dtype=np.int64)
            else:
                prox_ids, dist_ids = _infer_node_ids(starts, ends)
    else:
        raise ValueError("Custom geometry must be a .csv or .npz file.")

    if starts.ndim != 2 or starts.shape[1] != 3 or ends.shape != starts.shape:
        raise ValueError("Custom geometry starts and ends must have shape (n_segments, 3).")
    if radii.shape != (starts.shape[0],) or np.any(~np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("Custom geometry radii must be finite, positive, and one per segment.")
    if prox_ids.shape != (starts.shape[0],) or dist_ids.shape != (starts.shape[0],):
        raise ValueError("Custom geometry node ID arrays must contain one value per segment.")
    if np.any(prox_ids < 0) or np.any(dist_ids < 0) or np.any(prox_ids == dist_ids):
        raise ValueError("Custom geometry node IDs must be non-negative and each segment must connect distinct nodes.")

    lengths = np.linalg.norm(ends - starts, axis=1)
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(ends)) or np.any(lengths <= 0.0):
        raise ValueError("Custom geometry coordinates must be finite and segments must have positive length.")

    inferred_inlets = sorted(set(int(v) for v in prox_ids) - set(int(v) for v in dist_ids))
    inferred_outlets = sorted(set(int(v) for v in dist_ids) - set(int(v) for v in prox_ids))
    inlet_list = inferred_inlets if inlet_nodes is None else [int(v) for v in inlet_nodes]
    outlet_list = inferred_outlets if outlet_nodes is None else [int(v) for v in outlet_nodes]
    if not inlet_list or not outlet_list:
        raise ValueError("Custom geometry must define or imply at least one inlet and one outlet node.")
    return starts, ends, radii, lengths, inlet_list, outlet_list, prox_ids, dist_ids


def _infer_node_ids(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.vstack((starts, ends))
    rounded = np.round(points, decimals=12)
    _unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    count = starts.shape[0]
    return inverse[:count].astype(np.int64), inverse[count:].astype(np.int64)


def _snake_channel_points(z: float, arc_segments: int, straight_segments: int = 5) -> np.ndarray:
    def arc(center, radius, theta_start, theta_end, n_segments, *, x_sign):
        theta = np.linspace(theta_start, theta_end, n_segments + 1, dtype=float)
        cx, cy, cz = center
        x = cx + x_sign * radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        return np.column_stack((x, y, np.full_like(x, cz)))

    nseg = max(2, int(arc_segments))
    straight_nseg = max(1, int(straight_segments))
    inlet = np.array([0.53, -0.2, z], dtype=float)
    first_turn_start = np.array([-0.325, -0.2, z], dtype=float)
    first_turn_end = np.array([-0.325, 0.0, z], dtype=float)
    second_turn_start = np.array([0.325, 0.0, z], dtype=float)
    second_turn_end = np.array([0.325, 0.2, z], dtype=float)
    outlet = np.array([-0.53, 0.2, z], dtype=float)
    first_arc = arc((-0.325, -0.1, z), 0.1, -np.pi / 2.0, np.pi / 2.0, nseg, x_sign=-1.0)
    second_arc = arc((0.325, 0.1, z), 0.1, -np.pi / 2.0, np.pi / 2.0, nseg, x_sign=1.0)
    def straight(start, end):
        return np.linspace(start, end, straight_nseg + 1, dtype=float)

    # Every straight run and each half-turn receive their own controllable
    # tessellation.  This is the geometry used by both the solver and setup
    # preview, keeping the visible path identical to a completed result.
    return np.vstack((
        straight(inlet, first_turn_start),
        first_arc[1:],
        straight(first_turn_end, second_turn_start)[1:],
        second_arc[1:],
        straight(second_turn_end, outlet)[1:],
    ))


def _extended_tissue_geometry(starts: np.ndarray, ends: np.ndarray, *, dims: tuple[float, float, float], extension_frac: float):
    frac = float(extension_frac)
    if frac <= 0.0 or starts.size == 0:
        return starts.copy(), ends.copy(), np.zeros(starts.shape[0]), np.linalg.norm(ends - starts, axis=1)
    starts_ext = starts.copy()
    ends_ext = ends.copy()
    decay_start = np.zeros(starts.shape[0], dtype=float)
    decay_end = np.linalg.norm(ends - starts, axis=1)
    mins = -0.5 * np.asarray(dims, dtype=float)
    maxs = 0.5 * np.asarray(dims, dtype=float)
    extents = maxs - mins
    tol = max(1.0e-9, 1.0e-6 * float(np.max(extents)))
    for i in range(starts.shape[0]):
        vec = ends[i] - starts[i]
        seg_len = float(np.linalg.norm(vec))
        if seg_len <= 1.0e-12:
            continue
        direction = vec / seg_len
        axis = int(np.argmax(np.abs(direction)))
        extension = frac * float(extents[axis])
        start_hits = np.any(np.isclose(starts[i], mins, atol=tol) | np.isclose(starts[i], maxs, atol=tol))
        end_hits = np.any(np.isclose(ends[i], mins, atol=tol) | np.isclose(ends[i], maxs, atol=tol))
        if start_hits:
            starts_ext[i] -= direction * extension
            decay_start[i] += extension
            decay_end[i] += extension
        if end_hits:
            ends_ext[i] += direction * extension
    return starts_ext, ends_ext, decay_start, decay_end


def _solve_independent_channels(ts, starts, radii, lengths, q_inlet_cm3_s, inlet_conc, fluid, *, diffusivity, vmax, km):
    flows = np.full((starts.shape[0],), float(q_inlet_cm3_s), dtype=float)
    cin = np.full((starts.shape[0],), float(inlet_conc), dtype=float)
    cout = np.empty_like(cin)
    for i in range(starts.shape[0]):
        if str(fluid).lower() == "blood":
            ht = ts.tube_hematocrit(radii[i], hd=ts.HD_DISCHARGE)
            ccap = ts.segment_O2_capacity_from_HT(ht)
            decay = ts._blood_greens_decay_factor(
                flows[i] * ts.CM3_TO_M3,
                radii[i] * ts.CM_TO_M,
                lengths[i] * ts.CM_TO_M,
                float(diffusivity) * ts.CM2_TO_M2,
                float(vmax),
                float(km),
                cin[i],
                ccap,
            )
        else:
            decay = ts._greens_decay_factor(
                flows[i] * ts.CM3_TO_M3,
                radii[i] * ts.CM_TO_M,
                lengths[i] * ts.CM_TO_M,
                float(diffusivity) * ts.CM2_TO_M2,
                float(vmax),
                float(km),
                cin[i],
            )
        cout[i] = cin[i] * decay
    return flows, cin, cout


def _simple_tree_data(starts, ends, radii, lengths, flows) -> np.ndarray:
    n = int(starts.shape[0])
    data = np.full((n, 31), np.nan, dtype=float)
    data[:, 0:3] = starts
    data[:, 3:6] = ends
    data[:, 20] = lengths
    data[:, 21] = radii
    data[:, 22] = flows
    return data


def _fluid_mu_base(ts, fluid: str) -> float:
    fluid_mode = str(fluid or "").lower()
    if fluid_mode in {"water", "cell media", "media"}:
        rho = 0.99336
        nu = 0.6959 / 100.0
        return rho * nu
    if fluid_mode == "blood":
        return float(getattr(ts, "MU_PLASMA_CGS", 0.012))
    raise ValueError(f"Unsupported simple-channel fluid: {fluid!r}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)
