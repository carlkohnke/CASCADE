"""Load, parse, and validate user-supplied CASCADE configuration.

Each section rejects unknown keys, converts external values to typed models,
and performs cross-section checks before any domain or solver work begins.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .models import (
    DomainConfig,
    ExternalFieldConfig,
    GrowthConfig,
    NetworkConfig,
    OcclusionConfig,
    OutputsConfig,
    RootConfig,
    RunConfig,
    SimulationConfig,
)


def _reject_unknown(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(f"{section}.{name}" if section else name for name in unknown)
        raise ValueError(f"Unknown CASCADE setting(s): {names}")


def reject_unknown(data: dict[str, Any], allowed: set[str], section: str) -> None:
    """Reject keys outside an explicitly supported configuration vocabulary."""

    _reject_unknown(data, allowed, section)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1) and math.isfinite(float(value)):
            return bool(value)
        raise ValueError(f"Boolean values must be true/false or 1/0, got {value!r}.")
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unrecognized boolean value: {value!r}.")


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _as_int(value: Any, *, name: str) -> int:
    result = _as_float(value, name=name)
    if not result.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(result)


def _as_float_list(value: Any, *, name: str, length: int | None = None) -> list[float]:
    if value is None:
        raise ValueError(f"{name} is required.")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of numbers.")
    out = [_as_float(v, name=f"{name}[{index}]") for index, v in enumerate(value)]
    if length is not None and len(out) != length:
        raise ValueError(f"{name} must contain {length} values.")
    return out


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [_as_int(value, name=name)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an integer or list of integers.")
    return [_as_int(v, name=f"{name}[{index}]") for index, v in enumerate(value)]


def load_config(path: str | Path) -> RunConfig:
    settings_path = Path(path).expanduser().resolve()
    with settings_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Settings JSON must contain an object at the top level.")
    config = parse_config(raw)
    config.settings_path = settings_path
    if config.network_mode == "simple":
        simple = dict(config.network.simple or {})
        if str(simple.get("mode", "")).strip().lower() == "custom":
            custom_path = Path(
                str(simple.get("path", simple.get("geometry_path")))
            ).expanduser()
            if not custom_path.is_absolute():
                custom_path = settings_path.parent / custom_path
            if not custom_path.is_file():
                raise FileNotFoundError(f"Custom geometry file not found: {custom_path}")
    if config.network.input_path:
        input_path = Path(config.network.input_path).expanduser()
        if not input_path.is_absolute():
            input_path = settings_path.parent / input_path
        checkpoint = config.growth.checkpoint_path
        if config.growth.resume_from_checkpoint and checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = settings_path.parent / checkpoint_path
            if checkpoint_path.is_file():
                input_path = checkpoint_path
        if not input_path.is_file():
            raise FileNotFoundError(f"Network input file not found: {input_path}")
    return config


def parse_config(raw: dict[str, Any]) -> RunConfig:
    _reject_unknown(
        raw,
        {
            "schema_version",
            "domain",
            "network",
            "growth",
            "simulation",
            "outputs",
            "settings",
            "runtime_settings",
            "sweep",
            "gui",
        },
        "",
    )
    domain = _parse_domain(raw.get("domain", {}))
    network = _parse_network(raw.get("network", {}))
    growth = _parse_growth(raw.get("growth", {}))
    simulation = _parse_simulation(raw.get("simulation", {}))
    outputs = _parse_outputs(raw.get("outputs", {}))
    runtime_settings = _parse_runtime_settings(
        raw.get("settings", raw.get("runtime_settings", {}))
    )
    _validate(domain, network, growth, simulation, outputs)
    return RunConfig(
        domain=domain,
        network=network,
        growth=growth,
        simulation=simulation,
        outputs=outputs,
        runtime_settings=runtime_settings,
        raw=dict(raw),
    )


def _parse_domain(raw: Any) -> DomainConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {
            "type",
            "kind",
            "side_length",
            "side_len",
            "dimensions",
            "lengths",
            "x_length",
            "y_length",
            "z_length",
            "box_x_cm",
            "box_y_cm",
            "box_z_cm",
            "radius",
            "sphere_radius",
            "height",
            "cylinder_height",
            "center",
            "theta_resolution",
            "sphere_theta_resolution",
            "phi_resolution",
            "sphere_phi_resolution",
            "path",
            "mesh",
            "random_seed",
            "use_cache",
            "cache_dir",
        },
        "domain",
    )
    kind = str(data.get("type", data.get("kind", "cube"))).strip().lower()
    side_length = _as_float(
        data.get("side_length", data.get("side_len", 1.0)),
        name="domain.side_length",
    )
    dimensions = data.get("dimensions", data.get("lengths"))
    x_len = y_len = z_len = None
    if dimensions is not None:
        dims = _as_float_list(dimensions, name="domain.dimensions", length=3)
        x_len, y_len, z_len = dims
    radius_raw = data.get("radius", data.get("sphere_radius"))
    radius = None if radius_raw is None else _as_float(radius_raw, name="domain.radius")
    if radius is not None and radius <= 0.0:
        raise ValueError("domain.radius must be positive.")
    height_raw = data.get("height", data.get("cylinder_height"))
    height = None if height_raw is None else _as_float(height_raw, name="domain.height")
    if height is not None and height <= 0.0:
        raise ValueError("domain.height must be positive.")
    center_raw = data.get("center")
    center = (
        None
        if center_raw is None
        else _as_float_list(center_raw, name="domain.center", length=3)
    )
    theta_resolution = _as_int(
        data.get("theta_resolution", data.get("sphere_theta_resolution", 12)),
        name="domain.theta_resolution",
    )
    phi_resolution = _as_int(
        data.get("phi_resolution", data.get("sphere_phi_resolution", 8)),
        name="domain.phi_resolution",
    )
    if theta_resolution < 8:
        raise ValueError("domain.theta_resolution must be at least 8.")
    if phi_resolution < 8:
        raise ValueError("domain.phi_resolution must be at least 8.")
    return DomainConfig(
        kind=kind,
        side_length=side_length,
        x_length=None
        if data.get("x_length", data.get("box_x_cm", x_len)) is None
        else _as_float(
            data.get("x_length", data.get("box_x_cm", x_len)),
            name="domain.x_length",
        ),
        y_length=None
        if data.get("y_length", data.get("box_y_cm", y_len)) is None
        else _as_float(
            data.get("y_length", data.get("box_y_cm", y_len)),
            name="domain.y_length",
        ),
        z_length=None
        if data.get("z_length", data.get("box_z_cm", z_len)) is None
        else _as_float(
            data.get("z_length", data.get("box_z_cm", z_len)),
            name="domain.z_length",
        ),
        radius=radius,
        height=height,
        center=center,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
        path=data.get("path"),
        mesh=data.get("mesh"),
        random_seed=_as_int(data.get("random_seed", 42), name="domain.random_seed"),
        use_cache=_as_bool(data.get("use_cache"), True),
        cache_dir=data.get("cache_dir"),
    )


def _parse_roots(data: dict[str, Any]) -> list[RootConfig]:
    roots_raw = data.get("roots")
    if roots_raw is None and data.get("root") is not None:
        roots_raw = [data.get("root")]
    if roots_raw is None and data.get("start") is not None:
        roots_raw = [{"start": data.get("start"), "direction": data.get("direction")}]
    if roots_raw is None:
        roots_raw = [{"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]}]
    if not isinstance(roots_raw, list):
        raise ValueError("network.roots must be a list.")
    roots: list[RootConfig] = []
    for idx, root_raw in enumerate(roots_raw):
        if not isinstance(root_raw, dict):
            raise ValueError(f"network.roots[{idx}] must be an object.")
        _reject_unknown(root_raw, {"start", "direction"}, f"network.roots[{idx}]")
        start = _as_float_list(
            root_raw.get("start"), name=f"network.roots[{idx}].start", length=3
        )
        direction_raw = root_raw.get("direction")
        direction = (
            None
            if direction_raw is None
            else _as_float_list(
                direction_raw,
                name=f"network.roots[{idx}].direction",
                length=3,
            )
        )
        roots.append(RootConfig(start=start, direction=direction))
    return roots


def _parse_network(raw: Any) -> NetworkConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {
            "mode",
            "input_path",
            "path",
            "target_terminal_count",
            "target_count",
            "target_total_terminal_count",
            "target_total_terminals",
            "target_terminal_counts",
            "target_counts",
            "root",
            "roots",
            "start",
            "direction",
            "physical_clearance",
            "save_path",
            "repair_connectivity",
            "validate_connectivity",
            "fail_connectivity",
            "fail_on_connectivity_error",
            "connectivity_geometry_atol",
            "simple",
            "simple_geometry",
        },
        "network",
    )
    target_counts = _as_int_list(
        data.get("target_terminal_counts", data.get("target_counts")),
        name="network.target_terminal_counts",
    )
    target_single = data.get(
        "target_terminal_count", data.get("target_count", 100)
    )
    target_total = data.get(
        "target_total_terminal_count", data.get("target_total_terminals")
    )
    return NetworkConfig(
        mode=str(data.get("mode", "tree")).strip().lower(),
        input_path=data.get("input_path", data.get("path")),
        target_terminal_count=None
        if target_single is None
        else _as_int(target_single, name="network.target_terminal_count"),
        target_total_terminal_count=None
        if target_total is None
        else _as_int(target_total, name="network.target_total_terminal_count"),
        target_terminal_counts=target_counts,
        roots=_parse_roots(data),
        physical_clearance=_as_float(
            data.get("physical_clearance", 0.0), name="network.physical_clearance"
        ),
        save_path=data.get("save_path"),
        repair_connectivity=_as_bool(data.get("repair_connectivity"), True),
        validate_connectivity=_as_bool(data.get("validate_connectivity"), False),
        fail_connectivity=_as_bool(
            data.get("fail_connectivity", data.get("fail_on_connectivity_error")), True
        ),
        connectivity_geometry_atol=_as_float(
            data.get("connectivity_geometry_atol", 1.0e-6),
            name="network.connectivity_geometry_atol",
        ),
        simple=dict(data.get("simple", data.get("simple_geometry", {})) or {}),
    )


def _parse_growth(raw: Any) -> GrowthConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {
            "enabled",
            "assignment",
            "growth_assignment",
            "bulk_growth_mode",
            "n_closest_vessels",
            "n_points",
            "weighted_sampling",
            "ignore_collisions",
            "allow_inside_vessels",
            "n_ignore_collisions",
            "collision_retry_limit",
            "collision_failure_mode",
            "nearest_tree_batch_points",
            "nearest_batch_points",
            "strict_domain_segments",
            "strict_domain_max_terminals",
            "domain_line_samples",
            "domain_line_tolerance",
            "growth_report_every",
            "add_per_tree",
            "add_total",
            "add_split_mode",
            "checkpoint_path",
            "checkpoint_forest",
            "checkpoint_every_adds",
            "resume_from_checkpoint",
            "save_target_counts",
            "n_equal_bifurcations",
            "equal_terminal",
            "equal_bifurcation",
        },
        "growth",
    )
    equal = dict(data.get("equal_terminal", data.get("equal_bifurcation", {})) or {})
    n_equal = data.get("n_equal_bifurcations", equal.get("n_equal_bifurcations"))
    assignment = (
        str(data.get("assignment", data.get("growth_assignment", "bulk")))
        .strip()
        .lower()
        .replace("_", "-")
    )
    bulk_default = "equal-bifurcation" if assignment == "nearest-tree" else "always"
    return GrowthConfig(
        enabled=_as_bool(data.get("enabled"), True),
        assignment=assignment,
        bulk_growth_mode=str(data.get("bulk_growth_mode", bulk_default))
        .strip()
        .lower()
        .replace("_", "-"),
        n_closest_vessels=_as_int(
            data.get("n_closest_vessels", 2), name="growth.n_closest_vessels"
        ),
        n_points=_as_int(data.get("n_points", 50), name="growth.n_points"),
        weighted_sampling=_as_bool(data.get("weighted_sampling"), False),
        ignore_collisions=_as_bool(data.get("ignore_collisions"), True),
        allow_inside_vessels=_as_bool(data.get("allow_inside_vessels"), True),
        n_ignore_collisions=(
            None
            if data.get("n_ignore_collisions") is None
            else _as_int(
                data.get("n_ignore_collisions"), name="growth.n_ignore_collisions"
            )
        ),
        collision_retry_limit=_as_int(
            data.get("collision_retry_limit", 100),
            name="growth.collision_retry_limit",
        ),
        collision_failure_mode=str(data.get("collision_failure_mode", "error"))
        .strip()
        .lower(),
        nearest_tree_batch_points=_as_int(
            data.get("nearest_tree_batch_points", data.get("nearest_batch_points", 256)),
            name="growth.nearest_tree_batch_points",
        ),
        strict_domain_segments=_as_bool(data.get("strict_domain_segments"), False),
        strict_domain_max_terminals=_as_int(
            data.get("strict_domain_max_terminals", 10000),
            name="growth.strict_domain_max_terminals",
        ),
        domain_line_samples=_as_int(
            data.get("domain_line_samples", 4), name="growth.domain_line_samples"
        ),
        domain_line_tolerance=_as_float(
            data.get("domain_line_tolerance", 0.0),
            name="growth.domain_line_tolerance",
        ),
        growth_report_every=_as_int(
            data.get("growth_report_every", 0), name="growth.growth_report_every"
        ),
        add_per_tree=_as_int_list(data.get("add_per_tree"), name="growth.add_per_tree"),
        add_total=None
        if data.get("add_total") is None
        else _as_int(data.get("add_total"), name="growth.add_total"),
        add_split_mode=str(data.get("add_split_mode", "equal"))
        .strip()
        .lower()
        .replace("_", "-"),
        checkpoint_path=data.get("checkpoint_path", data.get("checkpoint_forest")),
        checkpoint_every_adds=_as_int(
            data.get("checkpoint_every_adds", 0),
            name="growth.checkpoint_every_adds",
        ),
        resume_from_checkpoint=_as_bool(data.get("resume_from_checkpoint"), False),
        save_target_counts=_as_int_list(
            data.get("save_target_counts"), name="growth.save_target_counts"
        ),
        n_equal_bifurcations=None
        if n_equal is None
        or _as_int(n_equal, name="growth.n_equal_bifurcations") < 0
        else _as_int(n_equal, name="growth.n_equal_bifurcations"),
        equal_terminal=equal,
    )


def _parse_simulation(raw: Any) -> SimulationConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {
            "fluid",
            "build_fluid",
            "qin_target_ul_min",
            "qin_target",
            "total_qin_ul_min",
            "concentration_solver",
            "distance_sample_count",
            "flow_source",
            "inlet_conditions",
            "sample_mode",
            "tissue_sample_mode",
            "sample_points_path",
            "sample_file",
            "tissue_grid",
            "grid",
            "geometry_only",
            "skip_tissue_oxygen",
            "compute_avg_distance_to_channel",
            "tissue_accel",
            "tissue_gpu_validate_points",
            "viability_threshold",
            "occlusion",
            "infarction",
            "external_field",
            "cext",
            "tissuesim",
            "overrides",
            "finite_radius_o2_terms",
            "lumen_wall_closure",
            "graetz_n_radial",
            "graetz_n_modes",
            "graetz_max_fp_iters",
            "nearest_tissue_vessels",
            "window_factor",
            "hematocrit_model",
            "hematocrit_flow_iterations",
            "kirchhoff_bc_mode",
            "kirchhoff_solver",
            "solver_timing_details",
        },
        "simulation",
    )
    tissuesim = dict(data.get("tissuesim", data.get("overrides", {})) or {})
    for key in (
        "finite_radius_o2_terms",
        "lumen_wall_closure",
        "graetz_n_radial",
        "graetz_n_modes",
        "graetz_max_fp_iters",
        "nearest_tissue_vessels",
        "window_factor",
        "hematocrit_model",
        "hematocrit_flow_iterations",
        "kirchhoff_bc_mode",
        "kirchhoff_solver",
        "solver_timing_details",
    ):
        if key in data:
            tissuesim[key] = data[key]
    fluid = _parse_fluid(data.get("fluid", "blood"), "simulation.fluid", allow_both=True)
    build_fluid = _parse_fluid(
        data.get("build_fluid", "blood" if fluid == "both" else fluid),
        "simulation.build_fluid",
        allow_both=False,
    )
    return SimulationConfig(
        fluid=fluid,
        build_fluid=build_fluid,
        qin_target_ul_min=_as_float(
            data.get("qin_target_ul_min", data.get("qin_target", 100.0)),
            name="simulation.qin_target_ul_min",
        ),
        total_qin_ul_min=(
            None
            if data.get("total_qin_ul_min") is None
            else _as_float(
                data.get("total_qin_ul_min"), name="simulation.total_qin_ul_min"
            )
        ),
        concentration_solver=str(data.get("concentration_solver", "network_ext"))
        .strip()
        .lower(),
        distance_sample_count=_as_int(
            data.get("distance_sample_count", 10000),
            name="simulation.distance_sample_count",
        ),
        flow_source=str(data.get("flow_source", "per_tree")).strip().lower(),
        inlet_conditions=_parse_inlet_conditions(data.get("inlet_conditions", [])),
        sample_mode=str(
            data.get("sample_mode", data.get("tissue_sample_mode", "random"))
        )
        .strip()
        .lower(),
        sample_points_path=data.get("sample_points_path", data.get("sample_file")),
        tissue_grid=dict(data.get("tissue_grid", data.get("grid", {})) or {}),
        geometry_only=_as_bool(data.get("geometry_only"), False),
        skip_tissue_oxygen=_as_bool(data.get("skip_tissue_oxygen"), False),
        compute_avg_distance_to_channel=_as_bool(
            data.get("compute_avg_distance_to_channel"), False
        ),
        tissue_accel=data.get("tissue_accel", "auto"),
        tissue_gpu_validate_points=(
            None
            if data.get("tissue_gpu_validate_points") is None
            else _as_int(
                data.get("tissue_gpu_validate_points"),
                name="simulation.tissue_gpu_validate_points",
            )
        ),
        viability_threshold=(
            None
            if data.get("viability_threshold") is None
            else _as_float(
                data.get("viability_threshold"),
                name="simulation.viability_threshold",
            )
        ),
        occlusion=_parse_occlusion(data.get("occlusion", data.get("infarction"))),
        external_field=_parse_external_field(data.get("external_field")),
        cext=dict(data.get("cext", {}) or {}),
        tissuesim=tissuesim,
    )


def _parse_occlusion(raw: Any) -> OcclusionConfig | None:
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValueError("simulation.occlusion must be an object.")
    data = dict(raw)
    _reject_unknown(
        data,
        {
            "global_segment_id",
            "infarction_global_segment_id",
            "fraction_blocked",
            "fraction_blocked_infarction",
            "include_downstream_when_complete",
        },
        "simulation.occlusion",
    )
    fraction = _as_float(
        data.get("fraction_blocked", data.get("fraction_blocked_infarction", 0.0))
        or 0.0,
        name="simulation.occlusion.fraction_blocked",
    )
    if fraction < 0.0:
        raise ValueError("simulation.occlusion.fraction_blocked must be in [0, 1].")
    if fraction == 0.0:
        return None
    if fraction > 1.0:
        raise ValueError("simulation.occlusion.fraction_blocked must be in [0, 1].")
    target = data.get("global_segment_id", data.get("infarction_global_segment_id"))
    if target is None:
        raise ValueError(
            "simulation.occlusion.global_segment_id is required when "
            "fraction_blocked > 0."
        )
    return OcclusionConfig(
        global_segment_id=_as_int(
            target, name="simulation.occlusion.global_segment_id"
        ),
        fraction_blocked=fraction,
        include_downstream_when_complete=_as_bool(
            data.get("include_downstream_when_complete"), True
        ),
    )


def _parse_external_field(raw: Any) -> ExternalFieldConfig:
    if raw in (None, {}):
        return ExternalFieldConfig()
    if not isinstance(raw, dict):
        raise ValueError("simulation.external_field must be an object.")
    data = dict(raw)
    _reject_unknown(
        data,
        {"enabled", "scope", "mode"},
        "simulation.external_field",
    )
    enabled = _as_bool(data.get("enabled"), True)
    scope = str(data.get("scope", "shared")).strip().lower().replace("_", "-")
    if scope not in {"per-network", "shared"}:
        raise ValueError(
            "simulation.external_field.scope must be 'per-network' or 'shared'."
        )
    mode = str(data.get("mode", "shared-global")).strip().lower().replace("_", "-")
    if mode != "shared-global":
        raise ValueError(f"Unsupported simulation.external_field.mode: {mode!r}.")
    return ExternalFieldConfig(
        enabled=enabled,
        scope=scope,
        mode=mode,
    )


def _parse_inlet_conditions(raw: Any) -> list[dict[str, float]]:
    if raw in (None, []):
        return []
    if not isinstance(raw, list):
        raise ValueError("simulation.inlet_conditions must be a list.")
    parsed = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"simulation.inlet_conditions[{index}] must be an object.")
        _reject_unknown(
            item,
            {
                "flow_ul_min",
                "inlet_pressure_pa",
                "outlet_pressure_pa",
                "inlet_concentration_mmol_l",
            },
            f"simulation.inlet_conditions[{index}]",
        )
        try:
            parsed.append(
                {
                    "flow_ul_min": _as_float(
                        item["flow_ul_min"],
                        name=f"simulation.inlet_conditions[{index}].flow_ul_min",
                    ),
                    "inlet_pressure_pa": _as_float(
                        item["inlet_pressure_pa"],
                        name=f"simulation.inlet_conditions[{index}].inlet_pressure_pa",
                    ),
                    "outlet_pressure_pa": _as_float(
                        item["outlet_pressure_pa"],
                        name=f"simulation.inlet_conditions[{index}].outlet_pressure_pa",
                    ),
                    "inlet_concentration_mmol_l": _as_float(
                        item["inlet_concentration_mmol_l"],
                        name=f"simulation.inlet_conditions[{index}].inlet_concentration_mmol_l",
                    ),
                }
            )
        except KeyError as exc:
            raise ValueError(
                f"simulation.inlet_conditions[{index}] is missing {exc.args[0]}."
            ) from exc
    return parsed


def _parse_outputs(raw: Any) -> OutputsConfig:
    data = dict(raw or {})
    _reject_unknown(
        data,
        {
            "out_dir",
            "prefix",
            "write_paraview",
            "write_vessels_vtp",
            "write_tissue_vtp",
            "write_summary_csv",
            "write_segments_csv",
            "write_points_csv",
            "include_tissue_nearest_fields",
            "save_network",
            "use_cache",
            "cache_path",
            "export_float_dtype",
            "export_index_dtype",
            "vessel_resolution",
            "write_combined_sweep_csv",
            "combined_sweep_filename",
            "overwrite",
        },
        "outputs",
    )
    return OutputsConfig(
        out_dir=str(data.get("out_dir", "cascade_run")),
        prefix=data.get("prefix"),
        write_paraview=_as_bool(data.get("write_paraview"), True),
        write_vessels_vtp=_as_bool(data.get("write_vessels_vtp"), True),
        write_tissue_vtp=_as_bool(data.get("write_tissue_vtp"), True),
        write_summary_csv=_as_bool(data.get("write_summary_csv"), True),
        write_segments_csv=_as_bool(data.get("write_segments_csv"), True),
        write_points_csv=_as_bool(data.get("write_points_csv"), True),
        include_tissue_nearest_fields=_as_bool(
            data.get("include_tissue_nearest_fields"), False
        ),
        save_network=_as_bool(data.get("save_network"), True),
        use_cache=_as_bool(data.get("use_cache"), True),
        cache_path=data.get("cache_path"),
        export_float_dtype=str(data.get("export_float_dtype", "float64"))
        .strip()
        .lower(),
        export_index_dtype=str(data.get("export_index_dtype", "int64")).strip().lower(),
        vessel_resolution=_as_int(
            data.get("vessel_resolution", 2), name="outputs.vessel_resolution"
        ),
        write_combined_sweep_csv=_as_bool(data.get("write_combined_sweep_csv"), True),
        combined_sweep_filename=str(
            data.get("combined_sweep_filename", "sweep_summary.csv")
        ),
        overwrite=_as_bool(data.get("overwrite"), False),
    )


def _parse_runtime_settings(raw: Any) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "settings must be an object whose keys are named settings sections."
        )
    parsed: dict[str, dict[str, Any]] = {}
    for section, values in raw.items():
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"settings.{section} must be an object.")
        parsed_values = dict(values)
        if str(section).lower() == "hematocrit":
            key = "model" if "model" in parsed_values else "HEMATOCRIT_MODEL"
            if str(parsed_values.get(key, "")).strip().lower() == "constant":
                parsed_values[key] = "uniform_tube"
        parsed[str(section)] = parsed_values
    return parsed


def _parse_fluid(value: Any, name: str, *, allow_both: bool) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a fluid name string; custom fluid objects are not supported."
        )
    mode = value.strip().lower().replace("_", " ")
    mode = {"cellmedia": "cell media", "media": "cell media"}.get(mode, mode)
    allowed = {"blood", "water", "cell media", "custom"}
    if allow_both:
        allowed.add("both")
    if mode not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported {name}: {value!r}. Supported fluids are {choices}."
        )
    return mode


def _validate(
    domain: DomainConfig,
    network: NetworkConfig,
    growth: GrowthConfig,
    simulation: SimulationConfig,
    outputs: OutputsConfig,
) -> None:
    if domain.side_length <= 0.0:
        raise ValueError("domain.side_length must be positive.")
    for name in ("x_length", "y_length", "z_length"):
        value = getattr(domain, name)
        if value is not None and value <= 0.0:
            raise ValueError(f"domain.{name} must be positive.")
    if network.physical_clearance < 0.0:
        raise ValueError("network.physical_clearance must be non-negative.")
    if network.connectivity_geometry_atol < 0.0:
        raise ValueError("network.connectivity_geometry_atol must be non-negative.")
    if network.mode not in {"tree", "forest", "simple"}:
        raise ValueError("network.mode must be 'tree', 'forest', or 'simple'.")
    if network.mode in {"tree", "forest"} and network.input_path is None:
        if not growth.enabled:
            raise ValueError(
                "network.input_path is required for tree/forest runs when "
                "growth.enabled=false."
            )
        if (
            network.target_terminal_count is not None
            and network.target_terminal_count < 1
        ):
            raise ValueError(
                "Generated SVV trees require a target growth count of at least 1."
            )
        if any(value < 1 for value in network.target_terminal_counts):
            raise ValueError(
                "Every generated SVV tree requires a target growth count of at least 1."
            )
        if (
            network.target_total_terminal_count is not None
            and network.target_total_terminal_count < len(network.roots)
        ):
            raise ValueError(
                "A generated SVV forest requires a target growth count of at least 1 "
                "per tree."
            )
    if network.mode == "simple":
        simple_mode = str(network.simple.get("mode", "onechannel")).strip().lower()
        if simple_mode == "custom":
            custom_path = network.simple.get("path", network.simple.get("geometry_path"))
            if not custom_path:
                raise ValueError(
                    "network.simple.path is required when network.simple.mode='custom'."
                )
            if Path(str(custom_path)).suffix.lower() not in {".csv", ".npz"}:
                raise ValueError("Custom geometry must be a .csv or .npz file.")
    if (
        network.mode == "tree"
        and len(network.roots) != 1
        and network.input_path is None
    ):
        raise ValueError(
            "tree mode requires exactly one root unless network.input_path is provided."
        )
    if (
        network.mode == "forest"
        and len(network.roots) < 1
        and network.input_path is None
    ):
        raise ValueError(
            "forest mode requires at least one root unless network.input_path is provided."
        )
    if simulation.flow_source not in {
        "per_tree",
        "per_inlet",
        "total_split",
        "tree-root-flow",
        "tree_root_flow",
        "total-qin-split",
        "total_qin_split",
    }:
        raise ValueError(
            "simulation.flow_source must be one of 'per_tree', 'total_split', "
            "'per_inlet', 'tree-root-flow', or 'total-qin-split'."
        )
    if simulation.flow_source == "per_inlet" and not simulation.inlet_conditions:
        raise ValueError("per-inlet flow requires simulation.inlet_conditions.")
    if simulation.qin_target_ul_min <= 0.0:
        raise ValueError("simulation.qin_target_ul_min must be positive.")
    if simulation.total_qin_ul_min is not None and simulation.total_qin_ul_min <= 0.0:
        raise ValueError("simulation.total_qin_ul_min must be positive.")
    if simulation.distance_sample_count < 0:
        raise ValueError("simulation.distance_sample_count must be non-negative.")
    if (
        simulation.tissue_gpu_validate_points is not None
        and simulation.tissue_gpu_validate_points < 0
    ):
        raise ValueError("simulation.tissue_gpu_validate_points must be non-negative.")
    if simulation.viability_threshold is not None and simulation.viability_threshold < 0:
        raise ValueError("simulation.viability_threshold must be non-negative.")
    if (
        simulation.external_field.enabled
        and simulation.external_field.scope == "shared"
        and simulation.concentration_solver != "topdown_ext_hybrid_bg"
    ):
        raise ValueError(
            "Shared external-field coupling requires "
            "simulation.concentration_solver='topdown_ext_hybrid_bg'."
        )
    if simulation.inlet_conditions and network.input_path is None:
        if len(simulation.inlet_conditions) != len(network.roots):
            raise ValueError(
                "The number of inlet condition sets must match the number of network roots."
            )
    for index, condition in enumerate(simulation.inlet_conditions):
        if condition["flow_ul_min"] <= 0:
            raise ValueError(f"Inlet {index + 1} flow must be positive.")
        if condition["inlet_pressure_pa"] <= condition["outlet_pressure_pa"]:
            raise ValueError(
                f"Inlet {index + 1} pressure must be above its outlet/reference pressure."
            )
        if condition["inlet_concentration_mmol_l"] < 0:
            raise ValueError(
                f"Inlet {index + 1} oxygen concentration cannot be negative."
            )
    if simulation.sample_mode not in {"random", "grid", "file"}:
        raise ValueError("simulation.sample_mode must be 'random', 'grid', or 'file'.")
    if simulation.sample_mode == "file" and not simulation.sample_points_path:
        raise ValueError(
            "simulation.sample_points_path is required when sample_mode is 'file'."
        )
    grid = dict(simulation.tissue_grid or {})
    _reject_unknown(
        grid,
        {
            "nx",
            "ny",
            "nz",
            "boundary_resolution",
            "implicit_margin",
            "disable_enclosed_check",
            "enclosed_tolerance",
            "inside_combine_mode",
            "chunk_points",
            "tissue_grid_chunk_points",
        },
        "simulation.tissue_grid",
    )
    for name in (
        "nx",
        "ny",
        "nz",
        "boundary_resolution",
        "chunk_points",
        "tissue_grid_chunk_points",
    ):
        if name in grid:
            try:
                value = _as_int(
                    grid[name], name=f"simulation.tissue_grid.{name}"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"simulation.tissue_grid.{name} must be a positive integer."
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"simulation.tissue_grid.{name} must be a positive integer."
                )
    if "inside_combine_mode" in grid and str(
        grid["inside_combine_mode"]
    ).strip().lower() not in {"and", "or"}:
        raise ValueError(
            "simulation.tissue_grid.inside_combine_mode must be 'and' or 'or'."
        )
    if "enclosed_tolerance" in grid:
        try:
            tolerance = _as_float(
                grid["enclosed_tolerance"],
                name="simulation.tissue_grid.enclosed_tolerance",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "simulation.tissue_grid.enclosed_tolerance must be finite and non-negative."
            ) from exc
        if tolerance != tolerance or tolerance < 0.0 or tolerance == float("inf"):
            raise ValueError(
                "simulation.tissue_grid.enclosed_tolerance must be finite and non-negative."
            )
    if growth.n_closest_vessels <= 0:
        raise ValueError("growth.n_closest_vessels must be positive.")
    if growth.n_points <= 0:
        raise ValueError("growth.n_points must be positive.")
    if growth.assignment not in {"bulk", "scheduled", "nearest-tree"}:
        raise ValueError(
            "growth.assignment must be 'bulk', 'scheduled', or 'nearest-tree'."
        )
    if growth.bulk_growth_mode not in {
        "always",
        "never",
        "after-collision-threshold",
        "equal-bifurcation",
    }:
        raise ValueError(
            "growth.bulk_growth_mode must be 'always', 'never', 'after-collision-threshold', or 'equal-bifurcation'."
        )
    if growth.add_split_mode not in {"equal", "flow-proportional"}:
        raise ValueError(
            "growth.add_split_mode must be 'equal' or 'flow-proportional'."
        )
    if growth.collision_retry_limit <= 0:
        raise ValueError("growth.collision_retry_limit must be positive.")
    if growth.collision_failure_mode not in {"error", "switch"}:
        raise ValueError("growth.collision_failure_mode must be 'error' or 'switch'.")
    if growth.nearest_tree_batch_points <= 0:
        raise ValueError("growth.nearest_tree_batch_points must be positive.")
    if growth.domain_line_samples < 2:
        raise ValueError("growth.domain_line_samples must be at least 2.")
    if growth.domain_line_tolerance < 0.0:
        raise ValueError("growth.domain_line_tolerance must be non-negative.")
    if growth.strict_domain_max_terminals <= 0:
        raise ValueError("growth.strict_domain_max_terminals must be positive.")
    if growth.n_ignore_collisions is not None and growth.n_ignore_collisions < 0:
        raise ValueError("growth.n_ignore_collisions must be non-negative.")
    if growth.growth_report_every < 0 or growth.checkpoint_every_adds < 0:
        raise ValueError("Growth reporting and checkpoint intervals must be non-negative.")
    if any(value < 0 for value in growth.add_per_tree):
        raise ValueError("growth.add_per_tree values must be non-negative.")
    if growth.add_total is not None and growth.add_total < 0:
        raise ValueError("growth.add_total must be non-negative.")
    if outputs.vessel_resolution <= 0:
        raise ValueError("outputs.vessel_resolution must be positive.")
    if outputs.export_float_dtype not in {"float32", "float64"}:
        raise ValueError("outputs.export_float_dtype must be 'float32' or 'float64'.")
    if outputs.export_index_dtype not in {"int32", "int64"}:
        raise ValueError("outputs.export_index_dtype must be 'int32' or 'int64'.")
