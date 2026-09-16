"""Keep the exhaustive user settings reference synchronized with the code."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from cascade.configuration.settings.registry import SETTINGS_SECTIONS

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "settings.md"


# Canonical user-facing paths outside the generated runtime registry. Aliases
# are documented next to their canonical setting but are not separate controls.
CORE_SETTING_PATHS = (
    "schema_version",
    "domain.type",
    "domain.side_length",
    "domain.dimensions",
    "domain.x_length",
    "domain.y_length",
    "domain.z_length",
    "domain.radius",
    "domain.height",
    "domain.center",
    "domain.theta_resolution",
    "domain.phi_resolution",
    "domain.path",
    "domain.mesh",
    "domain.random_seed",
    "domain.use_cache",
    "domain.cache_dir",
    "network.mode",
    "network.input_path",
    "network.target_terminal_count",
    "network.target_total_terminal_count",
    "network.target_terminal_counts",
    "network.roots[].start",
    "network.roots[].direction",
    "network.physical_clearance",
    "network.save_path",
    "network.repair_connectivity",
    "network.validate_connectivity",
    "network.fail_connectivity",
    "network.connectivity_geometry_atol",
    "network.simple.mode",
    "network.simple.axis",
    "network.simple.radius_cm",
    "network.simple.z_from_bottom_cm",
    "network.simple.flow_ul_min",
    "network.simple.concentration_inlet",
    "network.simple.solve_channels_separately",
    "network.simple.edge_extension_frac",
    "network.simple.y_offsets_cm",
    "network.simple.snake_arc_segments",
    "network.simple.snake_straight_segments",
    "network.simple.lattice_type",
    "network.simple.cells",
    "network.simple.sizing_mode",
    "network.simple.cell_spacing_cm",
    "network.simple.anisotropy_yx",
    "network.simple.anisotropy_zx",
    "network.simple.inlet_points_cm",
    "network.simple.outlet_points_cm",
    "network.simple.radius_expression",
    "network.simple.subdivisions",
    "network.simple.diffusivity",
    "network.simple.vmax",
    "network.simple.km",
    "network.simple.omega",
    "network.simple.window_factor",
    "network.simple.path",
    "network.simple.inlet_nodes",
    "network.simple.outlet_nodes",
    "growth.enabled",
    "growth.assignment",
    "growth.bulk_growth_mode",
    "growth.n_closest_vessels",
    "growth.n_points",
    "growth.weighted_sampling",
    "growth.ignore_collisions",
    "growth.allow_inside_vessels",
    "growth.n_ignore_collisions",
    "growth.collision_retry_limit",
    "growth.collision_failure_mode",
    "growth.nearest_tree_batch_points",
    "growth.strict_domain_segments",
    "growth.strict_domain_max_terminals",
    "growth.domain_line_samples",
    "growth.domain_line_tolerance",
    "growth.growth_report_every",
    "growth.add_per_tree",
    "growth.add_total",
    "growth.add_split_mode",
    "growth.checkpoint_path",
    "growth.checkpoint_every_adds",
    "growth.resume_from_checkpoint",
    "growth.save_target_counts",
    "growth.n_equal_bifurcations",
    "growth.equal_terminal",
    "simulation.fluid",
    "simulation.build_fluid",
    "simulation.qin_target_ul_min",
    "simulation.total_qin_ul_min",
    "simulation.concentration_solver",
    "simulation.distance_sample_count",
    "simulation.flow_source",
    "simulation.inlet_conditions[]",
    "simulation.sample_mode",
    "simulation.sample_points_path",
    "simulation.tissue_grid",
    "simulation.geometry_only",
    "simulation.skip_tissue_oxygen",
    "simulation.compute_avg_distance_to_channel",
    "simulation.tissue_accel",
    "simulation.tissue_gpu_validate_points",
    "simulation.viability_threshold",
    "simulation.occlusion",
    "simulation.external_field",
    "simulation.cext",
    "simulation.tissuesim",
    "flow_ul_min",
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "inlet_concentration_mmol_l",
    "simulation.occlusion.global_segment_id",
    "simulation.occlusion.fraction_blocked",
    "simulation.occlusion.include_downstream_when_complete",
    "simulation.external_field.enabled",
    "simulation.external_field.scope",
    "simulation.external_field.mode",
    "simulation.tissue_grid.nx",
    "simulation.tissue_grid.ny",
    "simulation.tissue_grid.nz",
    "simulation.tissue_grid.boundary_resolution",
    "simulation.tissue_grid.implicit_margin",
    "simulation.tissue_grid.disable_enclosed_check",
    "simulation.tissue_grid.enclosed_tolerance",
    "simulation.tissue_grid.inside_combine_mode",
    "simulation.tissue_grid.chunk_points",
    "outputs.out_dir",
    "outputs.prefix",
    "outputs.write_paraview",
    "outputs.write_vessels_vtp",
    "outputs.write_tissue_vtp",
    "outputs.write_summary_csv",
    "outputs.write_segments_csv",
    "outputs.write_points_csv",
    "outputs.include_tissue_nearest_fields",
    "outputs.save_network",
    "outputs.use_cache",
    "outputs.cache_path",
    "outputs.export_float_dtype",
    "outputs.export_index_dtype",
    "outputs.vessel_resolution",
    "outputs.write_combined_sweep_csv",
    "outputs.combined_sweep_filename",
    "outputs.overwrite",
    "sweep.target_terminal_counts",
    "sweep.fluids",
    "sweep.side_lengths",
    "sweep.qin_target_ul_min_values",
    "sweep.distance_sample_counts",
    "sweep.output_csv",
    "sweep.work_dir",
    "sweep.save_final_network",
    "sweep.legacy_columns_only",
    "sweep.legacy_single_trial_std_nan",
)


def test_generated_runtime_settings_reference_is_current() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_settings_reference.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_settings_reference_covers_every_registered_runtime_setting() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    constants = {
        constant
        for section_name, section in SETTINGS_SECTIONS.items()
        if section_name != "concentration"
        for constant in section.defaults
    }
    for constant in sorted(constants):
        assert f"`{constant}`" in text or f"`{constant.lower()}`" in text, constant


def test_settings_reference_covers_every_core_setting() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    for path in CORE_SETTING_PATHS:
        assert f"`{path}`" in text, path


def test_every_strictly_accepted_core_key_is_named() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    documented = set(re.findall(r"(?<!`)`([^`\n]+)`(?!`)", text))
    accepted: set[str] = set()
    for relative_path in (
        "src/cascade/configuration/parsing.py",
        "src/cascade/vessels/simple.py",
    ):
        tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = function.id if isinstance(function, ast.Name) else None
            if name not in {"_reject_unknown", "reject_unknown"} or len(node.args) < 2:
                continue
            allowed = node.args[1]
            if isinstance(allowed, (ast.Set, ast.Tuple, ast.List)):
                accepted.update(
                    item.value
                    for item in allowed.elts
                    if isinstance(item, ast.Constant) and isinstance(item.value, str)
                )

    missing = sorted(
        key
        for key in accepted
        if not any(
            token.replace("[]", "") == key
            or token.replace("[]", "").endswith(f".{key}")
            or key in token.replace("[]", "").split(".")
            for token in documented
        )
    )
    assert not missing
