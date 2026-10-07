# Settings reference

This page lists the settings accepted by CASCADE configuration files. Settings
are grouped by the object they control, then ordered by importance. Most users
should start with the short list below and leave advanced numerical and
performance settings at their defaults.

## Importance labels

| Label | Meaning |

| **Essential scientific** | Changes the physical problem, geometry, material model, or interpretation of the result. Record these values with study results. |
| **Common workflow** | Commonly changed input, output, or sampling behavior. |
| **Advanced scientific/numerical** | Changes a specialist model or numerical method. Change only with a reason and validation. |
| **Performance/runtime** | Changes speed, memory use, batching, caching, or hardware selection; it should not intentionally change the scientific problem. |
| **Compatibility/internal** | Legacy, debugging, or library-facing setting. Avoid in new files unless directed by a specific guide. |

## Start here: settings that define the study

These are the first settings to review. Their exact relevance depends on the
selected network and solver.

| Setting | Applies to | Why it matters |

| `domain.type` and domain dimensions | Domain | Define the tissue shape and physical size in centimetres. |
| `network.mode` | Network | Selects a generated tree, forest, or deterministic/simple network. |
| `network.input_path` | Network | Selects the saved vascular network used by the study. |
| `network.target_terminal_count` / `target_terminal_counts` | Generated network | Set vascular-tree size. |
| `network.roots[].start` and `direction` | Generated network | Set inlet locations and initial growth directions. |
| `simulation.fluid` | Simulation | Selects blood, water, cell media, custom fluid, or paired-fluid execution. |
| `simulation.flow_source` and flow inputs | Hemodynamics | Define whether flow is per tree, per inlet, or split from a total. |
| `simulation.inlet_conditions[]` | Per-inlet solve | Define inlet flow, inlet/outlet pressure, and inlet concentration. |
| `settings.hemodynamics.root_pressure` / `terminal_pressure` | Hemodynamics | Define the pressure boundary condition when the selected mode uses pressure. |
| `settings.hemodynamics.qin_target` | Hemodynamics | Defines the runtime flow target used by the underlying solver. |
| `settings.hematocrit.model` and `hd_discharge` | Blood | Define red-cell partitioning and discharge hematocrit. |
| `settings.oxygen.concentration_inlet_by_fluid` | Oxygen | Defines inlet solute concentration for each fluid. |
| `settings.oxygen.solute_diffusivity` | Oxygen | Defines tissue solute diffusivity. |
| `settings.oxygen.vmax_mm` | Oxygen | Defines maximum Michaelis-Menten tissue consumption rate. |
| `settings.oxygen.k_m_mm` | Oxygen | Defines the Michaelis-Menten half-saturation concentration. |
| `settings.oxygen.finite_radius_o2_terms` / `lumen_wall_closure` | Oxygen | Select finite-radius and lumen-wall transport physics. |
| `simulation.viability_threshold` | Results | Defines the concentration threshold used for viability statistics. |
| `simulation.sample_mode` and sample/grid controls | Tissue | Define where tissue concentration is evaluated. |

Example placement:

```json
{
  "simulation": {
    "fluid": "blood",
    "qin_target_ul_min": 100.0,
    "viability_threshold": 0.01
  },
  "settings": {
    "oxygen": {
      "vmax_mm": 0.04,
      "k_m_mm": 0.0069
    },
    "hematocrit": {
      "model": "pries_secomb",
      "hd_discharge": 0.45
    }
  }
}
```

## Where settings go

| Object | Controls |

| `domain` | Tissue geometry and domain cache. |
| `network` | Network source, roots, terminal targets, and connectivity validation. |
| `network.simple` | Deterministic channels, lattices, and custom CSV/NPZ networks. |
| `growth` | Vascular growth, collision policy, checkpoints, and growth allocation. |
| `simulation` | Fluids, boundary conditions, solver selection, tissue sampling, and run modes. |
| `settings.<section>` | Scientific model and expert solver/runtime defaults. |
| `outputs` | Output location, formats, data types, and replacement policy. |
| `sweep` | Values and output policy for `cascade sweep`. |
| `gui` | Studio-managed project state. It is not a simulation setting and should not be edited manually. |

Paths are resolved relative to the settings file unless a setting says
otherwise. Geometry lengths are in centimetres, flow is in microlitres per
minute, pressure is in pascals, and concentration is in mol/m³ (numerically
equal to mmol/L) unless noted otherwise.

The optional top-level `schema_version` value is reserved for configuration
format metadata. `runtime_settings` is an accepted historic alias for
`settings`; use `settings` in new files.

## Domain settings

| Setting | Importance | Default | Meaning |

| `domain.type` | Essential scientific | `"cube"` | `cube`, `box`, `sphere`, `cylinder`, or a path-backed mesh domain. |
| `domain.side_length` | Essential scientific | `1.0` | Cube side length, or fallback length for unspecified box axes, in cm. |
| `domain.x_length` | Essential scientific | `null` | Box x length in cm. |
| `domain.y_length` | Essential scientific | `null` | Box y length in cm. |
| `domain.z_length` | Essential scientific | `null` | Box z length in cm. |
| `domain.dimensions` | Essential scientific | `null` | Three box lengths `[x, y, z]` in cm; an alternative to individual lengths. |
| `domain.radius` | Essential scientific | `null` | Sphere or cylinder radius in cm. |
| `domain.height` | Essential scientific | `null` | Cylinder height in cm. |
| `domain.center` | Essential scientific | `null` | Domain centre `[x, y, z]` in cm. |
| `domain.path` | Essential scientific | `null` | STL or supported mesh path for a mesh-backed domain. |
| `domain.random_seed` | Common workflow | `42` | Seed used for reproducible domain sampling. |
| `domain.theta_resolution` | Advanced scientific/numerical | `12` | Sphere/cylinder angular boundary resolution; minimum 8. |
| `domain.phi_resolution` | Advanced scientific/numerical | `8` | Sphere boundary polar resolution; minimum 8. |
| `domain.use_cache` | Performance/runtime | `true` | Reuse compatible domain cache data. |
| `domain.cache_dir` | Performance/runtime | `null` | Optional external domain-cache directory. |
| `domain.mesh` | Compatibility/internal | `null` | In-memory mesh object for library use; JSON users should use `path`. |

Accepted aliases: `kind` for `type`, `side_len` for `side_length`, `lengths`
for `dimensions`, `box_x_cm`/`box_y_cm`/`box_z_cm` for the three lengths,
`sphere_radius` for `radius`, `cylinder_height` for `height`, and
`sphere_theta_resolution`/`sphere_phi_resolution` for the two resolutions.

## Network settings

| Setting | Importance | Default | Meaning |

| `network.mode` | Essential scientific | `"tree"` | `tree`, `forest`, or `simple`. |
| `network.input_path` | Essential scientific | `null` | Saved tree/forest NPZ path. Omit to generate a network. |
| `network.target_terminal_count` | Essential scientific | `100` | Growth count `N` for one generated tree. SVV starts with one root segment and performs `N` bifurcation additions, yielding `N + 1` terminal leaves and `2N + 1` total segments. |
| `network.target_total_terminal_count` | Essential scientific | `null` | Total growth count distributed over a generated forest; each tree contributes its initial root segment in addition to its assigned count. |
| `network.target_terminal_counts` | Essential scientific | `[]` | Per-tree growth counts for a generated forest, using the same `2N + 1` segment convention. |
| `network.roots[].start` | Essential scientific | generated default | Root start `[x, y, z]` in cm. |
| `network.roots[].direction` | Essential scientific | `null` | Initial direction vector or endpoint, as expected by the growth engine. |
| `network.physical_clearance` | Essential scientific | `0.0` | Required physical clearance between trees/vessels in cm. |
| `network.repair_connectivity` | Advanced scientific/numerical | `true` | Repair supported connectivity defects when loading a network. |
| `network.validate_connectivity` | Advanced scientific/numerical | `false` | Run connectivity validation on the loaded/generated graph. |
| `network.fail_connectivity` | Advanced scientific/numerical | `true` | Fail instead of warning when connectivity validation reports an error. |
| `network.connectivity_geometry_atol` | Advanced scientific/numerical | `1e-6` | Geometric absolute tolerance for connectivity checks in cm. |
| `network.save_path` | Common workflow | `null` | Optional explicit path for the saved generated network. |

Accepted aliases: `path` for `input_path`, `target_count` for
`target_terminal_count`, `target_total_terminals` for
`target_total_terminal_count`, `target_counts` for `target_terminal_counts`,
`root` or top-level `start`/`direction` for a single root,
`fail_on_connectivity_error` for `fail_connectivity`, and `simple_geometry`
for `simple`.

### Deterministic and simple networks

These settings apply only when `network.mode` is `simple`.

| Setting | Importance | Default | Meaning |

| `network.simple.mode` | Essential scientific | `"onechannel"` | Channel, snake, lattice, or `custom` network type. |
| `network.simple.axis` | Essential scientific | `"x"` | Main channel axis. |
| `network.simple.radius_cm` | Essential scientific | `0.015` | Default vessel radius in cm. |
| `network.simple.flow_ul_min` | Essential scientific | simulation flow | Flow assigned to the simple network override (in μL/min). |
| `network.simple.concentration_inlet` | Essential scientific | fluid default | Inlet concentration override. |
| `network.simple.diffusivity` | Essential scientific | oxygen default | Simple-network diffusivity override. |
| `network.simple.vmax` | Essential scientific | oxygen default | Maximum consumption-rate override. |
| `network.simple.km` | Essential scientific | oxygen default | Michaelis-Menten half-saturation override. |
| `network.simple.path` | Essential scientific | `null` | Required CSV/NPZ path when `mode` is `custom`. |
| `network.simple.inlet_nodes` | Essential scientific | inferred | Custom-network inlet node IDs. |
| `network.simple.outlet_nodes` | Essential scientific | inferred | Custom-network outlet node IDs. |
| `network.simple.lattice_type` | Essential scientific | mode | Lattice topology; see the custom geometry guide. |
| `network.simple.cells` | Essential scientific | `4` | Number of lattice cells along x in cell-count sizing mode. |
| `network.simple.sizing_mode` | Essential scientific | `"cells"` | Use cell count or cell spacing to size the lattice. |
| `network.simple.cell_spacing_cm` | Essential scientific | derived | Requested lattice spacing in cm when spacing mode is selected. |
| `network.simple.anisotropy_yx` | Essential scientific | `1.0` | Lattice y-to-x spacing ratio. |
| `network.simple.anisotropy_zx` | Essential scientific | `1.0` | Lattice z-to-x spacing ratio. |
| `network.simple.inlet_points_cm` | Essential scientific | topology default | Explicit lattice inlet points in cm. |
| `network.simple.outlet_points_cm` | Essential scientific | topology default | Explicit lattice outlet points in cm. |
| `network.simple.radius_expression` | Essential scientific | topology default | Lattice vessel-radius expression. |
| `network.simple.z_from_bottom_cm` | Essential scientific | half domain height | Channel z offset from the domain bottom. |
| `network.simple.y_offsets_cm` | Essential scientific | `[-0.2,0,0.2]` | Parallel-channel y offsets. |
| `network.simple.edge_extension_frac` | Advanced scientific/numerical | `0.0` | Fractional tissue-line extension beyond channel ends. |
| `network.simple.omega` | Advanced scientific/numerical | oxygen default | Simple-solver relaxation override. |
| `network.simple.window_factor` | Advanced scientific/numerical | oxygen default | Simple-solver spatial window override. |
| `network.simple.solve_channels_separately` | Advanced scientific/numerical | `true` | Solve independent channels separately where supported. |
| `network.simple.snake_arc_segments` | Advanced scientific/numerical | `5` | Segments per curved portion of snake geometry. |
| `network.simple.snake_straight_segments` | Advanced scientific/numerical | `5` | Segments per straight portion of snake geometry. |
| `network.simple.subdivisions` | Advanced scientific/numerical | `1` | Subdivide each generated lattice edge. |

Accepted aliases: `channel_mode`, `channel_axis`, `channel_radius_cm`,
`channel_z_from_bottom_cm`, `qin_ul_min`, `conc_inlet`,
`edge_channel_extension_frac`, `channel_y_offsets_cm`, `cells_per_axis`,
`segment_subdivisions`, and `geometry_path` map to the corresponding canonical
keys above.

## Growth settings

| Setting | Importance | Default | Meaning |

| `growth.enabled` | Essential scientific | `true` | Grow a generated or extendable network. |
| `growth.assignment` | Essential scientific | `"bulk"` | Growth assignment policy, including nearest-tree allocation. |
| `growth.bulk_growth_mode` | Essential scientific | context dependent | Bulk growth behavior for the selected assignment. |
| `growth.n_closest_vessels` | Essential scientific | `2` | Candidate vessels considered for new terminal assignment. |
| `growth.n_points` | Advanced scientific/numerical | `50` | Candidate points considered during a growth step. |
| `growth.weighted_sampling` | Advanced scientific/numerical | `false` | Weight growth-point sampling. |
| `growth.ignore_collisions` | Essential scientific | `true` | Permit growth without rejecting vessel collisions. |
| `growth.allow_inside_vessels` | Essential scientific | `true` | Permit points inside existing vessel volumes. |
| `growth.n_ignore_collisions` | Advanced scientific/numerical | `null` | Initial additions for which collision checks are ignored. |
| `growth.collision_retry_limit` | Advanced scientific/numerical | `100` | Attempts after a collision before applying failure policy. |
| `growth.collision_failure_mode` | Advanced scientific/numerical | `"error"` | Behavior after collision retries are exhausted. |
| `growth.strict_domain_segments` | Essential scientific | `false` | Require grown segments to remain inside the domain. |
| `growth.strict_domain_max_terminals` | Performance/runtime | `10000` | Terminal-count limit for strict segment-domain checking. |
| `growth.domain_line_samples` | Advanced scientific/numerical | `4` | Interior line samples used by strict domain checking. |
| `growth.domain_line_tolerance` | Advanced scientific/numerical | `0.0` | Domain containment tolerance for segment samples. |
| `growth.add_per_tree` | Essential scientific | `[]` | Explicit number of terminals to add to each tree. |
| `growth.add_total` | Essential scientific | `null` | Total terminals to add across a forest. |
| `growth.add_split_mode` | Essential scientific | `"equal"` | How `add_total` is divided among trees. |
| `growth.n_equal_bifurcations` | Essential scientific | `null` | Number of equal-bifurcation additions. |
| `growth.equal_terminal` | Advanced scientific/numerical | `{}` | Nested equal-terminal/equal-bifurcation controls. |
| `growth.checkpoint_path` | Common workflow | `null` | Growth checkpoint file. |
| `growth.checkpoint_every_adds` | Common workflow | `0` | Save a checkpoint after this many additions; zero disables. |
| `growth.resume_from_checkpoint` | Common workflow | `false` | Resume from `checkpoint_path`. |
| `growth.save_target_counts` | Common workflow | `[]` | Save intermediate networks at these growth counts. |
| `growth.growth_report_every` | Performance/runtime | `0` | Console progress interval; zero disables periodic reports. |
| `growth.nearest_tree_batch_points` | Performance/runtime | `256` | Batch size for nearest-tree assignment. |

Accepted aliases: `growth_assignment`, `nearest_batch_points`,
`checkpoint_forest`, `equal_bifurcation`, and an embedded
`equal_terminal.n_equal_bifurcations` value.

## Simulation settings

| Setting | Importance | Default | Meaning |

| `simulation.fluid` | Essential scientific | `"blood"` | `blood`, `water`, `cell media`, `custom`, or `both`. |
| `simulation.build_fluid` | Essential scientific | selected fluid | Fluid used while building geometry when `fluid` is `both`. |
| `simulation.qin_target_ul_min` | Essential scientific | `100.0` | Per-tree/inlet target flow, depending on `flow_source`. |
| `simulation.total_qin_ul_min` | Essential scientific | `null` | Total flow divided among inlets by total-split modes. |
| `simulation.flow_source` | Essential scientific | `"per_tree"` | Per-tree, per-inlet, total-split, or loaded-tree-root flow policy. |
| `simulation.inlet_conditions[]` | Essential scientific | `[]` | Per-inlet boundary-condition objects described below. |
| `simulation.concentration_solver` | Advanced scientific/numerical | `"network_ext"` | Concentration solver family. |
| `simulation.sample_mode` | Common workflow | `"random"` | Tissue point mode: random, grid, or loaded sample points. |
| `simulation.distance_sample_count` | Common workflow | `10000` | Number of random tissue samples. |
| `simulation.sample_points_path` | Common workflow | `null` | NPZ/compatible path for loaded tissue sample points. |
| `simulation.tissue_grid` | Common workflow | `{}` | Regular-grid controls described below. |
| `simulation.viability_threshold` | Essential scientific | `null` | Concentration threshold used for tissue viability statistics. |
| `simulation.occlusion` | Essential scientific | `null` | Reversible solve-time occlusion described below. |
| `simulation.external_field` | Advanced scientific/numerical | disabled | Cross-network external-field coupling described below. |
| `simulation.geometry_only` | Common workflow | `false` | Build/export geometry without running the transport simulation. |
| `simulation.skip_tissue_oxygen` | Common workflow | `false` | Solve vessels but skip tissue oxygen evaluation. |
| `simulation.compute_avg_distance_to_channel` | Common workflow | `false` | Compute the average tissue-to-vessel distance statistic. |
| `simulation.tissue_accel` | Performance/runtime | `"auto"` | Tissue accelerator selection. |
| `simulation.tissue_gpu_validate_points` | Performance/runtime | `null` | GPU tissue points compared against a validation path. |
| `simulation.cext` | Compatibility/internal | `{}` | Legacy/direct Cext override mapping; prefer `settings.cext`. |
| `simulation.tissuesim` | Compatibility/internal | `{}` | Legacy/direct solver overrides; prefer named `settings.*` sections. |

Accepted aliases: `qin_target`, `tissue_sample_mode`, `sample_file`, `grid`,
`infarction`, and `overrides`. Several historic flat solver keys are also
accepted and copied into `tissuesim`: `finite_radius_o2_terms`,
`lumen_wall_closure`, `graetz_n_radial`, `graetz_n_modes`,
`graetz_max_fp_iters`, `nearest_tissue_vessels`, `window_factor`,
`hematocrit_model`, `hematocrit_flow_iterations`, `kirchhoff_bc_mode`,
`kirchhoff_solver`, and `solver_timing_details`. Prefer their named runtime
sections in new files.

### Per-inlet boundary conditions

Every `simulation.inlet_conditions[]` object requires all four values.

| Setting | Importance | Meaning |

| `flow_ul_min` | Essential scientific | Inlet flow in µL/min. |
| `inlet_pressure_pa` | Essential scientific | Inlet pressure in Pa. |
| `outlet_pressure_pa` | Essential scientific | Outlet/reference pressure in Pa. |
| `inlet_concentration_mmol_l` | Essential scientific | Inlet concentration in mmol/L. |

### Occlusion

| Setting | Importance | Default | Meaning |

| `simulation.occlusion.global_segment_id` | Essential scientific | required | Global segment ID to occlude. |
| `simulation.occlusion.fraction_blocked` | Essential scientific | `0.0` | Blocked fraction from 0 to 1; zero disables the occlusion. |
| `simulation.occlusion.include_downstream_when_complete` | Essential scientific | `true` | Include downstream segments when the target is fully blocked. |

The historic keys `infarction_global_segment_id` and
`fraction_blocked_infarction` are accepted aliases.

### External-field coupling

| Setting | Importance | Default | Meaning |

| `simulation.external_field.enabled` | Advanced scientific/numerical | `false` | Enable vessel-to-vessel external-field coupling. |
| `simulation.external_field.scope` | Advanced scientific/numerical | `"per-network"` | `per-network` or `shared`; shared requires the supported hybrid solver. |
| `simulation.external_field.mode` | Advanced scientific/numerical | `"shared-global"` | Coupling implementation; currently only `shared-global` is supported. |

### Tissue grid

These apply when `simulation.sample_mode` is `grid`.

| Setting | Importance | Default | Meaning |

| `simulation.tissue_grid.nx` | Common workflow | `200` | Grid points along x. |
| `simulation.tissue_grid.ny` | Common workflow | `200` | Grid points along y. |
| `simulation.tissue_grid.nz` | Common workflow | `200` | Grid points along z. |
| `simulation.tissue_grid.boundary_resolution` | Advanced scientific/numerical | `28` | Boundary resolution used when a boundary must be built. |
| `simulation.tissue_grid.implicit_margin` | Advanced scientific/numerical | `0.0` | Margin applied to implicit inside-domain testing. |
| `simulation.tissue_grid.disable_enclosed_check` | Advanced scientific/numerical | `true` | Use only the implicit check instead of combining it with a mesh enclosure check. |
| `simulation.tissue_grid.enclosed_tolerance` | Advanced scientific/numerical | `1e-6` | Tolerance for mesh enclosed-point testing. |
| `simulation.tissue_grid.inside_combine_mode` | Advanced scientific/numerical | `"and"` | Combine implicit and enclosure masks with `and` or `or`. |
| `simulation.tissue_grid.chunk_points` | Performance/runtime | `250000` | Approximate point count processed per grid chunk. |

`tissue_grid_chunk_points` is accepted as an alias for `chunk_points`.

## Output settings

| Setting | Importance | Default | Meaning |

| `outputs.out_dir` | Common workflow | `"cascade_run"` | Output directory. |
| `outputs.prefix` | Common workflow | automatic | Filename prefix. |
| `outputs.write_paraview` | Common workflow | `true` | Master switch for ParaView-family outputs. |
| `outputs.write_vessels_vtp` | Common workflow | `true` | Write vessel polydata. |
| `outputs.write_tissue_vtp` | Common workflow | `true` | Write tissue point data. |
| `outputs.write_summary_csv` | Common workflow | `true` | Write run summary CSV. |
| `outputs.write_segments_csv` | Common workflow | `true` | Write per-segment CSV. |
| `outputs.write_points_csv` | Common workflow | `true` | Write tissue-point CSV. |
| `outputs.include_tissue_nearest_fields` | Common workflow | `false` | Include nearest-vessel fields with tissue outputs. |
| `outputs.save_network` | Common workflow | `true` | Save the network archive. |
| `outputs.overwrite` | Common workflow | `false` | Permit replacing an existing output set. |
| `outputs.vessel_resolution` | Advanced scientific/numerical | `2` | Vessel tube resolution used for VTK geometry. |
| `outputs.export_float_dtype` | Advanced scientific/numerical | `"float64"` | Floating-point dtype used in exported arrays. |
| `outputs.export_index_dtype` | Advanced scientific/numerical | `"int64"` | Integer dtype used in exported arrays. |
| `outputs.use_cache` | Performance/runtime | `true` | Reuse compatible output/network cache data. |
| `outputs.cache_path` | Performance/runtime | `null` | Optional explicit cache path. |
| `outputs.write_combined_sweep_csv` | Common workflow | `true` | Write one combined sweep summary. |
| `outputs.combined_sweep_filename` | Common workflow | `"sweep_summary.csv"` | Combined sweep filename. |

## Sweep settings

These settings are used by `cascade sweep`. A scalar is accepted where a
single-item list is meaningful.

| Setting | Importance | Default | Meaning |

| `sweep.target_terminal_counts` | Essential scientific | network target | Growth counts to run, using the `2N + 1` segment convention. |
| `sweep.fluids` | Essential scientific | simulation fluid | Fluids to run; `both` expands to water and blood. |
| `sweep.side_lengths` | Essential scientific | domain side length | Cube side lengths in cm. |
| `sweep.qin_target_ul_min_values` | Essential scientific | simulation flow | Flow targets in µL/min. |
| `sweep.distance_sample_counts` | Common workflow | simulation count | Tissue sample counts used for summary statistics. |
| `sweep.output_csv` | Common workflow | derived | Combined summary CSV path. |
| `sweep.work_dir` | Common workflow | output directory | Directory for individual sweep runs. |
| `sweep.save_final_network` | Common workflow | output policy | Save the final network for each compatible growth series. |
| `sweep.legacy_columns_only` | Compatibility/internal | `false` | Emit the historic reduced summary schema. |
| `sweep.legacy_single_trial_std_nan` | Compatibility/internal | `false` | Preserve historic NaN standard deviation for one-trial results. |

Accepted aliases: `target_counts`, `cube_side_lengths`, `qin_target_values`,
`summary_csv`, and `out_dir`.

## Expert solver and runtime settings

The remaining settings live inside the top-level `settings` object. They are
grouped by the model or solver they affect. The tables are generated directly
from CASCADE's setting registry so new runtime options cannot be added without
making this reference stale.

The displayed defaults are CLI/runtime fallback defaults. Studio normally
saves explicit project values, which take precedence and may intentionally
differ from a fallback. Always inspect the saved project settings when
reproducing a Studio study.

<!-- BEGIN GENERATED RUNTIME SETTINGS -->

The tables below are generated from CASCADE's runtime-setting registry.
Use the shorter JSON keys shown here. Exact uppercase constant names are
also accepted, but are intended mainly for compatibility and debugging.

<details><summary><code>settings.growth</code> — 41 options</summary>

Equal-terminal and equal-bifurcation growth controls.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `n_equal_bifurcations` | Common workflow | `null` | Terminal count at which growth switches from CCO bifurcation optimization to the fast equal-bifurcation rule. |
| `alpha_max_deg` `EQUAL_TERMINAL_ALPHA_MAX_DEG` | Advanced scientific/numerical | `85.0` | Largest allowed branch opening angle, in degrees, for equal-bifurcation candidate generation. |
| `alpha_min_deg` `EQUAL_TERMINAL_ALPHA_MIN_DEG` | Advanced scientific/numerical | `5.0` | Smallest allowed branch opening angle, in degrees, for equal-bifurcation candidate generation. |
| `alpha_mode_deg` `EQUAL_TERMINAL_ALPHA_MODE_DEG` | Advanced scientific/numerical | `37.5` | Preferred branch opening angle, in degrees, used when generating equal-bifurcation candidates. |
| `check_midpoint` `EQUAL_TERMINAL_CHECK_MIDPOINT` | Advanced scientific/numerical | `true` | If true, check the midpoint of each proposed segment against the domain as well as the endpoint. |
| `density_alpha` `EQUAL_TERMINAL_DENSITY_ALPHA` | Advanced scientific/numerical | `-0.17` | Density-model exponent controlling how local vessel crowding changes branch length. |
| `density_beta` `EQUAL_TERMINAL_DENSITY_BETA` | Advanced scientific/numerical | `0.6` | Density-model exponent controlling how generation/depth changes branch length. |
| `density_epsilon` `EQUAL_TERMINAL_DENSITY_EPSILON` | Advanced scientific/numerical | `0.99` | Small positive floor that prevents divide-by-zero in density-based length formulas. |
| `density_k` `EQUAL_TERMINAL_DENSITY_K` | Advanced scientific/numerical | `3` | Neighbor count used by density-based equal-bifurcation length estimates. |
| `domain_margin` `EQUAL_TERMINAL_DOMAIN_MARGIN` | Advanced scientific/numerical | `0.0` | Extra distance required between a candidate segment and the domain boundary. |
| `equal_terminal_n_candidates` | Advanced scientific/numerical | `3` | Number of candidate branch directions tested for each terminal in equal-bifurcation mode. |
| `equal_terminal_only` `EQUAL_TERMINAL_ONLY_ENABLE` | Advanced scientific/numerical | `true` | If true, equal-bifurcation mode only adds from terminal segments instead of any eligible segment. |
| `gen_f_k` `EQUAL_TERMINAL_GEN_F_K` | Advanced scientific/numerical | `3` | Neighbor count used by the generation-and-flow mixture length model. |
| `gen_f_mix_mu1_ln` `EQUAL_TERMINAL_GEN_F_MIX_MU1_LN` | Advanced scientific/numerical | `-1.4940805101469532` | Mean of the first log-normal component for the generation-and-flow length model. |
| `gen_f_mix_mu2_ln` `EQUAL_TERMINAL_GEN_F_MIX_MU2_LN` | Advanced scientific/numerical | `-0.9535671641507659` | Mean of the second log-normal component for the generation-and-flow length model. |
| `gen_f_mix_sig1_ln` `EQUAL_TERMINAL_GEN_F_MIX_SIG1_LN` | Advanced scientific/numerical | `0.6459130145614477` | Standard deviation of the first log-normal component for the generation-and-flow length model. |
| `gen_f_mix_sig2_ln` `EQUAL_TERMINAL_GEN_F_MIX_SIG2_LN` | Advanced scientific/numerical | `0.34151613489071186` | Standard deviation of the second log-normal component for the generation-and-flow length model. |
| `gen_f_mix_w` `EQUAL_TERMINAL_GEN_F_MIX_W` | Advanced scientific/numerical | `0.19928806732283386` | Mixture weight between the two log-normal components in the generation-and-flow length model. |
| `length` `EQUAL_TERMINAL_LENGTH` | Advanced scientific/numerical | `null` | Fixed equal-bifurcation child length. None means compute length from the selected length model. |
| `length_floor` `EQUAL_BIFURCATION_LENGTH_FLOOR` | Advanced scientific/numerical | `0.0001` | Lower bound on generated child lengths in equal-bifurcation mode. |
| `length_max` `EQUAL_TERMINAL_LENGTH_MAX` | Advanced scientific/numerical | `1.0` | Maximum equal-bifurcation segment length. |
| `length_min` `EQUAL_TERMINAL_LENGTH_MIN` | Advanced scientific/numerical | `0.0001` | Minimum equal-bifurcation segment length. |
| `length_mode` `EQUAL_TERMINAL_LENGTH_MODE` | Advanced scientific/numerical | `"gen_f_mix2"` | Length model used when fixed child length is not supplied. |
| `length_power` `EQUAL_TERMINAL_LENGTH_POWER` | Advanced scientific/numerical | `-0.33` | Power-law exponent used by the simple equal-terminal length model. |
| `length_scale` `EQUAL_TERMINAL_LENGTH_SCALE` | Advanced scientific/numerical | `0.01` | Multiplicative scale applied to generated equal-bifurcation segment lengths. |
| `length_shrink` `EQUAL_TERMINAL_LENGTH_SHRINK` | Advanced scientific/numerical | `0.7` | Factor used to shorten a candidate if the original length does not fit in the domain. |
| `psi_max_deg` `EQUAL_TERMINAL_PSI_MAX_DEG` | Advanced scientific/numerical | `120.0` | Maximum angular sweep, in degrees, around the parent segment when searching directions. |
| `psi_step_deg` `EQUAL_TERMINAL_PSI_STEP_DEG` | Advanced scientific/numerical | `10.0` | Angular step, in degrees, used when sweeping candidate directions around the parent segment. |
| `radius_floor` `EQUAL_BIFURCATION_RADIUS_FLOOR` | Advanced scientific/numerical | `0.0001` | Lower bound on generated child radii in equal-bifurcation mode. |
| `t_min` `EQUAL_BIFURCATION_T_MIN` | Advanced scientific/numerical | `0.001` | Minimum bifurcation location fraction along the parent segment in equal-bifurcation updates. |
| `total_length_log10_intercept` `EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_INTERCEPT` | Advanced scientific/numerical | `0.4795` | Intercept for estimating total network length from terminal count on a log10 scale. |
| `total_length_log10_slope` `EQUAL_TERMINAL_TOTAL_LENGTH_LOG10_SLOPE` | Advanced scientific/numerical | `0.688` | Slope for estimating total network length from terminal count on a log10 scale. |
| `defer_hnsw_updates` `EQUAL_TERMINAL_DEFER_HNSW_UPDATES` | Performance/runtime | `true` | If true, delay expensive nearest-neighbor index updates during equal-bifurcation batches. |
| `domain_chunk` `EQUAL_TERMINAL_DOMAIN_CHUNK` | Performance/runtime | `0` | Candidate chunk size for equal-bifurcation domain checks; 0 lets the code choose. |
| `domain_parallel_min_points` `EQUAL_TERMINAL_DOMAIN_PARALLEL_MIN_POINTS` | Performance/runtime | `0` | Minimum number of candidate points before parallel domain checks are used. |
| `domain_workers` `EQUAL_TERMINAL_DOMAIN_WORKERS` | Performance/runtime | `max(1, CPU count - 2)` | Worker count used for domain-inside checks during equal-bifurcation growth. |
| `equal_terminal_batch_size` | Performance/runtime | `500` | Number of equal-bifurcation additions attempted in one batch before bookkeeping is refreshed. |
| `record_add_times` `EQUAL_TERMINAL_RECORD_ADD_TIMES` | Performance/runtime | `false` | If true, store per-add timing details for equal-bifurcation debugging. |
| `report_every` `EQUAL_TERMINAL_REPORT_EVERY` | Performance/runtime | `1` | Report equal-bifurcation progress every this many batches when timing reports are enabled. |
| `report_timings` `EQUAL_TERMINAL_REPORT_TIMINGS` | Performance/runtime | `false` | If true, print equal-bifurcation timing summaries while growing. |
| `debug_add_vessel` | Compatibility/internal | `false` | If true, print low-level vessel-add debug output from svv/CASCADE growth routines. |

</details>

<details><summary><code>settings.hemodynamics</code> — 8 options</summary>

Physical pressures, flow targets, and fluid properties.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `custom_fluid_density_g_cm3` | Essential scientific | `1.0` | Density of a user-defined constant-viscosity perfusate, in g/cm^3. |
| `custom_fluid_dynamic_viscosity_cp` | Essential scientific | `1.0` | Dynamic viscosity of a user-defined perfusate, in centipoise. |
| `qin_target` | Essential scientific | `900.0` | Target inlet flow used when a run does not provide a different flow, in microliters per minute. |
| `root_pressure` | Essential scientific | `66661.0` | Pressure at the root/inlet of the tree, in pascals. |
| `terminal_pressure` | Essential scientific | `40000.0` | Pressure assigned to terminal outlets before any side-length scaling, in pascals. |
| `scale_dp_by_volume` `SCALE_dP_BY_VOLUME` | Advanced scientific/numerical | `true` | Scale dp by volume. |
| `scale_nterms_by_volume` | Advanced scientific/numerical | `false` | If true, scale requested terminal counts by the domain volume. Usually left false in CASCADE runs. |
| `scale_q_by_volume` | Advanced scientific/numerical | `true` | If true, scale inlet flow by side_length^3 so larger domains receive proportionally more flow. |

</details>

<details><summary><code>settings.kirchhoff</code> — 19 options</summary>

Pressure/flow linear-solver choices and convergence controls.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `bc_mode` `KIRCHHOFF_BC_MODE` | Advanced scientific/numerical | `"legacy_equal_terminal_flow"` | Boundary condition mode for flow. The legacy mode enforces equal terminal flow then reconstructs pressure. |
| `cg_maxiter` `KIRCHHOFF_CG_MAXITER` | Advanced scientific/numerical | `10000` | Maximum conjugate-gradient iterations before the solve is considered failed. |
| `cg_min_nodes` `KIRCHHOFF_CG_MIN_NODES` | Advanced scientific/numerical | `50000` | Node-count threshold where automatic sparse solving may switch to conjugate gradient. |
| `cg_rtol` `KIRCHHOFF_CG_RTOL` | Advanced scientific/numerical | `1e-10` | Relative residual tolerance for conjugate-gradient pressure solves. |
| `gmres_eq_diag_floor_rel` `KIRCHHOFF_GMRES_EQ_DIAG_FLOOR_REL` | Advanced scientific/numerical | `1e-12` | Lower bound for diagonal scaling entries, relative to the median positive diagonal. |
| `gmres_equilibrate` `KIRCHHOFF_GMRES_EQUILIBRATE` | Advanced scientific/numerical | `true` | If true, diagonally rescale the pressure system before GMRES to improve conditioning. |
| `gmres_maxiter` `KIRCHHOFF_GMRES_MAXITER` | Advanced scientific/numerical | `1000` | Maximum GMRES iterations before the solve is considered failed. |
| `gmres_restart` `KIRCHHOFF_GMRES_RESTART` | Advanced scientific/numerical | `300` | Number of Krylov vectors kept before GMRES restarts; larger values use more memory. |
| `gmres_retry_unscaled_if_eq_fail` `KIRCHHOFF_GMRES_RETRY_UNSCALED_IF_EQ_FAIL` | Advanced scientific/numerical | `true` | If true, retry GMRES without equilibration if the scaled system fails. |
| `gmres_rtol` `KIRCHHOFF_GMRES_RTOL` | Advanced scientific/numerical | `0.0001` | Relative residual tolerance for GMRES pressure solves. |
| `ilu_drop_tol` `KIRCHHOFF_ILU_DROP_TOL` | Advanced scientific/numerical | `1e-07` | Drop tolerance for incomplete-LU preconditioning; smaller values keep more matrix entries. |
| `ilu_fill_factor` `KIRCHHOFF_ILU_FILL_FACTOR` | Advanced scientific/numerical | `300` | Maximum fill allowed in incomplete-LU preconditioning; larger values can be more robust but use more memory. |
| `ilu_permc_specs` `KIRCHHOFF_ILU_PERMC_SPECS` | Advanced scientific/numerical | `["COLAMD","MMD_AT_PLUS_A","NATURAL"]` | Column-ordering methods tried when building the incomplete-LU preconditioner. |
| `ilu_shift_rels` `KIRCHHOFF_ILU_SHIFT_RELS` | Advanced scientific/numerical | `[0.0,1e-14,1e-12]` | Small diagonal shifts tried if incomplete-LU factorization is unstable. |
| `solver` `KIRCHHOFF_SOLVER` | Advanced scientific/numerical | `"auto"` | Kirchhoff flow solver family. "tree" uses the fast tree-specialized solver when possible. |
| `sparse_solver` `KIRCHHOFF_SPARSE_SOLVER` | Advanced scientific/numerical | `"spsolve"` | Sparse Kirchhoff solver used when not using the tree-specialized solver. |
| `validate_sparse_solver` `KIRCHHOFF_VALIDATE_SPARSE_SOLVER` | Advanced scientific/numerical | `"spsolve"` | Sparse solver used as the reference when validating the tree-specialized solver. |
| `validate_tree` `KIRCHHOFF_VALIDATE_TREE` | Advanced scientific/numerical | `false` | If true, compare the tree-specialized solver against a sparse solver for debugging. |
| `diagnostics` `KIRCHHOFF_DIAGNOSTICS` | Performance/runtime | `true` | If true, print flow-solver timing and residual diagnostics during runs. |

</details>

<details><summary><code>settings.hematocrit</code> — 25 options</summary>

Red-cell partitioning, viscosity, and Pries-Secomb parameters.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `hd_discharge` | Essential scientific | `0.42` | Root discharge hematocrit, meaning the red-cell volume fraction entering the root segment. |
| `model` `HEMATOCRIT_MODEL` | Essential scientific | `"pries_secomb"` | Hematocrit model used for blood; "pries_secomb" includes vessel-size-dependent red-cell partitioning. |
| `bifpar_1` `PRIES_SECOMB_BIFPAR_1` | Advanced scientific/numerical | `0.964` | Pries-Secomb bifurcation-fit coefficient 1 for red-cell splitting at vessel branches. |
| `bifpar_2` `PRIES_SECOMB_BIFPAR_2` | Advanced scientific/numerical | `6.98` | Pries-Secomb bifurcation-fit coefficient 2 for red-cell splitting at vessel branches. |
| `bifpar_3` `PRIES_SECOMB_BIFPAR_3` | Advanced scientific/numerical | `-13.29` | Pries-Secomb bifurcation-fit coefficient 3 for red-cell splitting at vessel branches. |
| `cpar_1` `PRIES_SECOMB_CPAR_1` | Advanced scientific/numerical | `0.8` | Pries-Secomb cell-fraction fit coefficient 1 used by the hematocrit/viscosity model. |
| `cpar_2` `PRIES_SECOMB_CPAR_2` | Advanced scientific/numerical | `-0.075` | Pries-Secomb cell-fraction fit coefficient 2 used by the hematocrit/viscosity model. |
| `cpar_3` `PRIES_SECOMB_CPAR_3` | Advanced scientific/numerical | `-11.0` | Pries-Secomb cell-fraction fit coefficient 3 used by the hematocrit/viscosity model. |
| `cpar_4` `PRIES_SECOMB_CPAR_4` | Advanced scientific/numerical | `12.0` | Pries-Secomb cell-fraction fit coefficient 4 used by the hematocrit/viscosity model. |
| `flow_iterations` `HEMATOCRIT_FLOW_ITERATIONS` | Advanced scientific/numerical | `2` | Number of times flow and hematocrit are recomputed together to account for viscosity changes. |
| `hdtol` `HEMATOCRIT_HDTOL` | Advanced scientific/numerical | `0.001` | Hematocrit-change tolerance for stopping hematocrit-flow iterations. |
| `max` `HEMATOCRIT_MAX` | Advanced scientific/numerical | `0.95` | Highest allowed discharge hematocrit after numerical updates. |
| `mcv_fl` `PRIES_SECOMB_MCV_FL` | Advanced scientific/numerical | `55.0` | Mean cell volume in femtoliters used to rescale Pries-Secomb blood-cell geometry. |
| `min` `HEMATOCRIT_MIN` | Advanced scientific/numerical | `0.0` | Lowest allowed discharge hematocrit after numerical updates. |
| `optw_um` `PRIES_SECOMB_OPTW_UM` | Advanced scientific/numerical | `1.1` | Pries-Secomb optimal vessel-width scale in micrometers for viscosity corrections. |
| `qtol_nl_min` `HEMATOCRIT_QTOL_NL_MIN` | Advanced scientific/numerical | `0.001` | Flow-change tolerance for stopping hematocrit-flow iterations, in nanoliters per minute. |
| `relaxation` `HEMATOCRIT_RELAXATION` | Advanced scientific/numerical | `1.0` | Relaxation factor for hematocrit updates; 1.0 fully accepts each new estimate. |
| `viscpar_1` `PRIES_SECOMB_VISCPAR_1` | Advanced scientific/numerical | `6.0` | Pries-Secomb viscosity-fit coefficient 1 for blood apparent viscosity in small vessels. |
| `viscpar_2` `PRIES_SECOMB_VISCPAR_2` | Advanced scientific/numerical | `-0.085` | Pries-Secomb viscosity-fit coefficient 2 for blood apparent viscosity in small vessels. |
| `viscpar_3` `PRIES_SECOMB_VISCPAR_3` | Advanced scientific/numerical | `3.2` | Pries-Secomb viscosity-fit coefficient 3 for blood apparent viscosity in small vessels. |
| `viscpar_4` `PRIES_SECOMB_VISCPAR_4` | Advanced scientific/numerical | `-2.44` | Pries-Secomb viscosity-fit coefficient 4 for blood apparent viscosity in small vessels. |
| `viscpar_5` `PRIES_SECOMB_VISCPAR_5` | Advanced scientific/numerical | `-0.06` | Pries-Secomb viscosity-fit coefficient 5 for blood apparent viscosity in small vessels. |
| `viscpar_6` `PRIES_SECOMB_VISCPAR_6` | Advanced scientific/numerical | `0.645` | Pries-Secomb viscosity-fit coefficient 6 for blood apparent viscosity in small vessels. |
| `vplas_cp` `PRIES_SECOMB_VPLAS_CP` | Advanced scientific/numerical | `1.0466` | Plasma viscosity in centipoise used by the Pries-Secomb hematocrit model. |
| `diagnostics` `HEMATOCRIT_DIAGNOSTICS` | Performance/runtime | `true` | If true, print per-iteration hematocrit convergence information. |

</details>

<details><summary><code>settings.oxygen</code> — 34 options</summary>

Oxygen transport, consumption, lumen exchange, and quadrature.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `alpha_mmhg` | Essential scientific | `0.001408` | Dissolved oxygen solubility coefficient in blood model units per mmHg. |
| `concentration_inlet_by_fluid` | Essential scientific | `{"blood":0.14,"cell media":0.2211,"custom":0.2211,"media":0.2211,"water":0.2211}` | Inlet oxygen concentration by fluid type, in the concentration units used by the solver. |
| `finite_radius_o2_terms` | Essential scientific | `"both"` | Finite-radius oxygen correction mode for vessel sources; "both" applies both supported corrections. |
| `k_m_mm` | Essential scientific | `0.0069` | Michaelis-Menten concentration where consumption is half of VMAX_MM. |
| `lumen_diffusivity_cm2_s` `LUMEN_DIFFUSIVITY_OVERRIDE_CM2_S` | Essential scientific | `null` | Optional explicit lumen diffusivity override; when set, it wins over fluid-specific values. |
| `lumen_wall_closure` | Essential scientific | `"graetz"` | Lumen-wall closure model; "graetz" models radial lumen gradients, "wellmixed" assumes uniform lumen concentration. |
| `n_hill` | Essential scientific | `2.7` | Hill exponent controlling the steepness of the hemoglobin saturation curve. |
| `o2_cap_per_hct` | Essential scientific | `20.3` | Hemoglobin-bound oxygen capacity per unit tube hematocrit. |
| `p50_mmhg` | Essential scientific | `26.5` | Oxygen partial pressure where hemoglobin is 50 percent saturated, in mmHg. |
| `solute_diffusivity` | Essential scientific | `2.41e-05` | Oxygen diffusivity in tissue, in cm^2/s. |
| `vmax_mm` | Essential scientific | `0.04` | Michaelis-Menten maximum tissue consumption rate. |
| `conc_max` `CONC_MAX_FOR_NORMALIZATION` | Common workflow | `0.14` | Reference concentration used to normalize output metrics such as C_tiss_over_Cmax. |
| `concentration_solver` | Common workflow | `"network_ext"` | Default intravascular concentration solver used when a run does not specify one. |
| `axial_blood_steps` | Advanced scientific/numerical | `5` | Axial discretization steps used by older blood concentration approximations. |
| `blood_convective_hematocrit` | Advanced scientific/numerical | `"discharge"` | Discharge hematocrit is appropriate for convective oxygen flux. |
| `diffusivity_blood_cm2_s` `LUMEN_DIFFUSIVITY_BLOOD_CM2_S` | Advanced scientific/numerical | `2.41e-05` | Oxygen diffusivity inside blood-filled lumens, in cm^2/s. |
| `diffusivity_water_cm2_s` `LUMEN_DIFFUSIVITY_WATER_CM2_S` | Advanced scientific/numerical | `3.2e-05` | Oxygen diffusivity inside water/media-filled lumens, in cm^2/s. |
| `fp_tol` `GRAETZ_FP_TOL` | Advanced scientific/numerical | `1e-05` | Convergence tolerance for Graetz fixed-point iterations. |
| `gl_order` | Advanced scientific/numerical | `5` | Number of Gauss-Legendre quadrature points per vessel segment for tissue Greens integrals. |
| `gl_order_cext` | Advanced scientific/numerical | `1` | Number of Gauss-Legendre quadrature points per segment for explicit Cext coupling. |
| `junction_oxygen_balance` | Advanced scientific/numerical | `"total_content"` | Conserve total oxygen flux, including hemoglobin-bound oxygen. |
| `max_bi` `GRAETZ_MAX_BI` | Advanced scientific/numerical | `1000000.0` | Graetz max bi. |
| `max_fp_iters` `GRAETZ_MAX_FP_ITERS` | Advanced scientific/numerical | `4` | Maximum fixed-point iterations for solving Graetz wall/lumen coupling per segment. |
| `min_bi` `GRAETZ_MIN_BI` | Advanced scientific/numerical | `1e-08` | Graetz min bi. |
| `n_modes` `GRAETZ_N_MODES` | Advanced scientific/numerical | `4` | Fixed retained modes in the validated, precomputed Graetz basis. |
| `n_radial` `GRAETZ_N_RADIAL` | Advanced scientific/numerical | `8` | Fixed radial nodes in the validated, precomputed Graetz basis. |
| `network_transport_accel` | Advanced scientific/numerical | `"auto"` | Network transport accel. |
| `omega` | Advanced scientific/numerical | `0.7` | Relaxation factor for iterative network concentration solves. |
| `porosity` | Advanced scientific/numerical | `0.9` | Tissue porosity used by legacy transport calculations. |
| `velocity_profile` `GRAETZ_VELOCITY_PROFILE` | Advanced scientific/numerical | `"poiseuille"` | Velocity profile assumed inside the vessel lumen for Graetz calculations. |
| `bi_cache_per_decade` `GRAETZ_BI_CACHE_PER_DECADE` | Performance/runtime | `128` | Resolution and range of the validated, precomputed Graetz basis. |
| `debug_diagnostics` `GRAETZ_DEBUG_DIAGNOSTICS` | Performance/runtime | `false` | If true, print extra diagnostics for Graetz basis and fixed-point behavior. |
| `diffusivity_cm2_s` `LUMEN_DIFFUSIVITY_CM2_S` | Compatibility/internal | `2.41e-05` | Effective lumen diffusivity currently used by legacy code paths. |
| `extravascular_concentration` | Compatibility/internal | `0.0` | Deprecated legacy background tissue concentration; current Cext paths compute local extravascular concentration. |

</details>

<details><summary><code>settings.tissue</code> — 38 options</summary>

Tissue sampling, neighborhood accuracy, and tissue execution.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `accel_mode` `TISSUE_ACCEL_MODE` | Common workflow | `"auto"` | Tissue oxygen backend selection: "gpu", "cpu", or "auto". |
| `compute_avg_distance_to_channel` | Common workflow | `false` | If true, compute average distance from sample points to the nearest vessel centerline. |
| `distance_sample_count` | Common workflow | `1000000` | Default number of tissue sample points for random tissue oxygen evaluation. |
| `brute_force_max_segments` `DISTANCE_BRUTE_FORCE_MAX_SEGMENTS` | Advanced scientific/numerical | `20000` | Largest vessel count where brute-force distance-to-vessel calculations are still allowed. |
| `cext_cell_list_enable` `TISSUE_CEXT_CELL_LIST_ENABLE` | Advanced scientific/numerical | `true` | If true, use a cell-list spatial grid to find Cext source vessels near tissue points on GPU. |
| `cext_cell_max_grid` `TISSUE_CEXT_CELL_MAX_GRID` | Advanced scientific/numerical | `256` | Maximum grid resolution per dimension for the GPU Cext tissue cell list. |
| `cext_cell_max_rad_cells` `TISSUE_CEXT_CELL_MAX_RAD_CELLS` | Advanced scientific/numerical | `8` | Maximum number of neighboring cell layers searched around each tissue point. |
| `cext_cell_min_grid` `TISSUE_CEXT_CELL_MIN_GRID` | Advanced scientific/numerical | `16` | Minimum grid resolution per dimension for the GPU Cext tissue cell list. |
| `cext_cell_target_occupancy` `TISSUE_CEXT_CELL_TARGET_OCCUPANCY` | Advanced scientific/numerical | `4` | Desired average number of vessel sources per spatial cell in the GPU Cext tissue cell list. |
| `cext_tissue_quadrature_mode` | Advanced scientific/numerical | `"independent"` | Tissue quadrature after a Cext solve: independent resamples to GL_ORDER; legacy_cext reuses the Cext source nodes for oracle compatibility only. |
| `distance_kdtree_candidate_mult` | Advanced scientific/numerical | `8` | Candidate multiplier for KD-tree searches; larger values reduce missed nearest vessels but cost more time. |
| `kdtree_candidate_mult` `TISSUE_KDTREE_CANDIDATE_MULT` | Advanced scientific/numerical | `2` | Candidate multiplier used while building tissue nearest-vessel caches. |
| `kdtree_max_candidates` `DISTANCE_KDTREE_MAX_CANDIDATES` | Advanced scientific/numerical | `4096` | Hard cap on candidate vessels per tissue point in KD-tree distance searches. |
| `kdtree_min_candidates` `DISTANCE_KDTREE_MIN_CANDIDATES` | Advanced scientific/numerical | `64` | Minimum number of candidate vessels considered per tissue point in KD-tree distance searches. |
| `min_segment_length_si` `TISSUE_MIN_SEGMENT_LENGTH_SI` | Advanced scientific/numerical | `1e-12` | Minimum segment length in SI units; shorter segments are treated as degenerate for tissue kernels. |
| `nearest_tissue_vessels` | Advanced scientific/numerical | `250` | Maximum number of nearby vessels kept per tissue point for tissue oxygen calculations. |
| `use_kdtree_for_large_trees` `DISTANCE_USE_KDTREE_FOR_LARGE_TREES` | Advanced scientific/numerical | `true` | If true, use a KD-tree nearest-neighbor search instead of a dense all-vessel distance matrix for large trees. |
| `window_factor` | Advanced scientific/numerical | `6` | Cutoff radius measured in oxygen decay lengths; farther vessels are ignored for tissue oxygen. |
| `cache_chunk_size` `TISSUE_CACHE_CHUNK_SIZE` | Performance/runtime | `512` | Number of tissue points per chunk when building CPU tissue geometry caches. |
| `cache_float_dtype` `TISSUE_CACHE_FLOAT_DTYPE` | Performance/runtime | `float32` | Floating-point dtype used inside tissue cache arrays. |
| `cache_force_float64` `TISSUE_CACHE_FORCE_FLOAT64` | Performance/runtime | `false` | If true, force tissue caches to store floats as float64 instead of the memory-saving default. |
| `cache_index_dtype` `TISSUE_CACHE_INDEX_DTYPE` | Performance/runtime | `int32` | Integer dtype used inside tissue cache index arrays. |
| `distance_chunk_size` | Performance/runtime | `256` | Number of tissue points processed together in some CPU distance/oxygen kernels. |
| `gpu_chunk_points` `TISSUE_GPU_CHUNK_POINTS` | Performance/runtime | `8192` | Number of tissue points processed per GPU kernel launch. |
| `gpu_min_chunk_points` `TISSUE_GPU_MIN_CHUNK_POINTS` | Performance/runtime | `512` | Lower bound for GPU chunk size when automatic chunk sizing is used. |
| `gpu_validate_points` `TISSUE_GPU_VALIDATE_POINTS` | Performance/runtime | `0` | Number of tissue points to cross-check against CPU results for GPU validation; 0 disables validation. |
| `kdtree_workers` `TISSUE_KDTREE_WORKERS` | Performance/runtime | `max(1, CPU count - 2)` | Worker count used for KD-tree queries while building tissue caches. |
| `parallel_workers` `TISSUE_PARALLEL_WORKERS` | Performance/runtime | `max(1, CPU count)` | Worker count for CPU tissue calculations that can run in parallel. |
| `streaming_chunk_workers` `TISSUE_STREAMING_CHUNK_WORKERS` | Performance/runtime | `max(1, CPU count - 2)` | Worker count for streaming tissue chunks. |
| `streaming_enabled` `TISSUE_STREAMING_ENABLED` | Performance/runtime | `false` | If true, process very large tissue point sets in streaming chunks to reduce peak memory. |
| `streaming_log_every_chunks` `TISSUE_STREAMING_LOG_EVERY_CHUNKS` | Performance/runtime | `25` | Print streaming progress every this many chunks. |
| `streaming_max_chunk_points` `TISSUE_STREAMING_MAX_CHUNK_POINTS` | Performance/runtime | `2048` | Largest allowed point count for a streaming chunk. |
| `streaming_min_chunk_points` `TISSUE_STREAMING_MIN_CHUNK_POINTS` | Performance/runtime | `256` | Smallest allowed point count for a streaming chunk. |
| `streaming_min_points` `TISSUE_STREAMING_MIN_POINTS` | Performance/runtime | `100000` | Minimum tissue point count before streaming mode is considered. |
| `streaming_numba_threads_per_worker` `TISSUE_STREAMING_NUMBA_THREADS_PER_WORKER` | Performance/runtime | `1` | Numba thread count assigned inside each streaming worker process. |
| `streaming_prune_by_window` `TISSUE_STREAMING_PRUNE_BY_WINDOW` | Performance/runtime | `true` | If true, remove vessels outside the local oxygen window during streaming to reduce work. |
| `streaming_target_candidate_slots` `TISSUE_STREAMING_TARGET_CANDIDATE_SLOTS` | Performance/runtime | `500000` | Target number of point-vessel candidate slots per streaming chunk; controls memory per chunk. |
| `use_numba` `TISSUE_USE_NUMBA` | Performance/runtime | `true` | If true, use numba-compiled tissue kernels when available. |

</details>

<details><summary><code>settings.cext</code> — 96 options</summary>

Extravascular concentration coupling and its acceleration algorithms.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `accel_mode` `CEXT_ACCEL_MODE` | Common workflow | `"auto"` | Device used for the main explicit extravascular concentration solve: "gpu", "cpu", or "auto". |
| `vess_coupling_max_iter` `CEXT_VESS_COUPLING_MAX_ITER` | Common workflow | `1` | Maximum outer iterations coupling intravascular oxygen to extravascular concentration. |
| `vess_coupling_tol` `CEXT_VESS_COUPLING_TOL` | Common workflow | `0.0001` | Absolute convergence tolerance in mol/m^3 for the selected residual norm. |
| `active_set_abs_tol` `CEXT_ACTIVE_SET_ABS_TOL` | Advanced scientific/numerical | `0.00025` | Absolute change threshold used to decide that a source vessel is stable. |
| `active_set_enable` `CEXT_ACTIVE_SET_ENABLE` | Advanced scientific/numerical | `true` | If true, freeze source vessels whose Cext contribution has converged enough. |
| `active_set_min_active_count` `CEXT_ACTIVE_SET_MIN_ACTIVE_COUNT` | Advanced scientific/numerical | `1024` | Minimum number of still-active source vessels required before freezing is allowed. |
| `active_set_min_active_fraction` `CEXT_ACTIVE_SET_MIN_ACTIVE_FRACTION` | Advanced scientific/numerical | `0.0` | Minimum active-source fraction required before freezing is allowed; 0 means use only the count limit. |
| `active_set_refresh_period` `CEXT_ACTIVE_SET_REFRESH_PERIOD` | Advanced scientific/numerical | `8` | Iteration period for refreshing frozen/active source-vessel status. |
| `active_set_rel_tol` `CEXT_ACTIVE_SET_REL_TOL` | Advanced scientific/numerical | `0.005` | Relative change threshold used to decide that a source vessel is stable. |
| `active_set_stable_iters` `CEXT_ACTIVE_SET_STABLE_ITERS` | Advanced scientific/numerical | `3` | Consecutive stable iterations required before a source vessel can be frozen. |
| `active_set_start` `CEXT_ACTIVE_SET_START` | Advanced scientific/numerical | `6` | First outer iteration where source-vessel active-set freezing is allowed. |
| `approx_window_scale` `CEXT_APPROX_WINDOW_SCALE` | Advanced scientific/numerical | `1.0` | Scale factor applied to approximate-window distances for Cext candidate pruning. |
| `grid_cell_factor` `CEXT_GRID_CELL_FACTOR` | Advanced scientific/numerical | `1.0` | Multiplier connecting vessel interaction length scales to hybrid/grid cell sizes. |
| `hybrid_bg_assignment` `CEXT_HYBRID_BG_ASSIGNMENT` | Advanced scientific/numerical | `"tsc"` | Source-to-grid assignment stencil for the hybrid background field, such as CIC or TSC. |
| `hybrid_bg_enable_source_freezing` `CEXT_HYBRID_BG_ENABLE_SOURCE_FREEZING` | Advanced scientific/numerical | `false` | If true, allow background-source freezing during hybrid background iterations. |
| `hybrid_bg_grid` `CEXT_HYBRID_BG_GRID` | Advanced scientific/numerical | `256` | Grid resolution per dimension for the hybrid background-field approximation. |
| `hybrid_bg_lambda_bins` `CEXT_HYBRID_BG_LAMBDA_BINS` | Advanced scientific/numerical | `5` | Number of lambda bins used when grouping source vessels for the hybrid background solve. |
| `hybrid_bg_mode` `CEXT_HYBRID_BG_MODE` | Advanced scientific/numerical | `"fft"` | Hybrid background mode; "fft" uses grid convolution, while local-only modes use direct local corrections. |
| `hybrid_bg_near_radius_mult` `CEXT_HYBRID_BG_NEAR_RADIUS_MULT` | Advanced scientific/numerical | `0.0` | Radius multiplier separating near-field direct corrections from background-field approximation. |
| `hybrid_bg_solver` `CEXT_HYBRID_BG_SOLVER` | Advanced scientific/numerical | `"auto"` | Solver used for the hybrid background field: "auto", "fft", or "jacobi". |
| `hybrid_bg_vcycles` `CEXT_HYBRID_BG_VCYCLES` | Advanced scientific/numerical | `2` | Number of multigrid V-cycles used by background solvers that support V-cycles. |
| `hybrid_fft_o2_correction` `CEXT_HYBRID_FFT_O2_CORRECTION` | Advanced scientific/numerical | `true` | If true, include finite-radius oxygen source corrections in the FFT background path. |
| `hybrid_fft_o2_fused_ifft` `CEXT_HYBRID_FFT_O2_FUSED_IFFT` | Advanced scientific/numerical | `true` | If true, fuse some oxygen-correction work into inverse FFT operations. |
| `hybrid_fft_quantile_bins` `CEXT_HYBRID_FFT_QUANTILE_BINS` | Advanced scientific/numerical | `true` | If true, choose lambda-bin edges from quantiles of the current vessel lambda values. |
| `hybrid_fft_self_sub_fused` `CEXT_HYBRID_FFT_SELF_SUB_FUSED` | Advanced scientific/numerical | `false` | If true, fuse self-subtraction operations where supported by the FFT implementation. |
| `hybrid_fft_self_sub_scale` `CEXT_HYBRID_FFT_SELF_SUB_SCALE` | Advanced scientific/numerical | `1.0` | Multiplicative scale for the FFT self-subtraction correction. |
| `hybrid_fft_self_sub_target_sampling` `CEXT_HYBRID_FFT_SELF_SUB_TARGET_SAMPLING` | Advanced scientific/numerical | `"cic"` | Grid interpolation method used when sampling the self-subtraction correction at target vessels. |
| `hybrid_fft_self_subtract` `CEXT_HYBRID_FFT_SELF_SUBTRACT` | Advanced scientific/numerical | `true` | If true, subtract each vessel's own gridded background contribution before adding direct self terms. |
| `init_mode` `CEXT_INIT_MODE` | Advanced scientific/numerical | `"decoupled_greens"` | Initial guess for Cext before vessel/extravascular coupling iterations begin. |
| `lambda_source` `CEXT_LAMBDA_SOURCE` | Advanced scientific/numerical | `"lambda_t"` | Source of the screening length lambda used by Cext kernels, usually tissue lambda from local uptake. |
| `local_exclude_hops` `CEXT_LOCAL_EXCLUDE_HOPS` | Advanced scientific/numerical | `2` | Number of graph hops excluded from local Cext coupling to avoid near-self double counting. |
| `max_candidates_per_target` `CEXT_MAX_CANDIDATES_PER_TARGET` | Advanced scientific/numerical | `100` | Maximum candidate source vessels retained per target in approximate Cext searches. |
| `max_cells_per_seg` `CEXT_MAX_CELLS_PER_SEG` | Advanced scientific/numerical | `128` | Maximum number of grid cells a single segment may touch during Cext spatial indexing. |
| `tail_core_abs_threshold` `CEXT_TAIL_CORE_ABS_THRESHOLD` | Advanced scientific/numerical | `0.00025` | Absolute residual threshold defining the active core for tail correction. |
| `tail_core_rel_threshold` `CEXT_TAIL_CORE_REL_THRESHOLD` | Advanced scientific/numerical | `0.005` | Relative residual threshold defining the active core for tail correction. |
| `tail_gmres_maxiter` `CEXT_TAIL_GMRES_MAXITER` | Advanced scientific/numerical | `96` | Maximum GMRES iterations inside the tail solver. |
| `tail_gmres_restart` `CEXT_TAIL_GMRES_RESTART` | Advanced scientific/numerical | `32` | Krylov restart length for GMRES inside the tail solver. |
| `tail_max_nonlinear_iters` `CEXT_TAIL_MAX_NONLINEAR_ITERS` | Advanced scientific/numerical | `6` | Maximum nonlinear iterations inside the tail correction solve. |
| `tail_rebound_arm_rel` `CEXT_TAIL_REBOUND_ARM_REL` | Advanced scientific/numerical | `0.4` | Armijo-style relative rebound limit used in guarded tail-solver step selection. |
| `tail_rebound_rel` `CEXT_TAIL_REBOUND_REL` | Advanced scientific/numerical | `0.9` | Relative rebound limit used to reject tail updates that worsen the residual too much. |
| `tail_solver` `CEXT_TAIL_SOLVER` | Advanced scientific/numerical | `"active_core_nk"` | Tail solver used when active-set iterations stall; "none" disables the tail correction. |
| `tail_trigger_active_count` `CEXT_TAIL_TRIGGER_ACTIVE_COUNT` | Advanced scientific/numerical | `4096` | Minimum active unknown count before the tail solver is worth using. |
| `tail_trigger_improvement_ratio` `CEXT_TAIL_TRIGGER_IMPROVEMENT_RATIO` | Advanced scientific/numerical | `0.8` | Required improvement ratio for accepting tail-solver progress. |
| `tail_trigger_stall_iters` `CEXT_TAIL_TRIGGER_STALL_ITERS` | Advanced scientific/numerical | `6` | Number of stalled iterations required before the tail solver is considered. |
| `tail_trigger_start_iter` `CEXT_TAIL_TRIGGER_START_ITER` | Advanced scientific/numerical | `8` | Earliest outer iteration where the tail solver can be triggered. |
| `target_active_set_abs_tol` `CEXT_TARGET_ACTIVE_SET_ABS_TOL` | Advanced scientific/numerical | `0.00025` | Absolute change threshold used to decide that a target vessel is stable. |
| `target_active_set_enable` `CEXT_TARGET_ACTIVE_SET_ENABLE` | Advanced scientific/numerical | `true` | If true, freeze target vessels whose received Cext field has converged enough. |
| `target_active_set_min_active_count` `CEXT_TARGET_ACTIVE_SET_MIN_ACTIVE_COUNT` | Advanced scientific/numerical | `1024` | Minimum number of still-active target vessels required before target freezing is allowed. |
| `target_active_set_neighbor_pad` `CEXT_TARGET_ACTIVE_SET_NEIGHBOR_PAD` | Advanced scientific/numerical | `1` | Extra neighbor layers kept active around active targets so local interactions remain accurate. |
| `target_active_set_rel_tol` `CEXT_TARGET_ACTIVE_SET_REL_TOL` | Advanced scientific/numerical | `0.005` | Relative change threshold used to decide that a target vessel is stable. |
| `target_active_set_stable_iters` `CEXT_TARGET_ACTIVE_SET_STABLE_ITERS` | Advanced scientific/numerical | `3` | Consecutive stable iterations required before a target vessel can be frozen. |
| `target_active_set_start` `CEXT_TARGET_ACTIVE_SET_START` | Advanced scientific/numerical | `6` | First outer iteration where target-vessel freezing is allowed. |
| `treecode_freeze_qrel_tol` `CEXT_TREECODE_FREEZE_QREL_TOL` | Advanced scientific/numerical | `0.025` | Relative flow-change tolerance used when deciding whether treecode sources can remain frozen. |
| `treecode_lambda_bins` `CEXT_TREECODE_LAMBDA_BINS` | Advanced scientific/numerical | `8` | Number of lambda bins used by treecode approximations. |
| `treecode_leaf_nodes` `CEXT_TREECODE_LEAF_NODES` | Advanced scientific/numerical | `128` | Maximum source count per leaf node in the treecode spatial hierarchy. |
| `treecode_near_radius_mult` `CEXT_TREECODE_NEAR_RADIUS_MULT` | Advanced scientific/numerical | `4.0` | Radius multiplier defining direct near-field work around treecode targets. |
| `treecode_order` `CEXT_TREECODE_ORDER` | Advanced scientific/numerical | `1` | Multipole expansion order used by treecode Cext approximations. |
| `treecode_theta` `CEXT_TREECODE_THETA` | Advanced scientific/numerical | `0.5` | Treecode opening angle; smaller values are more accurate and slower. |
| `vess_conc_floor` | Advanced scientific/numerical | `1e-12` | Lower bound for vessel concentration used to avoid zero or negative concentration in Cext formulas. |
| `vess_coupling_accel` `CEXT_VESS_COUPLING_ACCEL` | Advanced scientific/numerical | `"anderson"` | Iteration acceleration method: "none", "aitken", or "anderson". |
| `vess_coupling_accel_accept_factor` `CEXT_VESS_COUPLING_ACCEL_ACCEPT_FACTOR` | Advanced scientific/numerical | `0.9` | Required improvement factor for accepting any accelerated step. |
| `vess_coupling_accel_restart_factor` `CEXT_VESS_COUPLING_ACCEL_RESTART_FACTOR` | Advanced scientific/numerical | `1.1` | Residual-growth factor that triggers acceleration history restart. |
| `vess_coupling_accel_step_factor` `CEXT_VESS_COUPLING_ACCEL_STEP_FACTOR` | Advanced scientific/numerical | `2.0` | Scale applied to candidate accelerated step sizes. |
| `vess_coupling_anderson_depth` `CEXT_VESS_COUPLING_ANDERSON_DEPTH` | Advanced scientific/numerical | `4` | Number of previous residual vectors Anderson acceleration can use. |
| `vess_coupling_anderson_gate_ratio` `CEXT_VESS_COUPLING_ANDERSON_GATE_RATIO` | Advanced scientific/numerical | `0.9` | Residual-improvement gate required before accepting Anderson acceleration. |
| `vess_coupling_anderson_min_stable_iters` `CEXT_VESS_COUPLING_ANDERSON_MIN_STABLE_ITERS` | Advanced scientific/numerical | `2` | Number of stable iterations required before Anderson acceleration is trusted. |
| `vess_coupling_anderson_omega_gate_factor` `CEXT_VESS_COUPLING_ANDERSON_OMEGA_GATE_FACTOR` | Advanced scientific/numerical | `1.5` | Additional gate on Anderson steps based on the current relaxation factor. |
| `vess_coupling_anderson_reg` `CEXT_VESS_COUPLING_ANDERSON_REG` | Advanced scientific/numerical | `1e-10` | Small regularization added to Anderson least-squares problems for numerical stability. |
| `vess_coupling_anderson_start` `CEXT_VESS_COUPLING_ANDERSON_START` | Advanced scientific/numerical | `2` | First outer iteration where Anderson acceleration is allowed to start. |
| `vess_coupling_best_revert_factor` `CEXT_VESS_COUPLING_BEST_REVERT_FACTOR` | Advanced scientific/numerical | `1.02` | Residual increase over the best residual that can trigger reverting toward the best state. |
| `vess_coupling_best_stall_iters` `CEXT_VESS_COUPLING_BEST_STALL_ITERS` | Advanced scientific/numerical | `20` | Iterations allowed without improving the best residual before best-state recovery logic can act. |
| `vess_coupling_norm` `CEXT_VESS_COUPLING_NORM` | Advanced scientific/numerical | `"rms"` | Norm of the unrelaxed coupling equation residual, also used for step acceptance. |
| `vess_coupling_omega` `CEXT_VESS_COUPLING_OMEGA` | Advanced scientific/numerical | `1.0` | Base relaxation factor for vessel-Cext coupling updates before acceleration modifies it. |
| `vess_coupling_omega_max` `CEXT_VESS_COUPLING_OMEGA_MAX` | Advanced scientific/numerical | `1.4` | Upper bound on relaxation after acceleration; prevents overly aggressive updates. |
| `vess_coupling_omega_min` `CEXT_VESS_COUPLING_OMEGA_MIN` | Advanced scientific/numerical | `0.025` | Lower bound on relaxation after acceleration; prevents updates from becoming too tiny. |
| `vess_coupling_rel_tol` `CEXT_VESS_COUPLING_REL_TOL` | Advanced scientific/numerical | `0.0` | Relative convergence tolerance for vessel-Cext coupling; 0 disables this relative stop test. |
| `vess_coupling_step_reject_factor` `CEXT_VESS_COUPLING_STEP_REJECT_FACTOR` | Advanced scientific/numerical | `1.02` | Residual-growth factor that causes an accelerated step to be rejected. |
| `vess_coupling_step_retry_factor` `CEXT_VESS_COUPLING_STEP_RETRY_FACTOR` | Advanced scientific/numerical | `0.5` | Factor used to shrink and retry a rejected coupling step. |
| `vess_coupling_trust_abs` `CEXT_VESS_COUPLING_TRUST_ABS` | Advanced scientific/numerical | `0.001` | Absolute trust limit for accepting an accelerated coupling step. |
| `vess_coupling_trust_rel` `CEXT_VESS_COUPLING_TRUST_REL` | Advanced scientific/numerical | `0.4` | Relative trust limit for accepting an accelerated coupling step. |
| `window_factor` `CEXT_WINDOW_FACTOR` | Advanced scientific/numerical | `6` | Spatial interaction cutoff measured in oxygen decay lengths for local Cext calculations. |
| `float_dtype` `CEXT_FLOAT_DTYPE` | Performance/runtime | `float32` | Floating-point dtype used in Cext working arrays. |
| `frozen_accel_mode` `CEXT_FROZEN_ACCEL_MODE` | Performance/runtime | `"auto"` | Device used for frozen-source Cext steps, which reuse fixed vessel source strengths. |
| `gpu_validate_segments` `CEXT_GPU_VALIDATE_SEGMENTS` | Performance/runtime | `0` | Number of vessel segments to validate against a reference GPU path; 0 disables validation. |
| `hybrid_fft_bin_epoch_cache` `CEXT_HYBRID_FFT_BIN_EPOCH_CACHE` | Performance/runtime | `true` | If true, cache FFT bin data across coupling iterations when bin definitions do not change. |
| `hybrid_fft_o2_moment_batch` `CEXT_HYBRID_FFT_O2_MOMENT_BATCH` | Performance/runtime | `1` | Batch size for FFT oxygen moment calculations; larger batches can improve throughput but use more memory. |
| `hybrid_fft_response_batched` `CEXT_HYBRID_FFT_RESPONSE_BATCHED` | Performance/runtime | `false` | If true, batch FFT response calculations to reduce Python overhead at the cost of more temporary memory. |
| `hybrid_gpu_iteration_cache` `CEXT_HYBRID_GPU_ITERATION_CACHE` | Performance/runtime | `true` | Build hybrid FFT source weights on GPU, refreshing the current field and final tissue source handoff. |
| `hybrid_gpu_runtime_moments` `CEXT_HYBRID_GPU_RUNTIME_MOMENTS` | Performance/runtime | `true` | If true, compute FFT O2 moment weights inside runtime-stencil kernels instead of caching moment arrays. |
| `hybrid_gpu_runtime_stencil` `CEXT_HYBRID_GPU_RUNTIME_STENCIL` | Performance/runtime | `true` | If true, compute grid assignment stencils inside CUDA kernels instead of storing large stencil arrays. |
| `hybrid_gpu_runtime_weights` `CEXT_HYBRID_GPU_RUNTIME_WEIGHTS` | Performance/runtime | `true` | If true, update hybrid GPU work weights from measured runtime instead of static estimates. |
| `index_dtype` `CEXT_INDEX_DTYPE` | Performance/runtime | `int32` | Integer dtype used in Cext index arrays. |
| `local_only_fast_self` `CEXT_LOCAL_ONLY_FAST_SELF` | Performance/runtime | `true` | If true, use a faster self-interaction path for local-only Cext modes. |
| `precompute_workers` `CEXT_PRECOMPUTE_WORKERS` | Performance/runtime | `max(1, CPU count - 2)` | Worker count for CPU precomputation before Cext GPU or hybrid solves. |
| `streaming_target_candidate_slots` `CEXT_STREAMING_TARGET_CANDIDATE_SLOTS` | Performance/runtime | `500000` | Target number of source-target candidate slots when streaming Cext calculations. |
| `treecode_gpu_local` `CEXT_TREECODE_GPU_LOCAL` | Performance/runtime | `true` | If true, compute treecode near-field/local corrections on the GPU when available. |

</details>

<details><summary><code>settings.numerics</code> — 9 options</summary>

Data types, compilation, timing, and tree-cache behavior.

| JSON key | Importance | Default | Meaning |
| --- | --- | --- | --- |
| `cache_dirname` `TREE_CACHE_DIRNAME` | Performance/runtime | `"trees_cache"` | Directory name used by tree-cache helpers. |
| `cache_index_name` `TREE_CACHE_INDEX_NAME` | Performance/runtime | `"tree_index.csv"` | CSV index filename used by tree-cache helpers. |
| `conc_use_numba` | Performance/runtime | `true` | If true, use numba-compiled concentration kernels when numba is installed. |
| `save_trees` | Performance/runtime | `false` | If true, persist newly grown trees after construction. |
| `solver_timing_details` | Performance/runtime | `true` | If true, collect and print detailed timing fields for flow, concentration, Cext, and tissue steps. |
| `tree_data_dtype` | Performance/runtime | `float64` | Floating-point dtype used for tree geometry and flow arrays in public-svv-compatible runs. |
| `tree_fast_cache_enable` | Performance/runtime | `false` | If true, prefer a lightweight fast tree cache when loading saved trees. |
| `tree_index_dtype` | Performance/runtime | `int64` | Integer dtype used for tree connectivity and segment-index arrays. |
| `use_tree_cache` | Performance/runtime | `true` | If true, allow reusable tree-cache lookup during network construction. |

</details>

Compatibility views:

- `settings.concentration` accepts the basic concentration, diffusivity,
  consumption, relaxation, and quadrature keys from `settings.oxygen`.
- `settings.hemodynamics` also accepts the `settings.kirchhoff` keys.
- `settings.runtime` accepts any registered exact setting name. Prefer the
  named sections above in new files.

<!-- END GENERATED RUNTIME SETTINGS -->

## Choosing settings safely

1. Define the physical geometry, flow/boundary conditions, fluid, oxygen
   transport, and consumption model first.
2. Select sampling and outputs needed to answer the study question.
3. Change advanced numerical settings only with convergence or validation
   evidence.
4. Tune GPU batches, chunks, worker counts, caches, and data types last. Record
   those settings for reproducibility, but do not treat them as physical model
   parameters.

Run `cascade init-settings case.json` for a small starter file. Unknown keys in
the core sections are rejected to catch misspellings. See the [CLI guide](cli.md)
for commands, [Studio guide](gui.md) for GUI workflow, and
[examples](examples.md) for complete configurations.
