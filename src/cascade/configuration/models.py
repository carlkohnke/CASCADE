"""Typed configuration value objects for a CASCADE run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DomainConfig:
    kind: str = "cube"
    side_length: float = 1.0
    x_length: float | None = None
    y_length: float | None = None
    z_length: float | None = None
    radius: float | None = None
    height: float | None = None
    center: list[float] | None = None
    theta_resolution: int = 12
    phi_resolution: int = 8
    path: str | None = None
    mesh: Any | None = None
    random_seed: int = 42
    use_cache: bool = True
    cache_dir: str | None = None


@dataclass
class RootConfig:
    start: list[float]
    direction: list[float] | None = None


@dataclass
class NetworkConfig:
    mode: str = "tree"
    input_path: str | None = None
    target_terminal_count: int | None = 100
    target_total_terminal_count: int | None = None
    target_terminal_counts: list[int] = field(default_factory=list)
    roots: list[RootConfig] = field(default_factory=list)
    physical_clearance: float = 0.0
    save_path: str | None = None
    repair_connectivity: bool = True
    validate_connectivity: bool = False
    fail_connectivity: bool = True
    connectivity_geometry_atol: float = 1.0e-6
    simple: dict[str, Any] = field(default_factory=dict)


@dataclass
class GrowthConfig:
    enabled: bool = True
    assignment: str = "bulk"
    bulk_growth_mode: str = "always"
    n_closest_vessels: int = 2
    n_points: int = 50
    weighted_sampling: bool = False
    ignore_collisions: bool = True
    allow_inside_vessels: bool = True
    n_ignore_collisions: int | None = None
    collision_retry_limit: int = 100
    collision_failure_mode: str = "error"
    nearest_tree_batch_points: int = 256
    strict_domain_segments: bool = False
    strict_domain_max_terminals: int = 10000
    domain_line_samples: int = 4
    domain_line_tolerance: float = 0.0
    growth_report_every: int = 0
    add_per_tree: list[int] = field(default_factory=list)
    add_total: int | None = None
    add_split_mode: str = "equal"
    checkpoint_path: str | None = None
    checkpoint_every_adds: int = 0
    resume_from_checkpoint: bool = False
    save_target_counts: list[int] = field(default_factory=list)
    n_equal_bifurcations: int | None = None
    equal_terminal: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalFieldConfig:
    """Execution policy for vessel-to-vessel external-field coupling."""

    enabled: bool = False
    scope: str = "per-network"
    mode: str = "shared-global"


@dataclass(frozen=True)
class OcclusionConfig:
    """A reversible solve-time vascular occlusion."""

    global_segment_id: int
    fraction_blocked: float
    include_downstream_when_complete: bool = True


@dataclass
class SimulationConfig:
    fluid: str = "blood"
    build_fluid: str = "blood"
    qin_target_ul_min: float = 100.0
    total_qin_ul_min: float | None = None
    concentration_solver: str = "network_ext"
    distance_sample_count: int = 10000
    flow_source: str = "per_tree"
    inlet_conditions: list[dict[str, float]] = field(default_factory=list)
    sample_mode: str = "random"
    sample_points_path: str | None = None
    tissue_grid: dict[str, Any] = field(default_factory=dict)
    geometry_only: bool = False
    skip_tissue_oxygen: bool = False
    compute_avg_distance_to_channel: bool = False
    tissue_accel: str | None = "gpu"
    tissue_gpu_validate_points: int | None = None
    viability_threshold: float | None = None
    occlusion: OcclusionConfig | None = None
    external_field: ExternalFieldConfig = field(default_factory=ExternalFieldConfig)
    cext: dict[str, Any] = field(default_factory=dict)
    tissuesim: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputsConfig:
    out_dir: str = "cascade_run"
    prefix: str | None = None
    write_paraview: bool = True
    write_vessels_vtp: bool = True
    write_tissue_vtp: bool = True
    write_summary_csv: bool = True
    write_segments_csv: bool = True
    write_points_csv: bool = True
    include_tissue_nearest_fields: bool = False
    save_network: bool = True
    use_cache: bool = True
    cache_path: str | None = None
    export_float_dtype: str = "float64"
    export_index_dtype: str = "int64"
    vessel_resolution: int = 2
    write_combined_sweep_csv: bool = True
    combined_sweep_filename: str = "sweep_summary.csv"
    overwrite: bool = False


@dataclass
class RunConfig:
    domain: DomainConfig
    network: NetworkConfig
    growth: GrowthConfig
    simulation: SimulationConfig
    outputs: OutputsConfig
    runtime_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_setting_overrides: dict[str, Any] = field(default_factory=dict)
    runtime_setting_warnings: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    settings_path: Path | None = None

    @property
    def network_mode(self) -> str:
        return self.network.mode.strip().lower()

    @property
    def prefix(self) -> str:
        if self.outputs.prefix:
            return self.outputs.prefix
        return f"cascade_{self.network_mode}"
