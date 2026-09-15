"""Create, validate, estimate, and persist the GUI's project configuration model.

Widget-independent helpers translate project JSON into validated run settings,
resource estimates, sweep jobs, and preview descriptions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any, Iterable
from uuid import uuid4

from cascade.configuration.schema import example_config, parse_config
from cascade.utils.files import atomic_write_text
from cascade.vessels.lattice import channel_count, resolve_lattice_layout
from cascade.utils.resources import resolve_domain_path

from . import GUI_SCHEMA_VERSION


MMHG_TO_PA = 133.322387415
ALPHA_MMHG = 0.001408
QUEUE_FILENAME = "queue.json"
PROJECT_FILENAME = ".cascade-project.json"


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.errors


@dataclass
class HardwareInfo:
    total_ram_bytes: int
    available_ram_bytes: int
    cpu_count: int
    gpu_name: str = "Not detected"
    gpu_memory_bytes: int = 0
    platform_name: str = field(default_factory=platform.platform)


@dataclass
class ResourceEstimate:
    segment_count: int
    tissue_point_count: int
    host_memory_bytes: int
    gpu_memory_bytes: int
    level: str
    message: str


@dataclass
class JobRecord:
    id: str
    name: str
    settings_path: str
    output_dir: str
    status: str = "Queued"
    progress: int = 0
    stage: str = "Waiting"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None
    manifest_path: str | None = None
    error: str | None = None
    sweep_batch_id: str | None = None
    combined_csv_path: str | None = None
    sweep_parameters: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobRecord":
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in raw.items() if key in names})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_project() -> dict[str, Any]:
    config = example_config()
    config["settings"].pop("kirchhoff", None)
    config["schema_version"] = GUI_SCHEMA_VERSION
    config["domain"] = {
        "type": "box",
        "side_length": 1.0,
        "x_length": 1.0,
        "y_length": 1.0,
        "z_length": 1.0,
        "random_seed": 42,
    }
    config["network"]["target_terminal_count"] = 100
    config["simulation"].update(
        {
            "qin_target_ul_min": 100.0,
            "concentration_solver": "network_ext",
            "distance_sample_count": 10000,
            "tissue_grid": {"nx": 20, "ny": 20, "nz": 20},
            "viability_threshold": ALPHA_MMHG,
        }
    )
    config["settings"].update(
        {
            "hemodynamics": {
                "root_pressure": 66661.0,
                "terminal_pressure": 40000.0,
                "scale_q_by_volume": False,
                "scale_dp_by_volume": False,
            },
            "hematocrit": {
                "model": "pries_secomb",
                "flow_iterations": 2,
                "hd_discharge": 0.42,
            },
            "oxygen": {
                "concentration_inlet_by_fluid": {
                    "water": 100.0 * ALPHA_MMHG,
                    "blood": 100.0 * ALPHA_MMHG,
                    "cell media": 100.0 * ALPHA_MMHG,
                    "media": 100.0 * ALPHA_MMHG,
                },
                "conc_max_for_normalization": 100.0 * ALPHA_MMHG,
                "solute_diffusivity": 2.41e-5,
                "vmax_mm": 0.001,
                "k_m_mm": 0.005,
                "finite_radius_o2_terms": "both",
                "lumen_wall_closure": "graetz",
                "axial_blood_steps": 5,
                "gl_order": 5,
                "gl_order_cext": 1,
            },
            "cext": {
                "accel_mode": "gpu",
                "vess_coupling_max_iter": 1,
                "vess_coupling_tol": 1e-3,
                "window_factor": 6,
                "hybrid_bg_grid": 256,
                "hybrid_bg_lambda_bins": 5,
                "hybrid_bg_mode": "fft",
                "float_dtype": "float32",
                "index_dtype": "int32",
            },
            "tissue": {
                "accel_mode": "gpu",
                "nearest_vessels": 250,
                "window_factor": 6,
                "streaming_enabled": False,
                "cache_float_dtype": "float32",
                "cache_index_dtype": "int32",
            },
            "numerics": {"conc_use_numba": True, "solver_timing_details": True},
        }
    )
    config["outputs"].update(
        {
            "out_dir": "results",
            "prefix": "cascade_run",
            "write_summary_csv": True,
            "write_segments_csv": False,
            "write_points_csv": False,
            "write_paraview": True,
            "save_network": True,
            "write_combined_sweep_csv": True,
            "combined_sweep_filename": "sweep_summary.csv",
            "export_float_dtype": "float32",
            "export_index_dtype": "int32",
        }
    )
    config["gui"] = {
        "project_name": "Untitled CASCADE project",
        "boundary_conditions": {
            "mode": "flow_pressure",
            "pressure_unit": "mmHg",
            "flow_unit": "µL/min",
        },
        "rebuild_svv_for_pressure_drop": False,
        "oxygen_input_unit": "mmHg",
        "diffusivity_unit": "cm²/s",
        "solver_detail": "Guided",
        "auto_svv_roots": True,
        "reserve_gpu_for_simulation": True,
        "leave_ram_headroom": True,
        "sweeps": [],
        "analysis": {
            "colormap": "plasma",
            "vessel_opacity": 1.0,
            "tissue_opacity": 0.45,
        },
        "viewer": {
            "vessel_mode": "near",
            "vessel_limit": 5000,
            "tissue_mode": "near",
            "tissue_limit": 10000,
        },
    }
    return config


def merge_project(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge older/plain CASCADE JSON with GUI defaults without losing unknown keys."""
    merged = default_project()
    _deep_merge(merged, deepcopy(raw))
    merged["schema_version"] = int(raw.get("schema_version", GUI_SCHEMA_VERSION))
    return merged


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def save_project(path: str | Path, config: dict[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    materialize_project_assets(config, target.parent)
    _atomic_json(target, config)
    return target


def load_project(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("project file must contain a JSON object")
    return merge_project(raw)


def materialize_project_assets(
    config: dict[str, Any], project_dir: str | Path
) -> None:
    """Copy external domain/network inputs into the portable project folder."""
    root = Path(project_dir).expanduser().resolve()
    assets = root / "assets"
    paths: list[tuple[dict[str, Any], str]] = []
    domain = config.get("domain", {})
    network = config.get("network", {})
    simple = network.get("simple", {}) if isinstance(network, dict) else {}
    if isinstance(domain, dict) and domain.get("path"):
        paths.append((domain, "path"))
    if isinstance(network, dict) and network.get("input_path"):
        paths.append((network, "input_path"))
    if isinstance(simple, dict):
        for key in ("path", "geometry_path"):
            if simple.get(key):
                paths.append((simple, key))
                break
    for owner, key in paths:
        source = Path(str(owner[key])).expanduser()
        if not source.is_absolute():
            source = (root / source).resolve()
        else:
            source = source.resolve()
        if not source.is_file():
            continue
        try:
            owner[key] = str(source.relative_to(root))
            continue
        except ValueError:
            pass
        stat = source.stat()
        fingerprint = hashlib.sha256(
            f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:10]
        destination = assets / f"{source.stem}-{fingerprint}{source.suffix}"
        assets.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            shutil.copy2(source, destination)
        owner[key] = str(destination.relative_to(root))


def validate_project(
    config: dict[str, Any], base_dir: str | Path | None = None
) -> ValidationReport:
    report = ValidationReport()
    try:
        parse_config(config)
    except Exception as exc:
        report.errors.append(str(exc))

    gui = config.get("gui", {})
    bc = gui.get("boundary_conditions", {})
    bc_mode = bc.get("mode", "flow_pressure")
    flow = _float_at(config, "simulation.qin_target_ul_min", math.nan)
    settings = config.get("settings", {})
    hemo = settings.get("hemodynamics", settings.get("kirchhoff", {}))
    runtime_bc_mode = config.get("simulation", {}).get("kirchhoff_bc_mode")
    if runtime_bc_mode is None:
        runtime_bc_mode = hemo.get(
            "kirchhoff_bc_mode",
            hemo.get(
                "KIRCHHOFF_BC_MODE",
                settings.get("kirchhoff", {}).get("bc_mode"),
            ),
        )
    runtime_bc_mode = (
        str(runtime_bc_mode).strip().lower().replace("-", "_")
        if runtime_bc_mode is not None
        else None
    )
    if runtime_bc_mode in {"pressure_pressure", "fixed_pressure_drop", "dirichlet"}:
        bc_mode = "pressure_pressure"
    elif runtime_bc_mode in {
        "legacy_equal_terminal_flow",
        "equal_terminal_flow",
        "equal_terminal_flows",
    }:
        bc_mode = "legacy_equal_flow"
    elif runtime_bc_mode in {
        "terminal_pressure",
        "mixed",
        "pressure_terminals",
    }:
        bc_mode = "flow_pressure"
    root_p = float(hemo.get("root_pressure", hemo.get("ROOT_PRESSURE", math.nan)))
    outlet_p = float(
        hemo.get("terminal_pressure", hemo.get("TERMINAL_PRESSURE", math.nan))
    )
    if bc_mode in {"flow_pressure", "legacy_equal_flow"}:
        if not npfinite_positive(flow):
            report.errors.append(
                "Boundary conditions need a positive total inlet flow."
            )
        if not math.isfinite(outlet_p):
            report.errors.append(
                "Boundary conditions need a finite outlet/reference pressure."
            )
        if math.isfinite(root_p) and root_p <= outlet_p:
            report.warnings.append(
                "Inlet pressure is not above outlet pressure; verify pressure units and flow direction."
            )
    elif bc_mode == "pressure_pressure":
        if not math.isfinite(root_p) or not math.isfinite(outlet_p):
            report.errors.append(
                "Pressure boundary conditions need finite inlet and outlet pressures."
            )
        elif root_p <= outlet_p:
            report.errors.append(
                "Inlet pressure must be above outlet pressure for forward flow."
            )
        else:
            report.notes.append(
                "Inlet flow will be calculated directly from the prescribed pressure drop."
            )
    else:
        report.errors.append(f"Unknown boundary-condition mode: {bc_mode}")

    domain = config.get("domain", {})
    if domain.get("type") not in {"cube", "box", "sphere", "cylinder", "disk"} and not domain.get("path"):
        report.errors.append("An uploaded domain requires a readable domain file path.")
    domain_path = resolve_domain_path(domain.get("path"))
    if domain.get("path") and (domain_path is None or not domain_path.exists()):
        report.errors.append(f"Domain file does not exist: {domain['path']}")

    network = config.get("network", {})
    source = gui.get("network_source", "svv_generated")
    if source == "uploaded" and not network.get("input_path"):
        report.errors.append(
            "Uploaded vessel mode requires a .tree.npz, .forest, or .forest.simcache file."
        )
    if (
        network.get("input_path")
        and not Path(str(network["input_path"])).expanduser().exists()
    ):
        report.errors.append(f"Vessel file does not exist: {network['input_path']}")
    simple = network.get("simple", {})
    if str(simple.get("mode", "")).strip().lower() == "custom":
        custom_value = simple.get("path", simple.get("geometry_path"))
        if custom_value:
            custom_path = Path(str(custom_value)).expanduser()
            if not custom_path.is_absolute() and base_dir is not None:
                custom_path = Path(base_dir).expanduser() / custom_path
            if not custom_path.is_file():
                report.errors.append(f"Custom geometry file does not exist: {custom_path}")
    if source == "lattice" or simple.get("mode") == "lattice":
        if domain.get("type") not in {"cube", "box"}:
            report.notes.append(
                "The lattice will be generated across the domain bounding box; outside nodes and their incident edges will be removed."
            )
        lattice_type = str(simple.get("lattice_type", "cubic"))
        try:
            layout = _configured_lattice_layout(config, simple)
            segments = channel_count(
                int(layout["generator_cells"]), lattice_type
            ) * max(int(simple.get("subdivisions", 1)), 1)
            if segments > 1_000_000:
                report.warnings.append(
                    f"This lattice contains about {segments:,} vessel segments before export."
                )
        except Exception as exc:
            report.errors.append(str(exc))
        if simple.get("radius_expression"):
            report.notes.append(
                "The radius expression is evaluated at subdivided segment midpoints using x, y, z, r0 and L in cm."
            )

    sim = config.get("simulation", {})
    solver = str(sim.get("concentration_solver", "topdown"))
    if source == "lattice" and solver not in {
        "network_ext",
        "network_ext_hybrid_bg",
        "network",
    }:
        report.errors.append(
            "Lattice networks contain loops and require the general network concentration solver."
        )
    if solver in {
        "network_ext",
        "network_ext_hybrid_bg",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
    }:
        report.notes.append(
            "The external-concentration solve couples vessel oxygen to the tissue field."
        )
    if sim.get("viability_threshold") is None:
        report.notes.append(
            "Viability classification is disabled; the exported fallback remains 1% of inlet normalization."
        )

    outputs = config.get("outputs", {})
    if gui.get("sweeps") and not outputs.get("write_summary_csv", True):
        if outputs.get("write_combined_sweep_csv", False):
            report.errors.append(
                "Combined sweep CSV requires Summary metrics CSV. Enable the per-run summary output."
            )
        else:
            report.warnings.append(
                "Summary CSV is disabled, so the built-in sweep plotter will have no metrics to compare."
            )
    if outputs.get("write_paraview") and int(outputs.get("vessel_resolution", 2)) > 4:
        report.warnings.append(
            "High vessel VTK resolution duplicates every segment several times in memory."
        )
    return report


def hardware_info() -> HardwareInfo:
    total = available = 0
    try:
        import psutil

        vm = psutil.virtual_memory()
        total, available = int(vm.total), int(vm.available)
    except Exception:
        try:
            entries = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                entries[key] = int(value.strip().split()[0]) * 1024
            total, available = (
                entries.get("MemTotal", 0),
                entries.get("MemAvailable", 0),
            )
        except Exception:
            pass
    gpu_name, gpu_mem = _gpu_info()
    return HardwareInfo(total, available, os.cpu_count() or 1, gpu_name, gpu_mem)


def estimate_resources(
    config: dict[str, Any], hardware: HardwareInfo | None = None
) -> ResourceEstimate:
    hardware = hardware or hardware_info()
    gui = config.get("gui", {})
    network = config.get("network", {})
    source = gui.get("network_source", "svv_generated")
    if source == "lattice" or network.get("simple", {}).get("mode") == "lattice":
        simple = network.get("simple", {})
        try:
            layout = _configured_lattice_layout(config, simple)
            segments = channel_count(
                int(layout["generator_cells"]),
                str(simple.get("lattice_type", "cubic")),
            )
            segments *= max(int(simple.get("subdivisions", 1)), 1)
        except Exception:
            segments = 0
    elif network.get("input_path"):
        segments = int(network.get("estimated_segment_count", 0))
    else:
        targets = (
            network.get("target_total_terminal_count")
            or network.get("target_terminal_count")
            or 1
        )
        segments = max(2 * int(targets) + 1, 1)

    sim = config.get("simulation", {})
    if sim.get("geometry_only") or sim.get("skip_tissue_oxygen"):
        points = 0
    elif sim.get("sample_mode", "random") == "grid":
        grid = sim.get("tissue_grid", {})
        fallback = list(grid.get("shape", grid.get("dimensions", [20, 20, 20])) or [])
        fallback = (fallback + [20, 20, 20])[:3]
        shape = (
            grid.get("nx", fallback[0]),
            grid.get("ny", fallback[1]),
            grid.get("nz", fallback[2]),
        )
        points = math.prod(int(v) for v in shape)
    else:
        points = max(int(sim.get("distance_sample_count", 0)), 0)

    settings = config.get("settings", {})
    oxygen = settings.get("oxygen", {})
    cext = settings.get("cext", {})
    tissue = settings.get("tissue", {})
    tissue_gl = int(oxygen.get("gl_order", oxygen.get("GL_ORDER", 5)))
    cext_gl = int(oxygen.get("gl_order_cext", oxygen.get("GL_ORDER_CEXT", tissue_gl)))
    gl = max(tissue_gl, cext_gl)
    nearest = int(
        tissue.get("nearest_vessels", tissue.get("NEAREST_TISSUE_VESSELS", 250))
    )
    vessel_bytes = segments * (900 + 64 * max(gl, 1))
    streaming = bool(
        tissue.get("streaming_enabled", tissue.get("TISSUE_STREAMING_ENABLED", False))
    )
    point_bytes = points * (
        128 if streaming else 64 + 28 * min(max(nearest, 1), max(segments, 1))
    )
    fft_bytes = 0
    gpu_bytes = segments * (256 + 48 * max(gl, 1))
    if str(sim.get("concentration_solver", "")) in {
        "network_ext",
        "topdown_ext",
        "topdown_ext_hybrid_bg",
        "topdown_ext_treecode",
        "network_ext_hybrid_bg",
    }:
        mode = str(cext.get("hybrid_bg_mode", cext.get("CEXT_HYBRID_BG_MODE", "fft")))
        if mode == "fft":
            grid_n = int(
                cext.get("hybrid_bg_grid", cext.get("CEXT_HYBRID_BG_GRID", 256))
            )
            bins = int(
                cext.get(
                    "hybrid_bg_lambda_bins", cext.get("CEXT_HYBRID_BG_LAMBDA_BINS", 5)
                )
            )
            fft_bytes = grid_n**3 * max(bins, 1) * 4 * 8
            gpu_bytes += fft_bytes
        else:
            gpu_bytes += segments * 320
    export_multiplier = 1.0
    outputs = config.get("outputs", {})
    if outputs.get("write_segments_csv") or outputs.get("write_paraview"):
        export_multiplier += 0.6
    if outputs.get("write_points_csv") or outputs.get("write_paraview"):
        export_multiplier += 0.5
    host = int(
        (vessel_bytes + point_bytes + 0.35 * fft_bytes + 300_000_000)
        * export_multiplier
    )
    gpu = int(gpu_bytes)
    reserve = (
        max(2 * 1024**3, int(0.1 * hardware.total_ram_bytes))
        if hardware.total_ram_bytes
        else 2 * 1024**3
    )
    available_for_run = max(hardware.available_ram_bytes - reserve, 0)
    level = "ok"
    messages = []
    if hardware.available_ram_bytes and host > available_for_run:
        level = "danger"
        messages.append(
            "estimated host RAM exceeds available RAM after safety headroom"
        )
    elif hardware.available_ram_bytes and host > 0.7 * available_for_run:
        level = "warning"
        messages.append("estimated host RAM uses most currently available memory")
    if hardware.gpu_memory_bytes and gpu > 0.9 * hardware.gpu_memory_bytes:
        level = "danger"
        messages.append("estimated GPU memory exceeds the safe device budget")
    elif (
        hardware.gpu_memory_bytes
        and gpu > 0.7 * hardware.gpu_memory_bytes
        and level == "ok"
    ):
        level = "warning"
        messages.append("estimated GPU memory is close to device capacity")
    if not messages:
        messages.append("estimate fits the detected hardware budget")
    return ResourceEstimate(segments, points, host, gpu, level, "; ".join(messages))


def _configured_lattice_layout(
    config: dict[str, Any], simple: dict[str, Any]
) -> dict[str, Any]:
    """Use the same domain dimensions and lattice resolution as the preview."""
    from cascade.gui.visualization.geometry import _domain_dimensions

    dims, _center = _domain_dimensions(config.get("domain", {}))
    return resolve_lattice_layout(
        tuple(float(value) for value in dims),
        cells=int(simple.get("cells", simple.get("cells_per_axis", 4))),
        sizing_mode=str(simple.get("sizing_mode", "cells")),
        cell_spacing_cm=simple.get("cell_spacing_cm"),
        anisotropy_yx=float(simple.get("anisotropy_yx", 1.0)),
        anisotropy_zx=float(simple.get("anisotropy_zx", 1.0)),
        lattice_type=str(simple.get("lattice_type", "cubic")),
    )


def expand_sweeps(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    sweeps = [
        item
        for item in config.get("gui", {}).get("sweeps", [])
        if item.get("enabled", True)
    ]
    if not sweeps:
        return [("base", deepcopy(config))]
    paths, values = [], []
    for sweep in sweeps:
        path = str(sweep.get("path", "")).strip()
        vals = sweep.get("values", [])
        if not path or not isinstance(vals, list) or not vals:
            continue
        paths.append(path)
        values.append(vals)
    if not paths:
        return [("base", deepcopy(config))]
    # Geometry dimensions change slowest. This keeps every solver-only
    # combination for one geometry contiguous before the next domain/network
    # geometry is constructed.
    dimensions = sorted(
        zip(paths, values),
        key=lambda item: 0 if _sweep_affects_geometry(item[0]) else 1,
    )
    paths = [path for path, _values in dimensions]
    values = [dimension_values for _path, dimension_values in dimensions]
    if math.prod(len(v) for v in values) > 10_000:
        raise ValueError(
            "Sweep expands to more than 10,000 jobs; reduce its dimensions."
        )
    expanded = []
    for index, combination in enumerate(itertools.product(*values), start=1):
        run = deepcopy(config)
        labels = []
        for path, value in zip(paths, combination):
            set_path(run, path, value)
            if path == "settings.oxygen.conc_max_for_normalization":
                oxygen = run.setdefault("settings", {}).setdefault("oxygen", {})
                inlet = oxygen.get("concentration_inlet_by_fluid", {})
                if isinstance(inlet, dict):
                    oxygen["concentration_inlet_by_fluid"] = {
                        key: value
                        for key in (
                            inlet
                            or {
                                "blood": value,
                                "water": value,
                                "cell media": value,
                                "media": value,
                            }
                        )
                    }
                else:
                    oxygen["concentration_inlet_by_fluid"] = value
            labels.append(f"{path.split('.')[-1]}={value}")
        run.setdefault("gui", {})["sweeps"] = []
        expanded.append((f"{index:03d} · " + ", ".join(labels), run))
    return expanded


def _sweep_affects_geometry(path: str) -> bool:
    return (
        path.startswith(("domain.", "network.", "growth."))
        or path == "simulation.build_fluid"
    )


def create_jobs(config: dict[str, Any], project_dir: str | Path) -> list[JobRecord]:
    root = Path(project_dir).expanduser().resolve()
    materialize_project_assets(config, root)
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    gui_config = config.get("gui", {})
    project_name = str(gui_config.get("project_name", "CASCADE project"))
    requested_name = str(gui_config.get("run_name") or "").strip()
    run_name = requested_name or f"Run {len(list(runs_dir.glob('*')))+1:03d}"
    records: list[JobRecord] = []
    expanded = expand_sweeps(config)
    active_sweeps = [
        item
        for item in config.get("gui", {}).get("sweeps", [])
        if item.get("enabled", True)
        and item.get("path")
        and isinstance(item.get("values"), list)
        and item.get("values")
    ]
    for _label, run in expanded:
        _absolutize_input_paths(run, root)

    # A GUI sweep is queued as separate processes.  Give runs which have the
    # same domain/network/growth definition a common on-disk geometry bundle,
    # allowing all but the first run to skip domain patch solving and growth.
    geometry_keys = [_sweep_geometry_key(run) for _label, run in expanded]
    geometry_counts = {key: geometry_keys.count(key) for key in set(geometry_keys)}
    geometry_batch = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    combined_enabled = bool(
        active_sweeps
        and config.get("outputs", {}).get("write_combined_sweep_csv", True)
    )
    combined_filename = Path(
        str(
            config.get("outputs", {}).get(
                "combined_sweep_filename", "sweep_summary.csv"
            )
        )
    ).name
    if not combined_filename.lower().endswith(".csv"):
        combined_filename += ".csv"

    for (label, run), geometry_key in zip(expanded, geometry_keys):
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        display_name = run_name if label == "base" else f"{run_name} - {label}"
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", display_name).strip("-._")
        slug = slug[:64] or "run"
        output_dir = runs_dir / f"{job_id}-{slug}"
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_csv_path = (
            runs_dir / f"sweep-{geometry_batch}" / combined_filename
            if combined_enabled
            else None
        )
        run.setdefault("outputs", {})["out_dir"] = str(output_dir)
        gui = run.setdefault("gui", {})
        gui["job_id"] = job_id
        gui["run_name"] = display_name
        sweep_parameters = [
            {
                "path": str(item["path"]),
                "label": _sweep_parameter_label(str(item["path"])),
                "column": _sweep_csv_column(str(item["path"])),
                "unit": str(item.get("unit") or ""),
                "value": _get_path(run, str(item["path"])),
            }
            for item in active_sweeps
        ]
        if combined_csv_path is not None:
            gui["sweep_batch_id"] = geometry_batch
            gui["combined_sweep_csv_path"] = str(combined_csv_path)
            gui["sweep_parameters"] = sweep_parameters
        if geometry_counts[geometry_key] > 1:
            gui["shared_geometry_cache_dir"] = str(
                root / ".cascade_gui" / "geometry_cache" / geometry_batch / geometry_key
            )
        settings_path = output_dir / "settings.json"
        _atomic_json(settings_path, run)
        records.append(
            JobRecord(
                id=job_id,
                name=display_name,
                settings_path=str(settings_path),
                output_dir=str(output_dir),
                log_path=str(output_dir / "run.log"),
                manifest_path=str(output_dir / "manifest.json"),
                sweep_batch_id=geometry_batch
                if combined_csv_path is not None
                else None,
                combined_csv_path=None
                if combined_csv_path is None
                else str(combined_csv_path),
                sweep_parameters=sweep_parameters
                if combined_csv_path is not None
                else [],
            )
        )
    return records


_SWEEP_CSV_INFO = {
    "simulation.qin_target_ul_min": ("Inlet flow", "sweep_inlet_flow_uL_per_min"),
    "network.target_terminal_count": ("Terminal count", "sweep_target_terminals"),
    "settings.oxygen.solute_diffusivity": (
        "Diffusivity",
        "sweep_diffusivity_cm2_per_s",
    ),
    "settings.oxygen.vmax_mm": ("Vmax", "sweep_vmax_mol_per_m3_per_s"),
    "settings.oxygen.k_m_mm": ("Km", "sweep_km_mol_per_m3"),
    "settings.hematocrit.hd_discharge": (
        "Discharge hematocrit",
        "sweep_discharge_hematocrit",
    ),
    "settings.oxygen.conc_max_for_normalization": (
        "Inlet concentration",
        "sweep_inlet_concentration_mol_per_m3",
    ),
    "settings.cext.hybrid_bg_grid": ("FFT grid", "sweep_fft_grid_points"),
    "settings.cext.vess_coupling_max_iter": (
        "Cext iterations",
        "sweep_cext_iterations",
    ),
}


def _sweep_parameter_label(path: str) -> str:
    return _SWEEP_CSV_INFO.get(path, (path, ""))[0]


def _sweep_csv_column(path: str) -> str:
    fallback = "sweep_" + path.replace(".", "_")
    return _SWEEP_CSV_INFO.get(path, (path, fallback))[1]


def _get_path(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _sweep_geometry_key(config: dict[str, Any]) -> str:
    """Return a stable key for inputs which require rebuilding geometry."""
    simulation = dict(config.get("simulation", {}) or {})
    payload = {
        "cache_format": 1,
        "domain": config.get("domain", {}),
        "network": config.get("network", {}),
        "growth": config.get("growth", {}),
        # Fluid selection can affect the parameters used while constructing a
        # tree.  Flow, oxygen, hematocrit, and tissue settings are deliberately
        # excluded: those are recomputed from the cached fixed geometry.
        "build_fluid": simulation.get("build_fluid", simulation.get("fluid")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _absolutize_input_paths(config: dict[str, Any], project_root: Path) -> None:
    for section, key in (("domain", "path"), ("network", "input_path")):
        value = config.get(section, {}).get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            config[section][key] = str((project_root / path).resolve())
    simple = config.get("network", {}).get("simple", {})
    if str(simple.get("mode", "")).strip().lower() == "custom":
        key = "path" if "path" in simple else "geometry_path"
        value = simple.get(key)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                simple[key] = str((project_root / path).resolve())


class QueueStore:
    def __init__(self, project_dir: str | Path):
        self.root = Path(project_dir).expanduser().resolve() / ".cascade_gui"
        self.path = self.root / QUEUE_FILENAME
        self.results_path = self.root / "results.json"

    def load(self) -> list[JobRecord]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            jobs = [JobRecord.from_dict(item) for item in raw.get("jobs", [])]
            for job in jobs:
                if job.status == "Running":
                    job.status = "Interrupted"
                    job.stage = "Previous session ended during this run"
            return jobs
        except Exception:
            backup = self.path.with_suffix(".corrupt.json")
            try:
                self.path.replace(backup)
            except Exception:
                pass
            return []

    def save(self, jobs: Iterable[JobRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.path,
            {
                "schema_version": GUI_SCHEMA_VERSION,
                "jobs": [job.to_dict() for job in jobs],
            },
        )

    def archive_results(self, jobs: Iterable[JobRecord]) -> None:
        archived = {job.id: job for job in self.load_results()}
        for job in jobs:
            if job.manifest_path and Path(job.manifest_path).is_file():
                archived[job.id] = job
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.results_path,
            {
                "schema_version": GUI_SCHEMA_VERSION,
                "jobs": [job.to_dict() for job in archived.values()],
            },
        )

    def load_results(self) -> list[JobRecord]:
        archived: dict[str, JobRecord] = {}
        if self.results_path.is_file():
            try:
                raw = json.loads(self.results_path.read_text(encoding="utf-8"))
                archived.update(
                    (job.id, job)
                    for job in (
                        JobRecord.from_dict(item) for item in raw.get("jobs", [])
                    )
                    if job.manifest_path and Path(job.manifest_path).is_file()
                )
            except Exception:
                pass
        for results_root in (self.root.parent / "runs", self.root.parent / "results"):
            if not results_root.is_dir():
                continue
            for manifest_path in results_root.glob("*/manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    settings = dict(manifest.get("settings", {}) or {})
                    gui = dict(settings.get("gui", {}) or {})
                    project = dict(manifest.get("project", {}) or {})
                    job_id = str(project.get("job_id") or manifest_path.parent.name)
                    if job_id in archived:
                        continue
                    archived[job_id] = JobRecord(
                        id=job_id,
                        name=str(
                            gui.get("run_name")
                            or gui.get("project_name")
                            or job_id
                        ),
                        settings_path=str(manifest.get("settings_path") or ""),
                        output_dir=str(manifest_path.parent),
                        status="Completed",
                        progress=100,
                        stage="Finished",
                        manifest_path=str(manifest_path),
                    )
                except Exception:
                    continue
        return list(archived.values())

    def delete_results(self, jobs: Iterable[JobRecord]) -> None:
        targets = list(jobs)
        target_ids = {job.id for job in targets}
        project = self.root.parent.resolve()
        allowed_roots = tuple(
            (project / name).resolve() for name in ("runs", "results", ".cascade_gui")
        )
        for job in targets:
            candidates = [Path(job.output_dir)]
            if job.settings_path:
                candidates.append(Path(job.settings_path).parent)
            for candidate in candidates:
                resolved = candidate.expanduser().resolve()
                if resolved in allowed_roots or not any(
                    root in resolved.parents for root in allowed_roots
                ):
                    continue
                if resolved.is_dir():
                    shutil.rmtree(resolved)
        remaining = [
            job for job in self.load_results() if job.id not in target_ids
        ]
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            self.results_path,
            {
                "schema_version": GUI_SCHEMA_VERSION,
                "jobs": [job.to_dict() for job in remaining],
            },
        )


def pressure_to_pa(value: float, unit: str) -> float:
    return float(value) * MMHG_TO_PA if "mmhg" in unit.lower() else float(value)


def pressure_from_pa(value: float, unit: str) -> float:
    return float(value) / MMHG_TO_PA if "mmhg" in unit.lower() else float(value)


def flow_to_ul_min(value: float, unit: str) -> float:
    key = unit.lower().replace("μ", "µ")
    if "cm³/s" in key or "cm3/s" in key:
        return float(value) * 60_000.0
    if "ml" in key:
        return float(value) * 1000.0
    if "m³" in key or "m3" in key:
        return float(value) * 60.0 * 1e9
    return float(value)


def flow_from_ul_min(value: float, unit: str) -> float:
    key = unit.lower().replace("μ", "µ")
    if "cm³/s" in key or "cm3/s" in key:
        return float(value) / 60_000.0
    if "ml" in key:
        return float(value) / 1000.0
    if "m³" in key or "m3" in key:
        return float(value) / (60.0 * 1e9)
    return float(value)


def oxygen_to_concentration(value: float, unit: str) -> float:
    return float(value) * ALPHA_MMHG if "mmhg" in unit.lower() else float(value)


def oxygen_from_concentration(value: float, unit: str) -> float:
    return float(value) / ALPHA_MMHG if "mmhg" in unit.lower() else float(value)


def parse_points(text: str) -> list[list[float]]:
    points = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.replace(";", ",").split(",")]
        if len(parts) != 3:
            raise ValueError(f"line {line_no}: expected x, y, z")
        points.append([float(part) for part in parts])
    return points


def format_points(points: list[list[float]] | None) -> str:
    return "\n".join(
        ", ".join(f"{float(v):g}" for v in point) for point in (points or [])
    )


def parse_sweep_values(text: str) -> list[Any]:
    out: list[Any] = []
    for part in text.replace("\n", ",").split(","):
        value = part.strip()
        if not value:
            continue
        lowered = value.lower()
        if lowered in {"true", "false"}:
            out.append(lowered == "true")
            continue
        try:
            number = float(value)
            # Editing is deliberately permissive, but an incomplete or
            # overflowing numeric token must never take down a live Qt slot.
            if not math.isfinite(number):
                out.append(value)
            else:
                out.append(int(number) if number.is_integer() else number)
        except (ValueError, OverflowError):
            out.append(value)
    return out


def set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = mapping
    for part in parts[:-1]:
        target = target.setdefault(part, {})
        if not isinstance(target, dict):
            raise ValueError(f"cannot set {path}; {part} is not an object")
    target[parts[-1]] = value


def human_bytes(value: int | float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024.0 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def open_folder(path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if sys_platform() == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys_platform() == "darwin":
        subprocess.Popen(["open", str(target)])
    elif _is_wsl() and (explorer := _command_path("explorer.exe")):
        converted = subprocess.check_output(
            ["wslpath", "-w", str(target)], text=True
        ).strip()
        subprocess.Popen([explorer, converted])
    else:
        _open_linux_desktop_path(target)


def open_path(path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(target)
    if sys_platform() == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys_platform() == "darwin":
        subprocess.Popen(["open", str(target)])
    elif _is_wsl() and (cmd := _command_path("cmd.exe")):
        converted = subprocess.check_output(
            ["wslpath", "-w", str(target)], text=True
        ).strip()
        subprocess.Popen([cmd, "/c", "start", "", converted])
    else:
        _open_linux_desktop_path(target)


def sys_platform() -> str:
    import sys

    return sys.platform


def npfinite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _float_at(config: dict[str, Any], path: str, fallback: float) -> float:
    value: Any = config
    try:
        for part in path.split("."):
            value = value[part]
        return float(value)
    except Exception:
        return fallback


def _atomic_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _gpu_info() -> tuple[str, int]:
    """Discover NVIDIA hardware without confusing a cold WSL driver with absence."""
    from shutil import which

    candidates = [which("nvidia-smi")]
    if _is_wsl():
        candidates.append("/usr/lib/wsl/lib/nvidia-smi")
    candidates.extend(("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"))
    attempted: set[str] = set()
    for executable in candidates:
        if not executable or executable in attempted or not Path(executable).exists():
            continue
        attempted.add(executable)
        try:
            output = subprocess.check_output(
                [
                    executable,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                # WSL may need several seconds to initialize the GPU bridge
                # after launch. Two seconds produced a false negative here.
                timeout=12,
            ).splitlines()[0]
            name, memory_mib = output.rsplit(",", 1)
            return name.strip(), int(float(memory_mib.strip()) * 1024**2)
        except Exception:
            continue
    return "Not detected", 0


def _is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except Exception:
        return False


def _command_path(name: str) -> str | None:
    from shutil import which

    executable = which(name)
    if executable or not _is_wsl():
        return executable

    # Desktop launchers intentionally use a small, predictable PATH. Windows
    # interop still works in that environment, but its executables are no
    # longer discoverable through ``which``. Probe the standard WSL mounts so
    # folder/file actions do not incorrectly fall back to Linux desktop tools.
    relative_paths = {
        "explorer.exe": ("Windows", "explorer.exe"),
        "cmd.exe": ("Windows", "System32", "cmd.exe"),
    }
    relative = relative_paths.get(name.lower())
    if relative:
        for drive in "cdefghijklmnopqrstuvwxyz":
            candidate = Path("/mnt") / drive / Path(*relative)
            if candidate.is_file():
                return str(candidate)
    return None


def _open_linux_desktop_path(target: Path) -> None:
    """Open a path with an available Linux desktop integration."""
    for command, arguments in (("xdg-open", ()), ("gio", ("open",))):
        if executable := _command_path(command):
            subprocess.Popen([executable, *arguments, str(target)])
            return
    raise RuntimeError(
        "No desktop opener is available. Install 'xdg-utils' (which provides "
        "xdg-open) or GLib's 'gio' command. On WSL, also verify that Windows "
        "interop is enabled."
    )
