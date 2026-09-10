from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cascade.growth as growth_module
from cascade.config import parse_config
from cascade.gpu import gpu_requested
from cascade.gui.model import (
    HardwareInfo,
    JobRecord,
    QueueStore,
    create_jobs,
    default_project,
    estimate_resources,
    expand_sweeps,
    flow_from_ul_min,
    flow_to_ul_min,
    oxygen_from_concentration,
    oxygen_to_concentration,
    pressure_from_pa,
    pressure_to_pa,
    validate_project,
)
from cascade.gui.sweep_csv import rebuild_combined_sweep_csv
from cascade.lattice import channel_count, generate_lattice
from cascade.export import _segment_polydata
from cascade.gui.preview import _polyline_data
from cascade.growth import apply_runtime_settings
from cascade.runtime import tissuesim as ts


def test_vtk_vessels_retain_computed_external_field_quadrature_nodes():
    nodes, _ = np.polynomial.legendre.leggauss(5)
    t = 0.5 * (nodes + 1.0)
    bulk = np.linspace(0.14, 0.10, 5, dtype=np.float32)
    profile = {
        "gl_points_si": np.column_stack(
            (0.01 * t, np.zeros(5), np.zeros(5))
        )[None, :, :].astype(np.float32),
        "c_iv_gl": bulk[None, :],
        "c_bulk_gl": bulk[None, :],
        "c_wall_gl": (bulk - 0.01)[None, :],
        "c_ext_gl": np.linspace(0.02, 0.01, 5, dtype=np.float32)[None, :],
    }
    rows = [
        {
            "start_x": 0.0,
            "start_y": 0.0,
            "start_z": 0.0,
            "end_x": 1.0,
            "end_y": 0.0,
            "end_z": 0.0,
            "tree_id": 0,
            "local_segment_id": 0,
            "global_segment_id": 0,
            "cin": 0.14,
            "cout": 0.10,
            "flow_cm3_s": 1.0,
            "flow_ul_min": 60000.0,
            "radius_cm": 0.01,
            "length_cm": 1.0,
        }
    ]
    tree_results = [SimpleNamespace(tree_id=0, details={"vessel_quadrature": profile})]

    mesh = _segment_polydata(
        rows,
        resolution=2,
        float_dtype=np.dtype(np.float32),
        index_dtype=np.dtype(np.int32),
        tree_results=tree_results,
    )

    quadrature_mask = np.asarray(mesh["solver_quadrature_node"], dtype=bool)
    assert mesh.n_points == 7  # two endpoints plus the five computed nodes
    assert np.count_nonzero(quadrature_mask) == 5
    assert np.asarray(mesh["solver_quadrature_order"])[quadrature_mask].tolist() == [5] * 5
    assert np.asarray(mesh["bulk_oxygen"])[quadrature_mask] == pytest.approx(bulk)
    assert np.asarray(mesh["concentration"])[quadrature_mask] == pytest.approx(bulk)
    starts, ends, arrays, _inlets = _polyline_data(mesh)
    assert starts.shape == ends.shape == (6, 3)
    assert arrays["bulk_oxygen"].shape == (6,)
    assert np.ptp(arrays["bulk_oxygen"]) > 0.0


def test_distinct_quadrature_orders_reach_tissuesim(monkeypatch):
    raw = default_project()
    raw["settings"]["oxygen"]["gl_order"] = 3
    raw["settings"]["oxygen"]["gl_order_cext"] = 7
    config = parse_config(raw)
    monkeypatch.setattr(ts, "GL_ORDER", 5)
    monkeypatch.setattr(ts, "GL_ORDER_CEXT", 5)

    apply_runtime_settings(ts, config)

    assert ts.GL_ORDER == 3
    assert ts.GL_ORDER_CEXT == 7


@pytest.mark.parametrize(
    ("kind", "cells"),
    [("cubic", 2), ("octet", 2), ("bcc", 2), ("diamond", 2)],
)
def test_paper_lattice_counts_and_radius_expression(kind, cells):
    lattice = generate_lattice(
        cells,
        (1.0, 1.0, 1.0),
        5e-4,
        lattice_type=kind,
        subdivisions=2,
        radius_expression="r0 * (1 + 0.1*x/L)",
    )
    assert lattice["edge_nodes"].shape == (2 * channel_count(cells, kind), 2)
    assert (lattice["segment_radii_cm"] > 0).all()
    assert len(lattice["inlet_nodes"]) == 1
    assert len(lattice["outlet_nodes"]) == 1


def test_octet_requires_even_cells():
    with pytest.raises(ValueError, match="even"):
        generate_lattice(3, (1, 1, 1), 0.001, lattice_type="octet")


def test_lattice_nodes_and_incident_edges_are_clipped_to_domain():
    center = np.array([1.0, 2.0, 3.0])
    radius = 0.51

    def inside(points):
        offsets = points - center
        return (offsets * offsets).sum(axis=1) <= radius**2

    lattice = generate_lattice(
        4,
        (1.0, 1.0, 1.0),
        0.001,
        lattice_type="cubic",
        center_cm=tuple(center),
        node_inside=inside,
    )
    nodes = lattice["node_coords_cm"]
    edges = lattice["edge_nodes"]
    assert inside(nodes).all()
    assert edges.min() >= 0
    assert edges.max() < len(nodes)
    assert lattice["nodes_removed_by_domain"] > 0
    assert lattice["edges_removed_by_domain"] > 0
    assert len(lattice["inlet_nodes"]) == 1
    assert len(lattice["outlet_nodes"]) == 1


def test_prescribed_lattice_boundaries_are_added_and_connected_to_nearest_nodes():
    inlet = np.array([-0.62, -0.19, -0.11])
    outlet = np.array([0.63, 0.18, 0.12])
    lattice = generate_lattice(
        2,
        (1.0, 1.0, 1.0),
        0.001,
        lattice_type="cubic",
        inlet_points_cm=[inlet.tolist()],
        outlet_points_cm=[outlet.tolist()],
    )

    assert lattice["inlet_connection_count"] == 1
    assert lattice["outlet_connection_count"] == 1
    assert lattice["edge_nodes"].shape[0] == channel_count(2, "cubic") + 2
    assert np.allclose(lattice["inlet_points_cm"], [inlet])
    assert np.allclose(lattice["outlet_points_cm"], [outlet])
    endpoints = np.vstack(
        (lattice["segment_starts_cm"], lattice["segment_ends_cm"])
    )
    assert np.any(np.all(np.isclose(endpoints, inlet), axis=1))
    assert np.any(np.all(np.isclose(endpoints, outlet), axis=1))


def test_radius_expression_accepts_caret_exponentiation():
    caret = generate_lattice(
        2,
        (1.0, 1.0, 1.0),
        0.01,
        radius_expression="r0 * (1 + (x/L)^2)",
    )
    stars = generate_lattice(
        2,
        (1.0, 1.0, 1.0),
        0.01,
        radius_expression="r0 * (1 + (x/L)**2)",
    )
    assert np.array_equal(caret["segment_radii_cm"], stars["segment_radii_cm"])


def test_default_project_is_engine_compatible_and_memory_estimated():
    config = default_project()
    assert config["domain"] == {
        "type": "box",
        "side_length": 1.0,
        "x_length": 1.0,
        "y_length": 1.0,
        "z_length": 1.0,
        "random_seed": 42,
    }
    parsed = parse_config(config)
    assert parsed.network_mode == "tree"
    assert gpu_requested(parsed)
    assert config["settings"]["oxygen"]["gl_order"] == 5
    assert config["settings"]["oxygen"]["gl_order_cext"] == 1
    assert config["settings"]["oxygen"]["axial_blood_steps"] == 5
    assert config["settings"]["cext"]["vess_coupling_max_iter"] == 1
    assert config["settings"]["cext"]["hybrid_bg_grid"] == 256
    assert config["settings"]["cext"]["window_factor"] == 6
    assert config["settings"]["cext"]["float_dtype"] == "float32"
    assert config["outputs"]["write_paraview"] is True
    report = validate_project(config)
    assert report.runnable
    assert not any("export records" in warning.lower() for warning in report.warnings)
    estimate = estimate_resources(config)
    assert estimate.segment_count == 199
    assert estimate.tissue_point_count == 10_000
    assert estimate.host_memory_bytes > 0


def test_geometry_only_run_does_not_request_gpu():
    config = default_project()
    config["simulation"]["geometry_only"] = True
    assert not gpu_requested(parse_config(config))


def test_lattice_is_allowed_in_non_box_domain():
    config = default_project()
    config["domain"] = {"type": "sphere", "radius": 0.5, "center": [0, 0, 0]}
    config["gui"]["network_source"] = "lattice"
    config["network"]["simple"] = {
        "mode": "lattice",
        "lattice_type": "cubic",
        "cells": 4,
    }
    config["simulation"]["concentration_solver"] = "network"
    report = validate_project(config)
    assert report.runnable
    assert not any("cube" in error.lower() for error in report.errors)
    assert any("outside nodes" in note.lower() for note in report.notes)


def test_pressure_only_boundary_conditions_are_blocked():
    config = default_project()
    config["gui"]["boundary_conditions"]["mode"] = "pressure_pressure"
    report = validate_project(config)
    assert not report.runnable
    assert any("Pressure-only" in message for message in report.errors)


def test_general_sweep_cartesian_product():
    config = default_project()
    config["gui"]["sweeps"] = [
        {"enabled": True, "path": "simulation.qin_target_ul_min", "values": [10, 20]},
        {
            "enabled": True,
            "path": "settings.oxygen.vmax_mm",
            "values": [0.01, 0.02, 0.03],
        },
    ]
    runs = expand_sweeps(config)
    assert len(runs) == 6
    assert {run[1]["simulation"]["qin_target_ul_min"] for run in runs} == {10, 20}


def test_geometry_sweep_is_outermost_regardless_of_ui_row_order():
    config = default_project()
    config["gui"]["sweeps"] = [
        {"enabled": True, "path": "simulation.qin_target_ul_min", "values": [10, 20]},
        {"enabled": True, "path": "network.target_terminal_count", "values": [50, 100]},
        {"enabled": True, "path": "settings.oxygen.vmax_mm", "values": [0.001, 0.005]},
    ]

    runs = expand_sweeps(config)
    terminal_counts = [run["network"]["target_terminal_count"] for _label, run in runs]

    assert terminal_counts == [50, 50, 50, 50, 100, 100, 100, 100]


@pytest.mark.parametrize(
    ("to_base", "from_base", "value", "unit"),
    [
        (pressure_to_pa, pressure_from_pa, 75.0, "mmHg"),
        (flow_to_ul_min, flow_from_ul_min, 1.5, "mL/min"),
        (flow_to_ul_min, flow_from_ul_min, 0.002, "cm³/s"),
        (oxygen_to_concentration, oxygen_from_concentration, 95.0, "mmHg"),
    ],
)
def test_unit_round_trip(to_base, from_base, value, unit):
    assert math.isclose(from_base(to_base(value, unit), unit), value, rel_tol=1e-12)


def test_jobs_freeze_settings_and_queue_round_trip(tmp_path):
    config = default_project()
    config["outputs"]["out_dir"] = "chosen-results"
    records = create_jobs(config, tmp_path)
    assert len(records) == 1
    assert records[0].output_dir.startswith(str(tmp_path / "chosen-results"))
    assert (tmp_path / ".cascade_gui" / "jobs" / records[0].id / "settings.json").exists()
    store = QueueStore(tmp_path)
    store.save(records)
    assert store.load()[0].id == records[0].id


def test_solver_only_sweep_jobs_share_one_geometry_cache(tmp_path):
    config = default_project()
    config["gui"]["sweeps"] = [
        {"enabled": True, "path": "simulation.qin_target_ul_min", "values": [10, 20]},
        {"enabled": True, "path": "settings.oxygen.vmax_mm", "values": [0.001, 0.005]},
    ]

    records = create_jobs(config, tmp_path)
    frozen = [json.loads(open(record.settings_path, encoding="utf-8").read()) for record in records]
    cache_dirs = {run["gui"].get("shared_geometry_cache_dir") for run in frozen}

    assert len(records) == 4
    assert len(cache_dirs) == 1
    assert None not in cache_dirs
    assert len({record.sweep_batch_id for record in records}) == 1
    assert len({record.combined_csv_path for record in records}) == 1
    assert [parameter["column"] for parameter in records[0].sweep_parameters] == [
        "sweep_inlet_flow_uL_per_min",
        "sweep_vmax_mol_per_m3_per_s",
    ]


def test_combined_sweep_csv_is_long_form_atomic_and_keeps_failed_runs(tmp_path):
    config = default_project()
    config["outputs"]["out_dir"] = str(tmp_path / "results")
    config["gui"]["sweeps"] = [
        {
            "enabled": True,
            "path": "simulation.qin_target_ul_min",
            "unit": "µL/min",
            "values": [10, 20],
        }
    ]
    records = create_jobs(config, tmp_path)
    combined = Path(records[0].combined_csv_path)

    first_output = Path(records[0].output_dir)
    first_output.mkdir(parents=True)
    (first_output / "summary.csv").write_text(
        "number_of_trees,target_terminals,total_volume_mean,tree_id\n"
        "2,100,1.25,0\n"
        "2,100,1.75,1\n",
        encoding="utf-8",
    )
    records[0].status = "Completed"
    records[1].status = "Failed"
    records[1].error = "solver failed"

    rebuild_combined_sweep_csv(records, combined)
    rebuild_combined_sweep_csv(records, combined)
    with combined.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert [row["sweep_inlet_flow_uL_per_min"] for row in rows] == ["10", "10", "20"]
    assert [row["tree_id"] for row in rows[:2]] == ["0", "1"]
    assert rows[2]["run_status"] == "Failed"
    assert rows[2]["error"] == "solver failed"
    assert rows[2]["total_volume_mean"] == ""
    assert not combined.with_name(combined.name + ".tmp").exists()


def test_inlet_concentration_sweep_updates_actual_inlet_values():
    config = default_project()
    config["gui"]["sweeps"] = [
        {
            "enabled": True,
            "path": "settings.oxygen.conc_max_for_normalization",
            "values": [0.1, 0.2],
        }
    ]

    runs = expand_sweeps(config)

    assert [run[1]["settings"]["oxygen"]["concentration_inlet_by_fluid"]["blood"] for run in runs] == [
        0.1,
        0.2,
    ]


def test_terminal_count_sweep_does_not_share_different_geometries(tmp_path):
    config = default_project()
    config["gui"]["sweeps"] = [
        {"enabled": True, "path": "network.target_terminal_count", "values": [50, 100]},
    ]

    records = create_jobs(config, tmp_path)
    frozen = [json.loads(open(record.settings_path, encoding="utf-8").read()) for record in records]

    assert len(records) == 2
    assert all("shared_geometry_cache_dir" not in run["gui"] for run in frozen)


def test_each_terminal_count_group_reuses_geometry_for_solver_variants(tmp_path):
    config = default_project()
    config["gui"]["sweeps"] = [
        {"enabled": True, "path": "network.target_terminal_count", "values": [50, 100]},
        {"enabled": True, "path": "settings.oxygen.vmax_mm", "values": [0.001, 0.005]},
    ]

    records = create_jobs(config, tmp_path)
    frozen = [json.loads(open(record.settings_path, encoding="utf-8").read()) for record in records]
    by_target: dict[int, set[str]] = {}
    for run in frozen:
        by_target.setdefault(run["network"]["target_terminal_count"], set()).add(
            run["gui"]["shared_geometry_cache_dir"]
        )

    assert all(len(cache_dirs) == 1 for cache_dirs in by_target.values())
    assert next(iter(by_target[50])) != next(iter(by_target[100]))


def test_second_sweep_run_loads_geometry_instead_of_building_it(tmp_path, monkeypatch):
    cache_dir = tmp_path / "shared-geometry"
    raw = default_project()
    raw["gui"]["shared_geometry_cache_dir"] = str(cache_dir)
    config = parse_config(raw)
    config.settings_path = tmp_path / "settings.json"
    calls = {"domain_build": 0, "network_build": 0, "network_load": 0}

    class FakeDomain:
        random_seed = 42

        def set_random_generator(self):
            pass

        def save(self, path, **_kwargs):
            target = cache_dir / "domain.dmn"
            target.write_bytes(b"domain")
            return str(target)

    class FakeTree:
        segment_count = 3
        n_terminals = 101

        def save(self, path):
            target = cache_dir / "network.tree"
            target.write_bytes(b"tree")
            return target

    domain = FakeDomain()
    tree = FakeTree()

    def build_domain(*_args, **_kwargs):
        calls["domain_build"] += 1
        return domain

    def build_trees(*_args, **_kwargs):
        calls["network_build"] += 1
        return [tree]

    def load_network(_path, loaded_domain, loaded_config):
        calls["network_load"] += 1
        assert loaded_domain is domain
        assert loaded_config.growth.enabled is False
        return [tree], None

    monkeypatch.setattr(growth_module, "load_runtime_module", lambda: SimpleNamespace())
    monkeypatch.setattr(growth_module, "apply_runtime_settings", lambda *_args: None)
    monkeypatch.setattr(growth_module, "build_domain", build_domain)
    monkeypatch.setattr(growth_module, "_pre_sample_points", lambda *_args: (np.empty((0, 3)), {}))
    monkeypatch.setattr(growth_module, "_build_configured_trees", build_trees)
    monkeypatch.setattr(growth_module, "_repair_and_validate_if_requested", lambda *_args: ([], []))
    monkeypatch.setattr(growth_module.Domain, "load", staticmethod(lambda _path: domain))
    monkeypatch.setattr(growth_module, "_load_existing_network", load_network)

    first = growth_module.build_or_load_network(config)
    second = growth_module.build_or_load_network(config)

    assert first.trees == [tree]
    assert second.trees == [tree]
    assert calls == {"domain_build": 1, "network_build": 1, "network_load": 1}
    assert second.build_timings["geometry_cache_load_s"] >= 0.0
    assert config.growth.enabled is True


def test_resource_estimator_warns_before_oversubscribing_ram():
    config = default_project()
    config["simulation"]["sample_mode"] = "grid"
    config["simulation"]["tissue_grid"] = {"shape": [512, 512, 128]}
    tiny_machine = HardwareInfo(
        total_ram_bytes=4 * 1024**3,
        available_ram_bytes=1 * 1024**3,
        cpu_count=4,
    )
    assert estimate_resources(config, tiny_machine).level == "danger"
