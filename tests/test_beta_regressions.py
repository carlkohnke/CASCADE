from __future__ import annotations

import json

import pytest

from cascade.configuration.schema import example_config, load_config, parse_config
from cascade.gui.model import JobRecord, QueueStore, _absolutize_input_paths
from cascade.runtime.planning import estimate_run_resources
from cascade.vessels.conditions import terminal_flow_for_target
from cascade.vessels.conditions import _tree_terminal_segments
from cascade.vessels.metadata import inspect_network


def test_cli_example_matches_studio_defaults():
    config = example_config()
    assert config["network"]["target_terminal_count"] == 100
    assert config["simulation"]["qin_target_ul_min"] == 100.0
    assert config["simulation"]["distance_sample_count"] == 10000
    assert config["simulation"]["concentration_solver"] == "network_ext"
    assert config["simulation"]["tissue_accel"] == "gpu"
    assert config["settings"]["cext"]["accel_mode"] == "gpu"


@pytest.mark.parametrize("fluid", [{"density": 1.0}])
def test_unsupported_fluids_fail_during_parsing(fluid):
    raw = example_config()
    raw["simulation"]["fluid"] = fluid
    with pytest.raises(ValueError, match="custom fluid"):
        parse_config(raw)


def test_legacy_constant_hematocrit_is_canonicalized():
    raw = example_config()
    raw["settings"]["hematocrit"]["model"] = "constant"
    parsed = parse_config(raw)
    assert parsed.runtime_settings["hematocrit"]["model"] == "uniform_tube"


def test_custom_network_path_is_validated_and_frozen(tmp_path):
    network = tmp_path / "network.csv"
    network.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z\n0,0,0,1,0,0\n",
        encoding="utf-8",
    )
    raw = example_config()
    raw["network"] = {
        "mode": "simple",
        "simple": {"mode": "custom", "path": "network.csv"},
    }
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_config(settings)
    assert loaded.network.simple["path"] == "network.csv"
    _absolutize_input_paths(raw, tmp_path)
    assert raw["network"]["simple"]["path"] == str(network.resolve())


def test_custom_csv_metadata_reports_segments(tmp_path):
    network = tmp_path / "network.csv"
    network.write_text(
        "start_x,start_y,start_z,end_x,end_y,end_z\n"
        "0,0,0,1,0,0\n1,0,0,2,0,0\n",
        encoding="utf-8",
    )
    metadata = inspect_network(network)
    assert metadata.kind == "custom-csv"
    assert metadata.segments == 2
    assert metadata.exact_counts


def test_terminal_target_uses_svv_growth_count():
    config = parse_config(example_config())
    assert terminal_flow_for_target(config, 1.0, 4) == pytest.approx(0.25)
    estimate = estimate_run_resources(config)
    assert estimate.segments == 201


def test_forest_resource_estimates_use_two_n_plus_one_per_tree():
    raw = example_config()
    raw["network"]["mode"] = "forest"
    raw["network"]["roots"] = [
        {"start": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0]},
        {"start": [0.0, 1.0, 0.0], "direction": [1.0, 0.0, 0.0]},
    ]
    raw["network"]["target_terminal_counts"] = [10, 20]
    config = parse_config(raw)
    assert estimate_run_resources(config).segments == 62

    raw["network"].pop("target_terminal_counts")
    raw["network"]["target_total_terminal_count"] = 30
    config = parse_config(raw)
    assert estimate_run_resources(config).segments == 62


def test_generated_svv_tree_rejects_impossible_one_terminal_target():
    raw = example_config()
    raw["network"]["target_terminal_count"] = 1
    with pytest.raises(ValueError, match="at least 2 final terminal"):
        parse_config(raw)


def test_terminal_metric_counts_topological_leaves():
    tree = type(
        "Tree",
        (),
        {
            "n_terminals": 2,
            "vessel_map": {
                0: {"downstream": [1, 2]},
                1: {"downstream": []},
                2: {"downstream": []},
            },
        },
    )()
    assert _tree_terminal_segments(tree) == 2


def test_removed_queue_result_remains_in_catalog(tmp_path):
    output = tmp_path / "elsewhere" / "run-1"
    output.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    job = JobRecord(
        id="run-1",
        name="Project - run 1",
        settings_path=str(tmp_path / "settings.json"),
        output_dir=str(output),
        status="Completed",
        manifest_path=str(manifest),
    )
    store = QueueStore(tmp_path)
    store.archive_results([job])
    restored = store.load_results()
    assert [(item.id, item.name) for item in restored] == [
        ("run-1", "Project - run 1")
    ]


def test_output_overwrite_defaults_off():
    config = parse_config(example_config())
    assert config.outputs.overwrite is False


def test_studio_rejects_object_fluid_without_type_error(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from cascade.gui.pages.physics import PhysicsPage

    app = QApplication.instance() or QApplication([])
    page = PhysicsPage()
    raw = example_config()
    raw["simulation"]["fluid"] = {"density": 1.0}
    with pytest.raises(ValueError, match="cannot edit unsupported fluid"):
        page.load(raw)
    page.deleteLater()
    app.processEvents()


def test_wsl_auto_renderer_uses_visible_software_canvas(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.delenv("CASCADE_RENDER_BACKEND", raising=False)
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication, QWidget
    from cascade.gui.visualization.canvas import GeometryCanvas
    from cascade.gui.visualization.support import _create_geometry_canvas

    app = QApplication.instance() or QApplication([])
    parent = QWidget()
    canvas = _create_geometry_canvas(parent)
    assert isinstance(canvas, GeometryCanvas)
    assert canvas.renderer_backend == "software-qpaint"
    parent.deleteLater()
    app.processEvents()
