from __future__ import annotations

import numpy as np
import pytest

from cascade.configuration import parse_config
from cascade.domain.workflow import build_domain


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"domain": {"use_cache": "flase"}}, "Unrecognized boolean"),
        ({"domain": {"side_length": -1}}, "side_length must be positive"),
        ({"domain": {"center": [0, float("nan"), 0]}}, "must be finite"),
        ({"network": {"target_terminal_count": 3.9}}, "must be an integer"),
        ({"simulation": {"distance_sample_count": 2.9}}, "must be an integer"),
        ({"simulation": {"distance_sample_count": -1}}, "must be non-negative"),
        ({"simulation": {"qin_target_ul_min": 0}}, "must be positive"),
        ({"network": {"physical_clearance": -0.1}}, "must be non-negative"),
    ],
)
def test_invalid_scientific_configuration_fails_during_parsing(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_config(raw)


class _MeshCapturingRuntime:
    def build_domain(self, *args, mesh=None, random_seed=None):
        assert random_seed is not None
        return mesh if mesh is not None else args[0]


@pytest.mark.parametrize(
    ("kind", "extra", "expected_bounds"),
    [
        ("cube", {"side_length": 2.0}, (2.0, 4.0, -1.0, 1.0, 4.0, 6.0)),
        (
            "box",
            {"x_length": 2.0, "y_length": 4.0, "z_length": 6.0},
            (2.0, 4.0, -2.0, 2.0, 2.0, 8.0),
        ),
    ],
)
def test_center_is_applied_to_box_domains(kind, extra, expected_bounds):
    config = parse_config(
        {"domain": {"type": kind, "center": [3.0, 0.0, 5.0], **extra}}
    )
    mesh = build_domain(config, ts=_MeshCapturingRuntime())
    np.testing.assert_allclose(mesh.bounds, expected_bounds)


def test_cylinder_uses_configured_angular_resolution():
    low = parse_config(
        {"domain": {"type": "cylinder", "theta_resolution": 8}}
    )
    high = parse_config(
        {"domain": {"type": "cylinder", "theta_resolution": 24}}
    )
    runtime = _MeshCapturingRuntime()

    low_mesh = build_domain(low, ts=runtime)
    high_mesh = build_domain(high, ts=runtime)

    assert high_mesh.n_points > low_mesh.n_points


@pytest.mark.parametrize("frontend", ["cli", "gui"])
def test_acceleration_overrides_do_not_leak_between_serial_runs(frontend):
    from types import SimpleNamespace

    from cascade.configuration.bridge import apply_runtime_settings
    from cascade.configuration.schema import example_config
    from cascade.configuration.settings import default_settings
    from cascade.configuration.settings import cext
    from cascade.gui.model import default_project

    make_config = example_config if frontend == "cli" else default_project
    names = (
        "hybrid_gpu_iteration_cache",
        "hybrid_gpu_runtime_stencil",
        "hybrid_gpu_runtime_moments",
    )
    runtime = SimpleNamespace(
        **{
            key: value
            for section in default_settings().values()
            for key, value in section.items()
        }
    )
    overrides = make_config()
    overrides["settings"]["cext"].update({name: False for name in names})
    apply_runtime_settings(runtime, parse_config(overrides))
    assert all(getattr(runtime, cext.ALIASES[name]) is False for name in names)
    apply_runtime_settings(runtime, parse_config(make_config()))
    for name in names:
        constant = cext.ALIASES[name]
        assert getattr(runtime, constant) == cext.DEFAULTS[constant]


def _defaults_runtime():
    from types import SimpleNamespace

    from cascade.configuration.settings import default_settings

    return SimpleNamespace(
        **{
            key: value
            for section in default_settings().values()
            for key, value in section.items()
        }
    )


def _applied_inlet_flow(raw):
    from cascade.configuration.bridge import apply_runtime_settings

    config = parse_config(raw)
    runtime = _defaults_runtime()
    apply_runtime_settings(runtime, config)
    return config, runtime


@pytest.mark.parametrize(
    "alias",
    [
        {"settings": {"hemodynamics": {"qin_target": 5.0}}},
        {"settings": {"hemodynamics": {"QIN_TARGET": 5.0}}},
        {"settings": {"hemodynamics": {"qin_target_ul_min": 5.0}}},
        {"settings": {"flow": {"qin_target": 5.0}}},
        {"settings": {"runtime": {"QIN_TARGET": 5.0}}},
        {"simulation": {"tissuesim": {"qin_target": 5.0}}},
    ],
)
def test_deprecated_inlet_flow_alias_alone_sets_the_inlet_flow(alias):
    config, runtime = _applied_inlet_flow(alias)
    assert config.simulation.qin_target_ul_min == 5.0
    assert runtime.QIN_TARGET == 5.0
    assert any(
        "Deprecated" in warning and "simulation.qin_target_ul_min" in warning
        for warning in config.runtime_setting_warnings
    )


def test_deprecated_inlet_flow_alias_equal_to_simulation_flow_warns():
    config, runtime = _applied_inlet_flow(
        {
            "simulation": {"qin_target_ul_min": 5.0},
            "settings": {"hemodynamics": {"qin_target": 5}},
        }
    )
    assert config.simulation.qin_target_ul_min == 5.0
    assert runtime.QIN_TARGET == 5.0
    assert any("qin_target" in w for w in config.runtime_setting_warnings)


def test_deprecated_inlet_flow_alias_conflict_is_a_configuration_error():
    with pytest.raises(ValueError) as error:
        parse_config(
            {
                "simulation": {"qin_target_ul_min": 100.0},
                "settings": {"hemodynamics": {"qin_target": 5.0}},
            }
        )
    message = str(error.value)
    assert "simulation.qin_target_ul_min=100" in message
    assert "settings.hemodynamics.qin_target=5" in message


def test_conflicting_deprecated_inlet_flow_aliases_are_rejected():
    with pytest.raises(ValueError, match="settings.flow.qin_target=6"):
        parse_config(
            {
                "settings": {
                    "hemodynamics": {"qin_target": 5.0},
                    "flow": {"qin_target": 6.0},
                }
            }
        )


@pytest.mark.parametrize("frontend", ["cli", "gui"])
def test_runtime_inlet_flow_defaults_to_simulation_flow(frontend):
    from cascade.configuration.schema import example_config
    from cascade.gui.model import default_project

    raw = example_config() if frontend == "cli" else default_project()
    raw["simulation"].pop("qin_target_ul_min")
    config, runtime = _applied_inlet_flow(raw)
    assert config.simulation.qin_target_ul_min == 100.0
    assert runtime.QIN_TARGET == config.simulation.qin_target_ul_min
    assert not any("qin_target" in w for w in config.runtime_setting_warnings)


def test_runtime_inlet_flow_follows_simulation_flow_between_serial_runs():
    runtime = _defaults_runtime()
    from cascade.configuration.bridge import apply_runtime_settings

    apply_runtime_settings(
        runtime, parse_config({"settings": {"hemodynamics": {"qin_target": 5.0}}})
    )
    assert runtime.QIN_TARGET == 5.0
    apply_runtime_settings(runtime, parse_config({}))
    assert runtime.QIN_TARGET == 100.0


def test_studio_default_project_omits_deprecated_inlet_flow_alias():
    from cascade.configuration.parsing import inlet_flow_aliases
    from cascade.gui.model import default_project

    project = default_project()
    assert inlet_flow_aliases(project) == []
    assert "qin_target" not in project["settings"].get("hemodynamics", {})
    assert project["simulation"]["qin_target_ul_min"] == 100.0


def test_studio_loads_old_project_with_deprecated_inlet_flow_alias(tmp_path):
    import json

    from cascade.gui.model import load_project, save_project, validate_project

    path = tmp_path / "old.cascade-project.json"
    path.write_text(
        json.dumps(
            {
                "simulation": {"fluid": "blood"},
                "settings": {"hemodynamics": {"qin_target": 42.0}},
            }
        ),
        encoding="utf-8",
    )
    project = load_project(path)
    assert project["simulation"]["qin_target_ul_min"] == 42.0
    assert "qin_target" not in project["settings"]["hemodynamics"]
    assert parse_config(project).simulation.qin_target_ul_min == 42.0
    assert not validate_project(project, tmp_path).errors

    saved = json.loads(save_project(path, project).read_text(encoding="utf-8"))
    assert "qin_target" not in saved["settings"]["hemodynamics"]


def test_studio_reports_old_project_with_conflicting_inlet_flow_alias(tmp_path):
    import json

    from cascade.gui.model import load_project, validate_project

    path = tmp_path / "old.cascade-project.json"
    path.write_text(
        json.dumps(
            {
                "simulation": {"qin_target_ul_min": 100.0},
                "settings": {"hemodynamics": {"qin_target": 5.0}},
            }
        ),
        encoding="utf-8",
    )
    project = load_project(path)
    errors = validate_project(project, tmp_path).errors
    assert any("settings.hemodynamics.qin_target=5" in error for error in errors)


def test_sweep_rejects_flows_that_conflict_with_deprecated_alias(tmp_path):
    import json

    from cascade.simulation.sweep import run_sweep

    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "settings": {"hemodynamics": {"qin_target": 5.0}},
                "sweep": {"qin_target_ul_min_values": [5.0, 50.0]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conflicts with the sweep inlet flow"):
        run_sweep(path)


@pytest.mark.parametrize(
    "simulation",
    [
        {"kirchhoff_bc_mode": "pressure_pressure"},
        {"tissuesim": {"kirchhoff_bc_mode": "pressure_pressure"}},
    ],
)
def test_simulation_override_shadowed_by_named_setting_is_rejected(simulation):
    raw = {
        "simulation": simulation,
        "settings": {"kirchhoff": {"bc_mode": "legacy_equal_terminal_flow"}},
    }
    with pytest.raises(ValueError, match="settings.kirchhoff.bc_mode"):
        parse_config(raw)


def test_matching_simulation_override_and_named_setting_are_accepted():
    from cascade.configuration.runtime import RuntimeConfiguration
    from cascade.runtime import tissuesim

    config = parse_config(
        {
            "simulation": {"kirchhoff_bc_mode": "pressure-pressure"},
            "settings": {"kirchhoff": {"bc_mode": "pressure_pressure"}},
        }
    )
    RuntimeConfiguration.from_run_config(config).apply_compatibility_state(tissuesim)
    assert tissuesim.KIRCHHOFF_BC_MODE == "pressure_pressure"
