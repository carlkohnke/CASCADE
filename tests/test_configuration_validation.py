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
