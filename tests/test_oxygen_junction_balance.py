"""Physical conservation and backend parity for total-oxygen junctions."""

import numpy as np
import pytest
from scipy.optimize import brentq

from cascade.concentration.vessel import network
from cascade.concentration.vessel.oxygen_transport import (
    concentration_from_content,
    oxygen_content,
    transport_capacity,
)
from cascade.configuration import solver_state as state


@pytest.fixture(autouse=True)
def force_cpu_for_cpu_reference_tests(monkeypatch):
    """These tests patch CPU decay functions; CUDA is covered separately."""
    monkeypatch.setattr(state, "NETWORK_TRANSPORT_ACCEL", "cpu")


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("different_hematocrit", [False, True])
def test_unequal_state_merger_conserves_total_oxygen(
    monkeypatch, sparse, reverse, different_hematocrit
):
    monkeypatch.setattr(state, "_HAVE_SCIPY_SPARSE", sparse)
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "discharge")
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    monkeypatch.setattr(
        network,
        "_blood_greens_decay_factor",
        lambda q, r, length, *a: 0.25 if length == 0.01 else 1.0,
    )
    q = np.array([1.0, 1.0, 2.0])
    hd = (
        np.array([0.2, 0.6, 0.4])
        if different_hematocrit
        else np.full(3, state.HD_DISCHARGE)
    )
    args = {
        "starts": np.zeros((3, 3)),
        "ends": np.ones((3, 3)),
        "radii": np.full(3, 0.01),
        "lengths": np.array([1.0, 2.0, 3.0]),
        "flows": -q if reverse else q,
        "inlet_nodes": [0, 1],
        "outlet_nodes": [3],
        "inlet_concentration": 0.14,
        "fluid": "blood",
        "prox_ids": np.array([2, 2, 3]) if reverse else np.array([0, 1, 2]),
        "dist_ids": np.array([0, 1, 2]) if reverse else np.array([2, 2, 3]),
        "discharge_hematocrit": hd,
        "tol": 1e-11,
        "max_iter": 200,
    }
    cin, cout, nodes, history = network.solve_network_concentrations(**args)
    capacities = hd * state.O2_CAP_PER_HCT
    cap = capacities[2]
    target = (
        oxygen_content(0.035, capacities[0]) + oxygen_content(0.14, capacities[1])
    ) / 2
    expected = brentq(lambda c: oxygen_content(c, cap) - target, 0.035, 0.14)
    assert nodes[2] == pytest.approx(expected, abs=2e-11)
    assert 2 * oxygen_content(cin[2], cap) == pytest.approx(
        np.sum(oxygen_content(cout[:2], capacities[:2])), rel=1e-9
    )
    assert abs(history["MB_resid"][-1]) < 1e-8
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "dissolved")
    _, _, legacy, _ = network.solve_network_concentrations(**args)
    assert legacy[2] == pytest.approx(0.0875, abs=2e-11)
    assert oxygen_content(legacy[2], cap) > target


@pytest.mark.parametrize(
    "content,capacity",
    [(0.0, 9.0), (0.14, 0.0), (0.0001, 9.0), (6.0, 9.0), (30.0, 9.0)],
)
def test_content_inverse(content, capacity):
    c = concentration_from_content(content, capacity, state.ALPHA_MMHG, 0.14)
    assert 0 <= c <= content
    assert float(oxygen_content(c, capacity)) == pytest.approx(content, abs=2e-11)


def test_convective_capacity_uses_discharge_without_changing_tube(monkeypatch):
    hd, ht = np.array([0.45, 0.3]), np.array([0.25, 0.2])
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "discharge")
    np.testing.assert_allclose(transport_capacity(hd, ht), hd * state.O2_CAP_PER_HCT)
    np.testing.assert_array_equal(ht, [0.25, 0.2])


@pytest.mark.parametrize("convection", ["tube", "discharge"])
def test_network_phase_separation_requires_current_cache(monkeypatch, convection):
    from types import SimpleNamespace

    from cascade.concentration.vessel.oxygen_transport import (
        network_discharge_hematocrit,
    )
    from cascade.flow.hematocrit import _store_tree_hematocrit_cache

    monkeypatch.setattr(state, "HEMATOCRIT_MODEL", "pries_secomb")
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", convection)
    tree = SimpleNamespace()
    flows = np.array([2.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="flow-matched"):
        network_discharge_hematocrit(tree, flows)
    hd = np.array([0.4, 0.2, 0.6])
    _store_tree_hematocrit_cache(tree, hd, hd * 0.7, model="pries_secomb", flows=flows)
    np.testing.assert_allclose(network_discharge_hematocrit(tree, flows), hd)
    with pytest.raises(ValueError, match="flow-matched"):
        network_discharge_hematocrit(tree, flows.copy())


@pytest.mark.parametrize("fluid", ["water", "blood"])
def test_hemoglobin_free_limit_preserves_dissolved_solution(monkeypatch, fluid):
    monkeypatch.setattr(state, "HD_DISCHARGE", 0.0)
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "discharge")
    args = {
        "starts": np.zeros((3, 3)),
        "ends": np.ones((3, 3)),
        "radii": np.full(3, 0.01),
        "lengths": np.array([1.0, 2.0, 3.0]),
        "flows": np.array([1.0, 1.0, 2.0]),
        "inlet_nodes": [0, 1],
        "outlet_nodes": [3],
        "inlet_concentration": 0.14,
        "fluid": fluid,
        "prox_ids": np.array([0, 1, 2]),
        "dist_ids": np.array([2, 2, 3]),
        "tol": 1e-10,
        "max_iter": 200,
    }
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "dissolved")
    original = network.solve_network_concentrations(**args)
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    corrected = network.solve_network_concentrations(**args)
    for old, new in zip(original[:3], corrected[:3], strict=True):
        np.testing.assert_allclose(old, new, rtol=1e-10, atol=1e-12)


def test_oxygen_options_validate_and_invalidate_cext_cache(monkeypatch):
    from cascade.concentration.external_field.state import _cext_state_cache_key
    from cascade.configuration.settings import apply_settings

    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "dissolved")
    monkeypatch.setattr(state, "BLOOD_CONVECTIVE_HEMATOCRIT", "tube")
    kw = {
        "fluid_mode": "blood",
        "inlet_concentration": 0.14,
        "diffusivity_si": 2.41e-9,
        "vmax": 0.04,
        "km": 0.0069,
    }
    old = _cext_state_cache_key(**kw)
    apply_settings(
        state,
        {
            "oxygen": {
                "junction_oxygen_balance": "total_content",
                "blood_convective_hematocrit": "discharge",
            }
        },
    )
    assert old != _cext_state_cache_key(**kw)
    with pytest.raises(ValueError, match="JUNCTION_OXYGEN_BALANCE"):
        apply_settings(state, {"oxygen": {"junction_oxygen_balance": "typo"}})


@pytest.mark.parametrize("frontend", ["cli", "gui"])
@pytest.mark.parametrize("omit_options", [False, True])
def test_blood_defaults_resolve_for_cli_and_gui(frontend, omit_options):
    from types import SimpleNamespace

    from cascade.configuration.bridge import apply_runtime_settings
    from cascade.configuration.schema import example_config, parse_config
    from cascade.configuration.settings import default_settings
    from cascade.gui.model import default_project

    raw = example_config() if frontend == "cli" else default_project()
    if omit_options:
        raw["settings"]["oxygen"].pop("junction_oxygen_balance")
        raw["settings"]["oxygen"].pop("blood_convective_hematocrit")
    config = parse_config(raw)
    # A previous legacy run must not leak its settings into a new default run.
    runtime = SimpleNamespace(
        **{
            key: value
            for section in default_settings().values()
            for key, value in section.items()
        }
    )
    runtime.JUNCTION_OXYGEN_BALANCE = "dissolved"
    runtime.BLOOD_CONVECTIVE_HEMATOCRIT = "tube"
    apply_runtime_settings(runtime, config)
    assert runtime.JUNCTION_OXYGEN_BALANCE == "total_content"
    assert runtime.BLOOD_CONVECTIVE_HEMATOCRIT == "discharge"


def test_bulk_coupling_norm_and_nonfinite_guard(monkeypatch):
    from cascade.concentration.external_field.diagnostics import _cext_residual_metrics

    current = np.full(10000, 0.03, dtype=np.float32)
    mapped = current.copy()
    mapped[0] += 0.002
    monkeypatch.setattr(state, "CEXT_VESS_COUPLING_NORM", "rms")
    _, absolute, relative = _cext_residual_metrics(current, mapped)
    assert absolute == pytest.approx(2e-5, rel=1e-5)
    assert absolute < 1e-4
    assert relative < 0.001
    monkeypatch.setattr(state, "CEXT_VESS_COUPLING_NORM", "max")
    _, absolute, _ = _cext_residual_metrics(current, mapped)
    assert absolute == pytest.approx(0.002, rel=1e-5)
    assert absolute > 1e-4
    mapped[1] = np.nan
    assert np.isinf(_cext_residual_metrics(current, mapped)[1])


def test_coupling_norm_setting_invalidates_cache(monkeypatch):
    from cascade.concentration.external_field.state import _cext_state_cache_key
    from cascade.configuration.settings import apply_settings

    monkeypatch.setattr(state, "CEXT_VESS_COUPLING_NORM", "max")
    kw = {
        "fluid_mode": "blood",
        "inlet_concentration": 0.14,
        "diffusivity_si": 2.41e-9,
        "vmax": 0.04,
        "km": 0.0069,
    }
    old = _cext_state_cache_key(**kw)
    apply_settings(state, {"cext": {"vess_coupling_norm": "rms"}})
    assert old != _cext_state_cache_key(**kw)
    with pytest.raises(ValueError, match="CEXT_VESS_COUPLING_NORM"):
        apply_settings(state, {"cext": {"vess_coupling_norm": "typo"}})


@pytest.mark.parametrize("compiled", [False, True])
def test_coupled_merger_uses_same_content_balance(monkeypatch, compiled):
    from cascade.concentration.external_field.coupling_steps import (
        _run_network_ext_frozen_step,
    )

    monkeypatch.setattr(state, "CONC_USE_NUMBA", compiled)
    monkeypatch.setattr(state, "LUMEN_WALL_CLOSURE", "wellmixed")
    monkeypatch.setattr(state, "JUNCTION_OXYGEN_BALANCE", "total_content")
    context = {
        "network_topology": {
            "prox_ids": np.array([0, 1, 2]),
            "dist_ids": np.array([2, 2, 3]),
            "inlet_nodes": [0, 1],
        },
        "flows_si": np.array([1e-9, 1e-9, 2e-9]),
        "radii_si": np.full(3, 1e-5),
        "lengths_si": np.array([0.01, 0.0001, 0.001]),
        "gl_t": np.array([0.5]),
        "diffusivity_si": 2.41e-9,
    }
    ext = {"c_ext_gl": np.zeros((3, 1))}
    cap = np.full(3, 9.0)
    cin, cout, _, _, _ = _run_network_ext_frozen_step(
        context,
        ext,
        inlet_concentration=0.14,
        vmax=0.04,
        km=0.0069,
        chb_max=cap,
        fluid_mode="blood",
    )
    assert cout[0] != pytest.approx(cout[1], abs=1e-5)
    residual = 2 * float(oxygen_content(cin[2], 9.0)) - np.sum(
        oxygen_content(cout[:2], 9.0)
    )
    assert abs(residual) < 3e-5
