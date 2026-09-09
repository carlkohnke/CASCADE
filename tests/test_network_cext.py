from __future__ import annotations

import numpy as np
import pytest

from cascade.runtime import tissuesim as ts


def test_cext_sources_are_reweighted_for_independent_tissue_quadrature():
    state = {
        "gl_points_si": np.asarray([[[0.5, 0.0, 0.0]]], dtype=np.float32),
        "segment_vectors": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "q_line_gl": np.asarray([[2.0]], dtype=np.float32),
        "q_weighted_gl": np.asarray([[2.0]], dtype=np.float32),
        "mono2_weight_gl": np.asarray([[0.5]], dtype=np.float32),
        "dipole2_weight_gl": np.asarray([[0.25]], dtype=np.float32),
        "lambda_iv_gl": np.asarray([[0.1]], dtype=np.float32),
        "c_iv_gl": np.asarray([[0.14]], dtype=np.float32),
        "c_bulk_gl": np.asarray([[0.13]], dtype=np.float32),
        "c_wall_gl": np.asarray([[0.12]], dtype=np.float32),
        "c_ext_gl": np.asarray([[0.02]], dtype=np.float32),
    }

    resampled = ts._resample_cext_source_state_for_tissue(state, 5)

    assert resampled["gl_points_si"].shape == (1, 5, 3)
    assert resampled["q_weighted_gl"].shape == (1, 5)
    assert np.sum(resampled["q_weighted_gl"]) == pytest.approx(2.0)
    assert np.sum(resampled["mono2_weight_gl"]) == pytest.approx(0.5)
    assert np.sum(resampled["dipole2_weight_gl"]) == pytest.approx(0.25)
    assert resampled["tissue_gl_order"] == 5
    assert resampled["cext_gl_order"] == 1


def test_network_fft_solver_identifier_is_accepted():
    assert ts._resolve_concentration_solver("network_ext_hybrid_bg") == "network_ext_hybrid_bg"


def test_network_direct_solver_identifier_is_accepted():
    assert ts._resolve_concentration_solver("network_ext") == "network_ext"


def test_network_direct_solver_runs_the_coupled_graph_path(monkeypatch):
    monkeypatch.setattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")
    monkeypatch.setattr(ts, "CEXT_ACCEL_MODE", "cpu")
    monkeypatch.setattr(ts, "CEXT_FROZEN_ACCEL_MODE", "cpu")
    monkeypatch.setattr(ts, "CEXT_VESS_COUPLING_MAX_ITER", 1)
    monkeypatch.setattr(ts, "GL_ORDER_CEXT", 2)
    starts = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
    ends = np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float)

    cin, cout = ts._solve_channel_concentrations(
        None,
        np.asarray([1e-6, 1e-6], dtype=float),
        (0,),
        (2,),
        starts,
        ends,
        np.asarray([0.005, 0.005], dtype=float),
        np.asarray([0.1, 0.1], dtype=float),
        prox_ids=np.asarray([0, 1], dtype=np.int64),
        dist_ids=np.asarray([1, 2], dtype=np.int64),
        inlet_concentration=0.14,
        diffusivity=2.41e-5,
        vmax=0.001,
        km=0.0069,
        fluid="water",
        solver="network_ext",
    )

    assert cin.shape == cout.shape == (2,)
    assert np.all(np.isfinite(cin))
    assert np.all(cout <= cin + 1e-7)
    assert ts._LAST_CEXT_SOURCE_STATE["solver"] == "network_ext"


def test_network_frozen_step_balances_a_looped_graph(monkeypatch):
    monkeypatch.setattr(ts, "LUMEN_WALL_CLOSURE", "wellmixed")
    prox = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    dist = np.asarray([1, 2, 3, 3, 4], dtype=np.int64)
    flows_si = np.asarray([1.0, 1.0, 1.0, 1.0, 2.0], dtype=float) * 1e-12
    context = {
        "network_topology": {
            "prox_ids": prox,
            "dist_ids": dist,
            "inlet_nodes": (0,),
            "outlet_nodes": (4,),
        },
        "flows_si": flows_si,
        "radii_si": np.full((5,), 6e-6, dtype=float),
        "lengths_si": np.full((5,), 2e-4, dtype=float),
        "gl_t": np.asarray([0.25, 0.75], dtype=float),
        "diffusivity_si": 2.41e-9,
    }
    state = {
        "c_ext_gl": np.zeros((5, 2), dtype=np.float32),
        "cin_seg": np.full((5,), 0.1, dtype=np.float32),
    }

    cin, cout, civ, backend, _ = ts._run_network_ext_frozen_step(
        context,
        state,
        inlet_concentration=0.1,
        vmax=0.04,
        km=0.0069,
        chb_max=np.zeros((5,), dtype=float),
        fluid_mode="water",
    )

    assert backend == "cpu-network"
    assert cin.shape == cout.shape == (5,)
    assert civ.shape == (5, 2)
    assert np.all(np.isfinite(cout))
    assert np.all(cout <= cin + 1e-7)
    assert np.isclose(cin[4], 0.5 * (cout[2] + cout[3]), rtol=2e-3, atol=1e-7)
