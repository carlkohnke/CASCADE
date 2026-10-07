"""Physical and cache regression checks for large-network acceleration."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from cascade.configuration import solver_state as state


def require_cuda():
    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel("auto") != "gpu":
        pytest.skip("CUDA/CuPy unavailable")


def test_smoothing_bounds_preserve_stability():
    from cascade.flow.gpu import GPUAMGSolver

    solver = object.__new__(GPUAMGSolver)
    solver.smoothing_bound, solver.weight_factor = "positive", 1.8
    rng = np.random.default_rng(15)
    for matrix in (
        sp.diags([-np.ones(19), np.full(20, 2.01), -np.ones(19)], [-1, 0, 1]).tocsr(),
        sp.csr_matrix((lambda x: x.T @ x + np.eye(20))(rng.normal(size=(20, 20)))),
    ):
        d = 1 / np.sqrt(matrix.diagonal())
        rho = np.linalg.eigvalsh(d[:, None] * matrix.toarray() * d[None, :]).max()
        assert 0 < solver._smoothing_weight(matrix) * rho < 2


def test_lambda_bins_do_not_reuse_an_epoch_from_another_solve(monkeypatch):
    from cascade.concentration.external_field.hybrid_geometry import (
        _cext_hybrid_init_lambda_bins,
    )

    monkeypatch.setattr(state, "CEXT_HYBRID_FFT_QUANTILE_BINS", True)
    monkeypatch.setattr(state, "CEXT_HYBRID_FFT_BIN_EPOCH_CACHE", True)
    hybrid = {"lambda_bins": 2, "bg_mode": "fft"}
    first = {
        "lambda_iv_gl": np.array([[0.0001, 0.0002, 0.0004, 0.0008]]),
        "_lambda_bin_epoch": 1,
    }
    edges, _ = _cext_hybrid_init_lambda_bins(hybrid, first)
    repeated, _ = _cext_hybrid_init_lambda_bins(hybrid, first)
    np.testing.assert_array_equal(edges, repeated)
    second = {
        "lambda_iv_gl": np.array([[0.002, 0.004, 0.008, 0.016]]),
        "_lambda_bin_epoch": 1,
    }
    updated, _ = _cext_hybrid_init_lambda_bins(hybrid, second)
    assert updated[-1] > edges[-1] * 10
    second["lambda_iv_gl"] *= 2
    second["_lambda_bin_epoch"] += 1
    fresh, _ = _cext_hybrid_init_lambda_bins(hybrid, second)
    assert fresh[-1] > updated[-1] * 1.9


def test_hierarchy_cache_checks_matrix_options_and_recovers_corruption(
    monkeypatch, tmp_path
):
    pyamg = pytest.importorskip("pyamg")
    from cascade.flow.hierarchy_cache import _path, prepare_hierarchy

    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path))
    a = sp.diags([-np.ones(49), np.full(50, 2.1), -np.ones(49)], [-1, 0, 1]).tocsr()
    options = {"smooth": None, "max_coarse": 8}
    calls = []

    def builder(matrix, **kwargs):
        calls.append(1)
        return pyamg.smoothed_aggregation_solver(matrix, **kwargs)

    def prepare(matrix=a, opts=options):
        return prepare_hierarchy(
            matrix, opts, builder=builder, version=pyamg.__version__, use_cache=True
        )

    first, hit = prepare()
    assert not hit
    cached, hit = prepare()
    assert hit and len(calls) == 1
    for original, restored in zip(first.levels, cached.levels):
        np.testing.assert_array_equal(original.A.toarray(), restored.A.toarray())
    assert not prepare(a * 1.001)[1]
    assert not prepare(opts={**options, "max_coarse": 9})[1]
    _path(a, options, pyamg.__version__).write_bytes(b"invalid cache")
    assert not prepare()[1]
    assert prepare()[1]


@pytest.mark.parametrize("precision", ["float64", "float32"])
@pytest.mark.parametrize("smooth", [None, "jacobi"])
@pytest.mark.parametrize("captured", [True, False])
def test_pressure_preconditioners_and_updates_match_sparse_direct(
    precision, smooth, captured
):
    require_cuda()
    from cascade.flow.gpu import GPUAMGSolver

    size = 150
    rng = np.random.default_rng(4)
    g = 10 ** rng.uniform(-2, 2, size + 1)
    a = sp.diags([-g[1:-1], g[:-1] + g[1:], -g[1:-1]], [-1, 0, 1]).tocsr()
    rhs = np.zeros(size)
    rhs[0] = 1e-4
    solver = GPUAMGSolver(
        a,
        preconditioner_precision=precision,
        smoothing_bound="positive",
        hierarchy_options={"smooth": smooth},
        capture_krylov=captured,
        fused_cycle=captured,
        device_update=captured,
    )
    for updated in (a, a + sp.diags(rng.uniform(0.001, 0.01, size))):
        if updated is not a:
            from cupyx.scipy import sparse as gpu_sparse

            solver.update(gpu_sparse.csr_matrix(updated.tocsr()))
        expected = sp.linalg.spsolve(updated, rhs)
        actual = solver.solve(rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=1e-12)
        assert np.linalg.norm(updated @ actual - rhs) / np.linalg.norm(rhs) < 1e-8


def test_failed_float_preconditioner_recovers_in_double_precision(
    monkeypatch, tmp_path
):
    require_cuda()
    from cascade.flow.gpu import GPUAMGSolver, LAST_GPU_FLOW_TIMINGS

    monkeypatch.setenv("CASCADE_CACHE_DIR", str(tmp_path))
    a = sp.diags([-np.ones(4), np.full(5, 3.0), -np.ones(4)], [-1, 0, 1]).tocsr()
    solver = GPUAMGSolver(a, preconditioner_precision="float32")
    solver.coarse_inverse.fill(0.0)
    rhs = np.arange(1.0, 6.0)
    actual = solver.solve(rhs, max_iterations=16)
    np.testing.assert_allclose(actual, sp.linalg.spsolve(a, rhs), rtol=1e-9)
    assert solver.preconditioner_dtype == state._cp.float64
    assert LAST_GPU_FLOW_TIMINGS["preconditioner_fallback"]


@pytest.mark.parametrize("magnitude", [1e-90, 1e-4, 1e30])
def test_float_preconditioner_preserves_flow_scale(magnitude):
    require_cuda()
    from cascade.flow.gpu import GPUAMGSolver

    a = sp.diags([-np.ones(49), np.full(50, 2.1), -np.ones(49)], [-1, 0, 1]).tocsr()
    rhs = np.zeros(50)
    rhs[0] = magnitude
    solver = GPUAMGSolver(a)
    actual = solver.solve(rhs)
    expected = sp.linalg.spsolve(a, rhs)
    np.testing.assert_allclose(
        actual / magnitude, expected / magnitude, rtol=1e-7, atol=1e-10
    )


@pytest.mark.parametrize(
    "bc", ["terminal_pressure", "legacy_equal_terminal_flow", "pressure"]
)
def test_device_pressure_matches_cpu_and_invalidates_boundaries(monkeypatch, bc):
    require_cuda()
    from cascade.flow.device_pressure import solve_device_pressure
    from cascade.flow.kirchhoff import solve_kirchhoff
    from cascade.flow.linear_system import _solve_pressure_dirichlet

    up, down = np.array([0, 0, 0, 1, 2, 3]), np.array([1, 1, 2, 3, 3, 4])
    monkeypatch.setattr(
        state, "KIRCHHOFF_BC_MODE", bc if bc != "pressure" else "terminal_pressure"
    )
    for resistance, outlets in (
        (np.arange(1.0, 7.0), [4]),
        (np.arange(1.0, 7.0) ** 2, [4]),
        (np.arange(1.0, 7.0), [3, 4]),
    ):
        boundary = (100.0, 10.0) if bc == "pressure" else None
        actual = solve_device_pressure(
            up,
            down,
            state._cp.asarray(resistance),
            [0],
            outlets,
            0.015,
            pressure_boundary=boundary,
        )
        if boundary:
            expected = _solve_pressure_dirichlet(
                up, down, resistance, [0], 100.0, outlets, 10.0
            )
        else:
            expected = solve_kirchhoff(
                up,
                down,
                resistance,
                [0],
                0.015,
                outlets,
                solver_mode="spsolve",
                boundary_condition=bc,
            )[:2]
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(state._cp.asnumpy(a), b, rtol=2e-7, atol=1e-10)


@pytest.mark.parametrize("width", [4, 8, 16, 32])
def test_tissue_thread_tiles_preserve_sources_and_lumen_mask(monkeypatch, width):
    require_cuda()
    from cascade.concentration.tissue import gpu

    monkeypatch.setattr(state, "TISSUE_ACCEL_MODE", "gpu")
    monkeypatch.setattr(state, "TISSUE_GPU_VALIDATE_POINTS", 0)
    monkeypatch.setattr(state, "SOLVER_TIMING_DETAILS", False)
    starts = np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.02, -0.01, 0.0]])
    ends = starts + [0.1, 0.0, 0.0]
    radius = np.full(3, 0.001)
    nodes = np.array([0.1, 0.5, 0.9])
    gl = (
        starts[:, None, :] + (ends - starts)[:, None, :] * nodes[None, :, None]
    ) * state.CM_TO_M
    source = {
        "gl_points_si": gl,
        "lambda_iv_gl": np.full((3, 3), 0.001),
        "q_weighted_gl": np.full((3, 3), 1e-13),
        "mono2_weight_gl": np.full((3, 3), 1e-21),
        "dipole2_weight_gl": np.full((3, 3), 2e-21),
        "seg_cap_gl": np.full(3, 0.14),
        "diffusivity_si": 2.41e-9,
        "window_factor": 6.0,
    }
    points = np.random.default_rng(7).uniform(
        [-0.01, -0.015, -0.015], [0.11, 0.015, 0.015], (137, 3)
    )
    points[:3] = (starts + ends) / 2
    monkeypatch.setattr(gpu, "_CEXT_TISSUE_WARP_MIN_WORK", float("inf"))
    mask, values = gpu._compute_tissue_samples_greens_from_cext_state_gpu(
        points, starts, ends, radius, source
    )
    monkeypatch.setattr(gpu, "_CEXT_TISSUE_WARP_MIN_WORK", 0)
    monkeypatch.setattr(gpu, "_CEXT_TISSUE_GROUP_WIDTH", width)
    actual_mask, actual = gpu._compute_tissue_samples_greens_from_cext_state_gpu(
        points, starts, ends, radius, source
    )
    np.testing.assert_array_equal(actual_mask, mask)
    np.testing.assert_allclose(actual, values, rtol=3e-6, atol=1e-9)
    assert not np.any(mask[:3])


@pytest.mark.parametrize("source_assignment", [0, 1])
@pytest.mark.parametrize("target_assignment", [0, 1])
@pytest.mark.parametrize("moment", range(7))
def test_correlated_self_correction_matches_stencil_pair_oracle(
    source_assignment, target_assignment, moment
):
    require_cuda()
    from cascade.concentration.external_field.hybrid_corrections import (
        _get_self_correlation_kernel,
    )
    from cascade.concentration.external_field.hybrid_geometry import (
        _get_cext_fft_discrete_self_runtime_moment_kernel,
    )

    cp = state._cp
    rng = np.random.default_rng(123)
    points = rng.uniform(-0.2, 8.2, (3, 3, 3)).astype(np.float32)
    points[0, 0] = [-0.1, 0.0, 8.1]
    points[0, 1] = [0.1, 0.2, 7.9]
    weight = rng.normal(size=(3, 3)).astype(np.float32)
    weight[0, 1] = 0.0
    inputs = [
        cp.asarray(np.array([2, 0, 1], np.int32)),
        cp.asarray(points).ravel(),
        cp.asarray(weight).ravel(),
        cp.asarray(
            np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [-1.0, 3.0, 2.0]], np.float32)
        ).ravel(),
        cp.asarray(rng.uniform(0.1, 2.0, (3, 3)).astype(np.float32)).ravel(),
        cp.asarray(np.array([1, 0, 1], np.uint8)),
        cp.asarray(np.array([0.0, 0.8, 3.0], np.float32)),
        cp.asarray(rng.normal(size=(2, 8, 8, 8)).astype(np.float32)).ravel(),
    ]
    scalars = [
        np.int32(moment),
        np.int32(2),
        np.int32(8),
        np.int32(512),
        np.int32(3),
        np.int32(3),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(0.0),
        np.float32(1.0),
        np.int32(target_assignment),
        np.int32(source_assignment),
    ]
    results = []
    for kernel in (
        _get_cext_fft_discrete_self_runtime_moment_kernel(),
        _get_self_correlation_kernel(),
    ):
        out = cp.zeros(9, cp.float32)
        kernel((1,), (128,), (*inputs, *scalars, out))
        results.append(cp.asnumpy(out))
    np.testing.assert_allclose(results[1], results[0], rtol=3e-6, atol=2e-6)
    np.testing.assert_array_equal(results[1][-3:], 0.0)


def test_gpu_cell_reach_matches_cpu_and_refreshes_geometry(monkeypatch):
    require_cuda()
    from cascade.concentration.tissue import gpu

    points = np.random.default_rng(8).uniform(0.0, 0.01, (30, 3, 3)).astype(np.float32)
    lam = np.full((30, 3), 0.001, np.float32)
    radius, lengths = np.full(30, 0.0001), np.full(30, 0.0005)
    kw = dict(window_factor=6.1, radii_si=radius, seg_len_si=lengths)
    monkeypatch.setattr(gpu, "_CEXT_TISSUE_GPU_REACH_MIN_NODES", 1000000)
    reference, _ = gpu._build_cext_tissue_cell_list_gpu(
        points, lam, points.reshape(-1, 3), **kw
    )
    monkeypatch.setattr(gpu, "_CEXT_TISSUE_GPU_REACH_MIN_NODES", 0)
    cache = {}
    actual, metrics = gpu._build_cext_tissue_cell_list_gpu(
        points, lam, points.reshape(-1, 3), tissue_cache=cache, **kw
    )
    for key in ("cell_source_reach_g", "cell_segment_reach_g"):
        np.testing.assert_array_equal(
            state._cp.asnumpy(actual[key]), state._cp.asnumpy(reference[key])
        )
    assert not metrics["structure_reused"]
    radius_changed = dict(kw, radii_si=radius * 2)
    _, metrics = gpu._build_cext_tissue_cell_list_gpu(
        points, lam, points.reshape(-1, 3), tissue_cache=cache, **radius_changed
    )
    assert not metrics["structure_reused"]
    _, metrics = gpu._build_cext_tissue_cell_list_gpu(
        points, lam, points.reshape(-1, 3), tissue_cache=cache, **kw
    )
    assert metrics["structure_reused"]
    changed = points.copy()
    changed[10, 1] += 0.000001
    _, metrics = gpu._build_cext_tissue_cell_list_gpu(
        changed, lam, points.reshape(-1, 3), tissue_cache=cache, **kw
    )
    assert not metrics["structure_reused"]
