"""Treecode Cext accuracy against a direct sum of the same regularised kernel."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.concentration.external_field import treecode
from cascade.configuration import solver_state as state

DIFFUSIVITY_SI = 2.0e-9
N_BINS = 8


def cuda():
    from cascade.concentration.vessel.network_gpu import resolve_network_accel

    if resolve_network_accel("auto") != "gpu":
        pytest.skip("CUDA/CuPy unavailable")


def _case(seed: int = 7, nseg: int = 1500, gl_order: int = 2, frozen: float = 0.0):
    rng = np.random.default_rng(seed)
    domain = 1.0e-3
    mid = rng.uniform(0.0, domain, (nseg, 3))
    mid[: nseg // 3] = 0.5 * domain + 0.08 * domain * rng.standard_normal(
        (nseg // 3, 3)
    )
    dirs = rng.standard_normal((nseg, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    offsets = 4.0e-6 * np.linspace(-0.5, 0.5, gl_order)
    points = mid[:, None, :] + offsets[None, :, None] * dirs[:, None, :]
    q = rng.uniform(0.2, 1.0, (nseg, gl_order)).astype(np.float32)
    # Put every source lambda on a bin centre so binning adds no error; two
    # zero-strength anchors pin the bin range.
    lam_min, lam_max = np.float32(0.05 * domain), np.float32(2.0 * domain)
    edges = np.geomspace(float(lam_min), float(lam_max), N_BINS + 1).astype(np.float32)
    centers = np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)
    lam = centers[rng.integers(0, N_BINS, q.shape)]
    lam[0, 0], q[0, 0] = lam_min, 0.0
    lam[1, 0], q[1, 0] = lam_max, 0.0
    return {
        "points": points.astype(np.float32),
        "radii": np.full((nseg,), 2.0e-6, dtype=np.float32),
        "q": q,
        "lam": lam.astype(np.float32),
        "active": rng.random(nseg) >= frozen,
    }


def _direct(case: dict) -> np.ndarray:
    nseg, gl_order, _ = case["points"].shape
    pts = case["points"].reshape(-1, 3).astype(np.float64)
    seg = np.repeat(np.arange(nseg), gl_order)
    q = case["q"].reshape(-1).astype(np.float64)
    lam = case["lam"].reshape(-1).astype(np.float64)
    radii = case["radii"].astype(np.float64)
    out = np.zeros(pts.shape[0])
    for lo in range(0, pts.shape[0], 1024):
        hi = min(lo + 1024, pts.shape[0])
        d2 = np.sum((pts[lo:hi, None, :] - pts[None, :, :]) ** 2, axis=-1)
        radius_sum = radii[seg[lo:hi]][:, None] + radii[seg][None, :]
        r = np.sqrt(d2 + radius_sum**2)
        g = q * np.exp(-r / lam) / (4.0 * np.pi * DIFFUSIVITY_SI * r)
        g[seg[lo:hi][:, None] == seg[None, :]] = 0.0
        out[lo:hi] = g.sum(axis=1)
    return out.reshape(nseg, gl_order)


def _treecode(monkeypatch, case: dict, *, theta: float, order: int, active=None):
    cp = state._cp
    monkeypatch.setattr(state, "CEXT_TREECODE_THETA", theta)
    monkeypatch.setattr(state, "CEXT_TREECODE_ORDER", order)
    monkeypatch.setattr(state, "CEXT_TREECODE_LEAF_NODES", 16)
    monkeypatch.setattr(state, "CEXT_TREECODE_LAMBDA_BINS", N_BINS)
    monkeypatch.setattr(state, "CEXT_TREECODE_NEAR_RADIUS_MULT", 4.0)
    # Every lambda window spans the domain, so the direct sum has no cutoff.
    monkeypatch.setattr(state, "CEXT_WINDOW_FACTOR", 1.0e6)
    nseg = case["points"].shape[0]
    exclude_idx = np.arange(nseg, dtype=np.int32)[:, None]
    context = {
        "gl_points_si": case["points"],
        "max_reach_si": 0.0,
        "exclude_idx": exclude_idx,
        "diffusivity_si": DIFFUSIVITY_SI,
        "gpu_static": {
            "gl_points_si": cp.asarray(case["points"]),
            "radii_si": cp.asarray(case["radii"]),
            "exclude_idx": cp.asarray(exclude_idx),
            "exclude_count": cp.ones((nseg,), dtype=cp.uint8),
            "all_target_seg_ids": cp.arange(nseg, dtype=cp.int32),
        },
    }
    tree = treecode._build_cext_treecode_context(context)
    leaf_scale = float(tree["side"]) / float(1 << int(tree["max_depth"]))
    tree["near_radius_si"] = 4.0 * leaf_scale
    ext_state = {
        "q_weighted_gl": case["q"],
        "lambda_iv_gl": case["lam"],
        "treecode_active_source_mask": case["active"] if active is None else active,
    }
    out, _ = treecode._compute_cext_treecode_gpu(context, tree, ext_state)
    return np.asarray(out, dtype=np.float64), tree


def _rel_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(value - reference)) / np.max(np.abs(reference)))


def test_treecode_boxes_contain_their_sources(monkeypatch):
    monkeypatch.setattr(state, "CEXT_TREECODE_LEAF_NODES", 16)
    tree = treecode._build_cext_treecode_context(
        {"gl_points_si": _case()["points"], "max_reach_si": 0.0}
    )
    points = tree["sorted_points"]
    for node in range(int(tree["node_count"])):
        lo, hi = int(tree["node_start"][node]), int(tree["node_end"][node])
        offset = np.abs(points[lo:hi] - tree["node_center"][node])
        assert np.all(offset <= tree["node_half"][node] * (1.0 + 1.0e-5)), node


def test_treecode_dipole_improves_on_monopole(monkeypatch):
    cuda()
    case = _case()
    reference = _direct(case)
    monopole, _ = _treecode(monkeypatch, case, theta=0.5, order=0)
    dipole, _ = _treecode(monkeypatch, case, theta=0.5, order=1)
    err0 = _rel_error(monopole, reference)
    err1 = _rel_error(dipole, reference)
    assert err1 < 0.5 * err0, (err0, err1)
    assert err1 < 5.0e-3, err1


def test_treecode_matches_direct_sum_at_small_theta(monkeypatch):
    cuda()
    case = _case()
    reference = _direct(case)
    result, tree = _treecode(monkeypatch, case, theta=0.2, order=1)
    assert _rel_error(result, reference) < 1.0e-3
    # Root moments must be taken about the root centre after the upsweep.
    state_gpu = tree["runtime_state"]
    mass = state._cp.asnumpy(state_gpu["node_total_mass"])[:, 0]
    dipole = state._cp.asnumpy(state_gpu["node_total_dipole"])[:, 0, :]
    q = case["q"].reshape(-1).astype(np.float64)
    arm = case["points"].reshape(-1, 3) - tree["node_center"][0]
    np.testing.assert_allclose(mass.sum(), q.sum(), rtol=1.0e-5)
    np.testing.assert_allclose(
        dipole.sum(axis=0), (q[:, None] * arm).sum(axis=0), rtol=1.0e-3, atol=1.0e-9
    )


@pytest.mark.parametrize("frozen", [0.4, 0.95])
def test_treecode_counts_frozen_sources_once(monkeypatch, frozen):
    cuda()
    case = _case(frozen=frozen)
    assert 0 < np.count_nonzero(~case["active"]) < case["active"].size
    reference = _direct(case)
    mixed, _ = _treecode(monkeypatch, case, theta=0.2, order=1)
    all_active, _ = _treecode(
        monkeypatch, case, theta=0.2, order=1, active=np.ones_like(case["active"])
    )
    assert _rel_error(mixed, reference) < 1.0e-3
    assert _rel_error(mixed, all_active) < 1.0e-3
