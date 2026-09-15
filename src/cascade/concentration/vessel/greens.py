"""Green's-function oxygen kinetics and finite-radius corrections.

Numerical kernels and packaged Bessel lookup data live together here, without
depending on import order or implicit namespace injection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.flow.rheology import tube_hematocrit

try:
    from scipy.special import k0 as _bessel_k0, k1 as _bessel_k1, kve as _bessel_kve
except ImportError:  # pragma: no cover - approximation paths remain available
    _bessel_k0 = _bessel_k1 = _bessel_kve = None

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - scalar Python paths remain available

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate

    prange = range


def profile(func):
    """No-op hook retained for line-profiler compatibility."""
    return func


def _build_k_ratio_lut(
    xmin: float = 1e-12,
    xmax: float = 5e2,
    n: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an in-memory reference table for validation tooling.

    Production code loads the denser, versioned package asset below.  The
    exponentially scaled Bessel functions are essential here: direct K1/K0
    underflows for large arguments even though the ratio remains well behaved.
    """
    xs = np.exp(np.linspace(np.log(xmin), np.log(xmax), n))
    if _state._HAVE_SCIPY:
        k0_vals = _bessel_k0(xs)
        k1_vals = _bessel_k1(xs)
        ratio = _bessel_kve(1, xs) / _bessel_kve(0, xs)
    else:
        gamma = 0.5772156649015329
        small = xs < 1e-2
        large = xs > 8.0
        invphi = 1.0 / np.maximum(xs, 1e-30)
        r_small = invphi / np.maximum(-(np.log(xs / 2.0) + gamma), 1e-8)
        r_large = 1.0 + 0.5 * invphi + 0.375 * (invphi**2)
        r_mid = (1.0 + 0.5658 * xs + 0.1373 * xs * xs) / (
            1.0 + 1.0361 * xs + 0.5454 * xs * xs
        )
        ratio = np.where(small, r_small, np.where(large, r_large, r_mid))
        k0_vals = np.empty_like(xs)
        k1_vals = ratio * 0.0
    return xs, k0_vals, k1_vals, ratio


def _load_k_ratio_lut(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["format_version"]).item())
            xs = np.asarray(data["xs"], dtype=np.float64).copy()
            k0_vals = np.asarray(data["k0"], dtype=np.float64).copy()
            k1_vals = np.asarray(data["k1"], dtype=np.float64).copy()
            ratio = np.asarray(data["ratio"], dtype=np.float64).copy()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load the packaged Bessel table at {path}: {exc}"
        ) from exc
    if version != 1:
        raise RuntimeError(f"Unsupported Bessel table format {version} at {path}.")
    expected_shape = xs.shape
    if (
        xs.ndim != 1
        or xs.size < 2
        or any(values.shape != expected_shape for values in (k0_vals, k1_vals, ratio))
    ):
        raise RuntimeError(f"Invalid Bessel table array shapes at {path}.")
    if not np.all(np.isfinite(xs)) or not np.all(np.diff(xs) > 0.0):
        raise RuntimeError(
            f"Bessel table arguments must be finite and strictly increasing at {path}."
        )
    if not all(
        np.all(np.isfinite(values)) and np.all(values > 0.0)
        for values in (k0_vals, k1_vals, ratio)
    ):
        raise RuntimeError(
            f"Bessel table values must be finite and positive at {path}."
        )
    return xs, k0_vals, k1_vals, ratio


_state._KRATIO_LUT = _load_k_ratio_lut(_state.KRATIO_LUT_PATH)
_state._KRATIO_XS, _state._K0_LUT, _state._K1_LUT, _state._KRATIO_YS = (
    _state._KRATIO_LUT
)


def _k0est(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf)
    long_est = np.sqrt(np.pi / (2.0 * x) * np.exp(-x))
    short_est = -np.log(x / 2.0) - 0.5772 + ((x**2) / 4.0) * (np.log(x / 2.0) + 0.0772)
    asymp = 0.5 * (1.0 - vec_erf(3.0 * (x - 0.5)))
    return asymp * short_est + (1.0 - asymp) * long_est


@profile
def _k_ratio(phi: np.ndarray | float) -> np.ndarray:
    phi_arr = np.asarray(phi, dtype=float)
    phi_arr = np.maximum(phi_arr, 1e-12)
    flat = phi_arr.ravel()
    out = np.interp(
        flat,
        _state._KRATIO_XS,
        _state._KRATIO_YS,
        left=_state._KRATIO_YS[0],
        right=_state._KRATIO_YS[-1],
    )
    return out.reshape(phi_arr.shape)


def _k0_lookup(x: np.ndarray) -> np.ndarray:
    if _state._K0_LUT.size:
        flat = np.asarray(x, dtype=float).ravel()
        out = np.interp(
            flat,
            _state._KRATIO_XS,
            _state._K0_LUT,
            left=_state._K0_LUT[0],
            right=_state._K0_LUT[-1],
        )
        return out.reshape(np.asarray(x).shape)
    if _state._HAVE_SCIPY:
        return _bessel_k0(x)
    return _k0est(x)


def _k1_lookup(x: np.ndarray) -> np.ndarray:
    if _state._K1_LUT.size:
        flat = np.asarray(x, dtype=float).ravel()
        out = np.interp(
            flat,
            _state._KRATIO_XS,
            _state._K1_LUT,
            left=_state._K1_LUT[0],
            right=_state._K1_LUT[-1],
        )
        return out.reshape(np.asarray(x).shape)
    if _state._HAVE_SCIPY:
        return _bessel_k1(x)
    return _k0est(x) * 0.0


if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _interp_scalar(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
        if x <= xs[0]:
            return ys[0]
        if x >= xs[xs.shape[0] - 1]:
            return ys[ys.shape[0] - 1]
        idx = np.searchsorted(xs, x) - 1
        if idx < 0:
            idx = 0
        if idx >= xs.shape[0] - 1:
            idx = xs.shape[0] - 2
        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = ys[idx]
        y1 = ys[idx + 1]
        if x1 == x0:
            return y0
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @njit(cache=True)
    def _k0_lookup_numba(
        x_arr: np.ndarray, xs: np.ndarray, k0_lut: np.ndarray
    ) -> np.ndarray:
        out = np.empty_like(x_arr)
        for i in range(x_arr.size):
            out[i] = _interp_scalar(x_arr[i], xs, k0_lut)
        return out

    @njit(cache=True)
    def _k_ratio_scalar_numba(phi: float) -> float:
        x = phi
        if x < 1e-12:
            x = 1e-12
        return _interp_scalar(x, _state._KRATIO_XS, _state._KRATIO_YS)

    @njit(cache=True)
    def _lambda_if_scalar_numba(
        c_iv: float, diffusivity_si: float, vmax: float, km: float
    ) -> float:
        c_local = c_iv if c_iv > 0.0 else 0.0
        denom = km + c_local
        if denom < 1e-30:
            denom = 1e-30
        return np.sqrt(diffusivity_si / max(vmax / denom, 1e-30))

    @njit(cache=True)
    def _interfacial_transfer_coeff_scalar_numba(
        radius_si: float,
        lambda_if: float,
        diffusivity_si: float,
        xs_lut: np.ndarray,
        ys_lut: np.ndarray,
    ) -> float:
        lambda_local = max(lambda_if, 1e-30)
        phi = radius_si / lambda_local
        if phi < 1e-12:
            phi = 1e-12
        ratio = _interp_scalar(phi, xs_lut, ys_lut)
        return (2.0 * np.pi * radius_si) * (diffusivity_si / lambda_local) * ratio

    @njit(cache=True)
    def _greens_decay_ratio_numba(
        flow: float,
        radius: float,
        length: float,
        diffusivity: float,
        vmax: float,
        km: float,
        cin: float,
    ) -> float:
        if length <= 0.0 or radius <= 0.0:
            return 1.0
        flow_mag = abs(flow)
        if flow_mag <= 0.0:
            return 1.0
        cin_pos = cin if cin > 0.0 else 0.0
        denom = km + cin_pos
        if denom < 1e-30:
            denom = 1e-30
        k1 = vmax / denom
        if k1 < 1e-30:
            k1 = 1e-30
        lam = (diffusivity / k1) ** 0.5
        if lam < 1e-30:
            lam = 1e-30
        phi = radius / lam
        ratio = _k_ratio_scalar_numba(phi)
        beta = (
            (2.0 * np.pi * radius / (flow_mag if flow_mag > 1e-30 else 1e-30))
            * (diffusivity / lam)
            * ratio
        )
        exponent = -beta * length
        if exponent < -150.0:
            exponent = -150.0
        elif exponent > 50.0:
            exponent = 50.0
        return np.exp(exponent)

    @njit(cache=True)
    def _severinghaus_dSdP_scalar(P: float) -> float:
        den = P * P * P + 150.0 * P + 23400.0
        return 70200.0 * (P * P + 50.0) / (den * den)

    @njit(cache=True)
    def _blood_greens_decay_ratio_numba(
        flow: float,
        radius: float,
        length: float,
        diffusivity: float,
        vmax: float,
        km: float,
        cin: float,
        ccap: float,
        steps: int,
    ) -> float:
        if length <= 0.0 or radius <= 0.0:
            return 1.0
        flow_mag = abs(flow)
        if flow_mag <= 0.0:
            return 1.0
        cin_pos = cin if cin > 0.0 else 0.0
        denom = km + cin_pos
        if denom < 1e-30:
            denom = 1e-30
        k1 = vmax / denom
        if k1 < 1e-30:
            k1 = 1e-30
        lam = (diffusivity / k1) ** 0.5
        if lam < 1e-30:
            lam = 1e-30
        phi = radius / lam
        ratio = _k_ratio_scalar_numba(phi)
        base = (
            (2.0 * np.pi * radius / (flow_mag if flow_mag > 1e-30 else 1e-30))
            * (diffusivity / lam)
            * ratio
        )
        if steps < 1:
            steps = 1
        ds = length / steps
        c_local = cin_pos
        cap = ccap if ccap > 0.0 else 0.0
        for _ in range(steps):
            P_mmHg = c_local / _state.ALPHA_MMHG
            buffer = 1.0 + cap * _severinghaus_dSdP_scalar(P_mmHg) / _state.ALPHA_MMHG
            if buffer < 1e-30:
                buffer = 1e-30
            beta_local = base / buffer
            exponent = -beta_local * ds
            if exponent < -150.0:
                exponent = -150.0
            elif exponent > 50.0:
                exponent = 50.0
            c_local *= np.exp(exponent)
        denom_cin = cin if cin > 1e-30 else 1e-30
        return c_local / denom_cin

    @njit(cache=True)
    def _solve_channel_concentrations_topdown_numba(
        order: np.ndarray,
        parents: np.ndarray,
        flows_si: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        inlet_concentration: float,
        chb_max: np.ndarray,
        is_blood: bool,
        axial_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        nseg = parents.shape[0]
        cin = np.empty(nseg, dtype=np.float64)
        cout = np.empty(nseg, dtype=np.float64)
        for i in range(nseg):
            cin[i] = np.nan
            cout[i] = np.nan
        for i in range(nseg):
            if parents[i] < 0:
                cin[i] = inlet_concentration
        for oi in range(order.shape[0]):
            idx = int(order[oi])
            if idx < 0 or idx >= nseg:
                continue
            if not np.isfinite(cin[idx]):
                p = int(parents[idx])
                if p >= 0 and p < nseg and np.isfinite(cout[p]):
                    cin[idx] = cout[p]
                else:
                    cin[idx] = inlet_concentration
            cin_local = cin[idx]
            if not np.isfinite(cin_local) or cin_local < 0.0:
                cin_local = 0.0
            cin[idx] = cin_local
            if is_blood:
                decay = _blood_greens_decay_ratio_numba(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                    chb_max[idx],
                    axial_steps,
                )
            else:
                decay = _greens_decay_ratio_numba(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                )
            cout[idx] = cin_local * decay
        return cin, cout

    @njit(cache=True, parallel=True)
    def _tissue_kernel_numba(
        points_si: np.ndarray,
        starts_si: np.ndarray,
        segment_vectors: np.ndarray,
        seg_len: np.ndarray,
        radii_si: np.ndarray,
        nearest_idx: np.ndarray,
        proj_raw: np.ndarray,
        d_center: np.ndarray,
        keep_mask: np.ndarray,
        cin_pos: np.ndarray,
        alpha_edge: np.ndarray,
        flow_sign: np.ndarray,
        diffusivity_si: float,
        km: float,
        vmax: float,
        window_factor: float,
        lam_ref: float,
        gl_nodes: np.ndarray,
        gl_weights: np.ndarray,
        xs_lut: np.ndarray,
        k0_lut: np.ndarray,
    ) -> np.ndarray:
        n_points = points_si.shape[0]
        out = np.zeros(n_points, dtype=np.float64)
        for row in prange(n_points):
            if not keep_mask[row]:
                continue
            seg_idx = nearest_idx[row]
            if seg_idx.size == 0:
                continue
            proj_row = proj_raw[row]
            d_local = d_center[row]
            cap_max = 1e-6
            total = 0.0
            for local_i in range(seg_idx.size):
                seg_i = seg_idx[local_i]
                if seg_i < 0:
                    continue
                L = seg_len[seg_i]
                if L <= 0.0:
                    continue
                proj = proj_row[local_i]
                if flow_sign[seg_i] < 0.0:
                    proj = 1.0 - proj
                if proj < 0.0:
                    proj = 0.0
                if proj > 1.0:
                    proj = 1.0
                s_star = proj * L
                Cc_star = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s_star)
                denom_gate = km + max(Cc_star, 1e-12)
                if denom_gate < 1e-30:
                    denom_gate = 1e-30
                lam_gate = np.sqrt(diffusivity_si / max(vmax / denom_gate, 1e-30))
                if d_local[local_i] > window_factor * lam_gate:
                    continue
                if Cc_star > cap_max:
                    cap_max = Cc_star

                halfW = window_factor * lam_ref
                s0 = s_star - halfW
                if s0 < 0.0:
                    s0 = 0.0
                s1 = s_star + halfW
                if s1 > L:
                    s1 = L
                if s1 <= s0 + 1e-15:
                    continue

                mid = 0.5 * (s0 + s1)
                half = 0.5 * (s1 - s0)
                total_local = 0.0
                for g in range(gl_nodes.shape[0]):
                    s = mid + half * gl_nodes[g]
                    t = s / L
                    xs = starts_si[seg_i] + t * segment_vectors[seg_i]
                    r_vec0 = points_si[row, 0] - xs[0]
                    r_vec1 = points_si[row, 1] - xs[1]
                    r_vec2 = points_si[row, 2] - xs[2]
                    r = np.sqrt(r_vec0 * r_vec0 + r_vec1 * r_vec1 + r_vec2 * r_vec2)
                    if r <= 1e-12:
                        continue
                    cc_s = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s)
                    denom_loc = km + max(cc_s, 1e-12)
                    if denom_loc < 1e-30:
                        denom_loc = 1e-30
                    lam_loc = np.sqrt(diffusivity_si / max(vmax / denom_loc, 1e-30))
                    phi_loc = radii_si[seg_i] / max(lam_loc, 1e-30)
                    if phi_loc < 1e-12:
                        phi_loc = 1e-12
                    r_over_lam = r / max(lam_loc, 1e-30)
                    k0_num = _interp_scalar(r_over_lam, xs_lut, k0_lut)
                    k0_den = _interp_scalar(phi_loc, xs_lut, k0_lut)
                    if k0_den < 1e-300:
                        k0_den = 1e-300
                    Ci_R = cc_s * (k0_num / k0_den)

                    denom_corr = km + min(3.0 * Ci_R, cc_s)
                    if denom_corr < 1e-30:
                        denom_corr = 1e-30
                    lam_corr = np.sqrt(diffusivity_si / max(vmax / denom_corr, 1e-30))
                    phi = radii_si[seg_i] / max(lam_corr, 1e-30)
                    if phi < 1e-12:
                        phi = 1e-12
                    ratio = _interp_scalar(phi, _state._KRATIO_XS, _state._KRATIO_YS)
                    wall_factor = (diffusivity_si / max(lam_corr, 1e-30)) * ratio
                    q_s = (2.0 * np.pi * radii_si[seg_i]) * wall_factor * cc_s
                    kernel = np.exp(-r / max(lam_corr, 1e-30)) / (4.0 * np.pi * r)
                    integrand = q_s * (kernel / diffusivity_si)
                    total_local += gl_weights[g] * integrand

                total += half * total_local

            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > -np.inf:
                out[row] = min(total, cap_max)
            else:
                out[row] = total
        return out

    @njit(cache=True, parallel=True)
    def _tissue_dense_kernel_numba(
        points_si: np.ndarray,
        starts_si: np.ndarray,
        segment_vectors: np.ndarray,
        seg_len_sq: np.ndarray,
        seg_len: np.ndarray,
        radii_si: np.ndarray,
        cin_pos: np.ndarray,
        alpha_edge: np.ndarray,
        flow_sign: np.ndarray,
        diffusivity_si: float,
        km: float,
        vmax: float,
        window_factor: float,
        lam_ref: float,
        gl_nodes: np.ndarray,
        gl_weights: np.ndarray,
        xs_lut: np.ndarray,
        k0_lut: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse exact all-segment geometry and tissue integration on the CPU."""
        n_points = points_si.shape[0]
        n_segments = starts_si.shape[0]
        keep_mask = np.ones(n_points, dtype=np.bool_)
        out = np.zeros(n_points, dtype=np.float64)
        for row in prange(n_points):
            px = points_si[row, 0]
            py = points_si[row, 1]
            pz = points_si[row, 2]
            cap_max = 1.0e-6
            total = 0.0
            for seg_i in range(n_segments):
                len2 = seg_len_sq[seg_i]
                L = seg_len[seg_i]
                if len2 <= 0.0 or L <= 0.0:
                    continue
                sx0 = starts_si[seg_i, 0]
                sx1 = starts_si[seg_i, 1]
                sx2 = starts_si[seg_i, 2]
                vx0 = segment_vectors[seg_i, 0]
                vx1 = segment_vectors[seg_i, 1]
                vx2 = segment_vectors[seg_i, 2]
                wx0 = px - sx0
                wx1 = py - sx1
                wx2 = pz - sx2
                proj_raw = (wx0 * vx0 + wx1 * vx1 + wx2 * vx2) / len2
                proj_clip = min(1.0, max(0.0, proj_raw))
                dx = px - (sx0 + proj_clip * vx0)
                dy = py - (sx1 + proj_clip * vx1)
                dz = pz - (sx2 + proj_clip * vx2)
                d_center = np.sqrt(dx * dx + dy * dy + dz * dz)
                if (
                    proj_raw >= 0.0
                    and proj_raw <= 1.0
                    and d_center <= min(radii_si[seg_i], L)
                ):
                    keep_mask[row] = False
                    total = 0.0
                    break

                proj = 1.0 - proj_raw if flow_sign[seg_i] < 0.0 else proj_raw
                proj = min(1.0, max(0.0, proj))
                s_star = proj * L
                Cc_star = cin_pos[seg_i] * np.exp(-alpha_edge[seg_i] * s_star)
                denom_gate = max(km + max(Cc_star, 1.0e-12), 1.0e-30)
                lam_gate = np.sqrt(diffusivity_si / max(vmax / denom_gate, 1.0e-30))
                if d_center > window_factor * lam_gate:
                    continue
                if Cc_star > cap_max:
                    cap_max = Cc_star

                half_width = window_factor * lam_ref
                s0 = max(0.0, s_star - half_width)
                s1 = min(L, s_star + half_width)
                if s1 <= s0 + 1.0e-15:
                    continue
                mid = 0.5 * (s0 + s1)
                half = 0.5 * (s1 - s0)
                total_local = 0.0
                radius = radii_si[seg_i]
                cin_seg = cin_pos[seg_i]
                alpha = alpha_edge[seg_i]
                for g in range(gl_nodes.shape[0]):
                    s = mid + half * gl_nodes[g]
                    t = s / L
                    rx = px - (sx0 + t * vx0)
                    ry = py - (sx1 + t * vx1)
                    rz = pz - (sx2 + t * vx2)
                    r = np.sqrt(rx * rx + ry * ry + rz * rz)
                    if r <= 1.0e-12:
                        continue
                    cc_s = cin_seg * np.exp(-alpha * s)
                    denom_loc = max(km + max(cc_s, 1.0e-12), 1.0e-30)
                    lam_loc = np.sqrt(diffusivity_si / max(vmax / denom_loc, 1.0e-30))
                    phi_loc = max(radius / max(lam_loc, 1.0e-30), 1.0e-12)
                    r_over_lam = r / max(lam_loc, 1.0e-30)
                    k0_num = _interp_scalar(r_over_lam, xs_lut, k0_lut)
                    k0_den = max(_interp_scalar(phi_loc, xs_lut, k0_lut), 1.0e-300)
                    ci_r = cc_s * (k0_num / k0_den)
                    denom_corr = max(km + min(3.0 * ci_r, cc_s), 1.0e-30)
                    lam_corr = np.sqrt(diffusivity_si / max(vmax / denom_corr, 1.0e-30))
                    phi = max(radius / max(lam_corr, 1.0e-30), 1.0e-12)
                    ratio = _interp_scalar(phi, _state._KRATIO_XS, _state._KRATIO_YS)
                    wall_factor = (diffusivity_si / max(lam_corr, 1.0e-30)) * ratio
                    q_s = 2.0 * np.pi * radius * wall_factor * cc_s
                    green = np.exp(-r / max(lam_corr, 1.0e-30)) / (4.0 * np.pi * r)
                    total_local += gl_weights[g] * q_s * (green / diffusivity_si)
                total += half * total_local

            if keep_mask[row]:
                if total < 0.0 or not np.isfinite(total):
                    total = 0.0
                out[row] = min(total, cap_max)
        return keep_mask, out

    @njit(cache=True, parallel=True)
    def _cext_tissue_kernel_numba(
        points_si: np.ndarray,
        nearest_idx: np.ndarray,
        keep_mask: np.ndarray,
        valid_source_ids: np.ndarray,
        gl_points_si: np.ndarray,
        segment_vectors: np.ndarray,
        lambda_iv_gl: np.ndarray,
        q_weighted_gl: np.ndarray,
        mono2_weight_gl: np.ndarray,
        dipole2_weight_gl: np.ndarray,
        seg_cap_gl: np.ndarray,
        diffusivity_si: float,
        window_factor: float,
    ) -> np.ndarray:
        """Evaluate converged Cext sources without Python point/segment loops."""
        n_points = points_si.shape[0]
        gl_order = gl_points_si.shape[1]
        out = np.zeros(n_points, dtype=np.float64)
        for row in prange(n_points):
            if not keep_mask[row]:
                continue
            px = points_si[row, 0]
            py = points_si[row, 1]
            pz = points_si[row, 2]
            total = 0.0
            cap_max = 0.0
            for local_i in range(nearest_idx.shape[1]):
                compact_seg = nearest_idx[row, local_i]
                if compact_seg < 0 or compact_seg >= valid_source_ids.size:
                    continue
                source_seg = valid_source_ids[compact_seg]
                if source_seg < 0 or source_seg >= gl_points_si.shape[0]:
                    continue
                seg_contributed = False
                vx = segment_vectors[source_seg, 0]
                vy = segment_vectors[source_seg, 1]
                vz = segment_vectors[source_seg, 2]
                vector_length = np.sqrt(vx * vx + vy * vy + vz * vz)
                for source_node in range(gl_order):
                    source_lambda = max(lambda_iv_gl[source_seg, source_node], 1.0e-30)
                    dx = px - gl_points_si[source_seg, source_node, 0]
                    dy = py - gl_points_si[source_seg, source_node, 1]
                    dz = pz - gl_points_si[source_seg, source_node, 2]
                    radius = np.sqrt(dx * dx + dy * dy + dz * dz)
                    if radius <= 1.0e-12 or radius > window_factor * source_lambda:
                        continue
                    kernel = np.exp(-radius / source_lambda) / (
                        4.0 * np.pi * diffusivity_si * radius
                    )
                    total += q_weighted_gl[source_seg, source_node] * kernel
                    o2_weight = (
                        mono2_weight_gl[source_seg, source_node]
                        + dipole2_weight_gl[source_seg, source_node]
                    )
                    if o2_weight != 0.0 and vector_length > 1.0e-30:
                        tangent_dot_r = (vx * dx + vy * dy + vz * dz) / vector_length
                        mu2 = min(
                            (tangent_dot_r * tangent_dot_r)
                            / max(radius * radius, 1.0e-30),
                            1.0,
                        )
                        inv_radius = 1.0 / radius
                        inv_lambda = 1.0 / source_lambda
                        p_hessian = (
                            (1.0 - 3.0 * mu2)
                            * (inv_radius * inv_radius + inv_lambda * inv_radius)
                            + (1.0 - mu2) * inv_lambda * inv_lambda
                        ) * kernel
                        total += o2_weight * p_hessian
                    seg_contributed = True
                if seg_contributed:
                    cap_max = max(cap_max, seg_cap_gl[source_seg])
            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > 0.0:
                total = min(total, cap_max)
            out[row] = total
        return out


def _greens_lambda_char(
    diffusivity: float, vmax: float, km: float, cin: float
) -> float:
    denom = max(km + max(cin, 0.0), 1e-30)
    k1 = vmax / denom
    return float(np.sqrt(diffusivity / max(k1, 1e-30)))


def _greens_decay_factor(
    flow: float,
    radius: float,
    length: float,
    diffusivity: float,
    vmax: float,
    km: float,
    cin: float,
) -> float:
    if length <= 0.0 or radius <= 0.0:
        return 1.0
    flow_mag = abs(flow)
    if flow_mag <= 0.0:
        return 1.0
    lam = _greens_lambda_char(diffusivity, vmax, km, cin)
    phi = radius / max(lam, 1e-30)
    ratio = float(_k_ratio(phi))
    beta = (
        (2.0 * np.pi * radius / max(flow_mag, 1e-30))
        * (diffusivity / max(lam, 1e-30))
        * ratio
    )
    exponent = np.clip(-beta * length, -150.0, 50.0)
    return float(np.exp(exponent))


def _greens_segment_params(
    cin: np.ndarray,
    radii: np.ndarray,
    flows: np.ndarray,
    diffusivity: float,
    vmax: float,
    km: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cin_pos = np.maximum(cin, 0.0)
    denom = np.maximum(km + cin_pos, 1e-30)
    k1 = vmax / denom
    lam = np.sqrt(diffusivity / np.maximum(k1, 1e-30))
    phi = radii / np.maximum(lam, 1e-30)
    ratio = _k_ratio(phi)
    flow_mag = np.maximum(np.abs(flows), 1e-30)
    beta = (
        (2.0 * np.pi * radii / flow_mag)
        * (diffusivity / np.maximum(lam, 1e-30))
        * ratio
    )
    return lam, ratio, beta


def _hill_dS_dC(
    cin: float,
    *,
    alpha: float = _state.ALPHA_MMHG,
    p50: float = _state.P50_MMHG,
    n_hill: float = _state.N_HILL,
) -> float:
    cin_pos = max(cin, 0.0)
    alpha_n = alpha**n_hill
    p50_n = p50**n_hill
    denom = alpha_n * p50_n + cin_pos**n_hill
    return float(
        (n_hill * alpha_n * p50_n * cin_pos ** (n_hill - 1.0))
        / max(denom * denom, 1e-30)
    )


def severinghaus_saturation(P_mmHg: np.ndarray | float) -> np.ndarray | float:
    P = np.asarray(P_mmHg, dtype=float)
    num = P**3 + 150.0 * P
    den = num + 23400.0
    S = num / den
    if np.isscalar(P_mmHg):
        return float(S)
    return S


def severinghaus_dSdP(P_mmHg: np.ndarray | float) -> np.ndarray | float:
    P = np.asarray(P_mmHg, dtype=float)
    den = P**3 + 150.0 * P + 23400.0
    dSdP = 70200.0 * (P**2 + 50.0) / (den**2)
    if np.isscalar(P_mmHg):
        return float(dSdP)
    return dSdP


def segment_O2_capacity(radius_cm: float, hd: float = _state.HD_DISCHARGE) -> float:
    HT = tube_hematocrit(radius_cm, hd=hd)
    # O2_CAP_PER_HCT is already mol / m^3 per unit hematocrit.
    return float(HT * _state.O2_CAP_PER_HCT)


def segment_O2_capacity_from_HT(HT: float) -> float:
    # O2_CAP_PER_HCT is already mol / m^3 per unit hematocrit.
    return float(HT * _state.O2_CAP_PER_HCT)


def buffer_factor_B(C_plasma: float, Chb_max: float) -> float:
    C_plasma = max(C_plasma, 1e-12)
    P_mmHg = C_plasma / _state.ALPHA_MMHG
    dSdP = severinghaus_dSdP(P_mmHg)
    return float(1.0 + (Chb_max / _state.ALPHA_MMHG) * dSdP)


def _blood_greens_decay_factor(
    flow: float,
    radius: float,
    length: float,
    diffusivity: float,
    vmax: float,
    km: float,
    cin: float,
    ccap: float,
) -> float:
    if length <= 0.0 or radius <= 0.0:
        return 1.0
    flow_mag = abs(flow)
    if flow_mag <= 0.0:
        return 1.0
    lam = _greens_lambda_char(diffusivity, vmax, km, cin)
    phi = radius / max(lam, 1e-30)
    ratio = float(_k_ratio(phi))
    base = (
        (2.0 * np.pi * radius / max(flow_mag, 1e-30))
        * (diffusivity / max(lam, 1e-30))
        * ratio
    )
    steps = max(int(_state.AXIAL_BLOOD_STEPS), 1)
    ds = length / steps
    c_local = float(max(cin, 0.0))
    for _ in range(steps):
        buffer = (
            1.0
            + max(ccap, 0.0)
            * severinghaus_dSdP(c_local / _state.ALPHA_MMHG)
            / _state.ALPHA_MMHG
        )
        beta_local = base / max(buffer, 1e-30)
        c_local *= float(np.exp(np.clip(-beta_local * ds, -150.0, 50.0)))
    return c_local / max(cin, 1e-30)


def _lambda_if_from_civ(
    c_iv: np.ndarray | float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    c_local = np.maximum(np.asarray(c_iv, dtype=float), 0.0)
    denom = np.maximum(km + c_local, 1e-30)
    return np.sqrt(diffusivity_si / np.maximum(vmax / denom, 1e-30))


def _lambda_tissue_from_civ_ctissue(
    c_iv: np.ndarray | float,
    c_tissue: np.ndarray | float,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    c_iv_arr = np.maximum(np.asarray(c_iv, dtype=float), 0.0)
    c_tissue_arr = np.maximum(np.asarray(c_tissue, dtype=float), 0.0)
    effective_term = np.sqrt(
        np.maximum(km + c_iv_arr, 1e-30) * np.maximum(km + c_tissue_arr, 1e-30)
    )
    return np.sqrt(diffusivity_si / np.maximum(vmax / effective_term, 1e-30))


def _normalize_cext_lambda_source(value: str | None = None) -> str:
    mode = str(_state.CEXT_LAMBDA_SOURCE if value is None else value).strip().lower()
    aliases = {
        "vv": "lambda_vv",
        "lambda_vv": "lambda_vv",
        "vessel": "lambda_vv",
        "vessel_vessel": "lambda_vv",
        "if": "lambda_vv",
        "lambda_if": "lambda_vv",
        "wall": "lambda_vv",
        "t": "lambda_t",
        "lambda_t": "lambda_t",
        "tissue": "lambda_t",
        "lambda_tissue": "lambda_t",
    }
    if mode not in aliases:
        raise ValueError("SVV_CEXT_LAMBDA_SOURCE must be 'lambda_vv' or 'lambda_t'.")
    return aliases[mode]


def _cext_green_lambda_from_fields(
    c_wall: np.ndarray,
    c_ext: np.ndarray,
    lambda_wall: np.ndarray,
    diffusivity_si: float,
    vmax: float,
    km: float,
) -> np.ndarray:
    if _normalize_cext_lambda_source() == "lambda_vv":
        return np.maximum(np.asarray(lambda_wall, dtype=float), 1e-30)
    return np.maximum(
        _lambda_tissue_from_civ_ctissue(c_wall, c_ext, diffusivity_si, vmax, km),
        1e-30,
    )


def _interfacial_transfer_coefficient(
    radius_si: np.ndarray | float,
    lambda_if: np.ndarray | float,
    diffusivity_si: float,
) -> np.ndarray:
    radius_arr = np.maximum(np.asarray(radius_si, dtype=float), 0.0)
    lambda_arr = np.maximum(np.asarray(lambda_if, dtype=float), 1e-30)
    phi = np.maximum(radius_arr / lambda_arr, 1e-12)
    ratio = _k_ratio(phi)
    return (2.0 * np.pi * radius_arr) * (diffusivity_si / lambda_arr) * ratio


def _finite_radius_o2_term_flags() -> tuple[bool, bool]:
    mode = str(_state.FINITE_RADIUS_O2_TERMS or "none").strip().lower()
    if mode in ("0", "false", "off", "none"):
        return False, False
    if mode in ("monopole", "mono", "monopole2", "source"):
        return True, False
    if mode in ("dipole", "dipole2", "target"):
        return False, True
    if mode in ("both", "all", "monopole+dipole"):
        return True, True
    raise ValueError(
        "--finite-radius-o2-terms must be one of none, monopole, dipole, both."
    )


__all__ = [
    "_build_k_ratio_lut",
    "_load_k_ratio_lut",
    "_k0est",
    "_k_ratio",
    "_k0_lookup",
    "_k1_lookup",
    "_interp_scalar",
    "_k0_lookup_numba",
    "_k_ratio_scalar_numba",
    "_lambda_if_scalar_numba",
    "_interfacial_transfer_coeff_scalar_numba",
    "_greens_decay_ratio_numba",
    "_severinghaus_dSdP_scalar",
    "_blood_greens_decay_ratio_numba",
    "_solve_channel_concentrations_topdown_numba",
    "_tissue_kernel_numba",
    "_tissue_dense_kernel_numba",
    "_cext_tissue_kernel_numba",
    "_greens_lambda_char",
    "_greens_decay_factor",
    "_greens_segment_params",
    "_hill_dS_dC",
    "severinghaus_saturation",
    "severinghaus_dSdP",
    "segment_O2_capacity",
    "segment_O2_capacity_from_HT",
    "buffer_factor_B",
    "_blood_greens_decay_factor",
    "_lambda_if_from_civ",
    "_lambda_tissue_from_civ_ctissue",
    "_normalize_cext_lambda_source",
    "_cext_green_lambda_from_fields",
    "_interfacial_transfer_coefficient",
    "_finite_radius_o2_term_flags",
]
