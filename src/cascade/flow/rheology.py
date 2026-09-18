"""Viscosity, resistance, and Fahraeus-Lindqvist rheology."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.flow.hematocrit import (
    _normalize_hematocrit_model,
    _tree_exact_connectivity,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - dependency validation reports this earlier

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate


def _dbg(message: str) -> None:
    """Compatibility debug hook; detailed solver logging is opt-in elsewhere."""
    return None


def eta_rel_pries(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
    d = np.asarray(d_um, dtype=float)
    hd = np.asarray(hd, dtype=float)
    A = 4.0 / (1.0 + np.exp(-0.593 * (d - 6.74)))
    term = 110.0 * np.exp(-1.424 * d) + 3.0 - 3.45 * np.exp(-0.035 * d)
    num = np.exp(hd) - 1.0
    den = np.exp(0.45 * A) - 1.0
    factor = np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)
    return 1.0 + factor * term


def _use_pries_secomb_blood_rheology() -> bool:
    try:
        return _normalize_hematocrit_model() == "pries_secomb"
    except Exception:
        return False


def pries_secomb_viscor_cgs(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
    """Pries-Secomb apparent blood viscosity, returned in cgs units."""
    d = np.asarray(d_um, dtype=float)
    h = np.clip(
        np.asarray(hd, dtype=float), _state.HEMATOCRIT_MIN, _state.HEMATOCRIT_MAX
    )
    h = np.broadcast_to(h, d.shape)
    dcorr = np.maximum(d * float(_state.PRIES_SECOMB_MCV_CORR), 1.0e-9)
    denom = np.maximum(dcorr - float(_state.PRIES_SECOMB_OPTW_UM), 1.0e-9)
    geom_fac = (dcorr / denom) ** 2
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inv_term = 1.0 / (
            1.0
            + (10.0**_state.PRIES_SECOMB_CPAR_3) * (dcorr**_state.PRIES_SECOMB_CPAR_4)
        )
        c = (
            _state.PRIES_SECOMB_CPAR_1 + np.exp(_state.PRIES_SECOMB_CPAR_2 * dcorr)
        ) * (-1.0 + inv_term) + inv_term
        eta45 = (
            _state.PRIES_SECOMB_VISCPAR_1
            * np.exp(_state.PRIES_SECOMB_VISCPAR_2 * dcorr)
            + _state.PRIES_SECOMB_VISCPAR_3
            + _state.PRIES_SECOMB_VISCPAR_4
            * np.exp(
                _state.PRIES_SECOMB_VISCPAR_5 * (dcorr**_state.PRIES_SECOMB_VISCPAR_6)
            )
        )
        hdfac_den = (1.0 - 0.45) ** c - 1.0
        hdfac = np.divide(
            (1.0 - h) ** c - 1.0,
            hdfac_den,
            out=np.zeros_like(dcorr),
            where=hdfac_den != 0.0,
        )
        etarel = (1.0 + (eta45 - 1.0) * hdfac * geom_fac) * geom_fac
    mu_cgs = etarel * _state.PRIES_SECOMB_VPLAS_CP * 0.01
    return np.where(
        np.isfinite(mu_cgs) & (mu_cgs > 0.0),
        mu_cgs,
        _state.PRIES_SECOMB_VPLAS_CP * 0.01,
    )


if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _pries_secomb_viscor_cgs_numba(d_um: np.ndarray, hd: np.ndarray) -> np.ndarray:
        n = d_um.shape[0]
        out = np.empty(n, dtype=np.float64)
        fallback = _state.PRIES_SECOMB_VPLAS_CP * 0.01
        ten_cpar3 = 10.0**_state.PRIES_SECOMB_CPAR_3
        for i in range(n):
            d = d_um[i]
            if not np.isfinite(d) or d <= 0.0:
                d = 1.0e-9
            h = hd[i]
            if not np.isfinite(h):
                h = _state.HD_DISCHARGE
            if h < _state.HEMATOCRIT_MIN:
                h = _state.HEMATOCRIT_MIN
            elif h > _state.HEMATOCRIT_MAX:
                h = _state.HEMATOCRIT_MAX

            dcorr = d * _state.PRIES_SECOMB_MCV_CORR
            if dcorr < 1.0e-9:
                dcorr = 1.0e-9
            denom = dcorr - _state.PRIES_SECOMB_OPTW_UM
            if denom < 1.0e-9:
                denom = 1.0e-9
            geom_fac = (dcorr / denom) ** 2.0
            inv_term = 1.0 / (1.0 + ten_cpar3 * (dcorr**_state.PRIES_SECOMB_CPAR_4))
            c = (
                _state.PRIES_SECOMB_CPAR_1 + np.exp(_state.PRIES_SECOMB_CPAR_2 * dcorr)
            ) * (-1.0 + inv_term) + inv_term
            eta45 = (
                _state.PRIES_SECOMB_VISCPAR_1
                * np.exp(_state.PRIES_SECOMB_VISCPAR_2 * dcorr)
                + _state.PRIES_SECOMB_VISCPAR_3
                + _state.PRIES_SECOMB_VISCPAR_4
                * np.exp(
                    _state.PRIES_SECOMB_VISCPAR_5
                    * (dcorr**_state.PRIES_SECOMB_VISCPAR_6)
                )
            )
            hdfac_den = (1.0 - 0.45) ** c - 1.0
            if hdfac_den != 0.0:
                hdfac = ((1.0 - h) ** c - 1.0) / hdfac_den
            else:
                hdfac = 0.0
            etarel = (1.0 + (eta45 - 1.0) * hdfac * geom_fac) * geom_fac
            mu = etarel * fallback
            if np.isfinite(mu) and mu > 0.0:
                out[i] = mu
            else:
                out[i] = fallback
        return out


def _segment_permeation_rate(radius: float, diffusivity: float) -> float:
    if not np.isfinite(radius) or radius <= 0.0:
        return 0.0
    return float(3.66 * diffusivity / (radius * radius))


def tube_hematocrit(radius_cm: float, hd: float = _state.HD_DISCHARGE) -> float:
    if radius_cm <= 0.0 or not np.isfinite(radius_cm):
        return 0.0
    d_um = 2.0 * radius_cm * 1.0e4
    ratio = hd + (1.0 - hd) * (
        1.0 + 1.7 * np.exp(-0.415 * d_um) - 0.6 * np.exp(-0.011 * d_um)
    )
    return float(hd * ratio)


def segment_viscosity_from_radius(
    radii_cm: np.ndarray,
    mu_base: float,
    fluid: str,
    hd: float = _state.HD_DISCHARGE,
) -> np.ndarray:
    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    if fluid_mode != "blood":
        return np.full_like(radii_cm, float(mu_base))
    return segment_viscosity_from_radius_hd(
        radii_cm, mu_base, fluid, np.full_like(radii_cm, float(hd), dtype=float)
    )


def segment_viscosity_from_radius_hd(
    radii_cm: np.ndarray,
    mu_base: float,
    fluid: str,
    hd: np.ndarray,
) -> np.ndarray:
    fluid_mode = (fluid or _state.ACTIVE_FLUID).lower()
    if fluid_mode != "blood":
        return np.full_like(radii_cm, float(mu_base), dtype=float)
    d_um = 2.0 * radii_cm * 1.0e4
    hd_arr = np.asarray(hd, dtype=float)
    if hd_arr.ndim == 0:
        hd_arr = np.full_like(d_um, float(hd_arr), dtype=float)
    elif hd_arr.shape[0] != d_um.shape[0]:
        hd_arr = np.full_like(d_um, float(_state.HD_DISCHARGE), dtype=float)
    if _use_pries_secomb_blood_rheology():
        if _state._HAVE_NUMBA:
            return _pries_secomb_viscor_cgs_numba(
                np.asarray(d_um, dtype=np.float64),
                np.asarray(hd_arr, dtype=np.float64),
            )
        return pries_secomb_viscor_cgs(d_um, hd_arr)
    eta_rel = eta_rel_pries(d_um, hd_arr)
    return mu_base * eta_rel


def compute_segment_viscosity(
    tree,
    hd_root: float = _state.HD_DISCHARGE,
    hd_per_segment: np.ndarray | None = None,
    data_override: np.ndarray | None = None,
) -> np.ndarray:
    if data_override is not None:
        data = np.asarray(data_override, dtype=float)
        nseg = data.shape[0]
    else:
        nseg = int(getattr(tree, "segment_count", 0))
        if nseg <= 0:
            return np.empty((0,), dtype=float)
        data = np.asarray(tree.data[:nseg], dtype=float)

    radii_cm = data[:, 21]
    if hd_per_segment is None:
        HD = np.full(nseg, float(hd_root), dtype=float)
    else:
        HD = np.asarray(hd_per_segment, dtype=float)
        if HD.shape[0] != nseg:
            HD = np.full(nseg, float(hd_root), dtype=float)

    rho = getattr(getattr(tree, "parameters", None), "fluid_density", 1.06)
    nu = getattr(getattr(tree, "parameters", None), "kinematic_viscosity", 0.012 / 1.06)
    mu_base = rho * nu
    fluid_mode = (
        getattr(tree, "fluid", None)
        or getattr(getattr(tree, "parameters", None), "fluid", None)
        or _state.ACTIVE_FLUID
    )
    return segment_viscosity_from_radius_hd(radii_cm, mu_base, fluid_mode, HD)


def _update_resistance_variable_mu_py(
    data: np.ndarray, idx: np.ndarray, gamma: float, mu_seg: np.ndarray
) -> None:
    vessels = list(range(data.shape[0]))
    max_depth = float(np.nanmax(data[:, 26]))
    while vessels:
        tmp = []
        for i in vessels:
            if data[i, 26] != max_depth:
                tmp.append(i)
                continue

            local_mu = mu_seg[i]
            if np.isnan(data[i, 15:17]).all():
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20]
                data[i, 27] = 0.0
            elif np.isnan(data[i, 15]):
                right = idx[i, 1]
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + data[right, 25]
                data[i, 23] = 0.0
                data[i, 24] = 1.0
                data[i, 27] = data[right, 20] + data[right, 27]
            elif np.isnan(data[i, 16]):
                left = idx[i, 0]
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + data[left, 25]
                data[i, 23] = 1.0
                data[i, 24] = 0.0
                data[i, 27] = data[left, 20] + data[left, 27]
            else:
                left = idx[i, 0]
                right = idx[i, 1]
                lr = (
                    (data[left, 22] * data[left, 25])
                    / (data[right, 22] * data[right, 25])
                ) ** 0.25
                lbif = (1.0 + lr ** (-gamma)) ** (-1.0 / gamma)
                rbif = (1.0 + lr**gamma) ** (-1.0 / gamma)
                data[i, 25] = (8.0 * local_mu / np.pi) * data[i, 20] + (
                    (lbif**4 / data[left, 25]) + (rbif**4 / data[right, 25])
                ) ** -1.0
                data[i, 23] = lbif
                data[i, 24] = rbif
                data[i, 27] = lbif**2 * (data[left, 20] + data[left, 27]) + rbif**2 * (
                    data[right, 20] + data[right, 27]
                )

        vessels = tmp
        max_depth -= 1.0


def apply_fahraeus_lindqvist_resistance(
    tree, *, fluid: str | None = None, hd_root: float = _state.HD_DISCHARGE
) -> None:
    fluid_mode = (
        fluid
        or getattr(getattr(tree, "parameters", None), "fluid", None)
        or _state.ACTIVE_FLUID
    ).lower()
    if fluid_mode != "blood":
        _dbg("F-L skipped: fluid_mode != blood")
        return

    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        _dbg("F-L skipped: nseg <= 0")
        return

    data = np.asarray(tree.data[:nseg], dtype=float)
    idx = _tree_exact_connectivity(tree, data)
    original_root_flow = float(data[0, 22]) if data.size else float("nan")

    try:
        hd_existing = getattr(tree, "discharge_hematocrit", None)
    except Exception:
        hd_existing = None

    mu_seg = compute_segment_viscosity(
        tree,
        hd_root=hd_root,
        hd_per_segment=hd_existing,
        data_override=data,
    )

    _dbg(
        f"F-L: nseg={nseg}, mu_base*eta_rel sample={mu_seg[:3] if mu_seg.size else 'empty'}"
    )

    _update_resistance_variable_mu_py(
        data, idx, float(getattr(tree.parameters, "murray_exponent", 3.0)), mu_seg
    )
    _dbg(
        f"F-L: updated resistances sample={data[:3, 25] if data.shape[0] >= 3 else data[:, 25]}"
    )
    flows = np.full(nseg, np.nan, dtype=float)
    delta_p = float(tree.parameters.root_pressure) - float(
        tree.parameters.terminal_pressure
    )
    root_R = data[0, 25]
    if root_R > 0:
        flows[0] = delta_p / root_R
        stack = [0]
        while stack:
            i = stack.pop()
            f_i = flows[i]
            left = idx[i, 0]
            right = idx[i, 1]
            has_left = idx[i, 0] >= 0
            has_right = idx[i, 1] >= 0
            if has_left and not has_right:
                flows[left] = f_i
                stack.append(left)
            elif has_right and not has_left:
                flows[right] = f_i
                stack.append(right)
            elif has_left and has_right:
                Rl = data[left, 25]
                Rr = data[right, 25]
                lbif = data[i, 23]
                rbif = data[i, 24]
                denom = 0.0
                if Rl > 0.0:
                    denom += (lbif**4) / Rl
                if Rr > 0.0:
                    denom += (rbif**4) / Rr
                if denom > 0.0:
                    flows[left] = f_i * ((lbif**4) / Rl) / denom if Rl > 0 else 0.0
                    flows[right] = f_i * ((rbif**4) / Rr) / denom if Rr > 0 else 0.0
                else:
                    flows[left] = flows[right] = f_i * 0.5
                stack.append(left)
                stack.append(right)
    n_terms = max(int(getattr(tree, "n_terminals", 0)), 1)
    desired_root_flow = original_root_flow
    if not np.isfinite(desired_root_flow) or desired_root_flow == 0.0:
        desired_root_flow = float(tree.parameters.terminal_flow) * float(n_terms)
    if np.isfinite(desired_root_flow) and np.isfinite(flows[0]) and flows[0] != 0.0:
        scale = desired_root_flow / flows[0]
        flows *= scale
    _dbg(f"F-L: updated flows sample={flows[:3] if flows.size else 'empty'}")

    try:
        if flows.size == data.shape[0]:
            data[:, 22] = flows
        tree.data[:nseg] = data
    except Exception:
        pass


__all__ = [
    "eta_rel_pries",
    "_use_pries_secomb_blood_rheology",
    "pries_secomb_viscor_cgs",
    "_pries_secomb_viscor_cgs_numba",
    "_segment_permeation_rate",
    "tube_hematocrit",
    "segment_viscosity_from_radius",
    "segment_viscosity_from_radius_hd",
    "compute_segment_viscosity",
    "_update_resistance_variable_mu_py",
    "apply_fahraeus_lindqvist_resistance",
]
