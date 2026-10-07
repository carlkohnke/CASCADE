"""Total oxygen flux helpers; dissolved concentration remains the unknown."""

from __future__ import annotations

import numpy as np

from cascade.configuration import solver_state as _state

from .greens import severinghaus_saturation

try:
    from numba import njit
except ImportError:

    def njit(*args, **kwargs):
        return lambda fn: fn


def transport_capacity(discharge, tube):
    """Choose hematocrit for axial convection, without changing rheology."""
    mode = str(_state.BLOOD_CONVECTIVE_HEMATOCRIT).lower()
    if mode not in {"tube", "discharge"}:
        raise ValueError("blood_convective_hematocrit must be 'tube' or 'discharge'")
    return np.asarray(discharge if mode == "discharge" else tube, dtype=float) * float(
        _state.O2_CAP_PER_HCT
    )


def network_discharge_hematocrit(network, flows):
    """Use flow-matched hematocrit where available, otherwise uniform inflow H_D."""
    from cascade.flow.hematocrit import _get_tree_hematocrit_cache

    cached = _get_tree_hematocrit_cache(
        network,
        len(flows),
        model=_state.HEMATOCRIT_MODEL,
        flows=flows,
    )
    if cached is not None:
        return np.asarray(cached[0], dtype=float)
    if str(_state.HEMATOCRIT_MODEL).lower() == "pries_secomb":
        raise ValueError(
            "Network oxygen transport requires flow-matched per-edge hematocrit for pries_secomb"
        )
    return np.full(len(flows), float(_state.HD_DISCHARGE))


def total_content_enabled(fluid):
    mode = str(_state.JUNCTION_OXYGEN_BALANCE).lower()
    if mode not in {"dissolved", "total_content"}:
        raise ValueError(
            "junction_oxygen_balance must be 'dissolved' or 'total_content'"
        )
    return str(fluid).lower() == "blood" and mode == "total_content"


def oxygen_content(concentration, capacity):
    return np.asarray(concentration) + np.asarray(capacity) * severinghaus_saturation(
        np.asarray(concentration) / float(_state.ALPHA_MMHG)
    )


@njit(cache=True)
def concentration_from_content(content, capacity, alpha, guess):
    """Safeguarded Newton inverse of the monotone Severinghaus content law."""
    if content <= 0.0:
        return 0.0
    if capacity <= 0.0:
        return content
    lo = max(content - capacity, 0.0)
    hi = content
    c = min(max(guess, lo), hi)
    for _ in range(64):
        p = c / alpha
        num = p * p * p + 150.0 * p
        den = num + 23400.0
        residual = c + capacity * num / den - content
        if abs(residual) <= 1e-12 * max(content, 1e-30):
            return c
        if residual > 0.0:
            hi = c
        else:
            lo = c
        slope = 1.0 + capacity / alpha * 70200.0 * (p * p + 50.0) / (den * den)
        trial = c - residual / slope
        c = trial if lo < trial < hi else 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


def nodal_capacity(up, down, q, capacity, nnode):
    """Flow-weighted outgoing capacity; sinks use incoming capacity."""
    qout = np.bincount(up, weights=q, minlength=nnode)
    qin = np.bincount(down, weights=q, minlength=nnode)
    cout = np.bincount(up, weights=q * capacity, minlength=nnode)
    cin = np.bincount(down, weights=q * capacity, minlength=nnode)
    denominator = np.where(qout > 0.0, qout, qin)
    numerator = np.where(qout > 0.0, cout, cin)
    return np.divide(
        numerator, denominator, out=np.zeros(nnode), where=denominator > 0.0
    )


def junction_flux_residual(
    up, down, q, cin, cout, capacity, nnode, *, boundary_nodes=()
):
    """Maximum relative nonlinear total-oxygen flux error at internal nodes."""
    flux_in = np.bincount(
        down, weights=q * oxygen_content(cout, capacity), minlength=nnode
    )
    flux_out = np.bincount(
        up, weights=q * oxygen_content(cin, capacity), minlength=nnode
    )
    qin = np.bincount(down, weights=q, minlength=nnode)
    qout = np.bincount(up, weights=q, minlength=nnode)
    internal = (qin > 0.0) & (qout > 0.0)
    # Prescribed-concentration reservoirs can exchange oxygen with returning
    # flow. They are boundary conditions, not conservative interior junctions.
    internal[np.asarray(list(boundary_nodes), dtype=int)] = False
    if not np.any(internal):
        return 0.0
    return float(
        np.max(
            np.abs(flux_in[internal] - flux_out[internal])
            / np.maximum(
                np.maximum(np.abs(flux_in[internal]), np.abs(flux_out[internal])), 1e-30
            )
        )
    )
