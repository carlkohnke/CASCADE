"""Top-down intravascular concentration transport."""

from __future__ import annotations

from time import perf_counter
from typing import Tuple

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.flow.hematocrit import (
    _get_tree_hematocrit_cache,
    _hematocrit_context_for_tree,
    _normalize_hematocrit_model,
    _store_tree_hematocrit_cache,
    compute_tree_hematocrit,
)

from .greens import (
    _blood_greens_decay_factor,
    _greens_decay_factor,
    _solve_channel_concentrations_topdown_numba,
)


def _fmt_seconds(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}s"


def _solve_channel_concentrations_topdown(
    tree: _state.Tree,
    flows: np.ndarray,
    *,
    inlet_concentration: float,
    diffusivity: float,
    vmax: float,
    km: float,
    fluid: str,
) -> Tuple[np.ndarray, np.ndarray]:
    nseg = int(getattr(tree, "segment_count", 0))
    if nseg <= 0:
        empty = np.empty((0,), dtype=float)
        return empty, empty

    t_total = perf_counter()
    t0 = perf_counter()
    data = np.asarray(tree.data[:nseg])
    hct_context = _hematocrit_context_for_tree(tree)
    lengths = np.asarray(data[:, 20], dtype=float)
    radii = np.asarray(hct_context["radii"], dtype=float)
    flows = np.asarray(flows, dtype=float)

    diffusivity_si = diffusivity * _state.CM2_TO_M2
    lengths_si = lengths * _state.CM_TO_M
    radii_si = radii * _state.CM_TO_M
    flows_si = flows * _state.CM3_TO_M3
    t_arrays = perf_counter() - t0

    fluid_mode = (
        fluid
        or getattr(getattr(tree, "parameters", None), "fluid", None)
        or _state.ACTIVE_FLUID
    ).lower()
    t_hema = 0.0
    hct_source = "none"
    HD = np.empty((0,), dtype=float)
    HT = np.empty((0,), dtype=float)
    if fluid_mode == "blood":
        t0 = perf_counter()
        cached_hct = _get_tree_hematocrit_cache(
            tree,
            nseg,
            model=_state.HEMATOCRIT_MODEL,
            flows=flows,
        )
        if cached_hct is not None:
            HD, HT = cached_hct
            hct_source = "cache"
        else:
            HD, HT = compute_tree_hematocrit(
                tree,
                hd_root=_state.HD_DISCHARGE,
                flows=flows,
                model=_state.HEMATOCRIT_MODEL,
                order=np.asarray(hct_context["order"], dtype=np.int64),
            )
            _store_tree_hematocrit_cache(
                tree,
                HD,
                HT,
                model=_state.HEMATOCRIT_MODEL,
                flows=flows,
                fixed_flow_bc=False,
            )
            hct_source = "computed"
        Chb_max = np.asarray(HT, dtype=float) * float(_state.O2_CAP_PER_HCT)
        t_hema = perf_counter() - t0
    else:
        Chb_max = np.zeros_like(radii)

    t0 = perf_counter()
    parents = np.asarray(hct_context["parents"], dtype=np.int64)
    order = np.asarray(hct_context["order"], dtype=np.int64)
    order_mode = str(hct_context["order_mode"])
    t_order = perf_counter() - t0

    if _state._HAVE_NUMBA and _state.CONC_USE_NUMBA:
        t0 = perf_counter()
        cin, cout = _solve_channel_concentrations_topdown_numba(
            order,
            parents,
            flows_si.astype(np.float64),
            radii_si.astype(np.float64),
            lengths_si.astype(np.float64),
            float(diffusivity_si),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(Chb_max, dtype=np.float64),
            bool(fluid_mode == "blood"),
            int(_state.AXIAL_BLOOD_STEPS),
        )
        t_kernel = perf_counter() - t0
    else:
        t0 = perf_counter()
        cin = np.full(nseg, np.nan, dtype=float)
        cout = np.full(nseg, np.nan, dtype=float)
        root_mask = parents < 0
        cin[root_mask] = float(inlet_concentration)
        for idx in order:
            if not np.isfinite(cin[idx]):
                parent = parents[idx] if 0 <= idx < parents.size else -1
                if 0 <= parent < nseg and np.isfinite(cout[parent]):
                    cin[idx] = cout[parent]
                else:
                    cin[idx] = float(inlet_concentration)

            cin_local = float(max(cin[idx], 0.0))
            cin[idx] = cin_local
            if fluid_mode == "blood":
                decay = _blood_greens_decay_factor(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                    Chb_max[idx],
                )
            else:
                decay = _greens_decay_factor(
                    flows_si[idx],
                    radii_si[idx],
                    lengths_si[idx],
                    diffusivity_si,
                    vmax,
                    km,
                    cin_local,
                )
            cout[idx] = cin_local * decay
        t_kernel = perf_counter() - t0

    if _state.SOLVER_TIMING_DETAILS:
        hd_text = ""
        if fluid_mode == "blood" and HD.size:
            hd_q = (
                np.quantile(HD[np.isfinite(HD)], [0.01, 0.5, 0.99])
                if np.any(np.isfinite(HD))
                else [float("nan")] * 3
            )
            ht_q = (
                np.quantile(HT[np.isfinite(HT)], [0.01, 0.5, 0.99])
                if np.any(np.isfinite(HT))
                else [float("nan")] * 3
            )
            hd_text = (
                f" hematocrit_model={_normalize_hematocrit_model()} "
                f"source={hct_source} "
                f"HD[p01={hd_q[0]:.3f} p50={hd_q[1]:.3f} p99={hd_q[2]:.3f}] "
                f"HT[p01={ht_q[0]:.3f} p50={ht_q[1]:.3f} p99={ht_q[2]:.3f}]"
            )
        print(
            "  Concentration topdown: "
            f"nseg={nseg} fluid={fluid_mode} order={order_mode} axial_steps={_state.AXIAL_BLOOD_STEPS if fluid_mode == 'blood' else 0} "
            f"arrays={_fmt_seconds(t_arrays)} hematocrit={_fmt_seconds(t_hema)} "
            f"order={_fmt_seconds(t_order)} kernel={_fmt_seconds(t_kernel)} "
            f"total={_fmt_seconds(perf_counter() - t_total)}{hd_text}"
        )

    return cin, cout


__all__ = ["_solve_channel_concentrations_topdown"]
