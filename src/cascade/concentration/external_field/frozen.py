"""Frozen-source external-field solvers.

Frozen modes hold the external source estimate fixed during a top-down transport
pass and provide CPU, Numba, GPU, and Graetz implementations.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.configuration.solver_state import _lumen_diffusivity_cm2_s_for_fluid
from cascade.concentration.vessel.graetz import _graetz_get_basis_table
from cascade.concentration.vessel.greens import (
    _interfacial_transfer_coeff_scalar_numba,
    _interfacial_transfer_coefficient,
    _lambda_if_from_civ,
    _lambda_if_scalar_numba,
    _severinghaus_dSdP_scalar,
    severinghaus_dSdP,
)
from cascade.accelerators.cuda import load_cuda_source

from .direct import _ensure_cext_gpu_static

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - Python fallbacks remain available

    def njit(*args, **kwargs):
        def decorate(func):
            return func

        return decorate

    prange = range

if _state._HAVE_NUMBA:

    @njit(cache=True)
    def _build_topdown_level_slices_numba(
        order_arr: np.ndarray,
        parents_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nseg = int(parents_arr.shape[0])
        if nseg <= 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
        depth = np.zeros((nseg,), dtype=np.int32)
        max_depth = 0
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i < 0 or idx_i >= nseg:
                continue
            parent = int(parents_arr[idx_i])
            if parent >= 0 and parent < nseg:
                depth[idx_i] = np.int32(depth[parent] + 1)
            else:
                depth[idx_i] = np.int32(0)
            if int(depth[idx_i]) > max_depth:
                max_depth = int(depth[idx_i])

        counts = np.zeros((max_depth + 1,), dtype=np.int32)
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i >= 0 and idx_i < nseg:
                counts[int(depth[idx_i])] += np.int32(1)

        level_offsets = np.zeros((counts.shape[0] + 1,), dtype=np.int32)
        total = 0
        for ii in range(counts.shape[0]):
            total += int(counts[ii])
            level_offsets[ii + 1] = np.int32(total)

        write_pos = level_offsets[:-1].copy()
        level_order = np.empty((nseg,), dtype=np.int32)
        for ii in range(order_arr.shape[0]):
            idx_i = int(order_arr[ii])
            if idx_i < 0 or idx_i >= nseg:
                continue
            d = int(depth[idx_i])
            pos = int(write_pos[d])
            level_order[pos] = np.int32(idx_i)
            write_pos[d] = np.int32(pos + 1)
        return level_order, level_offsets, depth

    @njit(cache=True)
    def _build_local_exclusion_arrays_numba(
        parents: np.ndarray,
        left_child: np.ndarray,
        right_child: np.ndarray,
        max_local: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        nseg = int(parents.shape[0])
        out_idx = np.full((nseg, max_local), -1, dtype=np.int32)
        out_count = np.zeros((nseg,), dtype=np.uint8)
        for seg in range(nseg):
            buf = np.full((max_local,), -1, dtype=np.int32)
            count = 0

            # Passing the segment buffer explicitly avoids retaining state
            # between iterations when Numba inlines these local helpers.
            def add_unique(value: int, count_local: int, buffer: np.ndarray) -> int:
                if value < 0 or value >= nseg:
                    return count_local
                for ii in range(count_local):
                    if buffer[ii] == value:
                        return count_local
                if count_local < max_local:
                    buffer[count_local] = np.int32(value)
                    return count_local + 1
                return count_local

            def add_immediate(node: int, count_local: int, buffer: np.ndarray) -> int:
                if node < 0 or node >= nseg:
                    return count_local
                parent = int(parents[node])
                count_local = add_unique(parent, count_local, buffer)
                left = int(left_child[node])
                right = int(right_child[node])
                count_local = add_unique(left, count_local, buffer)
                count_local = add_unique(right, count_local, buffer)
                if parent >= 0 and parent < nseg:
                    pl = int(left_child[parent])
                    pr = int(right_child[parent])
                    if pl != node:
                        count_local = add_unique(pl, count_local, buffer)
                    if pr != node:
                        count_local = add_unique(pr, count_local, buffer)
                return count_local

            count = add_unique(seg, count, buf)
            count = add_immediate(seg, count, buf)
            seed_count = count
            for ii in range(seed_count):
                count = add_immediate(int(buf[ii]), count, buf)
            out_count[seg] = np.uint8(count)
            for ii in range(count):
                out_idx[seg, ii] = buf[ii]
        return out_idx, out_count

    @njit(cache=True, parallel=True)
    def _solve_topdown_ext_frozen_numba(
        level_order: np.ndarray,
        level_offsets: np.ndarray,
        parents: np.ndarray,
        flows_si: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        gl_t: np.ndarray,
        c_ext_gl: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        inlet_concentration: float,
        chb_max: np.ndarray,
        is_blood: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nseg = int(parents.shape[0])
        m = int(gl_t.shape[0])
        cin_seg = np.full((nseg,), inlet_concentration, dtype=np.float32)
        cout_seg = np.full((nseg,), inlet_concentration, dtype=np.float32)
        c_iv_gl = np.full((nseg, m), inlet_concentration, dtype=np.float32)
        for level_idx in range(level_offsets.shape[0] - 1):
            start = int(level_offsets[level_idx])
            end = int(level_offsets[level_idx + 1])
            for pos in prange(start, end):
                seg_idx = int(level_order[pos])
                if seg_idx < 0 or seg_idx >= nseg:
                    continue
                parent = int(parents[seg_idx])
                cin_local = inlet_concentration
                if parent >= 0 and parent < nseg:
                    cin_local = float(cout_seg[parent])
                if cin_local < _state.VESS_CONC_FLOOR:
                    cin_local = _state.VESS_CONC_FLOOR
                cin_seg[seg_idx] = np.float32(cin_local)
                flow_mag_si = abs(float(flows_si[seg_idx]))
                if flow_mag_si <= 1e-30:
                    flow_mag_si = 1e-30
                radius_si = float(radii_si[seg_idx])
                length_si = float(lengths_si[seg_idx])
                c_running = float(cin_local)
                prev_s = 0.0
                for node_idx in range(m):
                    s_target = float(gl_t[node_idx]) * length_si
                    ds_step = s_target - prev_s
                    if ds_step < 0.0:
                        ds_step = 0.0
                    c_ext_local = float(c_ext_gl[seg_idx, node_idx])
                    if c_ext_local < 0.0:
                        c_ext_local = 0.0
                    lambda_if_up = _lambda_if_scalar_numba(
                        c_running, diffusivity_si, vmax, km
                    )
                    k_if_up = _interfacial_transfer_coeff_scalar_numba(
                        radius_si,
                        lambda_if_up,
                        diffusivity_si,
                        _state._KRATIO_XS,
                        _state._KRATIO_YS,
                    )
                    beta_if = k_if_up / flow_mag_si
                    if is_blood > 0:
                        buffer = (
                            1.0
                            + max(float(chb_max[seg_idx]), 0.0)
                            * _severinghaus_dSdP_scalar(c_running / _state.ALPHA_MMHG)
                            / _state.ALPHA_MMHG
                        )
                        if buffer < 1e-30:
                            buffer = 1e-30
                        beta_if = beta_if / buffer
                    exponent = -beta_if * ds_step
                    if exponent < -150.0:
                        exponent = -150.0
                    elif exponent > 50.0:
                        exponent = 50.0
                    c_node = c_ext_local + (c_running - c_ext_local) * math.exp(
                        exponent
                    )
                    if c_node < _state.VESS_CONC_FLOOR:
                        c_node = _state.VESS_CONC_FLOOR
                    c_iv_gl[seg_idx, node_idx] = np.float32(c_node)
                    c_running = c_node
                    prev_s = s_target
                ds_tail = length_si - prev_s
                if ds_tail < 0.0:
                    ds_tail = 0.0
                c_ext_tail = 0.0
                if m > 0:
                    c_ext_tail = float(c_ext_gl[seg_idx, m - 1])
                    if c_ext_tail < 0.0:
                        c_ext_tail = 0.0
                lambda_if_tail = _lambda_if_scalar_numba(
                    c_running, diffusivity_si, vmax, km
                )
                k_if_tail = _interfacial_transfer_coeff_scalar_numba(
                    radius_si,
                    lambda_if_tail,
                    diffusivity_si,
                    _state._KRATIO_XS,
                    _state._KRATIO_YS,
                )
                beta_tail = k_if_tail / flow_mag_si
                if is_blood > 0:
                    buffer_tail = (
                        1.0
                        + max(float(chb_max[seg_idx]), 0.0)
                        * _severinghaus_dSdP_scalar(c_running / _state.ALPHA_MMHG)
                        / _state.ALPHA_MMHG
                    )
                    if buffer_tail < 1e-30:
                        buffer_tail = 1e-30
                    beta_tail = beta_tail / buffer_tail
                exponent_tail = -beta_tail * ds_tail
                if exponent_tail < -150.0:
                    exponent_tail = -150.0
                elif exponent_tail > 50.0:
                    exponent_tail = 50.0
                c_out = c_ext_tail + (c_running - c_ext_tail) * math.exp(exponent_tail)
                if c_out < _state.VESS_CONC_FLOOR:
                    c_out = _state.VESS_CONC_FLOOR
                cout_seg[seg_idx] = np.float32(c_out)
        return cin_seg, cout_seg, c_iv_gl

    @njit(cache=True)
    def _propagate_network_ext_frozen_numba(
        node_conc: np.ndarray,
        up: np.ndarray,
        q: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        gl_t: np.ndarray,
        c_ext_gl: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        chb_max: np.ndarray,
        is_blood: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate one fixed node-concentration state without Python loops."""
        nseg = int(q.shape[0])
        gl_order = int(gl_t.shape[0])
        cin = np.empty((nseg,), dtype=np.float32)
        cout = np.empty((nseg,), dtype=np.float32)
        civ = np.empty((nseg, gl_order), dtype=np.float32)
        for edge_idx in range(nseg):
            c_in = float(node_conc[int(up[edge_idx])])
            if c_in < _state.VESS_CONC_FLOOR:
                c_in = _state.VESS_CONC_FLOOR
            cin[edge_idx] = np.float32(c_in)
            cout[edge_idx] = np.float32(c_in)
            for node_idx in range(gl_order):
                civ[edge_idx, node_idx] = np.float32(c_in)
            if float(q[edge_idx]) <= 1e-30:
                continue

            c_running = c_in
            previous_s = 0.0
            flow_mag = max(float(q[edge_idx]), 1e-30)
            radius_si = float(radii_si[edge_idx])
            length_si = float(lengths_si[edge_idx])
            for node_idx in range(gl_order):
                target_s = float(gl_t[node_idx]) * length_si
                step = max(target_s - previous_s, 0.0)
                c_external = max(float(c_ext_gl[edge_idx, node_idx]), 0.0)
                c_local = max(c_running, 0.0)
                denominator = max(km + c_local, 1e-30)
                lambda_if = np.sqrt(
                    diffusivity_si / max(vmax / denominator, 1e-30)
                )
                lambda_local = max(lambda_if, 1e-30)
                radius_local = max(radius_si, 0.0)
                phi = max(radius_local / lambda_local, 1e-12)
                ratio = np.interp(phi, _state._KRATIO_XS, _state._KRATIO_YS)
                k_if = (
                    (2.0 * np.pi * radius_local)
                    * (diffusivity_si / lambda_local)
                    * ratio
                )
                beta = k_if / flow_mag
                if is_blood > 0:
                    pressure = c_running / _state.ALPHA_MMHG
                    severinghaus_denominator = (
                        pressure**3 + 150.0 * pressure + 23400.0
                    )
                    derivative = (
                        70200.0
                        * (pressure**2 + 50.0)
                        / (severinghaus_denominator**2)
                    )
                    buffer = (
                        1.0
                        + max(float(chb_max[edge_idx]), 0.0)
                        * derivative
                        / _state.ALPHA_MMHG
                    )
                    beta /= max(float(buffer), 1e-30)
                exponent = min(max(-beta * step, -150.0), 50.0)
                c_running = max(
                    c_external
                    + (c_running - c_external) * np.exp(exponent),
                    _state.VESS_CONC_FLOOR,
                )
                civ[edge_idx, node_idx] = np.float32(c_running)
                previous_s = target_s

            tail_step = max(length_si - previous_s, 0.0)
            c_external = (
                max(float(c_ext_gl[edge_idx, gl_order - 1]), 0.0)
                if gl_order
                else 0.0
            )
            c_local = max(c_running, 0.0)
            denominator = max(km + c_local, 1e-30)
            lambda_if = np.sqrt(
                diffusivity_si / max(vmax / denominator, 1e-30)
            )
            lambda_local = max(lambda_if, 1e-30)
            radius_local = max(radius_si, 0.0)
            phi = max(radius_local / lambda_local, 1e-12)
            ratio = np.interp(phi, _state._KRATIO_XS, _state._KRATIO_YS)
            k_if = (
                (2.0 * np.pi * radius_local)
                * (diffusivity_si / lambda_local)
                * ratio
            )
            beta = k_if / flow_mag
            if is_blood > 0:
                pressure = c_running / _state.ALPHA_MMHG
                severinghaus_denominator = (
                    pressure**3 + 150.0 * pressure + 23400.0
                )
                derivative = (
                    70200.0
                    * (pressure**2 + 50.0)
                    / (severinghaus_denominator**2)
                )
                buffer = (
                    1.0
                    + max(float(chb_max[edge_idx]), 0.0)
                    * derivative
                    / _state.ALPHA_MMHG
                )
                beta /= max(float(buffer), 1e-30)
            exponent = min(max(-beta * tail_step, -150.0), 50.0)
            cout[edge_idx] = np.float32(
                max(
                    c_external
                    + (c_running - c_external) * np.exp(exponent),
                    _state.VESS_CONC_FLOOR,
                )
            )
        return cin, cout, civ

    @njit(cache=True)
    def _solve_network_ext_frozen_numba(
        up: np.ndarray,
        q: np.ndarray,
        incoming_offsets: np.ndarray,
        incoming_edges: np.ndarray,
        sum_out: np.ndarray,
        sum_in: np.ndarray,
        inlet_mask: np.ndarray,
        cached_cin: np.ndarray,
        radii_si: np.ndarray,
        lengths_si: np.ndarray,
        gl_t: np.ndarray,
        c_ext_gl: np.ndarray,
        diffusivity_si: float,
        vmax: float,
        km: float,
        inlet_concentration: float,
        chb_max: np.ndarray,
        is_blood: int,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Solve general-network fixed-field transport in compiled CPU code."""
        nseg = int(q.shape[0])
        nnode = int(inlet_mask.shape[0])
        node_conc = np.full((nnode,), inlet_concentration, dtype=np.float64)
        if cached_cin.shape[0] == nseg:
            for edge_idx in range(nseg):
                upstream = int(up[edge_idx])
                if q[edge_idx] > 1e-30 and inlet_mask[upstream] == 0:
                    node_conc[upstream] = max(
                        float(cached_cin[edge_idx]), _state.VESS_CONC_FLOOR
                    )
        for node in range(nnode):
            if inlet_mask[node] != 0:
                node_conc[node] = inlet_concentration

        iterations = 0
        for iteration_number in range(1, 101):
            iterations = iteration_number
            _, cout, _ = _propagate_network_ext_frozen_numba(
                node_conc,
                up,
                q,
                radii_si,
                lengths_si,
                gl_t,
                c_ext_gl,
                diffusivity_si,
                vmax,
                km,
                chb_max,
                is_blood,
            )
            updated = node_conc.copy()
            for node in range(nnode):
                row_start = int(incoming_offsets[node])
                row_stop = int(incoming_offsets[node + 1])
                if inlet_mask[node] != 0 or row_start == row_stop:
                    continue
                denominator = (
                    float(sum_out[node])
                    if float(sum_out[node]) > 1e-30
                    else float(sum_in[node])
                )
                if denominator > 1e-30:
                    numerator = 0.0
                    for position in range(row_start, row_stop):
                        edge_idx = int(incoming_edges[position])
                        numerator += float(q[edge_idx]) * float(cout[edge_idx])
                    updated[node] = numerator / denominator
            delta = 0.0
            for node in range(nnode):
                difference = abs(float(updated[node]) - float(node_conc[node]))
                if difference > delta:
                    delta = difference
                node_conc[node] += omega * (updated[node] - node_conc[node])
                if inlet_mask[node] != 0:
                    node_conc[node] = inlet_concentration
            if delta < 1e-6:
                break

        cin, cout, civ = _propagate_network_ext_frozen_numba(
            node_conc,
            up,
            q,
            radii_si,
            lengths_si,
            gl_t,
            c_ext_gl,
            diffusivity_si,
            vmax,
            km,
            chb_max,
            is_blood,
        )
        return cin, cout, civ, iterations


def _solve_topdown_ext_frozen_python(
    level_order: np.ndarray,
    level_offsets: np.ndarray,
    parents: np.ndarray,
    flows_si: np.ndarray,
    radii_si: np.ndarray,
    lengths_si: np.ndarray,
    gl_t: np.ndarray,
    c_ext_gl: np.ndarray,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    is_blood: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nseg = int(np.asarray(parents).shape[0])
    m = int(np.asarray(gl_t).shape[0])
    cin_seg = np.zeros((nseg,), dtype=np.float32)
    cout_seg = np.zeros((nseg,), dtype=np.float32)
    c_iv_gl = np.zeros((nseg, m), dtype=np.float32)
    for level_idx in range(int(np.asarray(level_offsets).shape[0]) - 1):
        start_pos = int(level_offsets[level_idx])
        end_pos = int(level_offsets[level_idx + 1])
        for pos in range(start_pos, end_pos):
            seg_idx = int(level_order[pos])
            parent = int(parents[seg_idx])
            cin_local = (
                float(inlet_concentration) if parent < 0 else float(cout_seg[parent])
            )
            cin_local = max(cin_local, _state.VESS_CONC_FLOOR)
            cin_seg[seg_idx] = np.float32(cin_local)
            flow_mag_si = max(abs(float(flows_si[seg_idx])), 1e-30)
            length_si = float(lengths_si[seg_idx])
            radius_si = float(radii_si[seg_idx])
            c_running = cin_local
            prev_s = 0.0
            for node_idx, gl_t_i in enumerate(np.asarray(gl_t, dtype=float)):
                s_target = float(gl_t_i) * length_si
                ds_step = max(s_target - prev_s, 0.0)
                c_ext_local = max(float(c_ext_gl[seg_idx, node_idx]), 0.0)
                lambda_if_up = float(
                    _lambda_if_from_civ(
                        c_running, float(diffusivity_si), float(vmax), float(km)
                    )
                )
                k_if_up = float(
                    _interfacial_transfer_coefficient(
                        radius_si, lambda_if_up, float(diffusivity_si)
                    )
                )
                beta_if = k_if_up / flow_mag_si
                if is_blood:
                    buffer = (
                        1.0
                        + max(float(chb_max[seg_idx]), 0.0)
                        * severinghaus_dSdP(c_running / _state.ALPHA_MMHG)
                        / _state.ALPHA_MMHG
                    )
                    beta_if = beta_if / max(float(buffer), 1e-30)
                c_node = c_ext_local + (c_running - c_ext_local) * float(
                    np.exp(np.clip(-beta_if * ds_step, -150.0, 50.0))
                )
                c_node = max(c_node, _state.VESS_CONC_FLOOR)
                c_iv_gl[seg_idx, node_idx] = np.float32(c_node)
                c_running = c_node
                prev_s = s_target
            ds_tail = max(length_si - prev_s, 0.0)
            c_ext_tail = max(float(c_ext_gl[seg_idx, -1]), 0.0) if m > 0 else 0.0
            lambda_tail = float(
                _lambda_if_from_civ(
                    c_running, float(diffusivity_si), float(vmax), float(km)
                )
            )
            k_if_tail = float(
                _interfacial_transfer_coefficient(
                    radius_si, lambda_tail, float(diffusivity_si)
                )
            )
            beta_tail = k_if_tail / flow_mag_si
            if is_blood:
                buffer_tail = (
                    1.0
                    + max(float(chb_max[seg_idx]), 0.0)
                    * severinghaus_dSdP(c_running / _state.ALPHA_MMHG)
                    / _state.ALPHA_MMHG
                )
                beta_tail = beta_tail / max(float(buffer_tail), 1e-30)
            cout_seg[seg_idx] = np.float32(
                max(
                    c_ext_tail
                    + (c_running - c_ext_tail)
                    * float(np.exp(np.clip(-beta_tail * ds_tail, -150.0, 50.0))),
                    _state.VESS_CONC_FLOOR,
                )
            )
    return cin_seg, cout_seg, c_iv_gl

    @njit(cache=True)
    def _cext_is_local_excluded_numba(
        source_seg: int, row: np.ndarray, count: int
    ) -> int:
        for idx in range(count):
            if int(row[idx]) == source_seg:
                return 1
        return 0

    @njit(cache=True, parallel=True)
    def _compute_cext_batch_numba(
        target_seg_ids: np.ndarray,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        gl_points_si: np.ndarray,
        midpoints_si: np.ndarray,
        radii_si: np.ndarray,
        lambda_iv_gl: np.ndarray,
        q_weighted_gl: np.ndarray,
        seg_cap_gl: np.ndarray,
        exclude_idx: np.ndarray,
        exclude_count: np.ndarray,
        diffusivity_si: float,
        window_factor: float,
    ) -> np.ndarray:
        batch_n = int(target_seg_ids.shape[0])
        m = int(gl_points_si.shape[1])
        out = np.zeros((batch_n, m), dtype=np.float32)
        for flat_idx in prange(batch_n * m):
            batch_idx = flat_idx // m
            target_node = flat_idx - batch_idx * m
            target_seg = int(target_seg_ids[batch_idx])
            target_point_x = float(gl_points_si[target_seg, target_node, 0])
            target_point_y = float(gl_points_si[target_seg, target_node, 1])
            target_point_z = float(gl_points_si[target_seg, target_node, 2])
            target_lambda = float(lambda_iv_gl[target_seg, target_node])
            if target_lambda < 1e-30:
                target_lambda = 1e-30
            target_radius = float(radii_si[target_seg])
            total = 0.0
            cap_max = 0.0
            row_start = int(row_ptr[batch_idx])
            row_end = int(row_ptr[batch_idx + 1])
            excl_count = int(exclude_count[target_seg])
            for pos in range(row_start, row_end):
                source_seg = int(col_idx[pos])
                if (
                    _cext_is_local_excluded_numba(
                        source_seg, exclude_idx[target_seg], excl_count
                    )
                    != 0
                ):
                    continue
                radius_sum = target_radius + float(radii_si[source_seg])
                dx_mid = float(midpoints_si[source_seg, 0]) - float(
                    midpoints_si[target_seg, 0]
                )
                dy_mid = float(midpoints_si[source_seg, 1]) - float(
                    midpoints_si[target_seg, 1]
                )
                dz_mid = float(midpoints_si[source_seg, 2]) - float(
                    midpoints_si[target_seg, 2]
                )
                coarse_r = math.sqrt(
                    dx_mid * dx_mid
                    + dy_mid * dy_mid
                    + dz_mid * dz_mid
                    + radius_sum * radius_sum
                )
                source_lambda_max = 1e-30
                for source_node in range(m):
                    source_lambda_probe = float(lambda_iv_gl[source_seg, source_node])
                    if source_lambda_probe > source_lambda_max:
                        source_lambda_max = source_lambda_probe
                coarse_lambda = (
                    target_lambda
                    if target_lambda >= source_lambda_max
                    else source_lambda_max
                )
                if coarse_r > window_factor * coarse_lambda:
                    continue
                seg_cap = float(seg_cap_gl[source_seg])
                seg_contributed = 0
                for source_node in range(m):
                    source_lambda = float(lambda_iv_gl[source_seg, source_node])
                    if source_lambda < 1e-30:
                        source_lambda = 1e-30
                    source_point_x = float(gl_points_si[source_seg, source_node, 0])
                    source_point_y = float(gl_points_si[source_seg, source_node, 1])
                    source_point_z = float(gl_points_si[source_seg, source_node, 2])
                    dx = source_point_x - target_point_x
                    dy = source_point_y - target_point_y
                    dz = source_point_z - target_point_z
                    r = math.sqrt(dx * dx + dy * dy + dz * dz + radius_sum * radius_sum)
                    if r > window_factor * source_lambda:
                        continue
                    if r < 1e-30:
                        r = 1e-30
                    kernel = math.exp(-r / source_lambda) / (
                        4.0 * math.pi * diffusivity_si * r
                    )
                    total += float(q_weighted_gl[source_seg, source_node]) * kernel
                    seg_contributed = 1
                if seg_contributed == 1 and seg_cap > cap_max:
                    cap_max = seg_cap
            if total < 0.0 or not np.isfinite(total):
                total = 0.0
            if cap_max > 0.0 and total > cap_max:
                total = cap_max
            out[batch_idx, target_node] = np.float32(total)
        return out


_state._CEXT_GPU_KERNEL = None
_state._CEXT_GPU_DIRECT_KERNEL = None
_state._CEXT_GPU_LOCAL_CORR_KERNEL = None
_state._CEXT_ITERATION_CACHE_KERNEL = None
_state._CEXT_HYBRID_BG_KERNEL = None
_state._CEXT_HYBRID_DEPOSIT_KERNEL = None
_state._CEXT_FFT_MOMENT_DEPOSIT_KERNEL = None
_state._CEXT_FFT_MOMENT_DEPOSIT_BATCH_KERNEL = None
_state._CEXT_FFT_DISCRETE_SELF_KERNEL = None
_state._CEXT_FFT_DISCRETE_SELF_FUSED_O2_KERNEL = None
_state._CEXT_TREECODE_KERNEL = None
_state._CEXT_TREECODE_DEPOSIT_KERNEL = None
_state._CEXT_TREECODE_UPSWEEP_KERNEL = None
_state._FROZEN_TOPDOWN_GPU_KERNEL = None
_state._FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL = None
_state._CEXT_TISSUE_CELL_GPU_KERNEL = None


def _get_cext_gpu_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_GPU_KERNEL is not None:
        return _state._CEXT_GPU_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_kernel.cu")
    _state._CEXT_GPU_KERNEL = _state._cp.RawKernel(_state.code, "cext_kernel")
    return _state._CEXT_GPU_KERNEL


def _get_cext_gpu_direct_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._CEXT_GPU_DIRECT_KERNEL is not None:
        return _state._CEXT_GPU_DIRECT_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("cext_direct_kernel.cu")
    _state._CEXT_GPU_DIRECT_KERNEL = _state._cp.RawKernel(
        _state.code, "cext_direct_kernel"
    )
    return _state._CEXT_GPU_DIRECT_KERNEL


def _get_frozen_topdown_gpu_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._FROZEN_TOPDOWN_GPU_KERNEL is not None:
        return _state._FROZEN_TOPDOWN_GPU_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("frozen_topdown_kernel.cu")
    _state._FROZEN_TOPDOWN_GPU_KERNEL = _state._cp.RawKernel(
        _state.code, "frozen_topdown_kernel"
    )
    return _state._FROZEN_TOPDOWN_GPU_KERNEL


def _solve_topdown_ext_frozen_gpu(
    context: dict,
    c_ext_gl: np.ndarray,
    *,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    fluid_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    kernel = _get_frozen_topdown_gpu_kernel()
    gl_order = int(np.asarray(context["gl_t"]).shape[0])
    nseg = int(np.asarray(context["parents"]).shape[0])

    t_upload = perf_counter()
    c_ext_g = _state._cp.asarray(np.asarray(c_ext_gl, dtype=np.float32))
    chb_max_g = _state._cp.asarray(np.asarray(chb_max, dtype=np.float32))
    cin_seg_g = _state._cp.full(
        (nseg,), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    cout_seg_g = _state._cp.full(
        (nseg,), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    c_iv_gl_g = _state._cp.full(
        (nseg, gl_order), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    _state._cp.cuda.Stream.null.synchronize()
    upload_time = perf_counter() - t_upload

    threads = 128
    t_kernel = perf_counter()
    for level_idx in range(int(np.asarray(context["level_offsets"]).shape[0]) - 1):
        start = int(context["level_offsets"][level_idx])
        stop = int(context["level_offsets"][level_idx + 1])
        n_level = max(stop - start, 0)
        if n_level <= 0:
            continue
        seg_ids_g = static["level_order"][start:stop]
        blocks = (n_level + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                seg_ids_g,
                np.int32(n_level),
                static["parents"],
                static["flows_si"],
                static["radii_si"],
                static["lengths_si"],
                static["gl_t"],
                c_ext_g.ravel(),
                np.int32(gl_order),
                np.float32(diffusivity_si),
                np.float32(vmax),
                np.float32(km),
                np.float32(inlet_concentration),
                chb_max_g,
                np.int32(1 if str(fluid_mode).lower() == "blood" else 0),
                np.float32(_state.ALPHA_MMHG),
                np.float32(_state.VESS_CONC_FLOOR),
                static["xs_lut"],
                static["ratio_lut"],
                np.int32(int(np.asarray(_state._KRATIO_XS).size)),
                cin_seg_g,
                cout_seg_g,
                c_iv_gl_g.ravel(),
            ),
        )
    _state._cp.cuda.Stream.null.synchronize()
    kernel_time = perf_counter() - t_kernel

    t_download = perf_counter()
    cin_seg = _state._cp.asnumpy(cin_seg_g)
    cout_seg = _state._cp.asnumpy(cout_seg_g)
    c_iv_gl = _state._cp.asnumpy(c_iv_gl_g)
    _state._cp.cuda.Stream.null.synchronize()
    download_time = perf_counter() - t_download

    timings = {
        "upload_s": float(upload_time),
        "kernel_s": float(kernel_time),
        "download_s": float(download_time),
    }
    return cin_seg, cout_seg, c_iv_gl, timings


def _ensure_graetz_gpu_basis(context: dict) -> dict:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    if int(_state.GRAETZ_N_RADIAL) > 32 or int(_state.GRAETZ_N_MODES) > 16:
        raise RuntimeError(
            "GPU Graetz closure supports up to 32 radial nodes and 16 modes in this v1 kernel."
        )
    table = _graetz_get_basis_table(
        int(_state.GRAETZ_N_RADIAL),
        int(_state.GRAETZ_N_MODES),
        str(_state.GRAETZ_VELOCITY_PROFILE),
    )
    key = (
        table["profile"],
        int(table["n_radial"]),
        int(table["n_modes"]),
        int(table["per_decade"]),
        int(table["key_min"]),
        int(table["key_max"]),
    )
    cached = context.get("graetz_gpu_basis")
    if isinstance(cached, dict) and cached.get("key") == key:
        return cached
    cached = {
        "key": key,
        "mu2": _state._cp.asarray(np.asarray(table["mu2"], dtype=np.float32)),
        "phi": _state._cp.asarray(np.asarray(table["phi"], dtype=np.float32)),
        "project": _state._cp.asarray(np.asarray(table["project"], dtype=np.float32)),
        "cup_weights": _state._cp.asarray(
            np.asarray(table["cup_weights"], dtype=np.float32)
        ),
        "key_min": int(table["key_min"]),
        "key_max": int(table["key_max"]),
        "per_decade": int(table["per_decade"]),
        "min_bi": float(table["min_bi"]),
        "max_bi": float(table["max_bi"]),
        "n_radial": int(table["n_radial"]),
        "n_modes": int(table["n_modes"]),
    }
    context["graetz_gpu_basis"] = cached
    return cached


def _get_frozen_topdown_graetz_gpu_kernel():
    # Mutable runtime state is centralized in configuration.solver_state.
    if _state._FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL is not None:
        return _state._FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    _state.code = load_cuda_source("frozen_topdown_graetz_kernel.cu")
    _state._FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL = _state._cp.RawKernel(
        _state.code, "frozen_topdown_graetz_kernel"
    )
    return _state._FROZEN_TOPDOWN_GRAETZ_GPU_KERNEL


def _solve_topdown_ext_frozen_graetz_gpu(
    context: dict,
    c_ext_gl: np.ndarray,
    *,
    diffusivity_si: float,
    vmax: float,
    km: float,
    inlet_concentration: float,
    chb_max: np.ndarray,
    fluid_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if _state._cp is None:
        raise RuntimeError("CuPy is not available.")
    static = _ensure_cext_gpu_static(context)
    basis = _ensure_graetz_gpu_basis(context)
    kernel = _get_frozen_topdown_graetz_gpu_kernel()
    gl_order = int(np.asarray(context["gl_t"]).shape[0])
    nseg = int(np.asarray(context["parents"]).shape[0])
    lumen_diffusivity_si = (
        float(_lumen_diffusivity_cm2_s_for_fluid(fluid_mode)) * _state.CM2_TO_M2
    )

    t_upload = perf_counter()
    c_ext_g = _state._cp.asarray(np.asarray(c_ext_gl, dtype=np.float32))
    chb_max_g = _state._cp.asarray(np.asarray(chb_max, dtype=np.float32))
    cin_seg_g = _state._cp.full(
        (nseg,), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    cout_seg_g = _state._cp.full(
        (nseg,), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    c_bulk_gl_g = _state._cp.full(
        (nseg, gl_order), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    c_wall_gl_g = _state._cp.full(
        (nseg, gl_order), np.float32(inlet_concentration), dtype=_state._cp.float32
    )
    debug_enabled = bool(_state.GRAETZ_DEBUG_DIAGNOSTICS)
    if debug_enabled:
        debug_gl_fp_iters_g = _state._cp.zeros((nseg, gl_order), dtype=_state._cp.int32)
        debug_gl_fp_resid_g = _state._cp.zeros(
            (nseg, gl_order), dtype=_state._cp.float32
        )
        debug_gl_buffer_g = _state._cp.zeros((nseg, gl_order), dtype=_state._cp.float32)
        debug_tail_fp_iters_g = _state._cp.zeros((nseg,), dtype=_state._cp.int32)
        debug_tail_fp_resid_g = _state._cp.zeros((nseg,), dtype=_state._cp.float32)
        debug_tail_buffer_g = _state._cp.zeros((nseg,), dtype=_state._cp.float32)
    else:
        debug_gl_fp_iters_g = _state._cp.zeros((1,), dtype=_state._cp.int32)
        debug_gl_fp_resid_g = _state._cp.zeros((1,), dtype=_state._cp.float32)
        debug_gl_buffer_g = _state._cp.zeros((1,), dtype=_state._cp.float32)
        debug_tail_fp_iters_g = _state._cp.zeros((1,), dtype=_state._cp.int32)
        debug_tail_fp_resid_g = _state._cp.zeros((1,), dtype=_state._cp.float32)
        debug_tail_buffer_g = _state._cp.zeros((1,), dtype=_state._cp.float32)
    _state._cp.cuda.Stream.null.synchronize()
    upload_time = perf_counter() - t_upload

    threads = 128
    t_kernel = perf_counter()
    for level_idx in range(int(np.asarray(context["level_offsets"]).shape[0]) - 1):
        start = int(context["level_offsets"][level_idx])
        stop = int(context["level_offsets"][level_idx + 1])
        n_level = max(stop - start, 0)
        if n_level <= 0:
            continue
        seg_ids_g = static["level_order"][start:stop]
        blocks = (n_level + threads - 1) // threads
        kernel(
            (blocks,),
            (threads,),
            (
                seg_ids_g,
                np.int32(n_level),
                static["parents"],
                static["flows_si"],
                static["radii_si"],
                static["lengths_si"],
                static["gl_t"],
                c_ext_g.ravel(),
                np.int32(gl_order),
                np.float32(diffusivity_si),
                np.float32(lumen_diffusivity_si),
                np.float32(vmax),
                np.float32(km),
                np.float32(inlet_concentration),
                chb_max_g,
                np.int32(1 if str(fluid_mode).lower() == "blood" else 0),
                np.float32(_state.ALPHA_MMHG),
                np.float32(_state.VESS_CONC_FLOOR),
                static["xs_lut"],
                static["ratio_lut"],
                np.int32(int(np.asarray(_state._KRATIO_XS).size)),
                np.int32(basis["key_min"]),
                np.int32(basis["key_max"]),
                np.int32(basis["per_decade"]),
                np.float32(basis["min_bi"]),
                np.float32(basis["max_bi"]),
                np.int32(basis["n_radial"]),
                np.int32(basis["n_modes"]),
                np.int32(max(int(_state.GRAETZ_MAX_FP_ITERS), 1)),
                np.float32(float(_state.GRAETZ_FP_TOL)),
                basis["mu2"].ravel(),
                basis["phi"].ravel(),
                basis["project"].ravel(),
                basis["cup_weights"].ravel(),
                cin_seg_g,
                cout_seg_g,
                c_bulk_gl_g.ravel(),
                c_wall_gl_g.ravel(),
                np.int32(1 if debug_enabled else 0),
                debug_gl_fp_iters_g.ravel(),
                debug_gl_fp_resid_g.ravel(),
                debug_gl_buffer_g.ravel(),
                debug_tail_fp_iters_g.ravel(),
                debug_tail_fp_resid_g.ravel(),
                debug_tail_buffer_g.ravel(),
            ),
        )
    _state._cp.cuda.Stream.null.synchronize()
    kernel_time = perf_counter() - t_kernel

    t_download = perf_counter()
    cin_seg = _state._cp.asnumpy(cin_seg_g)
    cout_seg = _state._cp.asnumpy(cout_seg_g)
    c_bulk_gl = _state._cp.asnumpy(c_bulk_gl_g)
    c_wall_gl = _state._cp.asnumpy(c_wall_gl_g)
    _state._cp.cuda.Stream.null.synchronize()
    download_time = perf_counter() - t_download

    timings = {
        "upload_s": float(upload_time),
        "kernel_s": float(kernel_time),
        "download_s": float(download_time),
        "graetz_debug_diagnostics": bool(debug_enabled),
        "graetz_tail_buffer_patch": "enabled",
        "graetz_blood_buffer_active": bool(str(fluid_mode).lower() == "blood"),
    }
    if debug_enabled:
        gl_fp = _state._cp.asnumpy(debug_gl_fp_iters_g)
        gl_resid = _state._cp.asnumpy(debug_gl_fp_resid_g)
        gl_buffer = _state._cp.asnumpy(debug_gl_buffer_g)
        tail_fp = _state._cp.asnumpy(debug_tail_fp_iters_g)
        tail_resid = _state._cp.asnumpy(debug_tail_fp_resid_g)
        tail_buffer = _state._cp.asnumpy(debug_tail_buffer_g)
        gl_mask = gl_fp > 0
        tail_mask = tail_fp > 0
        timings.update(
            {
                "graetz_gl_fp_iters_mean": float(np.mean(gl_fp[gl_mask]))
                if np.any(gl_mask)
                else 0.0,
                "graetz_gl_fp_iters_max": float(np.max(gl_fp[gl_mask]))
                if np.any(gl_mask)
                else 0.0,
                "graetz_gl_fp_resid_mean": float(np.mean(gl_resid[gl_mask]))
                if np.any(gl_mask)
                else 0.0,
                "graetz_gl_fp_resid_max": float(np.max(gl_resid[gl_mask]))
                if np.any(gl_mask)
                else 0.0,
                "graetz_gl_buffer_mean": float(np.mean(gl_buffer[gl_mask]))
                if np.any(gl_mask)
                else 1.0,
                "graetz_gl_buffer_max": float(np.max(gl_buffer[gl_mask]))
                if np.any(gl_mask)
                else 1.0,
                "graetz_tail_fp_iters_mean": float(np.mean(tail_fp[tail_mask]))
                if np.any(tail_mask)
                else 0.0,
                "graetz_tail_fp_iters_max": float(np.max(tail_fp[tail_mask]))
                if np.any(tail_mask)
                else 0.0,
                "graetz_tail_fp_resid_mean": float(np.mean(tail_resid[tail_mask]))
                if np.any(tail_mask)
                else 0.0,
                "graetz_tail_fp_resid_max": float(np.max(tail_resid[tail_mask]))
                if np.any(tail_mask)
                else 0.0,
                "graetz_tail_buffer_mean": float(np.mean(tail_buffer[tail_mask]))
                if np.any(tail_mask)
                else 1.0,
                "graetz_tail_buffer_max": float(np.max(tail_buffer[tail_mask]))
                if np.any(tail_mask)
                else 1.0,
            }
        )
    return cin_seg, cout_seg, c_bulk_gl, c_wall_gl, timings


__all__ = [
    "_build_topdown_level_slices_numba",
    "_build_local_exclusion_arrays_numba",
    "_solve_topdown_ext_frozen_numba",
    "_solve_topdown_ext_frozen_python",
    "_get_cext_gpu_kernel",
    "_get_cext_gpu_direct_kernel",
    "_get_frozen_topdown_gpu_kernel",
    "_solve_topdown_ext_frozen_gpu",
    "_ensure_graetz_gpu_basis",
    "_get_frozen_topdown_graetz_gpu_kernel",
    "_solve_topdown_ext_frozen_graetz_gpu",
]
