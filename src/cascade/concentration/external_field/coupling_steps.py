"""External-field nonlinear coupling steps.

Each step propagates lumen concentrations, evaluates the resulting external
field, and returns the residual needed by the outer iteration.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cascade.configuration import solver_state as _state
from cascade.configuration.solver_state import _lumen_diffusivity_cm2_s_for_fluid
from cascade.concentration.vessel.graetz import _graetz_get_basis_table
from cascade.concentration.vessel.greens import (
    _interfacial_transfer_coefficient,
    _lambda_if_from_civ,
    severinghaus_dSdP,
)

from .diagnostics import _validate_cext_gpu_batch
from .direct import (
    _compute_cext_batch_cpu,
    _compute_cext_batch_gpu_into,
    _compute_cext_direct_gpu,
    _context_uses_cext_gpu_direct,
)
from .frozen import (
    _solve_network_ext_frozen_numba,
    _solve_topdown_ext_frozen_gpu,
    _solve_topdown_ext_frozen_graetz_gpu,
    _solve_topdown_ext_frozen_numba,
    _solve_topdown_ext_frozen_python,
)
from .geometry import (
    _coerce_candidate_batches_for_gpu,
    _ensure_cext_candidate_batches,
    _split_cext_candidate_batch,
)
from .state import _build_cext_iteration_cache


def _run_network_ext_frozen_step(
    context: dict,
    ext_state: dict,
    *,
    inlet_concentration: float,
    vmax: float,
    km: float,
    chb_max: np.ndarray,
    fluid_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Solve graph concentrations for a fixed external field.

    The FFT field iteration is topology-independent.  This replaces the
    tree-only parent sweep with nodal oxygen mass balance while retaining the
    same per-segment wall-exchange integration.
    """
    t_total = perf_counter()
    closure_mode = str(_state.LUMEN_WALL_CLOSURE or "wellmixed").strip().lower()
    if closure_mode not in {"wellmixed", "graetz"}:
        raise ValueError("--lumen-wall-closure must be 'wellmixed' or 'graetz'.")

    topology = context.get("network_topology") or {}
    prox = np.asarray(topology["prox_ids"], dtype=np.int64).reshape(-1)
    dist = np.asarray(topology["dist_ids"], dtype=np.int64).reshape(-1)
    flows_si = np.asarray(context["flows_si"], dtype=float).reshape(-1)
    q = np.abs(flows_si)
    up = prox.copy()
    down = dist.copy()
    flip = flows_si < 0.0
    up[flip], down[flip] = dist[flip], prox[flip]
    nseg = int(q.size)
    nnode = int(max(up.max(), down.max()) + 1) if nseg else 0
    inlet_nodes = {int(node) for node in topology.get("inlet_nodes", ())}
    inlet_value = float(inlet_concentration)
    gl_t = np.asarray(context["gl_t"], dtype=float)
    c_ext_gl = np.asarray(ext_state["c_ext_gl"], dtype=float)
    radii_si = np.asarray(context["radii_si"], dtype=float)
    lengths_si = np.asarray(context["lengths_si"], dtype=float)
    diffusivity_si = float(context["diffusivity_si"])
    chb = np.asarray(chb_max, dtype=float)
    valid = q > 1e-30

    graetz_basis = None
    lumen_diffusivity_si = 0.0
    if closure_mode == "graetz":
        graetz_basis = _graetz_get_basis_table(
            int(_state.GRAETZ_N_RADIAL),
            int(_state.GRAETZ_N_MODES),
            str(_state.GRAETZ_VELOCITY_PROFILE),
        )
        lumen_diffusivity_si = (
            float(_lumen_diffusivity_cm2_s_for_fluid(fluid_mode))
            * _state.CM2_TO_M2
        )

    if closure_mode == "wellmixed" and _state._HAVE_NUMBA and _state.CONC_USE_NUMBA:
        topology_cache = context.get("network_frozen_topology_cache")
        if topology_cache is None:
            valid_edges = np.flatnonzero(valid).astype(np.int64, copy=False)
            if valid_edges.size:
                edge_order = np.argsort(down[valid_edges], kind="stable")
                incoming_edges = np.asarray(
                    valid_edges[edge_order], dtype=np.int64
                )
                incoming_counts = np.bincount(
                    down[valid_edges], minlength=nnode
                ).astype(np.int64, copy=False)
            else:
                incoming_edges = np.empty((0,), dtype=np.int64)
                incoming_counts = np.zeros((nnode,), dtype=np.int64)
            incoming_offsets = np.empty((nnode + 1,), dtype=np.int64)
            incoming_offsets[0] = 0
            np.cumsum(incoming_counts, out=incoming_offsets[1:])
            sum_out = np.zeros((nnode,), dtype=np.float64)
            sum_in = np.zeros((nnode,), dtype=np.float64)
            np.add.at(sum_out, up[valid], q[valid])
            np.add.at(sum_in, down[valid], q[valid])
            inlet_mask = np.zeros((nnode,), dtype=np.uint8)
            for inlet in inlet_nodes:
                if 0 <= inlet < nnode:
                    inlet_mask[inlet] = 1
            topology_cache = {
                "incoming_offsets": incoming_offsets,
                "incoming_edges": incoming_edges,
                "sum_out": sum_out,
                "sum_in": sum_in,
                "inlet_mask": inlet_mask,
            }
            context["network_frozen_topology_cache"] = topology_cache
        cached_cin = np.asarray(ext_state.get("cin_seg", ()), dtype=np.float64)
        cin, cout, civ, iterations = _solve_network_ext_frozen_numba(
            np.asarray(up, dtype=np.int64),
            np.asarray(q, dtype=np.float64),
            topology_cache["incoming_offsets"],
            topology_cache["incoming_edges"],
            topology_cache["sum_out"],
            topology_cache["sum_in"],
            topology_cache["inlet_mask"],
            cached_cin,
            np.asarray(radii_si, dtype=np.float64),
            np.asarray(lengths_si, dtype=np.float64),
            np.asarray(gl_t, dtype=np.float64),
            np.asarray(c_ext_gl, dtype=np.float64),
            diffusivity_si,
            vmax,
            km,
            inlet_value,
            np.asarray(chb, dtype=np.float64),
            1 if fluid_mode == "blood" else 0,
            float(_state.OMEGA),
        )
        elapsed = perf_counter() - t_total
        _state._LAST_CEXT_FROZEN_STEP_TIMINGS = {
            "backend": "cpu-network-numba",
            "lumen_wall_closure": "wellmixed",
            "upload_s": 0.0,
            "kernel_s": float(elapsed),
            "download_s": 0.0,
            "transfer_s": 0.0,
            "total_s": float(elapsed),
            "network_iterations": int(iterations),
        }
        ext_state["c_iv_gl"] = civ
        ext_state["c_bulk_gl"] = civ
        ext_state["c_wall_gl"] = civ
        return cin, cout, civ, "cpu-network-numba", 0.0

    incoming: list[list[int]] = [[] for _ in range(nnode)]
    sum_out = np.zeros((nnode,), dtype=float)
    sum_in = np.zeros((nnode,), dtype=float)
    for edge_idx in range(nseg):
        if not valid[edge_idx]:
            continue
        incoming[int(down[edge_idx])].append(edge_idx)
        sum_out[int(up[edge_idx])] += q[edge_idx]
        sum_in[int(down[edge_idx])] += q[edge_idx]

    def graetz_step(
        profile: np.ndarray,
        c_external: float,
        step: float,
        edge_idx: int,
        bulk_in: float,
    ) -> tuple[np.ndarray, float, float]:
        if (
            step <= 0.0
            or q[edge_idx] <= 1e-30
            or radii_si[edge_idx] <= 1e-30
            or lumen_diffusivity_si <= 1e-30
        ):
            return profile.copy(), float(bulk_in), float(profile[-1])
        basis = graetz_basis
        wall_guess = max(float(profile[-1]), 0.0)
        accepted = profile.copy()
        accepted_bulk = float(bulk_in)
        accepted_wall = wall_guess
        lambda_guess = float(
            _lambda_if_from_civ(wall_guess, diffusivity_si, vmax, km)
        )
        buffer = 1.0
        if fluid_mode == "blood":
            buffer += (
                max(float(chb[edge_idx]), 0.0)
                * severinghaus_dSdP(float(bulk_in) / _state.ALPHA_MMHG)
                / _state.ALPHA_MMHG
            )
        velocity = q[edge_idx] / max(
            np.pi * radii_si[edge_idx] * radii_si[edge_idx], 1e-30
        )
        xi = lumen_diffusivity_si * step / max(
            buffer * radii_si[edge_idx] ** 2 * velocity, 1e-30
        )
        for _ in range(max(int(_state.GRAETZ_MAX_FP_ITERS), 1)):
            k_if = float(
                _interfacial_transfer_coefficient(
                    radii_si[edge_idx], lambda_guess, diffusivity_si
                )
            )
            bi = k_if / max(2.0 * np.pi * lumen_diffusivity_si, 1e-30)
            clipped_bi = min(max(bi, float(basis["min_bi"])), float(basis["max_bi"]))
            key = int(round(np.log10(clipped_bi) * int(basis["per_decade"])))
            basis_idx = min(
                max(key - int(basis["key_min"]), 0),
                int(np.asarray(basis["mu2"]).shape[0]) - 1,
            )
            project = np.asarray(basis["project"][basis_idx], dtype=float)
            modes = np.asarray(basis["phi"][basis_idx], dtype=float)
            decay = np.exp(
                -np.minimum(np.asarray(basis["mu2"][basis_idx], dtype=float) * xi, 80.0)
            )
            coefficients = project @ (profile - c_external)
            trial = np.maximum(
                c_external + modes @ (coefficients * decay),
                _state.VESS_CONC_FLOOR,
            )
            cup = np.asarray(basis["cup_weights"][basis_idx], dtype=float)
            accepted = trial
            accepted_bulk = max(float(cup @ trial), _state.VESS_CONC_FLOOR)
            accepted_wall = float(trial[-1])
            lambda_new = float(
                _lambda_if_from_civ(accepted_wall, diffusivity_si, vmax, km)
            )
            relative = abs(lambda_new - lambda_guess) / max(lambda_guess, 1e-30)
            if relative < float(_state.GRAETZ_FP_TOL):
                break
            lambda_guess = max(0.5 * (lambda_guess + lambda_new), 1e-30)
        return accepted, accepted_bulk, accepted_wall

    def propagate(
        node_conc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cin = np.maximum(node_conc[up], _state.VESS_CONC_FLOOR).astype(np.float32)
        cout = np.asarray(cin, dtype=np.float32).copy()
        civ = np.repeat(cin[:, None], int(gl_t.size), axis=1).astype(np.float32)
        cwall = civ.copy()
        for edge_idx in range(nseg):
            if not valid[edge_idx]:
                continue
            c_running = float(cin[edge_idx])
            previous_s = 0.0
            flow_mag = max(float(q[edge_idx]), 1e-30)
            profile = (
                np.full(
                    (int(graetz_basis["n_radial"]),),
                    c_running,
                    dtype=float,
                )
                if closure_mode == "graetz"
                else None
            )
            for node_idx, t_value in enumerate(gl_t):
                target_s = float(t_value) * float(lengths_si[edge_idx])
                step = max(target_s - previous_s, 0.0)
                c_external = max(float(c_ext_gl[edge_idx, node_idx]), 0.0)
                if closure_mode == "graetz":
                    profile, c_running, wall = graetz_step(
                        profile, c_external, step, edge_idx, c_running
                    )
                    civ[edge_idx, node_idx] = np.float32(c_running)
                    cwall[edge_idx, node_idx] = np.float32(wall)
                    previous_s = target_s
                    continue
                lambda_if = float(
                    _lambda_if_from_civ(c_running, diffusivity_si, vmax, km)
                )
                k_if = float(
                    _interfacial_transfer_coefficient(
                        radii_si[edge_idx], lambda_if, diffusivity_si
                    )
                )
                beta = k_if / flow_mag
                if fluid_mode == "blood":
                    buffer = (
                        1.0
                        + max(float(chb[edge_idx]), 0.0)
                        * severinghaus_dSdP(c_running / _state.ALPHA_MMHG)
                        / _state.ALPHA_MMHG
                    )
                    beta /= max(float(buffer), 1e-30)
                c_running = max(
                    c_external
                    + (c_running - c_external)
                    * float(np.exp(np.clip(-beta * step, -150.0, 50.0))),
                    _state.VESS_CONC_FLOOR,
                )
                civ[edge_idx, node_idx] = np.float32(c_running)
                previous_s = target_s
            tail_step = max(float(lengths_si[edge_idx]) - previous_s, 0.0)
            c_external = max(float(c_ext_gl[edge_idx, -1]), 0.0) if gl_t.size else 0.0
            if closure_mode == "graetz":
                profile, c_running, _wall = graetz_step(
                    profile, c_external, tail_step, edge_idx, c_running
                )
                cout[edge_idx] = np.float32(c_running)
                continue
            lambda_if = float(_lambda_if_from_civ(c_running, diffusivity_si, vmax, km))
            k_if = float(
                _interfacial_transfer_coefficient(
                    radii_si[edge_idx], lambda_if, diffusivity_si
                )
            )
            beta = k_if / flow_mag
            if fluid_mode == "blood":
                buffer = (
                    1.0
                    + max(float(chb[edge_idx]), 0.0)
                    * severinghaus_dSdP(c_running / _state.ALPHA_MMHG)
                    / _state.ALPHA_MMHG
                )
                beta /= max(float(buffer), 1e-30)
            cout[edge_idx] = np.float32(
                max(
                    c_external
                    + (c_running - c_external)
                    * float(np.exp(np.clip(-beta * tail_step, -150.0, 50.0))),
                    _state.VESS_CONC_FLOOR,
                )
            )
        return cin, cout, civ, cwall

    node_conc = np.full((nnode,), inlet_value, dtype=float)
    cached_cin = np.asarray(ext_state.get("cin_seg", ()), dtype=float).reshape(-1)
    if cached_cin.size == nseg:
        for edge_idx in range(nseg):
            if valid[edge_idx] and int(up[edge_idx]) not in inlet_nodes:
                node_conc[int(up[edge_idx])] = max(
                    float(cached_cin[edge_idx]), _state.VESS_CONC_FLOOR
                )
    for inlet in inlet_nodes:
        if 0 <= inlet < nnode:
            node_conc[inlet] = inlet_value

    iterations = 0
    for iteration_number in range(1, 101):
        iterations = iteration_number
        _, cout, _, _ = propagate(node_conc)
        updated = node_conc.copy()
        for node in range(nnode):
            if node in inlet_nodes or not incoming[node]:
                continue
            denominator = sum_out[node] if sum_out[node] > 1e-30 else sum_in[node]
            if denominator > 1e-30:
                updated[node] = (
                    sum(q[edge] * float(cout[edge]) for edge in incoming[node])
                    / denominator
                )
        delta = float(np.max(np.abs(updated - node_conc))) if nnode else 0.0
        node_conc += float(_state.OMEGA) * (updated - node_conc)
        for inlet in inlet_nodes:
            if 0 <= inlet < nnode:
                node_conc[inlet] = inlet_value
        if delta < 1e-6:
            break

    cin, cout, civ, cwall = propagate(node_conc)
    elapsed = perf_counter() - t_total
    _state._LAST_CEXT_FROZEN_STEP_TIMINGS = {
        "backend": "cpu-network",
        "lumen_wall_closure": closure_mode,
        "upload_s": 0.0,
        "kernel_s": float(elapsed),
        "download_s": 0.0,
        "transfer_s": 0.0,
        "total_s": float(elapsed),
        "network_iterations": int(iterations),
    }
    ext_state["c_iv_gl"] = civ
    ext_state["c_bulk_gl"] = civ
    ext_state["c_wall_gl"] = cwall
    return cin, cout, civ, "cpu-network", 0.0


def _run_topdown_ext_frozen_step(
    context: dict,
    ext_state: dict,
    *,
    inlet_concentration: float,
    vmax: float,
    km: float,
    chb_max: np.ndarray,
    fluid_mode: str,
    frozen_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    if context.get("network_topology") is not None:
        return _run_network_ext_frozen_step(
            context,
            ext_state,
            inlet_concentration=inlet_concentration,
            vmax=vmax,
            km=km,
            chb_max=chb_max,
            fluid_mode=fluid_mode,
        )
    # Mutable runtime state is centralized in configuration.solver_state.
    t_step_total = perf_counter()
    frozen_transfer = 0.0
    backend_local = str(frozen_backend)
    closure_mode = str(_state.LUMEN_WALL_CLOSURE or "wellmixed").strip().lower()
    if closure_mode not in ("wellmixed", "graetz"):
        raise ValueError("--lumen-wall-closure must be 'wellmixed' or 'graetz'.")
    if backend_local == "gpu":
        try:
            if closure_mode == "graetz":
                cin_seg, cout_seg, c_iv_gl, c_wall_gl, frozen_timings = (
                    _solve_topdown_ext_frozen_graetz_gpu(
                        context,
                        np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
                        diffusivity_si=float(context["diffusivity_si"]),
                        vmax=float(vmax),
                        km=float(km),
                        inlet_concentration=float(inlet_concentration),
                        chb_max=np.asarray(chb_max, dtype=np.float32),
                        fluid_mode=fluid_mode,
                    )
                )
            else:
                cin_seg, cout_seg, c_iv_gl, frozen_timings = (
                    _solve_topdown_ext_frozen_gpu(
                        context,
                        np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
                        diffusivity_si=float(context["diffusivity_si"]),
                        vmax=float(vmax),
                        km=float(km),
                        inlet_concentration=float(inlet_concentration),
                        chb_max=np.asarray(chb_max, dtype=np.float32),
                        fluid_mode=fluid_mode,
                    )
                )
                c_wall_gl = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_bulk_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
            ext_state["c_wall_gl"] = np.asarray(c_wall_gl, dtype=np.float32)
            frozen_transfer = float(frozen_timings.get("upload_s", 0.0)) + float(
                frozen_timings.get("download_s", 0.0)
            )
            _state._LAST_CEXT_FROZEN_STEP_TIMINGS = {
                "backend": backend_local,
                "lumen_wall_closure": closure_mode,
                "upload_s": float(frozen_timings.get("upload_s", 0.0)),
                "kernel_s": float(frozen_timings.get("kernel_s", 0.0)),
                "download_s": float(frozen_timings.get("download_s", 0.0)),
                "transfer_s": float(frozen_transfer),
                "total_s": float(perf_counter() - t_step_total),
            }
            for key, value in frozen_timings.items():
                if str(key).startswith("graetz_"):
                    _state._LAST_CEXT_FROZEN_STEP_TIMINGS[str(key)] = value
            return cin_seg, cout_seg, c_iv_gl, backend_local, frozen_transfer
        except Exception:
            if str(_state.CEXT_FROZEN_ACCEL_MODE).lower() == "gpu":
                raise
            backend_local = "cpu"
    if closure_mode == "graetz":
        raise RuntimeError(
            "Graetz lumen closure currently requires the GPU frozen top-down backend."
        )
    t_cpu = perf_counter()
    if _state._HAVE_NUMBA and _state.CONC_USE_NUMBA:
        cin_seg, cout_seg, c_iv_gl = _solve_topdown_ext_frozen_numba(
            np.asarray(context["level_order"], dtype=np.int32),
            np.asarray(context["level_offsets"], dtype=np.int32),
            np.asarray(context["parents"], dtype=np.int32),
            np.asarray(context["flows_si"], dtype=np.float32),
            np.asarray(context["radii_si"], dtype=np.float32),
            np.asarray(context["lengths_si"], dtype=np.float32),
            np.asarray(context["gl_t"], dtype=np.float32),
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
            float(context["diffusivity_si"]),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(chb_max, dtype=np.float32),
            1 if fluid_mode == "blood" else 0,
        )
    else:
        cin_seg, cout_seg, c_iv_gl = _solve_topdown_ext_frozen_python(
            np.asarray(context["level_order"], dtype=np.int32),
            np.asarray(context["level_offsets"], dtype=np.int32),
            np.asarray(context["parents"], dtype=np.int32),
            np.asarray(context["flows_si"], dtype=np.float32),
            np.asarray(context["radii_si"], dtype=np.float32),
            np.asarray(context["lengths_si"], dtype=np.float32),
            np.asarray(context["gl_t"], dtype=np.float32),
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32),
            float(context["diffusivity_si"]),
            float(vmax),
            float(km),
            float(inlet_concentration),
            np.asarray(chb_max, dtype=np.float32),
            fluid_mode == "blood",
        )
    _state._LAST_CEXT_FROZEN_STEP_TIMINGS = {
        "backend": backend_local,
        "lumen_wall_closure": closure_mode,
        "upload_s": 0.0,
        "kernel_s": float(perf_counter() - t_cpu),
        "download_s": 0.0,
        "transfer_s": 0.0,
        "total_s": float(perf_counter() - t_step_total),
    }
    ext_state["c_iv_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    ext_state["c_bulk_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    ext_state["c_wall_gl"] = np.asarray(c_iv_gl, dtype=np.float32)
    return cin_seg, cout_seg, c_iv_gl, backend_local, frozen_transfer


def _compute_cext_image_step(
    context: dict,
    ext_state: dict,
    *,
    backend: str,
    source_plan: dict | None = None,
    target_plan: dict | None = None,
    rebuild_cache: bool = True,
) -> tuple[np.ndarray, str, float, float]:
    if rebuild_cache:
        _build_cext_iteration_cache(context, ext_state)
    backend_local = str(backend)
    kernel_total = 0.0
    transfer_total = 0.0
    if backend_local == "gpu":
        if _context_uses_cext_gpu_direct(context):
            try:
                c_ext_total, c_ext_cap, kernel_total, transfer_total = (
                    _compute_cext_direct_gpu(
                        context,
                        ext_state,
                        source_plan=source_plan,
                        target_plan=target_plan,
                    )
                )
                c_ext_new = np.asarray(c_ext_total, dtype=np.float32)
                cap_arr = np.asarray(c_ext_cap, dtype=np.float32)
                positive_cap = cap_arr > 0.0
                if np.any(positive_cap):
                    c_ext_new = np.where(
                        positive_cap, np.minimum(c_ext_new, cap_arr), c_ext_new
                    ).astype(np.float32, copy=False)
                c_ext_new = np.maximum(c_ext_new, np.float32(0.0)).astype(
                    np.float32, copy=False
                )
                return c_ext_new, backend_local, kernel_total, transfer_total
            except Exception:
                if str(_state.CEXT_ACCEL_MODE).lower() == "gpu":
                    raise
                backend_local = "cpu"
                context["candidate_query_mode"] = (
                    "kdtree" if context.get("candidate_kdtree") is not None else "grid"
                )
        batches, _ = _ensure_cext_candidate_batches(context)
        batches = _coerce_candidate_batches_for_gpu(context)
        t_transfer = perf_counter()
        lambda_iv_g = _state._cp.asarray(
            np.asarray(ext_state["lambda_iv_gl"], dtype=np.float32)
        )
        q_weighted_g = _state._cp.asarray(
            np.asarray(ext_state["q_weighted_gl"], dtype=np.float32)
        )
        mono2_weight_g = _state._cp.asarray(
            np.asarray(
                ext_state.get(
                    "mono2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
        )
        dipole2_weight_g = _state._cp.asarray(
            np.asarray(
                ext_state.get(
                    "dipole2_weight_gl", np.zeros_like(ext_state["q_weighted_gl"])
                ),
                dtype=np.float32,
            )
        )
        seg_cap_g = _state._cp.asarray(
            np.asarray(ext_state["seg_cap_gl"], dtype=np.float32)
        )
        c_ext_new_g = _state._cp.zeros(
            np.asarray(ext_state["c_ext_gl"], dtype=np.float32).shape,
            dtype=_state._cp.float32,
        )
        _state._cp.cuda.Stream.null.synchronize()
        transfer_total += perf_counter() - t_transfer

        batch_idx = 0
        gpu_failed = False
        while batch_idx < len(batches):
            batch = batches[batch_idx]
            if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
                batch_idx += 1
                continue
            try:
                t_batch_kernel = perf_counter()
                _compute_cext_batch_gpu_into(
                    context,
                    batch,
                    lambda_iv_g,
                    q_weighted_g,
                    mono2_weight_g,
                    dipole2_weight_g,
                    seg_cap_g,
                    c_ext_new_g,
                )
                _state._cp.cuda.Stream.null.synchronize()
                kernel_total += perf_counter() - t_batch_kernel
                if int(_state.CEXT_GPU_VALIDATE_SEGMENTS) > 0:
                    t_validate_transfer = perf_counter()
                    gpu_out = _state._cp.asnumpy(
                        c_ext_new_g[batch["target_start"] : batch["target_stop"]]
                    )
                    transfer_total += perf_counter() - t_validate_transfer
                    _validate_cext_gpu_batch(
                        context,
                        ext_state,
                        np.asarray(batch["target_seg_ids"], dtype=np.int32),
                        np.asarray(batch["row_ptr"], dtype=np.int32),
                        np.asarray(batch["col_idx"], dtype=np.int32),
                        gpu_out,
                    )
                batch_idx += 1
            except Exception as exc:
                if _state._cp is None or not isinstance(
                    exc, _state._cp.cuda.memory.OutOfMemoryError
                ):
                    if str(_state.CEXT_ACCEL_MODE).lower() == "gpu":
                        raise
                    backend_local = "cpu"
                    gpu_failed = True
                    break
                if int(np.asarray(batch["target_seg_ids"]).size) <= 1:
                    if str(_state.CEXT_ACCEL_MODE).lower() == "gpu":
                        raise
                    backend_local = "cpu"
                    gpu_failed = True
                    break
                split_batches = _split_cext_candidate_batch(context, batch)
                batches[batch_idx : batch_idx + 1] = split_batches
                context["candidate_batches"] = batches
                _coerce_candidate_batches_for_gpu(context)
        if backend_local == "gpu" and not gpu_failed:
            t_download = perf_counter()
            c_ext_new = _state._cp.asnumpy(c_ext_new_g)
            transfer_total += perf_counter() - t_download
            context["candidate_batches"] = batches
            return c_ext_new, backend_local, kernel_total, transfer_total
        c_ext_new = np.zeros_like(ext_state["c_ext_gl"])
        for batch in batches:
            if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
                continue
            t_batch_kernel = perf_counter()
            out_batch = _compute_cext_batch_cpu(
                context,
                ext_state,
                np.asarray(batch["target_seg_ids"], dtype=np.int32),
                np.asarray(batch["row_ptr"], dtype=np.int32),
                np.asarray(batch["col_idx"], dtype=np.int32),
            )
            kernel_total += perf_counter() - t_batch_kernel
            c_ext_new[int(batch["target_start"]) : int(batch["target_stop"])] = (
                out_batch
            )
        context["candidate_batches"] = batches
        return c_ext_new, backend_local, kernel_total, transfer_total

    batches, _ = _ensure_cext_candidate_batches(context)
    c_ext_new = np.zeros_like(ext_state["c_ext_gl"])
    for batch in batches:
        if int(np.asarray(batch["target_seg_ids"]).size) <= 0:
            continue
        t_batch_kernel = perf_counter()
        out_batch = _compute_cext_batch_cpu(
            context,
            ext_state,
            np.asarray(batch["target_seg_ids"], dtype=np.int32),
            np.asarray(batch["row_ptr"], dtype=np.int32),
            np.asarray(batch["col_idx"], dtype=np.int32),
        )
        kernel_total += perf_counter() - t_batch_kernel
        c_ext_new[int(batch["target_start"]) : int(batch["target_stop"])] = out_batch
    return c_ext_new, backend_local, kernel_total, transfer_total


__all__ = [
    "_run_network_ext_frozen_step",
    "_run_topdown_ext_frozen_step",
    "_compute_cext_image_step",
]
