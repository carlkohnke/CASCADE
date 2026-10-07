"""Coupled graph flow, RBC transport and rheology with CPU/CUDA parity."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from cascade.configuration import solver_state as state

from .gpu import solve_kirchhoff_gpu, solve_pressure_dirichlet_gpu
from .hematocrit import _normalize_hematocrit_model
from .hematocrit_gpu import compute_network_hematocrit_gpu, segment_viscosity_gpu


_GPU_HEMATOCRIT_CONTEXTS = ({}, {})
_DEVICE_PRESSURE_ENABLE = True


def solve_network_blood(
    up,
    down,
    radii,
    lengths,
    inlets,
    outlets,
    inlet_flow,
    mu_base,
    *,
    pressure_boundary=None,
    pressure_solver="gpu_amg",
    accel="gpu",
):
    """Honor configured Pries/rheology iteration and retain graph caches.

    GPU solves return NumPy arrays at the existing public boundary. Topology,
    AMG interpolation, and hematocrit schedules are reused where valid. A flow
    direction change invalidates the RBC schedule; resistance changes refresh
    AMG matrix values without reusing stale conductances.
    """
    cp = state._cp if accel == "gpu" else np
    if cp is None:
        raise RuntimeError("GPU blood hemodynamics requires CUDA/CuPy")
    if accel not in {"cpu", "gpu"}:
        raise ValueError("blood hemodynamics accel must be 'cpu' or 'gpu'")
    to_cpu = cp.asnumpy if accel == "gpu" else np.asarray
    hematocrit = compute_network_hematocrit_gpu
    if accel == "cpu":
        from .hematocrit_network import compute_network_hematocrit_cpu

        hematocrit = compute_network_hematocrit_cpu
    started = perf_counter()
    model = _normalize_hematocrit_model()
    radius_gpu, length_gpu = cp.asarray(radii), cp.asarray(lengths)
    hd = cp.full(len(radii), state.HD_DISCHARGE, dtype=cp.float64)
    context = {}
    iterations = max(int(state.HEMATOCRIT_FLOW_ITERATIONS), 0)
    previous_flow = None
    log = []
    converged = model != "pries_secomb" or iterations == 0
    gpu_pressure = str(pressure_solver).lower() in {"gpu", "gpu_amg"} or (
        str(pressure_solver).lower() == "auto" and accel == "gpu"
    )
    for iteration in range(max(iterations, 1)):
        pass_started = perf_counter()
        if accel == "gpu":
            mu = segment_viscosity_gpu(
                radius_gpu,
                hd,
                mu_base,
                pries=model == "pries_secomb",
                return_device=True,
            )
        else:
            from .rheology import eta_rel_pries, pries_secomb_viscor_cgs

            diameter = 2 * radius_gpu * 1e4
            mu = (
                pries_secomb_viscor_cgs(diameter, hd)
                if model == "pries_secomb"
                else mu_base * eta_rel_pries(diameter, hd)
            )
        resistance = (
            8.0 * mu * length_gpu / (np.pi * cp.maximum(radius_gpu, 1e-12) ** 4)
        )
        use_device_pressure = (
            _DEVICE_PRESSURE_ENABLE and accel == "gpu" and gpu_pressure
        )
        if use_device_pressure:
            try:
                from cupyx.scipy import sparse  # noqa: F401
            except (ImportError, OSError):
                use_device_pressure = False
        if not use_device_pressure:
            resistance = to_cpu(resistance)
        viscosity_s = perf_counter() - pass_started
        pressure_started = perf_counter()
        if use_device_pressure:
            from .device_pressure import solve_device_pressure

            pressure, flow = solve_device_pressure(
                up,
                down,
                resistance,
                inlets,
                outlets,
                inlet_flow,
                pressure_boundary=pressure_boundary,
            )
        elif pressure_boundary is None:
            if gpu_pressure:
                implementation = solve_kirchhoff_gpu
                kwargs = {}
            else:
                from .kirchhoff import solve_kirchhoff

                implementation = solve_kirchhoff
                kwargs = {
                    "solver_mode": "spsolve"
                    if pressure_solver == "auto"
                    else pressure_solver
                }
            pressure, flow, _, _, _ = implementation(
                up,
                down,
                resistance,
                inlets,
                inlet_flow,
                outlets,
                boundary_condition=state.KIRCHHOFF_BC_MODE,
                **kwargs,
            )
        else:
            pin, pout = pressure_boundary
            implementation = solve_pressure_dirichlet_gpu
            if not gpu_pressure:
                from .linear_system import _solve_pressure_dirichlet

                implementation = _solve_pressure_dirichlet
            pressure, flow = implementation(
                up, down, resistance, inlets, pin, outlets, pout
            )
        pressure_s = perf_counter() - pressure_started
        pressure_details = {}
        if gpu_pressure:
            from .gpu import LAST_GPU_FLOW_TIMINGS

            pressure_details = dict(LAST_GPU_FLOW_TIMINGS)
        hematocrit_started = perf_counter()
        new_hd, tube = hematocrit(
            up,
            down,
            flow,
            radii,
            hd_root=state.HD_DISCHARGE,
            model=model,
            context=(
                _GPU_HEMATOCRIT_CONTEXTS[iteration]
                if accel == "gpu" and iteration < 2
                else context
            ),
            **({"return_device": True} if accel == "gpu" else {}),
        )
        change = float(cp.max(cp.abs(new_hd - hd)).item()) if len(radii) else 0.0
        hematocrit_s = perf_counter() - hematocrit_started
        qchange = (
            (
                float(cp.max(cp.abs(flow - previous_flow)).item())
                if use_device_pressure
                else float(np.max(abs(flow - previous_flow)))
            )
            if previous_flow is not None
            else float("inf")
        )
        log.append(
            {
                "iteration": iteration + 1,
                "max_hd_delta": change,
                "max_flow_delta_cm3_s": qchange,
                "viscosity_resistance_s": viscosity_s,
                "pressure_s": pressure_s,
                "pressure_details": pressure_details,
                "hematocrit_s": hematocrit_s,
            }
        )
        if iterations == 0 or model != "pries_secomb":
            hd = new_hd
            break
        # Keep the returned HD matched to returned flow: the unrelaxed
        # conservative propagation is the physical state, not a partial mix.
        hd_tol, q_tol = (
            float(state.HEMATOCRIT_HDTOL),
            float(state.HEMATOCRIT_QTOL_NL_MIN),
        )
        if (
            (hd_tol >= 0.0 or q_tol >= 0.0)
            and (hd_tol < 0.0 or change < hd_tol)
            and (q_tol < 0.0 or qchange < q_tol * 1e-6 / 60.0)
        ):
            hd, converged = new_hd, True
            break
        previous_flow = flow.copy()
        relaxation = float(np.clip(state.HEMATOCRIT_RELAXATION, 0.0, 1.0))
        hd = hd + relaxation * (new_hd - hd)
    # Always report flux-conservative HD for the final solved flow.
    hd, tube = to_cpu(new_hd), to_cpu(tube)
    if use_device_pressure:
        pressure, flow = to_cpu(pressure), to_cpu(flow)
    timing = {
        "backend": f"{accel}-network",
        "flow_backend": "gpu-amg" if gpu_pressure else str(pressure_solver),
        "total_s": perf_counter() - started,
        "iterations": len(log),
        "converged": converged,
        "history": log,
    }
    return pressure, flow, hd, tube, timing
