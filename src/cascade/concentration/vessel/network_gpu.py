"""GPU graph transport: independent edge propagation and CSR junction gather.

Geometry/topology uploads are cached across frozen-field steps. Concentrations,
edge propagation, junction fluxes, and iteration residuals remain on the GPU.
Only scalar stopping checks and the final public arrays cross to the CPU.
"""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np

from cascade.accelerators.cuda import load_cuda_source
from cascade.configuration import solver_state as state
from cascade.flow.graph_schedule import graph_schedule

from .oxygen_transport import (
    junction_flux_residual,
    nodal_capacity,
    total_content_enabled,
)

# One graph only, bounded like the pressure hierarchy cache. Exact comparisons
# handle in-place edits and avoid requiring callers to manage cache lifetimes.
_STANDALONE_GRAPH_CACHE = {}


def resolve_network_accel(value=None):
    mode = str(value or state.NETWORK_TRANSPORT_ACCEL).lower()
    if mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("network_transport_accel must be 'auto', 'cpu', or 'gpu'")
    if mode == "cpu":
        return "cpu"
    try:
        available = (
            state._cp is not None and state._cp.cuda.runtime.getDeviceCount() > 0
        )
    except (RuntimeError, OSError):
        available = False
    if mode == "gpu" and not available:
        raise RuntimeError(
            "GPU network transport was requested but CUDA/CuPy is unavailable"
        )
    return "gpu" if available else "cpu"


def solve_network_greens_gpu(
    up,
    down,
    q,
    radii,
    lengths,
    capacity,
    inlets,
    outlets,
    inlet,
    fluid,
    diffusivity,
    vmax,
    km,
    max_iter,
    tol,
    omega,
):
    """FP64 standalone transport, exact DAG sweep or explicit cyclic iteration."""
    cp = state._cp
    started = perf_counter()
    cached = _STANDALONE_GRAPH_CACHE
    inputs = (up, down, q, capacity, np.asarray(list(inlets), dtype=np.int64))
    graph_reused = (
        bool(cached)
        and cached["device"] == int(cp.cuda.Device().id)
        and all(np.array_equal(old, new) for old, new in zip(cached["inputs"], inputs))
    )
    if not graph_reused:
        cached.clear()
        cached.update(
            device=int(cp.cuda.Device().id),
            inputs=tuple(array.copy() for array in inputs),
            graph=gpu_graph_arrays(up, down, q, capacity, inlets, dtype=np.float64),
        )
    graph = cached["graph"]
    n = graph["nnode"]
    current = cp.full(n, inlet, dtype=cp.float64)
    updated, delta, residual = (
        cp.empty_like(current),
        cp.empty_like(current),
        cp.empty_like(current),
    )
    cin, cout = cp.empty(len(q), cp.float64), cp.empty(len(q), cp.float64)
    ids_all = cp.arange(len(q), dtype=cp.int32)
    radii_gpu = cp.asarray(radii * state.CM_TO_M)
    lengths_gpu = cp.asarray(lengths * state.CM_TO_M)
    flows_gpu = graph["q"] * state.CM3_TO_M3
    xs, ys = cp.asarray(state._KRATIO_XS), cp.asarray(state._KRATIO_YS)
    conserve = total_content_enabled(fluid)
    inlet_up = cp.asarray(np.isin(up, list(inlets)))
    inlet_down = cp.asarray(np.isin(down, list(inlets)))
    sinks = (
        list(outlets)
        if outlets
        else np.flatnonzero(np.bincount(up, weights=q, minlength=n) == 0.0)
    )
    sink_down, sink_up = (
        cp.asarray(np.isin(down, sinks)),
        cp.asarray(np.isin(up, sinks)),
    )
    internal = cp.asarray(
        (np.bincount(up, weights=q, minlength=n) > 0.0)
        & (np.bincount(down, weights=q, minlength=n) > 0.0)
    )
    internal[cp.asarray(list(inlets), dtype=cp.int32)] = False

    def record_metrics():
        def content(c):
            p = c / state.ALPHA_MMHG
            numerator = p * p * p + 150.0 * p
            return c + graph["capacity"] * numerator / (numerator + 23400.0)

        a, b = content(cin), content(cout)
        incoming = cp.bincount(graph["down"], weights=graph["q"] * b, minlength=n)
        outgoing = cp.bincount(graph["up"], weights=graph["q"] * a, minlength=n)
        true_res = cp.max(
            cp.where(
                internal,
                cp.abs(incoming - outgoing)
                / cp.maximum(cp.maximum(cp.abs(incoming), cp.abs(outgoing)), 1e-30),
                0.0,
            )
        )
        if not conserve:
            a, b = cin, cout
        mi = cp.sum(graph["q"][inlet_up] * a[inlet_up]) - cp.sum(
            graph["q"][inlet_down] * b[inlet_down]
        )
        mo = cp.sum(graph["q"][sink_down] * b[sink_down]) - cp.sum(
            graph["q"][sink_up] * a[sink_up]
        )
        drop = cp.sum(graph["q"] * (a - b))
        values = cp.asnumpy(cp.stack((mi, mo, drop, mi - mo - drop, true_res)))
        for key, value in zip(
            ("M_in", "M_out", "M_drop", "MB_resid", "total_oxygen_junction_rel_resid"),
            values,
        ):
            history[key].append(float(value))

    def propagate(ids):
        if not len(ids):
            return
        _kernel("network_greens_kernel")(
            ((len(ids) + 127) // 128,),
            (128,),
            (
                np.int32(len(ids)),
                ids,
                graph["up"],
                current,
                flows_gpu,
                radii_gpu,
                lengths_gpu,
                graph["capacity"],
                np.float64(diffusivity * state.CM2_TO_M2),
                np.float64(vmax),
                np.float64(km),
                np.float64(state.ALPHA_MMHG),
                np.int32(fluid == "blood"),
                np.int32(max(state.AXIAL_BLOOD_STEPS, 1)),
                xs,
                ys,
                np.int32(len(xs)),
                cin,
                cout,
            ),
        )

    history = {
        k: []
        for k in (
            "iter",
            "max_delta",
            "M_in",
            "M_out",
            "M_drop",
            "MB_resid",
            "total_oxygen_junction_rel_resid",
        )
    }
    schedule = graph["schedule"]
    converged = False
    if schedule is not None:
        nodes, offsets, edges, edge_offsets = schedule
        for level in range(len(offsets) - 1):
            _mix(
                graph,
                cout,
                current,
                current,
                delta,
                residual,
                inlet,
                1.0,
                conserve,
                double=True,
                nodes=nodes[offsets[level] : offsets[level + 1]],
            )
            propagate(edges[edge_offsets[level] : edge_offsets[level + 1]])
        iteration, change, converged = 1, 0.0, True
    else:
        for iteration in range(1, max_iter + 1):
            propagate(ids_all)
            _mix(
                graph,
                cout,
                current,
                updated,
                delta,
                residual,
                inlet,
                omega,
                conserve,
                double=True,
            )
            change = float(cp.max(delta).item())
            balance = float(cp.max(residual).item())
            history["iter"].append(iteration)
            history["max_delta"].append(change)
            record_metrics()
            current, updated = updated, current
            if change < tol and balance < tol:
                converged = True
                break
        propagate(ids_all)
    cin, cout, nodes_cpu = cp.asnumpy(cin), cp.asnumpy(cout), cp.asnumpy(current)
    from .oxygen_transport import oxygen_content

    inlet_edges = np.isin(up, list(inlets))
    if outlets:
        sink_edges = np.isin(down, list(outlets))
    else:
        sink_edges = np.bincount(up, weights=q, minlength=n)[down] == 0.0
    a, b = (
        (oxygen_content(cin, capacity), oxygen_content(cout, capacity))
        if conserve
        else (cin, cout)
    )
    mi, mo, drop = (
        float(
            np.sum(q[inlet_edges] * a[inlet_edges])
            - np.sum(q[np.isin(down, list(inlets))] * b[np.isin(down, list(inlets))])
        ),
        float(
            np.sum(q[sink_edges] * b[sink_edges])
            - (
                np.sum(q[np.isin(up, list(outlets))] * a[np.isin(up, list(outlets))])
                if outlets
                else 0.0
            )
        ),
        float(np.sum(q * (a - b))),
    )
    for key, value in {
        "M_in": mi,
        "M_out": mo,
        "M_drop": drop,
        "MB_resid": mi - mo - drop,
        "total_oxygen_junction_rel_resid": junction_flux_residual(
            up, down, q, cin, cout, capacity, n, boundary_nodes=inlets
        ),
    }.items():
        if schedule is not None:
            history[key] = [value]
        else:
            history[key][-1] = value
    if schedule is not None:
        history["iter"], history["max_delta"] = [1], [0.0]
    state._LAST_NETWORK_GPU_TIMINGS = {
        "backend": "gpu-network",
        "total_s": perf_counter() - started,
        "iterations": iteration,
        "converged": converged,
        "schedule": "dag" if schedule is not None else "cyclic-fixed-point",
        "graph_reused": graph_reused,
    }
    return cin, cout, nodes_cpu, history


@lru_cache(maxsize=4)
def _kernel(name, options=()):
    if state._cp is None:
        raise RuntimeError("CuPy is unavailable")
    return state._cp.RawKernel(load_cuda_source(name + ".cu"), name, options=options)


def gpu_graph_arrays(up, down, q, capacity, inlet_nodes, *, dtype=np.float32):
    cp = state._cp
    nnode = int(max(up.max(), down.max()) + 1) if len(up) else 0
    if nnode > np.iinfo(np.int32).max or len(up) > np.iinfo(np.int32).max:
        raise ValueError("GPU graph exceeds int32 CSR index capacity")
    valid = q > 1e-30
    ids = np.flatnonzero(valid)
    edges = ids[np.argsort(down[ids], kind="stable")]
    counts = np.bincount(down[ids], minlength=nnode)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    qout = np.bincount(up, weights=q, minlength=nnode)
    qin = np.bincount(down, weights=q, minlength=nnode)
    denominator = np.where(qout > 0.0, qout, qin)
    mask = np.zeros(nnode, np.uint8)
    mask[np.asarray(list(inlet_nodes), dtype=int)] = 1
    convert = lambda a: cp.asarray(np.asarray(a, dtype=dtype))
    schedule = graph_schedule(up, down, q)
    if schedule is not None:
        nodes, node_offsets, scheduled_edges, edge_offsets = schedule
        schedule = (
            cp.asarray(nodes),
            node_offsets,
            cp.asarray(scheduled_edges),
            edge_offsets,
        )
    return {
        "device_id": int(cp.cuda.Device().id),
        "nnode": nnode,
        "node_ids": cp.arange(nnode, dtype=cp.int32),
        "schedule": schedule,
        "up": cp.asarray(up, dtype=cp.int32),
        "down": cp.asarray(down, dtype=cp.int32),
        "offsets": cp.asarray(offsets),
        "edges": cp.asarray(edges, dtype=cp.int32),
        "q": convert(q),
        "capacity": convert(capacity),
        "denominator": convert(denominator),
        "node_capacity": convert(nodal_capacity(up, down, q, capacity, nnode)),
        "inlet_mask": cp.asarray(mask),
    }


def _mix(
    graph,
    cout,
    current,
    updated,
    delta,
    residual,
    inlet,
    omega,
    conserve,
    *,
    double=False,
    nodes=None,
):
    nodes = graph["node_ids"] if nodes is None else nodes
    if not len(nodes):
        return
    _kernel("network_junction_kernel", ("-DCASCADE_NODE_FLOAT64",) if double else ())(
        ((len(nodes) + 127) // 128,),
        (128,),
        (
            np.int32(len(nodes)),
            nodes,
            graph["offsets"],
            graph["edges"],
            graph["q"],
            cout,
            graph["capacity"],
            graph["denominator"],
            graph["node_capacity"],
            graph["inlet_mask"],
            current,
            updated,
            delta,
            residual,
            np.float64(state.ALPHA_MMHG),
            np.float64(inlet),
            np.float64(omega),
            np.int32(conserve),
        ),
    )


def solve_network_ext_gpu(
    context, ext_state, *, inlet_concentration, vmax, km, chb_max, fluid_mode
):
    cp = state._cp
    started = perf_counter()
    topology = context["network_topology"]
    flows = np.asarray(context["flows_si"], dtype=float)
    q = abs(flows)
    up = np.asarray(topology["prox_ids"], dtype=np.int64).copy()
    down = np.asarray(topology["dist_ids"], dtype=np.int64).copy()
    flip = flows < 0
    up[flip], down[flip] = down[flip], up[flip]
    nseg, order = len(q), len(context["gl_t"])
    closure = str(state.LUMEN_WALL_CLOSURE).lower()
    if closure not in {"wellmixed", "graetz"}:
        raise ValueError("lumen_wall_closure must be 'wellmixed' or 'graetz'")
    graph = context.get("network_gpu_graph")
    geometry_values = (
        up,
        down,
        q > 1e-30,
        np.asarray(context["radii_si"]),
        np.asarray(context["lengths_si"]),
        np.asarray(context["gl_t"]),
        np.asarray(topology.get("inlet_nodes", ())),
    )
    if (
        graph is None
        or graph["device_id"] != int(cp.cuda.Device().id)
        or not all(
            np.array_equal(old, new)
            for old, new in zip(graph["geometry_host"], geometry_values)
        )
    ):
        graph = gpu_graph_arrays(up, down, q, chb_max, topology.get("inlet_nodes", ()))
        graph.update(
            seg_ids=cp.arange(nseg, dtype=cp.int32),
            flows=cp.asarray(flows, dtype=cp.float32),
            radii=cp.asarray(context["radii_si"], dtype=cp.float32),
            lengths=cp.asarray(context["lengths_si"], dtype=cp.float32),
            gl_t=cp.asarray(context["gl_t"], dtype=cp.float32),
            xs=cp.asarray(state._KRATIO_XS, dtype=cp.float32),
            ratio=cp.asarray(state._KRATIO_YS, dtype=cp.float32),
            geometry_host=tuple(value.copy() for value in geometry_values),
            capacity_host=np.asarray(chb_max).copy(),
            flow_host=q.copy(),
        )
        graph["workspace"] = {
            name: cp.empty(shape, cp.float32)
            for name, shape in (
                ("current", (graph["nnode"],)),
                ("updated", (graph["nnode"],)),
                ("delta", (graph["nnode"],)),
                ("residual", (graph["nnode"],)),
                ("cin", (nseg,)),
                ("cout", (nseg,)),
                ("bulk", (nseg, order)),
                ("wall", (nseg, order)),
                ("external", (nseg, order)),
            )
        }
        context["network_gpu_graph"] = graph
    # The schedule depends on orientation and active edges, not flow magnitudes.
    # Refresh weights in place so captured kernels retain valid device pointers.
    flow_changed = not np.array_equal(graph["flow_host"], q)
    if flow_changed:
        graph["q"].set(np.asarray(q, dtype=np.float32))
        graph["flows"].set(np.asarray(flows, dtype=np.float32))
        qout = np.bincount(up, weights=q, minlength=graph["nnode"])
        qin = np.bincount(down, weights=q, minlength=graph["nnode"])
        graph["denominator"].set(np.asarray(np.where(qout > 0, qout, qin), np.float32))
        graph["flow_host"] = q.copy()
    # Capacity also changes when a geometry is reused with another O2 option.
    capacity_changed = not np.array_equal(graph["capacity_host"], chb_max)
    if capacity_changed:
        graph["capacity"].set(np.asarray(chb_max, dtype=np.float32))
        graph["capacity_host"] = np.asarray(chb_max).copy()
    if flow_changed or capacity_changed:
        graph["node_capacity"].set(
            np.asarray(
                nodal_capacity(up, down, q, chb_max, graph["nnode"]), dtype=np.float32
            )
        )
    nnode = graph["nnode"]
    work = graph["workspace"]
    current = work["current"]
    current.fill(inlet_concentration)
    cached = np.asarray(ext_state.get("cin_seg", ()), dtype=np.float32)
    if graph["schedule"] is None and cached.shape == (nseg,):
        # Deterministic initialization; shared upstream edges have the same inlet C.
        values = np.full(nnode, inlet_concentration, np.float32)
        values[up[q > 1e-30]] = cached[q > 1e-30]
        values[list(topology.get("inlet_nodes", ()))] = inlet_concentration
        current.set(values)
    updated, delta, residual = (work[key] for key in ("updated", "delta", "residual"))
    cin, cout, bulk, wall, external = (
        work[key] for key in ("cin", "cout", "bulk", "wall", "external")
    )
    external.set(np.asarray(ext_state["c_ext_gl"], dtype=np.float32))
    name = (
        "frozen_topdown_graetz_kernel"
        if closure == "graetz"
        else "frozen_topdown_kernel"
    )
    edge_kernel = _kernel(name, ("-DCASCADE_NETWORK",))
    basis = None
    if closure == "graetz":
        from cascade.concentration.external_field.frozen import _ensure_graetz_gpu_basis

        basis = _ensure_graetz_gpu_basis(context)
    cp.cuda.Stream.null.synchronize()
    upload_time = perf_counter() - started

    def propagate(ids=None):
        ids = graph["seg_ids"] if ids is None else ids
        if not len(ids):
            return
        args = (
            ids,
            np.int32(len(ids)),
            graph["up"],
            current,
            graph["flows"],
            graph["radii"],
            graph["lengths"],
            graph["gl_t"],
            external.ravel(),
            np.int32(order),
            np.float32(context["diffusivity_si"]),
        )
        if basis is not None:
            args += (
                np.float32(
                    state._lumen_diffusivity_cm2_s_for_fluid(fluid_mode)
                    * state.CM2_TO_M2
                ),
            )
        args += (
            np.float32(vmax),
            np.float32(km),
            np.float32(inlet_concentration),
            graph["capacity"],
            np.int32(fluid_mode == "blood"),
            np.float32(state.ALPHA_MMHG),
            np.float32(state.VESS_CONC_FLOOR),
            graph["xs"],
            graph["ratio"],
            np.int32(len(state._KRATIO_XS)),
        )
        if basis is not None:
            args += (
                np.int32(basis["key_min"]),
                np.int32(basis["key_max"]),
                np.int32(basis["per_decade"]),
                np.float32(basis["min_bi"]),
                np.float32(basis["max_bi"]),
                np.int32(basis["n_radial"]),
                np.int32(basis["n_modes"]),
                np.int32(max(state.GRAETZ_MAX_FP_ITERS, 1)),
                np.float32(state.GRAETZ_FP_TOL),
                basis["mu2"].ravel(),
                basis["phi"].ravel(),
                basis["project"].ravel(),
                basis["cup_weights"].ravel(),
            )
        args += (cin, cout, bulk.ravel())
        if basis is not None:
            # Disabled debug outputs are never dereferenced by the kernel.
            args += (
                wall.ravel(),
                np.int32(0),
                graph["seg_ids"],
                bulk,
                bulk,
                graph["seg_ids"],
                bulk,
                bulk,
            )
        edge_kernel(((len(ids) + 127) // 128,), (128,), args)

    kernel_started = perf_counter()
    conserve = total_content_enabled(fluid_mode)
    max_delta = 0.0
    iterations = 0
    schedule = graph["schedule"]
    if schedule is not None:
        nodes, offsets, edges, edge_offsets = schedule

        def sweep():
            for level in range(len(offsets) - 1):
                _mix(
                    graph,
                    cout,
                    current,
                    current,
                    delta,
                    residual,
                    inlet_concentration,
                    1.0,
                    conserve,
                    nodes=nodes[offsets[level] : offsets[level + 1]],
                )
                propagate(edges[edge_offsets[level] : edge_offsets[level + 1]])

        capture_key = (
            closure,
            fluid_mode,
            float(inlet_concentration),
            float(vmax),
            float(km),
            float(context["diffusivity_si"]),
            float(state.ALPHA_MMHG),
            float(state.VESS_CONC_FLOOR),
            conserve,
            repr(basis["key"]) if basis is not None else None,
            float(state._lumen_diffusivity_cm2_s_for_fluid(fluid_mode)),
            int(state.GRAETZ_MAX_FP_ITERS),
            float(state.GRAETZ_FP_TOL),
        )
        if graph.get("capture_key") != capture_key:
            edge_kernel.compile()
            _kernel("network_junction_kernel").compile()
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                sweep()
                graph["sweep_graph"] = stream.end_capture()
            graph["capture_key"] = capture_key
        graph["sweep_graph"].launch()
        iterations, max_delta = 1, 0.0
    else:
        for iterations in range(1, 101):
            propagate()
            _mix(
                graph,
                cout,
                current,
                updated,
                delta,
                residual,
                inlet_concentration,
                state.OMEGA,
                conserve,
            )
            max_delta = float(cp.max(delta).item()) if nnode else 0.0
            current, updated = updated, current
            if max_delta < 1e-6:
                break
        propagate()
    cp.cuda.Stream.null.synchronize()
    kernel_time = perf_counter() - kernel_started
    download_started = perf_counter()
    cin_cpu, cout_cpu, bulk_cpu = cp.asnumpy(cin), cp.asnumpy(cout), cp.asnumpy(bulk)
    wall_cpu = cp.asnumpy(wall) if basis is not None else bulk_cpu
    ext_state["c_iv_gl"] = bulk_cpu
    ext_state["c_bulk_gl"] = bulk_cpu
    ext_state["c_wall_gl"] = wall_cpu
    ext_state["junction_total_oxygen_rel_resid"] = junction_flux_residual(
        up,
        down,
        q,
        cin_cpu,
        cout_cpu,
        chb_max,
        nnode,
        boundary_nodes=topology.get("inlet_nodes", ()),
    )
    download_time = perf_counter() - download_started
    state._LAST_CEXT_FROZEN_STEP_TIMINGS = {
        "backend": "gpu-network",
        "lumen_wall_closure": closure,
        "upload_s": upload_time,
        "kernel_s": kernel_time,
        "download_s": download_time,
        "transfer_s": upload_time + download_time,
        "total_s": perf_counter() - started,
        "network_iterations": iterations,
        "network_max_delta": max_delta,
        "network_converged": max_delta < 1e-6,
        "network_schedule": "dag" if schedule is not None else "cyclic-fixed-point",
        "network_levels": len(schedule[1]) - 1 if schedule is not None else 0,
        "network_captured_sweep": schedule is not None,
    }
    return cin_cpu, cout_cpu, bulk_cpu, "gpu-network", upload_time + download_time
