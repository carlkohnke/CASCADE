"""GPU pressure solving with a reusable algebraic multigrid hierarchy.

Sparse hierarchy construction is a one-time CPU preprocessing operation.
All V-cycle smoothing, restriction, prolongation, Krylov vectors, reductions,
and pressure-to-flow updates run on CUDA. Repeated Galerkin matrix refreshes
use cupyx sparse multiplication when available, with a CPU refresh fallback.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import blake2b
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from cascade.configuration import solver_state as state

LAST_GPU_FLOW_TIMINGS = {}
_PRESSURE_CACHE = {}
_ASSEMBLY_CACHE = {}

_CSR_SOURCE = r"""
extern "C" __global__ void csr_mv(int n, const int* row, const int* col,
    const double* val, const double* x, double* y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double sum=0.;
    for(int j=row[i];j<row[i+1];++j) sum+=val[j]*x[col[j]];
    y[i]=sum;
}
extern "C" __global__ void csr_jacobi(int n, const int* row, const int* col,
    const double* val, const double* x, const double* b, const double* invdiag,
    double omega, double* y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double sum=0.;
    for(int j=row[i];j<row[i+1];++j) sum+=val[j]*x[col[j]];
    y[i]=x[i]+omega*invdiag[i]*(b[i]-sum);
}
extern "C" __global__ void csr_residual(int n, const int* row, const int* col,
    const double* val, const double* x, const double* b, double* y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double sum=0.;
    for(int j=row[i];j<row[i+1];++j) sum+=val[j]*x[col[j]];
    y[i]=b[i]-sum;
}
extern "C" __global__ void jacobi_initial(int n, const double* b,
    const double* invdiag, double omega, double* x) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=omega*invdiag[i]*b[i];
}
extern "C" __global__ void dense_coarse(int n, const double* a,
    const double* b, double* x) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    double sum=0.;
    for(int j=0;j<n;++j) sum+=a[i*n+j]*b[j];
    x[i]=sum;
}
extern "C" __global__ void cg_update(int n, double* x, double* r,
    const double* p, const double* ap, const double* rz, const double* pap) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { double alpha=rz[0]/fmax(pap[0],1e-300);
        x[i]+=alpha*p[i]; r[i]-=alpha*ap[i]; }
}
extern "C" __global__ void cg_direction(int n, const double* z, double* p,
    const double* rz, const double* newrz) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) p[i]=z[i]+(newrz[0]/fmax(rz[0],1e-300))*p[i];
}
extern "C" __global__ void csr_mv_warp(int n, const int* row, const int* col,
    const double* val, const double* x, double* y) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(i>=n) return;
    double sum=0.;
    for(int j=row[i]+lane;j<row[i+1];j+=32) sum+=val[j]*x[col[j]];
    for(int offset=16;offset;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0) y[i]=sum;
}
"""


@lru_cache(maxsize=1)
def _csr_kernel():
    return state._cp.RawKernel(_CSR_SOURCE, "csr_mv")


@lru_cache(maxsize=7)
def _cycle_kernel(name):
    return state._cp.RawKernel(_CSR_SOURCE, name)


class GPUAMGSolver:
    """Retain device matrices and workspaces for repeated right-hand sides.

    Rebuild when resistance/topology changes; reusing a stale hierarchy or
    conductance matrix is never permitted. FP64 and true residual validation
    protect low-flow branches from silent mixed-precision loss.
    """

    def __init__(
        self,
        matrix,
        *,
        fused_cycle=True,
        capture_cycle=True,
        weight_factor=1.8,
        capture_krylov=True,
        device_update=True,
        warp_threshold=16,
    ):
        import pyamg

        cp = state._cp
        if cp is None:
            raise RuntimeError("GPU pressure solver requires CUDA/CuPy")
        started = perf_counter()
        matrix = matrix.tocsr().astype(np.float64)
        # Symmetric diagonal scaling handles the wide vessel conductance range.
        diagonal = matrix.diagonal()
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            raise ValueError(
                "Pressure matrix has isolated nodes or invalid conductance"
            )
        scale = 1.0 / np.sqrt(diagonal)
        scaled = sp.diags(scale) @ matrix @ sp.diags(scale)
        hierarchy = pyamg.smoothed_aggregation_solver(
            scaled,
            symmetry="symmetric",
            max_coarse=32,
            presmoother=("jacobi", {"omega": 4.0 / 3.0, "iterations": 2}),
            postsmoother=("jacobi", {"omega": 4.0 / 3.0, "iterations": 2}),
        )
        self.scale = cp.asarray(scale)
        self.levels = []
        self.transfers = []
        self.pattern = (matrix.indptr.copy(), matrix.indices.copy())
        self.fused_cycle = fused_cycle
        self.capture_cycle = capture_cycle and fused_cycle
        self.weight_factor = float(weight_factor)
        if not 0.0 < self.weight_factor < 2.0:
            raise ValueError("Jacobi bound multiplier must be between zero and two")
        self.cycle_graph = None
        self.capture_krylov = capture_krylov and fused_cycle
        self.krylov_graph = None
        self.device_update = device_update
        self.warp_threshold = warp_threshold

        def upload(a):
            a = a.tocsr()
            return (
                a.shape[0],
                cp.asarray(a.indptr, dtype=cp.int32),
                cp.asarray(a.indices, dtype=cp.int32),
                cp.asarray(a.data, dtype=cp.float64),
            )

        for level in hierarchy.levels:
            item = {
                "A": upload(level.A),
                "invdiag": cp.asarray(1.0 / level.A.diagonal()),
                "x": cp.zeros(level.A.shape[0]),
                "tmp": cp.zeros(level.A.shape[0]),
                "rhs": cp.zeros(level.A.shape[0]),
                "residual": cp.zeros(level.A.shape[0]),
                "correction": cp.zeros(level.A.shape[0]),
            }
            if hasattr(level, "P"):
                self.transfers.append((level.P.tocsr(), level.R.tocsr()))
                item.update(P=upload(level.P), R=upload(level.R))
                # Jacobi relaxation weight bounded by Gershgorin spectral radius.
                bound = (
                    np.asarray(abs(level.A).sum(axis=1)).ravel() / level.A.diagonal()
                )
                item["omega"] = self.weight_factor / float(bound.max())
            self.levels.append(item)
        # Tiny coarse inverse is built once; its matvec also executes on CUDA.
        self.coarse_inverse = cp.asarray(
            np.linalg.inv(hierarchy.levels[-1].A.toarray())
        )
        cp.cuda.Stream.null.synchronize()
        self.setup_seconds = perf_counter() - started
        self.last_setup_seconds = self.setup_seconds
        self.previous_solution = None
        size = matrix.shape[0]
        self.krylov = {
            name: cp.empty(size, cp.float64) for name in ("x", "r", "p", "ap", "dot")
        }
        self.krylov.update(
            {name: cp.empty((), cp.float64) for name in ("rz", "newrz", "pap")}
        )

    def update(self, matrix):
        """Reuse aggregation for viscosity changes; refresh every matrix value."""
        cp = state._cp
        started = perf_counter()
        # Captured graphs contain matrix pointers and relaxation scalars by value.
        self.cycle_graph = None
        self.krylov_graph = None
        scale = 1.0 / np.sqrt(matrix.diagonal())
        a = (sp.diags(scale) @ matrix @ sp.diags(scale)).tocsr()
        self.scale.set(scale)
        if self.device_update:
            try:
                from cupyx.scipy import sparse as gpu_sparse
            except (ImportError, OSError):
                # Minimal CUDA installs still support the custom-kernel path.
                gpu_sparse = None
            if gpu_sparse is not None:
                a_gpu = gpu_sparse.csr_matrix(
                    (
                        cp.asarray(a.data),
                        cp.asarray(a.indices, dtype=cp.int32),
                        cp.asarray(a.indptr, dtype=cp.int32),
                    ),
                    shape=a.shape,
                )
                for idx, level in enumerate(self.levels):
                    level["A"] = (
                        a_gpu.shape[0],
                        a_gpu.indptr,
                        a_gpu.indices,
                        a_gpu.data,
                    )
                    diagonal = a_gpu.diagonal()
                    level["invdiag"] = 1.0 / diagonal
                    if idx < len(self.transfers):
                        bound = cp.asarray(abs(a_gpu).sum(axis=1)).ravel() / diagonal
                        level["omega"] = self.weight_factor / float(
                            cp.max(bound).item()
                        )
                        p, r = self.transfers[idx]
                        pg, rg = level["P"], level["R"]
                        p_gpu = gpu_sparse.csr_matrix(
                            (pg[3], pg[2], pg[1]), shape=p.shape
                        )
                        r_gpu = gpu_sparse.csr_matrix(
                            (rg[3], rg[2], rg[1]), shape=r.shape
                        )
                        a_gpu = (r_gpu @ a_gpu @ p_gpu).tocsr()
                self.coarse_inverse.set(np.linalg.inv(a_gpu.get().toarray()))
                cp.cuda.get_current_stream().synchronize()
                self.setup_seconds = perf_counter() - started
                self.last_setup_seconds = self.setup_seconds
                LAST_GPU_FLOW_TIMINGS["hierarchy_update_backend"] = "gpu"
                return
        for idx, level in enumerate(self.levels):
            level["A"] = (
                a.shape[0],
                cp.asarray(a.indptr, dtype=cp.int32),
                cp.asarray(a.indices, dtype=cp.int32),
                cp.asarray(a.data, dtype=cp.float64),
            )
            level["invdiag"].set(1.0 / a.diagonal())
            if idx < len(self.transfers):
                bound = np.asarray(abs(a).sum(axis=1)).ravel() / a.diagonal()
                level["omega"] = self.weight_factor / float(bound.max())
                p, r = self.transfers[idx]
                a = (r @ a @ p).tocsr()
        self.coarse_inverse.set(np.linalg.inv(a.toarray()))
        cp.cuda.Stream.null.synchronize()
        self.setup_seconds = perf_counter() - started
        self.last_setup_seconds = self.setup_seconds
        LAST_GPU_FLOW_TIMINGS["hierarchy_update_backend"] = "cpu"

    def matvec(self, matrix, x, out=None):
        cp = state._cp
        n, row, col, val = matrix
        if out is None:
            out = cp.empty(n, cp.float64)
        if (
            self.warp_threshold is not None
            and len(val) / max(n, 1) >= self.warp_threshold
        ):
            _cycle_kernel("csr_mv_warp")(
                ((n * 32 + 127) // 128,), (128,), (np.int32(n), row, col, val, x, out)
            )
        else:
            _csr_kernel()(
                ((n + 127) // 128,), (128,), (np.int32(n), row, col, val, x, out)
            )
        return out

    def _cycle(self, idx, b):
        cp = state._cp
        level = self.levels[idx]
        x = level["x"]
        if idx == len(self.levels) - 1:
            x[:] = cp.sum(self.coarse_inverse * b[None, :], axis=1)
            return x
        x.fill(0.0)
        for _ in range(2):
            self.matvec(level["A"], x, level["tmp"])
            x += level["omega"] * level["invdiag"] * (b - level["tmp"])
        self.matvec(level["A"], x, level["tmp"])
        restricted = self.matvec(level["R"], b - level["tmp"])
        correction = self._cycle(idx + 1, restricted)
        x += self.matvec(level["P"], correction)
        for _ in range(2):
            self.matvec(level["A"], x, level["tmp"])
            x += level["omega"] * level["invdiag"] * (b - level["tmp"])
        return x

    def _fused_cycle(self, idx, b):
        """Allocation-free symmetric V-cycle with fused sparse smoothing."""
        level = self.levels[idx]
        x, tmp = level["x"], level["tmp"]
        n = len(x)
        launch = ((n + 127) // 128,), (128,)
        if idx == len(self.levels) - 1:
            _cycle_kernel("dense_coarse")(
                *launch, (np.int32(n), self.coarse_inverse, b, x)
            )
            return x
        a = level["A"]
        # The first Jacobi step starts from zero; its matvec is identically zero.
        _cycle_kernel("jacobi_initial")(
            *launch, (np.int32(n), b, level["invdiag"], np.float64(level["omega"]), x)
        )
        smooth = _cycle_kernel("csr_jacobi")
        smooth(
            *launch,
            (
                np.int32(n),
                *a[1:],
                x,
                b,
                level["invdiag"],
                np.float64(level["omega"]),
                tmp,
            ),
        )
        _cycle_kernel("csr_residual")(
            *launch, (np.int32(n), *a[1:], tmp, b, level["residual"])
        )
        restricted = self.matvec(
            level["R"], level["residual"], self.levels[idx + 1]["rhs"]
        )
        correction = self._fused_cycle(idx + 1, restricted)
        self.matvec(level["P"], correction, level["correction"])
        tmp += level["correction"]
        smooth(
            *launch,
            (
                np.int32(n),
                *a[1:],
                tmp,
                b,
                level["invdiag"],
                np.float64(level["omega"]),
                x,
            ),
        )
        smooth(
            *launch,
            (
                np.int32(n),
                *a[1:],
                x,
                b,
                level["invdiag"],
                np.float64(level["omega"]),
                tmp,
            ),
        )
        return tmp

    def _precondition(self, b):
        if not self.fused_cycle:
            return self._cycle(0, b)
        if not self.capture_cycle:
            return self._fused_cycle(0, b)
        cp = state._cp
        rhs = self.levels[0]["rhs"]
        cp.copyto(rhs, b)
        if self.cycle_graph is None:
            # Compile all kernels before capture, on the calling stream.
            self._fused_cycle(0, rhs)
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                self._fused_cycle(0, rhs)
                self.cycle_graph = stream.end_capture()
        self.cycle_graph.launch()
        return self.levels[0]["tmp"] if len(self.levels) > 1 else self.levels[0]["x"]

    def _krylov_step(self):
        cp, w = state._cp, self.krylov
        n = len(w["x"])
        launch = ((n + 127) // 128,), (128,)
        self.matvec(self.levels[0]["A"], w["p"], w["ap"])
        cp.multiply(w["p"], w["ap"], out=w["dot"])
        cp.sum(w["dot"], out=w["pap"])
        _cycle_kernel("cg_update")(
            *launch, (np.int32(n), w["x"], w["r"], w["p"], w["ap"], w["rz"], w["pap"])
        )
        z = self._fused_cycle(0, w["r"])
        cp.multiply(w["r"], z, out=w["dot"])
        cp.sum(w["dot"], out=w["newrz"])
        _cycle_kernel("cg_direction")(
            *launch, (np.int32(n), z, w["p"], w["rz"], w["newrz"])
        )
        cp.copyto(w["rz"], w["newrz"])

    def _krylov_batch(self):
        """Replay eight CG steps between existing true-residual checks."""
        cp = state._cp
        if self.krylov_graph is None:
            # Warm operations without advancing the numerical iteration.
            saved = {name: self.krylov[name].copy() for name in ("x", "r", "p", "rz")}
            self._krylov_step()
            for name, value in saved.items():
                cp.copyto(self.krylov[name], value)
            cp.cuda.get_current_stream().synchronize()
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                stream.begin_capture()
                for _ in range(8):
                    self._krylov_step()
                self.krylov_graph = stream.end_capture()
        self.krylov_graph.launch()

    def solve(self, rhs, *, tolerance=1e-10, max_iterations=1000):
        cp = state._cp
        started = perf_counter()
        b = cp.asarray(rhs) * self.scale
        x = (
            cp.zeros_like(b)
            if self.previous_solution is None
            else self.previous_solution / self.scale
        )
        r = b - self.matvec(self.levels[0]["A"], x)
        norm = float(cp.sqrt(cp.sum(b * b)).item())
        if norm == 0.0:
            self.previous_solution = cp.zeros_like(b)
            return cp.asnumpy(self.previous_solution)
        initial_residual = float(cp.sqrt(cp.sum(r * r)).item()) / norm
        if initial_residual < tolerance:
            LAST_GPU_FLOW_TIMINGS.update(
                backend="gpu-amg",
                setup_s=self.last_setup_seconds,
                solve_s=perf_counter() - started,
                iterations=0,
                scaled_true_relative_residual=initial_residual,
                levels=len(self.levels),
            )
            return cp.asnumpy(x * self.scale)
        if self.capture_krylov:
            w = self.krylov
            cp.copyto(w["x"], x)
            cp.copyto(w["r"], r)
            z = self._fused_cycle(0, w["r"])
            cp.copyto(w["p"], z)
            cp.multiply(w["r"], z, out=w["dot"])
            cp.sum(w["dot"], out=w["rz"])
            residual = float("inf")
            iteration = 0
            while iteration < max_iterations:
                count = min(8, max_iterations - iteration)
                if count == 8:
                    self._krylov_batch()
                else:
                    for _ in range(count):
                        self._krylov_step()
                iteration += count
                true_r = b - self.matvec(self.levels[0]["A"], w["x"])
                residual = float(cp.sqrt(cp.sum(true_r * true_r)).item()) / norm
                if residual < tolerance:
                    break
            x = w["x"]
        else:
            x, iteration, residual = self._solve_uncaptured(
                b, x, r, norm, tolerance, max_iterations
            )
        if not np.isfinite(residual) or residual >= tolerance:
            raise RuntimeError(
                f"GPU AMG pressure solve did not converge: residual={residual:.3e}, iterations={iteration}"
            )
        self.previous_solution = (x * self.scale).copy()
        result = cp.asnumpy(self.previous_solution)
        LAST_GPU_FLOW_TIMINGS.update(
            backend="gpu-amg",
            setup_s=self.last_setup_seconds,
            solve_s=perf_counter() - started,
            iterations=iteration,
            scaled_true_relative_residual=residual,
            levels=len(self.levels),
            captured_krylov=self.capture_krylov,
        )
        return result

    def _solve_uncaptured(self, b, x, r, norm, tolerance, max_iterations):
        cp = state._cp
        z = self._precondition(r).copy()
        p, rz = z.copy(), cp.sum(r * z)
        residual = float("inf")
        for iteration in range(1, max_iterations + 1):
            ap = self.matvec(self.levels[0]["A"], p)
            alpha = rz / cp.maximum(cp.sum(p * ap), 1e-300)
            x += alpha * p
            r -= alpha * ap
            if iteration % 8 == 0 or iteration == max_iterations:
                # True, rather than recursively accumulated, residual.
                true_r = b - self.matvec(self.levels[0]["A"], x)
                residual = float(cp.sqrt(cp.sum(true_r * true_r)).item()) / norm
                if residual < tolerance:
                    break
            z = self._precondition(r).copy()
            new_rz = cp.sum(r * z)
            p = z + (new_rz / cp.maximum(rz, 1e-300)) * p
            rz = new_rz
        return x, iteration, residual


def cached_pressure_solver(matrix):
    """Cache by matrix values as well as sparsity; viscosity changes invalidate."""
    matrix = matrix.tocsr()
    digest = blake2b(digest_size=16)
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    key = (matrix.shape, digest.digest(), int(state._cp.cuda.Device().id))
    if key in _PRESSURE_CACHE:
        LAST_GPU_FLOW_TIMINGS["hierarchy_reused"] = True
        LAST_GPU_FLOW_TIMINGS["hierarchy_values_updated"] = False
        _PRESSURE_CACHE[key].last_setup_seconds = 0.0
        return _PRESSURE_CACHE[key]
    if _PRESSURE_CACHE:
        old_key, solver = next(iter(_PRESSURE_CACHE.items()))
        if (
            old_key[0] == matrix.shape
            and old_key[2] == key[2]
            and all(
                np.array_equal(a, b)
                for a, b in zip(solver.pattern, (matrix.indptr, matrix.indices))
            )
        ):
            solver.update(matrix)
            _PRESSURE_CACHE.clear()
            _PRESSURE_CACHE[key] = solver
            LAST_GPU_FLOW_TIMINGS.update(
                hierarchy_reused=True, hierarchy_values_updated=True
            )
            return solver
    solver = GPUAMGSolver(matrix)
    # Retain one hierarchy: prevents unbounded device memory across cases.
    _PRESSURE_CACHE.clear()
    _PRESSURE_CACHE[key] = solver
    LAST_GPU_FLOW_TIMINGS["hierarchy_reused"] = False
    LAST_GPU_FLOW_TIMINGS["hierarchy_values_updated"] = False
    return solver


def _assembled_operator(up, down, conductance, n, free):
    """Retain the sparse topology and reduced matrix like tree geometry caches.

    Changed conductances refresh every entry, including parallel-edge sums.
    Direction, node count, or boundary changes invalidate the affected cache.
    """
    started = perf_counter()
    cache = _ASSEMBLY_CACHE
    reused = (
        bool(cache)
        and cache["n"] == n
        and np.array_equal(cache["up"], up)
        and np.array_equal(cache["down"], down)
    )
    if not reused:
        cache.clear()
        lap = sp.coo_matrix(
            (
                np.concatenate((conductance, conductance, -conductance, -conductance)),
                (
                    np.concatenate((up, down, up, down)),
                    np.concatenate((up, down, down, up)),
                ),
            ),
            shape=(n, n),
        ).tocsr()
        cache.update(n=n, up=up.copy(), down=down.copy(), g=conductance.copy(), lap=lap)
    elif not np.array_equal(cache["g"], conductance):
        lap = cache["lap"]
        if "positions" not in cache:
            rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(lap.indptr))
            keys = rows * n + lap.indices
            coo_keys = np.concatenate(
                (up * n + up, down * n + down, up * n + down, down * n + up)
            )
            cache["positions"] = np.searchsorted(keys, coo_keys)
        lap.data[:] = np.bincount(
            cache["positions"],
            weights=np.concatenate(
                (conductance, conductance, -conductance, -conductance)
            ),
            minlength=lap.nnz,
        )
        cache["g"] = conductance.copy()
        cache.pop("reduced", None)
    if "reduced" not in cache or not np.array_equal(cache.get("free"), free):
        cache["reduced"] = cache["lap"][free][:, free].tocsr()
        cache["free"] = free.copy()
    LAST_GPU_FLOW_TIMINGS.update(
        assembly_s=perf_counter() - started, assembly_topology_reused=reused
    )
    return cache["lap"], cache["reduced"]


def solve_kirchhoff_gpu(
    up,
    down,
    resistance,
    inlet_nodes,
    inlet_flow,
    outlet_nodes,
    *,
    num_nodes=None,
    boundary_condition=None,
):
    from .topology import _normalize_kirchhoff_bc_mode

    up, down = np.asarray(up, dtype=np.int64), np.asarray(down, dtype=np.int64)
    n = int(num_nodes or max(up.max(), down.max()) + 1)
    g = 1.0 / np.maximum(np.asarray(resistance, dtype=float), 1e-30)
    rhs = np.zeros(n)
    rhs[list(inlet_nodes)] = inlet_flow / len(inlet_nodes)
    mode = _normalize_kirchhoff_bc_mode(boundary_condition)
    boundary = np.zeros(n, bool)
    if mode == "legacy_equal_terminal_flow":
        rhs[list(outlet_nodes)] -= inlet_flow / len(outlet_nodes)
        anchor = n - 1
        if anchor in inlet_nodes or anchor in outlet_nodes:
            anchor = max(n - 2, 0)
        boundary[anchor] = True
    else:
        boundary[list(outlet_nodes)] = True
    free = ~boundary
    _, reduced = _assembled_operator(up, down, g, n, free)
    solver = cached_pressure_solver(reduced)
    pressure = np.zeros(n)
    pressure[free] = solver.solve(rhs[free])
    cp = state._cp
    pressure_gpu = cp.asarray(pressure)
    flow_gpu = cp.asarray(g) * (
        pressure_gpu[cp.asarray(up)] - pressure_gpu[cp.asarray(down)]
    )
    flows = cp.asnumpy(flow_gpu)
    balance = np.bincount(up, weights=flows, minlength=n) - np.bincount(
        down, weights=flows, minlength=n
    )
    rel = np.linalg.norm((balance - rhs)[free]) / max(np.linalg.norm(rhs[free]), 1e-30)
    LAST_GPU_FLOW_TIMINGS["flow_balance_relative_residual"] = float(rel)
    if not np.isfinite(rel) or rel > 1e-8:
        raise RuntimeError(f"GPU flow balance failed: residual={rel:.3e}")
    return pressure, flows, up, down, np.empty((0, 3))


def solve_pressure_dirichlet_gpu(
    up, down, resistance, inlets, pin, outlets, pout, *, num_nodes=None
):
    up, down = np.asarray(up, dtype=np.int64), np.asarray(down, dtype=np.int64)
    n = int(num_nodes or max(up.max(), down.max()) + 1)
    g = 1.0 / np.asarray(resistance, dtype=float)
    boundary = np.asarray(list(inlets) + list(outlets), dtype=np.int64)
    values = np.concatenate((np.full(len(inlets), pin), np.full(len(outlets), pout)))
    free = np.ones(n, bool)
    free[boundary] = False
    lap, reduced = _assembled_operator(up, down, g, n, free)
    pressure = np.zeros(n)
    pressure[boundary] = values
    if np.any(free):
        rhs = -(lap[free][:, boundary] @ values)
        pressure[free] = cached_pressure_solver(reduced).solve(rhs)
        residual = lap[free] @ pressure
        rel = np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30)
        if not np.isfinite(rel) or rel > 1e-8:
            raise RuntimeError(f"GPU pressure boundary residual failed: {rel:.3e}")
    cp = state._cp
    p = cp.asarray(pressure)
    q = cp.asarray(g) * (p[cp.asarray(up)] - p[cp.asarray(down)])
    return pressure, cp.asnumpy(q)
