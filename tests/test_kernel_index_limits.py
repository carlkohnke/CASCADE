"""Regression checks for 32-bit index overflow in the CUDA kernels."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from cascade.accelerators.cuda import load_cuda_source
from cascade.accelerators.index_limits import (
    INT32_MAX,
    KernelIndexOverflowError,
    check_kernel_indices,
    launch_blocks,
)


def test_guard_accepts_sizes_at_the_limit():
    check_kernel_indices(
        "k",
        int32={"cells": 1024**3, "edge": INT32_MAX, "product": (46340, 46340)},
        int64={"bins x cells": (64, 1024**3), "stencil": (10**8, 8, 27)},
        blocks=launch_blocks(INT32_MAX, 1),
        threads=1,
    )


@pytest.mark.parametrize(
    "int32",
    [
        {"bins x cells": (5, 768**3)},
        {"stencil": (16_000_000, 5, 27)},
        {"nodes x 3": (800_000_000, 3)},
        {"scalar": INT32_MAX + 1},
        {"negative": -1},
    ],
)
def test_guard_rejects_overflowing_32bit_products(int32):
    with pytest.raises(KernelIndexOverflowError, match="32-bit index limit"):
        check_kernel_indices("demo_kernel", int32=int32)


def test_guard_rejects_thread_and_grid_overflow():
    threads = 128
    blocks = launch_blocks(600_000_000 * 4, threads)
    with pytest.raises(KernelIndexOverflowError, match="launched threads"):
        check_kernel_indices("warp", blocks=blocks, threads=threads)
    # Kernels with 64-bit thread indices only need the grid-dimension check.
    check_kernel_indices("warp", blocks=blocks)
    with pytest.raises(KernelIndexOverflowError, match="grid limit"):
        check_kernel_indices("huge", blocks=2**31)
    with pytest.raises(KernelIndexOverflowError, match="64-bit"):
        check_kernel_indices("k", int64={"x": (2**40, 2**30)})
    assert issubclass(KernelIndexOverflowError, OverflowError)


def test_deposit_guard_rejects_grids_beyond_int32_cells():
    from cascade.concentration.external_field.hybrid_deposit import (
        _check_deposit_indices,
    )

    static = {"grid_cells": 1024**3, "stencil_n": 27}
    blocks = launch_blocks(25_000_000 * 5, 256)
    # 5 bins x 1024^3 cells and 1.25e8 nodes x 27 stencil slots are 64-bit now.
    _check_deposit_indices("deposit", static, 25_000_000 * 5, 5, blocks, 256)
    with pytest.raises(KernelIndexOverflowError, match="grid cells"):
        _check_deposit_indices(
            "deposit", {"grid_cells": 1291**3, "stencil_n": 27}, 10, 5, 1, 256
        )


_OVERFLOWING_PATTERNS = (
    r"\bint\s+\w*base\s*=\s*\(?\s*\w+\s*\*",  # int base = bin * cells ...
    r"\bint\s+green_idx\b",
    r"\bint\s+idx\d{3}\b",
    r"\bint\s+stencil_base\b",
    r"\[\s*3\s*\*\s*(node_idx|source_flat|source_node)\b",
    r"\[\s*\(target_seg \* gl_order \+ target_node\) \* 3",
)


@pytest.mark.parametrize(
    "name",
    [
        "cext_direct_kernel.cu",
        "cext_fast_local_only_self_kernel.cu",
        "cext_fft_discrete_self_correlated_kernel.cu",
        "cext_fft_discrete_self_fused_o2_kernel.cu",
        "cext_fft_discrete_self_kernel.cu",
        "cext_fft_discrete_self_runtime_moment_kernel.cu",
        "cext_fft_discrete_self_runtime_stencil_kernel.cu",
        "cext_fft_moment_deposit_batch_kernel.cu",
        "cext_fft_moment_deposit_computed_runtime_stencil_kernel.cu",
        "cext_fft_moment_deposit_kernel.cu",
        "cext_fft_moment_deposit_runtime_stencil_kernel.cu",
        "cext_hybrid_bg_kernel.cu",
        "cext_hybrid_deposit_kernel.cu",
        "cext_hybrid_deposit_runtime_stencil_kernel.cu",
        "cext_kernel.cu",
        "cext_local_corr_kernel.cu",
        "cext_tissue_cell_greens_kernel.cu",
        "cext_tissue_greens_kernel.cu",
    ],
)
def test_kernel_sources_form_large_offsets_in_64_bits(name):
    source = load_cuda_source(name)
    for pattern in _OVERFLOWING_PATTERNS:
        assert re.search(pattern, source) is None, (name, pattern)


# ---------------------------------------------------------------------------
# Large-scale CUDA regressions (opt-in; each needs <= ~20 GB of device memory)
# ---------------------------------------------------------------------------

_LARGE = pytest.mark.skipif(
    os.environ.get("CASCADE_LARGE_GPU_TESTS") != "1",
    reason="set CASCADE_LARGE_GPU_TESTS=1 to run multi-GB CUDA regressions",
)


def _cupy(min_free_gb: float):
    from cascade.concentration.vessel.network_gpu import resolve_network_accel
    from cascade.configuration import solver_state as state

    if resolve_network_accel("auto") != "gpu" or state._cp is None:
        pytest.skip("CUDA/CuPy unavailable")
    cp = state._cp
    cp.get_default_memory_pool().free_all_blocks()
    free, _ = cp.cuda.runtime.memGetInfo()
    if free < min_free_gb * 1024**3:
        pytest.skip(f"needs {min_free_gb:.0f} GB free device memory")
    return cp


@pytest.fixture
def release_pool():
    yield
    from cascade.configuration import solver_state as state

    if state._cp is not None:
        state._cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.large_gpu
@_LARGE
def test_hybrid_deposit_beyond_2_31_bins_x_cells(release_pool):
    """3 bins x 1024^3 cells: the last bin starts at offset 2^31 (~13 GB)."""
    cp = _cupy(16)
    from cascade.concentration.external_field.hybrid_geometry import (
        _get_cext_hybrid_deposit_kernel,
    )

    grid_n, n_bins, n_nodes = 1024, 3, 1_000_000
    grid_cells = grid_n**3
    assert n_bins * grid_cells > INT32_MAX
    rng = np.random.default_rng(7)
    flat = rng.choice(grid_cells, size=n_nodes, replace=False).astype(np.int32)
    q = (np.arange(n_nodes) % 997 + 1).astype(np.float32)
    # Every node falls in the last lambda bin (edges 0, 1, 2, inf).
    edges = np.array([0.0, 1.0, 2.0, np.inf], dtype=np.float32)
    lam = np.full(n_nodes, 5.0, dtype=np.float32)
    out = cp.zeros((n_bins, grid_cells), dtype=cp.float32)
    threads = 256
    blocks = launch_blocks(n_nodes, threads)
    _get_cext_hybrid_deposit_kernel()(
        (blocks,),
        (threads,),
        (
            cp.asarray(q),
            cp.asarray(lam),
            cp.arange(n_nodes, dtype=cp.int32),
            cp.ones(n_nodes, dtype=cp.uint8),
            cp.asarray(flat),
            cp.ones(n_nodes, dtype=cp.float32),
            cp.asarray(edges),
            np.int32(n_bins),
            np.int32(grid_cells),
            np.int32(1),
            np.int32(n_nodes),
            out.ravel(),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    sample = rng.choice(n_nodes, size=4096, replace=False)
    got = cp.asnumpy(out[n_bins - 1][cp.asarray(flat[sample])])
    np.testing.assert_array_equal(got, q[sample])
    assert float(cp.sum(out[n_bins - 1], dtype=cp.float64)) == float(
        np.sum(q, dtype=np.float64)
    )
    assert int(cp.count_nonzero(out[: n_bins - 1])) == 0


@pytest.mark.large_gpu
@_LARGE
def test_hybrid_bg_samples_bin_beyond_2_31(release_pool):
    """Trilinear background sampling from a bin whose base offset is 2^31."""
    cp = _cupy(16)
    from cascade.concentration.external_field.hybrid_deposit import (
        _get_cext_hybrid_bg_kernel,
    )

    grid_n, n_bins, n_targets = 1024, 3, 8192
    phi = cp.zeros((n_bins, grid_n, grid_n, grid_n), dtype=cp.float32)
    # Last bin varies linearly along x, so trilinear sampling is exact.
    phi[n_bins - 1] = cp.arange(grid_n, dtype=cp.float32)[:, None, None]
    rng = np.random.default_rng(11)
    points = rng.uniform(2.0, grid_n - 2.0, size=(n_targets, 3)).astype(np.float32)
    out = cp.zeros(n_targets, dtype=cp.float32)
    threads = 128
    blocks = launch_blocks(n_targets, threads)
    _get_cext_hybrid_bg_kernel()(
        (blocks,),
        (threads,),
        (
            cp.arange(n_targets, dtype=cp.int32),
            cp.asarray(points.ravel()),
            phi.ravel(),
            cp.zeros(1, dtype=cp.float32),  # never read: near radius is zero
            cp.ones(n_bins, dtype=cp.float32),
            np.int32(n_bins),
            np.int32(grid_n),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(0.0),
            np.float32(1.0),
            np.float32(0.0),
            np.int32(1),
            np.int32(n_targets),
            np.float32(1.0),
            out,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    expected = points[:, 0].astype(np.float64) - 0.5
    np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=0, atol=2e-3)


@pytest.mark.large_gpu
@_LARGE
def test_stored_stencil_offsets_beyond_2_31(release_pool):
    """80M nodes x 27 slots: stencil offsets pass 2^31 (~19 GB of stencils)."""
    cp = _cupy(22)
    from cascade.concentration.external_field.hybrid_geometry import (
        _get_cext_hybrid_deposit_kernel,
    )

    n_nodes, stencil_n, grid_n = 80_000_000, 27, 64
    grid_cells = grid_n**3
    assert n_nodes * stencil_n > INT32_MAX
    # Broadcast rather than scatter: CuPy 13's advanced-index assignment passes
    # the indexed span (here n_nodes * stencil_n > 2^31) to its kernel as int32,
    # so `flat_idx[node, slot] = cell` silently misses every row past 2^31.
    node = cp.arange(n_nodes, dtype=cp.int64)[:, None]
    hit = node % stencil_n == cp.arange(stencil_n, dtype=cp.int64)[None, :]
    flat_idx = cp.where(hit, (node % grid_cells).astype(cp.int32), cp.int32(-1))
    weight = hit.astype(cp.float32)
    del node, hit
    # Only nodes whose stencil row starts beyond 2^31 carry a nonzero charge.
    first_tail = INT32_MAX // stencil_n + 1
    q = cp.zeros(n_nodes, dtype=cp.float32)
    q[first_tail:] = cp.float32(2.0)
    out = cp.zeros((1, grid_cells), dtype=cp.float32)
    threads = 256
    blocks = launch_blocks(n_nodes, threads)
    _get_cext_hybrid_deposit_kernel()(
        (blocks,),
        (threads,),
        (
            q,
            cp.ones(n_nodes, dtype=cp.float32),
            cp.zeros(n_nodes, dtype=cp.int32),
            cp.ones(1, dtype=cp.uint8),
            flat_idx.ravel(),
            weight.ravel(),
            cp.asarray(np.array([0.0, np.inf], dtype=np.float32)),
            np.int32(1),
            np.int32(grid_cells),
            np.int32(stencil_n),
            np.int32(n_nodes),
            out.ravel(),
        ),
    )
    cp.cuda.Stream.null.synchronize()
    expected = 2.0 * np.bincount(
        np.arange(first_tail, n_nodes, dtype=np.int64) % grid_cells,
        minlength=grid_cells,
    ).astype(np.float64)
    np.testing.assert_array_equal(cp.asnumpy(out[0]).astype(np.float64), expected)


def _python_scalar(node, assignments, depth=0):
    """Whether ``node`` evaluates to an untyped Python int or float."""
    import ast

    if depth > 4:
        return False
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"int", "float", "len", "round"}
    if isinstance(node, ast.BinOp):
        return any(
            _python_scalar(side, assignments, depth + 1)
            for side in (node.left, node.right)
        ) and all(
            isinstance(side, ast.Constant) or _python_scalar(side, assignments, depth + 1)
            for side in (node.left, node.right)
        )
    if isinstance(node, ast.Name):
        return any(
            _python_scalar(value, assignments, depth + 1)
            for value in assignments.get(node.id, ())
        )
    return False


def test_kernel_launch_scalars_are_numpy_typed():
    """CuPy 13 and 14 may convert bare Python scalars differently (int64/double
    against an int/float parameter), so every launch passes np.int32/np.float32-style
    scalars matching the kernel signature."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "cascade"
    launches, untyped = 0, []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assignments = {}
            for node in ast.walk(function):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignments.setdefault(target.id, []).append(node.value)
            for node in ast.walk(function):
                # kernel(grid, block, (arg, ...)) with literal grid/block tuples.
                if not (
                    isinstance(node, ast.Call)
                    and len(node.args) == 3
                    and all(isinstance(arg, ast.Tuple) for arg in node.args[:2])
                    and isinstance(node.args[2], ast.Tuple)
                ):
                    continue
                launches += 1
                for argument in node.args[2].elts:
                    if _python_scalar(argument, assignments):
                        untyped.append(
                            f"{path.relative_to(root)}:{argument.lineno}: "
                            f"{ast.unparse(argument)}"
                        )
    assert launches >= 20
    assert not untyped, "untyped kernel scalars:\n" + "\n".join(untyped)
