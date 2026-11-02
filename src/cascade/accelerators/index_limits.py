"""Host-side index-range checks for CUDA kernel launches.

CUDA kernels in this package receive sizes as ``int`` parameters and compute
some flattened offsets in 32-bit arithmetic.  Casting each operand with
``np.int32`` catches oversized scalars, but not products formed on the device
(for example ``n_bins * grid_cells``).  Launch sites call
:func:`check_kernel_indices` with every size and product the kernel forms so
that an overflowing launch raises a clear error instead of silently wrapping.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import prod

INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1
# CUDA limits gridDim.x to 2**31 - 1 blocks on every supported architecture.
MAX_GRID_X = 2**31 - 1

IndexTerm = int | Iterable[int]


class KernelIndexOverflowError(OverflowError):
    """A launch would exceed the index type a CUDA kernel uses."""


def _value(term: IndexTerm) -> int:
    if isinstance(term, Iterable):
        return int(prod(int(factor) for factor in term))
    return int(term)


def _check(kernel: str, terms: Mapping[str, IndexTerm], limit: int, kind: str):
    for label, term in terms.items():
        value = _value(term)
        if value < 0 or value > limit:
            raise KernelIndexOverflowError(
                f"{kernel}: {label} = {value:,} exceeds the {kind} index limit "
                f"({limit:,}) of this CUDA kernel. Reduce the problem size "
                "(grid size, lambda bins, quadrature order, segment or point "
                "count) or split the launch."
            )


def launch_blocks(total_threads: int, threads_per_block: int) -> int:
    """Return the 1-D block count covering ``total_threads`` threads."""
    return (int(total_threads) + int(threads_per_block) - 1) // int(threads_per_block)


def check_kernel_indices(
    kernel: str,
    *,
    int32: Mapping[str, IndexTerm] | None = None,
    int64: Mapping[str, IndexTerm] | None = None,
    blocks: int | None = None,
    threads: int | None = None,
) -> None:
    """Raise :class:`KernelIndexOverflowError` when a launch would overflow.

    ``int32`` and ``int64`` map a descriptive label to either a size or a
    sequence of factors whose exact (Python integer) product the kernel forms
    in that index type.  When ``blocks`` and ``threads`` are supplied, the
    launch is also checked against the grid-dimension limit and the 32-bit
    ``blockIdx.x * blockDim.x + threadIdx.x`` thread index the kernels use
    unless they state otherwise.
    """
    if blocks is not None:
        if int(blocks) > MAX_GRID_X:
            raise KernelIndexOverflowError(
                f"{kernel}: {int(blocks):,} blocks exceed the CUDA grid limit "
                f"({MAX_GRID_X:,})."
            )
        if threads is not None:
            _check(kernel, {"launched threads": (blocks, threads)}, INT32_MAX, "32-bit")
    _check(kernel, int32 or {}, INT32_MAX, "32-bit")
    _check(kernel, int64 or {}, INT64_MAX, "64-bit")


__all__ = [
    "INT32_MAX",
    "INT64_MAX",
    "MAX_GRID_X",
    "KernelIndexOverflowError",
    "check_kernel_indices",
    "launch_blocks",
]
