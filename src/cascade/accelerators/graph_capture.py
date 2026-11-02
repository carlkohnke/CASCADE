"""Allocation-free CUDA graph capture.

A captured graph replays raw device addresses. Memory-pool blocks requested
while capturing (reduction scratch, temporaries, casts) return to the pool as
soon as their Python owners are released, so a later replay would write into
whichever arrays reuse those blocks. Capture therefore rejects every pool
allocation. Owners keep the arrays a graph touches next to the graph and run
the recorded operations directly when capture is refused.
"""

from __future__ import annotations


class GraphCaptureAllocation(RuntimeError):
    """A captured operation requested device memory it would not own."""


def _reject_allocation(size):
    raise GraphCaptureAllocation(
        f"{size}-byte device allocation during CUDA graph capture"
    )


def capture_graph(record):
    """Return a CUDA graph of ``record()``, or None if recording allocates.

    ``record`` must only launch work on preallocated, owner-held arrays.
    Kernels should be compiled beforehand; nothing executes during capture.
    """
    from cascade.configuration import solver_state as state

    cp = state._cp
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    # The allocator override is thread-local, matching the capturing stream.
    with cp.cuda.using_allocator(_reject_allocation), stream:
        stream.begin_capture()
        try:
            record()
        except BaseException as error:
            try:
                stream.end_capture()
            except cp.cuda.runtime.CUDARuntimeError:
                pass
            if isinstance(error, GraphCaptureAllocation):
                return None
            raise
        return stream.end_capture()


__all__ = ["GraphCaptureAllocation", "capture_graph"]
