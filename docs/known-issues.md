# Known issues

## One-million-point tissue solve latency

Production validation on 2026-09-11 found that a 201-segment network with
1,000,000 tissue sample points takes about 10 seconds wall-clock on an NVIDIA
GeForce RTX 3080 Laptop GPU when launched as a fresh CLI process. Three current
post-refactor runs measured 10.01 s, 10.10 s, and 10.14 s. The corresponding
pre-refactor baseline runs measured 8.97-10.34 s, so the refactor has not shown
a clear solver regression, but the latency is too high for an interactive
workflow.

The CUDA Green's-function kernel itself is fast (0.087-0.091 s). Most of the
tissue-stage latency is currently outside that kernel:

- KD-tree candidate query: approximately 1.4-1.5 s
- candidate refinement: approximately 2.4-2.5 s
- GPU transfer: approximately 0.36 s
- complete tissue stage: approximately 4.4-4.6 s
- complete in-process simulation: approximately 7.0 s
- fresh-process import, initialization, and shutdown overhead: approximately
  3.0 s

Follow-up work should measure and address these components separately. A
persistent process, explicit startup warm-up, and precompiled/cached kernels may
reduce launch and JIT overhead. Reducing query and refinement time will require
algorithmic or data-lifecycle work, such as retaining spatial indices across
runs, reusing candidate maps when geometry and points are unchanged, batching
more efficiently, or evaluating an alternative GPU-resident spatial search.
Any optimization must preserve the existing CPU/GPU accuracy tolerances and be
benchmarked against the same fixed one-million-point fixture.
