# Known issues

## One-million-point tissue solve latency

Status: resolved for interactive and serial-worker use; fresh-process startup is
still intentionally reported separately.

The original 2026-09-11 profile used a 201-segment tree, 1,000,000 tissue
points, and an NVIDIA GeForce RTX 3080 Laptop GPU. A fresh CLI process took
10.01-10.14 seconds. The CUDA Green's-function kernel itself took only
0.087-0.091 seconds; KD-tree querying, candidate refinement, transfers, domain
construction, imports, and shutdown dominated the workflow.

The optimized implementation now has three complementary execution modes:

- A persistent `cascade worker` process, also used by Studio, keeps imports,
  compiled kernels, one geometry, and one compatible spatial context warm.
- `cascade batch --settings case1.json case2.json ...` exposes the same bounded,
  warm serial path to CLI users.
- Exact repeat/export-only requests reuse the last bounded simulation result.
- One-shot `cascade run` remains isolated and performs hard cleanup after all
  requested output files have been written.

### Current measurements

All timings below were measured on the same RTX 3080 Laptop GPU. Interactive
large-tree runs used a frozen 100,000-terminal tree (200,001 segments),
1,000,000 fixed tissue points, GL1 Cext, and the hybrid FFT solver.

| Workload | Previous | Current | Improvement |
| --- | ---: | ---: | ---: |
| 100,000-terminal application total | 7.180 s | 0.414 s changed-parameter worker run | 17.3x |
| 100,000-terminal simulation stages | 4.646 s | 0.329 s | 14.1x |
| 100,000-terminal exact repeat and export | 7.180 s | 0.0021 s | about 3,465x |
| Large-tree tissue stage | 0.2769 s | about 0.106 s | 2.6x |
| Dense CPU tissue, 201 segments / 50,000 points | 0.8437 s | 0.03977 s | 21.2x |
| Dense CPU tissue, 201 segments / 1,000,000 points | 16.453 s | 0.7021 s | 23.4x |
| Full CPU path, 201 segments / 1,000,000 points | 33.714 s external reference | 0.8448 s | 39.9x |
| Detailed VTK export, 200,001 vessels / 1,000,000 points | over 147 s (did not finish) | 7.14 s | at least 20.6x |
| HEART-L application total | 176.397 s | 179.019 s | 1.5% slower |
| HEART-L simulation stages | 124.968 s | 112.934 s | 1.11x (9.6% less) |

The simulation-stage figure is `t_allocation + t_flow + t_concentration +
t_tissue`. In ordinary CASCADE output those map to `t_assembly_s +
t_kirchhoff_s + t_concentration_s + t_tissue_s`. HEART-L's shared solver emits
allocation, flow, and concentration as one `t_cext_total_s`, so its equivalent
stage sum is `t_cext_total_s + t_tissue_s`. End-to-end application time remains
a separate measurement and includes loading, export, cleanup, and process
overhead.

The million-point dense CPU comparison had zero mask differences, relative L2
error `3.51e-13`, and maximum absolute error `1.25e-12`. The final large-tree
GPU comparison preserved all 103 scientific summary values exactly; only the
intentionally shortened fixed-flow iteration-count diagnostic changed. The
full HEART-L tissue/domain comparison also passed the release tolerance policy.

### What changed

- Exact dense tissue evaluation is fused into one CPU Numba loop or one CUDA
  kernel. It computes projection, lumen rejection, the physical window, and
  Green's quadrature without allocating point-by-segment candidate matrices.
- The large-tree GPU cell-list path now launches the entire point cloud when
  memory permits. Its storage is O(points), so the old 123 small launches can
  become one launch; an out-of-memory exception halves the chunk and retries.
- Dense candidate selection bypasses KD-tree construction and sorting when all
  segments would be retained anyway. Candidate/KD-tree CPU fallbacks remain
  available for sparse and non-GPU cases.
- Tree assembly, domain/sample geometry, tissue spatial context, and compatible
  hybrid Cext geometry are single-entry caches. Flow magnitude is refreshed in
  place, so changing inlet flow does not rebuild immutable geometry.
- Equal-terminal fixed-flow hematocrit stops after its first fully relaxed
  update because viscosity cannot feed back into prescribed flows. Other
  boundary conditions and under-relaxed iterations retain the general loop.
- Successful GPU preflight is cached within a process. Studio sends queued
  cases to a persistent newline-delimited worker instead of launching a new
  Python/CUDA process for every click.
- Studio starts that worker in the background, retires it after a configurable
  idle interval, batches log writes, updates only the changed queue row, and
  coalesces queue checkpoints and live combined-sweep rebuilds.
- VTK vessel construction is vectorized directly from solver arrays. Existing
  project defaults remain compatible; disabling detailed VTK/network outputs
  enables the lightest summary-only interactive replay mode.

### Memory lifecycle

The worker retains at most one compatible geometry, one tissue cache, one Cext
context, and one compact summary-only result. A geometry or scientific-settings
change drops the prior result before solving, and a geometry change collects
the old object graph before constructing its replacement. Results that contain
point rows, segment rows, or ParaView payloads are exported but not retained.
The compact replay object does not keep solver detail arrays or a second tree
reference. Host-RAM pressure evicts the result and spatial cache first, then
all reusable geometry at critical pressure. Linux hard cleanup also returns
free allocator arenas to the operating system.

Serial sweeps retain a tissue cache only across fluid variants that share the
same geometry and point set. Each case drops its result after its summary is
spooled; each group releases its spatial cache in a `finally` block. Summary
and timing records spill from RAM to a temporary file for very large sweeps.
Adaptive
cleanup keeps harmless allocator blocks only while device headroom is adequate,
whereas one-shot CLI completion and worker shutdown return unused accelerator
pools to the driver. Output export always finishes before cleanup begins.

A 16-case alternating-flow soak (every warm case recomputed) measured a median
of 0.426 seconds. Worker RSS changed from 904.4 MiB after the cold case
to 899.1 MiB after case 16, showing no case-count-dependent growth across the
periodic full-GC boundary.

### Remaining limitation

A brand-new process still pays Python/scientific imports, domain meshing,
network loading, CUDA initialization, and first-use compilation. Recent cold
100,000-terminal worker cases took about 4.7-4.9 seconds. This startup cost is
outside the sub-second steady-state interaction target; use `cascade worker`,
Studio, or a serial sweep when running more than one case.
