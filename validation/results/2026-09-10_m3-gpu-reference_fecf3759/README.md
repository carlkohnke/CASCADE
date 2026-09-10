# Frozen historical GPU speed reference

Tracker IDs: PERF-01, PERF-02, PERF-04, and PERF-07.

The owner identified the June 2026 GPU characterization represented by the
external `svva2/SCRIPTS/Cube_Memory_Improvement.csv` file. Its SHA-256 is
`fecf37597a34ac89533122b7563a36ddefb222a60bee5fd09e228517480b2016`.
The 74 source rows are not copied into Git; `legacy-gpu-reference-summary.json`
preserves their compact per-vessel-count/per-fluid timings and the exact source
hash.

The plotted algorithm time is not fresh-process wall time. It is exactly:

`t_assembly_s_mean + t_kirchhoff_s_mean + t_concentration_s_mean + t_tissue_s_mean`

Tree loading, interpreter/package startup, domain creation, exports, and
external monitoring are excluded. The profile used one million tissue points,
float32/int32 accelerator storage, `topdown_ext_hybrid_bg`, GPU Cext and tissue
paths, Cext GL1, tissue GL5, one Cext iteration, a 128-cubed FFT grid, and
window factor 4.

The benchmark snapshot at
`C:/Users/carl/svVascularize/svVascularize/tools/_TissueSim_cube_local_benchmark.py`
has SHA-256
`1c8225416f13b1d57b768fde6df6dc619b03f8a8c73aa44a2d268503b69f3a50`.
It differs from the current frozen SCRIPTS oracle in only 35 added and 39
removed lines, primarily a later same-node Cext FFT self-subtraction change and
workspace-reuse adjustment. New certification executes the current oracle and
wheel; this historical data is the speed target and a reproducibility anchor.

Representative source results are 0.880-1.031 s at 400,001 vessels,
12.36-18.37 s at 8,000,001 vessels, and 18.09-23.19 s at 10,000,001 vessels
across water/cell media and blood.
