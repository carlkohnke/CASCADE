# Full 200-cubed HEART-S GPU comparison

Tracker IDs: VAL-06, VAL-07, PERF-05, PERF-06.

The frozen legacy exporter selected 1,619,996 bivent3 coordinates from the
8,000,000-point candidate grid; the shared `.npy` fixture has SHA-256
`ab733bd07827e33e7458f9e474a5b6f6ec12f8eff45cf41c4bd00c268cbc9fdb`.
Every subsequent run used that exact point file. Healthy and global-segment-1
full-occlusion cases used blood, float32 GPU work arrays, Cext GL1, independent
tissue GL5, grid 256, window 6, one Cext iteration, and finite-radius/Graetz
controls.

The frozen exporter's native shared-Cext path is effective GL1 and is retained
in `healthy-production-comparison.json`; as expected, it is not equivalent work.
The validation wrapper then conserved each GL1 segment source weight while
resampling it to five nodes before calling the same frozen GPU tissue evaluator.
`healthy-gl5-reference-comparison.json` passes every geometry, field, finite-mask,
and viability rule against production CASCADE.

The first occlusion GL5 comparison retained passing viability fractions but
exposed that CASCADE propagated its temporary zero-radius solve override into
6,207 exported subtree segments and tissue masking. The pre-fix failure is kept
in `occlusion-gl5-reference-pre-radius-fix.json`; D-041 records the correction
that requires a fresh-wheel rerun.

The explicit-point healthy runs completed in approximately 10.0 s monitored for
legacy versus 8.7 s for CASCADE even though CASCADE performed the requested GL5
work. The initial legacy capture/filter run took 145 s and is classified as grid
construction rather than equivalent solver time. Peak RSS remained below 4.1 GiB.
