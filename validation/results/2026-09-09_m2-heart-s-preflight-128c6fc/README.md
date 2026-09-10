# HEART-S shared-Cext preflight

This bounded 16-cubed tissue-grid preflight retained the production Cext
workload: the frozen 19,999-segment HEART-S forest, blood, float32 accelerator
policy, 256-cubed FFT background, Cext quadrature 1, tissue quadrature 5, one
coupling iteration, and window factor 6. Runs were sequential and guarded by a
45 GiB host-memory ceiling.

The primary frozen exporter
`export_paraview_heart_forest_grid_cext_gfm.py` completed forest flow and shared
Cext, then failed in tissue evaluation with `KeyError: 'k_if_gl'`. Its
June-17 pre-state-slim backup completed successfully against the same current
SCRIPTS cube runtime because it retains the full shared Cext state. That backup
is now frozen as the tissue-capable comparison oracle with SHA-256
`3fed343b3c2bbce84b021f55f8b174fe618a21267a6edc074c91680ee6848165`;
the failed current-exporter result remains evidence of the historical
compatibility regression.

The exact `128c6fc` CASCADE wheel also completed, but its metadata showed that
all 4,096 bounding-box grid points were retained while the legacy bivent3
domain retained only 684. Inspection found that the packaged runtime's
mesh-based domain API was not wired into `cascade export-heart`; the path was
silently replaced by a cube. Its Cext concatenation also omitted wall/bulk and
interfacial-coefficient arrays, causing the packaged flux diagnostic to be
skipped. Both product defects now have source fixes and regression tests; this
successful but scientifically invalid CASCADE output is retained and must not
be treated as parity evidence.

Observed peaks were about 1.25 GiB legacy-current, 0.97 GiB legacy backup, and
1.02 GiB CASCADE. WSL `nvidia-smi` did not provide reliable per-process GPU
memory. None approached the 45 GiB host abort threshold.

This campaign is a failed/diagnostic preflight, not VAL-07 certification. A
fresh committed wheel must reproduce the backup oracle's domain mask, vessel
fields, Cext fields, tissue coordinates, and tissue oxygen before advancing to
the full 200-cubed cases.
