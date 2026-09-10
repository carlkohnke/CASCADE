# HEART-S shared-point harness bring-up

Tracker IDs: VAL-06 and VAL-07.

This append-only working campaign records two failed wrapper attempts followed
by a successful isolated legacy capture. The first attempt could not infer a
Python loader for the oracle's `.bak` suffix; the second removed the external
SCRIPTS directory from `sys.path` before the oracle lazily imported its heart
runtime. Both failures are preserved in their monitor JSON and raw logs.

The third attempt verified the tissue-capable oracle SHA-256
`3fed343b3c2bbce84b021f55f8b174fe618a21267a6edc074c91680ee6848165`,
ran the frozen 16-cubed HEART-S profile, and captured 684 inside-domain
coordinates. The ignored point fixture is:

`validation/runs/2026-09-10_m2-heart-shared-points_working/heart_s_grid16_inside.npy`

Its SHA-256 is
`9298fab8e103e595c4f674298695d445fed222ca410f5cd5d67a1757b04f9001`.
The successful solver output passed its Cext flux check and peaked below 1 GiB
host RSS. A fresh committed wheel is required for the actual pointwise parity
run; this bring-up campaign is not certification.
