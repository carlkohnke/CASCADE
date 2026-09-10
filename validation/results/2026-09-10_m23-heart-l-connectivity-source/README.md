# HEART-L compact connectivity and performance preflight

Tracker IDs: VAL-07, PERF-05, PERF-06.

This source-tree preflight uses the frozen 24,999,999-segment HEART-L simulation
cache with float32 physical arrays and exact int32 topology/node arrays. It
retains the initial failed `987c97e` attempt, which exposed float32 rounding of
embedded connectivity IDs above `2**24`.

After separating integer identifiers from physical fields, CASCADE completed
the compact GPU flow/Cext solve in 75.60 seconds application time and 80.97
seconds monitored wall time, peaking at 27.60 GiB RSS. The isolated `svva2`
oracle completed equivalent compact work in 80.93 seconds application time and
83.50 seconds monitored wall time, peaking at 29.95 GiB RSS. CASCADE was 6.6%
faster by application time and used 7.9% less peak resident memory. Tree outlet
pressures differed by less than `1.5e-4 dyn/cm^2` (approximately `1.3e-9`
relative).

The monitor JSON files contain exact commands and environments. Raw logs and
outputs are under ignored
`validation/runs/2026-09-10_m23-heart-l-connectivity-source/`.
