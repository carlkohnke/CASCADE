# Production-quadrature GPU preflight against the legacy summary

Status: scientifically informative failure. Tracker IDs: VAL-07 and PERF-04.

One target-1 blood case used the recovered historical accelerated profile
(128-cubed FFT grid, window 4, Cext GL1) while CASCADE correctly performed
independent tissue GL5 integration. Geometry, pressure, flow, vessel oxygen,
and every viability fraction passed. `C_tiss_over_Cmax_mean` failed because the
legacy oracle reused its single Cext source node for tissue evaluation, whereas
CASCADE resampled each segment to five tissue nodes.

This result is retained as the evidence that motivated D-034. The matrix stopped
at the required comparison failure, so it has no pair or aggregate benchmark
summary. It is not a performance result and it does not invalidate production
GL5; production accuracy must be evaluated as GL5 work, while legacy effective
GL1 is a separately labeled implementation-lineage diagnostic.
