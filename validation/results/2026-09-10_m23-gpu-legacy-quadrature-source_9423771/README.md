# Source-only legacy-quadrature GPU preflight

Status: non-certifying source preflight. Tracker IDs: VAL-07 and PERF-04.

One target-1 blood case selected `legacy_cext`, which intentionally reuses the
Cext GL1 source nodes during tissue evaluation. All 23 summary comparisons
passed, confirming the diagnosed quadrature cause.

This command invoked the editable development environment, as proven by
`cascade_module=/home/carl/svv_sweeps/GFM/src/cascade/__init__.py` in the
CASCADE record. The matrix header named the prior `e1cd36b` wheel because the
old driver validated only that the supplied wheel file had the requested hash;
it did not prove that the interpreter had installed that wheel. Consequently,
the reported timing is not exact-wheel evidence. The harness now records
distribution installation metadata and rejects any mismatch between the
declared wheel and the invoked CASCADE installation.
