# M3 clean cube CPU benchmark: 1 through 100 terminals

Tracker IDs: PERF-01, PERF-02, PERF-03, and PERF-07.

This campaign ran five isolated legacy/CASCADE pairs per target, strictly one
process at a time, with alternating execution order. Both sides loaded the same
hash-frozen tree and the same one-million-point tissue sample. CASCADE ran from
wheel SHA-256
`201c6459be5628cc5e79b3b2d0ae52cbdbba5c613bbe7a5a77a914476e6529c1`;
the external oracle SHA-256 was
`f2c4c89c8826b11246a239f7ae947872b955e5e47b44290033c13d1cef68bfda`.

Every one of the 15 scientific summary comparisons passed. The first pair at
target 1 includes CASCADE's one-time Numba compilation cost and is labeled
`cold_compile`. Every later run is a fresh process using the persistent compile
cache and is labeled `warmed_compile_cache`.

| Target | Legacy median (s) | CASCADE median (s) | Median CASCADE/legacy ratio |
| ---: | ---: | ---: | ---: |
| 1 | 6.002 | 6.278 | 1.035 |
| 10 | 7.933 | 8.107 | 1.018 |
| 100 | 27.273 | 26.755 | 0.974 |

Maximum observed host RSS across either implementation was 2.78 GiB. The full
pair records, component timings, commands, environment records, resource
samples, and comparison results are referenced by `benchmark-summary.json`.
Raw solver logs and emitted summaries are in the matching ignored runs folder.
This is the first scale tranche, not the final PERF-07 claim.
