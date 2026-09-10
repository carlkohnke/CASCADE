# GPU fresh-process baseline before persistent Numba caching

Tracker IDs: PERF-02, PERF-04, PERF-06, PERF-07.

Five alternating target-1 blood pairs ran the exact `c0db166` wheel and the
hash-frozen `svva2` oracle with the historical 128/window-4 GPU profile and
explicit legacy-GL1 tissue diagnostic. All 23 scientific summaries passed.
The warmed compile-cache median comparable compute ratio was approximately
`1.073` CASCADE/oracle. This campaign established that repeated fresh CASCADE
processes were recompiling CPU helper kernels and motivated commit `59f27e5`.

`benchmark-summary.json` is the compact machine-readable result. Raw logs and
outputs remain in the matching ignored `validation/runs/` directory.
