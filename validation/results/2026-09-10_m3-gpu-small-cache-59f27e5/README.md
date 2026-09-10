# GPU fresh-process result after persistent Numba caching

Tracker IDs: PERF-02, PERF-04, PERF-06, PERF-07.

The same five alternating target-1 blood pairs were repeated from the exact
`59f27e5` wheel after enabling on-disk Numba caches for the retained flow and
ordering helpers. All 23 scientific summaries passed. Warmed comparable compute
fell to a median of about `0.807 s`, versus `1.507 s` for the current external
oracle, a CASCADE/oracle ratio of `0.534`. Fresh-process wall time still includes
Python, CUDA, domain, and file setup and is not the historical plotted metric.

`benchmark-summary.json` records commands, installation provenance, timings,
memory, hashes, and comparisons. Raw products remain ignored.
