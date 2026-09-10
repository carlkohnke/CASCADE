# Installed-wheel GPU reference profile at one and five million terminals

Tracker IDs: PERF-01, PERF-02, PERF-04, PERF-07.

Five alternating isolated pairs were run at both 1,000,000 and 5,000,000
terminals (2,000,001 and 10,000,001 vessel segments). Every scientific summary
passed. Median comparable-compute ratios were approximately `0.849` and `0.947`:
at the largest case CASCADE required `26.16 s` versus `27.65 s` for the current
`svva2` oracle. Median monitored wall ratio at the largest case was `1.032`, with
the remaining overhead attributed to file loading/build/export outside the
historical plot boundary. Peak RSS stayed near 11-12 GiB, below the 45 GiB gate.

The historical June CSV reported 18.09 s for water and 23.19 s for blood at
10,000,001 vessels. Both current implementations were uniformly slower on this
campaign's host state, while CASCADE remained faster than the executable current
oracle under alternating order. `benchmark-summary.json` contains the exact
measurements and provenance.
