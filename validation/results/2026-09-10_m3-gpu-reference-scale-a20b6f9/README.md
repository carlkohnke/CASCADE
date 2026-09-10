# Installed-wheel GPU reference profile through 100,000 terminals

Tracker IDs: PERF-01, PERF-02, PERF-04, PERF-07.

Five strictly sequential alternating pairs at 10, 100, 1,000, 10,000, and
100,000 terminals used the exact `a20b6f9` wheel and current hash-frozen oracle.
The profile matches the historical plot's computation: one million common
tissue points, float32 GPU execution, Cext GL1, grid 128, window 4, one Cext
iteration, no finite-radius correction, and the explicit legacy-GL1 diagnostic.

Every scientific summary passed. Median CASCADE/oracle comparable-compute ratios
were `0.316`, `0.331`, `0.382`, `0.370`, and `0.513`, respectively. A separate
six-repeat long-lived target-10 check measured warmed CASCADE solve wall times of
`0.123-0.148 s`; its warmed comparable component time was about `0.103 s`, versus
about `0.132 s` for the current oracle. Thus the production command path reaches
the fractions-of-a-second regime represented by the historical curve.

`benchmark-summary.json` and the warm monitor/metadata files are the compact
evidence. Raw outputs and logs remain ignored.
