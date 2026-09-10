# CASCADE long-lived production-profile GPU diagnostic

Tracker IDs: PERF-04, PERF-06.

The exact installed `a20b6f9` wheel repeated one frozen target-1 network six
times inside one sequential `cascade sweep` process. It used the production
profile: float32, Cext GL1, independent tissue GL5, grid 256, window 6, one Cext
iteration, and the retained finite-radius/Graetz GPU functions. The first solve
was cold; the five subsequent comparable-component sums had a median of about
`1.38 s`. Peak RSS was about 1.45 GiB.

This is a production-work diagnostic, not an oracle ratio, because the frozen
legacy exporter reuses GL1 nodes for its nominal GL5 tissue stage. Exact settings,
per-run rows, process monitoring, and sweep timings are retained here.
