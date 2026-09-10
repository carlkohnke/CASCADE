# Legacy long-lived production-profile GPU diagnostic

Tracker IDs: PERF-04, PERF-06.

The hash-frozen current `svva2` cube oracle repeated the same target-1 structure
and point set six times inside one process with grid 256, window 6, one Cext
iteration, and finite-radius/Graetz controls. Its effective tissue quadrature is
GL1 despite the requested tissue GL5. Excluding the cold first solve, the median
comparable-component time was about `1.18 s`.

This run bounds current-oracle behavior but is not directly equivalent to
CASCADE's independent production GL5 work. The CSV, oracle metadata, and monitor
record are retained for attribution.
