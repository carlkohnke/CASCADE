# Bounded production GL5 HEART-S check

Tracker IDs: VAL-06, VAL-07, PERF-05.

The exact installed `ce6f48c` wheel ran production independent tissue GL5 on
the 684-point bounded HEART-S fixture using the qualified GPU profile and warm
content-keyed domain cache. It completed successfully in 6.30 s monitored.
Vessel geometry, IDs, flow, concentration, domain, viability, and fraction
metrics match the frozen oracle. Six tissue points differ from the oracle's
native effective-GL1 output, which is the expected non-equivalent quadrature
comparison retained in `production-vs-legacy-comparison.json`.

The full-grid campaign supplies the equivalent GL5 reference comparison.
