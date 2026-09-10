# Exact-wheel heart domain cache qualification

Tracker IDs: PERF-05 and PERF-06.

The clean `ce6f48c` wheel created and then reused an isolated bivent3 domain
cache keyed by the full STL hash and public `svv==0.0.48`. Cold construction
required 34.61 s. A new process loaded the cache in 0.05 s and reported 2.67 s
application time for the bounded GPU heart export (5.57 s externally monitored
including interpreter/CUDA startup). Both runs stayed below 1 GiB peak RSS.

Monitor JSON files contain the exact commands, host, and resource observations.
The cache, logs, and generated outputs remain under ignored validation paths.
