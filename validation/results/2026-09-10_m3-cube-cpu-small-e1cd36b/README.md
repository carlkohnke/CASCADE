# Retained failed M3 harness attempt

Tracker IDs: PERF-01 and PERF-02.

The first legacy target-1 run completed, but CASCADE failed before import. The
driver had resolved the virtual environment's `bin/python` symlink to the base
`svva2` interpreter, thereby discarding the isolated wheel environment and
raising `ModuleNotFoundError: cascade`.

The monitor JSON and ignored raw logs are retained. The driver now invokes the
virtual-environment path without resolving its symlink. The replacement
campaign is `../2026-09-10_m3-cube-cpu-small02-e1cd36b/`.
