# Historical pre-release audit — normalized index

Status: historical/inconclusive for current certification

Tracker IDs: M0-08, VAL-03, VAL-04, VAL-05, VAL-07, PERF-02

This record preserves the compact conclusions and execution commands from `docs/cascade_refactor_audit.md` under the current validation layout. The raw directories on this workstation are `.gfm_audit_runs/` and `.gfm_audit_runs_1000/`; some older prose calls them `.cascade_audit_runs*`. Raw generated files remain ignored.

## Recorded observations

| Case | Legacy wall time (s) | CASCADE wall time (s) | Result at the time |
| --- | ---: | ---: | --- |
| Cube target 1 | 5.270 | 5.566 | Compared aggregate summary metrics exactly matched |
| Cube target 100 | 6.128 | 6.675 | Compared aggregate summary metrics exactly matched |
| Cube target 1,000 | 13.471 | 13.877 | Compared aggregate summary metrics exactly matched |
| Tiny single-tree Cext | 8.464 | 7.019 | Compared outputs exactly matched |

Additional smoke observations included equal-bifurcation geometry, forest save/cache reload, grid export, simple channel, scheduled growth, nearest-tree growth, and legacy exporter/heart-growth loading.

## Why this does not certify M2 or M3

- The active full parity environment reported `svv==0.0.43`; public `svv==0.0.48` received smaller smoke tests only.
- The cases did not use the final isolated wheel-versus-SCRIPTS execution contract.
- Timings were not repeated and interleaved under the M3 protocol.
- The cube tests stop at 1,000 rather than the approved decade matrix through 10,000,000 terminals.
- Cext coverage was a tiny single-tree case, not frozen HEART-S/HEART-P shared-global Cext.
- Numerical comparisons were aggregate summaries, not the complete field-by-field rules required by VAL-02.

The data remain useful regression history and must not be overwritten by later campaigns.
