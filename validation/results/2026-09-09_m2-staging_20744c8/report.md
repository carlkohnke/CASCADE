# M2 staging and CUBE-1 preflight

Status: M0 complete; M2 preflight passed; full numerical and performance certification not claimed.

The owner-approved workloads, tolerances, and repetition rules are frozen in D-023. The legacy seed-42 cube sampler generated one million shared float64 coordinates (24,000,128 bytes; SHA-256 `251ca61e9a082c3ae10f46a882a77fde7cacc0312faaf0b2834d14e72004e6b7`). Both solvers consumed that exact file. The bivent3 heart contract uses byte-identical grid/filter functions on a `200 x 200 x 200` candidate grid; axis hashes are in `heart-grid.json`. HEART-S full occlusion targets global segment 1, whose frozen downstream subtree contains 6,207 segments.

The first legacy wrapper attempt failed before simulation because a Numba cache expected the oracle's real module name. That failure is retained. The corrected wrapper loads the same hash-checked external file under its filename-derived name without modifying or copying it. An initial unpaired GPU run is also retained but is not parity evidence.

The first valid CPU comparison found exact blood agreement but a large water oxygen discrepancy despite exact structure, flow, and resistance. The cause was three inconsistent CASCADE water/cell-media inlet defaults. The frozen oracle uses `0.2211`; CASCADE's active runtime used `0.165825`. CASCADE now consistently uses `0.2211` in the runtime, settings registry, and Studio fallback.

After rebuilding and installing the final staging wheel SHA-256 `76fa7d83fb2d11e7f9358ac51c20a731e7516d5e846c17c5ef2f1bb83ce53c79` into a fresh Python 3.9.20 environment outside the source import path, CUBE-1 passed all 13 selected summary comparisons for both blood and water/cell media. Blood's largest relative difference was `1.88e-16`; water/cell media was exact. The legacy environment used Python 3.9.20 and `svv==0.0.43`; the CASCADE environment used Python 3.9.20 and public `svv==0.0.48`.

This is a bounded summary preflight. It does not yet establish detailed per-segment/pointwise identity, the remaining cube decades, heart parity, CPU/GPU consistency, or performance.

Verification: 112 tests passed with 15 known warnings; wheel/sdist passed `twine check`; archive audit found 77 wheel entries and 149 source entries, with no internal docs, validation data, legacy oracle scripts, or repo-only M2 harness test; `git diff --check` passed.
