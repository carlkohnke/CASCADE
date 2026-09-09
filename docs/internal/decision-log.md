# CASCADE decision log

Decisions are append-only. If a decision changes, add a superseding entry rather than rewriting history.

## D-001 — CASCADE is the product name

- Date: 2026-09-09
- Status: accepted
- Decision: User-facing package, CLI, GUI, documentation, and output terminology use CASCADE. “GFM” remains only where it denotes the Green’s function method scientifically or in legacy identifiers needed for comparison.
- Consequence: Distribution is `cascade-vascular`; Python package and primary executable are `cascade`.

## D-002 — Target unmodified public svVascularize

- Date: 2026-09-09
- Status: accepted
- Decision: The release targets public `svv==0.0.48` and must not modify installed `site-packages/svv`.
- Consequence: The public version is pinned and reported in manifests and diagnostics.

## D-003 — Prefer CASCADE-local compatibility

- Date: 2026-09-09
- Status: accepted
- Decision: Implement incompatibility handling in CASCADE whenever it can be done without changing an upstream inner loop or shared file contract.
- Consequence: Cache detection/loading, connectivity repair, path relocation, custom geometries, settings, exporters, and adapters belong in CASCADE. Upstream proposals are reserved for growth/optimization loop behavior, unavailable performance-critical primitives, and shared interchange contracts.

## D-004 — Preserve legacy scripts as validation oracles

- Date: 2026-09-09
- Status: accepted
- Decision: Root TissueSim and legacy export scripts remain available during certification but are excluded from the installed wheel and are not production imports.
- Consequence: They can be removed or archived only after Step 2 establishes adequate replacement evidence.

## D-005 — Freeze the release candidate on Python 3.9

- Date: 2026-09-09
- Status: accepted for `0.1.0rc1`
- Decision: Require Python `>=3.9,<3.10`; validate on 3.9.20.
- Consequence: Broader Python support is not claimed and must be a separately tested release change.

## D-006 — CUDA 13 is the qualified GPU configuration

- Date: 2026-09-09
- Status: accepted for `0.1.0rc1`
- Decision: Qualify CuPy 13.6.0 with NVIDIA CUDA runtime 13.2.86, cuFFT 12.3.0.29, and nvJitLink 13.2.86 on the available RTX 3080 Laptop GPU.
- Consequence: CUDA 11/12 extras are convenience options, not qualified configurations. CASCADE discovers component-wheel libraries without a manual loader path.

## D-007 — Split operational readiness from scientific/performance certification

- Date: 2026-09-09
- Status: accepted
- Decision: Step 1 proves installability and basic workflow operation. Step 2 separately proves full numerical equivalence and equivalent-or-faster performance.
- Consequence: `0.1.0rc1` must not be described as performance-certified.

## D-008 — Internal evidence is tracked; heavy output is not

- Date: 2026-09-09
- Status: accepted
- Decision: Commit plans, compact reports, settings, hashes, and decisions. Keep large generated simulations in ignored `validation/runs/`.
- Consequence: Git history remains reviewable without losing the information needed to reproduce a result.

## D-009 — Public release requires an explicit license decision

- Date: 2026-09-09
- Status: pending owner decision
- Decision needed: Select a project license and confirm obligations for CASCADE-owned copies/derivatives of `svv` components.
- Consequence: Private Git hosting is operationally possible; public reuse should not be advertised until this is resolved.

## D-010 — `.dmn` is not yet a certified interchange format

- Date: 2026-09-09
- Status: pending
- Decision needed: Either make CASCADE/public-`svv` `.dmn` round trips a supported contract or document VTK/STL as the supported interchange route.
- Evidence: The historical audit observed an installed-public-`svv` reader rejecting an adapter-created `.dmn`.
