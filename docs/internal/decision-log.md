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
- Status: superseded by D-013
- Original decision needed: Either make CASCADE/public-`svv` `.dmn` round trips a supported contract or document VTK/STL as the supported interchange route.
- Evidence: The historical audit observed an installed-public-`svv` reader rejecting an adapter-created `.dmn`.

## D-011 — Run legacy and CASCADE as isolated installations

- Date: 2026-09-09
- Status: accepted; supersedes the repository-copy part of D-004
- Decision: The legacy oracle runs only from the frozen `svva2/SCRIPTS` installation in its own environment. CASCADE runs from its wheel in its own clean environment and may depend on public `svv` only for vascular growth. CASCADE does not import, copy, or link to the legacy scripts.
- Consequence: The two root-level legacy script copies, obsolete same-process comparison helpers, and the automatic SCRIPTS cache fallback were removed. A neutral comparison layer may launch both sides as subprocesses and compare files after they exit.

## D-012 — Freeze the default heart/Cext validation profile

- Date: 2026-09-09
- Status: accepted
- Decision: Use float32 accelerator work arrays, a 256-point Cext grid per axis, Cext quadrature 1, tissue quadrature 5, one vessel/Cext coupling iteration, and window factor 6.
- Consequence: Runtime registry, heart exporter, Studio defaults, tests, and M2 documentation use one consistent profile. Additional runtime-stencil/runtime-moment GPU paths remain required.

## D-013 — Use mesh formats as public domain interchange

- Date: 2026-09-09
- Status: accepted; resolves D-010
- Decision: STL and other supported PyVista mesh formats are domain inputs; VTP/VTU are the supported ParaView-oriented outputs. `.dmn` remains a legacy/internal serialized domain or cache that may be loaded for a frozen validation case but is not a public cross-version interchange promise.
- Consequence: No bidirectional public-`svv` `.dmn` certification gate is required. Load compatibility is retained, and migration uses the source mesh or VTK representation.

## D-014 — Keep large canonical structures external and hashed

- Date: 2026-09-09
- Status: accepted; case naming/scale details superseded by D-018 and D-021
- Decision: Multi-gigabyte tree, forest, cache, and simulation data remain in the legacy SCRIPTS hierarchy or ignored validation runs. Git stores paths, provenance, sizes, hashes, settings, and compact comparisons.
- Consequence: HEART-L is the frozen one-millimetre extended forest/simcache; cube tests use every available decade plus the highest available five-million-terminal tree without committing those structures.

## D-015 — Package bivent3 as a built-in CASCADE domain

- Date: 2026-09-09
- Status: accepted
- Decision: Include the owner-approved `bivent3.stl` in the wheel and source distribution and expose it in CASCADE Studio through the ordinary file-backed-domain path.
- Consequence: Projects store the portable symbolic path `bivent3.stl`; the resource resolver locates it in the installed package. M4 still reviews licensing/publication obligations.

## D-016 — Preserve float32 memory paths without changing unqualified tree growth

- Date: 2026-09-09
- Status: accepted for M0
- Decision: Float32 remains required for heart/Cext/tissue accelerator arrays and caches. The public-`svv` core growth arrays remain float64 until M2 provides evidence that float32 tree growth is reliable and equivalent.
- Consequence: M0 does not make an unsafe global dtype flip. A custom oxygen-consumption law remains a future TODO, and CASCADE retains nearest-tree growth locally while requesting an upstream high-scale primitive.

## D-017 — Validate computation on existing trees before growth

- Date: 2026-09-09
- Status: accepted
- Decision: The current M2 campaign loads identical prebuilt structures on both sides and validates flow, concentration, tissue oxygen, viability, and export behavior. It does not grow or extend a tree.
- Consequence: SLSQP versus L-BFGS-B growth semantics cannot affect computational-method certification. Growth is qualified later after public SVV exposes the requested optimizer flag.

## D-018 — Freeze the available cube family through five million terminals

- Date: 2026-09-09
- Status: accepted; supersedes the cube-scale portion of D-014
- Decision: Use the cache family selected through the current legacy compatibility key at requested targets 1, 10, 100, 1k, 10k, 100k, 1M, and the highest available target, 5M. Preserve legacy requested-target labels and record actual counts separately.
- Consequence: No 10M cube input was found. The current campaign stops at 5M and reports the missing expected input rather than silently growing a new tree.

## D-019 — Use frozen shared samples and functional agreement

- Date: 2026-09-09
- Status: accepted
- Decision: Pointwise comparisons pass the same hashed tissue coordinate file to both solvers. Begin with a 0.1% functional margin for oxygenation, flow, pressure, viability, and `FracAbove1pct`, while requiring exact structural identity and matching finite-value masks.
- Consequence: CASCADE supports fixed CSV/NPY/NPZ tissue coordinates as an installed-package feature. Field-specific absolute floors near zero are recorded before certification.

## D-020 — Permit only one resident simulation

- Date: 2026-09-09
- Status: accepted
- Decision: With approximately 50 GB host RAM, validation and normal queued operation run only one memory-intensive simulation at a time.
- Consequence: CLI simulation entry points use a per-user OS lock; Studio remains a sequential child-process queue; in-process sweeps explicitly release completed result and accelerator state between cases. The M2 driver must also serialize the legacy and CASCADE subprocesses.
- Implementation note: no-growth tree/forest loads alias the populated vessel table instead of allocating a growth buffer, omit build-only spatial indexes, and stream simulation-cache data into the requested working dtype. The heart exporter defaults its working representation to float32 data and int32 connectivity.

## D-021 — Use legacy heart units and prefer the production simcache

- Date: 2026-09-09
- Status: accepted
- Decision: Heart coordinates/radii use centimetres and other quantities follow the frozen TissueSim heart/export conversions. HEART-S uses the approved 10k forest. HEART-L uses the one-millimetre extended production `.forest.simcache` for practical loading, with its source `.forest` retained as provenance.
- Consequence: Both heart input hashes are frozen. M2 does not repeatedly decompress the approximately 3 GB production forest when the approximately 6.2 GB stored simulation cache is available.

## D-022 — Retain the future float64-to-float32 growth transition

- Date: 2026-09-09
- Status: accepted; clarifies D-016
- Decision: Future tree growth uses public-SVV float64 for true CCO and transitions to float32 at the default 300,000-terminal equal-bifurcation boundary. Existing-tree computation may load into float32 directly. The current validation campaign performs no growth.
- Consequence: CASCADE retains its dtype conversion/index-rebuild utility but does not exercise or certify the automatic growth transition in M2.

## D-023 — Freeze the M2 workload, tolerance, and repetition contract

- Date: 2026-09-09
- Status: accepted
- Decision: Final cube correctness uses the complete frozen one-million-point seed-42 legacy sample pool for both blood and cell media. Heart cases use the full `200 x 200 x 200` bivent3 candidate grid; HEART-S includes healthy and one frozen full downstream occlusion. Physical fields use `1e-3` relative tolerance with a `1e-6`-of-reference-case-scale absolute near-zero floor; fractions use `0.001` absolute tolerance; identity fields and finite masks are exact. CPU/GPU consistency is bounded to small/medium cases, while 1M/5M/HEART-L are GPU-only. Performance uses five pairs through 100k, three at 1M, and one initial pair at 5M/HEART-L, repeated when failed, memory-pressured, or within 5% of the release threshold.
- Consequence: Large cases emit compact evidence first. Full export parity is certified on bounded cases, followed by at most one carefully monitored HEART-L full export after numerical and memory gates pass. These choices close the remaining scientific/protocol questions in M0 without claiming M2 results.

## D-024 — Use the frozen cube oracle's cell-media inlet oxygen

- Date: 2026-09-09
- Status: accepted from M2 preflight evidence
- Decision: Normalize the CASCADE runtime, settings registry, and Studio fallback for water/cell-media inlet oxygen to `0.2211`, matching the frozen cube oracle. Blood remains `0.14`. The editable Studio new-project preset is not a runtime default and remains user-configurable.
- Consequence: The first CUBE-1 water preflight failure (25% vessel and 43% mean-tissue discrepancy) is retained. After the correction, the fresh-wheel blood and water/cell-media summary comparisons pass the frozen M2 rules. This is a bounded preflight, not full M2 certification.

## D-025 — Convert public pressure outputs from solver cgs units to pascals

- Date: 2026-09-09
- Status: accepted from M2 harness inspection
- Decision: Keep the Kirchhoff implementation and legacy-oracle comparison in `dyn/cm^2`, but divide solved pressures by 10 before CASCADE writes public summary, segment CSV, or VTK fields documented as pascals. The M2 array comparator performs the same normalization on both solver sides.
- Consequence: Existing historical pressure values are interpreted using their recorded implementation context. New CASCADE artifacts have truthful `pressure_pa` units without changing the numerical solver or resistance calculations.

## D-026 — Do not inflate growth payloads for simulation-only tree loads

- Date: 2026-09-09
- Status: accepted from M3 attribution
- Decision: When `growth.enabled=false`, load only the `data` member of a legacy `.tree.npz` archive. Do not read or unpickle its growth-only payload, which may contain a many-million-entry vessel-map object graph. Infer the terminal count from the stored child columns and attach only the compact simulation state, matching the frozen oracle's analysis-only loader.
- Consequence: Saved custom growth parameters in the payload are intentionally not an implicit simulation input; CASCADE's explicit settings remain authoritative. Growth-enabled loads continue to restore the full payload. This change must reproduce the existing array evidence through a freshly installed wheel before the earlier candidate is superseded.

## D-027 — Reject invalid tissue-grid specifications instead of coercing them

- Date: 2026-09-09
- Status: accepted from M2 negative testing
- Decision: Reject non-positive grid dimensions/resolutions/chunk sizes, unknown tissue-grid keys, invalid combine modes, and non-finite or negative enclosed-point tolerances during configuration validation. Do not silently clamp invalid user input to a different simulation.
- Consequence: The initial negative campaign's successful `nx=0` run is retained as failed evidence. A fresh committed wheel must pass the unchanged failure-path case before VAL-11 can close.
