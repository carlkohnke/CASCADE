# CASCADE production work log

This log records what was done. Current status and next actions belong in `release-tracker.md`.

## 2026-09-09 — Step 1 release candidate completed

### Scope

Turn the research checkout into an installable CASCADE release candidate containing the CLI solver, CASCADE Studio GUI, public-`svv` dependency, custom geometry support, and packaged heart/ParaView exporter.

### Work performed

- Created branch `release/cascade-0.1`.
- Preserved pre-release state in commit `3c040c8`.
- Moved the production Python package from `gfm` to `src/cascade` and renamed public interfaces to CASCADE.
- Added `pyproject.toml` packaging for CLI, GUI, viewer, doctor, and heart exporter entry points.
- Pinned Python 3.9 and public `svv==0.0.48` with Linux CPU and CUDA 13 lock files.
- Added strict settings validation, reproducibility manifests, custom CSV/NPZ networks, portable launchers, CI configuration, release/security/contribution documentation, and ParaView exports.
- Packaged the heart exporter and TissueSim-compatible runtime without hardcoded input or module paths.
- Implemented public-`svv` simulation-cache recognition and generic relocation of old package-owned tree-cache paths in CASCADE adapters.
- Documented the CASCADE-local-first public-`svv` compatibility policy.

### Release-gate findings and fixes

1. The first source archive included ignored generated example outputs. Added source-manifest pruning in commit `5321135`.
2. The installed CUDA wheel passed a basic CuPy kernel but failed the heart FFT path because `libcufft.so.12` was absent. Added pinned cuFFT/nvJitLink dependencies in commit `a4a9f82`.
3. NVIDIA component wheels installed their libraries outside the default Linux loader path. Added CASCADE discovery/preloading and expanded the doctor test to execute cuFFT in commit `25bb0e1`.
4. A benchmark helper retained a user-specific `site-packages` path. Replaced it with discovery through `svv.__file__` in commit `9da8b2e`.

### Verification performed

- Full source suite: 100 passed, 15 warnings.
- Final wheel installed into independent CPU-only and CUDA 13 temporary environments.
- Installed CLI ran tree, custom Y-channel, and two-tree forest workflows.
- CPU heart exporter completed tissue oxygen output.
- CUDA shared-global Cext exporter completed kernel, FFT, vessel, and tissue output on an RTX 3080 Laptop GPU.
- Generated VTP/VTU files reopened through PyVista.
- CASCADE Studio completed offscreen construction, display, event loop, and exit.
- Wheel and sdist passed `twine check` and archive-content audit.
- Final checksums verified.

### Result

- Release tag: `v0.1.0rc1`
- Release commit: `9da8b2e`
- Release report: `../release-0.1.0rc1.md`
- Artifacts and checksums: `../../dist/`
- Known warning: the retained L-BFGS-B bifurcation path does not enforce the legacy general inequality constraint through SciPy; tracked for Step 2 disposition.
- Not claimed: comprehensive numerical equivalence or equivalent/faster performance.

## 2026-09-09 — Internal production record established

### Work performed

- Added this internal documentation index and living release tracker.
- Added a Step 2 certification plan and repeatable release runbook.
- Added an append-only decision log.
- Defined tracked versus ignored validation evidence locations.
- Added `AGENTS.md` so future release/testing work updates these records as part of the task.

### Reason

Existing release reports and audits described individual snapshots but did not provide a single, continuously maintained view of current status, open work, evidence, and next actions.

## 2026-09-09 — M0 baseline normalized for M2

### Scope

Complete the unambiguous M0 work while retaining unresolved scientific choices for owner review.

### Work performed

- Froze exact paths, sizes, hashes, and environment versions for the cube, heart export, heart growth, heart accelerator, and lower-priority accelerated legacy scripts in `svva2/SCRIPTS`.
- Compared the former tracked cube script with the SCRIPTS oracle: they are `0.9699117570` line-similar but not identical. Recorded the root-only GPU/domain helpers and SCRIPTS-only control globals; retained SCRIPTS as the external oracle and CASCADE as an independent implementation.
- Froze the actual one-millimetre extended HEART-P forest (12.5M terminals, 24,999,999 segments), its legacy domain-cache hash, and the approved bivent3 STL hash.
- Added `m0-legacy-inventory.md` and `m0-traceability.md` with capability, parameter, output, case, and disposition mappings.
- Established a strict subprocess/file boundary between the wheel-installed CASCADE side and the legacy `svva2` side. Removed root-level oracle copies, obsolete same-process benchmark/audit helpers, and automatic SCRIPTS cache relocation from production code.
- Normalized the CASCADE profile to float32 accelerator arrays, Cext grid 256, Cext quadrature 1, tissue quadrature 5, one coupling iteration, and window factor 6 across runtime defaults, settings registry, heart exporter, and Studio.
- Added bivent3 as a packaged, portable Studio domain option through the ordinary file-backed-domain resolver.
- Classified `.dmn` as legacy/internal load compatibility rather than public interchange; retained STL/PyVista-readable input and VTP/VTU output as the supported contract.
- Retained explicit TODOs for a custom tissue oxygen-consumption law and an upstream high-scale nearest-tree primitive while preserving CASCADE's local nearest-tree implementation.
- Corrected the custom CSV documentation: radius is optional and falls back to `network.simple.radius_cm`.
- Preserved the historical 1/100/1000 and tiny-Cext observations under `validation/results/` with their `svv==0.0.43` and protocol limitations.
- Excluded the historical audit containing workstation-specific paths from future source archives.

### Verification

- Full source suite: 102 passed, 15 known warnings.
- Targeted installed-resource check loaded `bivent3.stl` from a wheel-only environment outside the checkout and reproduced size 652,584 bytes and SHA-256 `10fd497650eb881b37390861cb960db88e0c7e6f0b18267ffbb35d564c24a060`.
- PyVista read the packaged heart surface as 6,467 points and 13,050 cells; physical units remain an explicit M2 input-confirmation item.
- Temporary wheel and source archive built successfully and passed `twine check`; package audit confirmed bivent3 inclusion and no legacy script copies, internal docs, or validation results.
- `git diff --check` passed.

### Evidence

- `../../validation/results/2026-09-09_m0-inventory_20744c8/`
- `../../validation/results/2026-09-09_historical-audit_normalized/`

### Remaining owner/scientific choices

HEART-S, cube fixtures/seeds, shared sample coordinates, external input units, numerical tolerances/mapping rules, float32 public-SVV tree growth, and the L-BFGS-B constraint disposition remain open. M0 stays PARTIAL and no M2/M3 certification claim is made.

## 2026-09-09 — Existing-tree fixtures and memory boundary frozen

### Work performed

- Confirmed that the current legacy cache matcher selects stored cube family `a50c006ac491d07e399f94f1be37cfe0` through compatibility-key normalization.
- Hashed the complete available decade matrix through 1M and the highest available 5M tree. No 10M cube input exists in the inspected cache hierarchy; no tree was built.
- Froze HEART-S as `heart_seed2_grown100000_t10000.forest` and hashed the preferred HEART-L one-millimetre extended simulation cache.
- Confirmed heart centimetre units by loading HEART-S in the legacy environment and inspecting its CGS unit system and geometry bounds.
- Added fixed tissue-coordinate input to the installed CASCADE settings and Studio workflows for CSV, NPY, and NPZ, including resolved-path/hash provenance.
- Added a per-user OS lock to `cascade run`, `cascade sweep`, and `cascade export-heart` so independently launched CASCADE simulations cannot overlap.
- Added explicit result deletion, Cext/tissue global-state clearing, CUDA synchronization, CuPy pool trimming, and garbage collection between in-process sweep cases.
- Changed no-growth tree and forest loading to omit duplicate growth preallocation, midpoint/search structures, and vessel maps. Simulation-cache NPY members now stream directly from their stored dtype into the requested working dtype, avoiding a simultaneous full float64 source array plus float32 copy. Heart loading defaults to float32 data and int32 connectivity.
- Updated the test plan to validate computation on existing trees only, run legacy and CASCADE sequentially, report solver-only and end-to-end timings, and begin with a 0.1% functional agreement margin.

### Verification

- Streamed SHA-256 hashing without loading the large tree/simcache files into memory.
- Loaded only the 4 MB HEART-S forest for unit and structure metadata: two trees, 19,999 total segments, 10,000 terminals, float64 data, and int64 connectivity.
- Targeted loader, fixed-sample, lock, Studio sampling, and release-contract tests: 15 passed.
- Full source suite: 109 passed, 15 known warnings.
- Temporary final-candidate artifacts passed `twine check` and archive audit: 77 wheel entries and 149 sdist entries, bivent3 present, internal docs/validation/oracle scripts absent.
- The exact final temporary wheel (`9ee7a7e0b49e1363f89629d02589f7faa3212789eef39cc948bed6859527da25`) passed independent CPU and CUDA 13 installed-wheel smokes, including tree, custom geometry, forest simcache, CPU heart export, GPU shared-global FFT/Cext heart export, cuFFT/device checks, VTK reopen, and manifest checks.
- Evidence: `../../validation/results/2026-09-09_m0-fixtures_20744c8/`.

### Important finding

The current TissueSim cube script does not implement a general automatic float64 CCO to float32 equal-bifurcation transition. Its validation path finds existing float64 cache files and loads their working representation as float32. Because the current campaign performs no growth, CASCADE retains the conversion utility and records the desired 300k transition for the later growth campaign rather than claiming it is already qualified.

## 2026-09-09 — M0 closed and isolated M2 CUBE-1 preflight passed

### Work performed

- Recorded the owner's final M2 workload, tolerance, backend, export, occlusion, and performance-repetition choices as D-023.
- Added hash-checking validation utilities for frozen cube-coordinate generation, isolated legacy cube execution, deterministic heart-grid recording, and CSV comparison. The legacy wrapper does not copy or modify the oracle and CASCADE never imports it.
- Generated the complete one-million-point legacy seed-42 cube pool under ignored `validation/runs/` and recorded its SHA-256, bounds, units, dtype, and generator hash.
- Froze the bivent3 `200 x 200 x 200` axis/order contract. Confirmed the legacy and CASCADE grid/filter function blocks were byte-identical at staging time.
- Selected HEART-S global segment 1 (tree 0/local 1) for the approved full occlusion; its frozen downstream subtree contains 6,207 segments.
- Built a fresh wheel-installed Python 3.9.20/public-`svv==0.0.48` environment and ran legacy then CASCADE sequentially on CUBE-1 with the full shared point pool and no growth.
- Retained a failed wrapper attempt caused by Numba module-name cache metadata and an unpaired legacy GPU observation instead of overwriting them.
- Diagnosed the water/cell-media preflight failure to inconsistent CASCADE inlet oxygen defaults. Normalized runtime, settings, and Studio fallback values to the frozen oracle's `0.2211`, rebuilt the wheel, and reran both fluids.

### Verification

- CUBE-1 blood: 13/13 summary fields passed; maximum relative difference `1.88e-16`.
- CUBE-1 water/cell media: 13/13 summary fields passed exactly after D-024.
- Full source suite: 112 passed, 15 known warnings.
- Final staging wheel (`76fa7d83fb2d11e7f9358ac51c20a731e7516d5e846c17c5ef2f1bb83ce53c79`) and sdist (`625060f9dabd9d49bbfd2e73fea91dd5bb8e3bd68abac32feaf46219efec78f3`) passed `twine check`; archive audit found 77 wheel and 149 sdist entries, bivent3 present, and no internal docs, validation data, oracle scripts, or repo-only M2 harness test. Both fluids were rerun through that exact final wheel.
- `git diff --check` passed.

### Evidence and status

- Evidence: `../../validation/results/2026-09-09_m2-staging_20744c8/`.
- M0 is COMPLETE. M2 is IN PROGRESS; this summary preflight is not detailed-array, heart, GPU-consistency, scale-matrix, or performance certification.

## 2026-09-09 — Unified M2/M3 characterization harness staged

### Work performed

- Checkpointed the reviewed M0-complete/M2-staging state at commit `1f262f7` after 112 tests passed.
- Added isolated installed-wheel and legacy cube array runners. They write individually memory-mappable arrays for topology, geometry, flow, pressure, hematocrit, vessel concentration, tissue coordinates, and tissue oxygen.
- Added a chunked comparator enforcing exact identity and finite-mask rules plus the approved physical-field tolerance without loading duplicate full arrays.
- Added a process-tree resource monitor that records wall time, peak RSS, GPU memory where visible through `nvidia-smi`, logs, environment, and a 45 GiB host-memory safety abort. A smoke command verified its output and lock behavior.
- Expanded scalar comparison coverage to pressure, flow, resistance, Damkohler, and all reported viability fractions.
- Found that Kirchhoff pressure arrays use `dyn/cm^2` while public CASCADE fields promise pascals. Added explicit conversion for public summaries and segment/VTK output and recorded D-025; comparison evidence normalizes both internal solver sides to pascals.

### Verification

- Targeted release-contract, harness, and execution tests: 16 passed with 14 third-party warnings.
- The resource-monitor smoke recorded process-tree memory, host metadata, output hash, and a successful exit.
- `git diff --check` passed.

### Status

- The first characterization artifact built from `1f262f7` is superseded before numerical use because the pressure-unit correction changes packaged production code. A new checkpoint and wheel are required before the array campaign begins.

## 2026-09-09 — Frozen cube numerical matrix passed through five million terminals

### Work performed

- Checkpointed the pressure-corrected array harness at commit `9cb7712` and built an exact candidate wheel with SHA-256 `2bf57b8a39cfc39b2f18836c8c4eb83788698396a55693f7426311f13b0331ce`.
- Installed that wheel into an isolated Python 3.9 environment using public `svv==0.0.48`; added the pinned CUDA 13 component wheels to both isolated solver environments and verified compiled CuPy execution. CASCADE also passed `cascade doctor --require-gpu`.
- Strengthened the legacy runner to require an explicit tree path and SHA-256, override the oracle's cache lookup with only that file, and fail if any lower-count cache/growth path is requested.
- Executed legacy then CASCADE sequentially for blood and water/cell media at 1, 10, 100, 1k, 10k, 100k, 1M, and 5M terminals using the same one-million-point coordinate fixture. CPU was used through 100k and CUDA at 1M/5M.
- Captured 17 memory-mappable arrays and 23 summary fields per case. The 1-terminal cases were rerun in a separate append-only campaign after explicit-tree enforcement was added.
- Reduced external GPU-memory polling to a two-second cadence while retaining quarter-second process-tree RSS sampling and the 45 GiB abort.
- Added a deterministic campaign summarizer that consolidates pass/fail, exact mismatch counts, tolerance failures, evidence-enabled wall time, memory, and load time without treating those diagnostic walls as M3 certification.

### Results

- All 16 scale/fluid cases passed; there were zero exact mismatches and zero tolerance failures.
- The largest comparison covered 10,000,001 segments plus 1,000,000 tissue samples in each fluid.
- Peak CASCADE RSS was 16.65 GiB at 5M terminals, below the 45 GiB safety limit; legacy peak RSS was 6.20 GiB or less.
- Large-tree end-to-end overhead was attributed primarily to CASCADE network loading/conversion: 56-62 s at 5M versus about 8 s in the oracle. The numerical solve remained much closer and must be separated from load cost in the clean M3 protocol.
- `nvidia-smi` did not expose per-process WSL GPU allocation in these runs, so its recorded zero is not treated as proof of zero GPU memory use. Successful compiled kernels and solver GPU logs establish execution; host RSS remains the enforced safety metric.

### Evidence

- Consolidated report: `../../validation/results/2026-09-09_m23-characterization_9cb7712/cube-characterization.md` and `.json`.
- Exact-tree CUBE-1 rerun: `../../validation/results/2026-09-09_m23-cube1-explicit_9cb7712/`.
- Heavy arrays and logs: corresponding ignored directories under `../../validation/runs/`.

### Verification

- Each monitored record contains the exact command, environment, log hash, wall time, peak process-tree RSS, and memory-abort status.
- Each legacy record contains the frozen oracle, point, and tree hashes; each CASCADE record identifies the installed wheel module path and version.
- The report generator returned success only after all array and summary comparison files reported `pass`.
