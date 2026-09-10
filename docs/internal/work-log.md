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

## 2026-09-09 — Large simulation-only tree-load regression resolved

### Finding and change

- The cube matrix showed that CASCADE network loading grew to 16.6 seconds at 1M and 61.6 seconds at 5M terminals, versus 1.8 and 7.8 seconds for the frozen oracle.
- Inspection confirmed that CASCADE's analysis-only path still read and unpickled the archive payload, which can contain a many-million-entry growth vessel-map object graph. The oracle intentionally reads only `data`.
- Implemented D-026 at commit `fa2c958`: simulation-only `.tree.npz` loading reads only the vessel table, infers terminal count from child columns, and constructs compact state. Growth-enabled loading still restores the payload.
- Added a regression test with deliberately inconsistent payload counts to prove that analysis-only loading uses the stored numerical table rather than payload metadata.

### Fresh-wheel verification

- Full source suite before build: 115 passed, 15 known warnings.
- Built exact wheel SHA-256 `de21614b728dd3963db83176e3c7c99ed45b8f2342ea83a6c59985f24565f8cf` and sdist SHA-256 `15181a83dee50f1354d2f2e52428b8be05f0d0309285f1ef3b00df5a23344be9`; both passed `twine check`.
- Installed the wheel with pinned CUDA 13 dependencies in a new Python 3.9 environment; `cascade doctor --require-gpu` passed.
- At 1M terminals, network load fell from 16.602 to 2.052 seconds and peak RSS from 5.15 to 2.05 GiB.
- At 5M terminals, network load fell from 61.559 to 8.124 seconds and peak RSS from 16.65 to 6.48 GiB.
- Both optimized blood runs reproduced all 17 legacy arrays and 23 summary fields with zero exact mismatches and zero tolerance failures.

### Evidence

- `../../validation/results/2026-09-09_m23-loadfix-fa2c958/`.
- Heavy emitted arrays/logs remain under the corresponding ignored `../../validation/runs/` directory.

## 2026-09-09 — CUBE-100 CPU/GPU consistency passed

### Work performed

- Extended cube settings staging with explicit backend, fluid, and label controls so CPU and CUDA runs can coexist without overwriting evidence.
- Generalized the summary comparator's reference-pressure unit so CASCADE-to-CASCADE comparisons preserve already-SI pressure values while legacy comparisons retain cgs-to-SI normalization.
- Ran the exact installed `fa2c958` wheel sequentially on CUBE-100 blood with the same hashed tree and one-million tissue coordinates, once on CPU and once on CUDA.

### Results and evidence

- All 17 arrays and all 23 summary fields passed VAL-02.
- Identity fields had zero mismatches. The tissue field's maximum relative error was `2.4555e-05`, below `1e-3`; its maximum absolute error was `5.4288e-07`.
- Evidence-enabled wall time was 31.55 seconds CPU and 8.17 seconds GPU; peak host RSS was 2.84 and 1.43 GiB. These single cold observations are diagnostic only.
- Evidence: `../../validation/results/2026-09-09_m2-cpu-gpu-fa2c958/`; heavy arrays/logs under the matching ignored runs directory.

## 2026-09-09 — Custom geometry/domain and bounded normal export passed

### Work performed

- Added small canonical NPZ Y-channel, bivent3-contained CSV Y-channel, and VTP box-boundary fixtures with recorded hashes.
- Ran four exact-wheel cases from `/tmp`: CSV/box, NPZ/box, CSV/bivent3 STL, and NPZ/VTP domain.
- Added an export inspector that reopens every emitted VTK dataset, inventories field names/dtypes, verifies nonempty geometry and complete segment IDs, and compares VTK physical values with CSV rows.
- Retained the first failed CUSTOM-Y inspection, which incorrectly expected `tissue_oxygen`; corrected the validation contract to CASCADE's actual `local_concentration` field and emitted a new v2 result.

### Results

- All four cases completed outside checkout and each passed 23 export checks.
- The CSV and NPZ representations produced byte-identical segment and point CSVs; all 14 scientific summary fields available for simple networks matched exactly.
- The bivent3-contained fixture's endpoints and segment midpoints were checked as enclosed by the packaged surface before execution.
- VAL-09 is complete. VAL-06 remains partial until heart-specific exporter fields/values are compared.

### Evidence

- `../../validation/results/2026-09-09_m2-custom-export-fa2c958/`; large/generated run products remain under the corresponding ignored `../../validation/runs/` directory.

## 2026-09-09 — Negative preflight found invalid-grid coercion

### Work performed

- Added an installed-entry-point harness covering unknown settings, missing and malformed inputs, disconnected custom geometry, invalid grid dimensions, incompatible flow settings, corrupt VTK/forest inputs, and a requested GPU with no visible device.
- Ran all ten cases sequentially against the exact installed `fa2c958` wheel outside the checkout and required non-zero exit, matching root-cause text, and no successful manifest.
- Added strict tissue-grid validation and parameterized regression tests after the run showed that `nx=0` was silently clamped to one.

### Results and evidence

- Seven of ten initial harness checks passed. Two more commands failed correctly but used different message wording than the harness expected; the expectations were corrected.
- The invalid-grid case was a genuine product failure and its successful manifest is retained under ignored raw evidence. The compact failed result is `../../validation/results/2026-09-09_m2-negative-fa2c958/negative-matrix-initial.json`.
- `tests/test_release_contract.py` passed 12 tests after the source fix. A fresh committed wheel rerun remains required before VAL-11 completes.

## 2026-09-09 — Negative and reproducibility gates completed

### Work performed

- Ran the full source suite after the invalid-grid fix: 118 passed with the 15 already-classified warnings.
- Built wheel SHA-256 `51f75bf6accb916f1aa6b86861ef43d76491107bd0dbbd5f2622850fdd077608` and sdist SHA-256 `a010c72273063d33854f796e6e5d116fad5e2fa106da0912db1bc8be06a2571f` from clean commit `128c6fc`; both passed `twine check`.
- Installed that wheel into two independent Python 3.9 CPU environments using the frozen constraints.
- Reran all ten adversarial cases, then ran CUBE-100 blood with the same fixed tree and one-million tissue points once in each environment.
- Added a manifest comparator that explicitly removes only declared volatile fields before comparing every remaining leaf.

### Results and evidence

- Negative matrix: 10/10 pass, non-zero exits, expected root causes, no successful manifests. Evidence: `../../validation/results/2026-09-09_m2-negative-128c6fc/`.
- Reproducibility: all 17 arrays, 23 summaries, and 90 stable manifest leaves matched exactly. Evidence: `../../validation/results/2026-09-09_m2-repro-128c6fc/`.
- VAL-11 and VAL-12 are complete. Raw outputs remain under matching ignored `../../validation/runs/` directories.

## 2026-09-10 — HEART-S preflight exposed domain and state regressions

### Work performed

- Ran the primary legacy exporter, its retained pre-state-slim backup, and the exact installed `128c6fc` CASCADE wheel sequentially on HEART-S with a 16-cubed tissue grid and the full frozen 256-cubed shared Cext configuration.
- Preserved the primary exporter's `k_if_gl` failure and hash-froze the tissue-capable June-17 backup.
- Traced CASCADE's incorrect 4,096/4,096 inside-point count to the heart exporter calling the packaged runtime's default cube builder instead of its mesh constructor.
- Added the mesh-domain route and retained `c_bulk_gl`, `c_wall_gl`, and `k_if_gl` in the compact combined Cext state, with regression tests.

### Results and evidence

- Primary legacy exporter: failed after shared Cext, as expected from the newly identified SCRIPTS mismatch.
- Pre-state-slim legacy exporter: completed, retaining 684/4,096 bivent3 points and passing its flux consistency check.
- Initial CASCADE wheel: completed but was scientifically invalid because it retained all bounding-cube points and skipped the flux check.
- Evidence: `../../validation/results/2026-09-09_m2-heart-s-preflight-128c6fc/`; raw VTK/log products are in the matching ignored runs directory.
- VAL-07 remains partial until a fresh committed wheel passes this bounded comparison and the full 200-cubed healthy/occlusion cases.

## 2026-09-10 — HEART-S shared-coordinate contract implemented

### Work performed

- Reran the corrected source at commit `881d26b` on the bounded 16-cubed HEART-S case. CASCADE used the actual bivent3 mesh, retained 658 generated inside points, completed tissue evaluation, and passed the Cext flux diagnostic with relative L2 error approximately `4.98e-8`.
- Compared the corrected CASCADE vessel VTP to the tissue-capable legacy oracle: all 39,998 vessel points, 19,999 cells, geometry, flows, radii, lengths, and IDs matched exactly; concentration differences were at float32 scale.
- Identified a 26-coordinate difference between the two independently generated surface masks (684 legacy versus 658 CASCADE) despite using the same STL and grid axes.
- Added D-031 and `cascade export-heart --tissue-points` so production and validation runs can consume an explicit numeric `(N, 3)` `.npy` array in centimetres. Metadata records its resolved path, SHA-256, and count.
- Added a validation-only legacy wrapper that verifies the external exporter hash and either captures its generated points or substitutes a fixed point array without editing or copying the oracle.
- Captured the bounded oracle's 684 inside coordinates with SHA-256 `9298fab8e103e595c4f674298695d445fed222ca410f5cd5d67a1757b04f9001`. Two failed wrapper bring-up attempts are retained rather than overwritten.

### Verification and next gate

- Full source suite: 123 passed with 15 previously classified warnings.
- The successful capture run completed with peak RSS below 1 GiB and the frozen production Cext profile. Compact monitor evidence is under `../../validation/results/2026-09-10_m2-heart-shared-points_working/`; raw logs, VTK, and `.npy` data are under the matching ignored runs directory.
- A committed, freshly installed wheel must now rerun both sides against the captured points and pass the field comparator before the full 200-cubed coordinate fixture is generated.

## 2026-09-10 — HEART-S quadrature discrepancy and first clean M3 tranche

### Work performed

- Built and installed the exact `e1cd36b` wheel (SHA-256 `201c6459...`) outside the checkout, then ran both heart implementations sequentially against the same 684 frozen coordinates.
- Added a reusable heart VTP comparator covering exact identities/coordinates/finite masks, the frozen physical-field tolerance, fraction tolerances, and unordered boundary geometry.
- Added isolated installed-wheel cube timing runners. They enforce the 45 GiB memory ceiling, alternate legacy/CASCADE order, retain raw logs outside Git, avoid array-evidence writes, and record build/solve/export plus solver components.
- Preserved the first failed timing campaign: resolving a virtual-environment Python symlink invoked the base interpreter and lost the wheel installation. The corrected driver retains the virtual-environment executable path and gained a quiet-child logging mode.

### Verification and evidence

- HEART-S shared-point geometry, IDs, flows, radii, lengths, finite masks, viability, and fraction-above-1% agree. Field parity remains failed because the legacy shared-Cext path effectively performs tissue GL1 while CASCADE performs the requested tissue GL5; D-033 is pending before large heart work. Evidence: `../../validation/results/2026-09-10_m2-heart-s-shared-e1cd36b/`.
- The clean five-pair CPU cube tranche passed all 15 scientific summary comparisons. Warm compile-cache median CASCADE/legacy process ratios were 1.035 at target 1, 1.018 at target 10, and 0.974 at target 100; maximum observed RSS was 2.78 GiB. Evidence: `../../validation/results/2026-09-10_m3-cube-cpu-small02-e1cd36b/`.
- The superseded failed harness evidence remains at `../../validation/results/2026-09-10_m3-cube-cpu-small-e1cd36b/`.

## 2026-09-10 — GPU timing contract recovered and quadrature semantics resolved

### Work performed

- Recovered the exact 74-row June GPU characterization from the external `Cube_Memory_Improvement.csv`, froze its SHA-256, profile, and timing-field definition, and added a compact grouped result under `../../validation/results/2026-09-10_m3-gpu-reference_fecf3759/`.
- Verified that the plotted algorithm time is `t_assembly + t_kirchhoff + t_concentration + t_tissue`, not fresh-process wall time. The reference reports 0.880-1.031 seconds at 400,001 vessels and 18.09-23.19 seconds at 10,000,001 vessels.
- Stopped the in-progress CPU extension after five target-1,000 and three target-10,000 pairs when the owner limited CPU certification to targets 1/10/100. The incomplete evidence is retained under `../../validation/results/2026-09-10_m3-cube-cpu-medium_9423771/`.
- Added an explicit post-Cext quadrature policy. Production `independent` mode solves Cext with GL1 and resamples to independent tissue GL5; `legacy_cext` reproduces the oracle's effective-GL1 tissue evaluation only for diagnostics.
- Tightened the benchmark harness to configure the Cext GPU profile explicitly, report the historical four-component algorithm boundary, retain scientific failures when requested, record installed-distribution provenance, and reject an interpreter that is not running the declared wheel.

### Verification and evidence

- Targeted Cext tests pass, including the independent GL1-to-GL5 resampling contract, legacy diagnostic mode, and invalid-mode rejection.
- The production-GL5 target-1 preflight passed geometry, hemodynamics, vessel oxygen, and viability but failed mean tissue oxygen versus legacy effective GL1, as expected: `../../validation/results/2026-09-10_m23-gpu-reference-smoke_9423771/`.
- A source-only `legacy_cext` preflight passed all 23 summaries, confirming the cause. It is explicitly non-certifying because its metadata proves that it invoked the editable checkout while the old header merely named a wheel: `../../validation/results/2026-09-10_m23-gpu-legacy-quadrature-source_9423771/`.
- D-034 records the owner's final GL1/GL5 decision; D-035 records GPU-first cube and GPU-only heart certification; D-036 fixes the comparable timing boundary. A fresh exact wheel is required before the next GPU campaign.

## 2026-09-10 — GPU cube speed restored and verified through ten million vessels

### Work performed

- Enabled persistent Numba caches for retained flow/ordering helpers and corrected `cascade sweep` so an explicit cached structure is loaded once and may be repeated only at its actual target count.
- Kept completed result/state cleanup between sequential sweep cases while retaining CuPy allocator pools inside the same bounded worker.
- Extended the alternating exact-wheel harness to name finite-radius and lumen-wall settings explicitly and added an in-process legacy repetition mode.
- Ran five alternating GPU pairs at every frozen target from 10 through 5M terminals, always one process at a time under the 45 GiB monitor.

### Results and evidence

- All scientific summaries passed at every scale. CASCADE comparable compute was faster than the current frozen oracle at every target. At 10,000,001 vessels, median CASCADE/oracle compute was `26.16/27.65 s` (ratio `0.947`) and peak RSS remained about 11-12 GiB.
- A six-run target-10 sequential check reached `0.123-0.148 s` warmed CASCADE solve wall time and approximately `0.103 s` comparable compute, confirming the fractions-of-a-second regime behind the historical figure.
- Compact evidence: `../../validation/results/2026-09-10_m3-gpu-reference-scale-a20b6f9/`, `../../validation/results/2026-09-10_m3-gpu-reference-large-a20b6f9/`, and the before/after cache campaigns in `../../validation/results/2026-09-10_m3-gpu-small-legacy-c0db166/` and `../../validation/results/2026-09-10_m3-gpu-small-cache-59f27e5/`.

## 2026-09-10 — HEART-S diagnostic exposed and fixed repeated domain setup

### Work performed

- Repeated the bounded HEART-S GPU comparison from the exact installed `a20b6f9` wheel in explicit legacy-GL1 diagnostic mode.
- Retained the original strict comparator failure and clarified the float32 near-zero floor after the only failed field contained 66 `cext_mean` values with maximum absolute difference `6.51e-7`.
- Attributed roughly 33.5 seconds of CASCADE heart wall time to rebuilding the identical STL-backed public-`svv` domain on every isolated run.
- Added a source-hash/cache-schema/`svv`-version-keyed CASCADE user domain cache with explicit location and disable controls.

### Verification and next gate

- The corrected diagnostic passes structure, flow, concentration, tissue oxygen, viability, coordinates, and finite masks in `../../validation/results/2026-09-10_m2-heart-s-quadrature-a20b6f9/`.
- Targeted release/sweep/execution tests pass: 22 passed with only the known third-party warnings.
- Build a fresh exact wheel, verify cold cache creation and warm reuse, then execute production-independent-GL5 HEART-S full-grid healthy/occlusion followed by the monitored HEART-L gate.

## 2026-09-10 — Full-grid HEART-S GL5 reference and infarction anatomy correction

### Work performed

- Captured the frozen legacy bivent3 inside mask for the full 200-cubed candidate grid: 1,619,996 coordinates, SHA-256 `ab733bd07827e33e7458f9e474a5b6f6ec12f8eff45cf41c4bd00c268cbc9fdb`.
- Ran healthy and global-segment-1 full-occlusion cases sequentially in both environments on those exact coordinates with GPU-only execution.
- Added a validation-only, source-conserving GL1-to-GL5 resampler around the hash-frozen oracle's GPU tissue evaluator, because its native shared-Cext path ignores the requested tissue GL5 order.
- Corrected CASCADE so zero-radius/radius-reduction solve overrides are restored in exported solution geometry before tissue masking and VTP construction.

### Results and next gate

- The healthy production GL5 comparison passes every structural, field, finite-mask, and viability rule. The native effective-GL1 comparison is retained and shows the expected quadrature difference rather than being promoted to production evidence.
- The first occlusion GL5 comparison passes viability fractions but exposed exactly 6,207 temporarily zeroed subtree radii (12,414 VTP points) and the resulting tissue-mask difference. D-041 records the source fix; rerun from an exact fresh wheel before closing VAL-07.
- CASCADE full-grid healthy wall time was below the equivalent explicit-point legacy run despite performing five tissue nodes. All runs remained far below the 45 GiB RSS ceiling.
## 2026-09-10 — HEART-S closure and HEART-L production attribution

### Scope

VAL-06, VAL-07, PERF-05, PERF-06, and PERF-07.

### Work performed

- Accepted the explicit sparse GPU boundary comparison policy after the exact `987c97e` HEART-S occlusion rerun preserved anatomy and finite masks.
- Fixed HEART-L float32 connectivity corruption by streaming topology/node columns into exact int32 side arrays and routing analysis solvers through them.
- Added compact legacy-oracle output suppression and coordinate-aligned tissue comparison to the validation harness.
- Replaced the GL1-to-GL5 per-segment Python interpolation loop with vectorized constant interpolation.

### Verification and next gate

- Focused harness/loader tests pass.
- A compact 24,999,999-segment source run completed in 75.60 s versus 80.93 s legacy and used 27.60 versus 29.95 GiB peak RSS.
- The full 200-cubed-derived point campaign completed in 132.83 s versus 179.24 s legacy, used 33.76 versus 36.98 GiB peak RSS, and passed the D-042 field/fraction comparison.
- Evidence is in `../../validation/results/2026-09-10_m2-heart-s-occlusion-987c97e/`, `../../validation/results/2026-09-10_m23-heart-l-connectivity-source/`, and `../../validation/results/2026-09-10_m23-heart-l-full-vectorized/`.
- Final exact-wheel repetition remains before closing M2/M3.

## 2026-09-10 — RC2 exact-wheel release closure; M2/M3 complete

### Scope

M1, VAL-05, VAL-06, VAL-07, PERF-05, PERF-06, and PERF-07.

### Work performed

- Repeated HEART-L from the exact installed `af001fb` wheel rather than the editable source. The 24,999,999-segment, 1,619,996-input-point GPU workflow passed D-042 in 141.37 seconds application time and 145.21 seconds monitored wall time, versus 179.24/182.74 seconds for the isolated corrected-GL5 oracle. Peak RSS was 33.83 versus 36.98 GiB.
- Prepared CASCADE `0.1.0rc2`, then retained the first CPU smoke preflight failure: the shared lock still pinned `cascade-vascular==0.1.0rc1`. Commit `129c6a2` updates that lock, after which the artifacts were rebuilt.
- Installed the rebuilt exact wheel into independent temporary CPU and CUDA 13 environments outside the checkout. Both complete release smokes passed. CUDA diagnostics executed a compiled CuPy operation and cuFFT on the RTX 3080 Laptop GPU; the GPU smoke also completed the installed-wheel shared-global Cext heart workflow and reopened its VTK output.
- Reran the source suite: 133 passed with 15 already classified warnings.
- Audited the final archives: 77 wheel entries and 151 sdist entries, packaged bivent3 present, and zero forbidden-path/content hits. Both artifacts passed `twine check` and all four RC1/RC2 checksums in `dist/SHA256SUMS` verify.
- Added local annotated tag `v0.1.0rc2` at immutable source commit `129c6a2c60b7718a38085f0d3ef92b2184385aff` and copied the checksum-verified RC2 wheel/sdist beside the retained RC1 artifacts. No remote publication was attempted.
- Marked M2 and M3 complete for the existing-tree scope under D-045. Growth/optimizer qualification and all M4 governance/publication work remain explicitly separate.

### Exact artifacts and evidence

- Wheel: SHA-256 `a0a3f2f7d56fc29d1b161242879677d52bd696ccd9c3f07f128ab07547ae9ba7`, 901,059 bytes.
- Sdist: SHA-256 `8ae23fd8405303bcdf15305ed61eec60fa0979e27093bbeeace5d23c1a08665f`, 923,695 bytes.
- Release closure: `../../validation/results/2026-09-10_rc2-release_129c6a2/`.
- Exact-wheel HEART-L: `../../validation/results/2026-09-10_m23-heart-l-wheel-af001fb/`.
- Cube scale/performance: `../../validation/results/2026-09-10_m3-gpu-reference-scale-a20b6f9/` and `../../validation/results/2026-09-10_m3-gpu-reference-large-a20b6f9/`.

## 2026-09-10 — Post-M3 CLI/package hardening audit

### Scope

M1/RC-03 through RC-12, UX-02, GOV-01, and preservation of the completed M2/M3 conclusions.

### Work performed

- Audited high-signal static failures, every importable package module, CLI defaults, installed entry points, source/sdist/wheel contents, the public-SVV boundary, and the loaded-tree/forest lifecycle.
- Fixed undefined branches in the retained bifurcation, CPU Cext, GPU treecode, and linear tissue code; directly exercised treecode with both `decoupled_greens` and `zero` initialization on CUDA.
- Fixed forest connection restoration and analysis-only tree load/solve/save behavior, with regression tests.
- Changed generated settings to an immediately runnable CPU profile and added an installed `cascade self-test` CPU/GPU workflow.
- Removed the production heart exporter's arbitrary external-module hook and documented the process-isolated oracle boundary under D-046.
- Expanded CI and `scripts/release_smoke.py` to cover static checks, `pip check`, every console entry point, literal starter execution, installed self-test, public-SVV file integrity, heart CPU/CUDA workflows, manifests, and VTK reopening.

### Verification and disposition

- Source suite: 135 passed with 15 previously classified warnings; high-signal Ruff and compileall pass.
- All 69 non-executable package modules import successfully.
- Pre-freeze RC3 wheel CPU and CUDA release smokes passed outside the checkout, including compiled CuPy/cuFFT checks and real GPU Cext/tissue self-test.
- M2/M3 results remain valid because no accepted scientific settings, comparison rules, canonical structures, or certified performance algorithms were changed. Final immutable artifact hashes and archive audit are retained in the RC3 compact evidence directory.

### Immutable RC3 closure

- Tagged source commit `a1b0cf8848c95ad85cb39f2ab3b87a872067ae9a` as `v0.1.0rc3`.
- Exact wheel: 904,370 bytes, 78 entries, SHA-256 `6c7411f2b3e7dca4b4d426472c0292eb6b42d872da95dcd285e6aaf25d2dab1c`.
- Exact sdist: 929,906 bytes, 153 entries, SHA-256 `839d5b8f84dcbc717678aea198a33fea2ae0eecc510fbde82ff7bd7f7f36b2fc`.
- Both artifacts passed `twine check`; all entries in `dist/SHA256SUMS` verified. Archive inspection found no private/user paths, external oracle paths, generated validation output, or arbitrary external-module loader.
- Independent exact-wheel CPU and CUDA 13 smokes passed. The public `svv==0.0.48` distribution hash remained `c8767e211f732e8e38c20085989140123c3d2370531c349d4acb3d0717982660` before and after each smoke.
- Compact evidence: `../../validation/results/2026-09-10_rc3-cli-audit_a1b0cf8/`. Raw final transcripts remain ignored under `../../validation/tmp/rc3-final-{cpu,gpu}-smoke.log`.
