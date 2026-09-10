# CASCADE production and validation tracker

Last updated: 2026-09-10

Current release baseline: `v0.1.0rc3` at commit `a1b0cf8`; the exact CPU/CUDA wheel qualification and archive audit passed.

Current objective: preserve the completed M0-M3 existing-tree certification while preparing the governance, remote-CI, and human-acceptance work that remains in M4.

## Milestones

| Milestone | Status | Exit criterion | Evidence / next action |
| --- | --- | --- | --- |
| M0 — Legacy inventory and initial parity audit | COMPLETE | Relevant scripts, inputs, outputs, assumptions, and gaps identified | `m0-legacy-inventory.md`, `m0-traceability.md`, and `../../validation/results/2026-09-09_m2-staging_20744c8/`; completed M2 used the frozen inputs and rules |
| M1 — Reproducible release candidate | COMPLETE | Git baseline, installable CLI/GUI, pinned CPU/GPU environments, artifacts, basic workflows, and release report | `v0.1.0rc3` plus `../../validation/results/2026-09-10_rc3-cli-audit_a1b0cf8/` |
| M2 — Numerical and functional certification | COMPLETE for approved existing-tree scope | All agreed reference cases pass structural and numerical acceptance criteria on fixed inputs | Cube matrix, custom inputs, negative/reproducibility, HEART-S healthy/occlusion, and exact-wheel HEART-L pass; scientific growth qualification is explicitly deferred by D-023 |
| M3 — Performance certification | COMPLETE | CASCADE is not slower than the agreed `TissueSim_cube_local` baseline under the approved protocol, or every regression is resolved/accepted | Warm small GPU solve `0.123-0.148 s`; 10,000,001-vessel compute ratio `0.947`; exact-wheel HEART-L application-time ratio `0.789` with lower peak RSS |
| M4 — Public-release readiness | BLOCKED | License selected, remote/CI operational, supported-platform statement finalized, M2/M3 disposition recorded | Owner license decision and GitHub repository URL required |

## Step 0: frozen legacy baseline

| ID | Work item | Status | Evidence / next action |
| --- | --- | --- | --- |
| M0-01 | Freeze legacy oracle scripts and direct supporting modules | COMPLETE | Exact external paths, sizes, versions, and SHA-256 hashes in `m0-legacy-inventory.md`; tracked duplicate scripts removed from the CASCADE root |
| M0-02 | Establish isolated oracle/CASCADE execution boundary | COMPLETE | D-011; wheel-installed CASCADE and `svva2/SCRIPTS` run in separate environments with no imports or automatic cache relocation across the boundary; obsolete same-process helpers removed |
| M0-03 | Map legacy capabilities to CASCADE | COMPLETE | `m0-traceability.md` records command/settings, implementation, case, and disposition |
| M0-04 | Freeze canonical numerical profile | COMPLETE | D-012; float32, grid 256, Cext GL1, tissue GL5, one Cext iteration, window 6; defaults and contract tests updated |
| M0-05 | Freeze heart inputs | COMPLETE | HEART-S forest, HEART-L forest/simcache, bivent3, units, sizes, and hashes are frozen in `m0-legacy-inventory.md` |
| M0-06 | Freeze cube validation scale | COMPLETE | Current-default compatible family frozen at 1 through 1M plus highest available 5M; absent 10M reported without building |
| M0-07 | Map outputs and comparison fields | COMPLETE | Representation map plus D-023: exact identity fields, `1e-3` relative physical-field rule, `1e-6` case-scale near-zero floor, and `0.001` absolute fraction rule |
| M0-08 | Normalize historical audit evidence | COMPLETE | `../../validation/results/2026-09-09_historical-audit_normalized/` preserves measurements, commands, raw locations, and limitations without promoting them to current certification |
| M0-09 | Classify legacy scope and upstream boundaries | COMPLETE | Traceability matrix, D-013 through D-016, and `../svv-compatibility.md` |
| M0-10 | Enforce bounded sequential execution | COMPLETE | Per-user CLI simulation lock, sequential Studio workers, sweep cleanup, compact analysis-only loaders, streamed float32 simcache conversion, and contract tests; approximately 50 GB host budget documented |

## Step 1: release-candidate work

| ID | Work item | Status | Evidence | Follow-up |
| --- | --- | --- | --- | --- |
| RC-01 | Preserve pre-CASCADE repository state in Git | COMPLETE | Commit `3c040c8` | None |
| RC-02 | Rename public package/CLI/GUI branding from GFM to CASCADE while retaining scientific GFM terminology | COMPLETE | Commit `eec412a`; `../../src/cascade/`; `../../pyproject.toml` | Review new public text as it is added |
| RC-03 | Package CLI, GUI, viewer, diagnostics, and heart exporter | COMPLETE | Wheel entry points in `../../pyproject.toml`; artifact audit | Add new entry points only through packaging tests |
| RC-04 | Pin public `svv` and Python/dependency environments | COMPLETE | `svv==0.0.48`; `../../locks/`; Python 3.9.20 report | Re-resolve only as a deliberate release change |
| RC-05 | Validate CPU-only installation outside checkout | COMPLETE | Exact RC3 wheel smoke and installed self-test passed; `../../validation/results/2026-09-10_rc3-cli-audit_a1b0cf8/` | Repeat for every artifact build |
| RC-06 | Validate CUDA 13 installation outside checkout | COMPLETE | Exact RC3 wheel compiled CuPy, cuFFT, installed self-test, and heart Cext smoke passed on RTX 3080 Laptop GPU | CUDA 11/12 remain unvalidated |
| RC-07 | Support custom explicit CSV/NPZ vascular geometry | COMPLETE | `../../examples/custom_y_channel.csv`; release smoke and completed malformed-input matrix | Keep format regression coverage |
| RC-08 | Support custom/file-backed domains | COMPLETE for supported contract | VTP/STL and built-in domains operational; packaged `bivent3.stl` added | `.dmn` remains legacy/internal validation input, not public interchange |
| RC-09 | Export ParaView-compatible vascular/domain/tissue results | COMPLETE | VTP/VTU files reopened with PyVista | Validate field schema/value parity under VAL-06 |
| RC-10 | Package advanced heart forest exporter and Cext runtime | COMPLETE | CPU/CUDA smoke plus HEART-S/L M2/M3 certification passed | Keep exact-wheel and representative-scale regression coverage |
| RC-11 | Validate CASCADE Studio startup | COMPLETE for basic gate | Offscreen construct/show/event-loop/exit passed | Full human queue/cancel/result workflow remains UX-01 |
| RC-12 | Produce clean wheel/sdist and checksums | COMPLETE | RC3 wheel/sdist in `../../dist/`; `twine check`; 78/153-entry audit; SHA-256 verification in RC3 evidence | Rebuild/check for each release tag |
| RC-13 | Configure GitHub remote and verify CI | BLOCKED | Local branch/tag exist; no remote configured | Obtain repository URL, add remote, push branch/tag, observe CI |
| RC-14 | Select project license | BLOCKED | No `LICENSE` exists | Owner/legal decision; confirm compatibility with derived `svv` code |
| RC-15 | Complete post-M3 CLI/package audit | COMPLETE | Immutable tag `v0.1.0rc3` at `a1b0cf8`; 135-test suite, high-signal static/import audit, exact-wheel CPU/CUDA smokes, entry-point audit, public-SVV byte-integrity check, starter and installed self-test | Preserve the compact evidence and immutable artifacts |

## Step 2: numerical and functional certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAL-01 | Freeze canonical reference inputs | COMPLETE | Versioned settings plus hashes for geometry, domain, forest/tree, seeds, and tissue samples | One-million cube coordinates hash `251ca61e...`; heart axes/grid contract and HEART-S major-branch occlusion ID 1 in the M2 staging evidence |
| VAL-02 | Freeze comparison fields and tolerances | COMPLETE | Written exact/tolerant comparison rules for geometry, flow, pressure, concentration, and tissue oxygen | D-023, D-039, D-042, and `test-plan.md`; changes require a recorded scientific reason |
| VAL-03 | Cube existing-tree structural identity | COMPLETE | Both sides load the exact hashed topology, segment ordering/mapping, endpoints, radii, and terminal counts | `../../validation/results/2026-09-09_m23-characterization_9cb7712/cube-characterization.md`: 16 cases, 17 arrays each, zero exact mismatches through 10,000,001 segments |
| VAL-04 | Cube tree numerical parity | COMPLETE | All selected scalar/array metrics meet VAL-02 | Same report: zero tolerance failures in blood and water/cell-media at 1, 10, 100, 1k, 10k, 100k, 1M, and 5M terminals |
| VAL-05 | Forest reload/cache parity on existing structures | COMPLETE for approved scope | Existing forests and simulation caches reload with exact topology for computation; growth itself is not part of this campaign | Forest release smoke, HEART-S `.forest`, and HEART-L 24,999,999-segment simcache pass; scheduled/nearest growth qualification remains a later campaign under D-023/D-035 |
| VAL-06 | VTK/export parity | COMPLETE | Expected files, schemas, dtypes, IDs, units, point counts, and field values match approved baseline | Bounded CSV/NPZ/domain export passed 23 checks; HEART-S certifies complete vessel schema/values and HEART-L certifies full tissue/domain output with compact vessel output per D-023 |
| VAL-07 | Heart shared-global Cext parity | COMPLETE | CASCADE and legacy exporter agree for fixed multi-tree forest and tissue grid under the explicitly named quadrature contract | HEART-S healthy/occlusion and exact-wheel HEART-L pass production Cext GL1 followed by independent tissue GL5 under D-034/D-040/D-042 |
| VAL-08 | `.dmn` interchange decision and round trip | COMPLETE | `.dmn` is explicitly legacy/internal validation input; STL/VTP/VTU are the supported public interchange path | D-013; `m0-legacy-inventory.md`; retain load compatibility without claiming cross-version round trips |
| VAL-09 | Custom-domain/custom-geometry correctness | COMPLETE | CSV/NPZ plus VTP/STL domain workflows pass topology, unit, solver, and export checks | `../../validation/results/2026-09-09_m2-custom-export-fa2c958/`: all four installed-wheel cases passed outside checkout; CSV/NPZ outputs matched |
| VAL-10 | CPU/GPU numerical consistency | COMPLETE | Same frozen case meets VAL-02 across CPU and CUDA paths | `../../validation/results/2026-09-09_m2-cpu-gpu-fa2c958/`: CUBE-100 passed all 17 arrays and 23 summaries; max tissue relative error `2.46e-5` |
| VAL-11 | Failure-mode and input-validation tests | COMPLETE | Invalid settings, missing files, malformed geometry, unavailable GPU, and incompatible caches fail clearly | `../../validation/results/2026-09-09_m2-negative-128c6fc/`: exact committed wheel passed all 10 cases after the retained fa2c958 preflight exposed zero-grid coercion |
| VAL-12 | Reproducibility rerun | COMPLETE | Repeated fixed-seed run produces accepted identical/tolerant outputs and manifests | `../../validation/results/2026-09-09_m2-repro-128c6fc/`: two clean environments produced bit-identical 17 arrays/23 summaries and matching 90 stable manifest leaves |

## Step 2: performance certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| PERF-01 | Freeze benchmark protocol | COMPLETE | Same process isolation, geometry, arrays, settings, hardware state, warmups, run count, and timing boundaries | D-023/D-032/D-035/D-036: sequential alternating GPU pairs, cold compile separated, historical four-component algorithm boundary plus end-to-end reporting |
| PERF-02 | Cube scale matrix | COMPLETE | Compare every available decade and the highest cached target | Exact-wheel GPU scientific comparisons pass from 10 through 5M terminals; earlier exact-wheel target-1 and CPU 1/10/100 tranches pass; `../../validation/results/2026-09-10_m3-gpu-reference-scale-a20b6f9/` and `../../validation/results/2026-09-10_m3-gpu-reference-large-a20b6f9/` |
| PERF-03 | Tissue oxygen CPU benchmark | COMPLETE for approved scope | Component and end-to-end timings versus `TissueSim_cube_local` | Five pairs each at 1/10/100 pass all summaries; the interrupted 1k/10k CPU extension is retained but superseded by D-035 |
| PERF-04 | Tissue oxygen CUDA benchmark | COMPLETE | Kernel/component and end-to-end timings with synchronization | All GPU cube summaries pass; warmed target-10 solve wall is 0.123-0.148 s; at 10,000,001 vessels median comparable compute is 26.16 s CASCADE versus 27.65 s current oracle (ratio 0.947); historical June blood was 23.19 s |
| PERF-05 | Heart forest/export benchmark | COMPLETE | GPU-only load/cache, flow, Cext, tissue, and VTK timings at representative scale | Exact-wheel HEART-L: CASCADE `141.37 s` application/`145.21 s` monitored versus oracle `179.24/182.74 s`; peak RSS `33.83` versus `36.98 GiB` |
| PERF-06 | Profile regressions | COMPLETE | Attribute any statistically meaningful slowdown and implement or disposition fixes | D-037 compile/allocator reuse, D-038 domain cache, D-043 exact integer topology, and D-044 vectorized GL1-to-GL5 transition remove all observed release-blocking regressions |
| PERF-07 | Equivalent-or-faster release gate | COMPLETE | CASCADE meets agreed performance criterion against `TissueSim_cube_local` | All equivalent-work GPU cube pairs pass scientifically and CASCADE is faster at every scale; HEART-L ratio `0.789`; compact evidence linked above and in the RC2 closure report |

## Usability, CI, and governance

| ID | Work item | Status | Evidence / next action |
| --- | --- | --- | --- |
| UX-01 | Human GUI workflow: create project, preview, queue, cancel, resume, load results | PLANNED | Record manual checklist and screenshots/logs on supported display environment |
| UX-02 | Human CLI workflow from README on clean clone | COMPLETE locally / remote pending | Exact-wheel smoke literally runs `init-settings` then `run`; `self-test` works without checkout data; repeat from GitHub only after remote exists |
| CI-01 | CPU GitHub Actions | PARTIAL | Workflow is implemented at `../../.github/workflows/ci.yml`; requires remote run |
| CI-02 | GPU CI strategy | PLANNED | Decide self-hosted runner versus documented workstation qualification |
| GOV-01 | CASCADE-local compatibility review | COMPLETE for RC3 | Public `svv==0.0.48` remains unmodified on disk; adapter-only in-memory hooks documented/tested; production external-module hook removed |
| GOV-02 | Upstream issue package | PLANNED | Only inner-loop growth/optimization or shared-contract items, with minimal reproducers |
| GOV-03 | Supported platform statement | PARTIAL | Linux/WSL2 x86-64 Python 3.9 and CUDA 13 validated; broader support not yet claimed |

## Immediate next actions

1. Retain the immutable RC2 and RC3 tags and checksum-verified artifacts; any packaged-source change requires a new build and exact-wheel qualification.
2. Treat tree-growth/optimizer qualification as a separate post-M3 scientific campaign: retain float64 public-SVV CCO, the float32 equal-bifurcation transition, and the planned upstream SLSQP/L-BFGS-B selector.
3. When M4 is activated, select a license, configure the GitHub remote, push the branch/tag intentionally, observe CPU CI, and finalize the supported-platform statement.
4. Complete human CASCADE Studio queue/cancel/recovery/viewer acceptance before a public production claim.
