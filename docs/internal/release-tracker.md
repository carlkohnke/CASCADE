# CASCADE production and validation tracker

Last updated: 2026-09-09

Current release baseline: `v0.1.0rc1` at commit `9da8b2e`

Current objective: certify CASCADE against the selected legacy TissueSim workflows for numerical equivalence and equivalent-or-better performance, while keeping public-`svv` compatibility changes in CASCADE whenever possible.

## Milestones

| Milestone | Status | Exit criterion | Evidence / next action |
| --- | --- | --- | --- |
| M0 — Legacy inventory and initial parity audit | COMPLETE | Relevant scripts, inputs, outputs, assumptions, and gaps identified | `m0-legacy-inventory.md`, `m0-traceability.md`, and `../../validation/results/2026-09-09_m2-staging_20744c8/`; M2 starts from frozen inputs and rules |
| M1 — Reproducible release candidate | COMPLETE | Git baseline, installable CLI/GUI, pinned CPU/GPU environments, artifacts, basic workflows, and release report | `v0.1.0rc1`; `../release-0.1.0rc1.md` |
| M2 — Numerical and functional certification | IN PROGRESS | All agreed reference cases pass structural and numerical acceptance criteria on fixed inputs | CUBE-1 blood and water/cell-media summary preflight passes; continue the existing-tree correctness matrix before performance |
| M3 — Performance certification | PLANNED | CASCADE is not slower than the agreed `TissueSim_cube_local` baseline under the approved protocol, or every regression is resolved/accepted | Freeze benchmark protocol and execute interleaved repeated runs |
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
| RC-05 | Validate CPU-only installation outside checkout | COMPLETE | Final wheel smoke passed; `../release-0.1.0rc1.md` | Repeat for every artifact build |
| RC-06 | Validate CUDA 13 installation outside checkout | COMPLETE | Kernel, cuFFT, and heart Cext smoke passed on RTX 3080 Laptop GPU | CUDA 11/12 remain unvalidated |
| RC-07 | Support custom explicit CSV/NPZ vascular geometry | COMPLETE | `../../examples/custom_y_channel.csv`; release smoke | Expand malformed-input tests during M2 |
| RC-08 | Support custom/file-backed domains | COMPLETE for supported contract | VTP/STL and built-in domains operational; packaged `bivent3.stl` added | `.dmn` remains legacy/internal validation input, not public interchange |
| RC-09 | Export ParaView-compatible vascular/domain/tissue results | COMPLETE | VTP/VTU files reopened with PyVista | Validate field schema/value parity under VAL-06 |
| RC-10 | Package advanced heart forest exporter and Cext runtime | COMPLETE for operational gate | CPU and CUDA small solves passed | Numerical parity and representative-scale performance remain M2/M3 |
| RC-11 | Validate CASCADE Studio startup | COMPLETE for basic gate | Offscreen construct/show/event-loop/exit passed | Full human queue/cancel/result workflow remains UX-01 |
| RC-12 | Produce clean wheel/sdist and checksums | COMPLETE | `../../dist/`; `twine check`; artifact-content audit | Rebuild/check for each release tag |
| RC-13 | Configure GitHub remote and verify CI | BLOCKED | Local branch/tag exist; no remote configured | Obtain repository URL, add remote, push branch/tag, observe CI |
| RC-14 | Select project license | BLOCKED | No `LICENSE` exists | Owner/legal decision; confirm compatibility with derived `svv` code |

## Step 2: numerical and functional certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAL-01 | Freeze canonical reference inputs | COMPLETE | Versioned settings plus hashes for geometry, domain, forest/tree, seeds, and tissue samples | One-million cube coordinates hash `251ca61e...`; heart axes/grid contract and HEART-S major-branch occlusion ID 1 in the M2 staging evidence |
| VAL-02 | Freeze comparison fields and tolerances | COMPLETE | Written exact/tolerant comparison rules for geometry, flow, pressure, concentration, and tissue oxygen | D-023 and `test-plan.md`; refine only if a recorded scientific reason emerges during M2 |
| VAL-03 | Cube existing-tree structural identity | PLANNED | Both sides load the exact hashed topology, segment ordering/mapping, endpoints, radii, and terminal counts | Frozen family through 5M; no growth in this campaign |
| VAL-04 | Cube tree numerical parity | IN PROGRESS | All selected scalar/array metrics meet VAL-02 | Fresh-wheel CUBE-1 blood and water/cell-media summary preflight passed on the complete shared 1M-point pool; detailed arrays and remaining scales are not yet certified |
| VAL-05 | Forest growth and reload parity | PARTIAL | Deterministic scheduled/nearest assignment and cache reload meet structural/numerical rules | Existing smoke evidence; add canonical public-`svv` cases |
| VAL-06 | VTK/export parity | PLANNED | Expected files, schemas, dtypes, IDs, units, point counts, and field values match approved baseline | Compare normal and heart exporters field-by-field |
| VAL-07 | Heart shared-global Cext parity | PARTIAL | CASCADE and legacy exporter agree for fixed multi-tree forest and tissue grid | Operational CUDA smoke passed; representative numerical comparison not run |
| VAL-08 | `.dmn` interchange decision and round trip | COMPLETE | `.dmn` is explicitly legacy/internal validation input; STL/VTP/VTU are the supported public interchange path | D-013; `m0-legacy-inventory.md`; retain load compatibility without claiming cross-version round trips |
| VAL-09 | Custom-domain/custom-geometry correctness | PARTIAL | CSV/NPZ plus VTP/STL domain workflows pass topology, unit, solver, and export checks | Basic Y-channel case passed; broaden fixtures and negative tests |
| VAL-10 | CPU/GPU numerical consistency | PLANNED | Same frozen case meets VAL-02 across CPU and CUDA paths | Preserve identical sample points and reduction settings |
| VAL-11 | Failure-mode and input-validation tests | PARTIAL | Invalid settings, missing files, malformed geometry, unavailable GPU, and incompatible caches fail clearly | Strict settings and basic failures covered; build adversarial matrix |
| VAL-12 | Reproducibility rerun | PLANNED | Repeated fixed-seed run produces accepted identical/tolerant outputs and manifests | Run from two clean environments and compare hashes/arrays |

## Step 2: performance certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| PERF-01 | Freeze benchmark protocol | COMPLETE | Same process isolation, geometry, arrays, settings, hardware state, warmups, run count, and timing boundaries | D-023: five pairs through 100k, three at 1M, one initial at 5M/HEART-L with a recorded noise-triggered repeat rule |
| PERF-02 | Cube scale matrix | PARTIAL | Compare every available decade and the highest cached target | Historical timings exist for 1/100/1000; frozen matrix is 1, 10, 100, 1k, 10k, 100k, 1M, and 5M; missing 10M is an input gap, not a build request |
| PERF-03 | Tissue oxygen CPU benchmark | PLANNED | Component and end-to-end timings versus `TissueSim_cube_local` | Use shared samples and cached geometry separately from end-to-end test |
| PERF-04 | Tissue oxygen CUDA benchmark | PLANNED | Kernel/component and end-to-end timings with synchronization | Include first-run compile separately from warmed execution |
| PERF-05 | Heart forest/export benchmark | PLANNED | Load/cache, flow, Cext, tissue, and VTK timings at representative scale | Frozen HEART-S/HEART-L and 200-cubed grid; stage commands after bounded GPU/memory preflight |
| PERF-06 | Profile regressions | PLANNED | Attribute any statistically meaningful slowdown and implement or disposition fixes | Use profiler only after equivalent work is confirmed |
| PERF-07 | Equivalent-or-faster release gate | PLANNED | CASCADE meets agreed performance criterion against `TissueSim_cube_local` | No performance claim until PERF-01 through PERF-06 are complete |

## Usability, CI, and governance

| ID | Work item | Status | Evidence / next action |
| --- | --- | --- | --- |
| UX-01 | Human GUI workflow: create project, preview, queue, cancel, resume, load results | PLANNED | Record manual checklist and screenshots/logs on supported display environment |
| UX-02 | Human CLI workflow from README on clean clone | PARTIAL | Wheel automation passed; repeat literally from a GitHub clone after remote exists |
| CI-01 | CPU GitHub Actions | PARTIAL | Workflow is implemented at `../../.github/workflows/ci.yml`; requires remote run |
| CI-02 | GPU CI strategy | PLANNED | Decide self-hosted runner versus documented workstation qualification |
| GOV-01 | CASCADE-local compatibility review | IN PROGRESS | Review each incompatibility using `../svv-compatibility.md` policy |
| GOV-02 | Upstream issue package | PLANNED | Only inner-loop growth/optimization or shared-contract items, with minimal reproducers |
| GOV-03 | Supported platform statement | PARTIAL | Linux/WSL2 x86-64 Python 3.9 and CUDA 13 validated; broader support not yet claimed |

## Immediate next actions

1. Extend the isolated cube runner from summary preflight to exact topology/ID and pointwise/streaming field comparisons, then execute 10 through 5M in increasing order.
2. Stage HEART-S healthy and full-occlusion ID 1 commands against the full 200-cubed grid; run only after bounded GPU/memory preflight.
3. Run correctness before performance and only one solver subprocess at a time; use compact outputs for 1M/5M/HEART-L and bounded cases for full export parity.
4. Save every compact result under `validation/results/` and update this tracker plus `work-log.md` in the same commit.
5. Obtain the GitHub URL and license decision when public publication becomes the active milestone.
