# CASCADE production and validation tracker

Last updated: 2026-09-09

Current release baseline: `v0.1.0rc1` at commit `9da8b2e`

Current objective: certify CASCADE against the selected legacy TissueSim workflows for numerical equivalence and equivalent-or-better performance, while keeping public-`svv` compatibility changes in CASCADE whenever possible.

## Milestones

| Milestone | Status | Exit criterion | Evidence / next action |
| --- | --- | --- | --- |
| M0 — Legacy inventory and initial parity audit | PARTIAL | Relevant scripts, inputs, outputs, assumptions, and gaps identified | `../cascade_refactor_audit.md`; refresh against public `svv==0.0.48` during M2 |
| M1 — Reproducible release candidate | COMPLETE | Git baseline, installable CLI/GUI, pinned CPU/GPU environments, artifacts, basic workflows, and release report | `v0.1.0rc1`; `../release-0.1.0rc1.md` |
| M2 — Numerical and functional certification | PLANNED | All agreed reference cases pass structural and numerical acceptance criteria on fixed inputs | Execute `test-plan.md`; save reports under `validation/results/` |
| M3 — Performance certification | PLANNED | CASCADE is not slower than the agreed `TissueSim_cube_local` baseline under the approved protocol, or every regression is resolved/accepted | Freeze benchmark protocol and execute interleaved repeated runs |
| M4 — Public-release readiness | BLOCKED | License selected, remote/CI operational, supported-platform statement finalized, M2/M3 disposition recorded | Owner license decision and GitHub repository URL required |

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
| RC-08 | Support custom/file-backed domains | PARTIAL | VTP/STL and built-in domains operational | Decide and test `.dmn` interchange under VAL-08 |
| RC-09 | Export ParaView-compatible vascular/domain/tissue results | COMPLETE | VTP/VTU files reopened with PyVista | Validate field schema/value parity under VAL-06 |
| RC-10 | Package advanced heart forest exporter and Cext runtime | COMPLETE for operational gate | CPU and CUDA small solves passed | Numerical parity and representative-scale performance remain M2/M3 |
| RC-11 | Validate CASCADE Studio startup | COMPLETE for basic gate | Offscreen construct/show/event-loop/exit passed | Full human queue/cancel/result workflow remains UX-01 |
| RC-12 | Produce clean wheel/sdist and checksums | COMPLETE | `../../dist/`; `twine check`; artifact-content audit | Rebuild/check for each release tag |
| RC-13 | Configure GitHub remote and verify CI | BLOCKED | Local branch/tag exist; no remote configured | Obtain repository URL, add remote, push branch/tag, observe CI |
| RC-14 | Select project license | BLOCKED | No `LICENSE` exists | Owner/legal decision; confirm compatibility with derived `svv` code |

## Step 2: numerical and functional certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| VAL-01 | Freeze canonical reference inputs | PLANNED | Versioned settings plus hashes for geometry, domain, forest/tree, seeds, and tissue samples | Select representative cube, heart, and custom-domain fixtures |
| VAL-02 | Freeze comparison fields and tolerances | BLOCKED | Written exact/tolerant comparison rules for geometry, flow, pressure, concentration, and tissue oxygen | Scientific tolerance decision required before certification |
| VAL-03 | Cube tree structural parity | PARTIAL | Same topology, segment ordering/mapping, endpoints, radii, and terminal counts | Historical 1/100/1000 aggregate parity exists; rerun full structure checks with public `svv==0.0.48` |
| VAL-04 | Cube tree numerical parity | PARTIAL | All selected scalar/array metrics meet VAL-02 | Historical aggregate summaries matched for 1/100/1000 but active environment then used `svv==0.0.43`; not current certification |
| VAL-05 | Forest growth and reload parity | PARTIAL | Deterministic scheduled/nearest assignment and cache reload meet structural/numerical rules | Existing smoke evidence; add canonical public-`svv` cases |
| VAL-06 | VTK/export parity | PLANNED | Expected files, schemas, dtypes, IDs, units, point counts, and field values match approved baseline | Compare normal and heart exporters field-by-field |
| VAL-07 | Heart shared-global Cext parity | PARTIAL | CASCADE and legacy exporter agree for fixed multi-tree forest and tissue grid | Operational CUDA smoke passed; representative numerical comparison not run |
| VAL-08 | `.dmn` interchange decision and round trip | BLOCKED | Either supported and tested bidirectionally, or explicitly unsupported with migration path | Decide whether public `.dmn` is a required contract |
| VAL-09 | Custom-domain/custom-geometry correctness | PARTIAL | CSV/NPZ plus VTP/STL domain workflows pass topology, unit, solver, and export checks | Basic Y-channel case passed; broaden fixtures and negative tests |
| VAL-10 | CPU/GPU numerical consistency | PLANNED | Same frozen case meets VAL-02 across CPU and CUDA paths | Preserve identical sample points and reduction settings |
| VAL-11 | Failure-mode and input-validation tests | PARTIAL | Invalid settings, missing files, malformed geometry, unavailable GPU, and incompatible caches fail clearly | Strict settings and basic failures covered; build adversarial matrix |
| VAL-12 | Reproducibility rerun | PLANNED | Repeated fixed-seed run produces accepted identical/tolerant outputs and manifests | Run from two clean environments and compare hashes/arrays |

## Step 2: performance certification

| ID | Work item | Status | Required result | Evidence / next action |
| --- | --- | --- | --- | --- |
| PERF-01 | Freeze benchmark protocol | PLANNED | Same process isolation, geometry, arrays, settings, hardware state, warmups, run count, and timing boundaries | Approve protocol in `test-plan.md` before measuring |
| PERF-02 | Cube scale matrix | PARTIAL | Compare agreed small/medium/large target counts | Historical timings exist for 1/100/1000; rerun using current package/public `svv` and repeated interleaving |
| PERF-03 | Tissue oxygen CPU benchmark | PLANNED | Component and end-to-end timings versus `TissueSim_cube_local` | Use shared samples and cached geometry separately from end-to-end test |
| PERF-04 | Tissue oxygen CUDA benchmark | PLANNED | Kernel/component and end-to-end timings with synchronization | Include first-run compile separately from warmed execution |
| PERF-05 | Heart forest/export benchmark | PLANNED | Load/cache, flow, Cext, tissue, and VTK timings at representative scale | Select forest size and grid sizes that fit production use |
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

1. Receive the Step 2 scientific scope and select canonical fixtures.
2. Agree numerical fields and tolerances (VAL-02).
3. Freeze the performance protocol and representative scales (PERF-01).
4. Run correctness before speed; do not optimize a path that is not performing equivalent work.
5. Save every compact result under `validation/results/` and update this tracker plus `work-log.md` in the same commit.
6. Obtain the GitHub URL and license decision when public publication becomes the active milestone.
