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
