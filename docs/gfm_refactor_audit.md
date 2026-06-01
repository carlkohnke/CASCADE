# GFM Public-SVV Refactor Audit

Date: 2026-06-01

## Commands Run

```bash
/home/carl/miniconda3/envs/svva2/bin/python tests/gfm_refactor_audit.py --targets 1 100 --sample-count 1000 --timeout 3600
/home/carl/miniconda3/envs/svva2/bin/python tests/gfm_refactor_audit.py --targets 1 100 --sample-count 1000 --timeout 3600 --skip-cext
/home/carl/miniconda3/envs/svva2/bin/python tests/gfm_refactor_audit.py --work-dir .gfm_audit_runs_1000 --targets 1000 --sample-count 1000 --timeout 3600 --skip-cext
PIP_CACHE_DIR=/home/carl/GFM/.pip_cache_public_svv /home/carl/miniconda3/envs/svva2/bin/python -m pip install --target /home/carl/GFM/.tmp_public_svv svv==0.0.48
PYTHONPATH=/home/carl/GFM/.tmp_public_svv:/home/carl/GFM /home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_tree_smoke.json
PYTHONPATH=/home/carl/GFM/.tmp_public_svv:/home/carl/GFM /home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_forest_nearest_smoke.json
/home/carl/miniconda3/envs/svva2/bin/python /home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/export_paraview_heart_forest_grid_cext_gfm.py --forest /home/carl/GFM/.gfm_audit_runs/forest_smoke/cli/forest.forest --domain /home/carl/.cache/svv/heart_domain_35283d011b896a6e.dmn --out-dir /home/carl/GFM/.gfm_audit_runs/grid_export_legacy/out_heart_domain --prefix legacy_grid_heart_domain --side-length 1.0 --fluid blood --no-cext --concentration-solver topdown --finite-radius-o2-terms none --lumen-wall-closure wellmixed --flow-source total-qin-split --total-qin-ul-min 900 --nx 8 --ny 8 --nz 8 --tissue-accel cpu --tissue-gpu-validate-points 0 --vessel-resolution 2 --export-float-dtype float64 --export-index-dtype int64
SVV_DOMAIN_CACHE_DIR=/home/carl/GFM/.gfm_audit_runs/heart_forest_cache /home/carl/miniconda3/envs/svva2/bin/python /home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/TissueSim_heart_forest.py --forest-dir /home/carl/GFM/.gfm_audit_runs/forest_smoke/cli --seed-forest forest.forest --out-forest /home/carl/GFM/.gfm_audit_runs/heart_forest_smoke/out.forest --target-total-terminals 0 --target-per-tree 101,101 --domain-stl /mnt/c/Users/carl/svVascularize/svVascularize/meshes/tight_cube_outer.stl --side-length 1.0 --fluid blood --n-closest-vessels 2 --n-points 7 --n-equal-bifurcations -1 --growth-assignment scheduled --allow-scheduled-growth true --bulk-growth-mode never --progress false --strict-domain-segments false --dry-run true --save-target-counts ''
```

Detailed generated reports:

- `.gfm_audit_runs/audit_report.md`
- `.gfm_audit_runs_1000/audit_report.md`

## Passes

- Cube tree parity against `TissueSim_cube_local.py` is exact for the tested deterministic cases.
- `target_terminal_count=1`: all compared summary metrics matched exactly; latest audited legacy 5.270 s, CLI 5.566 s.
- `target_terminal_count=100`: all compared summary metrics matched exactly; latest audited legacy 6.128 s, CLI 6.675 s.
- `target_terminal_count=1000`: all compared summary metrics matched exactly; latest audited legacy 13.471 s, CLI 13.877 s.
- Cext single-tree smoke matched exactly for the tested target-1 case; legacy 8.464 s, CLI 7.019 s.
- Equal-bifurcation geometry smoke produced the expected 201 segments for 100 requested terminals.
- Forest smoke produced `summary.csv`, `segments.csv`, `points.csv`, `vessels.vtp`, `oxygen_points.vtp`, `manifest.json`, `forest.forest`, and `forest.forest.simcache`.
- Forest cache reload reproduced the forest smoke summary rows, 402 segment rows, and all compared per-tree summary metrics exactly.
- Direct `.forest.simcache` input also reproduced the same summary rows, 402 segment rows, and all compared per-tree summary metrics exactly.
- Regular grid export is now exposed through `simulation.sample_mode="grid"` with `simulation.tissue_grid.nx/ny/nz`; the audit smoke used `8x8x8`, wrote CSV/VTP outputs, and recorded grid metadata in `manifest.json`.
- Simple channel geometries are now exposed through `network.mode="simple"` and `network.simple`. The audit smoke ran a one-channel rectangular box, solved flow/concentration/tissue oxygen, saved `simple.simple.npz`, and wrote CSV/VTP outputs. The implementation also includes multichannel and snake geometry builders.
- Scheduled forest growth is now exposed through `growth.assignment="scheduled"` with checkpoint and target-save controls. The audit smoke extended two 100-terminal trees to 102/102, wrote checkpoint files, wrote target saves, and exported 410 segment rows.
- Nearest-tree forest growth is now exposed through `growth.assignment="nearest-tree"`. The audit smoke used fixed-point nearest assignment, inter-tree collision rejection, connectivity validation, and exported the expected 18 segment rows.
- Connectivity repair/validation is now available through `network.repair_connectivity`, `network.validate_connectivity`, `network.fail_connectivity`, and `network.connectivity_geometry_atol`.
- Flow-source aliases now include `tree-root-flow` and `total-qin-split`, with total-qin splitting weighted by loaded root flows where available.
- Infarction-style solve overrides are now available through `simulation.infarction.fraction_blocked` and `simulation.infarction.global_segment_id`.
- Export array dtype controls now accept `outputs.export_float_dtype` of `float32` or `float64`.
- Public `svv==0.0.48` temporary install under GFM imported correctly and ran the standard tree and nearest-tree forest smoke settings.
- The legacy Cext grid exporter can load the new CLI forest through its simulation cache and write the four ParaView-style outputs for a small `8x8x8` grid when given an installed-svv-readable `.dmn` domain.
- `TissueSim_heart_forest.py` can load the new CLI forest and complete dry-run planning with its domain cache redirected under GFM.

## Not Yet Equivalent

- The new CLI is not a full replacement for `TissueSim_heart_forest.py`.
- Remaining heart forest gaps are mostly high-scale controls: shared forest HNSW nearest index controls, the exact legacy candidate queue semantics, domain-only validation mode, full seed-forest metadata export, and float32 transition controls.
- The new CLI forest mode now supports scheduled growth, nearest-tree fixed-point growth, optional inter-tree collision-aware CCO, collision fallback to bulk mode, strict proposed-segment domain rejection, checkpoints, target saves, root-flow-weighted target/flow splitting, and analysis cache loading.
- The new CLI is not a full replacement for `export_paraview_heart_forest_grid_cext_gfm.py`.
- Missing grid/export controls include shared-global Cext forest modes, explicit backend-per-tree Cext mode selection, full exporter connectivity reporting format, and legacy prefix-named output files.
- New CLI tissue points can now be random samples or a regular grid. Grid export currently uses the same TissueSim concentration/tissue solve path as the normal CLI, not the full shared-global Cext exporter path.
- Simple geometry mode is functional for one-channel, multichannel, and snake channels, but it does not yet reproduce all plotting/profile image outputs from `single_channel_box_greens.py`.
- V1 remains float64-only by design. Public-svv float32 parity is not implemented.

## Bugs Or Risks Found

- A GFM-created cube `.dmn` written by the current adapter was not readable by the legacy grid exporter path using installed `svv.domain.domain.Domain.load`; it failed with `Unsupported .dmn format or version.` This matters if `.dmn` interchange is expected between the GFM adapter and old scripts.
- Direct `.forest.simcache` input originally failed because the loader only recognized `.forest` paths. This was fixed in `gfm.growth._load_existing_network` and is covered by `forest_direct_simcache`.
- The public `svv==0.0.48` check was smoke-level only. Full 100/1000 parity was run in the active env, which currently reports `svv==0.0.43`.
- The Cext parity check was limited to a very small single-tree case. Multi-tree shared Cext is still legacy-exporter-only.
- Nearest-tree assignment currently uses exact GFM-side point-to-segment distance for the small/medium target range, not the old shared `usearch` forest index. That is simpler and deterministic, but it is not the final high-scale strategy for million-vessel forests.

## Recommended Next Fixes

1. Decide whether `.dmn` is part of the supported public interface. If yes, align the GFM domain writer/reader with installed public `svv`.
2. Add the high-scale shared forest nearest index path behind the current nearest-tree interface.
3. Port or redesign shared-global forest Cext behind `simulation.cext.forest_mode`.
4. Add a public-`svv==0.0.48` CI-style audit target for the 100 and 1000 terminal parity cases.
