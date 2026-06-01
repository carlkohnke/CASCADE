# GFM

GFM is a command-line simulation and export layer for vascular growth and tissue oxygen analysis. It wraps public `svv` with GFM-owned compatibility code so workflows that previously depended on locally edited `svv` files can run from this repository instead of from `site-packages/svv/SCRIPTS`.

The current implementation keeps the numerical behavior of `TissueSim_cube_local.py` as the source of truth, but exposes it through structured JSON settings and a package layout:

```bash
python -m gfm.cli run --settings settings.json
python -m gfm.cli sweep --settings sweep_settings.json
```

The goal is to support reproducible tree, forest, simple-channel, simulation, cache, CSV, and ParaView workflows while avoiding edits outside this repo.

## Capabilities

### Network Types

GFM supports three network modes:

- `tree`: one vascular tree grown from a configured root.
- `forest`: multiple trees grown from configured roots, with per-tree or total target counts.
- `simple`: idealized channel geometries for microfluidic-style analysis.

Tree and forest workflows can:

- Grow new networks in cube, box, or file-backed domains.
- Load existing `.tree.npz`, `.forest`, and `.forest.simcache` files.
- Reuse simulation caches for fast analysis-only forest loading.
- Save generated trees/forests and optional forest simulation caches.
- Use normal CCO-style growth.
- Use the fast equal-bifurcation growth mode through `growth.n_equal_bifurcations`.
- Use scheduled forest growth.
- Use nearest-tree forest assignment with optional inter-tree collision rejection.
- Validate and repair connectivity.

Simple geometry mode currently supports:

- `onechannel`
- `multichannel`
- `snake`

Simple mode is intended for channel-design studies and uses the same flow/concentration/tissue oxygen kernels where possible.

### Simulation

GFM exposes the main TissueSim simulation controls through JSON:

- Fluids: `blood` and `water`.
- Flow sources: per-tree flow, loaded tree root flow, and total-Qin splitting.
- Concentration solvers supported by `TissueSim_cube_local.py`, including:
  - `topdown`
  - `network`
  - `topdown_ext`
  - `topdown_ext_hybrid_bg`
  - `topdown_ext_treecode`
- Random tissue sampling.
- Regular grid tissue sampling for grid/ParaView workflows.
- Geometry-only runs.
- Optional tissue nearest-vessel fields.
- Cext controls through `simulation.cext`.
- TissueSim constant overrides through `simulation.tissuesim`.
- Infarction-style downstream blocking through `simulation.infarction`.

Compute is float64 for the core simulation path. Export arrays can be written as float32 or float64, but public-svv float32 compute parity is not part of the current supported path.

### Outputs

Every normal `run` can write:

- `manifest.json`: settings, resolved defaults, dependency version, timings, output paths.
- `summary.csv`: TissueSim-style wide summary rows.
- `segments.csv`: one row per vessel segment.
- `points.csv`: tissue sample/grid points and oxygen fields.
- `vessels.vtp`: ParaView vessel polyline data.
- `oxygen_points.vtp`: ParaView tissue oxygen point data.
- `domain_boundary.vtp`: ParaView domain boundary.
- `domain_mesh.vtu`: ParaView domain mesh.
- Saved `.tree.npz`, `.forest`, `.forest.simcache`, or `.simple.npz` files when enabled.

Summary-only runs skip expensive segment/point row materialization when those outputs are disabled.

### Sweeps

The `sweep` command runs a target/fluid/sample/Qin sweep and writes one combined summary CSV. It is useful for recreating legacy files such as `Cube_FFT.csv`.

The sweep runner grows targets in non-decreasing order and reuses the in-memory tree, matching the old `BUILD_ON_PREVIOUS` style workflow more closely than running each target independently.

### Compatibility Status

Known-good coverage includes:

- Tree parity against `TissueSim_cube_local.py` for deterministic cube cases.
- Equal-bifurcation geometry smoke tests.
- Forest generation, export, cache reload, and direct `.forest.simcache` loading.
- Scheduled forest growth.
- Nearest-tree forest growth.
- Simple-channel smoke tests.
- Public `svv==0.0.48` smoke tests for tree and nearest-tree forest settings.
- Runtime benchmarks against direct `TissueSim_cube_local.py` for 10k and 1M terminal trees.

Known limitations:

- DLP mode is intentionally not carried forward.
- The Qt seed editor UI is not ported.
- Full high-scale shared-global forest Cext behavior from the legacy exporter is not fully redesigned behind the new CLI yet.
- The new CLI is not yet a full replacement for every `TissueSim_heart_forest.py` option.
- `.dmn` interchange with older installed `svv` paths needs caution; prefer normal GFM settings or already verified domain files.
- Float32 compute is out of scope for now.

## Repository Layout

```text
gfm/
  cli.py               Command-line entry point.
  config.py            JSON parsing, defaults, and validation.
  growth.py            Domain creation, tree/forest loading, growth, cache handling.
  simulation.py        Flow, concentration, tissue oxygen, summary/segment/point data.
  export.py            CSV and ParaView writers.
  sweep.py             Multi-target/fluid summary sweep runner.
  simple.py            Simple channel geometry backend.
  svv_adapter.py       Public-svv import boundary and compatibility hooks.
  _svv_*.py            Vendored compatibility modules for local-only svv behavior.

examples/
  cube_tree_smoke.json
  cube_tree_equal_smoke.json
  cube_forest_smoke.json
  cube_forest_nearest_smoke.json
  simple_channel_smoke.json
  cube_fft_recreate.json

tests/
  gfm_refactor_audit.py
  benchmark_tissuesim_vs_gfm.py

docs/
  gfm_refactor_audit.md
```

## Setup

On Carl's machine, use the existing `svva2` environment:

```bash
cd /home/carl/GFM
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli --help
```

`TissueSim_cube_local.py` is still required in the repo root. The refactor has moved orchestration, configuration, loading, caching, and export behavior into `gfm/`, but the current simulation kernels still come from that file.

For a fresh environment, install public `svv` and the scientific dependencies used by the simulation/export path:

```bash
python -m pip install svv==0.0.48 numpy scipy pyvista
```

GPU/Cext acceleration requires a working CUDA/CuPy setup compatible with the machine. If CuPy/CUDA is not available, use CPU-compatible solvers/settings or avoid GPU-only Cext cases.

## Quick Start

Run the smallest tree smoke example:

```bash
cd /home/carl/GFM
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_tree_smoke.json
```

Outputs are written relative to the settings file unless an absolute `outputs.out_dir` is used. For the example above:

```text
examples/gfm_run_tree_smoke/
  manifest.json
  summary.csv
  segments.csv
  points.csv
  vessels.vtp
  oxygen_points.vtp
  domain_boundary.vtp
  domain_mesh.vtu
  cube_tree.tree.npz
```

Create a starter settings file:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli init-settings my_settings.json
```

## Settings Structure

All runs use a JSON object with these top-level sections:

```json
{
  "domain": {},
  "network": {},
  "growth": {},
  "simulation": {},
  "outputs": {}
}
```

Sweeps add:

```json
{
  "sweep": {}
}
```

### Domain

Cube domain:

```json
{
  "domain": {
    "type": "cube",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

Rectangular box domain:

```json
{
  "domain": {
    "type": "box",
    "dimensions": [1.06, 0.811, 0.46],
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

File-backed domain:

```json
{
  "domain": {
    "type": "file",
    "path": "path/to/domain_or_mesh.dmn",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

Supported file-backed inputs are `.dmn` files readable by the adapter and mesh files readable by PyVista.

### Network

Single tree:

```json
{
  "network": {
    "mode": "tree",
    "target_terminal_count": 1000,
    "root": {
      "start": [0.49, -0.49, -0.49],
      "direction": [-0.49, 0.49, 0.49]
    }
  }
}
```

Forest:

```json
{
  "network": {
    "mode": "forest",
    "target_terminal_counts": [100, 100],
    "roots": [
      {
        "start": [0.49, -0.49, -0.49],
        "direction": [-0.49, 0.49, 0.49]
      },
      {
        "start": [-0.49, 0.49, 0.49],
        "direction": [0.49, -0.49, -0.49]
      }
    ]
  }
}
```

Load an existing tree or forest:

```json
{
  "network": {
    "mode": "tree",
    "input_path": "runs/cube_tree/cube_tree.tree.npz",
    "target_terminal_count": 1000
  },
  "growth": {
    "enabled": false
  }
}
```

For forests, `input_path` can point to:

- `.forest`
- `.forest.simcache`

When `outputs.use_cache` is true, GFM will prefer or create a `.forest.simcache` for fast analysis-only loading.

### Growth

Basic CCO-style tree growth:

```json
{
  "growth": {
    "enabled": true,
    "n_closest_vessels": 2,
    "n_points": 50,
    "weighted_sampling": false,
    "ignore_collisions": true,
    "allow_inside_vessels": true
  }
}
```

Equal-bifurcation mode:

```json
{
  "growth": {
    "n_equal_bifurcations": 200000,
    "equal_terminal": {
      "batch_size": 100,
      "length": 0.05,
      "length_shrink": 0.5,
      "report_timings": false
    }
  }
}
```

`n_equal_bifurcations` switches to equal-bifurcation growth after the configured terminal threshold. Set it to `null` or a negative value to disable.

Scheduled forest growth:

```json
{
  "growth": {
    "assignment": "scheduled",
    "bulk_growth_mode": "never",
    "add_total": 200,
    "add_split_mode": "equal",
    "checkpoint_path": "runs/forest/checkpoint.forest",
    "checkpoint_every_adds": 50,
    "save_target_counts": [100, 200]
  }
}
```

Nearest-tree forest growth:

```json
{
  "growth": {
    "assignment": "nearest-tree",
    "bulk_growth_mode": "never",
    "nearest_tree_batch_points": 256,
    "ignore_collisions": false,
    "n_ignore_collisions": -1,
    "collision_retry_limit": 100,
    "collision_failure_mode": "error"
  }
}
```

Domain segment checks:

```json
{
  "growth": {
    "strict_domain_segments": true,
    "strict_domain_max_terminals": 10000,
    "domain_line_samples": 4,
    "domain_line_tolerance": 0.0
  }
}
```

### Simulation

Basic blood simulation:

```json
{
  "simulation": {
    "fluid": "blood",
    "build_fluid": "blood",
    "qin_target_ul_min": 900.0,
    "concentration_solver": "topdown",
    "distance_sample_count": 1000,
    "sample_mode": "random",
    "geometry_only": false
  }
}
```

Summary-only geometry run:

```json
{
  "simulation": {
    "geometry_only": true,
    "distance_sample_count": 0
  }
}
```

Regular grid sampling:

```json
{
  "simulation": {
    "sample_mode": "grid",
    "tissue_grid": {
      "nx": 64,
      "ny": 64,
      "nz": 64
    }
  }
}
```

Cext/hybrid FFT path:

```json
{
  "simulation": {
    "concentration_solver": "topdown_ext_hybrid_bg",
    "tissue_accel": "gpu",
    "tissue_gpu_validate_points": 0,
    "cext": {
      "accel_mode": "gpu",
      "frozen_accel_mode": "gpu",
      "vess_coupling_accel": "anderson",
      "hybrid_bg_mode": "fft",
      "hybrid_bg_solver": "fft"
    }
  }
}
```

TissueSim constant overrides:

```json
{
  "simulation": {
    "tissuesim": {
      "finite_radius_o2_terms": "none",
      "lumen_wall_closure": "wellmixed",
      "nearest_tissue_vessels": 250,
      "window_factor": 6
    }
  }
}
```

Infarction-style block:

```json
{
  "simulation": {
    "infarction": {
      "global_segment_id": 12345,
      "fraction_blocked": 1.0
    }
  }
}
```

### Outputs

Full output:

```json
{
  "outputs": {
    "out_dir": "runs/case_001",
    "prefix": "case_001",
    "write_paraview": true,
    "write_summary_csv": true,
    "write_segments_csv": true,
    "write_points_csv": true,
    "include_tissue_nearest_fields": false,
    "save_network": true,
    "use_cache": true,
    "export_float_dtype": "float64",
    "export_index_dtype": "int64",
    "vessel_resolution": 2
  }
}
```

Fast summary-only output:

```json
{
  "outputs": {
    "out_dir": "runs/summary_only",
    "write_paraview": false,
    "write_summary_csv": true,
    "write_segments_csv": false,
    "write_points_csv": false,
    "save_network": false
  }
}
```

Use summary-only settings for large sweeps unless you explicitly need per-segment, per-point, or ParaView files.

## Common Workflows

### 1. Run A Tree Case

Edit or copy `examples/cube_tree_smoke.json`, then run:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_tree_smoke.json
```

Increase the tree size by changing:

```json
{
  "network": {
    "target_terminal_count": 10000
  }
}
```

### 2. Run Equal-Bifurcation Growth

Use:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_tree_equal_smoke.json
```

The important setting is:

```json
{
  "growth": {
    "n_equal_bifurcations": 2
  }
}
```

For production-scale runs, use a larger threshold such as `200000`.

### 3. Run A Forest

Use:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_forest_smoke.json
```

The forest roots and target counts are defined in:

```json
{
  "network": {
    "mode": "forest",
    "target_terminal_counts": [1, 1],
    "roots": []
  }
}
```

### 4. Run Nearest-Tree Forest Growth

Use:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/cube_forest_nearest_smoke.json
```

This mode assigns candidate points to the nearest tree and can reject inter-tree collisions.

### 5. Load A Saved Tree Or Forest

Use `network.input_path` and disable growth:

```json
{
  "network": {
    "mode": "tree",
    "input_path": "runs/tree/tree.tree.npz",
    "target_terminal_count": 10000
  },
  "growth": {
    "enabled": false
  }
}
```

For forests:

```json
{
  "network": {
    "mode": "forest",
    "input_path": "runs/forest/forest.forest"
  },
  "growth": {
    "enabled": false
  },
  "outputs": {
    "use_cache": true
  }
}
```

### 6. Run A Simple Channel Geometry

Use:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli run --settings examples/simple_channel_smoke.json
```

Core settings:

```json
{
  "network": {
    "mode": "simple",
    "simple": {
      "mode": "onechannel",
      "axis": "x",
      "radius_cm": 0.015,
      "flow_ul_min": 24.0384615385,
      "concentration_inlet": 0.22471
    }
  },
  "growth": {
    "enabled": false
  }
}
```

For multiple channels, set `network.simple.mode` to `multichannel` and provide `y_offsets_cm`. For a serpentine channel, use `snake`.

### 7. Export For ParaView

Set:

```json
{
  "outputs": {
    "write_paraview": true,
    "write_segments_csv": true,
    "write_points_csv": true
  }
}
```

For grid-style oxygen points:

```json
{
  "simulation": {
    "sample_mode": "grid",
    "tissue_grid": {
      "nx": 64,
      "ny": 64,
      "nz": 64
    }
  }
}
```

The main ParaView files are:

- `vessels.vtp`
- `oxygen_points.vtp`
- `domain_boundary.vtp`
- `domain_mesh.vtu`

### 8. Recreate `Cube_FFT.csv`

Use the provided sweep settings:

```bash
/home/carl/miniconda3/envs/svva2/bin/python -m gfm.cli sweep --settings examples/cube_fft_recreate.json
```

This writes:

```text
runs/cube_fft/Cube_FFT.csv
runs/cube_fft/Cube_FFT_manifest.json
```

The config mirrors the observed legacy sweep:

- cube side length `1.0`
- target list starting with `[0, 0, 1, 2, ...]`
- water and blood rows
- `distance_sample_count=1000000`
- `topdown_ext_hybrid_bg`
- legacy 123-column summary output

The full sweep goes to 3,000,000 terminals and is a long GPU-heavy job.

## Testing And Validation

Run the main audit:

```bash
/home/carl/miniconda3/envs/svva2/bin/python tests/gfm_refactor_audit.py --targets 1 100 --sample-count 1000 --timeout 3600
```

Run the larger 1000-terminal audit:

```bash
/home/carl/miniconda3/envs/svva2/bin/python tests/gfm_refactor_audit.py --work-dir .gfm_audit_runs_1000 --targets 1000 --sample-count 1000 --timeout 3600 --skip-cext
```

Audit reports are written under:

```text
.gfm_audit_runs/audit_report.md
.gfm_audit_runs_1000/audit_report.md
```

See `docs/gfm_refactor_audit.md` for the latest summarized validation notes.

## Runtime Benchmarks

Compare direct `TissueSim_cube_local.py` simulation against GFM using the same loaded tree and same tissue sample points:

```bash
/home/carl/miniconda3/envs/svva2/bin/python tests/benchmark_tissuesim_vs_gfm.py \
  --tree .gfm_benchmark_cache/legacy_cube_t10000.tree.npz \
  --target 10000 \
  --solver topdown_ext_hybrid_bg \
  --fluid blood \
  --sample-count 10000 \
  --runs 3 \
  --warmup 1
```

Recent summary-only benchmark results:

| Tree size | Solver | Tissue points | TissueSim median | GFM median | Ratio |
|---:|---|---:|---:|---:|---:|
| 10,000 terminals | `topdown` | 10,000 | 0.178 s | 0.180 s | 1.012x |
| 10,000 terminals | `topdown` | 50,000 | 0.698 s | 0.759 s | 1.086x |
| 10,000 terminals | `topdown_ext_hybrid_bg` | 10,000 | 2.362 s | 2.424 s | 1.026x |
| 1,000,000 terminals | `topdown` | 10,000 | 5.286 s | 5.154 s | 0.975x |
| 1,000,000 terminals | `topdown_ext_hybrid_bg` | 10,000 | 13.516 s | 13.689 s | 1.013x |

These timings exclude tree loading, growth, and file export. Enabling `segments.csv`, `points.csv`, or ParaView exports adds work by design.

## Practical Notes

- Keep generated runs under `runs/`, `.gfm_audit_runs/`, or `.gfm_benchmark_cache/`.
- For large sweeps, disable `write_segments_csv`, `write_points_csv`, and `write_paraview` unless needed.
- Use `growth.enabled=false` when analyzing a saved tree or forest without further growth.
- Use `.forest.simcache` for fast forest analysis loading.
- Use absolute paths only when a workflow must reference data outside the settings directory.
- Keep edits to public-svv compatibility inside GFM; do not modify installed `site-packages/svv`.

## Troubleshooting

### Import Errors

Make sure you are running from the repo root or have `/home/carl/GFM` on `PYTHONPATH`:

```bash
cd /home/carl/GFM
python -m gfm.cli --help
```

### GPU/CuPy Errors

Use `topdown` or CPU-compatible settings first. For Cext GPU workflows, verify the CUDA/CuPy environment before running large jobs.

### Slow Large Runs

Check output settings. Writing millions of segment rows or large point grids can dominate runtime and disk use. For TissueSim-style sweeps, use summary-only output.

### Cache Loading Problems

For forest analysis, prefer:

```json
{
  "growth": {"enabled": false},
  "outputs": {"use_cache": true}
}
```

If a `.forest.simcache` is stale or missing, GFM will try to build one from the `.forest` file.
