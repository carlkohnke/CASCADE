# CASCADE

CASCADE is a vascular growth, hemodynamics, and tissue-oxygen simulation toolkit built around the public `svVascularize` (`svv`) package. It provides a reproducible command-line interface, CASCADE Studio desktop GUI, and ParaView-compatible VTK exports.

CASCADE supports:

- Single vascular trees and multi-tree forests.
- Cube, rectangular-box, sphere, and file-backed mesh domains.
- Loaded `.tree.npz`, `.forest`, and `.forest.simcache` networks.
- Simple channels, vascular lattices, and explicit CSV/NPZ segment networks.
- Blood and water flow and oxygen transport.
- Tree, network, finite-radius, and Green's function Cext solvers.
- Random and structured Cartesian tissue sampling.
- CSV, VTP, and VTU export for analysis in ParaView.

This repository targets the public `svv==0.0.48` API. CASCADE-owned compatibility behavior is documented in [docs/svv-compatibility.md](docs/svv-compatibility.md).

> **Legacy file safety:** `.tree.npz`, `.forest`, and `.forest.simcache` files
> may contain Python pickle payloads for compatibility with existing `svv`
> archives. Load these legacy files only when they come from a trusted source.
> CSV segment networks and CASCADE's non-object NPZ inputs do not require
> pickle deserialization.

## Status

The current version is `0.1.0rc5`. This is a release candidate: configuration
and output formats may still change before the first stable release. See
[known issues](docs/known-issues.md) for current platform and scientific
limitations.

The implementation is organized by scientific responsibility rather than by
entry point: domain and vessel architecture, flow, vessel concentration, Cext,
Green's Function Method tissue oxygen, exporting, simulation, and GUI code each
have dedicated packages. See [docs/architecture.md](docs/architecture.md) for
the package map and compatibility boundaries.

## Requirements

- Windows 10/11 or Linux on x86-64. Native Windows qualification status is
  tracked in [docs/windows-qualification.md](docs/windows-qualification.md).
- Python 3.12.
- For GPU execution, an NVIDIA driver compatible with the selected CUDA package.
- ParaView is optional and is used only to inspect exported VTK files.

## Install

For a released wheel on native Windows, follow the
[native Windows installation guide](docs/windows.md). The supported CPU and
CUDA 13 installations use an isolated CPython 3.12 environment, install binary
wheels, and launch Studio without WSL or a console window.

For Linux development from the repository, create a CPU environment with
Studio from the repository root:

```bash
python setup_env.py \
  --venv .venv \
  --gui
source .venv/bin/activate
cascade doctor --no-gpu-probe
```

For a CUDA 13 environment:

```bash
python setup_env.py \
  --venv .venv \
  --gui \
  --gpu cu13 \
  --cuda-path /path/to/targets/x86_64-linux

source .venv/bin/activate
cascade doctor --require-gpu
```

The GPU extra installs CuPy plus the matching wheel-provided CUDA toolkit. A
compatible host NVIDIA driver is still required.

Normal package installation is also supported:

```bash
python -m pip install .
python -m pip install '.[gui]'
python -m pip install '.[gui,gpu-cu13]'
```

## Command-line interface

```text
cascade run --settings case.json
cascade batch --settings case-001.json case-002.json case-003.json
cascade sweep --settings sweep.json
cascade inspect --settings case.json
cascade prepare --settings case.json
cascade init-settings case.json
cascade doctor
cascade self-test
cascade self-test --require-gpu
```

Create and execute a starter configuration:

```bash
cascade init-settings case.json
cascade run --settings case.json
```

The generated starter is CPU-safe. `cascade self-test` verifies a bounded
loaded-tree solve, VTK/CSV export, manifest provenance, and the required public
`svv` dependency using only installed package contents. Add `--require-gpu` to
exercise an actual CUDA Cext solve and GPU tissue calculation.

Settings use six principal sections:

```json
{
  "domain": {},
  "network": {},
  "growth": {},
  "simulation": {},
  "settings": {},
  "outputs": {}
}
```

Unknown top-level and core-section settings are rejected so spelling mistakes cannot silently change a simulation.

## Example tree run

```json
{
  "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
  "network": {
    "mode": "tree",
    "target_terminal_count": 100,
    "root": {"start": [0.49, -0.49, -0.49], "direction": [-0.49, 0.49, 0.49]}
  },
  "growth": {
    "n_closest_vessels": 2,
    "n_points": 50,
    "ignore_collisions": true,
    "allow_inside_vessels": true
  },
  "simulation": {
    "fluid": "blood",
    "qin_target_ul_min": 900.0,
    "concentration_solver": "topdown",
    "distance_sample_count": 1000,
    "sample_mode": "random",
    "tissue_accel": "cpu"
  },
  "outputs": {
    "out_dir": "runs/tree_100",
    "prefix": "tree_100",
    "write_paraview": true,
    "save_network": true
  }
}
```

For a pressure-driven run, prescribe the inlet and outlet pressures and select
the pressure-pressure Kirchhoff mode. The inlet flow is then a solved output;
CASCADE does not sweep or iterate over candidate flow rates:

```json
{
  "simulation": {
    "fluid": "blood",
    "kirchhoff_bc_mode": "pressure_pressure"
  },
  "settings": {
    "hemodynamics": {
      "root_pressure": 7080.254371993,
      "terminal_pressure": 5999.51,
      "scale_dp_by_volume": false
    },
    "kirchhoff": {"solver": "tree"}
  }
}
```

Pressures in settings JSON are pascals. The solved inlet flow is reported in
`summary.csv`, and segment pressures and flows are written to `segments.csv`
and `vessels.vtp` when those outputs are enabled. The same mode is available in
Studio as **Inlet pressure + outlet pressure**.

## Custom domains

File-backed domains use surface/volume meshes readable by PyVista. CASCADE Studio also includes the packaged `bivent3.stl` heart surface. Historical `.dmn` files can be loaded for compatibility, but `.dmn` is not a supported cross-version interchange format:

```json
{
  "domain": {
    "type": "file",
    "path": "my_domain.vtp",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

Relative paths are resolved against the settings file. Store a mesh beside the configuration that uses it, or provide an absolute path.

## Custom vascular geometry

Use `network.mode="simple"` and `network.simple.mode="custom"` to load an explicit network:

```json
{
  "network": {
    "mode": "simple",
    "simple": {
      "mode": "custom",
      "path": "geometry/channels.csv",
      "flow_ul_min": 24.0,
      "concentration_inlet": 0.22471
    }
  },
  "growth": {"enabled": false}
}
```

CSV coordinates are in centimetres. Required columns are:

```text
start_x,start_y,start_z,end_x,end_y,end_z
```

Optional `radius_cm`, `prox_id`, and `dist_id` columns provide segment radius and explicit node connectivity. When `radius_cm` is absent, `network.simple.radius_cm` supplies it. Without node IDs, CASCADE infers connectivity by matching endpoints after rounding to 12 decimal places. `inlet_nodes` and `outlet_nodes` can be supplied in JSON when they cannot be inferred from graph degree.

NPZ geometry uses `starts`, `ends`, and optional `radii`, `prox_ids`, and `dist_ids` arrays. Built-in simple modes are `onechannel`, `multichannel`, `snake`, and `lattice`.

## ParaView output

A normal run can write:

- `vessels.vtp`: vessel centerlines and solved fields.
- `oxygen_points.vtp`: random or grid tissue samples.
- `domain_boundary.vtp`: domain surface.
- `domain_mesh.vtu`: tetrahedral domain mesh.
- `summary.csv`, `segments.csv`, and `points.csv`.
- `manifest.json`: settings, input hashes, software versions, environment, timings, and output paths.

Use structured sampling for regular ParaView grids:

```json
{
  "simulation": {
    "sample_mode": "grid",
    "tissue_grid": {"nx": 64, "ny": 64, "nz": 64}
  }
}
```

Use a frozen coordinate fixture when two runs must evaluate identical tissue points:

```json
{
  "simulation": {
    "sample_mode": "file",
    "sample_points_path": "fixtures/tissue_points.npz",
    "distance_sample_count": 0
  }
}
```

NPZ files contain `points` or `sample_points` with shape `(N, 3)`. NPY files contain the array directly. CSV files use `x,y,z` headers. Coordinates are in centimetres, paths resolve relative to the settings file, and the manifest records the resolved path and SHA-256.

CASCADE permits one memory-intensive simulation or preparation job per user at
a time. A competing CLI or Studio job exits with an active-owner error instead
of risking two resident simulations. Studio processes its own queue
sequentially.

For several related cases, `cascade batch` keeps one compatible network,
spatial context, and compiled accelerator state warm while still running cases
strictly serially. Results are exported before per-case arrays are released;
pass `--continue-on-error` for unattended batches that should proceed after an
individual failure. A one-shot `cascade run` performs hard cleanup when it
finishes.

When growth is disabled, existing tree and forest inputs load in analysis-only mode. CASCADE omits growth preallocation and spatial indexes, and `.forest.simcache` members stream directly into the selected working dtype to avoid retaining a second full vessel table.

## Anatomical forests and occlusion

Heart simulations use the same `cascade run` engine as cubes, generated forests,
and simple geometries. A loaded anatomical forest can opt into one external field
shared by every vascular network and can apply a reversible partial stenosis or
complete downstream occlusion using collection-wide segment IDs:

```json
{
  "domain": {
    "type": "file",
    "path": "inputs/heart.stl",
    "use_cache": true
  },
  "network": {
    "mode": "forest",
    "input_path": "inputs/heart.forest"
  },
  "growth": {"enabled": false},
  "simulation": {
    "fluid": "blood",
    "flow_source": "tree-root-flow",
    "concentration_solver": "topdown_ext_hybrid_bg",
    "sample_mode": "grid",
    "tissue_grid": {"nx": 64, "ny": 64, "nz": 64},
    "external_field": {
      "enabled": true,
      "scope": "shared",
      "mode": "shared-global"
    },
    "occlusion": {
      "global_segment_id": 120,
      "fraction_blocked": 1.0,
      "include_downstream_when_complete": true
    }
  },
  "outputs": {
    "out_dir": "runs/heart",
    "prefix": "heart",
    "export_float_dtype": "float32",
    "export_index_dtype": "int64"
  }
}
```

Omit `simulation.occlusion` for a baseline run. Set `fraction_blocked` between
zero and one for a radius-reduction stenosis; a value of one can also suppress
the target's downstream subtree. Solver tuning remains in the normal named
`settings` sections, so anatomical cases do not have a parallel set of CLI-only
defaults.

## CASCADE Studio

After installing the `gui` extra:

```bash
cascade-gui
```

On Windows, `cascade-gui.exe` is the normal console-free launcher and
`cascade-gui-console.exe` is the diagnostic launcher. See the
[native Windows guide](docs/windows.md).

The GUI configures domains, networks, solver settings, sweeps, queued runs, and
result visualization. Its persistent local worker reuses compatible geometry
and accelerator state between serial jobs, evicts incompatible state before
the next solve, and retires after an idle interval or when Studio closes.

For a saved large tree, inspect its scale without loading scientific arrays and
optionally prepare its one-time memory-mapped representation:

```bash
cascade inspect --settings case.json
cascade prepare --settings case.json
```

`cascade run`, `cascade batch`, and Studio automatically use a fresh prepared
tree whose configured float/index dtypes match. The original archive remains
the fallback and source of truth. If `--cache-dir` is supplied to `prepare`, set
`CASCADE_PREPARED_CACHE_DIR` to that directory for later runs.

See [docs/gui.md](docs/gui.md).

## Documentation

- [CASCADE Studio guide](docs/gui.md)
- [Native Windows installation](docs/windows.md)
- [Custom vascular geometry format](docs/custom-geometry.md)
- [Architecture and package map](docs/architecture.md)
- [Public svVascularize compatibility](docs/svv-compatibility.md)
- [Known issues and limitations](docs/known-issues.md)
- [Dependency and lock files](requirements/README.md)

## Development

```bash
python -m pip install '.[dev,gui]'
python -m ruff check --select E9,F63,F7,F82,F601,F811,E741 src setup_env.py
python -m build
cascade self-test
```

## Reproducibility

Keep the settings JSON, domain/network inputs, `manifest.json`, dependency lock, and raw command log together for every production run. Do not modify the installed public `svv` package; CASCADE-specific compatibility code belongs under `src/cascade/` and should be retired when the corresponding behavior is available upstream.

## License

CASCADE is provided under Stanford's academic, non-commercial license. Review
the complete terms in [LICENSE](LICENSE) before accessing or using the software.
