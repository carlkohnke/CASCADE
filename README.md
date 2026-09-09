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

## Status

The current version is `0.1.0rc1`. It is a release candidate while full numerical and performance parity against the legacy TissueSim workflows is being audited.

The completed installation and operational gates are recorded in [the rc1 release report](docs/release-0.1.0rc1.md). Full legacy parity and speed acceptance remain a separate validation step.

## Requirements

- Linux or WSL2 on x86-64.
- Python 3.9. The release candidate is validated on Python 3.9.20.
- An NVIDIA driver compatible with CUDA 13 for the optional GPU configuration.
- ParaView is optional and is used only to inspect exported VTK files.

## Install

Create a CPU environment from the repository root:

```bash
python setup_env.py \
  --venv .venv \
  --dev \
  --gui \
  --constraints locks/requirements-py39-cpu.txt
source .venv/bin/activate
cascade doctor --no-gpu-probe
```

For the CUDA 13 configuration used by the development workstation:

```bash
python setup_env.py \
  --venv .venv \
  --dev \
  --gui \
  --gpu cu13 \
  --constraints locks/requirements-py39-cu13.txt \
  --cuda-path /path/to/targets/x86_64-linux

source .venv/bin/activate
cascade doctor --require-gpu
```

The GPU extra installs CuPy plus the matching CUDA runtime, cuFFT, and nvJitLink component wheels. A compatible host NVIDIA driver is still required.

Normal package installation is also supported:

```bash
python -m pip install .
python -m pip install '.[gui]'
python -m pip install '.[gui,gpu-cu13]'
```

## Command-line interface

```text
cascade run --settings case.json
cascade sweep --settings sweep.json
cascade export-heart --forest heart.forest --domain heart.stl --out-dir results/heart
cascade init-settings case.json
cascade doctor
```

Create and execute a starter configuration:

```bash
cascade init-settings case.json
cascade run --settings case.json
```

Settings use five principal sections:

```json
{
  "domain": {},
  "network": {},
  "growth": {},
  "simulation": {},
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

More configurations are provided in `examples/`.

## Custom domains

File-backed domains can use `.dmn` files readable by the CASCADE adapter or surface/volume meshes readable by PyVista:

```json
{
  "domain": {
    "type": "file",
    "path": "domains/my_domain.vtp",
    "side_length": 1.0,
    "random_seed": 42
  }
}
```

Relative paths are resolved against the settings file. Store reusable meshes under `domains/` or beside the configuration that uses them.

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

CSV coordinates and radii are in centimetres. Required columns are:

```text
start_x,start_y,start_z,end_x,end_y,end_z,radius_cm
```

Optional `prox_id` and `dist_id` columns provide explicit node connectivity. Without them, CASCADE infers connectivity by matching endpoints after rounding to 12 decimal places. `inlet_nodes` and `outlet_nodes` can be supplied in JSON when they cannot be inferred from graph degree.

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

The heart exporter exposes shared forest Cext and heart-specific export controls:

```bash
cascade export-heart \
  --forest inputs/heart.forest \
  --domain inputs/heart.stl \
  --out-dir runs/heart \
  --cext-forest-mode shared-global \
  --nx 64 --ny 64 --nz 64
```

Run `cascade export-heart --help` for the complete interface.

## CASCADE Studio

After installing the `gui` extra:

```bash
cascade-gui
```

The GUI configures domains, networks, solver settings, sweeps, queued runs, and result visualization. Each simulation runs in a separate worker process so its CPU and GPU allocations are released when the job exits.

See [docs/gui.md](docs/gui.md).

## Development

```bash
python -m pip install '.[dev,gui]'
pytest -q
python -m build
```

The release smoke suite must also be run against an installed wheel from outside the source checkout.

Legacy equivalence tools are retained for validation but are not installed with CASCADE:

```bash
python tests/cascade_refactor_audit.py --targets 1 100 --sample-count 1000 --skip-cext
python tests/benchmark_tissuesim_vs_cascade.py --help
```

## Reproducibility

Keep the settings JSON, domain/network inputs, `manifest.json`, dependency lock, and raw command log together for every production run. Do not modify the installed public `svv` package; CASCADE-specific compatibility code belongs under `src/cascade/` and should be retired when the corresponding behavior is available upstream.
