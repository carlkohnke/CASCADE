# Outputs and visualization

CASCADE can write tables for analysis and VTK-family files for ParaView or
PyVista.

## Output files

- `summary.csv`: run-level measurements and timings.
- `segments.csv`: per-vessel solved fields.
- `points.csv`: sampled tissue fields.
- `vessels.vtp`: vessel centerlines and solved fields.
- `oxygen_points.vtp`: sampled tissue points.
- `domain_boundary.vtp`: domain surface.
- `domain_mesh.vtu`: tetrahedral domain mesh.
- `manifest.json`: settings, input hashes, software versions, environment,
  timings, and output paths.

Detailed CSV and VTK outputs can be disabled when only the run summary is
needed.

## Structured tissue sampling

Use a Cartesian grid for regular ParaView sampling:

```json
{
  "simulation": {
    "sample_mode": "grid",
    "tissue_grid": {"nx": 64, "ny": 64, "nz": 64}
  }
}
```

## Fixed tissue coordinates

Use a coordinate file when multiple runs must evaluate the same points:

```json
{
  "simulation": {
    "sample_mode": "file",
    "sample_points_path": "fixtures/tissue_points.npz",
    "distance_sample_count": 0
  }
}
```

NPZ files contain `points` or `sample_points` with shape `(n, 3)`. NPY files
contain the array directly. CSV files use `x,y,z` headers. Coordinates are in
centimetres. Relative paths resolve from the settings file, and the manifest
records the resolved path and SHA-256 hash.

PyVista can reopen the VTP and VTU files directly. ParaView is optional and is
not required to run a simulation.
