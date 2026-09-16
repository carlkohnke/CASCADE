# Example simulations

Save an example as JSON and run it with:

```bash
cascade run --settings case.json
```

The generated-tree example is self-contained. The later sections show settings
to add to a case or require the referenced input files.

## Generate a vascular tree

```json
{
  "domain": {"type": "cube", "side_length": 1.0, "random_seed": 42},
  "network": {
    "mode": "tree",
    "target_terminal_count": 100,
    "root": {
      "start": [0.49, -0.49, -0.49],
      "direction": [-0.49, 0.49, 0.49]
    }
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

## Pressure-driven flow

Use the pressure-pressure Kirchhoff mode and prescribe inlet and outlet
pressures in pascals:

```json
{
  "simulation": {
    "fluid": "blood",
    "kirchhoff_bc_mode": "pressure_pressure"
  },
  "settings": {
    "hemodynamics": {
      "root_pressure": 7080.0,
      "terminal_pressure": 5999.0,
      "scale_dp_by_volume": false
    },
    "kirchhoff": {"solver": "tree"}
  }
}
```

The inlet flow is a solved output. It is reported in `summary.csv`; segment
pressures and flows are written to `segments.csv` and `vessels.vtp` when those
outputs are enabled. Studio exposes the same mode as **Inlet pressure + outlet
pressure**.

## Anatomical forest (multiple trees) with occlusion

Generated and loaded anatomical forests use the same simulation engine. This
example loads a heart surface and forest, enables a shared external field, and
occludes one collection-wide segment:

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

Omit `simulation.occlusion` for a baseline run. Values between zero and one
apply a partial radius reduction; a value of one can also suppress the target's
downstream subtree.

Legacy `.tree.npz`, `.forest`, and `.forest.simcache` files may contain Python
pickle payloads. Load them only from a trusted source.
