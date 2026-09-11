# Runtime Settings

This folder owns the JSON-to-runtime settings boundary.

Each module contains defaults for one area of the solver:

- `growth.py`: CCO/equal-bifurcation and growth-time controls.
- `hemodynamics.py`: pressure, inlet flow, scaling, and Kirchhoff solver defaults.
- `hematocrit.py`: discharge/tube hematocrit and Pries-Secomb constants.
- `oxygen.py`: concentration, Graetz/lumen closure, O2 transport, Vmax, Km, diffusivity.
- `tissue.py`: tissue sampling, nearest-vessel search, cache, streaming, and GPU tissue kernel controls.
- `cext.py`: explicit extravascular concentration coupling, hybrid FFT/treecode, active sets, and acceleration controls.
- `numerics.py`: dtypes, caches, numba, and timing toggles.

Run files override these defaults with a top-level `settings` object:

```json
{
  "settings": {
    "kirchhoff": {
      "solver": "tree",
      "cg_rtol": 1e-10
    },
    "hematocrit": {
      "model": "pries_secomb",
      "flow_iterations": 2
    },
    "oxygen": {
      "solute_diffusivity": 2.41e-5,
      "vmax_mm": 0.04,
      "km": 0.0069
    },
    "cext": {
      "vess_coupling_omega_min": 0.025,
      "vess_coupling_omega_max": 1.4
    },
    "tissue": {
      "accel_mode": "gpu",
      "nearest_vessels": 250,
      "gpu_chunk_points": 8192
    },
    "growth": {
      "n_equal_bifurcations": 200000
    }
  }
}
```

Keys can use the short names above or exact runtime constant names such as
`CEXT_VESS_COUPLING_OMEGA_MIN`.

Compatibility inputs still work:

- `simulation.cext` maps to `settings.cext`.
- `simulation.tissuesim` maps through the same registry.
- `growth.equal_terminal` maps to `settings.growth`.
