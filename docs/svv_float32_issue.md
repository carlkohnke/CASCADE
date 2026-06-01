# svVascularize float32 compatibility issue

The GFM script relies on float32 support in svVascularize.

Observed failure after moving TissueSim_cube_local.py out of site-packages:

ValueError: Buffer dtype mismatch, expected 'double' but got 'float'

The installed svVascularize source appears to contain float32-aware changes in:

- svv/tree/utils/c_basis.py
- svv/tree/utils/c_basis.pyx

Specifically, c_basis.pyx contains _basis_f32, _basis_inplace_f32, and dtype dispatch in basis(...).

However, the runtime error comes from the compiled extension:

- svv_accel.tree.utils.c_basis

This suggests the compiled .so is stale or was not rebuilt after the float32 Cython edits.

Also, svv/tree/branch/root.py still contains dtype=float conversions for start and direction, which default to float64 and may be incompatible with full float32 support.

Likely next step:
- Create a clean svVascularize fork or patch branch.
- Rebuild the Cython extensions.
- Confirm svv_accel.tree.utils.c_basis.basis accepts float32 arrays.

Current GFM-local direction:
- The public-svv adapter pass keeps tree data as float64.
- `TissueSim_cube_local.py` and `export_paraview_heart_forest_grid.py` import `Domain`, `Tree`, and `Forest` through `gfm.svv_adapter`.
- The adapter keeps GFM-owned domain setup, root/add growth wrappers, save/load, simulation-cache, and equal-terminal bifurcation methods without requiring changes to the installed/public `svv` package.

Remaining long-term issue:
- The installed `svv_accel.tree.utils.c_basis` extension is still stale/float64-only.
- A future clean package/fork should rebuild the accelerated extension or provide maintained float32 support before tree data is switched back to float32.
