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
