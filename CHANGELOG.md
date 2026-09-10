# Changelog

## Unreleased

- Added the packaged bivent3 heart surface as a portable CASCADE Studio domain option.
- Normalized default heart/Cext controls to float32 accelerator arrays, grid 256, Cext quadrature 1, tissue quadrature 5, one coupling iteration, and window factor 6.
- Removed bundled legacy TissueSim copies and obsolete same-process comparison helpers; M2 uses isolated wheel and external-oracle environments.
- Defined STL/PyVista-readable mesh input and VTP/VTU output as the public domain interchange path; legacy `.dmn` loading remains internal compatibility.
- Added fixed CSV/NPY/NPZ tissue-coordinate inputs for pointwise validation.
- Enforced one resident CLI simulation per user and added explicit sweep/CUDA cleanup between cases.
- Made analysis-only tree and forest loads avoid growth buffers and spatial indexes; simulation-cache arrays stream directly into the requested float32 working dtype.
- Aligned the water/cell-media inlet oxygen default with the frozen TissueSim cube oracle (`0.2211`) across the runtime, settings registry, and Studio fallback.

## 0.1.0rc1

- Renamed the production package and public interfaces from GFM to CASCADE while retaining Green's function method terminology where scientifically meaningful.
- Added one installable `cascade` CLI for simulation, sweeps, diagnostics, and heart/ParaView export.
- Added the installable CASCADE Studio GUI and viewer entry points.
- Added explicit CSV/NPZ segment geometry input and file-backed custom domains.
- Added strict core configuration validation and reproducibility-rich run manifests.
- Added CPU and CUDA environment bootstrap paths and release locks.
- Documented public `svv` compatibility behavior and upstream gaps.

This release candidate does not yet claim numerical or performance equivalence to all legacy TissueSim workflows. That validation is the next release gate.
