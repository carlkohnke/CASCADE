# Changelog

## 0.1.0rc1

- Renamed the production package and public interfaces from GFM to CASCADE while retaining Green's function method terminology where scientifically meaningful.
- Added one installable `cascade` CLI for simulation, sweeps, diagnostics, and heart/ParaView export.
- Added the installable CASCADE Studio GUI and viewer entry points.
- Added explicit CSV/NPZ segment geometry input and file-backed custom domains.
- Added strict core configuration validation and reproducibility-rich run manifests.
- Added CPU and CUDA environment bootstrap paths and release locks.
- Documented public `svv` compatibility behavior and upstream gaps.

This release candidate does not yet claim numerical or performance equivalence to all legacy TissueSim workflows. That validation is the next release gate.
