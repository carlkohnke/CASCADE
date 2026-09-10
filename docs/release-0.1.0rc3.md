# CASCADE 0.1.0rc3 release-candidate report

Date: 2026-09-10

## Result

CASCADE 0.1.0rc3 is the post-M3 CLI/package hardening candidate. It preserves
the M2 numerical and M3 performance conclusions recorded for 0.1.0rc2; no
scientific algorithm, accepted tolerance, canonical input, or benchmark result
was changed by this audit.

The CLI is locally release-ready for the qualified Linux/WSL2, Python 3.9, and
CUDA 13 scope. M4 publication is not claimed: license selection, a GitHub
remote, observed remote CI, and the remaining human GUI acceptance remain
separate gates.

## Audit findings resolved

- Corrected undefined-name branches in clamped-root compatibility, CPU Cext,
  GPU treecode initialization/timing, and retained linear tissue helpers.
- Restored growth-mode forest connections and completed analysis-only tree
  reconstruction so a loaded tree can solve and save again.
- Made `cascade init-settings` CPU-safe and verified its literal generated
  workflow from an installed wheel.
- Added `cascade self-test` for a bounded installed-tree solve, VTK/CSV export,
  manifest/provenance check, and public-`svv` version check. With
  `--require-gpu`, it executes a real GL1 Cext and GPU tissue calculation.
- Removed arbitrary external Python solver loading from the production heart
  exporter. CASCADE and the frozen TissueSim oracle remain independent
  processes and exchange only explicit hashed files/results.
- Expanded CI/release smoke coverage to include high-signal static analysis,
  package dependency consistency, all five console entry points, public-SVV
  on-disk integrity, starter execution, installed self-tests, heart export, and
  VTK reopening.

## Package and compatibility conclusion

The wheel contains CASCADE's complete solver, adapter, domain, GUI, exporter,
settings, and built-in bivent3 code/data. It is a normal Python wheel, not a
single-file executable: declared third-party dependencies are installed by
`pip`. The exact dependency boundary is public `svv==0.0.48`; CASCADE does not
modify that distribution on disk. Adapter-scoped substitutions of public SVV
module globals occur only in the CASCADE process because those globals are how
SVV constructs trees and resolves two growth callbacks.

Existing-tree/forest/cache computation is M2/M3-qualified. True CCO remains
float64, the retained equal-bifurcation/simulation/export transition supports
float32, and the L-BFGS-B versus constrained SLSQP selector plus nearest-tree
growth remain explicitly outside the completed growth-qualification scope.

## User-visible validation boundary

Every ordinary production workflow is available through installed commands:
`cascade run`, `cascade sweep`, `cascade export-heart`, `cascade doctor`, and
`cascade self-test`, with separate GUI/viewer entry points. The complete M2/M3
oracle comparison is intentionally not a wheel-only command because it needs
the external `svva2` environment and multi-gigabyte private/hash-frozen
structures. Repository users can run that campaign through the retained
validation harness; installed-package users can run the bounded operational
self-test without repository files.

## Verification

- High-signal static checks: no syntax/undefined-name failures.
- Source suite: 135 tests passed; 15 warnings (14 third-party deprecations and
  the documented L-BFGS-B constraint warning).
- All 69 importable package modules loaded successfully (excluding executable
  `__main__` modules).
- CPU and CUDA 13 exact-wheel release smokes passed in independent temporary
  environments outside the checkout.
- CUDA diagnostics ran a compiled CuPy operation and cuFFT on the NVIDIA
  GeForce RTX 3080 Laptop GPU; installed GPU self-test ran vessel Cext and
  tissue kernels.
- Wheel and sdist passed `twine check`; archive contents, forbidden paths,
  checksums, and public-SVV byte integrity are recorded in the compact release
  evidence under `validation/results/` in the repository.

## Remaining limitations

- Tree-growth/optimizer equivalence and high-scale nearest-tree growth require
  their separate scientific/upstream campaign.
- CUDA 11 and CUDA 12 extras are packaged but not hardware-qualified.
- The supported release platform remains Linux/WSL2 x86-64 with Python 3.9;
  broader platforms are not claimed.
- Full human Studio queue/cancel/recovery/viewer acceptance remains open.
- Public release still requires the M4 license, remote, and observed-CI work.
