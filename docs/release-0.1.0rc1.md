# CASCADE 0.1.0rc1 release-candidate report

Validation date: 2026-09-09

## Step 1 result

The repository is installable as the `cascade-vascular` Python distribution and exposes the CLI, CASCADE Studio GUI, viewer, diagnostic command, and packaged heart exporter. Both CPU-only and CUDA 13 wheel installations completed from temporary environments outside the source checkout.

Validated platform:

- WSL2 Ubuntu, Linux x86-64
- Python 3.9.20
- Public `svv==0.0.48`
- NumPy 1.26.4, SciPy 1.13.1, Numba 0.60.0, PyVista 0.46.5, VTK 9.5.2
- CUDA configuration: CuPy 13.6.0 and NVIDIA CUDA runtime 13.2.86
- GPU: NVIDIA GeForce RTX 3080 Laptop GPU

## Passed gates

- 100 pytest tests passed against the clean public-`svv` environment.
- The CPU wheel smoke install passed from a temporary virtual environment.
- The CUDA 13 wheel smoke install passed from a separate temporary virtual environment.
- `cascade doctor --require-gpu` compiled and executed a CuPy kernel and cuFFT transform from the wheel-only environment with no source checkout on `sys.path`.
- The wheel-only smoke tests created, solved, and exported a vascular tree and a custom CSV Y-channel network.
- Tree, two-tree forest, nearest-tree forest, lattice, built-in channel, and custom-geometry examples executed through `cascade run`.
- `cascade export-heart` loaded a CASCADE `.forest.simcache` through public `svv`, ran geometry-only and CPU tissue solves, and wrote ParaView outputs.
- The packaged shared-global Cext heart workflow completed a small CUDA/FFT solve and tissue export.
- 27 generated VTP/VTU files reopened successfully with PyVista and contained points.
- CASCADE Studio constructed, displayed on the Qt offscreen platform, entered its event loop, and exited cleanly.
- Wheel and source distributions built successfully and passed `twine check`.

The exact dependency constraints are under `locks/`. Artifact hashes are written to `dist/SHA256SUMS` after the final build.

## Known release-candidate limitations

- Full numerical equivalence and performance parity against all selected TissueSim reference cases are Step 2 gates and are not claimed by this report.
- The legacy L-BFGS-B bifurcation path emits a SciPy warning because L-BFGS-B does not enforce the legacy general inequality constraint. This is retained for historical parity and documented in `docs/svv-compatibility.md`.
- Cross-version `.dmn` interchange with unwrapped public `svv` needs a dedicated round-trip decision and validation.
- The dependency locks are platform-specific: Python 3.9 on Linux x86-64. CUDA 11/12 extras remain available but were not validated on this workstation.
- The release candidate does not yet have a configured Git remote.
- No project license has been selected. A public repository should not be advertised for third-party reuse until the owner chooses a license and confirms compatibility with derived `svv` components. A private GitHub repository is not blocked by this packaging issue.

## Public `svv` boundary

CASCADE owns orchestration, strict input validation, cache detection/loading, connectivity repair, simulation settings, VTK export, custom geometry, and public-version compatibility wrappers. Upstream requests are reserved for changes that must run within `svv` growth/optimization loops, unavailable inner-loop primitives needed for performance, or shared interchange contracts.
