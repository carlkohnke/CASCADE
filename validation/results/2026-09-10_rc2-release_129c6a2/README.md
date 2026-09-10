# CASCADE 0.1.0rc2 release closure

Tracker IDs: M1, VAL-05, VAL-06, VAL-07, PERF-05, PERF-06, PERF-07.

Result: pass. The immutable source candidate is commit
`129c6a2c60b7718a38085f0d3ef92b2184385aff` on `release/cascade-0.1`, tagged
locally as `v0.1.0rc2`.
This result closes M2 and M3 for the owner-approved existing-tree computational
scope; it does not activate M4 publication or certify future tree growth.

## Exact artifacts

- `cascade_vascular-0.1.0rc2-py3-none-any.whl`: 901,059 bytes,
  SHA-256 `a0a3f2f7d56fc29d1b161242879677d52bd696ccd9c3f07f128ab07547ae9ba7`.
- `cascade_vascular-0.1.0rc2.tar.gz`: 923,695 bytes,
  SHA-256 `8ae23fd8405303bcdf15305ed61eec60fa0979e27093bbeeace5d23c1a08665f`.
- Both passed `twine check`; `dist/SHA256SUMS` verifies both RC1 and RC2
  artifacts.
- Archive audit: 77 wheel entries and 151 sdist entries. The packaged bivent3
  STL is present. Internal docs, validation data, legacy oracle/package trees,
  caches, backups, compiled Python files, and user-specific paths are absent.

## Verification commands

```text
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m build --outdir validation/tmp/dist-129c6a2
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m twine check validation/tmp/dist-129c6a2/*
/home/carl/.venvs/cascade-0.1.0rc1/bin/python scripts/release_smoke.py --wheel /home/carl/svv_sweeps/GFM/validation/tmp/dist-129c6a2/cascade_vascular-0.1.0rc2-py3-none-any.whl --constraints /home/carl/svv_sweeps/GFM/locks/requirements-py39-cpu.txt
/home/carl/.venvs/cascade-0.1.0rc1/bin/python scripts/release_smoke.py --wheel /home/carl/svv_sweeps/GFM/validation/tmp/dist-129c6a2/cascade_vascular-0.1.0rc2-py3-none-any.whl --constraints /home/carl/svv_sweeps/GFM/locks/requirements-py39-cu13.txt --gpu-extra gpu-cu13
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m pytest -q
(cd dist && sha256sum --check SHA256SUMS)
```

The CPU smoke passed version/doctor, GUI import, tree, explicit custom geometry,
forest save/cache reload, non-Cext heart tissue, VTK reopening, and manifest
checks. The CUDA smoke additionally passed compiled CuPy and cuFFT diagnostics
and a wheel-installed shared-global FFT/Cext heart solve with GPU tissue output.
The source suite passed 133 tests with 15 already classified warnings.

## Environment

- Python 3.9.20; WSL2 Ubuntu/Linux x86-64, kernel
  `6.18.33.2-microsoft-standard-WSL2`.
- Public `svv==0.0.48`, NumPy 1.26.4, SciPy 1.13.1, Numba 0.60.0.
- CUDA qualification: CuPy 13.6.0, `nvidia-cuda-runtime==13.2.86`,
  `nvidia-cufft==12.3.0.29`, `nvidia-nvjitlink==13.2.86`, NVIDIA GeForce RTX
  3080 Laptop GPU.
- CPU and CUDA smokes each used a newly created temporary environment outside
  the source checkout and ran sequentially.

## Retained preflight failure

The first RC2 CPU smoke was stopped by pip before installation because
`locks/requirements-py39-common.txt` still required
`cascade-vascular==0.1.0rc1`. Commit `129c6a2` changes only that release lock to
RC2. The artifacts were rebuilt afterward, and both exact-wheel smokes above
passed. This was a release-constraint mismatch, not a scientific or runtime
failure.

The numerical/performance evidence used to close M2/M3 remains in the linked
HEART-S, HEART-L, cube-scale, negative, reproducibility, and custom/export
campaign directories. Heavy outputs remain under ignored `validation/runs/`.
