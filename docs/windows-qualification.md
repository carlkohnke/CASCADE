# Native Windows qualification record

Status: **in progress — not release-qualified**

This record distinguishes artifact availability from behavior that has been
executed successfully. A blank or pending result must not be interpreted as a
pass.

## Qualification target

| Item | Value |
| --- | --- |
| Host | Windows 11 x86-64, build 22621 |
| Python | CPython 3.12.14, isolated portable runtime |
| CASCADE baseline | `deea5e525383613e36a8b79d5e9e3d68b83c7658` |
| CASCADE candidate | `0.1.0rc5`, filesystem qualification through `363e7f7aa0f95b9349cec09830d2ee0654a1db79` |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB |
| Driver observed | 616.92 |
| Test isolation | Native NTFS sandbox; no application tests from the WSL checkout |

The portable qualification runtime, virtual environments, downloads, caches,
logs, and generated test data are outside the repository in the dedicated
Windows sandbox. No global PATH, registry, driver, system CUDA, Windows
feature, WSL, or unrelated Python installation was changed.

## Stage results

| Stage | Result | Evidence |
| --- | --- | --- |
| 0 — baseline audit | Pass | Clean `main` at the recorded baseline; five local branches retained; no remote configured |
| 1 — packaging/dependencies | Pass | Python 3.12 Windows CPU/GUI/dev graph: 163 wheels, 0 sdists; corrected CUDA 13 toolkit graph: 174 wheels, 0 sdists; wheel/sdist build, metadata, resource, entry-point, and Twine checks pass |
| 2 — isolated installs | Pass | Separate CPU and CUDA Python 3.12 virtual environments installed the wheel from native NTFS with binary-only constraints; both pass `pip check` and import CASCADE from their own `site-packages` |
| 3 — compatibility fixes | Pass | Exact wheel installed on native NTFS; 18/18 focused filesystem, CUDA-bootstrap, host-memory, process-tree, atomic-state, and Studio lifecycle tests pass |
| 4 — native CPU qualification | Pass | Exact installed wheel: fatal/static check passes and 62/62 tests pass; all documented CLI commands and the CPU self-test complete from native NTFS |
| 5 — filesystem torture | Pass | Exact installed wheel: 66/66 tests pass from an NTFS-only test snapshot; deep space/Unicode paths, stale caches, mmap release, repeat writes, lock contention/recovery, and a read-only installed package pass |
| 6 — Studio qualification | Pending | — |
| 7 — native CUDA qualification | Pending | Artifact resolution only; no CUDA execution claimed |
| 8 — Linux regression | Pending | — |
| 9 — clean install | Pending | — |
| 10 — user tooling/docs | Pending | — |
| 11 — CI | Pending | — |

## Windows dependency evidence

Binary-only resolution succeeded for the declared CASCADE graph on CPython
3.12/Windows x86-64. The principal resolved artifacts are:

| Dependency | Resolved version | Windows artifact |
| --- | ---: | --- |
| SVV | 0.0.48 | `py3-none-win_amd64` wheel |
| NumPy | 2.5.3 | `cp312-win_amd64` wheel |
| SciPy | 1.18.1 | `cp312-win_amd64` wheel |
| Numba / llvmlite | 0.67.0 / 0.49.0 | `cp312-win_amd64` wheels |
| PyVista / VTK | 0.49.0 / 9.7.0 | pure Python / `cp312-win_amd64` wheels |
| Open3D | 0.19.0 | `cp312-win_amd64` wheel |
| TetGen | 0.8.4 | `cp312-abi3-win_amd64` wheel |
| PyMeshFix | 0.17.0 | `cp312-win_amd64` wheel |
| USearch | 2.26.2 | `cp312-win_amd64` wheel |
| PySide6 | 6.11.2 | `cp310-abi3-win_amd64` wheels |
| CuPy CUDA 13 | 14.2.0 | `cp312-win_amd64` wheel |
| NVIDIA runtime / NVRTC | 13.4.49 / 13.4.59 | `win_amd64` wheels |
| NVIDIA cuBLAS / cuFFT | 13.7.0.27 / 12.4.0.34 | `win_amd64` wheels |
| NVIDIA cuRAND / cuSOLVER | 10.4.4.49 / 12.3.2.15 | `win_amd64` wheels |
| NVIDIA cuSPARSE / nvJitLink | 12.8.6.49 / 13.4.52 | `win_amd64` wheels |

SVV 0.0.48 declares `trimesh[all]`, PySide6, psutil, PyVistaQt, and a broad
scientific/desktop dependency set. Consequently the normal dependency graph
is large. Qualification intentionally honors that metadata; it does not use
`--no-deps` or omit declared packages.

## Significant issues found

- The previous package metadata rejected Python 3.12 and advertised no native
  Windows support.
- Existing Windows launchers are WSL wrappers rather than native launchers.
- The initial GPU extra omitted NVRTC and other required toolkit components;
  it now declares CuPy's complete wheel-provided CUDA toolkit extra.
- CUDA bootstrap now discovers wheel-provided headers and nested DLLs before
  CuPy is imported, retaining Windows DLL-directory handles for process life.
- Runtime caches, configuration, state, logs, and Studio project defaults now
  use deterministic platform-native user-writable directories.
- Persistent-worker memory pressure now uses a cross-platform host-memory
  provider rather than relying on Linux `/proc`.
- Studio and TetGen workers are assigned to Windows Job Objects so cancellation
  and shutdown terminate descendants rather than leaving orphan processes.
- Queue, project, cache-manifest, and combined sweep state is published with
  same-directory atomic replacement and bounded Windows sharing-violation
  retries. A native concurrent-writer test reproduced and now covers this case.
- Studio now has a per-user native instance lock and records unhandled GUI
  exceptions outside the installed package for console-free launches.
- Redirected Windows command output now escapes only characters unsupported by
  the active console encoding instead of crashing on a valid Unicode path.
- The Windows simulation lock now reserves a byte solely for locking, keeps
  holder metadata readable during contention, and tolerates the brief release
  delay observed after forced process termination.
- Queue recovery, complete GUI lifecycle, and CUDA kernel families still
  require their later qualification stages.

## Known limitations and unperformed checks

- Native interactive Studio behavior has not yet been qualified; only
  installed-wheel imports, page construction, and offscreen startup/shutdown
  have passed.
- CUDA has not yet been imported or exercised in the candidate environment.
- No clean-install acceptance run has yet been performed.
- Windows CI workflow execution and final Linux regression are pending.

## Stage 2 installed-wheel isolation

The Stage 1 wheel (`SHA-256
 cce75d0d98ad3d10b0e17485a5be9367e99ddfe2a60f8bef12c1c15cd3703cba`)
was installed into independent CPU and CUDA virtual environments. Checks were
run from an unrelated NTFS directory with `PYTHONPATH` removed. In both
environments:

- `cascade.__file__` resolves beneath the environment's `site-packages`;
- CASCADE reports version `0.1.0rc5` on CPython 3.12.14;
- no `.pth` file refers to the WSL checkout;
- `pip check` reports no broken requirements.

The installed `cascade-gui.exe` has the Windows GUI PE subsystem, while
`cascade-gui-console.exe` has the console subsystem. The CPU environment also
passed CLI version/help and `cascade doctor --no-gpu-probe` smoke checks. The
CUDA environment's distributions were verified from metadata without importing
CuPy or creating a CUDA context; execution qualification remains Stage 7.

## Stage 3 compatibility qualification

The exact wheel built from commit `eeda05e0abbed451a41ac7309d89f9a116404b94`
has SHA-256
`76312cb98de8acfa93eddb519748cabbeef31994228d96ea14ebb4df955c93cf2`.
It was force-installed into the native Windows CPU environment and tested from
an unrelated NTFS working directory. CASCADE resolved from that environment's
`site-packages`; no source checkout was on the import path.

The focused suite passed 18 tests covering runtime directories, CUDA toolkit
discovery and pre-import setup, Windows host-memory reporting, recursive
process termination, Qt worker Job Object attachment, atomic state publishing,
concurrent replacement, Studio instance locking, and persistent Studio error
logging. The same wheel was installed into the CUDA environment, where all
declared requirements pass `pip check`; wheel-provided CUDA headers and the
NVRTC DLL were found without importing CuPy or creating a CUDA context.

## Stage 4 native CPU qualification

The exact wheel built from commit `7cc16555b0afb1939f77b64c3797f6b1d9b5f0cb`
has SHA-256
`7d122d06d1f25c0ae621265dad1a71339b46ee12b8f5abefc81d5591bc95dafa`.
It was installed into the native CPU environment and executed from a separate
NTFS working directory. The repository's fatal/static Ruff selection passed,
and the complete installed-wheel suite passed 62 tests in 80.07 seconds.

The command-level qualification covered version and help output,
`init-settings`, both `doctor` modes, `self-test`, `inspect`, `prepare`, `run`,
`batch`, and `sweep`. Scientific cases covered cube, sphere, and uploaded STL
domains; TetGen volume construction; custom CSV and NPZ networks; generated,
saved, and loaded vascular trees; generated, saved, and loaded legacy forests
and simulation caches; and a prepared memory-mapped tree. Both flow-driven and
pressure-pressure boundary modes completed, as did custom constant-density and
viscosity fluid settings. CSV, VTP, and VTU outputs were reopened with PyVista;
required geometry and pressure, flow, oxygen, and viability arrays were
verified.

The installed-wheel CPU self-test completed in 13.63 seconds and retained its
manifest, command log, saved tree, CSV tables, and VTK-family artifacts in the
Windows sandbox. Native offscreen Qt constructed the Studio window and all
primary pages, then shut down cleanly with exit code zero. Interactive GUI
qualification remains Stage 6.

Stage 4 found and fixed three non-platform-specific release blockers: generated
projects forced GPU despite the documented CPU-safe contract; network archive
inspection and preparation used a NumPy private header function removed in
NumPy 2.5; and simple-network pressure runs omitted the aggregate pressure-drop
field. Regression tests cover each fix.

## Stage 5 Windows filesystem qualification

The exact wheel built from commit `363e7f7aa0f95b9349cec09830d2ee0654a1db79`
has SHA-256
`4bc70c6cdf013408d386ce045f6d79cf97b4204fc65db639df169d00926b972a`.
The test sources and their standalone setup helper were copied to a fresh NTFS
sandbox snapshot; CASCADE imported from the CPU environment's `site-packages`,
not from that snapshot or the WSL repository. All 66 tests passed in 152.88
seconds.

The Windows-specific cases exercised a path longer than 200 characters with
spaces and the Unicode components `血管` and `Ω`; absolute and relative settings
and input paths; execution from different current working directories; stale
uploaded-STL domain-cache detection and rebuild; repeated replacement of an
existing result and manifest; release of prepared-tree memory maps followed by
an immediate directory rename; and cross-process simulation-lock contention,
forced owner termination, and immediate recovery. The broader suite retained
the atomic concurrent queue/state writers and process-tree interruption tests.

A second, disposable native environment installed the same wheel and its
declared dependency graph. Its installed `cascade` package was ACL-restricted
to read and execute access; an attempted write failed with
`UnauthorizedAccessException`. Version reporting, JSON doctor without a GPU
probe, and the installed-package CPU self-test then passed from an external
NTFS project directory. The package contained 244 files, and its aggregate
content digest was unchanged before and after the run. The self-test completed
in 39.594 seconds with all outputs, logs, temporary files, bytecode, and CASCADE
state redirected beneath the Windows sandbox.
