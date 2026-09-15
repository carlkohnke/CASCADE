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
| CASCADE candidate | `0.1.0rc5`; Stage 1 checkpoint recorded in repository history |
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
| 1 — packaging/dependencies | Pass | Python 3.12 Windows CPU/GUI/dev graph: 163 wheels, 0 sdists; CUDA 13 graph: 168 wheels, 0 sdists; wheel/sdist build, metadata, resource, entry-point, and Twine checks pass |
| 2 — isolated installs | Pending | — |
| 3 — compatibility fixes | Pending | — |
| 4 — native CPU qualification | Pending | — |
| 5 — filesystem torture | Pending | — |
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
| NVIDIA runtime / cuFFT / nvJitLink | 13.2.86 / 12.3.0.29 / 13.2.86 | `win_amd64` wheels |

SVV 0.0.48 declares `trimesh[all]`, PySide6, psutil, PyVistaQt, and a broad
scientific/desktop dependency set. Consequently the normal dependency graph
is large. Qualification intentionally honors that metadata; it does not use
`--no-deps` or omit declared packages.

## Significant issues found

- The previous package metadata rejected Python 3.12 and advertised no native
  Windows support.
- Existing Windows launchers are WSL wrappers rather than native launchers.
- CUDA configuration and cache locations are Linux-biased, and CuPy can be
  imported before Windows DLL directories are configured.
- Some runtime caches can resolve to POSIX home paths or package-adjacent
  locations unsuitable for a read-only wheel installation.
- Persistent-worker memory pressure uses `/proc` and is ineffective on native
  Windows.
- Native Windows process-tree cancellation, mmap cleanup, queue recovery,
  path edge cases, GUI lifecycle, and CUDA kernel families lack qualification.

## Known limitations and unperformed checks

- CPU scientific workflows have not yet been executed from the installed
  wheel.
- Studio has not yet been launched from the installed wheel.
- CUDA has not yet been imported or exercised in the candidate environment.
- No clean-install acceptance run has yet been performed.
- Windows CI workflow execution and final Linux regression are pending.
