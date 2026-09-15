# Native Windows qualification record

Status: **automated qualification complete — physical UI and hosted CI evidence
remain outstanding**

This record distinguishes artifact availability from behavior that has been
executed successfully. A blank or pending result must not be interpreted as a
pass.

## Qualification target

| Item | Value |
| --- | --- |
| Host | Windows 11 x86-64, build 22621 |
| Python | CPython 3.12.14, isolated portable runtime |
| CASCADE baseline | `deea5e525383613e36a8b79d5e9e3d68b83c7658` |
| CASCADE candidate | `0.1.0rc5`, final exact artifact from `b5b282439cefddc37be60117e62d3c1cdf357964` |
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
| 6 — Studio qualification | Partial | Exact installed wheel: 76/76 native tests pass; persistent worker reuse, cancellation/recovery, preview recovery, launch subsystems, native OpenGL/software rendering, and 125% scaling pass; remaining physical UI checks are recorded below |
| 7 — native CUDA qualification | Pass | Exact installed wheel: `doctor --require-gpu`, `self-test --require-gpu`, and 78/78 tests pass; all CUDA kernel families, CPU/GPU comparisons, repeated persistent-worker jobs, cancellation/recovery, and memory stability are covered |
| 8 — Linux regression | Pass | CPython 3.12 isolated environment: dependency integrity and static checks pass; 65 tests pass with 13 intentional Windows/GPU skips; installed-package CPU self-test passes and scientific outputs match Windows |
| 9 — clean install | Pass | Fresh CPU and CUDA environments installed the exact wheel and declared dependencies as binary wheels; version, doctor, self-test, Studio launch/project/run/cancel/recovery, exports, and GPU acceleration pass without source, compiler, WSL, PATH, or DLL workarounds |
| 10 — user tooling/docs | Pass | Native CPU/CUDA wheel installation, console-free and diagnostic launchers, CLI use, state paths, troubleshooting, uninstall, and support boundaries are documented and validated against the clean-install workflow |
| 11 — CI | Implemented, not hosted | Ubuntu 22.04 and Windows Python 3.12 installed-wheel CPU lanes plus a self-hosted native Windows/NVIDIA release gate are defined and locally validated; no remote is configured, so no GitHub run is claimed |

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
- The source tree's historical `.bat` and `.vbs` launchers are WSL development
  helpers. Native wheel installations now expose a GUI-subsystem Studio
  launcher and a separate console-subsystem diagnostic launcher.
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
- A deferred queue callback could clear a newly active Studio job; queue
  scheduling is now guarded against stale callbacks and cancellation is
  followed by a successful job in the native lifecycle suite.
- New-project display and timestamp formatting contained Windows-specific
  failures: the UI attempted to stat an unsaved project and used POSIX-only
  `strftime` directives. Both paths are now platform-neutral.
- Studio's native OpenGL selector imported the renderer from a nonexistent
  subpackage and silently fell back to software. The corrected import is
  covered by a regression test and the native Windows renderer now initializes.
- GPU validation previously replaced the real CUDA timing record with its
  internal CPU reference timings, causing diagnostics to report CPU execution
  after a successful GPU run. The GPU timing record is now preserved and the
  regression is covered by the native kernel-family qualification test.

## Known limitations and unperformed checks

- Native automated and programmatic Studio checks pass, including real Windows
  OpenGL and software rendering, but the available computer-control surface did
  not expose native applications. Open/Save/folder dialogs, title-bar dragging,
  Windows Snap, visible Explorer/result-viewer behavior, and the visible
  second-instance notice were therefore not manually verified.
- The host exposed one display at 125% scaling. Native 100%, 150%, and 200%
  scaling and multiple-monitor movement remain unperformed physical checks.
- Windows WDDM did not expose per-process VRAM counters for the qualification
  worker. Device-level VRAM was measured instead and remained flat across the
  repeated persistent-worker jobs.
- The CI workflows have not executed on GitHub because this repository has no
  remote configured. The GPU workflow additionally requires a real self-hosted
  Windows x64 runner labeled `cascade-gpu`; no such runner is fabricated here.

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

## Stage 6 native Studio qualification

The exact wheel built from commit `6fa6a42b78af915cdd925e4562897691906cb4a1`
has SHA-256
`1542ed556743d0b83bfe71601d5f860ca5e88ae6760b546d94bc6f70597c74f9`.
It was force-installed into the native CPU environment and imported from that
environment's `site-packages` in a fresh NTFS test snapshot. After correcting a
qualification-harness omission of the standalone `setup_env.py` fixture, the
complete installed-wheel suite passed 76 tests in 110.70 seconds. The initial
74-pass/2-fail result and successful rerun are both retained in sandbox logs;
the two failures were missing-fixture errors, not CASCADE runtime failures.

Automated native Windows coverage now constructs every primary Studio page;
creates, saves, and reopens a project; restores interrupted queue state; loads
exported result arrays; starts, reuses, stops, and restarts the persistent
`QProcess` worker; cancels a running job and completes the following job; and
cancels and recovers the preview worker. Native Qt dialog selection is tested
without invoking WSL, and PE headers verify that `cascade-gui.exe` is a GUI
subsystem launcher while `cascade-gui-console.exe` retains a diagnostic console.

Separate real Windows-platform probes—not Qt's offscreen plugin—initialized
both the software renderer and the OpenGL instanced renderer. OpenGL preflight
reported 3.3 support, the widget became valid, and the active renderer was
Intel UHD Graphics. The normal auto-selected GUI path used OpenGL, ran at the
host's native 125% scale factor, exercised minimize/maximize/restore and resize
state transitions, started and stopped its worker, closed cleanly, and reported
no attached console window. An actual normal GUI launcher process was also
started and its exact sandbox process tree was terminated for cleanup; no
sandbox-environment child process remained.

Physical interaction checks that could not be automated are explicitly left
open in the limitations above. Stage 6 is therefore recorded as partial rather
than silently treating those manual checks as passed.

## Stage 7 native CUDA qualification

The exact wheel built from commit `855989981a8be684e0ff5cae5455fa0538465a14`
has SHA-256
`c5038a6d3b375c59f601562681cf371b500ba4a1c0d51e5f1c985e2bfea8a1ae`.
It was installed into the isolated native Windows CUDA environment and imported
from that environment's `site-packages` with `PYTHONPATH` and `CUDA_PATH`
removed. Wheel-provided CUDA components and every runtime cache remained in the
Windows sandbox. `pip check` passed.

The device was an NVIDIA GeForce RTX 3080 Laptop GPU with compute capability
8.6 and 16 GiB of memory. CuPy reported driver API 13.4 and CUDA runtime 13.2.
`cascade doctor --require-gpu` passed after a cold toolkit/bootstrap probe, and
`cascade self-test --require-gpu` completed in 18.58 seconds with its outputs
retained in the sandbox. The complete installed-wheel suite, including the
opt-in GPU qualification cases, passed 78 tests in 283.85 seconds.

The qualification test forces execution through the direct Cext solver, the
frozen/top-down solver, treecode, pure FFT deposition/correction/sampling,
hybrid FFT with local correction, dense fused tissue kernels, and sparse
cell-list/KD-tree tissue kernels. It checks pressure, flow, vessel
concentration, tissue concentration, viability/statistics, and VTK exports
against CPU references or equivalent CUDA formulations. Export paths were
tested in both float32 and float64; CASCADE's CUDA Cext working arrays remain
float32 by design, so this record does not claim float64 CUDA arithmetic.

For the deterministic five-segment qualification tree, the largest observed
relative vessel-concentration error was approximately `1.58e-7` for direct GPU
versus CPU and `8.59e-6` for hybrid-local versus direct GPU. Top-down tissue
concentration differed from the CPU reference by approximately `6.33e-6` in
dense mode and `8.37e-6` in sparse mode. Pressure and flow satisfied the tighter
solver tolerances, tissue masks matched, and exported VTP/VTU data reopened
successfully with PyVista.

Three consecutive GPU jobs in one persistent Studio worker held aggregate
worker-process-tree RAM at 677.9 MiB and device VRAM at 145 MiB for all three
runs. A job was then cancelled after entering CUDA coupling, all recorded
worker and descendant PIDs exited, and a subsequent GPU job completed in a
fresh worker. Final shutdown removed that worker and its descendants as well.
This covers stable repeated execution, cancellation without poisoning the next
run, and clean CUDA worker teardown on the qualification host.

## Stage 8 Linux regression

Linux regression was run from the source repository at commit
`331be35f55030ab3870f26b03aa1c149254749cb` using CPython 3.12.3 in an isolated,
repository-local environment. The environment, package downloads, temporary
files, caches, configuration, state, logs, and test reports remained beneath
the ignored `.venv` directory. No system Python packages or user configuration
were changed.

`pip check` reported no broken requirements, and the fatal/static Ruff
selection passed across `src`, `setup_env.py`, and `tests`. The full suite
completed with 65 passes and 13 intentional skips in 56.40 seconds. Skips were
limited to tests requiring native Windows DLL, PE, Job Object, or QProcess
semantics and the explicitly opt-in dedicated-CUDA qualification cases.

The Linux environment installed a non-editable wheel built from the repository;
`cascade.__file__` resolved from its environment's `site-packages`. Its CPU
`cascade self-test` completed in 5.18 seconds with all evidence retained inside
the repository-local environment. The Windows and Linux self-test `segments.csv`
and `points.csv` files were byte-for-byte identical. All 137 non-timing
scientific fields in `summary.csv` also matched; only expected elapsed/timing
measurements differed. These results provide direct regression evidence that
the Windows filesystem, process, GUI, packaging, and CUDA changes did not alter
the existing Linux scientific path.

## Stage 9 clean-install qualification

The exact wheel built from commit `623901534a1fbb6ca51d21462276d594f22ac9db`
has SHA-256
`1c5152f221548caaa26fdbcee9f9ec63acbc7234206b484d07eed2b1ae6d5345`.
The corresponding sdist has SHA-256
`dc2edfab425ca1afbf97e1ff039c2d0f6f86603dbb984e537f0a1232bd2dcdb6`.
Both were built from a fresh Git archive copied to native NTFS rather than from
the WSL checkout.

A fresh CPU environment installed the exact wheel with the GUI extra using
only binary Windows wheels. It contained no editable install or source-tree
path, passed `pip check`, imported CASCADE from `site-packages`, reported the
expected version, passed both doctor modes, and completed the installed-package
CPU self-test in 9.81 seconds. The normal GUI-subsystem launcher stayed running
without a startup error or stderr output and left no process behind when closed.

In that environment, Studio created and saved a project whose path and name
contained Unicode, reopened its materialized portable project configuration,
and completed a scientific CPU run through its persistent worker. The run
produced CSV, VTP, and VTU outputs; PyVista reopened the exports and verified
the pressure, flow, concentration, tissue concentration, and viability arrays.
A subsequent job was cancelled, its worker exited, a recovery job completed,
and final shutdown left no worker process behind.

A separate fresh CUDA environment installed the same exact wheel with
`[gui,gpu-cu13]` from normal package indexes using `--only-binary=:all:`. Pip
resolved CuPy and the complete wheel-provided CUDA 13 toolkit without a compiler
or system CUDA installation, and `pip check` passed. With inherited
`PYTHONPATH`, `CUDA_PATH`, and source access absent, `doctor --require-gpu`
identified the RTX 3080 Laptop GPU and the installed-package GPU self-test
completed in 14.73 seconds.

Two earlier CUDA environment attempts are retained as failed qualification
fixtures. They stopped before installing CASCADE because the local GPU
wheelhouse was only a CUDA delta and did not contain the common dependency or
`cuda-toolkit` metapackage wheels. The normal release-user installation path
resolved those declared dependencies correctly; no package metadata or runtime
workaround was needed.

## Stage 10 Windows tooling and documentation

The native Windows guide at `docs/windows.md` records the clean, released-wheel
workflow for CPU and CUDA 13 environments. Its commands select CPython 3.12
explicitly, keep the environment isolated, require no global `PATH` change, and
invoke the environment's executables directly. The documented CUDA path relies
only on the NVIDIA driver and wheel-provided toolkit; it explicitly rejects
manual DLL copying and a system CUDA installation as setup steps.

The wheel-generated `cascade-gui.exe` is the normal GUI-subsystem launcher and
`cascade-gui-console.exe` is the diagnostic console launcher. Both resolve the
Python environment that installed them and neither invokes WSL. The guide also
documents the persistent Studio error log, software-renderer fallback,
platform-native configuration/cache/state/log locations, environment-specific
overrides, safe uninstall behavior, common NVIDIA failures, and the precise
supported/unsupported platform boundary. Historical source-tree `.bat` and
`.vbs` wrappers are clearly labeled as legacy WSL development helpers.

README, Studio, and known-issues documentation now direct Windows users to the
native workflow and no longer describe WSL as the supported Windows runtime.
Local Markdown links were checked, fatal/static checks passed, and a fresh
sdist/wheel build succeeded. The sdist allowlist was updated and its archive
was inspected to confirm that the new Windows guide is included.

## Stage 11 CI support

The primary workflow now runs a Python 3.12 matrix on Ubuntu 22.04 and
`windows-latest`. Each lane builds the wheel and sdist, checks both artifacts,
installs the wheel with its declared GUI and test extras, proves that CASCADE is
imported from the active environment's `site-packages`, runs fatal/static
checks and the complete test suite, and executes version, doctor, and
installed-package self-test smoke checks. Build products are retained as CI
artifacts. Runtime state, logs, caches, and self-test outputs are directed to
the runner's temporary directory.

The separate Windows GPU workflow is both manually dispatchable and reusable.
It requires the labels `self-hosted`, `Windows`, `X64`, and `cascade-gpu`,
installs the exact built wheel with `[dev,gui,gpu-cu13]` using binary artifacts,
runs required-GPU diagnostics and self-test, enables the opt-in CUDA
qualification suite, and uploads release-gate evidence even on failure. It does
not imply that a GitHub-hosted GPU runner exists.

Both workflow files passed local YAML syntax and structural checks. Their build,
install, isolation, CPU test, CUDA diagnostic, CUDA kernel-family, and
installed-package self-test operations were exercised in the preceding local
Linux and native Windows stages. With no Git remote configured, neither
workflow has been submitted to or executed by GitHub; Stage 11 is therefore
recorded as implemented rather than as a hosted CI pass.

## Final exact-candidate gate

The final wheel built from commit `b5b282439cefddc37be60117e62d3c1cdf357964`
has SHA-256
`d8ca856337fa4630f16cb85fba95fa13e30d1cac5db4754639277e1c60c39519`.
The matching sdist has SHA-256
`1be5ebb5c38055b15eb96e1bdbdac001f27ade1164ec194b050670816a9391bd`.
Both artifacts passed Twine checks.

The exact wheel was installed over the previously clean native Windows CPU and
CUDA environments with no dependency or source-tree substitution. The CPU
environment passed `pip check`, fatal/static checks, and the complete native
CPU suite: 76 passes and two intentional dedicated-GPU skips in 116.65 seconds.
The CUDA environment passed `pip check`, `doctor --require-gpu`, the
installed-package GPU self-test in 12.76 seconds, and the complete suite with
the GPU qualification flag: 78 passes in 273.12 seconds. Both imports resolved
from their environment's `site-packages`; `PYTHONPATH` and `CUDA_PATH` were
absent, and all caches, logs, temporary data, and evidence remained in the
Windows sandbox.

This closes all automatable local gates for the current candidate. It does not
convert the explicitly listed physical UI checks or unexecuted hosted CI jobs
into passes.
