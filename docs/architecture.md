# CASCADE architecture

CASCADE is organized by scientific responsibility. The command-line interface
and CASCADE Studio both call the same configuration, workflow, solver, and
export packages.

```text
JSON / CASCADE Studio
        |
        v
configuration -> workflows -> domain + vessels
                              |
                              v
                 flow -> vessel concentration
                              |
                              v
                    external field (Cext)
                              |
                              v
             tissue concentration (Green's Function Method)
                              |
                              v
                  exporting -> CSV / VTP / VTU / manifest
```

## Source layout

`src/` is the conventional Python packaging boundary: importing `cascade`
uses the installed package rather than accidentally importing files from the
repository root. The codebase itself begins at `src/cascade/`.

| Package | Responsibility |
| --- | --- |
| `cascade.configuration` | Validated run models, immutable runtime settings, and the temporary legacy-state bridge |
| `cascade.core` | Typed problem/result contracts and package infrastructure |
| `cascade.domain` | Domain construction, file-backed geometry, grids, and point sampling |
| `cascade.vessels` | Simple/lattice geometry, connectivity, growth, caching, target allocation, and vessel result contracts |
| `cascade.vessels.generation` | Public `svv` adapter and isolated compatibility implementations |
| `cascade.flow` | Topology, Kirchhoff pressure/flow solving, viscosity, rheology, and hematocrit |
| `cascade.concentration.vessel` | Intravascular top-down, network, Green's-function, and Graetz concentration solving |
| `cascade.concentration.external_field` | Direct, frozen, treecode, and hybrid-FFT Cext evaluation and coupling |
| `cascade.concentration.tissue` | Green's Function Method tissue oxygen, sampling caches, GPU execution, and viability metrics |
| `cascade.exporting` | Output schemas, tables, statistics, plots, VTK construction, and run manifests |
| `cascade.workflows` | Tree, sweep, simulation, and advanced whole-heart orchestration |
| `cascade.gui` | GUI pages, window coordination, preview geometry, rendering, workers, and widgets |
| `cascade.commands` | Thin command-line parsing and dispatch |
| `cascade.accelerators` | GPU availability and backend selection |
| `cascade.diagnostics` | Installation and runtime diagnostics |
| `cascade.validation` | Bounded installed-package smoke validation |
| `cascade.utils` | Execution guards and packaged-resource resolution |
| `cascade.assets` | Packaged domains, vascular seeds, and immutable numerical tables |

The package root intentionally contains only `cascade/__init__.py`; application
modules live in the package that owns their behavior.

Within `cascade.concentration.vessel`, `network.py` owns the arbitrary-network
node-mixing solver and `topdown.py` owns the tree-specialized solver. Workflow
modules may call or temporarily re-export these functions, but do not contain
their numerical implementations.

Within `cascade.flow`, `kirchhoff.py` owns backend-independent orchestration
and the tree-specialized solve, while `linear_system.py` owns sparse/dense
matrix assembly, conditioning diagnostics, and linear-solver dispatch.

## Numerical boundaries

The solver passes explicit typed inputs and results at its public boundaries:
`FlowProblem`, `FlowResult`, `VesselConcentrationProblem`,
`VesselConcentrationResult`, `TissueOxygenProblem`, and `TissueOxygenResult`.
These contracts validate array shape, topology, and physical scalar inputs
before numerical work begins.

The finite-radius Green's-function kernels load a versioned Bessel table from
`cascade.assets.numerics`. Its `K1/K0` values are generated with exponentially
scaled Bessel functions so the ratio remains stable at large arguments. The
Graetz closure likewise loads precomputed 8-radial-node, 4-mode bases for plug
and Poiseuille profiles over `1e-8 <= Bi <= 1e6`, with 128 samples per decade.
These are fixed release assets, not per-run caches or user-selectable
discretizations; corrupt or missing assets fail immediately with a diagnostic.

The former TissueSim implementation is decomposed across the flow,
concentration, domain, exporting, workflow, and diagnostic packages.
`cascade.runtime.tissuesim` is now a small compatibility facade for existing
callers and frozen validation workflows. It is not the owner of the numerical
implementation. Its mutable state is isolated in
`cascade.configuration._legacy_state`; new application code should use typed
configuration and subsystem entry points. The facade and state bridge are
transitional and should be removed after downstream callers have migrated and
the final parity campaign passes.

The advanced whole-heart path is likewise separated into domain/connectivity,
flow preparation, whole-forest Cext, output construction, and CLI orchestration
modules under `cascade.workflows`. Normal tree, forest, channel, lattice, and
custom-network exports use `cascade run`.

## Native source boundary

CASCADE keeps code in the language and subsystem where it executes:

- Python coordinates configuration, vascular models, simulation, CLI, and GUI behavior.
- CUDA C/C++ implements GPU Cext, treecode, FFT-correction, and tissue kernels compiled at runtime by CuPy/NVRTC.
- GLSL vertex and fragment shaders implement the retained OpenGL viewport pipeline.
- C# implements the Windows common-dialog interop class loaded by the WSL PowerShell bridge.

CUDA, GLSL, and C# sources are package resources included in wheels and source
distributions. CPU-only imports do not read CUDA resources until a GPU kernel is
requested.

## Dependency direction

End-to-end single-tree orchestration lives in
`cascade.workflows.tree_simulation`. The legacy `summarize_tree` name remains
as a compatibility alias; new code uses `run_tree_simulation`, while
`cascade.workflows.summaries` contains only compatibility imports.

Scientific packages do not import GUI or command modules. Commands and GUI
pages depend on workflows; workflows compose domain, vessel, flow,
concentration, and export services. Public `svv` adaptation is confined to
`cascade.vessels.generation` so it can be replaced without changing solver or
interface code.

Scientific modules import their dependencies explicitly. Concentration package
initializers expose the stable API lazily, which prevents low-level numerical
imports from loading unrelated solver backends or creating initialization
cycles. The TissueSim facade re-exports completed implementations for legacy
callers but no longer injects names into implementation modules.

## Reproducibility and memory

A run is identified by its settings file, input hashes, CASCADE version/commit,
dependency versions, random seed, execution environment, and generated
manifest. Relative inputs resolve from the settings file, not from the caller's
current directory.

The GUI launches simulations as child processes. It stores settings and
lightweight result metadata but does not retain full solver arrays after a
worker exits. Large output tables and VTK files can be disabled when only
summary results are needed.

## Validation boundary

The public package contains a bounded installed-package self-test. Detailed
numerical-equivalence fixtures, frozen legacy oracles, performance campaigns,
and internal regression tests live in the separate CASCADE workbench and are
not shipped in source or binary releases.
