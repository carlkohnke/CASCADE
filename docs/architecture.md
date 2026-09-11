# CASCADE architecture

CASCADE is organized by scientific responsibility. The command-line interface
and CASCADE Studio both call the same configuration, workflow, solver, and
export packages.

```text
JSON / CASCADE Studio
        |
        v
configuration -> simulation -> domain + vessels
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
| `cascade.configuration` | Typed run models, parsing/validation, examples, immutable runtime configuration, nested solver-setting sections, and the temporary legacy-state bridge |
| `cascade.domain` | Domain construction, SVV-compatible geometry implementation, file-backed geometry, grids, and point sampling |
| `cascade.vessels` | Simple/lattice geometry, connectivity, growth, caching, target allocation, and vessel result contracts |
| `cascade.vessels.generation` | Public `svv` adapter and isolated compatibility implementations |
| `cascade.flow` | Topology, Kirchhoff pressure/flow solving, viscosity, rheology, and hematocrit |
| `cascade.concentration.vessel` | Intravascular top-down, network, Green's-function, and Graetz concentration solving |
| `cascade.concentration.external_field` | Direct, frozen, treecode, and hybrid-FFT Cext evaluation and coupling |
| `cascade.concentration.tissue` | Green's Function Method tissue oxygen, sampling caches, GPU execution, and viability metrics |
| `cascade.exporting` | Output schemas, tables, statistics, plots, VTK construction, and run manifests |
| `cascade.simulation` | End-to-end network/forest execution, sweeps, interventions, and result aggregation |
| `cascade.gui` | GUI pages, window coordination, preview geometry, rendering, workers, and widgets |
| `cascade.commands` | Thin command-line parsing and dispatch |
| `cascade.accelerators` | GPU availability, backend selection, and packaged CUDA kernels |
| `cascade.diagnostics` | Installation/runtime diagnostics and the bounded installed-package self-test |
| `cascade.utils` | Generic execution, hashing, lazy-export, and packaged-resource helpers |
| `cascade.assets` | Packaged domains, vascular seeds, and immutable numerical tables |

The package root intentionally contains only `cascade/__init__.py` plus a
private `_svv_domain.py` compatibility bridge for loading historical domain
objects. The implementation itself lives in `cascade.domain.svv`; application
modules live in the package that owns their behavior.

Within `cascade.concentration.vessel`, `network.py` owns the arbitrary-network
node-mixing solver and `topdown.py` owns the tree-specialized solver. Workflow
modules may call or temporarily re-export these functions, but do not contain
their numerical implementations.

Within `cascade.flow`, `kirchhoff.py` owns backend-independent orchestration
and the tree-specialized solve, while `linear_system.py` owns sparse/dense
matrix assembly, conditioning diagnostics, and linear-solver dispatch.

The scientific subsystems are peers rather than children of `simulation`.
`cascade.simulation` conducts them; it does not own their implementations.
Its `network.py` adapts trees and arbitrary graphs to those subsystem APIs,
while `simple.py` performs the same orchestration for channels, lattices, and
custom segment networks. Their geometry remains under `cascade.vessels`.

## Numerical boundaries

The solver passes explicit typed inputs and results at its public boundaries:
`FlowProblem`, `FlowResult`, `VesselConcentrationProblem`,
`VesselConcentrationResult`, `TissueOxygenProblem`, and `TissueOxygenResult`.
Each contract lives beside the subsystem that owns it and validates array
shape, topology, and physical scalar inputs before numerical work begins.

The finite-radius Green's-function kernels load a versioned Bessel table from
`cascade.assets.numerics`. Its `K1/K0` values are generated with exponentially
scaled Bessel functions so the ratio remains stable at large arguments. The
Graetz closure loads precomputed 8-radial-node, 4-mode bases for plug and
Poiseuille profiles over `1e-8 <= Bi <= 1e6`, with 128 samples per decade. The
reference-compatible 6-node, 3-mode, 16-sample profile remains packaged and
selectable for reproducing the locked validation campaigns. These are fixed
release assets, not per-run caches; unsupported discretizations and corrupt or
missing assets fail immediately with a diagnostic.

`cascade.runtime.tissuesim` provides a unified runtime namespace, while each
numerical implementation lives in its owning flow, concentration, domain, or
simulation package. Mutable settings are centralized in
`cascade.configuration.solver_state`; typed configuration and subsystem entry
points are preferred for focused library integrations.

Anatomical heart forests, generated forests, single trees, lattices, channels,
and custom networks all use `cascade run`. Shared multi-network Cext lives under
`cascade.concentration.external_field`; occlusion and global/local segment
indexing live under `cascade.vessels`; the simulation engine only composes those
capabilities. There is no parallel heart-only solver or exporter.

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

End-to-end orchestration lives in `cascade.simulation`. The same engine accepts
one network or an ordered multi-network collection and delegates numerical work
to the owning scientific package.

Scientific packages do not import GUI or command modules. Commands and GUI
pages depend on simulation services; those services compose domain, vessel,
flow, concentration, and export packages. Public `svv` adaptation is confined to
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

The GUI sends serial simulations to one local child worker. That worker keeps
at most one compatible geometry, one spatial context, and one compact summary
result warm; incompatible state and detailed solver arrays are evicted after
export. It retires when idle or when the GUI closes. Large output tables and
VTK files can be disabled when only summary results are needed. The CLI exposes
the same bounded reuse through `cascade batch`; one-shot `cascade run` performs
hard cleanup at completion.

## Validation boundary

The public package contains a bounded installed-package self-test. Detailed
numerical-equivalence fixtures, frozen legacy oracles, performance campaigns,
and internal regression tests live in the separate CASCADE workbench and are
not shipped in source or binary releases.
