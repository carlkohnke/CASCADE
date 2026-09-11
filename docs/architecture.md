# CASCADE architecture
CASCADE separates configuration, vascular geometry, numerical simulation, and export so command-line and GUI runs use the same engine.

```text
JSON / CASCADE Studio
        |
        v
cascade.config -> cascade.growth -> cascade.simulation -> cascade.export
                        |                   |
                        v                   v
               public svv adapter    packaged TissueSim runtime
                                            |
                                            v
                                   CSV / VTP / VTU / manifest
```

The `cascade.runtime.tissuesim` module owns the Python control path for legacy-compatible flow and oxygen computation. Its CuPy/NVRTC kernels are packaged as CUDA translation units under `cascade.runtime.cuda`; the Python module loads those resources lazily and retains the existing kernel cache and call interfaces. This is a package-level separation, not yet a full decomposition of the Python numerical core: flow, rheology, concentration, Cext, and tissue-sampling routines still share one compatibility module and its runtime configuration state. Legacy TissueSim validation oracles are maintained outside the public repository and are not imported by the installed package.

`cascade.heart_export` provides the advanced forest-global Cext export path. Normal tree, forest, channel, lattice, and custom-network exports use `cascade run`.

## Native source boundary

CASCADE keeps code in the language and subsystem where it executes:

- Python coordinates configuration, vascular models, simulation, CLI, and GUI behavior.
- CUDA C/C++ implements the GPU Cext, treecode, FFT-correction, and tissue kernels compiled at runtime by CuPy/NVRTC.
- GLSL vertex and fragment shaders implement the retained OpenGL viewport pipeline.
- C# implements the Windows common-dialog interop class loaded by the WSL PowerShell bridge.

CUDA, GLSL, and C# sources are ordinary package resources included in both wheels and source distributions. CPU-only imports read none of the CUDA resources until a GPU kernel is requested.

## Reproducibility boundary

A run is identified by its settings file, input hashes, CASCADE version/commit, dependency versions, random seed, execution environment, and generated manifest. Relative inputs resolve from the settings file, not from the caller's current directory.

## Memory boundary

The GUI launches simulations as child processes. It stores settings and lightweight result metadata but does not retain full solver arrays after a worker exits. Large output tables and VTK files should be disabled when only summary results are needed.
