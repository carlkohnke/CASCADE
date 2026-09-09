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

The `cascade.runtime.tissuesim` module owns the legacy-compatible flow and oxygen kernels. Root-level TissueSim scripts are validation oracles and are not imported by the installed package.

`cascade.heart_export` provides the advanced forest-global Cext export path. Normal tree, forest, channel, lattice, and custom-network exports use `cascade run`.

## Reproducibility boundary

A run is identified by its settings file, input hashes, CASCADE version/commit, dependency versions, random seed, execution environment, and generated manifest. Relative inputs resolve from the settings file, not from the caller's current directory.

## Memory boundary

The GUI launches simulations as child processes. It stores settings and lightweight result metadata but does not retain full solver arrays after a worker exits. Large output tables and VTK files should be disabled when only summary results are needed.
