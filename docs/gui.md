# CASCADE Studio GUI

CASCADE Studio is the native graphical interface for CASCADE.

## Install and launch

For a released wheel on native Windows, install the `gui` extra and use the
wheel-generated launchers:

```powershell
& "$env:USERPROFILE\CASCADE\Scripts\cascade-gui.exe"
```

This normal launcher uses the Windows GUI subsystem and does not retain a
console. `cascade-gui-console.exe` is the visible diagnostic launcher. Complete
CPU and CUDA 13 setup commands, log locations, and troubleshooting are in the
[native Windows installation guide](windows.md).

For Linux or WSL installation from the repository:

From the repository root...

```bash
python3.12 setup_linux.py --gui --gpu cu13
.venv/bin/python -m cascade.gui
```

New projects default to `~/CASCADE Projects/Untitled`. Set
`CASCADE_PROJECT_DIR` to choose another location. The repository launcher also
uses a sibling `CASCADE-workbench/projects/CASCADE_Project` directory when that
workbench exists, keeping generated projects and GUI state out of the source
tree.

Use the CUDA option that matches the machine (`cu11`, `cu12`, `cu13`, or omit
`--gpu` for CPU-only work). After setup, `cascade-gui` is equivalent. WSL users
can run `bash scripts/wsl/launch-studio.sh`; the `.cmd` and `.vbs` files in that
directory are WSL convenience wrappers and are not the native Windows launch
path. See the [Linux](linux.md) and [WSL](wsl.md) installation guides.

## Guided workflow

1. **Project** names the study and reports detected CPU, RAM, GPU, and GPU memory.
2. **Domain** creates a box or sphere, uses the packaged bivent3 heart surface, or references a mesh/legacy `.dmn` file without copying it into the GUI.
3. **Network** grows a tree/forest with public svVascularize, loads a saved svVascularize object, builds a paper lattice, or creates a simple channel.
4. **Physics** sets unit-aware flow/pressure boundary conditions, diffusivity, Vmax, Km, inlet oxygen, hematocrit, hemoglobin oxygen capacity, and an optional viability threshold.
5. **Solver** selects the flow/concentration/Cext path, Graetz or well-mixed closure, finite-radius terms, quadrature, iterations, convergence, backend, and expert runtime overrides.
6. **Outputs** chooses sampling and file products, creates parameter sweeps, and shows a hardware estimate before queueing.
7. **Run** saves a frozen settings file for every job and runs jobs sequentially through a memory-bounded local worker, with stage, progress, cancellation, and logs.
8. **Results** inspects summary data and scalar-colored vessel/tissue output in the shared interactive viewport.

The tissue-point selector supports random points, a structured Cartesian grid, or a fixed CSV/NPY/NPZ coordinate file. Fixed files make validation runs evaluate identical coordinates; CSV columns are `x,y,z` in centimetres, while NPZ uses `points` or `sample_points` with shape `(N, 3)`.

Lattice sizing can use either the number of cells along X or a physical X
unit-cell spacing in centimetres. X is the baseline: **Y:X anisotropy** and
**Z:X anisotropy** multiply that spacing, so values above 1 stretch cells on
that axis and values below 1 compress them. CASCADE derives the Y and Z grid
counts from the domain while preserving those physical spacings.

Project files are ordinary CASCADE JSON. GUI-only state is stored under the versioned top-level `gui` key, which the command-line parser safely ignores. Queue state and frozen job settings live in `<project>/.cascade_gui/`; results live in `<project>/results/`.

## Lattices

The GUI supports the four networks in the paper:

- simple cubic;
- tetrahedral / diamond cubic;
- body-centered cubic;
- octet.

Inlet and outlet locations are entered as one `x, y, z` point per line and snap to the nearest generated graph nodes. Multiple inlets and outlets are supported. A radius law can use `x`, `y`, `z`, `r0`, and `L`; it is evaluated safely at segment midpoints. Increase **Subsegments per strut** when the radius must vary along a strut. Because lattices have loops, CASCADE Studio forces the sparse Kirchhoff and general network concentration solvers.