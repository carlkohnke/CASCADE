# CASCADE Studio GUI

CASCADE Studio is the native graphical interface for GFM. It is designed for experimentalists and keeps the numerical worker separate from the interface so the simulation can use nearly all available host and GPU memory.

## Install and launch

From `/home/carl/svv_sweeps/GFM`:

```bash
python setup_env.py --gui --gpu cu13
.venv/bin/python -m gfm.gui
```

Use the CUDA option that matches the machine (`cu11`, `cu12`, `cu13`, or omit `--gpu` for CPU-only work). After an editable install, `gfm-gui` is equivalent. On Carl's current WSL installation, double-click `launch_gui_windows.vbs` for a console-free launch; `launch_gui_windows.bat` is the visible diagnostic fallback. Forwarding launchers remain under `/home/carl/GFM` so existing Windows shortcuts continue to work. Linux and macOS users can run `./launch_gui.sh` after making it executable.

## Guided workflow

1. **Project** names the study and reports detected CPU, RAM, GPU, and GPU memory.
2. **Domain** creates a cube, box, or sphere, or references a mesh/`.dmn` file without copying it into the GUI.
3. **Network** grows a tree/forest with public svVascularize, loads a saved svVascularize object, builds a paper lattice, or creates a simple channel.
4. **Physics** sets unit-aware flow/pressure boundary conditions, diffusivity, Vmax, Km, inlet oxygen, hematocrit, hemoglobin oxygen capacity, and an optional viability threshold.
5. **Solver** selects the flow/concentration/Cext path, Graetz or well-mixed closure, finite-radius terms, quadrature, iterations, convergence, backend, and expert runtime overrides. Controls that do not apply are grayed out.
6. **Outputs** chooses sampling and file products, creates Cartesian parameter sweeps, and shows a hardware estimate before queueing.
7. **Run** saves a frozen settings file for every job and runs jobs sequentially in isolated Python processes, with stage, progress, cancellation, and logs.
8. **Results** inspects summary data and scalar-colored vessel/tissue output in the shared interactive viewport. A separate full viewer remains available for unusually large or detailed views.

The viewport remains the primary workspace through setup and analysis. Drag to rotate, scroll to zoom, and use **Home** to reset the camera. In Results, click a rendered vessel or tissue point to inspect its exported numeric fields in the right-side inspector.

For generated SVV trees, a lightweight branching cue appears immediately and is replaced automatically by the exact reusable seed. The inlet is placed on the primitive-domain boundary by default. Segment radii are hydraulically sized from total inlet flow and the inlet–outlet pressure drop; changing either value rebuilds the preview seed and updates the reported inlet radius. Disable automatic inlet placement to enter one or more explicit root locations and directions.

Project files are ordinary GFM JSON. GUI-only state is stored under the versioned top-level `gui` key, which the command-line parser safely ignores. Queue state and frozen job settings live in `<project>/.gfm_gui/`; results live in `<project>/results/`.

## Lattices

The GUI supports the four networks in the paper:

- simple cubic;
- tetrahedral / diamond cubic;
- body-centered cubic;
- octet.

Inlet and outlet locations are entered as one `x, y, z` point per line and snap to the nearest generated graph nodes. Multiple inlets and outlets are supported. A radius law can use `x`, `y`, `z`, `r0`, and `L`; it is evaluated safely at segment midpoints. Increase **Subsegments per strut** when the radius must vary along a strut. Because lattices have loops, CASCADE Studio forces the sparse Kirchhoff and general network concentration solvers.

## Memory behavior

- Only one simulation worker is launched at a time.
- Full numerical arrays are never sent to the GUI.
- A worker exits after its job, releasing its Python, CUDA, and allocator state.
- Summary-only output is the default. Per-segment CSV, tissue CSV, and VTK output display a peak-memory warning.
- The hardware estimate reserves the larger of 2 GiB or 10% of system RAM for the OS.
- The in-window viewport caps previews at 5,000 vessels by default and releases its arrays while the queue is running. The optional full viewer is a separate process and should be closed before a large GPU simulation.
- Float32 and Int32 are the default export/cache choices where the runtime supports them; the core svVascularize compute path remains float64.

The estimate is intentionally conservative, not a guarantee. FFT workspaces, sparse factorization fill-in, driver allocations, and user-selected expert settings can change peak memory substantially.

## Interface and accessibility

CASCADE uses centralized visual tokens in `gfm/gui/theme.py` and reusable form primitives in `gfm/gui/widgets.py`. Neutral graphite surfaces carry the interface; plasma/inferno accents indicate selection, transport, progress, and primary actions. The subtle perimeter flow only runs during active computation. Set `CASCADE_REDUCED_MOTION=1` before launch to disable it completely. All primary controls retain keyboard focus styling, state labels accompany color, and scientific values use a monospaced font.

The preserved pre-instrument-redesign GUI is stored at `/home/carl/svv_sweeps/GFM/backups/cascade_gui_pre_instrument_redesign_20260908.tar.gz`.

## Current scientific boundaries

- Pressure-only (inlet pressure plus outlet pressure) flow boundary conditions are shown but blocked because the current GFM solve path requires inlet flow plus a pressure reference.
- A custom `q = κ(C-Cext)` law is displayed but disabled until the runtime exposes a reproducible user-law interface.
- Primitive CSG combinations and arbitrary uploaded graph CSV/NPZ schemas are not yet part of the backend. Uploaded svVascularize tree/forest objects are supported.
- Lattice flow and intravascular oxygen are supported through the general graph solver; the existing top-down external-concentration solvers remain tree-specific.
- Detailed 3D visualization requires VTK output to have been enabled before the run.
- The current segment exporter provides geometry, radius, flow, hematocrit, and centerline oxygen. Per-segment pressure, wall oxygen, and retained Cext fields need a future runtime/export extension before those scalar choices can appear in the viewer.
- Moveable tissue slices currently select the exported sample points in a thin slab. The paper's nearest-distance Gaussian resampling is not yet implemented in the viewer.
