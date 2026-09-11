# CASCADE Studio GUI

CASCADE Studio is the native graphical interface for CASCADE. It is designed for experimentalists and keeps the numerical worker separate from the interface so the simulation can use nearly all available host and GPU memory.

## Install and launch

From the repository root:

```bash
python setup_env.py --gui --gpu cu13
.venv/bin/python -m cascade.gui
```

New projects default to `~/CASCADE Projects/Untitled`. Set
`CASCADE_PROJECT_DIR` to choose another location. The repository launcher also
uses a sibling `CASCADE-workbench/projects/CASCADE_Project` directory when that
workbench exists, keeping generated projects and GUI state out of the source
tree.

Use the CUDA option that matches the machine (`cu11`, `cu12`, `cu13`, or omit `--gpu` for CPU-only work). After an editable install, `cascade-gui` is equivalent. On WSL, double-click `GUI Launchers/launch_gui_windows_silent.vbs` for a console-free launch; `GUI Launchers/launch_gui_windows_shell.bat` is the visible diagnostic fallback. Both Windows launchers resolve the checkout containing their folder, so they do not depend on a user-specific path. Linux and macOS users can run `./GUI\ Launchers/launch_gui_linux.sh` after making it executable.

## Guided workflow

1. **Project** names the study and reports detected CPU, RAM, GPU, and GPU memory.
2. **Domain** creates a box or sphere, uses the packaged bivent3 heart surface, or references a mesh/legacy `.dmn` file without copying it into the GUI.
3. **Network** grows a tree/forest with public svVascularize, loads a saved svVascularize object, builds a paper lattice, or creates a simple channel.
4. **Physics** sets unit-aware flow/pressure boundary conditions, diffusivity, Vmax, Km, inlet oxygen, hematocrit, hemoglobin oxygen capacity, and an optional viability threshold.
5. **Solver** selects the flow/concentration/Cext path, Graetz or well-mixed closure, finite-radius terms, quadrature, iterations, convergence, backend, and expert runtime overrides. Controls that do not apply are grayed out.
6. **Outputs** chooses sampling and file products, creates Cartesian parameter sweeps, and shows a hardware estimate before queueing.
7. **Run** saves a frozen settings file for every job and runs jobs sequentially in isolated Python processes, with stage, progress, cancellation, and logs.
8. **Results** inspects summary data and scalar-colored vessel/tissue output in the shared interactive viewport.

The tissue-point selector supports random points, a structured Cartesian grid, or a fixed CSV/NPY/NPZ coordinate file. Fixed files make validation runs evaluate identical coordinates; CSV columns are `x,y,z` in centimetres, while NPZ uses `points` or `sample_points` with shape `(N, 3)`.

The viewport remains the primary workspace through setup and analysis. Drag to rotate, scroll to zoom, and use **Home** to reset the camera. The gear beside Home opens viewer-wide vessel and tissue sampling controls: choose nearest-inlet, deterministic random, all, or none, and type an explicit display limit. These project-backed controls apply consistently to setup and Results; Results retains only field, colormap, range, scale, opacity, and unit controls. In Results, click a rendered vessel or tissue point to inspect its exported numeric fields in the right-side inspector.

The interactive viewport uses a retained OpenGL renderer by default. Vessel
endpoints, radii, colors, and opacity are uploaded once and expanded into
round screen-space capsules in one instanced draw; tissue samples are uploaded
as one point cloud. Camera movement changes shader uniforms rather than
reprojecting and repainting every object in Python. Hardware OpenGL displays up
to one million tissue points, while software OpenGL uses a conservative 50,000
point budget. Headless/minimal Qt sessions automatically retain the QPainter
fallback. Set `CASCADE_RENDER_BACKEND=software` before launch to force that
fallback for graphics-driver troubleshooting, or `CASCADE_RENDER_BACKEND=gpu`
to require OpenGL.

For generated SVV trees, a lightweight branching cue appears immediately and is replaced automatically by the exact reusable seed. The inlet is placed on the primitive-domain boundary by default. Segment radii are hydraulically sized from total inlet flow and the inlet–outlet pressure drop; changing either value rebuilds the preview seed and updates the reported inlet radius. Disable automatic inlet placement to enter one or more explicit root locations and directions.

Project files are ordinary CASCADE JSON. GUI-only state is stored under the versioned top-level `gui` key, which the command-line parser safely ignores. Queue state and frozen job settings live in `<project>/.cascade_gui/`; results live in `<project>/results/`.

## Lattices

The GUI supports the four networks in the paper:

- simple cubic;
- tetrahedral / diamond cubic;
- body-centered cubic;
- octet.

Inlet and outlet locations are entered as one `x, y, z` point per line and snap to the nearest generated graph nodes. Multiple inlets and outlets are supported. A radius law can use `x`, `y`, `z`, `r0`, and `L`; it is evaluated safely at segment midpoints. Increase **Subsegments per strut** when the radius must vary along a strut. Because lattices have loops, CASCADE Studio forces the sparse Kirchhoff and general network concentration solvers.

## Memory behavior

- Only one simulation worker is launched at a time.
- A per-user operating-system lock also prevents a separately launched CASCADE CLI simulation from overlapping the Studio worker.
- Full numerical arrays are never sent to the GUI.
- A worker exits after its job, releasing its Python, CUDA, and allocator state.
- Summary-only output is the default. Per-segment CSV, tissue CSV, and VTK output display a peak-memory warning.
- The hardware estimate reserves the larger of 2 GiB or 10% of system RAM for the OS.
- Viewer-wide typed limits default to 5,000 vessels and 10,000 tissue points and apply to both setup and Results. Generated-network seed construction has its own approximately 1,000-segment responsiveness budget because tree growth, rather than rendering, dominates that path. The OpenGL viewport admits up to 250,000 vessels and 1,000,000 tissue points on hardware acceleration, or 50,000 of each with software OpenGL; the QPainter fallback uses 5,000 vessels and 10,000 tissue points. The status strip always reports displayed versus underlying counts, and the viewport releases CPU/GPU scene buffers while the queue is running.
- Float32 and Int32 are the default export/cache choices where the runtime supports them; the core svVascularize compute path remains float64.

The estimate is intentionally conservative, not a guarantee. FFT workspaces, sparse factorization fill-in, driver allocations, and user-selected expert settings can change peak memory substantially.

## Interface and accessibility

CASCADE uses centralized visual tokens in `cascade/gui/theme.py` and reusable form primitives in `cascade/gui/widgets.py`. Neutral graphite surfaces carry the interface; plasma/inferno accents indicate selection, transport, progress, and primary actions. The subtle perimeter flow only runs during active computation. Set `CASCADE_REDUCED_MOTION=1` before launch to disable it completely. All primary controls retain keyboard focus styling, state labels accompany color, and scientific values use a monospaced font.

## Current scientific boundaries

- Pressure-only (inlet pressure plus outlet pressure) flow boundary conditions are shown but blocked because the current CASCADE solve path requires inlet flow plus a pressure reference.
- A custom wall-exchange `q = κ(C-Cext)` law is displayed but disabled until the runtime exposes a reproducible user-law interface. A separately configurable tissue oxygen-consumption law is a retained future TODO; the current contract is Michaelis-Menten Vmax/Km.
- Primitive CSG combinations are not yet part of the backend. Explicit segment graphs can be loaded from CASCADE's documented CSV/NPZ schema, and uploaded svVascularize tree/forest objects are supported.
- Lattice flow and intravascular oxygen are supported through the general graph solver; the existing top-down external-concentration solvers remain tree-specific.
- Detailed 3D visualization requires VTK output to have been enabled before the run.
- The current segment exporter provides geometry, radius, flow, hematocrit, and centerline oxygen. Per-segment pressure, wall oxygen, and retained Cext fields need a future runtime/export extension before those scalar choices can appear in the viewer.
- Moveable tissue slices currently select the exported sample points in a thin slab. The paper's nearest-distance Gaussian resampling is not yet implemented in the viewer.
