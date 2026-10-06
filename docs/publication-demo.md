# Publication demo

The supplied [demo configuration](../examples/publication/demo.json) and
[simulated vascular network](../examples/publication/demo.tree.npz) demonstrate
loading a vascular tree, solving blood flow and vessel oxygen transport,
evaluating tissue oxygen at 10,000 locations, and exporting numerical and
visualization results. The network contains 201 segments in a cube of side
length 1 cm. It was generated with CASCADE and public
`svv`; no patient data or external dataset download is needed. The saved network
avoids repeating stochastic vascular growth. Tissue sampling uses seed 42.
This small CPU example uses the `topdown` vessel solver; it does not exercise
every optional solver or the large-network CUDA paths.

See [tested versions and hardware](tested-environments.md) for the separate
native Windows and WSL test systems, dependency versions, and validation scope.

The demo explicitly matches the GUI's new-project values: Vmax = 0.001 mM/s,
Km = 0.005 mM, diffusivity = 2.41 × 10⁻⁵ cm²/s, and inlet flow = 100 µL/min.

## Installation for the demo

Download and extract the source distribution, or clone the
[CASCADE repository](https://github.com/carlkohnke/CASCADE), and work from its
root directory. Python 3.12 must already be installed. On Linux/WSL, create a
separate environment and install the package:

```bash
python3.12 -m venv .venv-publication
source .venv-publication/bin/activate
python -m pip install -c requirements/locks/py312-linux-x86_64-publication-cpu.txt ".[gui]"
python -m pip check
cascade self-test
```

Native Windows users should follow the [Windows installation guide](windows.md), then run the
same demo command below in the extracted repository root.

The installation process should take a few minutes.

## Run the demo

With the environment activated, run from the repository root:

```bash
cascade run --settings examples/publication/demo.json
```

## Expected output

The command should exit successfully and create these files:

| File | Expected content |
| --- | --- |
| `summary.csv` | One run-summary row with flow, transport, and timing measurements |
| `segments.csv` | 201 vascular-segment rows, including flow and inlet/outlet concentrations |
| `points.csv` | Tissue-sample coordinates and oxygen concentration (9,995 exported rows from 10,000 requested samples in the reference run) |
| `manifest.json` | Settings, input hashes, software/environment versions, timings, and output paths |
| `vessels.vtp` | Vessel geometry and solved fields |
| `oxygen_points.vtp` | Sampled tissue oxygen field |
| `domain_boundary.vtp`, `domain_mesh.vtu` | Cube boundary and tetrahedral domain mesh |
| `publication_demo.tree.npz` | Exported vascular network |

CSV files can be opened in a spreadsheet or Python. VTP/VTU files can be opened
in ParaView or PyVista; ParaView is not required to run the demo. In
`summary.csv`, expect exactly 201 total segments and
inlet flow of approximately 100 microlitres/minute. Truncated numeric reference values for
the tested environment are listed below.

| Summary column | Reference value |
| --- | --- |
| `total_segments` | 201 |
| `terminal_segments` | 101 |
| `inlet_flow_ul_per_min` | 100. |
| `pressure_drop` | 5518. Pa |
| `C_LQ_over_Cmax` | 0.872 |
| `C_tiss_over_Cmax` | 0.0343 |


## Expected demo run time

The current 10,000-point CPU demo took 5.17 seconds on the
[tested WSL laptop](tested-environments.md#ubuntu-under-wsl-2) with existing
caches. Allow approximately 10 seconds on a comparable machine; the first run
and different hardware can take longer.
