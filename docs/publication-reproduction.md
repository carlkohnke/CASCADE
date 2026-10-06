# Main-text simulations

This guide summarizes the inputs used for Figs. 2–5. Use the paper's Methods
and Supplementary Table S3 for full details.

## Set up a case

Create a settings file with `cascade init-settings case.json`, or configure a
project in Studio. Generate or select the tissue domain and vascular network, then set the
flow boundary conditions, perfusate, oxygen uptake, and sampling resolution.
Run with:

```bash
cascade run --settings case.json
```

Use [custom geometry](custom-geometry.md) for imported segments,
[domains](domains.md) for tissue surfaces, and the [settings reference](settings.md)
for individual controls. For comparisons, reuse the same saved network and
sample coordinates, disable growth, and give each case a separate output folder.

## Physical inputs

Lengths in settings files are in cm, flow in µL/min, and pressure in Pa.
Concentration is in mol/m³ (numerically equal to mM); Vmax is in mol/m³/s.
Convert paper diffusivities from m²/s to cm²/s by multiplying by 10,000.

| Input | Setting |
| --- | --- |
| Inlet flow | `simulation.qin_target_ul_min` |
| Inlet/outlet pressure | `settings.hemodynamics.root_pressure`, `terminal_pressure` |
| Perfusate | `simulation.fluid` (`blood` or `cell media`) |
| Inlet dissolved oxygen | `settings.oxygen.concentration_inlet_by_fluid` |
| Vmax and Km | `settings.oxygen.vmax_mm`, `k_m_mm` |
| Tissue and lumen diffusivity | `settings.oxygen.solute_diffusivity`, `lumen_diffusivity_cm2_s` |
| Inlet hematocrit | `settings.hematocrit.hd_discharge` |
| Viability threshold | `simulation.viability_threshold` |

See the Supplemental Info of the publication for all these values.

## Fig. 2: engineered constructs

Use the GUI's built-in straight or serpentine channel, implemented in
[`src/cascade/vessels/simple.py`](../src/cascade/vessels/simple.py).
In JSON, select `network.mode="simple"` and
`network.simple.mode="onechannel"` or `"snake"`, respectively.

Use direct vessel–vessel coupling, a 6λ cutoff, three external-field iterations,
and approximately 20 µm tissue sampling. Compare oxygen on the sensor plane
and viability on the experimental cross-section.

## Figs. 3–4: cube networks and design sweeps

Generate svVascularize (SVV) trees inside a 1 cm³ cube using
`domain.type="cube"`, `domain.side_length=1.0`, and `network.mode="tree"`.
Set `network.target_terminal_count` to the corresponding terminal count for
each network size. The CCO convention is approximately 2N + 1 vessel segments
for N growth additions; check the exported count against the figure.
For large cube trees, switch to equal-bifurcation growth at approximately
150,000 terminals (300,000 segments) using `growth.n_equal_bifurcations`.

For Fig. 4a–d, vary vascular density
with inlet flow 900 µL/min and outlet pressure 4000 Pa; compare blood and
cell-culture medium. Use the myocardial uptake settings in Supplementary Table S3 and a
100 × 100 × 100 tissue grid. Fig. 4a cross-sections use 500 × 500 samples.

For Fig. 3, save and reload the generated networks to exclude growth from timing.
Use five quadrature nodes per segment, vary tissue sample count, and compare
CPU/GPU and direct/FFT execution. The paper's FFT configuration uses a
288³ grid per cm³ and five reaction-diffusion-length bins.

For Fig. 4e–g, vary flow (0.005–20,000 µL/min), inlet oxygen (0.07–0.16),
hematocrit (0–0.42), Vmax (0.0005–0.04), Km (0.00069–0.0036), tissue
diffusivity (10⁻⁸–10⁻⁵ cm²/s), and viability threshold (0.0014–0.007).
Compare branched, tetrahedral, simple-cubic, body-centered-cubic, and octet
networks under matched conditions. Lattice radii span 10–500 µm; omit
geometrically infeasible cases. The plotted volume-adjusted viability is
viable fraction × (1 − vascular volume / domain volume).

## Fig. 5: biventricular heart

Use the packaged myocardial surface
[`src/cascade/assets/domains/bivent.stl`](../src/cascade/assets/domains/bivent.stl)
and generate SVV coronary trees within it. Extend the distal vasculature until
channel diameters are approximately 3–4 µm (radius 0.00015–0.0002 cm), checking exported
radii as you increase the terminal count. Use a forest for separate LCA/RCA
trees and allocate terminals according to their inlet-flow fractions.

Switch from CCO to terminal-only equal-bifurcation growth near 350,000 total
terminals (approximately 700,000 segments), as described in the paper. Set
`growth.n_equal_bifurcations` to the switching terminal count; when growing
LCA/RCA trees separately, use each tree's share of that threshold.
CASCADE provides this mode through its compatibility layer for public
`svv==0.0.48`; see the [growth settings](settings.md) for controls.

The study used approximately 25 million vessel segments and 8 million equally spaced
tissue samples. Use direct coupling with a shared external field across the
coronary trees; the heart simulations did not use FFT. Baseline LCA/RCA inlet
flows were 189,600 / 85,000 µL/min. A newly generated network reproduces the
simulation approach, rather than the exact published branch territories.

| Comparison | Change from baseline |
| --- | --- |
| Complete LD2 occlusion | Block the selected LD2-equivalent segment and its downstream subtree |
| Partial LD2 reperfusion | Adjust the lesion until LD2 flow is 30% of baseline |
| Branch stenosis | Sweep fractional radius reduction from 0 to 1 at each selected branch site |
| Hematocrit | Sweep 0–0.80 at fixed inlet/outlet pressures of 12,665.6 / 4000 Pa |
| Hypoxemia | Sweep inlet dissolved oxygen from 0 to 0.14 with other inputs fixed |

Use `simulation.occlusion` for a selected segment's radius reduction.
Its `fraction_blocked` is a radius reduction, not a flow reduction; setting it
to 0.7 does not prescribe 30% flow. Select segment IDs and downstream territory masks
from your generated network. Keep inlet-flow boundary conditions for
Figs. 5a–d,f; use pressure boundary conditions for Fig. 5e so flow responds to
hematocrit-dependent viscosity. See [example settings](examples.md) for
pressure-driven and multi-tree cases.

## Read the results

Use `summary.csv` for case comparisons, `segments.csv` for vessel flow and
oxygen, and `points.csv` for tissue oxygen and viability. For Fig. 5d, compute
viability within each selected downstream territory. Save the manifest with
each case to retain its settings and input hashes. See [outputs](outputs.md)
for visualization and export details.
