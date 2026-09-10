# M0 legacy-to-CASCADE traceability

This matrix defines what M2 will compare. “Supported” means implemented and operational, not numerically certified.

| Legacy capability | CASCADE command/settings | CASCADE implementation | Validation case | Status |
| --- | --- | --- | --- | --- |
| Build/load cube domain | `cascade run`; `domain.type=cube` | `cascade.growth.build_domain`, `cascade.runtime.tissuesim` | CUBE decade matrix | Supported; M2 certification pending |
| Load one vascular tree | `cascade run`; `network.mode=tree`, explicit `input_path` | `cascade.growth`, `cascade.svv_adapter` | CUBE-1 through CUBE-5M | Supported; M2 uses frozen existing trees only |
| Grow one vascular tree | future growth campaign | public SVV plus CASCADE transition/orchestration | growth fixture TBD | Deferred until after computational-method validation; public SLSQP baseline and requested upstream L-BFGS-B flag |
| Flow/pressure solve | `cascade run`; hemodynamic settings | `cascade.simulation`, runtime TissueSim | CUBE, FOREST, HEART | Supported; D-023 tolerances frozen |
| Intravascular and tissue oxygen | `cascade run`; oxygen/tissue settings | `cascade.simulation`, `cascade.runtime.tissuesim` | CUBE and CUSTOM-Y | Supported; D-023 tolerances frozen |
| File-backed/custom domain | `domain.type=file` with STL/VTP/VTU | `cascade.resources`, `cascade.growth` | CUSTOM-DOMAIN, HEART | Supported public mesh path |
| Packaged biventricular domain | Studio “Biventricular heart (bivent3)” or `path=bivent3.stl` | packaged asset and `cascade.resources` | HEART-S/HEART-L | Supported; centimetre units and hash frozen |
| Scheduled forest growth | `network.mode=forest`; scheduled growth settings | `cascade.growth` | FOREST-S | Supported CASCADE orchestration |
| Nearest-tree forest growth | `growth.assignment=nearest-tree` | `cascade.growth` | FOREST-S and heart growth fixture | Supported locally; upstream primitive requested, local path retained |
| Heart growth oracle behavior | separate legacy `TissueSim_heart_forest.py` subprocess | no legacy runtime linkage; CASCADE growth APIs | later heart growth campaign | Deferred from computational M2; HEART-S/HEART-L existing structures preserve current provenance |
| Forest/cache load and connectivity repair | forest input plus repair/validation settings | `cascade.svv_adapter`, `cascade.connectivity` | FOREST-S, HEART | Supported; parity pending |
| Per-tree heart solve | `cascade export-heart` per-tree mode | `cascade.heart_export` | HEART-S | Supported |
| Shared-global Cext | `cascade export-heart` shared-global mode | `cascade.heart_export`, runtime TissueSim | HEART-S/HEART-L | Supported; numerical/performance certification pending |
| FFT background Cext | grid 256, Cext GL1, one iteration, window 6 | runtime FFT/Cext GPU paths | HEART-S/HEART-L | Required and supported |
| Float32 memory path | `float_dtype=float32` | heart exporter, runtime GPU arrays/caches, retained tree dtype conversion utility | HEART-S/HEART-L, CPU/GPU consistency | Required; validation loads existing trees and performs no growth |
| Runtime-stencil/runtime-moment GPU acceleration | selected by runtime Cext modes | `cascade.runtime.tissuesim` | HEART-S/HEART-L | Required and retained |
| Frozen tissue coordinates | `simulation.sample_mode=file`, `sample_points_path` | `cascade.growth`, `cascade.simulation`, Studio | CUBE/FOREST/CUSTOM | Supported with CSV/NPY/NPZ and manifest hash |
| Tissue grid export | heart exporter grid settings | `cascade.heart_export`, `cascade.export` | HEART-S/HEART-L | Supported |
| ParaView vessels/domain/tissue export | `write_paraview=true` or `cascade export-heart` | `cascade.export`, `cascade.heart_export` | VAL-06 cases | Supported; schema/value parity pending |
| Explicit CSV/NPZ vessels | `network.simple.mode=custom` | `cascade.simple` | CUSTOM-Y | Supported; negative matrix pending |
| Legacy `.dmn` load | `domain.type=file` for a frozen validation cache | CASCADE domain adapter | HEART-L load-only if needed | Internal/validation-only; not public interchange |
| `TissueSim_accel` behavior | none | none | none | Deferred/lower priority |
| Supplementary single-channel plotting/analysis | none | none | none | Intentionally excluded; comparable geometry/solve remains supported |
| Custom oxygen-consumption law | future setting/API | TODO in Studio/runtime contract | future case | Deferred |

## Frozen parameter equivalence

| Scientific meaning | Legacy control | CASCADE control | Frozen M2 value |
| --- | --- | --- | --- |
| Accelerator precision | legacy dtype/global/CLI | `settings.cext.float_dtype`, export dtype | float32 |
| Tissue quadrature | `GL_ORDER` | `settings.oxygen.gl_order` | 5 |
| Explicit Cext quadrature | `GL_ORDER_CEXT` | `settings.oxygen.gl_order_cext` | 1 |
| FFT grid | Cext background grid | `settings.cext.hybrid_bg_grid` | 256 per axis |
| Coupling work | vessel/Cext max iterations | `settings.cext.vess_coupling_max_iter` | 1 |
| Interaction window | window factor | `settings.cext.window_factor` and tissue equivalent | 6 |
| Vascular size | requested target terminal count | explicit frozen `network.input_path` | cube decades through 1,000,000 plus 5,000,000; HEART-L 12,500,000 total |
| Randomness | script seed/global RNG | `domain.random_seed` plus manifest | exact value frozen per fixture |
| Flow and pressure | legacy globals/CLI | `simulation` and `settings.hemodynamics` | exact value frozen per fixture |
| Hematocrit | legacy hematocrit controls | `settings.hematocrit` | exact model/value frozen per fixture |
| Output scope | legacy export flags | `outputs` and heart-export flags | identical requested fields per comparison |

Values still marked per-fixture are not guessed in M0. M2 settings files will freeze them before execution.

## Output equivalence

| Quantity | Legacy representation | CASCADE representation | Comparison rule status |
| --- | --- | --- | --- |
| Tree/forest topology | tree/forest arrays and saved objects | saved network plus manifest | Exact segment-order/local-ID mapping; forest global ID is tree-order offset plus local ID |
| Segment identity | legacy row/local IDs | `segments.csv`, VTP local/global IDs | Exact mapping required |
| Endpoints/radii | tree arrays and VTK geometry | segment table and `vessels.vtp` | Exact when loaded from the same frozen float32 representation; otherwise D-023 physical-field rule |
| Flow/pressure | solver arrays/export fields | segment table/VTP/summary | D-023 physical-field rule; required fields may not be silently omitted |
| Inlet/outlet concentration | legacy segment arrays | segment/VTP oxygen fields | D-023 physical-field rule |
| Wall/intravascular/Cext fields | legacy Cext state/VTK arrays | heart-export VTP arrays/runtime result | Field-name/ID mapping exact; physical arrays use D-023 tolerance |
| Tissue sample coordinates | legacy sample/grid coordinates | fixed CSV/NPY/NPZ input and `oxygen_points.vtp`/`points.csv` | Exact shared input coordinates required |
| Tissue oxygen | legacy tissue result | tissue VTP/CSV values | Exact coordinate/finite mask plus D-023 physical-field rule |
| Summary metrics | legacy JSON/CSV/log | `summary.csv`, `manifest.json` | Same named legacy aggregate/formula where present; D-023 rule |
| Domain output | legacy VTK | `domain_boundary.vtp`, `domain_mesh.vtu` | IDs/counts/schema exact; coordinates use the D-023 physical-field rule unless emitted from the same frozen source, when exact |

The frozen functional acceptance rule is 0.1% relative agreement for oxygenation, flow, pressure, and viability conclusions. Near zero, the absolute tolerance is one millionth of the field's recorded reference-case scale. Fractions including `FracAbove1pct` use ±0.001 absolute. Structures, IDs, shared coordinates, ordering/mapping, and finite-value masks remain exact. See D-023 and `test-plan.md`.

Units are explicit: geometry and tissue coordinates are centimetres, segment flow is compared in `cm^3/s` (and may also be reported in `uL/min`), and public pressure fields are pascals. The legacy and CASCADE Kirchhoff internals use `dyn/cm^2`; M2 normalizes those arrays to pascals before comparison, and CASCADE converts them before writing public summary/CSV/VTK fields. Raw concentration fields retain the frozen solver concentration units, normalized `*_over_Cmax` metrics and viability fractions are dimensionless, and elapsed/component times are seconds. The comparison layer rejects a missing required field rather than treating absence as equality.
