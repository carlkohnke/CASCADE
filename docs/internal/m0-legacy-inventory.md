# M0 legacy inventory and frozen validation boundary

Status: frozen for M2 staging. No scientific certification is implied by this inventory.

## Isolation contract

The M2 comparison has two independent execution sides:

1. The CASCADE side installs the wheel into a clean CASCADE environment and uses only installed `cascade` entry points plus the pinned public `svv` dependency for vascular growth.
2. The legacy side runs the files in the `svva2` environment's installed `svv/SCRIPTS` directory.
3. CASCADE never imports, copies, or dynamically loads a file from `svva2/SCRIPTS`. A comparison driver may launch the two environments as separate subprocesses and compare emitted files.

The obsolete same-process benchmark/audit helpers were removed because they imported the former root script copy and could not enforce this boundary. Hash-checking M2 fixture/oracle runners and a file-only comparison utility now live under `validation/harness/`; they preserve separate interpreter processes and are excluded from release archives.

Large legacy trees, forests, caches, and raw outputs remain outside Git. Their exact paths, sizes, and hashes are evidence, not package inputs.

The former tracked root `TissueSim_cube_local.py` and the frozen SCRIPTS copy are substantially the same lineage but are not identical: a line-level comparison measured `0.9699117570` similarity. The former root file had eight additional named helpers for runtime-stencil/runtime-moment GPU work and PyVista domain construction, while the SCRIPTS file had four additional control globals. The later owner direction makes the SCRIPTS file the external comparison oracle, not the source of CASCADE code. CASCADE independently retains the additional GPU and flexible-domain capabilities. Exact comparison metadata is stored in `../../validation/results/2026-09-09_m0-inventory_20744c8/oracle-comparison.json`.

## Frozen legacy programs

All paths below are under `/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/`.

| Role | File | Bytes | SHA-256 | Disposition |
| --- | --- | ---: | --- | --- |
| Primary cube oracle | `TissueSim_cube_local.py` | 894,463 | `f2c4c89c8826b11246a239f7ae947872b955e5e47b44290033c13d1cef68bfda` | Required for M2/M3 |
| Primary heart/export oracle | `export_paraview_heart_forest_grid_cext_gfm.py` | 130,114 | `cc916e26217bc64af23a2f25ff6a9306c2264dbe165bc4007b66de4c310df0a3` | Required for M2/M3 |
| Tissue-capable heart/export oracle | `export_paraview_heart_forest_grid_cext_gfm.py.pre_cext_state_slim.bak` | 128,371 | `3fed343b3c2bbce84b021f55f8b174fe618a21267a6edc074c91680ee6848165` | Required for HEART-S/L tissue comparison because the newer state-slim exporter omits fields required by the current cube runtime |
| Heart growth oracle | `TissueSim_heart_forest.py` | 143,919 | `a2f07948c38397509fb98d87af66117ebccf8cbddac1117e0c7df3ca40b39dbc` | Capability/reference input for nearest-tree growth |
| Heart runtime used by growth/export | `TissueSim_heart_accel.py` | 360,718 | `d5a180932a6fef944f7ebcd3ca37edaa83ad076fafcae88d556e46708e170a9e` | Required supporting oracle module |
| Older accelerated path | `TissueSim_accel.py` | 191,051 | `edd22973542d41766d36a4e7d29b1d4c6c751ad26987ab660d78cf0bdea96f9d` | Deferred/lower priority |

The Cext exporter does not import `TissueSim_heart_forest.py`. It imports the heart accelerator for per-tree behavior and dynamically loads a cube TissueSim module for shared Cext. `TissueSim_heart_forest.py` is retained as a separate oracle for the heart growth capabilities that produced the production structure.

The external Cext exporter is byte-identical to the version stored in Git history at pre-CASCADE commit `3c040c8`. The exact cube module used by the June 2026 production export cannot be proven from that run: its JSON names the now-missing `/home/carl/GFM/TissueSim_cube_local.py` but does not record its hash. M2 must use the frozen `svva2/SCRIPTS/TissueSim_cube_local.py` above and label this as a new comparison baseline, not a recreation of the June run.

HEART-S preflight established that the primary June-23 exporter is not tissue-compatible with the current August-10 SCRIPTS cube runtime: shared Cext completes, but its slim concatenated state omits `k_if_gl` and tissue evaluation raises `KeyError`. The retained June-17 pre-state-slim backup completes the identical case and is therefore the frozen tissue-capable oracle. CASCADE does not import either file; the legacy environment executes the selected oracle as its own subprocess.

## Legacy environment

The currently frozen oracle environment reports:

| Component | Version |
| --- | --- |
| Python | 3.9.20 |
| `svv` | 0.0.43 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| CuPy | 13.6.0 |

Directly imported `svv` implementation files are also frozen by hash:

| Installed module | SHA-256 |
| --- | --- |
| `svv/domain/domain.py` | `adb3bab63553cf9a9de8fe08f016d8439965a4191a7af0aabd9942084cdd058e` |
| `svv/forest/forest.py` | `b784783e4bd198c943a60f6976a711071bf8ddcc0e711d8c804e06411b06d9cb` |
| `svv/tree/collision/tree_collision.py` | `61fb12415e2818620d3613e1333201c213223f0dda84f18f7efac30455c0615f` |
| `svv/tree/data/data.py` | `2b27d75635db734d3b8b77156e61cf8eff9ebdce4e85c94439a403c29b652504` |
| `svv/tree/utils/TreeManager.py` | `545ed5c360880f807378e923e28007046c3dc9a17c8a25e9d1956af95fe7bc5f` |
| `svv/tree/branch/bifurcation.py` | `571e12f4c83935f46b2b24ddd682681efe8a09ba3e103f6056512ee841430118` |
| `svv/tree/tree.py` | `4ad748939050c7e6a48df13836d4f7608e65fabd1c5fbaa87a56b7c17514d20f` |

M2 must capture the full package inventory again at execution time. The legacy `svv==0.0.43` environment is intentional oracle history; the CASCADE side remains the clean wheel environment with public `svv==0.0.48`.

## Frozen production heart inputs

### HEART-L (large/production; formerly HEART-P)

`HEART-L` means the actual one-millimetre extended production forest, not an arbitrary large heart case.

- Link used by the exporter: `/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/FORESTS/coverage_terminal_extend_0gen_1mm_gl5_20260623/heart_seed2_coverage_terminal_extend_0gen_1mm.forest`
- Resolved file: `/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/FORESTS/coverage_joint_reduced_from300k_to25M_20260616_211532/heart_seed2_coverage_joint_reduced_from300k_to25M.forest`
- Size: 2,965,278,179 bytes
- SHA-256: `c0d4093adf1eaa3b79c4c2814bed03a0a6158175969d9172dd852260f3ed8f6a`
- Structure metadata: 24,999,999 segments and 12,500,000 terminals across two trees.
- Legacy domain cache: `tree_87b94fb1cc22bdd55af637e8910d3fa5_t1001.f64.tree.dmn`, 2,531,847 bytes, SHA-256 `4cdd5ba2ec4606b7a78a356b20b49c22866895a8a0c55e4ae8c7b0946f858cc2`.
- Preferred M2 load source: `/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/FORESTS/coverage_terminal_extend_0gen_1mm_gl5_20260623/heart_seed2_coverage_terminal_extend_0gen_1mm_equal_terminal_true_kirchhoff_radii.forest.simcache`.
- Simulation cache size: 6,200,001,691 bytes.
- Simulation cache SHA-256: `6bcb5863e86dad79dec96aea0622e5dc42f13b989eb7b6cb124d6568145707e6`.
- The stored, uncompressed simulation cache is preferred because loading the compressed `.forest` is prohibitively slow. The `.forest` remains the provenance/original-structure reference.

### HEART-S (small/debug)

`HEART-S` is the approved reduced two-tree debug forest using the same bivent3 surface and frozen numerical profile.

- Path: `/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/FORESTS/heart_seed2_grown100000_t10000.forest`.
- Size: 3,998,966 bytes.
- SHA-256: `9c418f84ea4b6019c0b46fcfea7e019f38f15fc74edf974dbae44b611fd0972c`.
- Structure metadata: 19,999 segments and 10,000 terminals across two trees.
- Stored tree arrays: float64 geometry and int64 connectivity. CASCADE converts its working numerical representation according to the frozen float32 accelerator policy without modifying this file.

### bivent3 domain

The permitted heart surface is packaged in CASCADE as `src/cascade/assets/domains/bivent3.stl` and exposed as a Studio domain option.

- Original location: `/mnt/c/Users/carl/OneDrive - Stanford/Desktop/O2 Dynamics/Vascular Geometries/Domains/Organs/bivent3.stl`
- Size: 652,584 bytes
- SHA-256: `10fd497650eb881b37390861cb960db88e0c7e6f0b18267ffbb35d564c24a060`
- Units: centimetres. The heart scripts use `SCALE = 1.0`; the loaded forest reports a centimetre/gram/second base unit system and its coordinate bounds match the bivent3 bounds.
- Publication/privacy: owner approved repository and GUI inclusion; license review remains part of M4.

## Frozen validation profile

Unless a case explicitly tests a different setting, the M2 heart/Cext profile is:

| Setting | Value |
| --- | ---: |
| Accelerator working precision | float32 |
| Cext background grid per axis | 256 |
| Cext quadrature | 1 |
| Tissue quadrature | 5 |
| Cext vessel-coupling iterations | 1 |
| Window factor | 6 |

Cext quadrature and tissue quadrature are independent stages. Production CASCADE uses GL1 to solve vessel/Cext coupling, then constructs GL5 source points for tissue oxygen integration. The frozen exporter historically reuses the GL1 Cext node during tissue evaluation even when passed tissue GL5; that behavior is retained only through an explicit validation diagnostic and is not the production default.

The runtime-stencil and runtime-moment GPU functions, flexible mesh-backed domain construction, and automatic CUDA component-library discovery are required CASCADE capabilities.

These are deliberately owner-frozen validation values, not an assertion that every current legacy file-level default already matches them. In particular, the frozen `TissueSim_cube_local.py` currently declares a 128 Cext grid and window factor 4; M2 must pass explicit overrides for grid 256 and window factor 6 to both sides and record those effective settings.

Float32 here means the heart/Cext/tissue accelerator work arrays and caches used to control memory. Validation uses existing trees and performs no growth. Future growth qualification will keep true CCO in public SVV float64, use 300,000 terminals as the default equal-bifurcation transition, and convert to float32 at that boundary. The conversion utility is retained, but that automatic transition is deliberately not activated during computational-method validation.

## Cube scale matrix

The historical 1/100/1000 cases are smoke/history only. The current legacy defaults match cache family `a50c006ac491d07e399f94f1be37cfe0` through the legacy compatibility-key normalization. The current computed key is `887b415362f2342df5e5c6f57aee9a33`; the matcher confirms that the stored family is compatible.

The frozen available matrix is:

| Requested terminals | Expected total segments (`2T + 1`) | Bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| 1 | 3 | 2,905 | `8d0e6b8a3f6b9c0e28384337888d43c1df135d0a33292b4b7e98e426a63352be` |
| 10 | 21 | 6,938 | `02e1c3e563f27481a1462108f2e8fb0a8ed1c996c23b4d54fcb556c3dadb385d` |
| 100 | 201 | 46,120 | `ad547a2d40ddfcc93880d0dd51860c4390c67eb818540b72fb01a27ef5218283` |
| 1,000 | 2,001 | 465,281 | `f7d0c209be4cd62e08a72be4e3e8fbf7f5604a7fd0dbea89e9962fc562b833b7` |
| 10,000 | 20,001 | 5,297,787 | `e5051e152e6ca0b2773e40f0305b8babfcb04c3f305cf59c59e390b3e057da35` |
| 100,000 | 200,001 | 59,391,226 | `245423388a9e97781c439eef12312ad479d7624bdd5a54e5945783ff1b64f062` |
| 1,000,000 | 2,000,001 | 356,536,777 | `eda0008cbc0fcabd894adc917a221330ce1aaa7440adccedcd6f393d3db2d170` |
| 5,000,000 | 10,000,001 | 1,401,237,357 | `8f1126a60b93ba777feea4b0f05e2127e623b22060baf4ad913567459c0af8c7` |

Every path is `trees_cache/tree_a50c006ac491d07e399f94f1be37cfe0_t<T>.tree.npz` under the frozen SCRIPTS directory. No ten-million-terminal cube file exists in the inspected cache hierarchy. M2/M3 stop at five million and must not build a replacement; the missing expected ten-million input is reported as a legacy cache gap.

Large structures stay in the external SCRIPTS hierarchy. Small publishable fixtures and settings may be committed under `validation/fixtures/` once selected.

The legacy key lookup is used only to establish this inventory. M2 passes these paths explicitly to both sides; CASCADE contains no legacy key lookup or SCRIPTS cache discovery.

## Shared tissue coordinates

Pointwise oxygen comparisons use exactly the same coordinate array on both sides. CASCADE supports `simulation.sample_mode=file` with CSV (`x,y,z` in cm), NPY, or NPZ (`points` or `sample_points`) input. The resolved path, point count, units, and SHA-256 are recorded in the run manifest. Exact point files are generated or selected when each M2 case is staged; they are not regenerated independently by the two solvers.

## Memory execution contract

- Host budget: approximately 50 GB RAM.
- Only one legacy or CASCADE simulation may be resident at once.
- Legacy and CASCADE comparisons run sequentially as isolated subprocesses and are compared only after each subprocess exits.
- CASCADE CLI run, sweep, and heart-export entry points use a per-user operating-system lock to reject overlapping CASCADE simulations.
- Studio already uses a sequential child-process queue. The process exit releases all host/GPU allocations before the next job starts.
- Sequential in-process sweeps delete completed result arrays, clear module-held Cext/tissue state, synchronize CUDA, and run garbage collection between cases. A single bounded worker retains CuPy allocator pools for warm reuse; process exit releases them before an unrelated or large case begins.
- Large computational cases default to compact summaries and validation statistics. Per-segment CSV/VTK materialization is tested separately at an approved size because Python row dictionaries can dominate memory.

## Interchange boundary

- STL/VTP/VTU are the supported public geometry interchange path: STL and compatible meshes are inputs; VTP/VTU are ParaView-oriented outputs.
- `.dmn` is treated as a legacy/internal serialized domain or cache. CASCADE may read a specifically frozen `.dmn` for validation, but it is not a promised public cross-version interchange format.
- A custom tissue oxygen-consumption law is deferred; the code and tracker retain an explicit TODO.
- High-scale nearest-tree growth remains required. CASCADE keeps its local implementation while an upstream primitive is requested and until that replacement passes saved-fixture validation.

## Owner-approved M2 workload contract

- Cube correctness uses one frozen one-million-point pool generated by the frozen legacy cube/domain seed-42 sampler. Smaller cases may use leading prefixes for debugging, but final correctness uses all one million coordinates.
- HEART-S and HEART-L use the exporter's full `200 x 200 x 200` candidate grid. Exact retained tissue coordinates and inside-domain masks are compared.
- Cube certification covers blood and water/cell-media analysis; heart remains blood-only.
- Heart certification includes the healthy case and one full downstream occlusion case on HEART-S. The frozen target is global segment 1 (tree 0, local segment 1), a first major branch whose downstream set contains 6,207 of tree 0's 12,842 segments in the frozen structure.
- Full VTK/CSV schema and value parity is exercised on bounded cases. The 1M/5M cubes and HEART-L first produce compact solver evidence; one full HEART-L export is attempted only as a separately monitored case after numerical and memory gates pass.
- Performance uses five paired runs through 100k terminals, three paired runs at 1M, and one initial paired run at 5M and HEART-L. A large-case pair is repeated when it fails, when either process reports memory pressure, or when its CASCADE/legacy ratio lies within 5% of the release threshold and therefore cannot support a stable conclusion.
- Physical fields and nonzero summary metrics use relative tolerance `1e-3`. Near zero, absolute tolerance is `1e-6` times that field's frozen reference-case scale. Fractions such as `FracAbove1pct` use absolute tolerance `0.001`. Structure, IDs, shared coordinates, ordering/mapping, and finite/non-finite masks are exact.
- CPU/GPU consistency is checked on small and medium cases. Production-scale 1M/5M/HEART-L work is GPU-only.
- CPU certification is limited to cube targets 1, 10, and 100. GPU is primary at every cube size, mandatory above 100, and mandatory for every HEART-S/HEART-L validation and performance run.
- Growth-only SLSQP/L-BFGS-B qualification and the ignored-constraint warning remain deferred until growth validation.

## Frozen historical GPU timing reference

The owner's June 2026 characterization is stored externally at
`/home/carl/miniconda3/envs/svva2/lib/python3.9/site-packages/svv/SCRIPTS/Cube_Memory_Improvement.csv`
(SHA-256 `fecf37597a34ac89533122b7563a36ddefb222a60bee5fd09e228517480b2016`).
Its 74 rows use one million tissue samples, float32/int32 accelerated arrays,
`topdown_ext_hybrid_bg`, GPU Cext and tissue, Cext GL1, tissue GL5, one Cext
iteration, a 128-cubed FFT grid, and window factor 4. The plotted compute time is
exactly `t_assembly + t_kirchhoff + t_concentration + t_tissue`; it excludes
process startup, tree loading, domain construction, and export. Compact grouped
evidence is retained in
`../../validation/results/2026-09-10_m3-gpu-reference_fecf3759/`.

The corresponding August benchmark snapshot is
`C:/Users/carl/svVascularize/svVascularize/tools/_TissueSim_cube_local_benchmark.py`
(SHA-256 `1c8225416f13b1d57b768fde6df6dc619b03f8a8c73aa44a2d268503b69f3a50`).
It is functionally close but not byte-identical to the current frozen oracle.
Certification executes the current hash-frozen oracle; the recovered CSV remains
the historical performance target. Its 128/window-4 profile is not substituted
for the owner-frozen production heart profile of 256/window-6.
