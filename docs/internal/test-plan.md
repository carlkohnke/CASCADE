# CASCADE Step 2 certification plan

Status: staged; canonical existing structures, workloads, tolerances, and performance repetitions are approved. Numerical campaigns have not yet certified the solver.

## Objective

Demonstrate that CASCADE performs the same scientific work as the selected legacy TissueSim references when given identical structures, domains, settings, and sample points, and that its execution time is equivalent or faster under a controlled comparison.

The primary legacy references are:

- `TissueSim_cube_local.py` for tree simulation and tissue oxygen behavior.
- `export_paraview_heart_forest_grid_cext_gfm.py` for heart forest, shared Cext, tissue grid, and ParaView export behavior.
- Custom channel/domain workflows only as geometry-input capability tests; their extra plotting and profile analyses are not release requirements.

Legacy scripts are test oracles, not production imports. Their exact external paths and hashes are frozen in `m0-legacy-inventory.md`; the CASCADE side must never add that SCRIPTS directory to `PYTHONPATH` or dynamically load a file from it.

## Principles

1. Compare identical work. Geometry, domain, flow inputs, fluid model, solver options, sample points, dtypes, stopping criteria, and output requests must match.
2. Correctness precedes performance. A faster result is irrelevant if fields or solver behavior differ outside approved tolerances.
3. Separate deterministic identity from floating-point agreement. IDs/topology/counts may require exact equality; physical fields generally require explicit absolute/relative tolerances.
4. Separate cold-start, warm solver, and end-to-end timings.
5. Do not silently repair one side only. Any normalization or repair must be applied symmetrically and recorded.
6. Preserve failures and regressions as evidence rather than rerunning until a favorable sample appears.
7. Do not grow trees during the computational-method campaign. Growth/optimizer qualification is a later campaign.
8. Never overlap simulations. With approximately 50 GB host RAM, each legacy or CASCADE subprocess must exit before the next begins.

## Environment matrix

| Environment | Purpose | Required baseline |
| --- | --- | --- |
| CPU release environment | Packaging, CPU solver, deterministic/reference checks | Python 3.9.20 and `locks/requirements-py39-cpu.txt` |
| CUDA 13 release environment | GPU Cext/tissue and performance checks | Python 3.9.20 and `locks/requirements-py39-cu13.txt` |
| Legacy oracle environment | Execute selected reference scripts from `svva2/SCRIPTS` | Python 3.9.20 and the frozen `svv==0.0.43` oracle inventory; do not let it leak into CASCADE imports |
| GitHub CPU CI | Regression checks on clean checkout | Activate after remote is configured |

## Canonical case matrix

Final sizes and fixtures are selected before running. This is the proposed minimum matrix.

| Case | Structure/domain | Solver path | Purpose |
| --- | --- | --- | --- |
| CUBE-1 through CUBE-5M | Frozen cache-family trees at 1, 10, 100, 1k, 10k, 100k, 1M, and 5M requested terminals; one shared 1M-point cube pool | CPU/GPU through medium scale; GPU-only at 1M/5M | Blood and cell-media computational parity, scaling, performance, and memory gates without growth |
| FOREST-S | Fixed two-tree cube forest | CPU, cache reload | Forest mapping/connectivity/cache parity |
| HEART-S | Frozen `heart_seed2_grown100000_t10000.forest`: 10k terminals, 19,999 segments; healthy plus one frozen full downstream occlusion | CPU/GPU consistency plus shared-global CUDA FFT/Cext | Debuggable solver and exporter parity on one hash-frozen legacy-derived point set from the full 200-cubed candidate grid |
| HEART-L | Frozen one-millimetre extended simulation cache: 12.5M terminals, 24,999,999 segments | GPU-only target configuration | Production solver/performance/memory gate on the full 200-cubed grid; full export only after those gates pass |
| CUSTOM-Y | Explicit CSV Y channel in box domain | CPU tissue GFM | Custom topology/units/export correctness |
| CUSTOM-DOMAIN | Fixed user-supplied VTP/STL domain and explicit network | Approved solver | File-path portability and enclosure behavior |

Each case receives a stable ID, versioned settings, and SHA-256 hashes for every input.

## Frozen heart/Cext profile

Unless a named case is explicitly testing a different control, both sides use:

| Control | Value |
| --- | ---: |
| Cext/tissue accelerator work arrays | float32 |
| FFT background grid | 256 per axis |
| Cext quadrature | 1 |
| Tissue quadrature | 5 |
| Vessel/Cext coupling iterations | 1 |
| Interaction window factor | 6 |

The runtime-stencil and runtime-moment GPU paths remain in scope. Growth is not exercised here. A later growth campaign will retain public-SVV CCO in float64, transition to float32 at the default 300,000-terminal equal-bifurcation boundary, and test the upstream optimizer selector when available.

## Functional checks

For each CLI workflow, begin outside the source checkout and use only installed entry points:

```bash
<clean-cascade-venv>/bin/cascade --version
<clean-cascade-venv>/bin/cascade doctor --no-gpu-probe
<clean-cascade-venv>/bin/cascade run --settings validation/fixtures/<case>.json
<clean-cascade-venv>/bin/cascade export-heart --forest <forest> --domain <domain> --out-dir <output>
<svva2>/bin/python <svva2>/lib/python3.9/site-packages/svv/SCRIPTS/<oracle>.py <legacy-args>
```

For heart tissue parity, the validation-only legacy wrapper first captures the
oracle's inside-domain coordinates, then supplies that unchanged `.npy` array on
all subsequent oracle runs. CASCADE receives the identical file through
`--tissue-points`. The fixture hash, point count, units, and source grid are
recorded with every result. Domain-mask counts may be reported diagnostically,
but they are not allowed to change the pointwise solver workload.

Run the two commands in separate processes and output directories. The comparison layer reads their artifacts after both processes exit; it does not import either solver implementation.

For pointwise cube comparisons, both settings use one frozen NPY coordinate file containing one million seed-42 legacy-domain points. Leading prefixes are allowed only for debugging; final cube correctness consumes the complete array. Heart comparisons independently construct the same deterministic `200 x 200 x 200` bivent3 candidate grid and require exact equality of the retained coordinates and inside-domain mask. Large computational cases disable per-segment CSV and VTK unless the case is specifically testing export; compact summaries and streaming array-comparison statistics prevent Python object materialization from consuming the host budget.

Validate:

- Exit code and actionable errors.
- Expected output set.
- Manifest completeness and hashes.
- Reloadability of saved tree/forest/cache.
- VTP/VTU readability.
- Field names, dtypes, units, point/cell association, and IDs.
- Relative path handling from settings-file location.
- No dependence on checkout path, current working directory, or user cache.

## Numerical comparison layers

### Structure

Compare exactly unless a documented mapping is required:

- Tree/forest counts and per-tree segment counts.
- Parent/child topology.
- Root and terminal identities.
- Segment ID mapping and export global IDs.
- Endpoints and radii, with an explicit geometry tolerance if optimization introduces floating-point variation.
- Tissue sample coordinates and inside-domain mask.

### Hemodynamics

Compare:

- Flow per segment and inlet/terminal conservation.
- Pressure fields and boundary conditions.
- Resistance and reported summary metrics.
- Hematocrit fields and iteration diagnostics.

### Oxygen and Cext

Compare:

- Segment inlet/outlet concentration.
- Wall/intravascular/extravascular concentration fields as applicable.
- Source strengths and finite-radius terms.
- Tissue oxygen at identical coordinates.
- Aggregate min, percentiles, mean, max, norms, and conservation/flux diagnostics.

Before tests are called certifying, document for every field:

- Exact equality, absolute tolerance, relative tolerance, or distributional criterion.
- NaN/Inf policy.
- dtype conversion policy.
- Whether ordering is significant or an ID mapping is applied.

The approved functional rule is 0.1% agreement for the physical conclusion: oxygenation statistics/percentiles, segment and total flow, pressures, viability, and `FracAbove1pct`. Fractions use an absolute 0.001 limit (0.1 percentage points). Other nonzero physical values use relative tolerance `1e-3`; near zero they use absolute tolerance `1e-6` times the field's frozen reference-case scale. The comparison report records that scale and resulting absolute tolerance per field. Exact topology, IDs, shared coordinates, ordering/mapping, and finite/non-finite masks are not relaxed.

Kirchhoff solver arrays on both sides use `dyn/cm^2`. Array evidence converts them to pascals before comparison, and CASCADE public summary/CSV/VTK pressure fields are also converted to pascals. Resistance metrics retain their documented legacy cgs interpretation unless a separately named SI field is added.

## Performance protocol

The benchmark must execute legacy and CASCADE cases in alternating/interleaved order to reduce thermal and background-load bias, but strictly sequentially: legacy exits and releases memory before CASCADE starts, and vice versa.

Record:

- Host, WSL/kernel, CPU, RAM, GPU, driver, CUDA, Python, and package inventory.
- Thread variables and worker counts.
- GPU power/performance state when available.
- Input hashes and exact settings.
- Warmup count and measured run count.
- Per-run wall time plus component timings.
- Peak host memory and peak GPU memory where practical.
- First-run compilation separately from warmed runs.
- Explicit CUDA synchronization at timing boundaries.

Report median, spread, minimum, maximum, and CASCADE/legacy ratio. The target is “no slower than `TissueSim_cube_local` for equivalent work.” The statistical/noise rule used to interpret “equivalent” must be approved before the gate is closed.

Report solver-only and end-to-end measurements separately. Solver-only starts from the already loaded identical structure. End-to-end includes explicit input load, float32 preparation, solve, and requested export. Both are release evidence; solver-only is the direct computational-method comparison and end-to-end identifies operational overhead.

Measured repetitions alternate legacy/CASCADE while remaining strictly sequential: five paired runs for each cube scale through 100k terminals, three pairs at 1M, and one initial pair at 5M and HEART-L. Repeat a large-case pair when it fails, reports memory pressure, or its time ratio lies within 5% of the release threshold. Small and medium cases include CPU/GPU consistency; the 1M, 5M, and HEART-L production cases are GPU-only. Full export performance is measured on bounded cases first; a full HEART-L export is a separate monitored run after solver correctness and memory gates pass.

Investigate rather than average away:

- Different geometry preparation.
- Extra serialization or VTK output.
- Different sample queries or candidate counts.
- Different convergence work.
- JIT compilation or cache reuse.
- CPU oversubscription.
- GPU transfer, allocation, or synchronization overhead.

## Negative and recovery tests

Exercise at minimum:

- Unknown/misspelled settings.
- Missing or unreadable domain/network files.
- Invalid CSV columns and disconnected/custom topology.
- Incompatible cache metadata.
- Requested GPU with no CuPy/device/component libraries.
- Invalid grid dimensions and impossible solver settings.
- Interrupted GUI/CLI jobs and partial output directories.
- Corrupt VTK or forest input.

Failures must be non-zero, preserve the root cause, and avoid presenting partial outputs as successful runs.

## Evidence outputs

Each campaign creates:

```text
validation/
  fixtures/                 versioned small/non-private inputs
  results/<campaign-id>/    tracked JSON/CSV/Markdown summaries
  runs/<campaign-id>/       ignored heavy raw output
```

The result summary must link the release tracker IDs it satisfies. Failed campaigns are retained with status `failed` or `inconclusive` and an explanation.

## Completion rule

Step 2 is complete only when all in-scope VAL and PERF items in `release-tracker.md` are either COMPLETE or explicitly DEFERRED/ACCEPTED with the owner’s rationale. Smoke tests alone cannot close numerical or performance gates.
