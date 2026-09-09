# CASCADE Step 2 certification plan

Status: planned; acceptance tolerances and canonical production scales require confirmation before execution.

## Objective

Demonstrate that CASCADE performs the same scientific work as the selected legacy TissueSim references when given identical structures, domains, settings, and sample points, and that its execution time is equivalent or faster under a controlled comparison.

The primary legacy references are:

- `TissueSim_cube_local.py` for tree simulation and tissue oxygen behavior.
- `export_paraview_heart_forest_grid_cext_gfm.py` for heart forest, shared Cext, tissue grid, and ParaView export behavior.
- Custom channel/domain workflows only as geometry-input capability tests; their extra plotting and profile analyses are not release requirements.

Legacy scripts are test oracles, not production imports.

## Principles

1. Compare identical work. Geometry, domain, flow inputs, fluid model, solver options, sample points, dtypes, stopping criteria, and output requests must match.
2. Correctness precedes performance. A faster result is irrelevant if fields or solver behavior differ outside approved tolerances.
3. Separate deterministic identity from floating-point agreement. IDs/topology/counts may require exact equality; physical fields generally require explicit absolute/relative tolerances.
4. Separate cold-start, warm solver, and end-to-end timings.
5. Do not silently repair one side only. Any normalization or repair must be applied symmetrically and recorded.
6. Preserve failures and regressions as evidence rather than rerunning until a favorable sample appears.

## Environment matrix

| Environment | Purpose | Required baseline |
| --- | --- | --- |
| CPU release environment | Packaging, CPU solver, deterministic/reference checks | Python 3.9.20 and `locks/requirements-py39-cpu.txt` |
| CUDA 13 release environment | GPU Cext/tissue and performance checks | Python 3.9.20 and `locks/requirements-py39-cu13.txt` |
| Legacy oracle environment | Execute selected reference scripts | Freeze and record its full package inventory; do not let it leak into CASCADE imports |
| GitHub CPU CI | Regression checks on clean checkout | Activate after remote is configured |

## Canonical case matrix

Final sizes and fixtures are selected before running. This is the proposed minimum matrix.

| Case | Structure/domain | Solver path | Purpose |
| --- | --- | --- | --- |
| CUBE-S | Fixed cube tree, small terminal count | CPU top-down + tissue GFM | Fast exact/debug reference |
| CUBE-M | Fixed cube tree, medium terminal count | CPU and CUDA where applicable | Functional and scaling comparison |
| CUBE-L | Fixed cube tree, production-representative terminal count | Target production solver | Performance and memory gate |
| FOREST-S | Fixed two-tree cube forest | CPU, cache reload | Forest mapping/connectivity/cache parity |
| HEART-S | Fixed small heart forest and domain | Shared-global CUDA FFT/Cext | Debuggable exporter parity |
| HEART-P | Production-representative heart forest/grid | Target CUDA configuration | Performance, memory, and VTK certification |
| CUSTOM-Y | Explicit CSV Y channel in box domain | CPU tissue GFM | Custom topology/units/export correctness |
| CUSTOM-DOMAIN | Fixed user-supplied VTP/STL domain and explicit network | Approved solver | File-path portability and enclosure behavior |

Each case receives a stable ID, versioned settings, and SHA-256 hashes for every input.

## Functional checks

For each CLI workflow, begin outside the source checkout and use only installed entry points:

```bash
cascade --version
cascade doctor --no-gpu-probe
cascade run --settings validation/fixtures/<case>.json
cascade export-heart --forest <forest> --domain <domain> --out-dir <output>
```

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

## Performance protocol

The benchmark must execute legacy and CASCADE cases in alternating/interleaved order to reduce thermal and background-load bias.

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
