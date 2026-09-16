# Command-line interface

Run `cascade --help` or `cascade <command> --help` for the current command
options.

## Quick start

Create a CPU-safe starter configuration, then run it:

```bash
cascade init-settings case.json
cascade run --settings case.json
```

Settings files use six main sections:

```json
{
  "domain": {},
  "network": {},
  "growth": {},
  "simulation": {},
  "settings": {},
  "outputs": {}
}
```

Unknown top-level and core-section keys are rejected so spelling mistakes do
not silently alter a simulation.

## Commands

| Command | Purpose |
| --- | --- |
| `cascade init-settings case.json` | Write a starter settings file. |
| `cascade inspect --settings case.json` | Report case scale without loading all scientific arrays. |
| `cascade prepare --settings case.json` | Prepare a large saved tree for memory-mapped loading. |
| `cascade run --settings case.json` | Run one simulation. |
| `cascade batch --settings a.json b.json` | Run related cases sequentially in one process. |
| `cascade sweep --settings sweep.json` | Run a configured parameter sweep. |
| `cascade doctor` | Report installation and runtime diagnostics. |
| `cascade self-test` | Run a bounded CPU installation test. |
| `cascade self-test --require-gpu` | Test the installed CUDA path. |

## Related cases and large networks

`cascade batch` reuses compatible network geometry, spatial context, and
compiled accelerator state while still running cases one at a time. Add
`--continue-on-error` when later cases should run after an individual failure.

For a large saved tree, run `inspect` first and `prepare` when a reusable
memory-mapped representation is useful. `run`, `batch`, and Studio
automatically use a compatible prepared tree and fall back to the original
archive when the prepared data is stale or incompatible.

When `prepare` uses `--cache-dir`, set `CASCADE_PREPARED_CACHE_DIR` to the same
directory for later runs.

CASCADE permits one memory-intensive simulation or preparation job per user.
A competing CLI or Studio job exits with an active-owner message instead of
running two large simulations concurrently.

When growth is disabled, existing tree and forest inputs load in analysis-only
mode without growth preallocation.

## Installation checks

`cascade self-test` verifies a small loaded-tree solve, CSV/VTK export,
manifest creation, and the installed public `svv` dependency. The GPU form
also executes CUDA Cext and tissue calculations. These checks validate the
installation; they do not validate the scientific settings of a study.

See [example simulations](examples.md) for complete settings files and
[outputs and visualization](outputs.md) for generated files.
