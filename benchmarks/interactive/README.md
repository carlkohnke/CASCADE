# Interactive performance benchmarks

Run the persistent worker from the repository root:

```bash
PYTHONPATH=src SVV_CEXT_GPU_POOL_TRIM=false \
python -m cascade.commands.main worker
```

Then send one JSON request per line. Repeating the request measures an exact
result replay; changing a scientific setting such as `qin_target_ul_min`
measures a new solve while retaining compatible geometry and CUDA state.

```json
{"id":"run-1","command":"run","settings":"/absolute/path/to/case.json"}
```

The CPU dense-path comparison is reproducible with:

```bash
PYTHONPATH=src python benchmarks/interactive/benchmark_dense_tissue_cpu.py \
  /path/to/cascade-arrays --points 50000
```

The benchmark prints timing and numerical-parity results as JSON. For worker
benchmarks, use a normal settings file whose output section writes only summary
artifacts; that makes its last exact result eligible for bounded replay without
embedding machine-specific fixture paths in this repository.

For a normal shell workflow, the same reuse is available without speaking the
worker protocol:

```bash
PYTHONPATH=src python -m cascade.commands.main batch \
  --settings case-001.json case-002.json case-003.json
```

Use `--continue-on-error` for an unattended study that should continue with
later cases after one configuration fails.
