# CASCADE release runbook

This is the repeatable operator procedure. Replace `<version>` and paths deliberately; do not use broad deletion commands against the repository root.

## 1. Establish the candidate

```bash
git status --short --branch
git log --oneline --decorate -10
python3.9 --version
```

Require a clean worktree or document every pre-existing change. Record the candidate commit in the release tracker before testing.

## 2. Create the CPU development environment

```bash
python3.9 setup_env.py \
  --venv /path/to/cascade-cpu-venv \
  --dev \
  --gui \
  --constraints locks/requirements-py39-cpu.txt

/path/to/cascade-cpu-venv/bin/cascade doctor --no-gpu-probe
/path/to/cascade-cpu-venv/bin/python -m pytest -q
```

Record test count and warnings. Warnings are tracked work, even when they do not fail the build.

## 3. Create the CUDA environment

```bash
python3.9 setup_env.py \
  --venv /path/to/cascade-cu13-venv \
  --dev \
  --gui \
  --gpu cu13 \
  --constraints locks/requirements-py39-cu13.txt

/path/to/cascade-cu13-venv/bin/cascade doctor --require-gpu
```

The doctor gate must execute both a compiled CuPy operation and cuFFT. A device-count-only check is insufficient.

## 4. Exercise source workflows

Run representative settings through the public CLI, not private Python helpers:

```bash
cascade run --settings examples/cube_tree_smoke.json
cascade run --settings examples/cube_forest_smoke.json
cascade run --settings examples/cube_forest_nearest_smoke.json
cascade run --settings examples/lattice_cubic_smoke.json
cascade run --settings examples/simple_channel_smoke.json
cascade run --settings examples/custom_geometry_smoke.json
```

Run the packaged heart exporter with a fixed forest/domain for geometry-only, CPU tissue, and CUDA shared-Cext modes. Save the exact command in the release report.

Open every emitted VTP/VTU programmatically with PyVista and manually inspect representative production outputs in ParaView.

## 5. Exercise the GUI

At minimum:

1. Construct and show CASCADE Studio.
2. Enter the Qt event loop and exit normally.
3. Create/load/save a project.
4. Preview a tree and forest.
5. Execute a queued job.
6. Cancel a running job and verify state/output handling.
7. Load results in the viewer.

Offscreen startup is automation evidence; it does not replace the full human interaction check.

## 6. Build artifacts

After verifying the exact artifact paths under `dist/`, replace only the candidate’s prior wheel/sdist and run:

```bash
python -m build
python -m twine check \
  dist/cascade_vascular-<version>-py3-none-any.whl \
  dist/cascade_vascular-<version>.tar.gz
```

Audit archive contents. Reject caches, generated runs, backups, internal process notes, compiled Python files, private inputs, user paths, and a legacy `gfm/` package.

## 7. Test the exact wheel outside the checkout

CPU:

```bash
python scripts/release_smoke.py \
  --wheel dist/cascade_vascular-<version>-py3-none-any.whl \
  --constraints locks/requirements-py39-cpu.txt
```

CUDA 13:

```bash
python scripts/release_smoke.py \
  --wheel dist/cascade_vascular-<version>-py3-none-any.whl \
  --constraints locks/requirements-py39-cu13.txt \
  --gpu-extra gpu-cu13
```

The smoke runner installs into temporary environments, invokes installed console scripts, runs tree/custom/forest cases, runs the heart exporter, and reopens VTK outputs.

If any source or packaged file changes after these commands, rebuild and repeat the exact-wheel tests.

## 8. Record and tag

```bash
sha256sum \
  dist/cascade_vascular-<version>-py3-none-any.whl \
  dist/cascade_vascular-<version>.tar.gz

git status --short --branch
git tag -a v<version> -m "CASCADE <version>"
```

Write a release-specific report containing environment, commands, results, known limitations, checksums, commit, and tag. Verify the tag points to the intended commit.

## 9. Publish only after governance gates

Before a public push/release:

- Select and add a compatible project license.
- Review the archive for private data and derived-code obligations.
- Configure the GitHub remote.
- Push the branch and tag intentionally.
- Observe the clean GitHub Actions result.
- Attach only checksum-verified artifacts from the tagged commit.

Do not rewrite or move a tag that has been published.
