# Command inventory

Commands were executed from `/home/carl/svv_sweeps/GFM` under WSL Ubuntu. The
release smoke runner creates independent temporary virtual environments outside
the source checkout and deletes them only after the run completes.

```text
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m ruff check --select E9,F63,F7,F82 src tests scripts
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m compileall -q src/cascade
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m pytest -q
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m build
/home/carl/.venvs/cascade-0.1.0rc1/bin/python -m twine check dist/cascade_vascular-0.1.0rc3-py3-none-any.whl dist/cascade_vascular-0.1.0rc3.tar.gz
/home/carl/.venvs/cascade-0.1.0rc1/bin/python scripts/release_smoke.py --wheel dist/cascade_vascular-0.1.0rc3-py3-none-any.whl --constraints locks/requirements-py39-cpu.txt
/home/carl/.venvs/cascade-0.1.0rc1/bin/python scripts/release_smoke.py --wheel dist/cascade_vascular-0.1.0rc3-py3-none-any.whl --constraints locks/requirements-py39-cu13.txt --gpu-extra gpu-cu13
sha256sum -c dist/SHA256SUMS
git rev-parse v0.1.0rc3^{}
```

Additional direct checks imported all 69 non-executable package modules and
ran `topdown_ext_treecode` on CUDA with both `decoupled_greens` and `zero`
initialization. The smoke runner itself checks `pip check`, distribution entry
points, version output, generated starter execution, installed self-test,
separate doctor/export-heart/viewer commands, GUI import, tree/custom/forest
workflows, CPU or CUDA heart Cext, VTK reopening, manifests, and public-SVV
file hashes.
