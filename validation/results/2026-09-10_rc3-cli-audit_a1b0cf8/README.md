# CASCADE 0.1.0rc3 CLI/package audit

Campaign ID: `2026-09-10_rc3-cli-audit_a1b0cf8`

Status: **PASS**

This campaign closes RC-03, RC-05, RC-06, RC-10, RC-12, RC-15, UX-02,
and GOV-01 for the locally qualified release scope. It audits the immutable
CASCADE `0.1.0rc3` source at commit
`a1b0cf8848c95ad85cb39f2ab3b87a872067ae9a`, tagged `v0.1.0rc3`.

## Acceptance results

- The exact wheel installs into independent CPU and CUDA 13 environments
  outside the checkout with `pip check` reporting no broken requirements.
- All five console entry points are installed: `cascade`, `cascade-gui`,
  `cascade-viewer`, `cascade-export-heart`, and `cascade-doctor`. The smoke
  exercises the non-interactive commands separately and imports the GUI.
- `cascade init-settings` emits a CPU-runnable starter, and the emitted file
  completes through `cascade run` without repository-only data.
- `cascade self-test` passes on CPU; `cascade self-test --require-gpu` executes
  compiled CuPy, GL1 vessel Cext, GPU tissue evaluation, VTK/CSV export, and
  manifest/provenance validation.
- CPU and CUDA release smokes build/load trees and forests, exercise the
  packaged heart exporter, reopen VTK outputs, and validate manifests.
- The installed public `svv==0.0.48` distribution is byte-identical before and
  after both smoke runs. CASCADE uses only process-local adapter hooks.
- The source suite passes 135 tests. High-signal Ruff, compileall, and imports
  of all 69 non-executable package modules pass.
- Wheel and sdist pass `twine check`, contain the required solver/runtime/data,
  and contain no audited private paths, external-oracle paths, generated runs,
  or arbitrary external solver loader.

## Exact artifacts

| Artifact | Bytes | Entries | SHA-256 |
| --- | ---: | ---: | --- |
| `cascade_vascular-0.1.0rc3-py3-none-any.whl` | 904,370 | 78 | `6c7411f2b3e7dca4b4d426472c0292eb6b42d872da95dcd285e6aaf25d2dab1c` |
| `cascade_vascular-0.1.0rc3.tar.gz` | 929,906 | 153 | `839d5b8f84dcbc717678aea198a33fea2ae0eecc510fbde82ff7bd7f7f36b2fc` |

## Environment

- Python 3.9.20; public `svv==0.0.48`
- WSL2 Ubuntu/Linux x86-64, kernel 6.18.33.2
- Intel Core i9-11950H, 16 logical CPUs, approximately 52 GiB RAM
- NVIDIA GeForce RTX 3080 Laptop GPU, driver 596.58, 16,384 MiB
- CUDA 13 component wheels; CuPy 13.6.0

## Raw evidence

The exact final smoke transcripts are retained outside Git under
`validation/tmp/rc3-final-cpu-smoke.log` and
`validation/tmp/rc3-final-gpu-smoke.log`. Heavy temporary environments and
generated outputs were deleted by the bounded smoke runner after success.
Structured results and the command inventory are stored beside this file.

## Preserved audit discoveries

The audit first exposed and then fixed 13 undefined-name findings, a GPU
treecode `zero`-initialization `UnboundLocalError`, incomplete analysis-only
tree reconstruction, missing forest connection restoration, a generated
starter that selected GPU by default, and an initial GPU self-test fixture that
did not satisfy the well-mixed closure contract. Regression tests and direct
branch exercises cover the corrected paths. These findings did not change the
accepted M2/M3 scientific profile or results.

## Boundaries

The wheel is self-contained for CASCADE code and bundled data, but it remains a
normal Python distribution whose declared dependencies are installed by
`pip`. Full M2/M3 oracle comparisons require the separate external `svva2`
environment and hash-frozen multi-gigabyte structures, so they remain
repository validation workflows rather than installed user commands. Growth
optimizer equivalence, the requested SLSQP/L-BFGS-B selector, high-scale
nearest-tree growth, CUDA 11/12 qualification, remote CI, licensing, and human
Studio acceptance remain outside this local CLI audit.
