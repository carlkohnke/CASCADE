# M0 existing-tree fixture and memory report

- Status: fixture inventory complete; shared coordinate files and field-level tolerance floors are staged during M2
- Source baseline: branch `release/cascade-0.1`, starting HEAD `20744c89f06437ad81703a53a758b8ce477f8d9e`
- Tracker IDs: M0-05, M0-06, M0-07, M0-10, VAL-01, VAL-02, PERF-01, PERF-02

## Result

The current legacy cache selector recognizes stored family `a50c006ac491d07e399f94f1be37cfe0` through its compatibility-key matcher. Exact cube inputs are frozen for 1, 10, 100, 1k, 10k, 100k, 1M, and the highest available target, 5M. No 10M cube tree exists in the inspected cache hierarchy, and no tree was built.

HEART-S is the approved 10k-terminal, 19,999-segment two-tree forest. HEART-L is the one-millimetre extended 12.5M-terminal production structure, with its stored 6.2 GB simulation cache preferred for loading. Heart coordinates/radii use centimetres.

CASCADE now accepts fixed CSV/NPY/NPZ tissue coordinates through settings and Studio and records their hash in manifests. CLI simulation entry points reject overlap through a per-user OS lock, while sequential sweeps explicitly release completed result and accelerator state. No-growth tree/forest inputs avoid growth-only duplicate buffers and spatial indexes; simulation-cache arrays stream directly into float32 working storage for the default heart workflow. The validation protocol additionally waits for each legacy or CASCADE subprocess to exit before launching the next.

Focused loader, locking, fixed-sample, Studio, and release-contract tests passed (15 tests). The full source suite passed (109 tests, 15 known warnings).

A temporary final-candidate wheel and sdist passed `twine check` and content audit (77 and 149 entries respectively). The wheel SHA-256 was `9ee7a7e0b49e1363f89629d02589f7faa3212789eef39cc948bed6859527da25`; that exact wheel passed independent CPU and CUDA 13 release smokes outside the checkout, including forest simcache loading, CPU heart export, GPU shared-global FFT/Cext heart export, cuFFT/device checks, VTK reopen, and manifest validation. These temporary artifacts are verification products, not replacements for the tagged `v0.1.0rc1` artifacts.

No numerical simulation or tree growth was run in this inventory campaign. No correctness or performance certification is claimed.
