# M0 inventory normalization report

- Status: partial M0 closure; settled inventory work complete, owner/scientific choices retained as open
- Tracker IDs: M0-01 through M0-09, VAL-01, VAL-08, PERF-02
- Source baseline: branch `release/cascade-0.1`, starting HEAD `20744c89f06437ad81703a53a758b8ce477f8d9e`
- Tagged release baseline: `v0.1.0rc1` at `9da8b2e`
- Host context: WSL2 Ubuntu/Linux x86-64

## Result

The legacy oracle boundary is now unambiguous: hashed programs in the `svva2/SCRIPTS` installation run in their own Python environment, while CASCADE runs only from a clean wheel environment. The tracked root script duplicates, obsolete same-process comparison helpers, and automatic SCRIPTS cache relocation were removed. No legacy path is imported by CASCADE.

The production heart input is the one-millimetre extended two-tree forest with 12,500,000 terminals and 24,999,999 segments. Its file hash and the associated legacy domain-cache hash are frozen in `docs/internal/m0-legacy-inventory.md`. The approved bivent3 STL was copied into the CASCADE package without byte changes and exposed through the ordinary file-domain path in Studio.

PyVista read the packaged STL successfully as 6,467 points and 13,050 cells with bounds approximately x `[-4.5746, 4.5671]`, y `[-3.9323, 3.9330]`, and z `[-4.5161, 4.5150]`. This proves file readability, not the still-open unit interpretation.

The frozen heart/Cext profile is float32 accelerator work arrays, grid 256, Cext quadrature 1, tissue quadrature 5, one Cext coupling iteration, and window factor 6. CASCADE defaults and tests were normalized to this profile. Runtime-stencil/runtime-moment GPU functions remain present.

`.dmn` is classified as legacy/internal load compatibility, not public interchange. STL/PyVista-readable meshes are the public input path and VTP/VTU are the public ParaView output path.

## Historical limitation retained

The old aggregate cube comparisons ran primarily with `svv==0.0.43` and are historical observations. They are not reclassified as current M2 certification. The exact cube module used by the June production heart export is not recoverable from its unhashed stale path, so M2 will use the newly frozen SCRIPTS cube oracle and label it as a new baseline.

## Open items intentionally not guessed

- HEART-S forest and shared sample coordinates.
- Exact cube structure/seed fixtures at each decade.
- Input units confirmation for bivent3/legacy heart structures.
- Production simcache hash at campaign start.
- Field-by-field numerical tolerances and ordering/mapping policy.
- Public-SVV tree-array float32 behavior and the L-BFGS-B constraint disposition.

These remain visible in the release tracker; no M2 correctness or performance claim is made by this campaign.
