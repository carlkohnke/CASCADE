# CASCADE repository working agreement

Before doing release, compatibility, numerical-validation, performance, or packaging work, read `docs/internal/README.md` and `docs/internal/release-tracker.md`.

For every material production-release or testing change:

1. Update the relevant item in `docs/internal/release-tracker.md`.
2. Add a dated entry to `docs/internal/work-log.md` describing what changed, how it was verified, and where the evidence lives.
3. Record durable design or process choices in `docs/internal/decision-log.md`.
4. Store compact, reviewable validation results under `validation/results/`; keep large generated outputs under ignored `validation/runs/`.
5. Do not mark a gate complete without a command, artifact, test report, or commit that another person can inspect.
6. Do not overwrite historical release reports. Create a new report for a new release or validation campaign.
7. Keep CASCADE compatibility behavior local by default. Propose upstream `svv` changes only when the required behavior must execute inside an upstream loop/primitive or is a shared interchange contract.

Release artifacts must not contain local caches, generated runs, internal process notes, user-specific absolute paths, or private data.
