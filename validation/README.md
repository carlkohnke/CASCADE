# CASCADE validation evidence

This directory holds reproducible numerical, functional, and performance evidence.

```text
validation/
  fixtures/              small, non-private, versioned canonical inputs
  results/               compact tracked reports and comparison tables
  runs/                  ignored heavy solver output
  tmp/                   ignored disposable intermediate data
```

Use campaign IDs of the form `YYYY-MM-DD_<scope>_<commit>`, for example:

```text
2026-09-10_cube-parity_9da8b2e
```

Every result must identify the corresponding tracker IDs from `docs/internal/release-tracker.md`, exact command, Git commit, input hashes, environment, hardware, acceptance criterion, result, and raw-output location.

Do not commit private anatomical data or large VTK/NumPy/cache artifacts. If raw evidence is retained outside the repository, record its stable storage location and checksum without embedding credentials.
