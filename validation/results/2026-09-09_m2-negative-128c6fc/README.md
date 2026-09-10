# M2 negative-path certification (128c6fc wheel)

All ten failure cases passed through an independently installed CPU wheel
environment outside the checkout. Every case exited non-zero, included its
expected root-cause text, and emitted no successful `manifest.json`.

The matrix covers unknown top-level and nested settings, missing and malformed
custom inputs, disconnected geometry, zero grid extent, incompatible flow
settings, corrupt VTK and forest files, and a GPU requirement with no visible
device. The previously successful zero-grid case is preserved in the earlier
`2026-09-09_m2-negative-fa2c958` campaign rather than overwritten.

- Commit: `128c6fc`
- Wheel SHA-256: `51f75bf6accb916f1aa6b86861ef43d76491107bd0dbbd5f2622850fdd077608`
- Python: 3.9.20
- Public `svv`: 0.0.48
- Result: `negative-matrix.json` (`10/10`, pass)
- Raw malformed fixtures and any partial directories: ignored matching
  `validation/runs/2026-09-09_m2-negative-128c6fc/`
