# M2 clean-environment reproducibility certification

CUBE-100 blood was run sequentially from two independently created Python 3.9
virtual environments. Each installed the exact `128c6fc` wheel and the frozen
CPU constraints, loaded the same hashed external tree, and used the same hashed
one-million-point tissue coordinate file.

Results:

- All 17 array comparisons passed with zero mismatched values and zero numerical
  error, including topology, geometry, flow, pressure, hematocrit,
  concentration, retained tissue coordinates, and tissue oxygen.
- All 23 summary fields passed with zero numerical error.
- All 90 stable manifest leaves matched. The comparator explicitly excludes
  timestamps, environment-specific executable/output paths, settings hashes
  changed only by those output paths, prefixes, and measured timings.
- Evidence-enabled wall times were 32.89 and 32.49 seconds; these are not M3
  benchmark measurements.
- Peak host RSS was approximately 2.83 GiB in both runs.

Inputs/artifact:

- Commit: `128c6fc`
- Wheel SHA-256: `51f75bf6accb916f1aa6b86861ef43d76491107bd0dbbd5f2622850fdd077608`
- Tree SHA-256: `ad547a2d40ddfcc93880d0dd51860c4390c67eb818540b72fb01a27ef5218283`
- Tissue points SHA-256: `251ca61e9a082c3ae10f46a882a77fde7cacc0312faaf0b2834d14e72004e6b7`

Compact comparisons and settings are stored here. Arrays, logs, and emitted
manifests remain under the matching ignored `validation/runs/` campaign.
