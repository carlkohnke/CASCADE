# Corrected bounded HEART-S CASCADE run

Tracker IDs: VAL-06, VAL-07, PERF-05 (diagnostic only).

The exact installed `881d26b` candidate wheel ran HEART-S with the real
packaged bivent3 mesh, the frozen production Cext profile, and a 16-cubed
candidate tissue grid. It completed successfully, retained 658 generated
inside-domain coordinates, exported 633 finite tissue samples, and reported a
passing flux check with relative L2 error approximately `4.98e-8`.

The vessel output has 39,998 points and 19,999 cells. Against the successful
legacy pre-state-slim run it matched vessel geometry, flow, radii, lengths, and
all IDs exactly; concentration differences were at float32 scale. This is not
pointwise tissue certification because the legacy domain implementation
retained 684 candidate coordinates. D-031 replaces independent surface masks
with one frozen explicit tissue-point array.

The monitor JSON records the exact command, environment, timing, memory, and
raw-log hash. Heavy VTK and log products remain under the matching ignored
`validation/runs/` directory.
