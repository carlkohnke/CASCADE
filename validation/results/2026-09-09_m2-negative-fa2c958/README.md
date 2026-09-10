# M2 negative-path preflight (fa2c958 wheel)

This append-only preflight executed ten independent failure cases through the
installed `cascade` entry point in
`validation/tmp/m2-wheel-fa2c958-py39`, outside the source checkout.

Seven cases passed immediately. Unknown top-level and nested settings, missing
and malformed custom inputs, disconnected custom geometry, incompatible flow
settings, corrupt VTK, corrupt forest input, and a requested GPU hidden with
`CUDA_VISIBLE_DEVICES=-1` all exited non-zero without writing a successful
manifest. Two of those correct failures were initially counted as harness
misses because the expected substrings were more specific than CASCADE's actual
`Unknown CASCADE setting(s)` wording.

The preflight found one product defect: a grid axis of zero was silently clamped
to one and the command succeeded. `negative-matrix-initial.json` preserves this
failed gate. The source fix validates grid dimensions before execution and is
covered by parameterized regression tests. The final negative matrix must be
rerun against a wheel built from the committed fix; this historical result is
not overwritten.
