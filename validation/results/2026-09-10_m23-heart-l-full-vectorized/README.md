# HEART-L full-grid GL5 production preflight

Tracker IDs: VAL-06, VAL-07, PERF-05, PERF-06, PERF-07.

This source-tree optimization campaign evaluates the frozen 24,999,999-segment
HEART-L simulation cache on all 1,619,996 frozen inside-domain points derived
from the 200-cubed candidate grid. Both sides use GPU, float32 accelerator
fields, a 256-cubed FFT background, Cext GL1, independent tissue GL5, one Cext
iteration, and window factor 6. Vessel VTP construction is intentionally
suppressed because its schema and values are certified on HEART-S; the full
tissue and domain products are written.

The first CASCADE attempt is retained as an interrupted regression: a Python
row loop in GL1-to-GL5 interpolation would have performed about 175 million
iterations. Vectorizing the mathematically constant GL1 interpolation reduced
the completed CASCADE application time to 132.83 seconds, versus 179.24 seconds
for the isolated corrected-GL5 legacy oracle (CASCADE/legacy ratio `0.741`).
Monitored wall times were 136.63 and 182.74 seconds. Peak RSS was 33.76 GiB for
CASCADE and 36.98 GiB for the oracle, both below the 45 GiB limit.

After exact-coordinate intersection, 1,501,705 tissue output points are common;
each side has six unique vessel-boundary points (`3.996e-6` of its output).
Only 151 normalized-concentration values (`1.006e-4`) exceed the ordinary
pointwise float32 tolerance. `FracAbove1pct` and `FracAbove5pct` each differ by
only `6.66e-7`, and one viability value differs among the common points. The
comparison passes the D-042 distributional boundary rule. The unaligned failed
comparison is retained to show why row-order alignment is required.

This is source-candidate evidence. Final certification requires one repeat from
the exact installed wheel containing these fixes.
