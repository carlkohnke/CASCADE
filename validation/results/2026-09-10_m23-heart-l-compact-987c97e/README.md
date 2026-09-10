# HEART-L initial compact failure

Tracker IDs: VAL-07, PERF-05, PERF-06.

The exact `987c97e` wheel loaded the frozen HEART-L simulation cache in float32
and then rejected duplicate child references. The first affected identifiers
were just above `2**24`, demonstrating that integer topology embedded in the
float64 SVV vessel table had been rounded during float32 conversion. The run
failed before a solve, peaked at 4.03 GiB RSS, and is retained rather than
overwritten. D-043 and the succeeding connectivity campaign record the fix.
