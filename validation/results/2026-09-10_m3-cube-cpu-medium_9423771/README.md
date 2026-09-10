# Superseded CPU medium-scale timing extension

Status: interrupted and superseded by D-035. Tracker IDs: PERF-02 and PERF-03.

This campaign began extending the clean CPU matrix beyond the owner-approved
small CPU range. It completed five sequential alternating pairs at target 1,000
and three pairs at target 10,000, with every emitted scientific summary passing.
The median fresh-process CASCADE/legacy ratios were approximately 0.967 and
0.983, respectively. Target-10,000 repetition 4 was stopped before either
solver completed when the owner clarified that CPU testing should end at target
100 and GPU is the release goal thereafter.

The completed compact pair/settings/comparison records are retained rather than
discarded. No `benchmark-summary.json` exists because the campaign was
interrupted. Heavy logs and the partial final run remain under the matching
ignored `validation/runs/` directory. These observations are diagnostic only
and are not used to close the GPU performance gate.
