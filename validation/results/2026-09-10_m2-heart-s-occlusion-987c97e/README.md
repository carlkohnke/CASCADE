# HEART-S occlusion rerun after anatomical-radius repair

Tracker IDs: VAL-06, VAL-07.

This campaign reran the full-grid HEART-S downstream-occlusion case from the
exact wheel built at commit `987c97e`. The frozen point file contains 1,619,996
inside-domain coordinates from the 200-cubed candidate grid and has SHA-256
`ab733bd07827e33e7458f9e474a5b6f6ec12f8eff45cf41c4bd00c268cbc9fdb`.

The repaired CASCADE output preserves the frozen anatomy and finite mask. The
coordinate, identity, point-count, viability-fraction, and finite-mask checks
pass. Seven of approximately 1.56 million accelerator-field values exceed the
ordinary pointwise float32 rule; the largest population fraction is
`3.20e-6`. These are isolated GPU spatial-cell boundary choices and pass the
explicit D-042 sparse-outlier rule. The strict pre-policy comparison is retained
beside the accepted report.

Commands and resource data are in `cascade-monitor.json`; large VTP/log output
remains under ignored `validation/runs/2026-09-10_m2-heart-s-grid200-ce6f48c/`.
