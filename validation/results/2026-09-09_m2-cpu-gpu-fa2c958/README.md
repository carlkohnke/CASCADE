# CUBE-100 CPU/GPU consistency

Tracker ID: VAL-10. Candidate: installed wheel from commit `fa2c958`, SHA-256 `de21614b728dd3963db83176e3c7c99ed45b8f2342ea83a6c59985f24565f8cf`.

The same explicit 201-segment blood tree and the same one-million-point coordinate fixture were solved once with CASCADE's CPU tissue path and once with its CUDA path. Runs were sequential and protected by the 45 GiB monitor.

- Array comparison: pass, 17/17 arrays, zero exact mismatches and zero tolerance failures.
- Summary comparison: pass, 23/23 fields.
- Maximum tissue-oxygen absolute error: `5.428808308727273e-07`.
- Maximum tissue-oxygen relative error: `2.4554614108283122e-05`, below the approved `1e-3` limit.
- Evidence-enabled wall: CPU `31.55 s`, GPU `8.17 s`.
- Peak host RSS: CPU `2.84 GiB`, GPU `1.43 GiB`.

The wall figures include cold process startup and array evidence and are diagnostic rather than repeated M3 certification.
