# Simulation-only tree-load optimization

Candidate: commit `fa2c958`, exact wheel SHA-256 `de21614b728dd3963db83176e3c7c99ed45b8f2342ea83a6c59985f24565f8cf`.

The optimized installed wheel was compared with the frozen legacy arrays already emitted from the same SHA-256-checked trees and shared one-million-point coordinate fixture. Both targeted blood cases passed all 17 array and 23 summary comparisons, with zero exact mismatches and zero tolerance failures.

| Terminals | Segments | Old load (s) | New load (s) | Load speedup | Old peak RSS (GiB) | New peak RSS (GiB) | Array/summary status |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1,000,000 | 2,000,001 | 16.602 | 2.052 | 8.1x | 5.15 | 2.05 | pass/pass |
| 5,000,000 | 10,000,001 | 61.559 | 8.124 | 7.6x | 16.65 | 6.48 | pass/pass |

The change avoids decompressing and unpickling the growth-only payload in a `.tree.npz` file when growth is disabled. Growth-enabled loading is unchanged. These are single evidence-enabled observations, not the repeated M3 performance result.
