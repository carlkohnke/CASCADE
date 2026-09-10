# Cube existing-tree characterization

Overall status: **PASS** (16 fluid/scale cases).

Every row used one explicit SHA-256-checked tree and the shared one-million-point coordinate fixture. Wall times include evidence-array writes and are diagnostic, not M3 certification.

| Terminals | Segments | Fluid | Backend | Arrays | Summary | Exact mismatches | Tol failures | Legacy/CASCADE wall (s) | Legacy/CASCADE RSS (GiB) | Legacy/CASCADE load (s) |
|---:|---:|:---|:---|:---:|:---:|---:|---:|---:|---:|---:|
| 1 | 3 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 8.33/8.97 | 1.20/1.17 | 0.001/0.002 |
| 1 | 3 | water | cpu | pass (17) | pass (23) | 0 | 0 | 7.84/8.15 | 0.91/0.99 | 0.001/0.002 |
| 10 | 21 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 12.25/13.24 | 1.01/1.23 | 0.001/0.003 |
| 10 | 21 | water | cpu | pass (17) | pass (23) | 0 | 0 | 11.08/11.17 | 1.09/0.96 | 0.001/0.002 |
| 100 | 201 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 33.38/34.92 | 2.75/2.75 | 0.001/0.003 |
| 100 | 201 | water | cpu | pass (17) | pass (23) | 0 | 0 | 32.92/32.85 | 2.75/2.75 | 0.001/0.003 |
| 1,000 | 2,001 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 60.86/62.26 | 3.32/3.32 | 0.003/0.012 |
| 1,000 | 2,001 | water | cpu | pass (17) | pass (23) | 0 | 0 | 60.54/58.94 | 3.31/3.31 | 0.003/0.013 |
| 10,000 | 20,001 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 64.04/63.46 | 3.32/3.33 | 0.023/0.240 |
| 10,000 | 20,001 | water | cpu | pass (17) | pass (23) | 0 | 0 | 64.57/64.40 | 3.32/3.33 | 0.019/0.261 |
| 100,000 | 200,001 | blood | cpu | pass (17) | pass (23) | 0 | 0 | 84.26/86.45 | 3.42/3.51 | 0.186/2.826 |
| 100,000 | 200,001 | water | cpu | pass (17) | pass (23) | 0 | 0 | 77.68/77.98 | 3.41/3.50 | 0.180/2.889 |
| 1,000,000 | 2,000,001 | blood | gpu | pass (17) | pass (23) | 0 | 0 | 30.67/47.94 | 1.93/5.15 | 1.840/16.602 |
| 1,000,000 | 2,000,001 | water | gpu | pass (17) | pass (23) | 0 | 0 | 25.32/43.12 | 1.81/5.15 | 1.591/17.149 |
| 5,000,000 | 10,000,001 | blood | gpu | pass (17) | pass (23) | 0 | 0 | 58.75/111.73 | 6.20/16.65 | 7.775/61.559 |
| 5,000,000 | 10,000,001 | water | gpu | pass (17) | pass (23) | 0 | 0 | 44.98/96.87 | 5.67/16.65 | 7.955/56.065 |

The maximum reported relative error is not used alone for pass/fail near zero; each physical field uses the frozen relative-or-reference-scale-floor rule, with finite-mask equality required.
