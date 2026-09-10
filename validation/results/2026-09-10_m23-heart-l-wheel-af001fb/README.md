# HEART-L exact-wheel certification

Tracker IDs: VAL-06, VAL-07, PERF-05, PERF-06, PERF-07.

Candidate commit: `af001fb6b29b0ff028fc30ceedf6c214ced3756b`.
Installed wheel SHA-256:
`b5f69ca1a79a3e35a908c23609b859ea37b1d7871587360862918c96b27ece65`.

The wheel was force-installed without dependencies into the isolated CUDA 13
validation environment, confirmed to import from `site-packages`, and passed
`cascade doctor --require-gpu`. It then completed the frozen 24,999,999-segment
HEART-L workflow on all 1,619,996 shared heart points using float32, FFT grid
256, Cext GL1, tissue GL5, one Cext iteration, and window factor 6.

CASCADE application time was 141.37 seconds and monitored wall time was 145.21
seconds, versus 179.24 and 182.74 seconds for the isolated corrected-GL5 oracle.
The application-time ratio is `0.789`; peak RSS was 33.83 GiB versus 36.98 GiB.

The coordinate-aligned comparison passes D-042. There are 1,501,705 common
tissue points and six unique vessel-boundary points per side. The largest
physical outlier population is 152 normalized-concentration values
(`1.012e-4`); one viability value differs. Both `FracAbove1pct` and
`FracAbove5pct` differ by `6.66e-7`.

The monitor and comparator reports are tracked here. Large output and raw logs
remain under ignored `validation/runs/2026-09-10_m23-heart-l-wheel-af001fb/`.
