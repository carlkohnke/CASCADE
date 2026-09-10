# CASCADE 0.1.0rc2 release-candidate report

Validation date: 2026-09-10

## Result

CASCADE 0.1.0rc2 is the locally qualified release candidate for the selected
existing-tree computational scope. M0, M1, M2, and M3 are complete. M4 is not
included: no Git remote or license has been selected, and remote CI has not been
observed.

The qualified platform is WSL2 Ubuntu/Linux x86-64, Python 3.9.20, public
`svv==0.0.48`, and CUDA 13 with CuPy 13.6.0 on an NVIDIA GeForce RTX 3080 Laptop
GPU. CPU qualification is intentionally limited to small cube and release-smoke
workflows; GPU is the production backend above 100 cube terminals and for every
heart case.

## Numerical certification

- Identical cached cube structures passed blood and cell-media comparisons at
  requested terminal counts 1, 10, 100, 1k, 10k, 100k, 1M, and 5M (10,000,001
  actual segments at the largest available case).
- CPU/GPU consistency, negative inputs, deterministic reproducibility, custom
  CSV/NPZ geometry, file-backed domains, and VTK schemas passed their frozen
  acceptance rules.
- HEART-S healthy and full downstream occlusion passed on 1,619,996 frozen
  inside-domain coordinates from the full 200-cubed candidate grid.
- HEART-L passed with the frozen 24,999,999-segment one-millimetre extended
  simulation cache, Cext GL1, independent tissue GL5, float32 accelerator
  arrays, a 256-cubed FFT background, one Cext iteration, and window factor 6.
- At HEART-L, `FracAbove1pct` and `FracAbove5pct` each differed from the isolated
  corrected-GL5 legacy oracle by `6.66e-7`. Six vessel-boundary coordinates per
  side differed after float32/float64 geometry classification; 152 of 1,501,705
  aligned normalized-concentration values exceeded the ordinary pointwise
  float32 rule. These sparse counts pass the documented distributional rule;
  topology and segment identity are not relaxed.

## Performance certification

- Warming a bounded sequential GPU worker gives approximately 0.10-0.15 second
  small cached-tree solves under the historical measurement boundary.
- All paired GPU cube cases passed scientifically. At 10,000,001 actual
  segments, median comparable compute was 26.16 seconds for CASCADE versus
  27.65 seconds for the current `svva2` oracle (ratio 0.947).
- The exact installed pre-rc2 wheel completed full HEART-L application work in
  141.37 seconds and 145.21 seconds monitored wall time, versus 179.24 and
  182.74 seconds for the isolated legacy oracle. Peak RSS was 33.83 GiB versus
  36.98 GiB, below the 45 GiB validation limit.

## Scope and remaining non-M2/M3 items

This certification validates computation on existing trees. Scientific growth
qualification is deliberately later: public SVV CCO remains float64, CASCADE
retains the future float32 transition at equal-bifurcation growth, and the owner
will request an upstream SLSQP/L-BFGS-B selector. CUDA 11/12, non-Linux
platforms, human end-to-end Studio acceptance, a project license, GitHub remote,
and observed remote CI are not claimed by this release candidate.

Compact evidence and exact commands are tracked under `validation/results/` in
the repository; large structures and outputs remain external or under ignored
`validation/runs/` and are not included in release archives.
