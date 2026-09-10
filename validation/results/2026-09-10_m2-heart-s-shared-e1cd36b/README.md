# HEART-S fresh-wheel shared-point comparison

Tracker IDs: VAL-06 and VAL-07.

The frozen tissue-capable legacy exporter and the exact installed CASCADE wheel
ran sequentially on HEART-S using the same 684-coordinate point array (SHA-256
`9298fab8e103e595c4f674298695d445fed222ca410f5cd5d67a1757b04f9001`).
Both used bivent3, blood, float32 accelerator arrays, shared-global Cext, a
256-cubed FFT background, Cext quadrature 1, requested tissue quadrature 5, one
coupling iteration, and window factor 6. The wheel SHA-256 was
`201c6459be5628cc5e79b3b2d0ae52cbdbba5c613bbe7a5a77a914476e6529c1`;
the legacy exporter SHA-256 was
`3fed343b3c2bbce84b021f55f8b174fe618a21267a6edc074c91680ee6848165`.

Both runs completed, passed their flux diagnostics, selected the same 659
finite tissue points, and produced identical viability and fraction-above-1%
(`0.0015174507`). Vessel geometry, IDs, flow, radii, and length were exact for
39,998 points and 19,999 cells. The domain boundary had the same unordered
6,048-point geometry and 12,152 cells.

The strict comparison intentionally remains failed. The legacy exporter accepts
`--gl-order 5` but its shared-Cext tissue path reuses the Cext quadrature-1 source
state. CASCADE actually resamples that state to five tissue quadrature nodes.
Consequently, six tissue values exceed the frozen field tolerance (maximum
absolute difference `0.00130024`), and 66 near-zero vessel `cext_mean` values
exceed the near-zero rule. Legacy wall time was 9.56 s versus 44.52 s for
CASCADE, but these timings represent different integration work and are not a
valid performance ratio.

`heart-vtp-comparison.json` contains the field-level result. The exact commands,
environment, log hashes, and memory records are in the two monitor JSON files;
raw VTP and logs are in the matching ignored runs folder. D-033 records the
owner choice needed before full 200-cubed HEART-S/HEART-L certification.
