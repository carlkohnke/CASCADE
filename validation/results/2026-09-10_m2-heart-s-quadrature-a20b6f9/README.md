# HEART-S GPU quadrature diagnostic after cube speed fixes

Tracker IDs: VAL-06, VAL-07, PERF-05, PERF-06.

The tissue-capable frozen heart exporter and exact installed `a20b6f9` wheel ran
sequentially on the same 684 hashed HEART-S coordinates with the production GPU
profile. CASCADE selected `legacy_cext` only for this implementation-lineage
diagnostic. Vessel structure, geometry, IDs, flow, tissue coordinates, finite
masks, tissue oxygen, and viability passed. The first comparator result is
retained; it rejected 66 near-zero vessel `cext_mean` values whose maximum
absolute difference was only `6.51e-7`. The corrected float32 absolute-floor
rule passes in `legacy-mode-comparison-v2.json`.

The run also exposed a non-solver regression: CASCADE rebuilt/tetrahedralized the
same STL domain for about 33.5 s while the legacy domain cache loaded in about
0.05 s. The source now uses a content- and `svv`-version-keyed CASCADE user cache;
a fresh wheel must verify cold creation and warm reuse before full-grid heart
certification.
