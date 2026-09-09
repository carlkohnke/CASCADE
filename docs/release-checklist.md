# CASCADE release checklist

Run these gates from a clean checkout on the release commit.

1. Create the pinned CPU environment and run `cascade doctor --no-gpu-probe`.
2. Run `pytest -q` and record the pass count and warnings.
3. Execute the tree, forest, simple-channel, lattice, and custom-geometry examples through `cascade run`.
4. Open every emitted VTP/VTU with PyVista and inspect representative files in ParaView.
5. Create the pinned CUDA environment and require a compiled GPU kernel with `cascade doctor --require-gpu`.
6. Run a CPU/GPU numerical comparison with fixed geometry, settings, samples, and tolerances.
7. Launch CASCADE Studio and verify project creation, preview, queue execution, cancellation, and result loading.
8. Build the wheel and source distribution, then run `python -m twine check dist/*`.
9. Run `scripts/release_smoke.py` against the wheel from outside the checkout.
10. Record artifact SHA-256 hashes, Git commit, lock files, hardware, driver, commands, results, and known limitations in the release report.

The `0.1.0rc1` Step 1 gate establishes installation and basic operational readiness. Full legacy numerical equivalence and performance parity are explicitly separate Step 2 acceptance gates.
