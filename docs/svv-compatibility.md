# Public svVascularize compatibility

CASCADE targets the unmodified public `svv==0.0.48` package. CASCADE does not edit or replace files in `site-packages/svv`; the wheel pins that exact public release.

Several capabilities used by the research workflows are not yet exposed by the public package. They are isolated in CASCADE so they can be removed when equivalent upstream APIs become available.

| Capability | Public `svv==0.0.48` | CASCADE implementation | Upstream action |
| --- | --- | --- | --- |
| Equal-bifurcation bulk growth | Not available through the required public API | Package-owned compatibility tree/forest methods and runtime settings | Open an upstream feature request with the deterministic geometry tests |
| Bifurcation optimization with L-BFGS-B | Public growth uses a different optimization path | `cascade.vessels.generation.compatibility.branch_bifurcation` supplies the internal L-BFGS-B behavior | Open an issue describing the performance motivation and constraint semantics |
| Nearest-tree forest growth | Required heart-growth assignment is not available | `cascade.vessels.growth` performs exact candidate-to-segment assignment with collision handling | Open an upstream high-scale primitive request; retain the CASCADE path until the replacement passes saved-fixture validation |
| Incremental scheduled forest growth | Not available with the required checkpoint controls | `cascade.vessels.growth` schedules one-tree-at-a-time growth and checkpoints | Keep in CASCADE unless an inner-loop performance primitive is required |
| Connectivity repair before solve/export | Public serialized forests can contain stale parent columns | `cascade.vessels.connectivity` repairs and validates topology | Keep in CASCADE |
| Fast forest simulation cache | No stable public interchange contract | CASCADE reads/writes `.forest.simcache` through its adapter | Keep in CASCADE; request upstream only if cross-package interchange becomes necessary |
| Legacy `.dmn` load | Format compatibility differs across public/internal versions | CASCADE retains local load support for frozen validation inputs | No public interchange claim; use STL/PyVista-readable meshes as inputs and VTP/VTU as outputs |
| Float32 heart/Cext memory path | Public growth is qualified here only with float64 tree arrays | CASCADE uses float32 accelerator work arrays/caches while retaining float64 at the public growth boundary | Revisit tree-array float32 only after M2 evidence |

## Optimizer note

The internal bifurcation implementation calls SciPy's L-BFGS-B method with bounds. SciPy does not enforce general inequality constraints for L-BFGS-B, so the existing `a[0] + a[1] <= 1` constraint is not handled by the optimizer itself. CASCADE retains this behavior for legacy parity in `0.1.0rc4`, records the warning in validation, and must not claim mathematical equivalence to constrained SLSQP until public SVV exposes the requested optimizer selector and the growth campaign validates it.

## Maintenance rule

Compatibility behavior must remain behind `cascade.vessels.generation.svv_adapter` or `cascade.vessels.generation.compatibility`. Application code consumes CASCADE interfaces and must not monkey-patch public `svv` outside that adapter boundary. The adapter currently replaces the public module's in-memory `Tree` factory and two growth callbacks because public `svv` resolves those names as module globals inside its constructors/methods. This affects only the current CASCADE process; it does not modify the installed distribution on disk. The adapter wrappers remain float64 at the true CCO boundary and may convert completed structures to float32 for equal-bifurcation, simulation-cache, and export work.

The production CLI never imports a legacy TissueSim file or an arbitrary external solver module. Legacy comparisons run the frozen oracle in its own environment and exchange only hashed inputs/results with the independently installed CASCADE wheel.

The default is to implement compatibility and workflow behavior in CASCADE. An
upstream `svv` request is warranted only when the change must occur inside an
`svv` growth/optimization loop, requires access to an unavailable internal
primitive for acceptable performance, or defines a stable interchange contract
that both packages must share. CLI orchestration, input validation, cache
detection, connectivity repair, simulation controls, and export behavior belong
in CASCADE.

Before removing a compatibility implementation:

1. Pin the public `svv` release containing the replacement.
2. Run saved-tree and deterministic-growth parity tests.
3. Run the TissueSim performance matrix.
4. Record the migration in the release notes.
