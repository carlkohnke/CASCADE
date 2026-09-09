# Public svVascularize compatibility

CASCADE targets the unmodified public `svv==0.0.48` package. CASCADE must not edit or replace files in `site-packages/svv`.

Several capabilities used by the research workflows are not yet exposed by the public package. They are isolated in CASCADE so they can be removed when equivalent upstream APIs become available.

| Capability | Public `svv==0.0.48` | CASCADE implementation | Upstream action |
| --- | --- | --- | --- |
| Equal-bifurcation bulk growth | Not available through the required public API | Package-owned compatibility tree/forest methods and runtime settings | Open an upstream feature request with the deterministic geometry tests |
| Bifurcation optimization with L-BFGS-B | Public growth uses a different optimization path | `cascade._svv_branch_bifurcation` supplies the internal L-BFGS-B behavior | Open an issue describing the performance motivation and constraint semantics |
| Nearest-tree forest growth | Required heart-growth assignment is not available | `cascade.growth` performs exact candidate-to-segment assignment with collision handling | Open an upstream feature request with scheduled/nearest fixtures |
| Incremental scheduled forest growth | Not available with the required checkpoint controls | `cascade.growth` schedules one-tree-at-a-time growth and checkpoints | Keep in CASCADE unless an inner-loop performance primitive is required |
| Connectivity repair before solve/export | Public serialized forests can contain stale parent columns | `cascade.connectivity` repairs and validates topology | Keep in CASCADE |
| Fast forest simulation cache | No stable public interchange contract | CASCADE reads/writes `.forest.simcache` through its adapter | Keep in CASCADE; request upstream only if cross-package interchange becomes necessary |
| Legacy `.dmn` interchange | Format compatibility differs across public/internal versions | CASCADE has a local domain adapter; cross-version round trips require validation | Open a format/versioning issue |

## Optimizer note

The internal bifurcation implementation calls SciPy's L-BFGS-B method with bounds. SciPy does not enforce general inequality constraints for L-BFGS-B, so the existing `a[0] + a[1] <= 1` constraint is not handled by the optimizer itself. CASCADE retains this behavior for legacy parity in `0.1.0rc1`, records the warning in validation, and must not claim mathematical equivalence to constrained SLSQP until a constrained reparameterization or penalty formulation is validated.

## Maintenance rule

Compatibility behavior must remain behind `cascade.svv_adapter`, `cascade.growth`, or explicitly named `_svv_*` modules. Application code should consume CASCADE interfaces and must not monkey-patch public `svv` outside that adapter boundary.

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
