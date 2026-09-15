# Public svVascularize compatibility

CASCADE targets the unmodified public `svv==0.0.48` package. CASCADE does not edit or replace files in `site-packages/svv`; the wheel pins that exact public release.

Several capabilities are not exposed by the public package. CASCADE keeps them
behind a compatibility boundary so application code can use a stable interface
and the implementations can be replaced if equivalent public APIs become
available.

| Capability | Public `svv==0.0.48` | CASCADE behavior |
| --- | --- | --- |
| Equal-bifurcation bulk growth | Not available through the required public API | Compatibility tree/forest methods and runtime settings provide it. |
| Bifurcation optimization with L-BFGS-B | Public growth uses a different optimization path | `cascade.vessels.generation.compatibility.branch_bifurcation` supplies the L-BFGS-B path. |
| Nearest-tree forest growth | The required multi-tree assignment is unavailable | `cascade.vessels.growth` performs candidate-to-segment assignment with collision handling. |
| Incremental scheduled forest growth | The required checkpoint controls are unavailable | `cascade.vessels.growth` schedules and checkpoints one tree at a time. |
| Connectivity repair before solve/export | Saved forests can contain stale parent columns | `cascade.vessels.connectivity` repairs and validates topology. |
| Fast forest simulation cache | No stable public interchange contract | CASCADE reads and writes its own `.forest.simcache` format through the adapter. |
| Historical `.dmn` load | Format behavior differs between versions | CASCADE retains local read support; use PyVista-readable meshes for interchange. |
| Float32 heart/Cext memory path | Growth expects float64 tree arrays | CASCADE keeps float64 at the growth boundary and may use float32 accelerator arrays and caches. |

## Optimizer note

The compatibility bifurcation implementation calls SciPy's L-BFGS-B method
with bounds. L-BFGS-B does not enforce the general inequality
`a[0] + a[1] <= 1`, so this path is not mathematically equivalent to a
constrained SLSQP solve. See [known issues](known-issues.md) before selecting
this growth mode.

## Contributor maintenance rule

Compatibility behavior must remain behind `cascade.vessels.generation.svv_adapter` or `cascade.vessels.generation.compatibility`. Application code consumes CASCADE interfaces and must not monkey-patch public `svv` outside that adapter boundary. The adapter currently replaces the public module's in-memory `Tree` factory and two growth callbacks because public `svv` resolves those names as module globals inside its constructors/methods. This affects only the current CASCADE process; it does not modify the installed distribution on disk. The adapter wrappers remain float64 at the true CCO boundary and may convert completed structures to float32 for equal-bifurcation, simulation-cache, and export work.

The CLI does not import arbitrary external solver modules. CLI orchestration,
input validation, cache detection, connectivity repair, simulation controls,
and export behavior remain CASCADE responsibilities.

Before removing a compatibility implementation:

1. Pin the public `svv` release containing the replacement.
2. Run saved-tree and deterministic-growth compatibility tests.
3. Run the solver and performance test suites.
4. Record the behavior change in the release notes.
