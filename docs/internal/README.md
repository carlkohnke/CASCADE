# CASCADE internal production record

This directory is the internal source of truth for the CASCADE production-release and validation program. It answers five questions continuously:

1. What are we trying to accomplish?
2. What is complete, partial, blocked, or not started?
3. What did we do and why?
4. How can another person reproduce the work?
5. Where is the supporting evidence?

## Read this first

| Document | Purpose | Update cadence |
| --- | --- | --- |
| `release-tracker.md` | Current status, priorities, open work, acceptance gates, and next actions | Every material work session |
| `test-plan.md` | Step 2 numerical, workflow, performance, and usability test design | Whenever scope or acceptance criteria change |
| `release-runbook.md` | Commands and procedures for building and validating a release | Whenever the tooling or environment changes |
| `decision-log.md` | Durable technical and process decisions, including rejected alternatives | When a decision is made or reversed |
| `work-log.md` | Chronological record of work performed and evidence produced | Every material work session |
| `m0-legacy-inventory.md` | Frozen external oracle programs, environments, production inputs, hashes, and owner-approved M2 fixture contract | When an oracle/input is deliberately changed |
| `m0-traceability.md` | Legacy capability/parameter/output mapping to CASCADE and M2 cases | When scope or implementation mapping changes |
| `../../validation/README.md` | Storage and naming convention for test evidence | When evidence structure changes |

Supporting public-facing or release-specific records remain in:

- `../release-0.1.0rc1.md`: immutable result report for the tagged Step 1 release candidate.
- `../release-0.1.0rc2.md`: result report for the locally tagged M0-M3 existing-tree release candidate.
- `../release-0.1.0rc3.md`: result report for the post-M3 CLI/package hardening candidate.
- `../release-checklist.md`: concise release gate list.
- `../svv-compatibility.md`: current public-`svv` boundary and upstream candidates.
- `../architecture.md`: production package architecture.
- `../cascade_refactor_audit.md`: historical pre-release parity audit; useful evidence but not current certification.
- `../../CHANGELOG.md`: user-facing release changes.

## Status vocabulary

- **COMPLETE**: acceptance criterion met and evidence is linked.
- **PARTIAL**: useful evidence exists, but the full acceptance criterion is not met.
- **IN PROGRESS**: work is actively underway.
- **PLANNED**: scoped but not started.
- **BLOCKED**: requires a named decision, input, dependency, or external change.
- **DEFERRED**: intentionally outside the current milestone.

“Passed once” and “production certified” are not synonyms. A result is only a certification when the applicable test matrix, environment, tolerance, repeat count, and evidence-retention requirements have all been satisfied.

## Evidence policy

Every completed validation item must record:

- Git commit and CASCADE version.
- Exact command or machine-readable invocation.
- Settings and input hashes.
- Python and dependency versions.
- CPU/GPU, driver, CUDA, OS, and relevant thread settings.
- Random seed and sample geometry identity.
- Wall-clock and component timings when performance is relevant.
- Pass/fail criteria and actual measurements.
- Paths to compact results and, when retained, raw outputs.

Secrets, credentials, private datasets, and machine-specific caches must never be committed. Large raw outputs belong in ignored `validation/runs/`; compact results belong in tracked `validation/results/`.

## Source-of-truth precedence

If records disagree, use this order:

1. Tagged release report for facts about that immutable release.
2. Machine-readable result under `validation/results/` for an individual test run.
3. Current `release-tracker.md` for present status and next actions.
4. `work-log.md` for chronology.
5. Historical audits for context only.
