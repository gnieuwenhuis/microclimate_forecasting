# 5. Purpose-built package adopting Kedro's patterns, not Kedro itself

- **Status:** Accepted
- **Date:** 2026-05-30

## Context

The initial sketch proposed Kedro for the data/ML pipeline. Kedro's genuine value is three
patterns — layered data (bronze/silver/gold), a declarative I/O catalog, and config-driven
parameters — which map well onto a configurable, multi-deployment system. But Kedro also
has real friction here:

- **Dynamic per-target config fights Kedro's static catalog.** Neighbor-station sets vary
  per deployment; Kedro is happiest with statically declared datasets.
- **Dependency weight in free CI.** Kedro + plugins is a heavy install on every hourly
  Action run, against a free-runner-minutes budget.
- **Two dissimilar workloads.** Training is a batch job Kedro suits; inference is a tiny
  hourly "fetch → build snapshot → predict → write JSON" job for which Kedro is overkill.

The project explicitly optimizes for AI-navigability and for making bad future decisions
structurally hard.

## Decision

**Skip Kedro the framework; keep Kedro the patterns.** Build a purpose-built `src/`
package with explicit layers: `connectors/`, `features/` (the single snapshot builder),
`pipelines/{training,inference}`, `config/deployments/*.yml`, a thin model registry, and
the forecast-JSON writer. Retain bronze/silver/gold layering and declarative config as
*conventions*, enforced by the project's guardrails rather than by a framework.

## Consequences

- No Kedro dependency; lighter, faster CI; full control over the I/O layer (which we
  hand-roll — roughly a day of plumbing).
- Structure is purpose-shaped, not framework-shaped — more AI-navigable.
- Architectural rules are enforced by our own guardrails (typed boundaries, schema
  validation, import-linter layering, a shared connector contract-test harness) — see the
  scaffolding spec.
- We own the maintenance of the catalog/IO and config-loading code Kedro would have
  provided.

## Alternatives considered

- **Adopt Kedro** — rejected: dynamic-pipeline friction, dependency weight, and split
  workloads outweigh the framework's structure for this project's scale.
