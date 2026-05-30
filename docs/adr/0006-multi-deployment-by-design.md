# 6. Multi-deployment by design, one deployment in v1

- **Status:** Accepted
- **Date:** 2026-05-30

## Context

The system is designed around Lethbridge but must be deployable for a different
microclimate, and the target station may change over time (which requires repulling
neighbor data and retraining). A "deployment" bundles everything that pins the problem to
one place: target station, neighbor list, enabled sources, HRDPS grid sampling, horizon,
lag depth, feature switches, and output destination.

The choice is between a single active config (redeploying elsewhere = edit/fork + retrain)
and a multi-deployment design (a directory of validated configs, every pipeline
parameterized by a `deployment_id`).

## Decision

**Multi-deployment by design, exactly one config (`lethbridge`) in v1.** Configs live in
`config/deployments/*.yml`, each a validated `DeploymentConfig`. Every pipeline takes a
`deployment_id`; artifacts, model-registry keys, and output filenames are **namespaced by
`deployment_id`**. The Action runs a matrix over the configs directory. "Lethbridge" is
never hardcoded.

## Consequences

- Adding a microclimate is *drop a new YAML + run training*, not a refactor — honoring the
  configurability goal at near-zero cost.
- Champion/challenger promotion is **per `(deployment_id, task)`** (composes with
  ADR-0004's two independent models).
- Every artifact path, registry key, and output filename must carry `deployment_id`; the
  Action gains a matrix dimension.
- Prediction targets are limited to stations with sufficient labeled history; the default
  target should be a genuine in-city microclimate station, **not** the airport (which raw
  HRDPS likely already matches, leaving no skill to gain).

## Alternatives considered

- **Single active config** — rejected: turns "deploy elsewhere" into a fork/refactor and
  invites hardcoding Lethbridge.
- **Build full multi-deployment orchestration now** — rejected as speculative; keying by
  `deployment_id` is enough, with one config shipped.
