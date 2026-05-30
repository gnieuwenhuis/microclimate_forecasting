# 3. Server-side inference with thin clients; on-device TFLite retired

- **Status:** Accepted — **storage homes amended by ADR-0009**
- **Date:** 2026-05-30

> **Amendment (ADR-0009):** the published-artifact homes below stand, but the raw
> training-data store is **private** (a separate repo), not a public branch — only derived
> products (forecast JSON, models) are public, for data-licensing reasons.

## Context

The initial sketch had GitHub Actions publish a tiny "current feature JSON," with the
Android app running the model on-device via TensorFlow Lite. That premise assumes the
client can assemble the model's inputs. ADR-0002 makes inputs include **live neighbor
observations**, so someone must fetch HRDPS, fetch every neighbor feed, reconstruct lag
features, and apply masks. Three places can do that:

- **(a) Server-side assembly + server-side inference** — a scheduled Action builds the
  snapshot, runs the model, and publishes the *forecast output* JSON. Clients only display
  it.
- **(b) Server-side assembly, on-device inference** — Action publishes the *feature*
  snapshot; client runs TFLite. Still depends on the hourly job, more complex than (a).
- **(c) Fully on-device** — client fetches all feeds and runs the model. Bakes API keys
  into the app; heavy, fragile, per-platform.

## Decision

Adopt **(a)**: feature assembly and inference run **server-side in a scheduled GitHub
Action**, which publishes a forecast-output JSON. Clients are **thin viewers** that only
read that JSON. The on-device TFLite requirement is **retired**.

## Consequences

- The model is no longer constrained to TFLite-convertible architectures, unblocking
  gradient-boosted trees (ADR-0004).
- Android app and dashboard collapse onto a **single published artifact** — built once,
  rendered many times. The JSON's `schema_version` is the cross-client contract.
- All secrets (Tempest/WU/CaSPAr) and all fragile live-feed plumbing live in one
  server-side place, never on devices.
- Forecast freshness is bounded by the last successful Action run; clients must handle a
  `stale`/`degraded` status. Acceptable given ADR-0002's downtime tolerance.

## Alternatives considered

- **(b)** — rejected: worst of both (still job-dependent, but more complex).
- **(c)** — rejected: ships secrets to clients, fragile, duplicated per platform.
