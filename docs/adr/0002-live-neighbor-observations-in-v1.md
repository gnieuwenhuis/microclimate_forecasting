# 2. Live neighbor observations are model inputs in v1

- **Status:** Accepted — **amended by ADR-0008**
- **Date:** 2026-05-30

> **Amendment (ADR-0008):** the live-neighbor-observation intent below stands, but the
> *sources* changed. Consumer-PWS APIs (Weather Underground, Tempest, Ambient) turned out
> to be device-gated and unavailable for free. Observation inputs are therefore sourced
> from **Environment Canada** and **ACIS** (deep-history, dual-feed), plus **CWOP** (live
> only) for cold-start deployments. Read this ADR together with ADR-0008.

## Context

Within the downscaling framing (ADR-0001), the model's inputs can range from minimal to
rich:

1. **HRDPS-only** — forecast + time/geo features. Target observations used only as labels.
   Inference depends on HRDPS alone; maximally robust.
2. **HRDPS + target's own recent observations** — adds a persistence signal. Inference now
   needs a live feed of the target station.
3. **HRDPS + target + neighbor-station observations** — adds recent readings from upstream
   neighbors as lag features, capturing weather advecting toward the target (e.g. a rain
   cell moving through the county). Most powerful for convective PoP; needs multiple live
   feeds at inference.

Option (3) carries the real microclimate/advection signal but makes every live observation
feed a runtime dependency that can break, rate-limit, or disagree between training and
serving.

## Decision

Ship **option (3) in v1**: live target *and* neighbor observations are model inputs from
the start. This is acceptable because v1 is not shipped to a wide audience, so downtime is
tolerable and errors can be fixed as they surface.

## Consequences

- Every observation source must be **dual-feed** (hourly historical *and* hourly live for
  the same measurement); daily-only sources are ineligible as inputs.
- The feature-snapshot contract and its as-of/no-leakage invariant become critical (they
  are the only defense against train/serve skew introduced by live obs).
- A down feed at inference must degrade gracefully (impute + missingness mask), not crash.
- The neighbor list is per-deployment configuration; re-targeting re-selects neighbors and
  retrains.
- Inference cannot run client-side without shipping API keys and heavy networking — this
  forces server-side inference (ADR-0003).

## Alternatives considered

- **Build for (3), ship (1)** — the conservative path; rejected because the audience and
  downtime tolerance make the extra robustness unnecessary for v1, and the advection signal
  is the point.
- **Option (2)** — rejected as a halfway house that still needs live feeds but omits the
  most valuable (neighbor) signal.
