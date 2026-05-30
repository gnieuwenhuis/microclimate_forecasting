# 8. Observation-source availability constraint; CWOP cold-start and the hybrid deployment

- **Status:** Accepted
- **Date:** 2026-05-30
- **Amends:** ADR-0002 (observation feature inputs)

## Context

ADR-0002 assumed neighbor/target observations would come from dense consumer PWS networks
(Weather Underground, Tempest, Ambient). Investigation showed those APIs are **device-gated
and unavailable for free**:

- **Weather Underground** issues a free API key only to owners of a PWS that uploads to WU.
- **Tempest/WeatherFlow** killed its generic public-station key on 2025-03-27; without
  owning a station you cannot obtain a token, and public-station data now needs a paid
  TempestONE subscription.
- **Ambient** is the same device-gated model.
- **Synoptic Data's** free "Open Access" tier is restricted to US `.edu` students and caps
  at 1 year of history — the project owner does not qualify.

The free, no-device sources that remain are **Environment Canada** (SWOB live + historical
CSV) and **ACIS** (Alberta Climate Information Service current + historical) — both
deep-history and keyless — plus **CWOP** (Citizen Weather Observer Program), which provides
**free live** observations for any PWS that reports to it, but **no reliable free
multi-year history**.

The project's headline microclimate is **Henderson Lake** (central Lethbridge). The only
stations physically there are consumer PWS — exactly the networks now ruled out for free,
deep history. There is therefore **no free, deep-history, dual-feed station at Henderson
Lake**. Pursuing a Henderson target on free data collapses to a **cold start**: live data
is free, but training labels can only accumulate forward (via the ADR-0007 logger), with no
historical seed.

## Decision

1. **Free observation sources only:** Environment Canada and ACIS (deep-history, dual-feed)
   and CWOP (live-only). Direct WU/Tempest/Ambient network APIs are **not used**.
2. **Introduce a deployment `training_strategy`:**
   - **`seeded`** — every observation source must have deep historical coverage
     (dual-feed). Training uses the CaSPAr historical seed + the logger. Trainable from day
     one.
   - **`cold_start`** — the target source may be **live-only** (e.g. CWOP). There is no
     historical seed; the logger is the sole label source. **Not trainable until a
     configured minimum of logged rows accumulates**; until then the deployment publishes no
     model (the pipeline reports "insufficient data", it does not fail).
3. **Per-source historical coverage is explicit and machine-checked.** Each
   `ObservationSource` declares a `historical_coverage` capability (`none` / `shallow` /
   `deep`). Source-eligibility validation is **strategy-aware**: `seeded` requires all
   sources `deep`; `cold_start` permits a `none`/`shallow` target source.
4. **v1 ships two deployments (ADR-0006):**
   - **`lethbridge`** — `seeded`; target = the closest free *official* deep-history station
     to Henderson Lake (an ACIS city-edge station, e.g. Demo Farm AGDM); neighbors = free
     ACIS county stations + YQL (ECCC). Trainable immediately; proves the pipeline.
   - **`lethbridge_henderson`** — `cold_start`; target = a Henderson-area CWOP PWS read
     live; labels accumulate forward. Keeps the true microclimate goal alive on the only
     viable free basis.

## Consequences

- **ADR-0002 is amended:** observation inputs come from ECCC/ACIS (and CWOP-live for
  cold-start deployments), not consumer-PWS APIs. The live-neighbor-obs intent of ADR-0002
  survives — ACIS county stations serve as the neighbors.
- The `ObservationSource` ABC keeps both `fetch_historical` and `fetch_live` (the dual-feed
  structural contract is unchanged); a live-only source like CWOP implements
  `fetch_historical` as best-effort and declares `historical_coverage = "none"` (or
  `"shallow"`). Depth is now a declared, validated capability rather than an assumption.
- The source-eligibility validator (`connectors/registry.validate_config_sources`) becomes
  strategy-aware; the connector contract-test harness gains a case asserting each source's
  declared coverage matches what its historical fetch returns over a probe window.
- The training pipeline and publish gate must handle a `cold_start` deployment with
  insufficient data gracefully (skip, report), never publishing an untrained model.
- If the owner later installs hardware at Henderson Lake (ADR-0008 supersedes nothing here —
  it stays a config swap), the `lethbridge_henderson` target connector changes from CWOP to
  that device's source; accumulated logged data carries over.

## Alternatives considered

- **All-free official sources, single deployment, re-target near Henderson** — rejected as
  the *sole* path: it abandons the Henderson goal. (It is exactly the `lethbridge` half of
  the hybrid.)
- **Own hardware at Henderson Lake** — deferred: not free, and still a cold start; can be
  adopted later as a connector swap.
- **Pay for Synoptic/TempestONE** — rejected: violates the free-to-deploy constraint.
