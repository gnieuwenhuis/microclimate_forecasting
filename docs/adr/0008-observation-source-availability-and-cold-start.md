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
   - **`cold_start`** *(designed but not implemented in v1 — see finding below)* — a target
     source may be live-only; there is no historical seed and the logger is the sole label
     source, so the deployment is not trainable until enough logged rows accumulate.
3. **Per-source historical coverage is explicit and machine-checked.** Each
   `ObservationSource` declares a `historical_coverage` capability (`deep` / `shallow` /
   `none`). **Eligibility requires every observation source to be `deep`** — the dual-feed /
   deep-history rule, now mechanically enforced (a `none`/`shallow` source like CWOP is
   ineligible).
4. **v1 ships one deployment (ADR-0006):**
   - **`lethbridge`** — `seeded`; target = the closest free *official* deep-history station
     to Henderson Lake: **ACIS Lethbridge Demo Farm IMCIN (#9835)**, ~6 km E; neighbors =
     free ACIS county stations (Picture Butte #710547, Iron Springs #9883, Blood Tribe
     #9747) + YQL airport (ECCC Climate ID 3033875). Trainable immediately.

### Finding: Henderson Lake is currently unreachable on free data

A station-by-station check (2026-05-30) of the live CWOP/APRS feed found **no active,
free PWS within ~6 km of Henderson Lake**. Every station in that radius is a government
ECCC/ACIS station *not* pushing to the live CWOP feed, or inactive. The nearest actual
personal stations (WU `ILETHB87`, `IALBERTA72`, `ILETHB19`) are WU-only (no CWOP
dual-push), currently offline, and behind WU's device-gated API. There is therefore no
free live *or* historical feed at Henderson today.

Consequently the planned `lethbridge_henderson` cold-start deployment **was dropped from
v1** — it had no station to point at. The `cold_start` strategy and a CWOP connector remain
documented here as the **deferred path**, to be reintroduced (with a new ADR) only when a
trigger occurs: a CWOP PWS appears near Henderson, or the owner installs hardware there
(which unlocks that network's API and makes the owner the ground truth — still a cold start,
but a real, controlled Henderson station).

## Consequences

- **ADR-0002 is amended:** observation inputs come from ECCC/ACIS, not consumer-PWS APIs.
  The live-neighbor-obs intent of ADR-0002 survives — ACIS county stations serve as the
  neighbors.
- The `ObservationSource` ABC keeps both `fetch_historical` and `fetch_live` (dual-feed
  contract unchanged) and declares `historical_coverage`; the eligibility validator
  (`connectors/registry.validate_config_sources`) rejects any non-`deep` source, and the
  connector contract-test harness asserts a source's declared coverage matches what its
  historical fetch returns over a probe window.
- v1 implements only the `seeded` path; the `cold_start` branch (logger-only labels,
  insufficient-data handling) is **not built** until Henderson (or another live-only target)
  becomes reachable.

## Alternatives considered

- **Park `lethbridge_henderson` as a disabled config** — rejected in favour of dropping it:
  it had no real station, so a disabled placeholder would carry unused cold-start machinery
  for no v1 benefit. The intent is preserved in this ADR instead.
- **Own hardware at Henderson Lake** — deferred: not free, still a cold start; adoptable
  later as a new deployment + ADR.
- **Pay for Synoptic/TempestONE** — rejected: violates the free-to-deploy constraint.
