# 10. Retarget the lethbridge deployment to a live ECCC hourly station; drop ACIS

- **Status:** Accepted
- **Date:** 2026-05-30
- **Amends:** ADR-0001 (target choice rationale), ADR-0002 (neighbor observation sources), ADR-0008 (observation-source availability)
- **Informed by:** Spike #3 (`docs/spikes/0003-acis-ungated-access.md`)

## Context

ADR-0008 chose **ACIS Lethbridge Demo Farm IMCIN (#9835)** as the `lethbridge` target — the
closest free, deep-history *official* station to Henderson Lake — and ACIS county stations
(Picture Butte #710547, Iron Springs #9883, Blood Tribe #9747) as neighbors. ADR-0001
justified preferring an ACIS ag-station over YQL airport on the grounds that *"HRDPS does not
already nail it"* (more downscaling headroom). Both assumed ACIS could supply the **hourly**
observations the system needs (the forecast product is hourly; the feature snapshot,
`lead_hour`, and as-of reconstruction are all hourly).

**Spike #3 (2026-05-30) refuted that assumption.** Every empirical claim below was confirmed
by two independent agents against real HTTP responses:

- The only *ungated* ACIS endpoint (the IMCIN/AIMM "climate file") serves **daily** data
  only — one row per calendar day, no time field. ACIS's only hourly route is the
  session/captcha-gated Data Viewer API, which the unattended cron job forbids.
- The ACIS ag stations have ECCC/MSC "AGDM" hourly twins on the same ungated bulk-CSV feed
  the `envcanada` connector already uses (Demo Farm → 42726, Blood Tribe → 42703, Iron
  Springs → 42728) — but **those hourly feeds are dead** (last real obs: Demo Farm
  2024-04-02, Blood Tribe 2024-05-10, Iron Springs 2023-05-15).
- Therefore **there is no free, ungated source of *live hourly* observations at Demo Farm or
  any ACIS ag station.** This also closes the cold-start escape hatch: cold-start (ADR-0008)
  still needs a *live* feed for the logger to accumulate labels forward, and there is no live
  hourly feed there to accumulate.

A live, verifiable, self-improving **hourly** forecast at Demo Farm is impossible on free
data. Of the three things the project wanted — **hourly · free · ag-microclimate target** —
one had to give.

## Decision

The product's **hourly** cadence and its identity as a **live, self-improving, verifiable
service** are both non-negotiable (CONTEXT.md). The ag-microclimate *target* is the
constraint that yields.

1. **Retarget `lethbridge` to a station that is live on the ungated ECCC bulk hourly CSV.**
   A regional search (and independent verification) of that feed selected:

   - **Target = `LETHBRIDGE CDA`, MSC StationID `2265`** (Climate ID 3033890), 49.70 / -112.77,
     elev 910 m. A non-airport agricultural research station ~5 km E of Lethbridge and ~4 km
     from Henderson Lake — so the geographic story is **preserved**, not relocated. Live to
     2026-05-29 23:00; deep hourly history (inventory 1994→2026; full-variable rows verified
     back to 2015). Carries temp, dewpoint, RH, **precip, and station pressure**; only
     visibility is absent (irrelevant to a temp/PoP target). Its non-airport, valley-bench
     siting gives HRDPS's 2.5 km cell more local bias to learn than the flat airport — the
     headroom ADR-0001 wanted, now on a station that is actually live.

2. **Fallback target = `LETHBRIDGE` (YQL airport), MSC StationID `49268`** (Climate ID
   3033875), live + deep (2011→2026). If `LETHBRIDGE CDA` ever fails the publish gate (see
   §5) or loses continuity, retargeting to YQL is a one-line config change. v1 ships **one**
   deployment (ADR-0006); YQL is a documented fallback, not a parallel deployment.

3. **Neighbors are now live ECCC hourly stations only.** The close 8–60 km advection ring is
   **dead** on the ungated feed (every ag-network station there returns an empty scaffold;
   verified). The nearest live ECCC hourly neighbors are 60–100 km out:

   | Direction | Station | StationID | Climate ID | Note |
   |---|---|---|---|---|
   | W  | PINCHER CREEK | 49368 | 3035198 | chinook gateway; full coverage |
   | NW | CLARESHOLM    | 2224  | 3031640 | no station pressure |
   | SW | CARDSTON      | 26971 | 3031322 | precip sensor dropped in 2026 (temp/RH/wind) |
   | SE | MILK RIVER    | 8804  | 3044533 | full coverage |

   (BOW ISLAND #10915, ~100 km E, is live and available as an alternative/addition.) A
   60–100 km lever arm is coarse but real: at storm speeds (~30–50 km/h) it is ~1.5–3 h of
   advection lead, which is useful across a 1–48 h horizon. Per-variable missingness masks
   cover the Cardston-precip / Claresholm-pressure gaps.

4. **Drop ACIS from v1 entirely (ECCC-only observations).** ACIS offers only *daily* live
   data, and a daily total carries no sub-day advection signal — filling the dead close ring
   with ACIS daily neighbors would add cadence/skew complexity for negligible hourly value.
   `acis` is removed from the `lethbridge` deployment's `enabled_sources` and neighbor list.
   The `AcisSource` connector code is **retained but unused** (not deleted), and the **ACIS
   connector work in #1 is descoped from v1** (deferred, not abandoned). `envcanada` is now
   the sole observation connector for v1, covering both target and neighbors through one
   mechanism — eliminating the cadence-skew surface ACIS would have introduced.

5. **Headroom is de-risked by the publish gate, not a pre-probe.** Skill cannot be measured
   until the model trains, and staging CaSPAr purely to pre-rank candidates is not worth it.
   The target was chosen by defensible microclimate proxies (valley-bench, non-airport
   siting). The **publish gate is the arbiter**: it refuses to publish a model that cannot
   beat raw HRDPS, so a wrong headroom bet **fails safe** (no bad forecast is shipped). If
   `LETHBRIDGE CDA` cannot beat HRDPS, fall back to YQL (§2).

## Consequences

- **ADR-0001 amended:** the target is a live ECCC hourly station (`LETHBRIDGE CDA`), not an
  ACIS ag-station. The "more headroom than YQL" intent survives — CDA is a non-airport
  station — but is now subordinate to *liveness* and *deep hourly history*, which are hard
  requirements for a seeded, live, hourly service.
- **ADR-0002 amended:** neighbor observations come from live ECCC hourly stations (60–100 km
  advection vectors), not ACIS county stations. The live-neighbor-obs intent survives.
- **ADR-0008 amended:** the free observation source for v1 is **ECCC only**. ACIS is dropped
  (no ungated live-hourly feed). The `seeded` strategy and `deep`-coverage eligibility rule
  are unchanged; `LETHBRIDGE CDA` satisfies them via the ECCC bulk-CSV deep hourly history.
- **`config/deployments/lethbridge.yml` re-authored:** target → CDA (`envcanada`); neighbors
  → the four ECCC hourly stations above; `acis` removed from `enabled_sources` and neighbors;
  the spike-#3 ACIS annotations removed (superseded by this ADR).
- **CONTEXT.md and DATA_LICENSES.md updated:** "Available sources" and the `lethbridge`
  deployment description now read ECCC-only; ACIS attribution is no longer mandatory for v1
  (the ACIS licence section is kept, marked "not used in v1 — see ADR-0010").
- **Issue #1 re-scoped:** its ACIS-connector slice is deferred out of v1; `envcanada` (the
  connector already merged) is the v1 observation connector. Spike #3's "blocked pending
  pivot" status is resolved by this ADR.
- The deployment remains **seeded and trainable from day one**, **live**, **verifiable**
  (real CDA obs land hourly), and **self-improving** (the logger accumulates real CDA labels
  forward) — the guarantees a frozen Demo Farm model could not provide.

## Alternatives considered

- **"Frozen-seeded" Demo Farm** — keep Demo Farm as target, train on its dead AGDM hourly
  history (2004→2024), publish live off HRDPS with no live labels. Rejected: no live
  verification ever, a permanently frozen model (the logger can never add a label), and
  train/serve skew on the target-obs feature (present in 100 % of training, 0 % of serving).
  Betrays the live-service guarantees.
- **Daily product** — re-scope the forecast to daily Tmax/Tmin + daily PoP so ACIS Demo Farm
  works fully. Rejected: hourly downscaling of HRDPS is the project's identity (CONTEXT.md).
- **Keep ACIS daily neighbors** to fill the dead 20–40 km ring. Rejected: daily cadence
  carries no sub-day advection signal; it would add a daily→hourly skew surface and keep a
  second connector alive for negligible value (see §4).
- **Nowcast/analysis grids as labels** (ERA5-Land, RDPA, RTMA-style) to target *any* point,
  including Henderson Lake itself. Parked as a **future idea, not adopted:** most free hourly
  grids near Lethbridge are *coarser* than HRDPS (ERA5-Land ≈9 km, RDRS/RDPA ≈10 km) — training
  2.5 km HRDPS toward a coarser grid is upscaling, not downscaling — and same-family analyses
  are circular with HRDPS. It is only viable with a ≥HRDPS-resolution, observationally
  *independent* product (e.g. NOAA RTMA/URMA, *if* its domain reaches 49.7 °N), and it changes
  the thesis to "predict the best gridded estimate at a point." Revisit behind a product spike.
- **Auxiliary independent forecasts as input features** (multi-model post-processing via e.g.
  Open-Meteo's GFS/ICON/ECMWF, including its historical-forecast archive). Parked as a
  **deferred fast-follow feature**, gated on confirming a *free historical forecast archive*
  exists (else it is train/serve-skew-prone or a cold-start feature). HRDPS stays the
  backbone-being-downscaled and the skill-score baseline.
