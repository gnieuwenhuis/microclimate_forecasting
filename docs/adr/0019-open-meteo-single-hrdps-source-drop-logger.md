# 19. Open-Meteo is the single HRDPS source (both feeds); drop the inference logger for retrain-time backfill

- **Status:** Accepted
- **Date:** 2026-06-02
- **Supersedes:** ADR-0007 (CaSPAr seed + inference logger), ADR-0014 (nwp_core de-accumulation)
- **Amends:** ADR-0009 (raw store privacy), ADR-0010 (Open-Meteo was parked there as a deferred auxiliary feature), ADR-0015/0017/0018 (training store — survive, now fed by backfill)
- **Informed by:** two `deep-research` passes (forecast-model availability; HRDPS-archive paths) plus an **empirical Open-Meteo API probe (2026-06-02)** that corrected the deep-archive assumption (see §1b), recorded in this PR's discussion.

## Context

ADR-0007 sourced **historical** HRDPS from **CaSPAr** (one-time seed) and **live** HRDPS from
**MSC GeoMet/Datamart**, both decoded from native GRIB2 through the shared `nwp_core`
(de-accumulation etc., ADR-0014). The inference pipeline doubled as a **logger**, persisting
every hourly snapshot, because *"CaSPAr is a queued bulk-request research archive, not an API"*
— a cron could not re-pull from it.

Two facts broke that design:

1. **CaSPAr is dead.** `caspar-data.ca` is offline and unmaintained since ~mid-2025 (site
   refuses connections; the project's own GitHub issue #3 "site not accessible", opened Sept
   2025, sits unanswered; last commit Sept 2024; no successor). The deep HRDPS record from 2017→present is
   **unrecoverable** from CaSPAr. Verified by independent agents against live evidence.
2. **No other free source gives a deep, Canada-wide HRDPS-resolution archive.** MSC Datamart and
   the Herbie client both retain only ~30 days. US deep archives (HRRR/NAM/GEFS reforecast) do
   not cover Canada. ECMWF/ICON cover Canada but are ~10× coarser and keep no deep free
   forecast archive.

The one free path that covers HRDPS itself, for both history and live, is **Open-Meteo** — which
mirrors **GEM HRDPS Continental (2.5 km, hourly)**, serves it live via `/v1/forecast`, and keeps a
**deep historical archive from ~2024** via the **Historical Forecast API**, under **CC-BY-4.0** and
free for non-commercial use (both verified by direct API probe, no key required). ADR-0010 had
already parked Open-Meteo as a *deferred auxiliary feature, gated on confirming a free historical
forecast archive exists*. That archive is now confirmed — but we adopt Open-Meteo for a larger
role: as **the HRDPS backbone source itself**.

## Decision

### 1. Open-Meteo is the single source of HRDPS, for **both** training and inference.

A new `NWPSource` connector (pure HTTP+JSON) implements `fetch_forecast` → `FORECAST_FRAME`
directly. For a recent `issue_time` it calls `/v1/forecast` (`api.open-meteo.com`); for a past
`issue_time` it calls the **Historical Forecast API** (`historical-forecast-api.open-meteo.com`,
`start_date`/`end_date`) and slices the hourly series to leads `1…48` relative to `t0`. The two
config slots collapse: `live_connector == historical_connector == openmeteo`.

- **`cell_selection=land` is pinned identically on both feeds** (elevation-aware grid-cell
  selection via Open-Meteo's 90 m DEM). We accept that Open-Meteo's HRDPS is a *reprocessed
  mirror*, not the raw 2.5 km grid cell, and that it applies its own elevation downscaling — fine
  because the physics still originates from ECCC HRDPS. The skill-score baseline ("raw HRDPS at the
  target") is now "Open-Meteo HRDPS at the selected land cell" — an internally consistent floor.
- **One Open-Meteo request spec**, shared by both feeds (coordinates, model, variable set, units,
  `cell_selection`), **enforced by a fitness-function test**. This pins everything *except the
  lead-time provenance* (see §1b), and replaces CONTEXT.md's old "one HRDPS spec" convention.
- All 8 `PHYSICAL_VARS` are available; the connector does only two unit conversions (cloud
  %→fraction, wind km/h→m/s). Precip and shortwave arrive **already de-accumulated** to hourly.

### 1b. Accepted limitation: lead-time skew between the deep seed and live serving (v1).

An empirical API probe (2026-06-02) corrected an earlier assumption: there is **no free
Open-Meteo product giving deep-history, full 1–48-lead-*per-run* HRDPS**. What the deep archive
(Historical Forecast API, ~2024→) actually serves is a **stitched short-lead series** — each hour
is the freshest run's early-lead value. The full per-run horizon (`/v1/forecast`, leads 1–48) and
the run-offset Previous Runs variables (`_previous_dayN`) only reach ~1 day back for HRDPS.

Consequence: the model **trains on short-lead HRDPS** (the deep seed) but is **served full-lead
HRDPS** (`/v1/forecast`), so a train/serve skew grows with lead hour. **For v1 we accept it**
(Option 1), on two grounds: (a) the *local bias* a downscaler corrects — the HRDPS-cell-vs-station
microclimate offset — is largely lead-time-stable, so it should transfer across leads; and (b) the
**publish gate (ADR-0016) fails safe** — a model that doesn't beat raw HRDPS at a given lead is not
published. True-parity *forward capture* (re-introducing a lightweight logger) is deferred as a
fast-follow (Option 3) if the gate shows the skew hurts long-lead skill.

### 2. De-accumulation becomes a per-connector concern; remove the native GRIB2 path.

The `h−1`/accumulated-baseline assumption was verified to live **only** in `nwp_core`
(`_check_lead_hours_present` + the de-accumulation loop), **not** in `build_snapshot`,
`FORECAST_FRAME`, or the snapshot contract — `build_snapshot` requests leads `1…48` and is
agnostic to how a connector produces them. The Open-Meteo connector therefore bypasses `nwp_core`
entirely and emits canonical hourly values with no lead-0 dependency.

Because Open-Meteo is now the only v1 NWP path, the native GRIB2 machinery is **deleted, not
retained** (a deliberate deviation from the "retain-but-unused" precedent of the ACIS connector —
justified because, unlike cold-start/ACIS, the native-GRIB2 path has *no planned future*; git
history preserves it): `nwp_core.py`, `hrdps_datamart.py`, `hrdps_caspar.py`, their tests, and the
`xarray`/`cfgrib` dependencies. **Bonus:** this drops the ecCodes native system library from CI,
strengthening the zero-maintenance/free-infra identity. The native-GRIB2 HRDPS path is recorded
here as a deliberately-removed alternative, recoverable from git if ever needed.

### 3. Drop the inference logger; training data comes from retrain-time backfill.

ADR-0007's rationale for the logger ("CaSPAr is not an API") is void — the Historical Forecast API
*is* an API a cron can re-pull — and the GitHub-Actions hourly cron proved flaky. The logger is
removed:

- **Inference pipeline → stateless, publish-only.** It still runs hourly (the product is hourly)
  but no longer logs snapshots.
- **Seed backfill → a retrain-time step.** At each retrain, pull the deep HRDPS history from the
  Historical Forecast API (+ as-of ECCC obs), assemble labeled rows, and coalesce into the training
  store. The backfill is **idempotent and additive**: it coalesces by `issue_time`×`lead_hour` and
  **never prunes**, so the store survives Open-Meteo pruning or outage and accumulates monotonically.
  **This additivity is a durability guarantee — no future cleanup step may delete rows absent from a
  backfill.**
- This also **decouples training-data completeness from cron reliability**: a missed inference
  run only makes a forecast hour stale; the training set stays gapless because backfill pulls the
  full record from Open-Meteo regardless.

The store stays public (ADR-0017/0018) and `seeded` (ADR-0010) is unchanged — `seeded` eligibility
is about *observation* depth (ECCC, deep), which is untouched; only the seed *source* changes.

### 4. Licensing & attribution.

- The free tier is **non-commercial only**; this project (free, open, no ads/subscription)
  qualifies. If a deployment ever monetizes, it needs Open-Meteo's paid plan or self-hosting.
- The data is **CC-BY-4.0**, which permits public redistribution **with attribution**. Therefore
  the **public `training-data` branch must carry its own attribution notice** (a redistribution of
  CC-BY-4.0 data) — in addition to the forecast JSON `attribution` field and dashboard footer.
  Attribution credits **Open-Meteo (CC-BY-4.0, changes indicated)** and **ECCC** (HRDPS + station
  obs). The **CaSPAr / Mai et al. 2020 citation is dropped.** This finishes superseding ADR-0009's
  "raw store must be private" stance (begun by ADR-0017): the raw data is now CC-BY-4.0/ECCC,
  redistributable with attribution. The notice text lives at
  `scripts/training_store_attribution.txt` (CI checks this file exists in-repo); **copying it onto
  the `training-data` branch root is the responsibility of the (deferred) store-publish slice** —
  that slice is not yet implemented.

## Consequences

- **Resolution & depth degrade vs the CaSPAr plan.** Open-Meteo HRDPS is a reprocessed ~2.5 km
  mirror (acceptable — same on both feeds); seed depth collapses from ~8 yr (2017) to ~2.3 yr
  (Jan 2024 →), leaving ~1.3 yr of training data before the unchanged 12-month holdout. Thin but
  workable; the publish gate (ADR-0016) fails safe if it isn't, and the store grows forward.
- **Minor residual obs skew:** backfilled obs may include late/QC-revised values that were missing
  at live time, so a backfilled snapshot can show fewer missingness-mask gaps than a live one. The
  mask mechanism degrades gracefully; magnitude is small for punctual ECCC hourly obs.
- **Lead-time skew is the headline limitation** (§1b): short-lead seed vs full-lead serving,
  accepted for v1, fail-safe via the publish gate, with forward-capture (Option 3) as the
  documented fast-follow if long-lead skill suffers.
- **Vendor dependence on Open-Meteo** is the new systemic risk (the CaSPAr lesson). Mitigated by
  the additive store (history we've captured survives Open-Meteo going away). A public-S3
  (`open-meteo/open-data`) bulk path exists as a scale-up/contingency, but it reprocesses raw grids
  and would diverge from the API's cell-selection/downscaling, so it is **not** the v1 path.
- **First-backfill API budget is safe.** NWP is single-point (target cell only; neighbors are ECCC
  obs). The Historical Forecast API returns a whole date range per call, so a full backfill is
  ~dozens of calls (monthly chunks over ~2.4 yr) — far within the free 10k/day (and 5k/hour) limit.
  The backfill is **resumable/idempotent** and throttles `<600/min`.
- Reverts the inference-logger work of PR #19–#23; `inference.yml` loses its logging/force-push
  half (the hourly publish remains). The backfill + `pipelines.training_data` become the sole
  training-data path.
- **Free non-commercial deep access was verified** by direct API probe (2026-06-02): both
  `api.open-meteo.com` and `historical-forecast-api.open-meteo.com` return GEM HRDPS data for 2024
  dates with no API key. (Resolves the earlier ambiguous pricing-page line.)
- `cold_start` (ADR-0008, deferred) loses its forward-accumulation mechanism (the logger) and needs
  a new forward-capture design when revisited — the same Option-3 capture would serve it.

## Alternatives considered

- **Open-Meteo for the seed only, keep native Datamart live** — rejected: reintroduces the exact
  train/serve skew "one spec" exists to prevent (reprocessed vs native HRDPS diverge).
- **Keep the inference logger now (Option 2)** — deferred, not rejected outright. It is the only
  way to get true-parity *full-lead* training rows, but it starts empty (no deep seed) and the
  flaky hourly cron is a real cost. For v1 we prefer the deep short-lead seed + fail-safe gate;
  forward capture returns as Option 3 if needed.
- **Hybrid: deep short-lead seed + forward full-lead capture (Option 3)** — deferred fast-follow.
  Best long-term (deep history now, true parity later) but adds a forward-capture step we don't
  need until the gate shows the §1b skew hurts.
- **Ephemeral backfill (no persisted store)** — rejected: re-makes the CaSPAr mistake (total
  dependence on an external archive's uptime).
- **Retain the native GRIB2 connectors unused** — rejected: no planned future for the path; git
  history suffices; deletion drops the ecCodes/cfgrib burden.
- **Pivot away from HRDPS to a globally-covering model (ECMWF/ICON/GFS)** — rejected: ~10× coarser
  than HRDPS and no deep free forecast archive either; HRDPS-via-Open-Meteo keeps ADR-0010 intact.
- **Bulk-pull from the Open-Meteo public S3 archive instead of the API** — parked as a scale-up
  contingency: free and unthrottled, but raw grids requiring us to replicate Open-Meteo's
  cell-selection/downscaling, which breaks the parity guarantee.
