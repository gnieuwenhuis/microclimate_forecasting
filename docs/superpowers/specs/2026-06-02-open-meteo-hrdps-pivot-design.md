# Open-Meteo HRDPS Source Pivot — Design Spec

- **Date:** 2026-06-02
- **Authoritative decision:** ADR-0019 (supersedes ADR-0007, ADR-0014)
- **Supersedes spec:** `2026-05-31-inference-logger-design.md` (the logger is removed)
- **Status:** Awaiting review → implementation plan

## Goal

Replace the dead CaSPAr seed and the native MSC GRIB2 HRDPS feeds with **Open-Meteo** as the
single source of HRDPS, for **both** training (Historical Forecast API, deep archive) and live
inference (`/v1/forecast`), and **remove the inference logger** in favour of an idempotent
retrain-time backfill. The HRDPS *model* and the downscaling thesis (ADR-0001, ADR-0010) are
unchanged — only the *source* and the training-data accumulation mechanism change.

**Accepted v1 limitation (ADR-0019 §1b):** the deep archive is a **stitched short-lead** series,
so the model trains on short-lead HRDPS but is served full-lead HRDPS — a lead-time skew we accept
for v1 (local bias is largely lead-stable; the publish gate fails safe).

Success = the `lethbridge` deployment trains and serves end-to-end off Open-Meteo HRDPS, with
spatial/variable parity mechanically enforced, the native GRIB2 stack (and `cfgrib`/`xarray`/ecCodes)
gone, and the public training store carrying lawful CC-BY-4.0 attribution.

## Architecture changes (before → after)

```
BEFORE                                          AFTER
  historical: hrdps_caspar ─┐                     historical: openmeteo ─┐ (Historical Forecast API)
  live:       hrdps_datamart┼─ nwp_core (GRIB2)   live:       openmeteo ─┼─ direct JSON→FORECAST_FRAME
                            └─ FORECAST_FRAME                            └─ FORECAST_FRAME (no nwp_core)
  inference  → build_snapshot → publish + LOG     inference  → build_snapshot → publish (STATELESS)
  training   ← logger-accumulated store           training   ← retrain-time BACKFILL → store (additive)
```

`build_snapshot`, `FORECAST_FRAME`, and the snapshot contract are **unchanged** — the `h−1`
accumulated-baseline assumption was verified to live only in `nwp_core`, which is being deleted.

## Components (units of work)

### 1. New `openmeteo` NWP connector — `connectors/sources/openmeteo.py`

- Implements the `NWPSource` contract: `fetch_forecast(issue_time, lat, lon, lead_hours) →
  FORECAST_FRAME`. Registered via `@register_source("openmeteo")`.
- **Endpoint routing by `issue_time`:** recent → `api.open-meteo.com/v1/forecast`; past →
  `historical-forecast-api.open-meteo.com/v1/forecast` (with `start_date`/`end_date` covering
  `t0 … t0+48h`, then slice the hourly series to leads `1…48` relative to `t0`). Both return the
  same `{hourly: {time, <vars>}}` shape, so one parser serves both. One HTTP+JSON client (reuse
  `connectors/http.py` + `json.loads`); no `cfgrib`, no `xarray`.
- **Request spec (pinned, identical on both routes):** `latitude`, `longitude`,
  `model=gem_hrdps_continental`, the 8-variable hourly set, explicit units, and
  **`cell_selection=land`**. Centralised in one builder so both routes are provably identical
  (see unit 7).
- **Variable mapping → `PHYSICAL_VARS`:** direct for temp/dewpoint/pressure/precip/solar/
  wind_dir; **cloud %→fraction (÷100)**; **wind km/h→m/s** (request `wind_speed_unit=ms`).
  Precip (`precipitation`, mm/h) and solar (`shortwave_radiation`, mean W/m²) arrive
  **already de-accumulated** — pass through, no lead-0 needed.
- Emits leads `1…horizon_hours`, validates against `FORECAST_FRAME`.

### 2. Delete the native GRIB2 stack

Remove: `connectors/nwp_core.py`, `connectors/sources/hrdps_datamart.py`,
`connectors/sources/hrdps_caspar.py`, and their tests (`test_nwp_core.py`,
`test_hrdps_datamart.py`, `test_hrdps_caspar.py`). Update `connectors/sources/__init__.py`
(drop the two HRDPS imports, add `openmeteo`) and the `xarray`/`cfgrib` deps in `pyproject.toml`.
Touch-ups: `tests/connectors/{conftest.py,test_connector_contract.py,test_sources_registered.py}`,
`tests/config/test_schema.py`. (Deliberate departure from the ACIS "retain-but-unused" precedent —
ADR-0019; git history preserves the native path.)

### 3. Config — `config/schema.py` + `config/deployments/lethbridge.yml`

- `enabled_sources: [openmeteo, envcanada]`.
- `nwp.live_connector = nwp.historical_connector = openmeteo` (the two-slot design collapses).
- Set `nwp.sampling: land` (the value the connector passes through as Open-Meteo
  `cell_selection`); keep the field name to avoid a schema migration.
- `training.seed`: `source: openmeteo`, `start: "2024-01-01"`. `holdout_months: 12` unchanged.
- `validate_config_sources` already enforces the NWP-is-`NWPSource` rule — verify `openmeteo`
  passes; no eligibility change for `seeded` (it concerns *observation* depth only).

### 4. Inference pipeline → stateless — `pipelines/inference.py`

- Remove the `TrainingStore` dependency and the `has_snapshot`/`append_snapshot` calls
  (lines ~91, 97, 105–126); inference builds the snapshot, writes the forecast JSON, and stops.
- Keep the `_HRDPS_PUBLISH_LAG` run-selection logic (still picks the latest available run).
- `inference.yml`: drop the eccodes/cfgrib install step, the store env/checkout, and the
  force-push of the `training-data` branch; keep the hourly publish.

### 5. Training pipeline + seed backfill — `pipelines/training.py` (stub today) + `pipelines/training_data.py`

- **Backfill step:** iterate issue-times over `[seed.start, now]` at the HRDPS run cadence,
  calling the existing `training_data` assembly with `nwp=openmeteo` (historical route) and
  `observations=envcanada` (as-of `fetch_historical`), producing labeled rows.
- **Coalesce into `TrainingStore`:** reuse the ADR-0018 read-modify-write coalescing. The
  backfill is **idempotent and additive** — dedupe by `issue_time`×`lead_hour`, latest
  `written_at` wins, and **never delete rows absent from the current backfill** (retention
  independence — ADR-0019). A "prune" operation is explicitly forbidden.
- **Throttle + resume:** cap request rate `<600/min`; the additive coalesce makes interruption
  safe (re-run resumes). Worst-case first backfill ≈ 3,500 single-point calls (≤4 runs/day ×
  ~2.4 yr) — within the free 10k/day; log the count and any skipped runs (no silent caps).
- **Train + gate:** read store → `build_features`/`attach_labels` → train temp+PoP → publish gate
  (ADR-0016). (Model training itself is the existing notebook/assembly path — this spec wires the
  backfill in front of it; full training-pipeline implementation may be a follow-on slice.)

### 6. Attribution — store-level + existing surfaces, CI-checked

- Add `ATTRIBUTION.md` (or `LICENSE`) at the `training-data` branch root: *"Contains data from
  Open-Meteo.com under CC-BY-4.0 (modified: normalized, de-accumulated, resampled to the target
  cell). Underlying model: HRDPS © Environment and Climate Change Canada (ECCC open-data
  licence)."*
- Update `DATA_LICENSES.md` (add Open-Meteo CC-BY-4.0; drop CaSPAr/Mai as required source), the
  forecast JSON `attribution` field, and the dashboard footer → **Open-Meteo + ECCC**.
- **CI check** asserts the store branch carries the attribution file (guardrails-over-discipline).

### 7. Request-spec parity fitness function — `tests/`

A test that asserts the live route and the historical route emit **identical** request parameters
for the shared keys (coords, model, variable set, units, `cell_selection`). This pins
spatial/variable parity (the route URL and `start_date`/`end_date` legitimately differ; lead-time
provenance is the accepted §1b skew, *not* asserted). Replaces the old "one HRDPS spec" convention
(ADR-0019).

### 8. Network smoke test + fixture capture (do first)

Free non-commercial deep access was already confirmed by direct probe (2026-06-02: both
`api.open-meteo.com` and `historical-forecast-api.open-meteo.com` return GEM HRDPS for 2024 dates,
no key). This task **records real JSON responses** from both endpoints as test fixtures
(`tests/connectors/fixtures/openmeteo_forecast.json`, `openmeteo_historical.json`) so the connector
parser (unit 1) is tested against true response shapes, and adds one `@pytest.mark.network` test per
endpoint. Note the empirical finding for the record: HRDPS `_previous_day2+` is null, so the deep
archive is short-lead only (the §1b basis).

### 9. README "Project status"

Reflect: native GRIB2 connectors removed, Open-Meteo connector added, logger removed, backfill-fed
store.

## Data flow

- **Inference (hourly):** `latest run → openmeteo /v1/forecast (target cell) → FORECAST_FRAME →
  build_snapshot → champion → forecast JSON`. No store writes.
- **Training (retrain cadence):** `issue-times → openmeteo Historical Forecast API + envcanada
  historical → training_data assembly → labeled rows → coalesce into store → build_features →
  train → gate`.

## Error handling / edge cases

- **Unpublished/missing run:** `fetch_forecast` raises (NWP is complete-or-fail per
  `_flatten_forecast`); inference relies on the hourly retry, backfill logs+skips and continues.
- **Open-Meteo outage during backfill:** safe — the additive store retains prior history; the run
  resumes later. (Total/long outage caps *new* history until it returns — accepted risk.)
- **Late/QC-revised obs:** backfilled snapshots may have fewer missingness-mask gaps than live;
  masks degrade gracefully (accepted minor skew, ADR-0019).
- **Rate limit:** throttle keeps us under 600/min; backfill is resumable if ever throttled.

## Testing strategy

- Unit: `openmeteo` connector variable mapping + unit conversions against a recorded JSON fixture;
  endpoint routing by `issue_time`; `FORECAST_FRAME` validity.
- Fitness: request-spec parity (unit 7); architecture test still passes (single `build_snapshot`
  path, ADR-0011); import-linter contract holds after deletions.
- `@pytest.mark.network`: one live `/v1/forecast` and one Historical Forecast API call (deselected
  by default).
- Backfill: idempotency (re-run = no dupes, no deletions) and additivity (rows absent from a
  second, narrower backfill survive) on a small date range.

## Out of scope / deferred

- Multi-deployment scale-up and the Open-Meteo **S3 bulk** path (contingency only — breaks parity).
- `cold_start` redesign (lost its logger mechanism — ADR-0019; revisit later).
- Full training-pipeline/model-training implementation beyond wiring the backfill (existing
  notebook path remains; may be sliced separately).
- Migration of already-logged native-Datamart snapshots: **discard** (the store just started; no
  migration tooling — start fresh from the Open-Meteo backfill).

## Open risks

- **Lead-time skew (ADR-0019 §1b):** short-lead seed vs full-lead serving. Accepted for v1; the
  fast-follow if long-lead skill suffers is forward full-lead capture (Option 3, a lightweight
  logger). Watch per-lead skill scores in the first publish-gate run.
- Vendor dependence on Open-Meteo (mitigated by the additive store; S3 contingency documented).
- ~1.3 yr effective training depth before the 12-month holdout — thin but the publish gate fails
  safe; revisit the holdout dial once the store has grown a year.
