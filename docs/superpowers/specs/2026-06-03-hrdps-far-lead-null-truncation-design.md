# Tolerate live HRDPS far-lead nulls (truncate to available coverage) — Design Spec

- **Date:** 2026-06-03
- **Relates to:** ADR-0019 (Open-Meteo HRDPS source; the §1b lead-time reality), ADR-0016 (live-always / never-dark), ADR-0011 (the single snapshot path), the inference-serves-champion slice (PR #29).
- **Status:** Awaiting review → implementation plan

## Goal

Stop the hourly inference run from going dark when the live Open-Meteo `/v1/forecast` series has
`null` values at far leads (the tail beyond the freshest HRDPS run's ~48 h reach). Instead of the
connector hard-failing the whole forecast, **truncate to the available contiguous lead prefix**,
publish a shorter forecast marked `stale`, and only treat the run as unavailable (retry next hour)
when coverage falls below a minimum floor.

Success = a run whose live series goes null at lead *k+1* publishes a `1..k` forecast (`status="stale"`
when `k < horizon_hours`); a run with `< min_horizon_hours` of coverage raises `ForecastUnavailable`
and retries; full-coverage runs are unchanged (`status="ok"`).

## Background

The openmeteo connector's `_parse_hourly_to_forecast_frame` raises `ForecastUnavailable` on the
first missing time slot or null variable (PR #26 / ADR-0019). HRDPS is a 48 h model; issuing at
`now − 4 h` and requesting 48 leads can push the far leads past the freshest run Open-Meteo has
stitched, so those hours come back `null` → the whole forecast fails. Observed live:
`ForecastUnavailable: Open-Meteo 'temp_c' is null at valid_time 2026-06-05T07:00 (lead_hour=43)`.
This is a pre-existing intermittent failure (the hourly cron has been silently failing on unlucky
timing); it blocks the train→publish→serve loop from going live.

## Scope

In: connector truncation, `build_snapshot` actual-leads + minimum-floor, the `min_horizon_hours`
config field, inference `stale` status. Out: changing the issue-time selection; the §1b
forward-capture; any model/gate change.

## Components

### 1. Connector — `openmeteo._parse_hourly_to_forecast_frame`

Replace "raise on first missing/null" with "**return the contiguous non-null prefix**":
- Iterate leads in order. For each lead, if its time slot is missing OR any of the 8 vars is
  `null`, **stop** — return the frame for the leads accumulated so far (`1..k`).
- If the very first requested lead is unavailable (`k == 0`), raise `ForecastUnavailable` (genuinely
  nothing to serve) — the existing message style is fine.
- Otherwise build/validate the `FORECAST_FRAME` for leads `1..k` exactly as today (same unit
  conversions/clamps). `FORECAST_FRAME` validates each row, so a `k`-row frame is valid.
- This is route-agnostic: the live route truncates the unreachable tail; the historical route (past
  data, complete) returns the full set as before. `fetch_forecast`'s docstring/contract updates to
  "returns the available contiguous lead prefix (≤ requested), raising only if none is available."

### 2. `build_snapshot` (`features/snapshot_builder.py`)

Two changes:
- **Store actual leads, not requested.** Today it passes `lead_hours = range(1, horizon+1)` to
  `fetch_forecast` and then sets `FeatureSnapshot(lead_hours=lead_hours, ...)` — the *requested*
  tuple — while `_flatten_forecast` derives keys from the returned frame. With truncation these
  diverge (a latent bug). Fix: after `fetch_forecast`, derive `actual_leads = tuple(int(h) for h in
  frame["lead_hour"])` and use `actual_leads` for both `_flatten_forecast` (already frame-driven)
  and the `FeatureSnapshot.lead_hours` field, so the snapshot is internally consistent.
- **Minimum-coverage floor.** If `len(actual_leads) < config.min_horizon_hours`, raise
  `ForecastUnavailable(f"only {len(actual_leads)} HRDPS leads available (< min_horizon_hours=...)")`.
  This propagates uniformly: the seed backfill already catches `ForecastUnavailable` and skips that
  issue-time; inference propagates → the next hourly run retries.
- The floor check applies only when the NWP feature group is on (it already only fetches NWP then);
  if `config.feature_groups.nwp` is false there is no frame and the floor is moot.

### 3. Config — `config/schema.py` + `config/deployments/lethbridge.yml`

Add `min_horizon_hours: int = Field(ge=1, default=12)` to `DeploymentConfig` (top-level, beside
`horizon_hours`/`lag_hours`). `lethbridge.yml` sets `min_horizon_hours: 12`. Validation: a config
with `min_horizon_hours > horizon_hours` is nonsensical — add a Pydantic model validator that
rejects it (`min_horizon_hours <= horizon_hours`).

### 4. Inference status — `pipelines/inference.py::run_inference`

After serving (champion/baseline per task), compute status with this precedence:
- `"degraded"` if any task fell back from an *expected* champion (existing logic), **else**
- `"stale"` if `len(snapshot.lead_hours) < config.horizon_hours` (the forecast is shorter than the
  target horizon), **else** `"ok"`.

So `degraded` wins over `stale`. Pass the resulting status into `_assemble_forecast` (which already
takes a `status` param). The published `series` already has only the available `k` steps because the
frame/matrix are truncated.

## Data flow & invariants

- **Never dark above the floor:** any run with ≥ `min_horizon_hours` of contiguous coverage
  publishes. Below the floor, it retries — degenerate tiny forecasts are never published.
- **Internally consistent snapshot:** `snapshot.lead_hours` always equals the leads actually present
  in `nwp_features`, so `feature_builder` (which builds rows from `snapshot.lead_hours`) and the
  models/forecast assembly stay aligned.
- **Truncation is contiguous from lead 1:** a forecast is always `1..k` (no interior gaps) — a null
  anywhere stops accumulation, so an interior anomaly conservatively truncates the tail rather than
  producing a sparse series.
- **Both routes:** live truncates the unreachable tail; historical (complete past data) is
  unaffected in practice; a rare short historical issue-time is skipped by the floor in backfill.

## Error handling / edge cases

- **Lead 1 null/missing** → `k==0` → `ForecastUnavailable` (truly unavailable; retry/skip).
- **`min_horizon_hours ≤ k < horizon_hours`** → publish `1..k`, `status="stale"`.
- **`k < min_horizon_hours`** → `ForecastUnavailable` from `build_snapshot` → backfill skips /
  inference retries.
- **Champion fell back AND truncated** → `status="degraded"` (degraded wins).
- **Config `min_horizon_hours > horizon_hours`** → rejected at config load (model validator).

## Testing strategy

- **Connector (unit, hermetic):**
  - trailing-null fixture (vars null from lead k+1) → returns rows `1..k` only; `FORECAST_FRAME`-valid.
  - complete fixture → returns all requested leads (unchanged).
  - lead-1 null → raises `ForecastUnavailable`.
  - (interior null at lead j → returns `1..j-1`.)
- **`build_snapshot` (unit, hermetic, fake NWP returning a truncated frame):**
  - `snapshot.lead_hours` equals the returned leads (not the requested 1..48).
  - returns < `min_horizon_hours` → raises `ForecastUnavailable`.
- **`run_inference` (unit):** truncated-but-≥-floor snapshot → `status="stale"`, series length == k;
  truncated AND champion-fallback → `status="degraded"`; full coverage → `status="ok"`.
- **Config:** `min_horizon_hours` defaults to 12 / loads from `lethbridge.yml`; `> horizon_hours`
  rejected.
- Full gate green: `ruff format --check`, `ruff check`, `lint-imports`, `pyright`, `pytest`.
- (Out-of-band) after merge, a `workflow_dispatch` inference run should now publish
  `forecasts/lethbridge.json` with the champion `model_versions` and `status` ∈ {`ok`,`stale`}.

## Open risks

- **Persistent short coverage at certain hours-of-day** could mean some hours always retry below the
  floor; `min_horizon_hours=12` is low enough this should be rare (the observed case gave ~42 leads).
  If it recurs, revisit the issue-time selection (out of scope here).
- **`stale` semantics:** this activates the contract's `stale` status for "horizon truncated"; if a
  future run-freshness meaning is wanted too, the two will need disambiguating then.
- Truncating in the shared connector slightly changes the historical/training contract (a null in
  the deep archive truncates that issue-time); benign and floor-guarded, but noted.
