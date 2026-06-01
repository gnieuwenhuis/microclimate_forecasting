# Datamart HRDPS connector reconciliation + nwp_core solar fix — design

- **Date:** 2026-05-31
- **Status:** Approved (brainstorming) — pending implementation plan
- **Relates to:** ADR-0007 (one HRDPS spec; CaSPAr seed ↔ live Datamart parity), the
  `HrdpsDatamartSource` connector, the shared `nwp_core` normalization core, and the CaSPAr
  subsystem-A spec (`2026-05-31-caspar-seed-acquisition-design.md`), which shares `nwp_core`.

## Purpose

`HrdpsDatamartSource` (the **live** HRDPS NWP source for inference) was built against
unverified guesses. A spike — downloading real HRDPS GRIB2 from MSC Datamart and decoding it
with eccodes — proved the connector is wrong on three axes and uncovered a contract bug in the
shared `nwp_core`:

1. **URL/layout wrong.** The connector's `…/model_hrdps/continental/2.5km/{run}/{hhh}/…`
   path returns **404**. The current Datamart layout is date-partitioned.
2. **Variable map wrong.** The connector expects ECMWF shortNames (`t2m`, `tp`, `tcc`, …);
   real files are `MSC_HRDPS_{VAR}_{LEVEL}` and cfgrib decodes some as `unknown`.
3. **`nwp_core` solar bug (shared).** `nwp_core` treats `solar_radiation_wm2` as instantaneous
   W/m² (pass-through). Real HRDPS `DSWRF` decodes as `ssrd` = **accumulated J/m²** — wrong by
   ~3 orders of magnitude. This is wrong for **both** HRDPS connectors.

This subsystem reconciles `HrdpsDatamartSource` to the verified live layout and fixes the
`nwp_core` solar handling, with an ADR for the shared-contract change.

## Spike findings (verified — eccodes-decoded real GRIB2, run 2026-05-31 18Z)

Real Datamart layout:
```
https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{hhh}/
  {YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LEVEL}_RLatLon0.0225_PT{hhh}H.grib2
```
Per-canonical-variable file + decoded result (each file holds exactly one data-var):

| canonical | MSC VAR / LEVEL | cfgrib name | units (raw) | nwp_core handling |
|---|---|---|---|---|
| temp_c | `TMP` / `AGL-2m` | `t2m` | K | K→°C ✓ (already) |
| dewpoint_c | `DPT` / `AGL-2m` | `d2m` | K | K→°C ✓ |
| surface_pressure_hpa | `PRES` / `Sfc` | `sp` | Pa | Pa→hPa ✓ |
| precip_mm | `APCP` / `Sfc` | **`unknown`** | accumulated mm | de-accumulate ✓ |
| cloud_cover_fraction | `TCDC` / `Sfc` | **`unknown`** | % (0–100) | %→fraction ✓ |
| solar_radiation_wm2 | `DSWRF` / `Sfc` | `ssrd` | **accumulated J/m²** | **pass-through ✗ — BUG** |
| wind_speed_ms | `WIND` / `AGL-10m` | `si10` | m/s | pass-through ✓ |
| wind_dir_deg | `WDIR` / `AGL-10m` | `wdir10` | degrees | pass-through ✓ |

So `nwp_core`'s K/Pa/%/precip handling already matches reality; only **solar** is wrong, and
**variable selection by shortName** breaks (precip/cloud decode as `unknown`).

## Scope

**In scope**

- `HrdpsDatamartSource` (`hrdps_datamart.py`): real date-partitioned URL + `MSC_HRDPS_{VAR}_{LEVEL}`
  filenames; robust single-data-var extraction (not keyed on shortName); canonical dataset
  variable naming.
- `nwp_core.py`: fix solar from instantaneous pass-through to **de-accumulate + ÷Δt → mean
  W/m²** (mirrors precip; clamps ≥ 0). Shared by both connectors.
- A **new ADR** recording the corrected HRDPS solar convention (shared-contract change).
- Tests: injected-`opener` unit tests aligned to the real names/units, a focused `nwp_core`
  solar-de-accumulation unit test, and a `network`-marked live integration test.

**Out of scope**

- The **CaSPAr connector** itself (subsystem-A spec). It shares `nwp_core` and therefore
  **inherits** the solar fix; subsystem-A's plan gets a one-line amendment to verify CaSPAr's
  solar against the new core behavior at its gate. CaSPAr's own variable/unit reconciliation
  stays in that spec (still gated on its sample).
- Committing GRIB2 fixtures (verification is injected-synthetic + the `network`-marked live
  test). The `scratch/` spike artifacts remain throwaway (gitignored or deleted).

## Components

### 1. `HrdpsDatamartSource` (`src/microclimate/connectors/sources/hrdps_datamart.py`)

- **`_DATAMART_BASE` + `_build_datamart_url`** → the verified date-partitioned URL above. The
  variable token is the MSC `(VAR, LEVEL)` pair, not an ECMWF shortName.
- **`HRDPS_VAR_MAP`** → the verified MSC tokens:
  `temp_c→(TMP, AGL-2m)`, `dewpoint_c→(DPT, AGL-2m)`, `surface_pressure_hpa→(PRES, Sfc)`,
  `precip_mm→(APCP, Sfc)`, `cloud_cover_fraction→(TCDC, Sfc)`,
  `solar_radiation_wm2→(DSWRF, Sfc)`, `wind_speed_ms→(WIND, AGL-10m)`,
  `wind_dir_deg→(WDIR, AGL-10m)`. (Shape changes from `str` shortName to a `(var, level)`
  pair used to build the filename; the exact representation is a plan detail.)
- **`_open_latest_run`** → for each `(lead_hour, canonical_var)`: build URL, download, decode
  the single-variable GRIB2 by taking its **sole `data_var`** (robust to cfgrib naming it
  `unknown`/`t2m`/`ssrd`), rename it to the **canonical** name, and stack into
  `(lead_hour, y, x)` with 2-D `latitude`/`longitude`. The Dataset handed to `nwp_core` then
  uses canonical names, so the `var_map` passed to `dataset_to_forecast_frame` is the identity
  map — eliminating shortName fragility. Keep the lazy `cfgrib` import and the typed
  `SourceUnavailable`/`ForecastUnavailable` errors.

### 2. `nwp_core` solar fix (`src/microclimate/connectors/nwp_core.py`)

- Replace the solar **pass-through** with de-accumulation parallel to precip:
  `solar_wm2 = max(0, (solar_acc(h) − solar_acc(h−1)) / SECONDS_PER_HOUR)`, where
  `SECONDS_PER_HOUR = 3600`. `h−1` is already fetched and validated for precip, so no new
  coverage requirement. Update the module docstring's solar line and add the constant.
- Applies to both connectors (shared core); CaSPAr verifies its own solar at its gate.

### 3. ADR

- New ADR: **"HRDPS solar is accumulated J/m²; nwp_core de-accumulates to mean W/m²."** Records
  the verified encoding (DSWRF→`ssrd`, accumulated J/m²), the corrected convention, that it
  affects both HRDPS connectors, and that CaSPAr confirms at its gate.

## Data flow (after fix)

`fetch_forecast(issue_time, lat, lon, lead_hours)` → `_open_latest_run` downloads per
`(lead, var)` from the date-partitioned URL → decodes sole-data-var → canonical-named Dataset
`(lead_hour, y, x)` + 2-D lat/lon → `dataset_to_forecast_frame(ds, identity_map, …)` →
nearest-cell sample + unit conversions (incl. **fixed solar de-accumulation**) → `FORECAST_FRAME`.

## Error handling

- `_open_latest_run` keeps the lazy `cfgrib` import → `SourceUnavailable` if eccodes absent;
  `http_get_bytes` network errors → `SourceUnavailable`; decode/missing-data → `ForecastUnavailable`.
- A wrong URL now manifests as the live integration test failing loudly (404 → `SourceUnavailable`).

## Testing

- **Offline unit (injected `opener`):** a synthetic `xr.Dataset` aligned to the real reality —
  K (`t2m`/`d2m`), Pa (`sp`), % (cloud), **accumulated** precip and solar, m/s wind, deg dir —
  with at least one variable left **`unknown`-named** to exercise sole-data-var selection.
  Assert the `FORECAST_FRAME` outputs: °C, hPa, fraction∈[0,1], de-accumulated precip mm, and
  **de-accumulated mean-W/m² solar**, finite, correct row count.
- **`nwp_core` solar unit test:** known accumulated solar `acc(h-1), acc(h)` → expected
  `(acc(h)-acc(h-1))/3600` W/m²; clamp-≥0 on a decreasing pair.
- **`network`-marked integration test:** live Datamart, dynamically find the latest available
  run, fetch the Lethbridge point for a few leads → valid `FORECAST_FRAME`, finite values,
  sensible physical ranges (e.g. temp −60..60 °C, cloud 0..1, solar 0..1500 W/m²). Deselected
  by default (repo's existing `network` marker).
- Full gate: `ruff format --check`, `ruff check`, `lint-imports`, `pyright` (strict), `pytest`.

## Documentation updates (same PR, per CLAUDE.md)

- The new solar ADR.
- `nwp_core` module docstring: solar line corrected (accumulated J/m² → de-accumulated mean W/m²).
- `hrdps_datamart` module docstring: replace the "unverified URL/shortName" caveats with
  "verified against live Datamart (run 2026-05-31 18Z)"; document the date-partitioned layout
  and the sole-data-var decode.
- README "Project status": note the live Datamart HRDPS connector is verified.
- One-line cross-reference amendment to the CaSPAr subsystem-A plan: verify CaSPAr solar
  against the new `nwp_core` accumulated-solar handling at its gate.

## Decomposition (for the plan)

1. `nwp_core` solar de-accumulation fix + unit test (pure; no network).
2. `HrdpsDatamartSource` URL + var-map + sole-data-var decode + injected-`opener` unit tests.
3. `network`-marked live integration test.
4. ADR + docstrings + README + CaSPAr-plan cross-reference.

## Open items deferred to the plan

- Exact representation of the `(VAR, LEVEL)` mapping (a dataclass/tuple vs two dicts) and how
  `_build_datamart_url` consumes it.
- Whether the connector retains a thin `var_map` indirection or fully canonical-names the
  Dataset (lean: canonical-name in the connector, identity `var_map` into `nwp_core`).
- The live test's run-selection (probe back from now over 00/06/12/18Z) and how it skips
  cleanly when Datamart/network is unavailable.
