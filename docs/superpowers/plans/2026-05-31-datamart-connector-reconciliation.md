# Datamart HRDPS Connector Reconciliation + nwp_core Solar Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the shared `nwp_core` solar handling (accumulated J/m² → mean W/m²) and reconcile `HrdpsDatamartSource` to the verified live MSC Datamart layout (date-partitioned URL, MSC variable codes, sole-data-var decode).

**Architecture:** Two coupled-but-orderable changes. Task 1 fixes `nwp_core` solar in isolation (pure; verified by overriding the synthetic fixture's solar). Task 2 reconciles the connector + aligns the shared synthetic fixture to reality (canonical var names, accumulated solar). Task 3 adds a `network`-marked live test. Task 4 finalizes ADR/docs. Verification is injected-synthetic (offline) + one live test; no committed GRIB2.

**Tech Stack:** Python 3.12, xarray + cfgrib/eccodes (eccodes now installed via Homebrew), pandas, Pandera (`FORECAST_FRAME`), pytest (`network` marker), pyright strict, ruff, uv.

---

## Conventions for every task

- TDD: failing test → confirm fail → implement → confirm pass → commit.
- Full gate before each commit: `uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`. `pytest` deselects `network`-marked tests by default; the "Cannot find the ecCodes library" warning no longer appears now that eccodes is installed, but a cfgrib import is still lazy — a clean run is the bar.
- Commit on the current branch `spec/datamart-connector-reconciliation` (main is PR-only). Push only at Final Integration.
- Verified facts driving this plan (spike, real run 2026-05-31 18Z): real URL is `https://dd.weather.gc.ca/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{hhh}/{YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LEVEL}_RLatLon0.0225_PT{hhh}H.grib2`; the 8 vars are `TMP`/AGL-2m, `DPT`/AGL-2m, `PRES`/Sfc, `APCP`/Sfc, `TCDC`/Sfc, `DSWRF`/Sfc, `WIND`/AGL-10m, `WDIR`/AGL-10m; each single-variable file decodes to exactly one data-var (cfgrib names some `unknown`); `DSWRF` decodes as `ssrd` = accumulated J/m².

## File structure

**Modify**
- `src/microclimate/connectors/nwp_core.py` — solar pass-through → de-accumulate + ÷3600 (Task 1).
- `src/microclimate/connectors/sources/hrdps_datamart.py` — URL, var map, `_open_latest_run`, `fetch_forecast` var_map, docstring (Task 2, 4).
- `tests/connectors/conftest.py` — `build_hrdps_dataset` / `VAR_MAP`: canonical names + accumulated solar (Task 2).
- `tests/connectors/test_nwp_core.py` — solar de-accumulation unit test (Task 1).
- `tests/connectors/test_hrdps_datamart.py` — align var refs; pinned-solar assertion; upgrade the network test (Task 2, 3).
- `README.md`, `docs/superpowers/plans/2026-05-31-caspar-seed-acquisition.md` (one-line cross-ref) (Task 4).

**Create**
- `docs/adr/0014-hrdps-solar-accumulated.md` (Task 4). (If the CaSPAr branch lands first and also adds an ADR-0014, renumber to the next free ADR number at implementation time.)

---

### Task 1: Fix `nwp_core` solar (accumulated J/m² → mean W/m²)

**Files:**
- Modify: `src/microclimate/connectors/nwp_core.py`
- Test: `tests/connectors/test_nwp_core.py`

- [ ] **Step 1: Write the failing test** (append to `tests/connectors/test_nwp_core.py`; it imports `build_hrdps_dataset`/`VAR_MAP` from conftest and overrides solar to accumulated J/m²):

```python
def test_solar_is_de_accumulated_to_mean_wm2() -> None:
    from datetime import UTC, datetime

    from microclimate.connectors.nwp_core import dataset_to_forecast_frame
    from tests.connectors.conftest import VAR_MAP, build_hrdps_dataset

    ds = build_hrdps_dataset(lead_hours=(0, 1, 2))
    solar_var = VAR_MAP["solar_radiation_wm2"]
    # Accumulated downward shortwave (J/m²) at the target cell (0,0): run-total.
    # h0=0, h1=3_600_000, h2=5_400_000  → per-hour mean flux: h1=1000 W/m², h2=500 W/m².
    ds[solar_var].values[0, 0, 0] = 0.0
    ds[solar_var].values[1, 0, 0] = 3_600_000.0
    ds[solar_var].values[2, 0, 0] = 5_400_000.0

    out = dataset_to_forecast_frame(
        ds, VAR_MAP, issue_time=datetime(2026, 5, 31, 0, tzinfo=UTC),
        lat=51.0, lon=-114.0, lead_hours=[1, 2],
    ).set_index("lead_hour")
    assert out.loc[1, "solar_radiation_wm2"] == 1000.0
    assert out.loc[2, "solar_radiation_wm2"] == 500.0


def test_solar_clamps_negative_to_zero() -> None:
    from datetime import UTC, datetime

    from microclimate.connectors.nwp_core import dataset_to_forecast_frame
    from tests.connectors.conftest import VAR_MAP, build_hrdps_dataset

    ds = build_hrdps_dataset(lead_hours=(0, 1))
    solar_var = VAR_MAP["solar_radiation_wm2"]
    ds[solar_var].values[0, 0, 0] = 5_000.0   # h0 accumulation > h1 (e.g. reset/noise)
    ds[solar_var].values[1, 0, 0] = 0.0
    out = dataset_to_forecast_frame(
        ds, VAR_MAP, issue_time=datetime(2026, 5, 31, 0, tzinfo=UTC),
        lat=51.0, lon=-114.0, lead_hours=[1],
    )
    assert out["solar_radiation_wm2"].iloc[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/connectors/test_nwp_core.py::test_solar_is_de_accumulated_to_mean_wm2 -v`
Expected: FAIL — current code passes solar through (would assert 3_600_000.0 ≠ 1000.0).

- [ ] **Step 3: Implement the fix** in `src/microclimate/connectors/nwp_core.py`:

(a) Add the constant near the other unit constants (after `_PCT_TO_FRACTION`):
```python
_SECONDS_PER_HOUR: float = 3600.0
```

(b) In `dataset_to_forecast_frame`, replace the solar pass-through. Change the sampling block — delete the line `solar = _sample(solar_da, h, iy, ix)` and add solar de-accumulation next to the precip de-accumulation:
```python
        # -- Precip: de-accumulate acc(h) − acc(h-1), clamp ≥ 0 ----
        acc_h = _sample(precip_da, h, iy, ix)
        acc_prev = _sample(precip_da, h - 1, iy, ix)
        precip_mm = max(0.0, acc_h - acc_prev)

        # -- Solar: HRDPS DSWRF is accumulated J/m² from run start. De-accumulate
        #    like precip, then ÷Δt → mean W/m² over the hour. Clamp ≥ 0. (ADR-0014)
        solar_acc_h = _sample(solar_da, h, iy, ix)
        solar_acc_prev = _sample(solar_da, h - 1, iy, ix)
        solar_wm2 = max(0.0, (solar_acc_h - solar_acc_prev) / _SECONDS_PER_HOUR)
```
and change the row dict entry from `"solar_radiation_wm2": solar,` to `"solar_radiation_wm2": solar_wm2,`.

(c) Update the module docstring's solar line:
```
    solar_radiation_wm2    accumulated J/m² from run-start → mean W/m² over the hour
                           (de-accumulate: diff vs previous hour, ÷3600 s; clamp ≥ 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/connectors/test_nwp_core.py -v`
Expected: PASS (new + existing nwp_core tests).

- [ ] **Step 5: Confirm no regressions in the broader suite**

Run: `uv run pytest tests/connectors -q`
Expected: PASS. (The existing `build_hrdps_dataset` has constant solar 300 → de-accumulates to 0.0; no existing test asserts a solar value, and 0.0 satisfies `FORECAST_FRAME`'s `ge(0)`.)

- [ ] **Step 6: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/connectors/nwp_core.py tests/connectors/test_nwp_core.py
git commit -m "fix(connectors): nwp_core de-accumulates HRDPS solar (accumulated J/m² → mean W/m²)"
```

---

### Task 2: Reconcile `HrdpsDatamartSource` to the live layout + align the synthetic fixture

**Files:**
- Modify: `src/microclimate/connectors/sources/hrdps_datamart.py`
- Modify: `tests/connectors/conftest.py`
- Modify: `tests/connectors/test_hrdps_datamart.py`

- [ ] **Step 1: Align the synthetic fixture to reality** in `tests/connectors/conftest.py`.

Change `VAR_MAP` to the identity map (the connector will produce canonical-named vars):
```python
VAR_MAP: dict[str, str] = {
    "temp_c": "temp_c",
    "dewpoint_c": "dewpoint_c",
    "surface_pressure_hpa": "surface_pressure_hpa",
    "precip_mm": "precip_mm",
    "cloud_cover_fraction": "cloud_cover_fraction",
    "solar_radiation_wm2": "solar_radiation_wm2",
    "wind_speed_ms": "wind_speed_ms",
    "wind_dir_deg": "wind_dir_deg",
}
```

In `build_hrdps_dataset`, name the data variables by these canonical keys (not `t2m`/`d2m`/…), and make **solar accumulated J/m²** at the target cell so it de-accumulates to a known flux. Set the target-cell (0,0) values to:
```
temp_c                = 288.15 K   → 15.0 °C
dewpoint_c            = 278.15 K   → 5.0 °C
surface_pressure_hpa  = 90000 Pa   → 900.0 hPa
precip_mm  (accum)    = [0.0, 0.5, 2.0, 2.0]   → per-hour [0.5, 1.5, 0.0] mm
cloud_cover_fraction  = 50 %       → 0.5
solar_radiation_wm2 (accum J/m²) = [0.0, 3_600_000, 7_200_000, 7_200_000]
                                   → per-hour mean flux [1000.0, 1000.0, 0.0] W/m²
wind_speed_ms         = 5.0 m/s    → 5.0
wind_dir_deg          = 270 deg    → 270.0
```
Update the builder's docstring value table to match (canonical names, accumulated solar). Keep the alternate-cell (1,1) distinct values. The variable arrays are still produced via the existing `_fill(target, other)` helper — only the **names** (dict keys for `data_vars`) and the **solar/precip accumulation arrays** change.

- [ ] **Step 2: Update datamart tests** in `tests/connectors/test_hrdps_datamart.py`:
  - Any reference to a shortName data-var (e.g. asserting `ds["t2m"]`) → canonical name. (The happy-path output assertions are already on `FORECAST_FRAME` columns like `temp_c`, so most are unaffected.)
  - Add a pinned-solar assertion mirroring the precip one:

```python
def test_happy_path_pinned_solar_de_accumulation() -> None:
    source = _make_source()
    df = source.fetch_forecast(
        datetime(2026, 5, 31, 0, tzinfo=UTC), lat=51.0, lon=-114.0, lead_hours=[1, 2, 3]
    ).set_index("lead_hour")
    assert df.loc[1, "solar_radiation_wm2"] == 1000.0
    assert df.loc[2, "solar_radiation_wm2"] == 1000.0
    assert df.loc[3, "solar_radiation_wm2"] == 0.0
```

- [ ] **Step 3: Run the fixture-aligned tests to verify they fail against the OLD connector var_map**

Run: `uv run pytest tests/connectors/test_hrdps_datamart.py -v`
Expected: FAIL — `fetch_forecast` still passes the old shortName `_HRDPS_VAR_MAP` to `dataset_to_forecast_frame`, which won't find canonical-named vars in the fixture. This drives Step 4.

- [ ] **Step 4: Reconcile the connector** in `src/microclimate/connectors/sources/hrdps_datamart.py`.

(a) Base URL + var map (replace the `_DATAMART_BASE`, `HRDPS_VAR_MAP`, `_HRDPS_VAR_MAP`, `_GRIB_SHORT_NAMES` block):
```python
# MSC Datamart root. The HRDPS continental path is date-partitioned (verified 2026-05-31).
_DATAMART_BASE: str = "https://dd.weather.gc.ca"

# canonical column → (MSC variable code, level tag) used to build the Datamart filename.
# Verified against live HRDPS continental 2.5 km GRIB2 (run 2026-05-31 18Z).
HRDPS_VAR_MAP: dict[str, tuple[str, str]] = {
    "temp_c": ("TMP", "AGL-2m"),
    "dewpoint_c": ("DPT", "AGL-2m"),
    "surface_pressure_hpa": ("PRES", "Sfc"),
    "precip_mm": ("APCP", "Sfc"),
    "cloud_cover_fraction": ("TCDC", "Sfc"),
    "solar_radiation_wm2": ("DSWRF", "Sfc"),
    "wind_speed_ms": ("WIND", "AGL-10m"),
    "wind_dir_deg": ("WDIR", "AGL-10m"),
}

# The connector names dataset variables by their canonical names, so the var_map handed
# to nwp_core is the identity map (no ECMWF-shortName indirection).
_IDENTITY_VAR_MAP: dict[str, str] = {canon: canon for canon in HRDPS_VAR_MAP}
```

(b) URL builder (replace `_build_datamart_url`):
```python
def _build_datamart_url(issue_time: datetime, lead_hour: int, var: str, level: str) -> str:
    """Build the (verified) MSC Datamart HRDPS continental 2.5 km GRIB2 URL.

    Layout (verified 2026-05-31):
      {BASE}/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{hhh}/
        {YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LEVEL}_RLatLon0.0225_PT{hhh}H.grib2
    """
    date = issue_time.strftime("%Y%m%d")
    hh = issue_time.strftime("%H")
    hhh = f"{lead_hour:03d}"
    fn = f"{date}T{hh}Z_MSC_HRDPS_{var}_{level}_RLatLon0.0225_PT{hhh}H.grib2"
    return f"{_DATAMART_BASE}/{date}/WXO-DD/model_hrdps/continental/2.5km/{hh}/{hhh}/{fn}"
```

(c) `_open_latest_run`: download per `(lead, canonical var)`, decode the **sole** data-var (robust to cfgrib `unknown` names), name it by the canonical key, stack `(lead_hour, y, x)`. Replace the body's download/decode/combine loop with:
```python
    with tempfile.TemporaryDirectory() as tmpdir:
        # {lead_hour: {canonical_var: DataArray}}
        per_lh: dict[int, dict[str, xr.DataArray]] = {}
        for lh in all_lead_hours:
            per_lh[lh] = {}
            for canon, (var, level) in HRDPS_VAR_MAP.items():
                url = _build_datamart_url(issue_time, lh, var, level)
                data_bytes = http_get_bytes(url)  # SourceUnavailable propagates
                tmp_path = f"{tmpdir}/{lh}_{canon}.grib2"
                try:
                    with open(tmp_path, "wb") as fh:
                        fh.write(data_bytes)
                except OSError as exc:
                    raise SourceUnavailable(
                        f"Disk I/O error writing temp GRIB2 for {canon!r}, lead_hour={lh}: {exc}"
                    ) from exc
                try:
                    ds_single: xr.Dataset = cfgrib.open_dataset(  # type: ignore[reportUnknownMemberType]
                        tmp_path, indexpath=""
                    )
                except Exception as exc:
                    raise ForecastUnavailable(
                        f"Failed to decode GRIB2 for {canon!r} ({var}/{level}), lead_hour={lh}: {exc}"
                    ) from exc
                # Each MSC single-variable file holds exactly one data var; cfgrib may name
                # it 'unknown' (e.g. APCP/TCDC) so select by sole-var, not by name.
                names = list(ds_single.data_vars)
                if len(names) != 1:
                    raise ForecastUnavailable(
                        f"Expected exactly one data variable in {canon!r} ({var}/{level}) "
                        f"file at lead_hour={lh}, got {names}."
                    )
                per_lh[lh][canon] = ds_single[names[0]]
```
Then update the combine block to iterate canonical names instead of `_GRIB_SHORT_NAMES`:
```python
        first_da = next(iter(per_lh[all_lead_hours[0]].values()))
        spatial_dims: tuple[str, ...] = tuple(str(d) for d in first_da.dims)  # type: ignore[reportUnknownMemberType]
        spatial_shape: tuple[int, ...] = tuple(int(s) for s in first_da.shape)  # type: ignore[reportUnknownMemberType]

        data_vars: dict[str, xr.DataArray] = {}
        for canon in HRDPS_VAR_MAP:
            stacked: np.ndarray = np.stack(  # type: ignore[reportUnknownMemberType]
                [per_lh[lh][canon].values for lh in all_lead_hours],  # type: ignore[reportUnknownMemberType]
                axis=0,
            )
            data_vars[canon] = xr.DataArray(stacked, dims=("lead_hour", *spatial_dims))

        first_var_da = per_lh[all_lead_hours[0]][next(iter(HRDPS_VAR_MAP))]
```
Keep the existing latitude/longitude extraction block (the 1-D→2-D broadcast logic is unchanged) and the `lh_coord`/`coords`/`xr.Dataset(...).load()` tail unchanged.

(d) `fetch_forecast`: pass the identity map. Change the `dataset_to_forecast_frame(ds, _HRDPS_VAR_MAP, …)` call to `dataset_to_forecast_frame(ds, _IDENTITY_VAR_MAP, …)`.

- [ ] **Step 5: Run datamart tests to verify they pass**

Run: `uv run pytest tests/connectors/test_hrdps_datamart.py tests/connectors/test_nwp_core.py -v`
Expected: PASS (incl. the new pinned-solar test).

- [ ] **Step 6: Run pyright** (the `tuple[str,str]` var map + reworked loop)

Run: `uv run pyright`
Expected: 0 errors. Add/adjust the narrow `# pyright: ignore[reportUnknownMemberType]` comments on cfgrib/xarray `.values`/`.dims` access exactly where pyright reports them (the surrounding code already uses this pattern).

- [ ] **Step 7: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add src/microclimate/connectors/sources/hrdps_datamart.py tests/connectors/conftest.py tests/connectors/test_hrdps_datamart.py
git commit -m "feat(connectors): reconcile HrdpsDatamartSource to live Datamart layout (verified); canonical-name vars + sole-data-var decode"
```

---

### Task 3: `network`-marked live integration test

**Files:**
- Modify: `tests/connectors/test_hrdps_datamart.py`

- [ ] **Step 1: Replace the existing `test_network_smoke_open_latest_run`** with a real end-to-end live test that finds the latest available run and asserts a valid forecast frame:

```python
@pytest.mark.network
def test_network_live_datamart_forecast_frame() -> None:  # pragma: no cover
    """Live: fetch the latest HRDPS run from MSC Datamart for the Lethbridge point.

    Deselected by default (network marker). Requires eccodes + network to dd.weather.gc.ca.
    Dynamically finds the most recent published run (publish lag ~3-4 h) so it doesn't pin a
    run that has rolled off Datamart's recent window.
    """
    import urllib.request
    from datetime import timedelta

    from microclimate.connectors.sources.hrdps_datamart import (
        HrdpsDatamartSource,
        _build_datamart_url,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
    )
    from microclimate.contracts.forecast_frame import FORECAST_FRAME

    # Find the latest run whose lead-1 TMP file exists.
    now = datetime.now(UTC)
    issue: datetime | None = None
    for hours_back in range(0, 36):
        t = now - timedelta(hours=hours_back)
        if t.hour not in (0, 6, 12, 18):
            continue
        url = _build_datamart_url(t, 1, "TMP", "AGL-2m")
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status == 200:
                    issue = t.replace(minute=0, second=0, microsecond=0)
                    break
        except Exception:  # noqa: BLE001 — any network/HTTP error: try an earlier run
            continue
    if issue is None:
        pytest.skip("No recent HRDPS run reachable on Datamart")

    df = HrdpsDatamartSource().fetch_forecast(issue, lat=49.70, lon=-112.77, lead_hours=[1, 2])
    FORECAST_FRAME.validate(df)
    assert list(df["lead_hour"]) == [1, 2]
    assert df["temp_c"].between(-60, 60).all()
    assert df["cloud_cover_fraction"].between(0, 1).all()
    assert df["solar_radiation_wm2"].between(0, 1500).all()
```

- [ ] **Step 2: Confirm it's deselected by default, and (optionally) run it live**

Run: `uv run pytest tests/connectors/test_hrdps_datamart.py -q` → the network test is deselected (default `-m 'not network'`).
Optional live check (eccodes is installed): `uv run pytest -m network tests/connectors/test_hrdps_datamart.py::test_network_live_datamart_forecast_frame -v` → PASS or a clean `skip` if no run is reachable.

- [ ] **Step 3: Commit**

```bash
git add tests/connectors/test_hrdps_datamart.py
git commit -m "test(connectors): network-marked live Datamart forecast-frame integration test"
```

---

### Task 4: ADR + docstrings + README + CaSPAr cross-reference

**Files:**
- Create: `docs/adr/0014-hrdps-solar-accumulated.md`
- Modify: `src/microclimate/connectors/sources/hrdps_datamart.py` (docstring), `README.md`, `docs/superpowers/plans/2026-05-31-caspar-seed-acquisition.md`

- [ ] **Step 1: Write the ADR** (`docs/adr/0014-hrdps-solar-accumulated.md`; if 0014 is taken on main by then, use the next free number):

```markdown
# 14. HRDPS solar is accumulated J/m²; nwp_core de-accumulates to mean W/m²

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0007 (one HRDPS spec — seed/live parity), the shared nwp_core core.

## Context

`nwp_core` originally treated `solar_radiation_wm2` as an instantaneous W/m² flux
(pass-through). A spike decoding real MSC Datamart HRDPS GRIB2 (run 2026-05-31 18Z) with
eccodes showed `DSWRF` decodes as `ssrd` = **accumulated downward shortwave J/m² from run
start** — so pass-through was wrong by ~3 orders of magnitude.

## Decision

`nwp_core` de-accumulates solar exactly like precip — `solar(h) − solar(h−1)` — then divides
by 3600 s to yield the **mean W/m² over the hour**, clamped ≥ 0. This is a shared-core change
applying to **both** HRDPS connectors (Datamart live + CaSPAr seed). The Datamart connector is
verified against real data here; the CaSPAr connector inherits the behavior and confirms its
own solar encoding when its sample lands (subsystem-A gate).

## Consequences

- Solar values are now physically correct mean hourly fluxes.
- Requires `h−1` in the dataset (already required for precip de-accumulation — no new constraint).
- If a future HRDPS source encodes solar instantaneously, this becomes per-connector config;
  deferred (YAGNI) until proven.
```

- [ ] **Step 2: Refresh the `hrdps_datamart.py` module docstring** — replace the "URL pattern unverified / shortNames unverified" caveats with: verified against live MSC Datamart (run 2026-05-31 18Z); the date-partitioned URL layout; netCDF-N/A (GRIB2); and the sole-data-var decode (MSC single-variable files, some named `unknown` by cfgrib).

- [ ] **Step 3: Update `README.md` Project status** — note the live MSC Datamart HRDPS connector (`hrdps_datamart`) is verified against real data, and `nwp_core` solar is de-accumulated (ADR-0014).

- [ ] **Step 4: Add a one-line cross-reference** to `docs/superpowers/plans/2026-05-31-caspar-seed-acquisition.md` (its connector-reconciliation task), e.g. under Task 4 (connector reconciliation): "Solar: verify CaSPAr's solar encoding against `nwp_core`'s de-accumulated-J/m² handling (ADR-0014); adjust only if CaSPAr differs from Datamart." *(If that plan file isn't present on this branch — it lives on the CaSPAr branch — skip this step and note it in the PR description so it's applied when the branches integrate.)*

- [ ] **Step 5: Full gate + commit**

```bash
uv run ruff format . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest
git add docs/adr/0014-hrdps-solar-accumulated.md src/microclimate/connectors/sources/hrdps_datamart.py README.md docs/superpowers/plans/2026-05-31-caspar-seed-acquisition.md 2>/dev/null || git add docs/adr/0014-hrdps-solar-accumulated.md src/microclimate/connectors/sources/hrdps_datamart.py README.md
git commit -m "docs: ADR-0014 HRDPS solar accumulated; datamart docstring verified; README status"
```

---

## Final Integration

- [ ] Delete or gitignore the throwaway spike artifacts so they don't get committed:

```bash
rm -rf scratch/
```

- [ ] Push and open a PR (main is PR-only):

```bash
git push -u origin spec/datamart-connector-reconciliation
gh pr create --fill --base main
```

- [ ] After automated review + CI, address feedback and merge.

---

## Self-review notes

- **Spec coverage:** nwp_core solar fix (Task 1) ✓; connector URL+var-map+sole-data-var decode (Task 2) ✓; injected-synthetic aligned fixture (Task 2) ✓; network-marked live test (Task 3) ✓; ADR + docstrings + README + CaSPAr cross-ref (Task 4) ✓; `scratch/` cleanup (Final) ✓.
- **Coupling handled:** Task 1 verifies solar by *overriding* the fixture's solar (doesn't depend on Task 2's fixture rename); existing tests stay green because no current test asserts a solar value and 0.0 satisfies `FORECAST_FRAME`. Task 2 owns the fixture's canonical-naming + accumulated-solar default and the connector's identity `var_map`.
- **Type consistency:** `HRDPS_VAR_MAP` is `dict[str, tuple[str,str]]`; `_IDENTITY_VAR_MAP` is `dict[str,str]` and is what `dataset_to_forecast_frame` receives; `_build_datamart_url(issue_time, lead_hour, var, level)` matches its call site; `build_hrdps_dataset` data-vars and `VAR_MAP` keys are both the 8 canonical names.
- **Out of scope confirmed:** CaSPAr connector (inherits solar fix, verifies at its gate), committed GRIB2 fixtures.
```
