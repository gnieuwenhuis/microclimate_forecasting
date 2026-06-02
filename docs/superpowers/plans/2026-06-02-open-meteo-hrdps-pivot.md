# Open-Meteo HRDPS Pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dead CaSPAr seed and native MSC GRIB2 HRDPS feeds with a single pure-HTTP **Open-Meteo** connector serving both live inference (`/v1/forecast`) and the deep historical seed backfill (Historical Forecast API), and make inference stateless (logger removed).

**Architecture:** A new `openmeteo` `NWPSource` returns `FORECAST_FRAME` directly from JSON (no `nwp_core`, no `cfgrib`/`xarray`). Training data comes from an idempotent/additive retrain-time backfill into the existing `TrainingStore`. The deep archive is short-lead-stitched while live serves full leads — the accepted v1 lead-time skew (ADR-0019 §1b), fail-safe via the publish gate.

**Tech Stack:** Python 3, `requests` (via `connectors/http.py`), `pandas`, `pandera` (`FORECAST_FRAME`), `pytest` (with a `network` marker), `uv`.

**Authoritative docs:** ADR-0019, `docs/superpowers/specs/2026-06-02-open-meteo-hrdps-pivot-design.md`, CONTEXT.md.

**Conventions to honor:** UTC everywhere; `register_source` registry; injectable fetcher for hermetic tests (mirror `EnvCanadaSource(fetcher=...)` / `HrdpsDatamartSource(opener=...)`); commit after each green step; run `uv run ruff format . && uv run ruff check . && uv run pyright && uv run lint-imports && uv run pytest` before declaring a task done.

**The 8 canonical variables** (`PHYSICAL_VARS`, fixed order) and their Open-Meteo hourly names:

| canonical | Open-Meteo `hourly` var | conversion |
|---|---|---|
| `temp_c` | `temperature_2m` | none (°C) |
| `dewpoint_c` | `dew_point_2m` | none (°C) |
| `surface_pressure_hpa` | `surface_pressure` | none (hPa) |
| `precip_mm` | `precipitation` | none (mm/h) |
| `cloud_cover_fraction` | `cloud_cover` | **÷100** (% → fraction) |
| `solar_radiation_wm2` | `shortwave_radiation` | none (W/m²) |
| `wind_speed_ms` | `wind_speed_10m` | none (request `wind_speed_unit=ms`) |
| `wind_dir_deg` | `wind_direction_10m` | none (degrees) |

---

## File Structure

- Create: `src/microclimate/connectors/sources/openmeteo.py` — the connector (request builder, parser, `OpenMeteoSource`).
- Create: `src/microclimate/pipelines/backfill.py` — issue-time generator + `backfill_store`.
- Create: `tests/connectors/test_openmeteo.py`, `tests/connectors/test_openmeteo_network.py`, `tests/pipelines/test_backfill.py`, `tests/connectors/fixtures/openmeteo_forecast.json`, `tests/connectors/fixtures/openmeteo_historical.json`, `docs/spikes/0004-openmeteo-hrdps-access.md`.
- Modify: `src/microclimate/connectors/sources/__init__.py`, `src/microclimate/pipelines/inference.py`, `config/deployments/lethbridge.yml`, `pyproject.toml`, `DATA_LICENSES.md`, `.github/workflows/inference.yml`, `README.md`, `tests/connectors/test_connector_contract.py`, `tests/connectors/conftest.py`, `tests/config/test_schema.py`.
- Delete: `src/microclimate/connectors/nwp_core.py`, `src/microclimate/connectors/sources/hrdps_datamart.py`, `src/microclimate/connectors/sources/hrdps_caspar.py`, `tests/connectors/test_nwp_core.py`, `tests/connectors/test_hrdps_datamart.py`, `tests/connectors/test_hrdps_caspar.py`.

---

## Task 1: Capture real Open-Meteo responses as fixtures + network smoke test

**Files:**
- Create: `tests/connectors/fixtures/openmeteo_forecast.json`
- Create: `tests/connectors/fixtures/openmeteo_historical.json`
- Create: `tests/connectors/test_openmeteo_network.py`
- Create: `docs/spikes/0004-openmeteo-hrdps-access.md`

- [ ] **Step 1: Capture the live `/v1/forecast` fixture** (Lethbridge CDA target cell)

Run:
```bash
mkdir -p tests/connectors/fixtures
curl -s "https://api.open-meteo.com/v1/forecast?latitude=49.70&longitude=-112.77&models=gem_hrdps_continental&cell_selection=land&wind_speed_unit=ms&timezone=GMT&forecast_days=3&hourly=temperature_2m,dew_point_2m,surface_pressure,precipitation,cloud_cover,shortwave_radiation,wind_speed_10m,wind_direction_10m" -o tests/connectors/fixtures/openmeteo_forecast.json
python3 -c "import json;d=json.load(open('tests/connectors/fixtures/openmeteo_forecast.json'));assert 'hourly' in d and 'temperature_2m' in d['hourly'];print('ok',len(d['hourly']['time']),'hours')"
```
Expected: `ok 72 hours` (or similar; ≥49).

- [ ] **Step 2: Capture the deep historical fixture** (a fixed 2024 date so the fixture is stable)

Run:
```bash
curl -s "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=49.70&longitude=-112.77&models=gem_hrdps_continental&cell_selection=land&wind_speed_unit=ms&timezone=GMT&start_date=2024-06-01&end_date=2024-06-03&hourly=temperature_2m,dew_point_2m,surface_pressure,precipitation,cloud_cover,shortwave_radiation,wind_speed_10m,wind_direction_10m" -o tests/connectors/fixtures/openmeteo_historical.json
python3 -c "import json;d=json.load(open('tests/connectors/fixtures/openmeteo_historical.json'));assert len(d['hourly']['time'])==72, d['hourly']['time'][:2];print('ok',d['hourly']['time'][0],'->',d['hourly']['time'][-1])"
```
Expected: `ok 2024-06-01T00:00 -> 2024-06-03T23:00`.

- [ ] **Step 3: Write the spike note**

Create `docs/spikes/0004-openmeteo-hrdps-access.md`:
```markdown
# Spike 0004 — Open-Meteo HRDPS access (2026-06-02)

Empirical probe (no API key) confirming the ADR-0019 source decision.

- `api.open-meteo.com/v1/forecast` and `historical-forecast-api.open-meteo.com/v1/forecast`
  both return GEM HRDPS Continental (model=`gem_hrdps_continental`) with no key — free
  non-commercial access confirmed.
- Deep history confirmed: 2024-06 returns full data on the historical host.
- **Key finding (ADR-0019 §1b):** the deep archive is a STITCHED SHORT-LEAD series.
  `temperature_2m_previous_day1` returns data (~24h lead) but `temperature_2m_previous_day2`
  and beyond are NULL for HRDPS — so deep full-1..48-lead-per-run history is NOT available
  for free. v1 trains on the short-lead seed and accepts the lead-time skew.
- Response shape: `{"hourly": {"time": [...], "<var>": [...], ...}}`, times are UTC-naive
  ISO `YYYY-MM-DDTHH:MM` when `timezone=GMT`.
- Fixtures captured: `tests/connectors/fixtures/openmeteo_forecast.json`,
  `openmeteo_historical.json`.
```

- [ ] **Step 4: Write the network smoke test**

Create `tests/connectors/test_openmeteo_network.py`:
```python
"""Live Open-Meteo smoke tests (network-marked; deselected by default)."""
from __future__ import annotations

import json

import pytest

from microclimate.connectors.http import http_get

_SHARED = (
    "latitude=49.70&longitude=-112.77&models=gem_hrdps_continental"
    "&cell_selection=land&wind_speed_unit=ms&timezone=GMT"
    "&hourly=temperature_2m,dew_point_2m,surface_pressure,precipitation,"
    "cloud_cover,shortwave_radiation,wind_speed_10m,wind_direction_10m"
)


@pytest.mark.network
def test_live_forecast_returns_hourly() -> None:
    body = http_get(f"https://api.open-meteo.com/v1/forecast?{_SHARED}&forecast_days=3")
    hourly = json.loads(body)["hourly"]
    assert "temperature_2m" in hourly and len(hourly["time"]) >= 49


@pytest.mark.network
def test_historical_forecast_returns_deep_hourly() -> None:
    body = http_get(
        "https://historical-forecast-api.open-meteo.com/v1/forecast?"
        f"{_SHARED}&start_date=2024-06-01&end_date=2024-06-02"
    )
    hourly = json.loads(body)["hourly"]
    assert len(hourly["time"]) == 48 and hourly["temperature_2m"][0] is not None
```

- [ ] **Step 5: Run the network tests to confirm they pass**

Run: `uv run pytest -m network tests/connectors/test_openmeteo_network.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/connectors/fixtures/openmeteo_forecast.json tests/connectors/fixtures/openmeteo_historical.json tests/connectors/test_openmeteo_network.py docs/spikes/0004-openmeteo-hrdps-access.md
git commit -m "test(openmeteo): capture API fixtures + network smoke tests (spike 0004)"
```

---

## Task 2: Open-Meteo JSON → FORECAST_FRAME parser (pure, hermetic)

**Files:**
- Create: `src/microclimate/connectors/sources/openmeteo.py` (parser + var map only this task)
- Test: `tests/connectors/test_openmeteo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/connectors/test_openmeteo.py`:
```python
"""Hermetic tests for the Open-Meteo connector (no network)."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from microclimate.contracts.forecast_frame import FORECAST_FRAME

_FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def test_parse_historical_fixture_to_forecast_frame() -> None:
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    payload = _load("openmeteo_historical.json")
    # Fixture starts 2024-06-01T00:00; pick t0 there so leads 1..3 map to 01:00/02:00/03:00.
    t0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    df = _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1, 2, 3])

    FORECAST_FRAME.validate(df)
    assert list(df["lead_hour"]) == [1, 2, 3]
    # cloud_cover is % in the payload → fraction in the frame (0..1).
    assert (df["cloud_cover_fraction"] >= 0).all() and (df["cloud_cover_fraction"] <= 1).all()
    # valid_time == t0 + lead_hour.
    import pandas as pd

    for _, row in df.iterrows():
        assert row["valid_time"] == pd.Timestamp(t0) + pd.Timedelta(hours=int(row["lead_hour"]))


def test_parse_raises_when_lead_hour_absent() -> None:
    from microclimate.connectors.base import ForecastUnavailable
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    payload = _load("openmeteo_historical.json")
    far = datetime(2024, 6, 3, 23, 0, tzinfo=UTC)  # t0+1 falls outside the fixture window
    with pytest.raises(ForecastUnavailable):
        _parse_hourly_to_forecast_frame(payload, issue_time=far, lead_hours=[1])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/connectors/test_openmeteo.py -v`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (`openmeteo` not created yet).

- [ ] **Step 3: Write the parser**

Create `src/microclimate/connectors/sources/openmeteo.py`:
```python
"""HRDPS via Open-Meteo — single pure-HTTP+JSON NWP source for live + historical (ADR-0019).

Live  → api.open-meteo.com/v1/forecast (current run, full leads).
Hist  → historical-forecast-api.open-meteo.com/v1/forecast (deep, stitched short-lead).
Both return {"hourly": {"time": [...], "<var>": [...]}}; one parser serves both.
No nwp_core, no cfgrib/xarray — precip/solar arrive already de-accumulated.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.connectors.base import ForecastUnavailable
from microclimate.contracts.forecast_frame import FORECAST_FRAME
from microclimate.contracts.physical_vars import PHYSICAL_VARS

# canonical column → Open-Meteo hourly variable name.
_OPENMETEO_VAR_MAP: dict[str, str] = {
    "temp_c": "temperature_2m",
    "dewpoint_c": "dew_point_2m",
    "surface_pressure_hpa": "surface_pressure",
    "precip_mm": "precipitation",
    "cloud_cover_fraction": "cloud_cover",
    "solar_radiation_wm2": "shortwave_radiation",
    "wind_speed_ms": "wind_speed_10m",
    "wind_dir_deg": "wind_direction_10m",
}
_PCT_TO_FRACTION: float = 100.0
# Open-Meteo emits naive ISO timestamps in UTC when timezone=GMT.
_OM_TIME_FMT: str = "%Y-%m-%dT%H:%M"


def _parse_hourly_to_forecast_frame(
    payload: dict[str, object],
    *,
    issue_time: datetime,
    lead_hours: Sequence[int],
) -> pd.DataFrame:
    """Map an Open-Meteo `hourly` payload to a FORECAST_FRAME-valid DataFrame.

    Selects, for each requested lead h, the value at valid_time = issue_time + h.
    Raises ForecastUnavailable if any requested valid_time is absent or null.
    """
    issue_utc = (
        issue_time.astimezone(UTC) if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)
    )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ForecastUnavailable("Open-Meteo payload missing 'hourly'/'time'.")
    times: list[str] = list(hourly["time"])  # type: ignore[arg-type]
    index_by_time: dict[str, int] = {t: i for i, t in enumerate(times)}

    missing_vars = [
        om for om in _OPENMETEO_VAR_MAP.values() if om not in hourly
    ]
    if missing_vars:
        raise ForecastUnavailable(f"Open-Meteo payload missing variable(s): {missing_vars}.")

    rows: list[dict[str, object]] = []
    for h in lead_hours:
        valid = issue_utc + timedelta(hours=int(h))
        key = valid.strftime(_OM_TIME_FMT)
        idx = index_by_time.get(key)
        if idx is None:
            raise ForecastUnavailable(
                f"Open-Meteo series has no entry for valid_time {key} (lead_hour={h})."
            )
        row: dict[str, object] = {
            "issue_time": pd.Timestamp(issue_utc),
            "lead_hour": int(h),
            "valid_time": pd.Timestamp(valid),
        }
        for canon in PHYSICAL_VARS:
            raw = hourly[_OPENMETEO_VAR_MAP[canon]][idx]  # type: ignore[index]
            if raw is None:
                raise ForecastUnavailable(
                    f"Open-Meteo {canon!r} is null at valid_time {key} (lead_hour={h})."
                )
            value = float(raw)
            if canon == "cloud_cover_fraction":
                value = max(0.0, min(1.0, value / _PCT_TO_FRACTION))
            row[canon] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    return FORECAST_FRAME.validate(df)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/connectors/test_openmeteo.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/connectors/sources/openmeteo.py tests/connectors/test_openmeteo.py
git commit -m "feat(openmeteo): JSON hourly -> FORECAST_FRAME parser"
```

---

## Task 3: Request builder + endpoint routing (pure)

**Files:**
- Modify: `src/microclimate/connectors/sources/openmeteo.py`
- Test: `tests/connectors/test_openmeteo.py`

- [ ] **Step 1: Add the failing test** (append to `tests/connectors/test_openmeteo.py`)

```python
def test_request_routing_and_shared_params() -> None:
    from microclimate.connectors.sources.openmeteo import _build_request

    now = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    live_url, live_params = _build_request(
        datetime(2026, 6, 2, 6, 0, tzinfo=UTC), 49.70, -112.77, [1, 48], now=now
    )
    hist_url, hist_params = _build_request(
        datetime(2024, 6, 1, 0, 0, tzinfo=UTC), 49.70, -112.77, [1, 48], now=now
    )

    assert live_url.startswith("https://api.open-meteo.com/")
    assert hist_url.startswith("https://historical-forecast-api.open-meteo.com/")
    # Shared parity keys must be identical across both routes.
    shared = ("latitude", "longitude", "models", "cell_selection", "wind_speed_unit", "timezone", "hourly")
    for k in shared:
        assert live_params[k] == hist_params[k], k
    assert live_params["cell_selection"] == "land"
    assert live_params["models"] == "gem_hrdps_continental"
    # Historical route pins the date window; live route does not.
    assert hist_params["start_date"] == "2024-06-01" and hist_params["end_date"] == "2024-06-03"
    assert "start_date" not in live_params
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/connectors/test_openmeteo.py::test_request_routing_and_shared_params -v`
Expected: FAIL — `_build_request` not defined.

- [ ] **Step 3: Implement the request builder** (add to `openmeteo.py`, below the constants)

```python
_LIVE_URL: str = "https://api.open-meteo.com/v1/forecast"
_HISTORICAL_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_GEM_MODEL: str = "gem_hrdps_continental"
_HOURLY_CSV: str = ",".join(_OPENMETEO_VAR_MAP[c] for c in PHYSICAL_VARS)
# Use the live endpoint when the run is recent enough to still be on it; else the deep archive.
_LIVE_CUTOFF = timedelta(days=2)


def _build_request(
    issue_time: datetime,
    lat: float,
    lon: float,
    lead_hours: Sequence[int],
    *,
    now: datetime,
) -> tuple[str, dict[str, str | int | float]]:
    """Return (url, params). Recent issue_time → live endpoint; older → historical archive."""
    issue_utc = (
        issue_time.astimezone(UTC) if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)
    )
    params: dict[str, str | int | float] = {
        "latitude": lat,
        "longitude": lon,
        "models": _GEM_MODEL,
        "cell_selection": "land",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "hourly": _HOURLY_CSV,
    }
    if issue_utc >= now.astimezone(UTC) - _LIVE_CUTOFF:
        return _LIVE_URL, params
    # Historical: pin a date window covering t0 .. t0 + max(lead).
    end = (issue_utc + timedelta(hours=max(lead_hours))).date()
    params["start_date"] = issue_utc.date().isoformat()
    params["end_date"] = end.isoformat()
    return _HISTORICAL_URL, params
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/connectors/test_openmeteo.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/connectors/sources/openmeteo.py tests/connectors/test_openmeteo.py
git commit -m "feat(openmeteo): request builder + live/historical routing"
```

---

## Task 4: OpenMeteoSource connector (registered NWPSource, injectable fetcher)

**Files:**
- Modify: `src/microclimate/connectors/sources/openmeteo.py`
- Test: `tests/connectors/test_openmeteo.py`

- [ ] **Step 1: Add the failing test**

```python
def test_source_fetch_forecast_hermetic() -> None:
    import json
    from datetime import UTC, datetime

    from microclimate.connectors.sources.openmeteo import OpenMeteoSource

    payload = _load("openmeteo_historical.json")

    def fake_fetcher(url: str, *, params: object) -> str:  # matches http_get(url, params=...)
        return json.dumps(payload)

    source = OpenMeteoSource(fetcher=fake_fetcher, now=datetime(2026, 6, 2, 12, 0, tzinfo=UTC))
    df = source.fetch_forecast(datetime(2024, 6, 1, 0, 0, tzinfo=UTC), 49.70, -112.77, [1, 2, 3])
    FORECAST_FRAME.validate(df)
    assert source.is_live is True
    assert list(df["lead_hour"]) == [1, 2, 3]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/connectors/test_openmeteo.py::test_source_fetch_forecast_hermetic -v`
Expected: FAIL — `OpenMeteoSource` not defined.

- [ ] **Step 3: Implement the connector class** (append to `openmeteo.py`)

```python
import json
from collections.abc import Callable, Mapping

from microclimate.connectors.base import NWPSource, SourceUnavailable
from microclimate.connectors.http import http_get
from microclimate.connectors.registry import register_source

_Fetcher = Callable[..., str]


@register_source("openmeteo")
class OpenMeteoSource(NWPSource):
    """HRDPS via Open-Meteo — live + historical behind one connector (ADR-0019).

    The registry instantiates this with no args, so the defaults must work argument-free.
    ``fetcher`` (default ``http_get``) and ``now`` are injectable for hermetic tests.
    """

    def __init__(self, fetcher: _Fetcher | None = None, now: datetime | None = None) -> None:
        self._fetcher: _Fetcher = fetcher if fetcher is not None else http_get
        self._now: datetime | None = now

    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        now = self._now if self._now is not None else datetime.now(UTC)
        url, params = _build_request(issue_time, lat, lon, lead_hours, now=now)
        body = self._fetcher(url, params=params)  # SourceUnavailable propagates from http_get
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise SourceUnavailable(f"Open-Meteo returned non-JSON for {url!r}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SourceUnavailable(f"Open-Meteo returned a non-object body for {url!r}.")
        if payload.get("error"):
            raise SourceUnavailable(f"Open-Meteo error for {url!r}: {payload.get('reason')}")
        return _parse_hourly_to_forecast_frame(
            payload, issue_time=issue_time, lead_hours=lead_hours
        )
```

Note: keep the `Mapping` import only if used; remove unused imports to satisfy ruff. (`Mapping` is not needed — delete it if you added it.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/connectors/test_openmeteo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run ruff + pyright on the new module**

Run: `uv run ruff check src/microclimate/connectors/sources/openmeteo.py && uv run pyright src/microclimate/connectors/sources/openmeteo.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/connectors/sources/openmeteo.py tests/connectors/test_openmeteo.py
git commit -m "feat(openmeteo): OpenMeteoSource NWPSource (registered, injectable fetcher)"
```

---

## Task 5: Register the source + request-spec parity fitness test

**Files:**
- Modify: `src/microclimate/connectors/sources/__init__.py`
- Test: `tests/connectors/test_openmeteo.py`

- [ ] **Step 1: Add the failing parity test**

```python
def test_request_spec_parity_live_vs_historical() -> None:
    """Spatial/variable parity: shared keys identical across routes (ADR-0019 §1)."""
    from microclimate.connectors.sources.openmeteo import _build_request

    now = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)
    _, live = _build_request(datetime(2026, 6, 2, 6, 0, tzinfo=UTC), 49.70, -112.77, list(range(1, 49)), now=now)
    _, hist = _build_request(datetime(2024, 1, 5, 0, 0, tzinfo=UTC), 49.70, -112.77, list(range(1, 49)), now=now)
    shared_keys = set(live) & set(hist)
    for k in shared_keys:
        assert live[k] == hist[k], f"parity break on {k!r}: {live[k]!r} != {hist[k]!r}"
    # Lead-time provenance is NOT pinned (accepted §1b skew): the route URLs differ by design.
```

- [ ] **Step 2: Register the connector** — edit `src/microclimate/connectors/sources/__init__.py`

Replace the two HRDPS imports with the Open-Meteo one (envcanada/acis lines stay):
```python
from microclimate.connectors.sources import acis as acis  # noqa: F401
from microclimate.connectors.sources import envcanada as envcanada  # noqa: F401
from microclimate.connectors.sources import openmeteo as openmeteo  # noqa: F401
```

- [ ] **Step 3: Run the parity test + confirm registration**

Run: `uv run pytest tests/connectors/test_openmeteo.py::test_request_spec_parity_live_vs_historical tests/connectors/test_connector_contract.py::test_source_conforms_to_contract -v`
Expected: parity test passes; the contract test now parametrizes `openmeteo` and passes structurally.

- [ ] **Step 4: Commit**

```bash
git add src/microclimate/connectors/sources/__init__.py tests/connectors/test_openmeteo.py
git commit -m "feat(openmeteo): register source + request-spec parity fitness test"
```

---

## Task 6: Delete the native GRIB2 stack + drop xarray/cfgrib

**Files:**
- Delete: `src/microclimate/connectors/nwp_core.py`, `src/microclimate/connectors/sources/hrdps_datamart.py`, `src/microclimate/connectors/sources/hrdps_caspar.py`, `tests/connectors/test_nwp_core.py`, `tests/connectors/test_hrdps_datamart.py`, `tests/connectors/test_hrdps_caspar.py`
- Modify: `pyproject.toml`, `tests/connectors/test_connector_contract.py`, `tests/connectors/conftest.py`, `tests/config/test_schema.py`

- [ ] **Step 1: Delete the native modules and their tests**

```bash
git rm src/microclimate/connectors/nwp_core.py \
       src/microclimate/connectors/sources/hrdps_datamart.py \
       src/microclimate/connectors/sources/hrdps_caspar.py \
       tests/connectors/test_nwp_core.py \
       tests/connectors/test_hrdps_datamart.py \
       tests/connectors/test_hrdps_caspar.py
```

- [ ] **Step 2: Remove the `xarray`/`cfgrib` dependencies** — edit `pyproject.toml`

Delete these two lines from `[project].dependencies`:
```
  "xarray>=2026.4.0",
  "cfgrib>=0.9.15.1",
```
Then run: `uv sync`
Expected: lockfile updates, xarray/cfgrib removed.

- [ ] **Step 3: Update `tests/connectors/test_connector_contract.py`**

In `test_source_behavioral_contract`, remove the `hrdps_datamart`/`hrdps_caspar` branches and add an `openmeteo` branch; delete `_assert_hrdps_datamart_behavioral_contract` and `_assert_hrdps_caspar_behavioral_contract`; drop the now-unused `build_hrdps_dataset` import from `.conftest`. Replace the dispatch block with:
```python
    if key == "envcanada":
        _assert_envcanada_behavioral_contract()
        return

    if key == "openmeteo":
        _assert_openmeteo_behavioral_contract()
        return

    pytest.fail(f"No behavioral contract assertion defined for source key {key!r}")
```
And add:
```python
def _assert_openmeteo_behavioral_contract() -> None:
    """Hermetic behavioural assertions for OpenMeteoSource (fixture-backed fetcher)."""
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    from microclimate.connectors.sources.openmeteo import OpenMeteoSource

    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "openmeteo_historical.json").read_text()
    )
    source = OpenMeteoSource(
        fetcher=lambda _url, *, params: json.dumps(payload),
        now=datetime(2026, 6, 2, 12, 0, tzinfo=UTC),
    )
    df = source.fetch_forecast(datetime(2024, 6, 1, 0, 0, tzinfo=UTC), 49.70, -112.77, [1, 2, 3])
    FORECAST_FRAME.validate(df)
    assert list(df["lead_hour"]) == [1, 2, 3]
```

- [ ] **Step 4: Update `tests/connectors/conftest.py` and `tests/config/test_schema.py`**

In `conftest.py`, delete the `build_hrdps_dataset` helper and any `import xarray`/`numpy` solely supporting it. In `tests/config/test_schema.py`, replace any `hrdps_datamart`/`hrdps_caspar` literals used as connector keys with `openmeteo` (search the file for those strings).

Run: `grep -rn "hrdps_datamart\|hrdps_caspar\|nwp_core\|build_hrdps_dataset\|import xarray\|import cfgrib" src tests`
Expected: no matches.

- [ ] **Step 5: Run the full suite + import contract**

Run: `uv run lint-imports && uv run pytest`
Expected: all pass (network deselected). If `pyright` flags removed-import references, fix them.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(connectors): delete native GRIB2 HRDPS stack; drop xarray/cfgrib (ADR-0019)"
```

---

## Task 7: Point the lethbridge deployment at Open-Meteo

**Files:**
- Modify: `config/deployments/lethbridge.yml`
- Test: existing `tests/config/test_schema.py` + `validate_config_sources`

- [ ] **Step 1: Edit `config/deployments/lethbridge.yml`**

Replace the `enabled_sources`, `nwp`, and `training.seed` blocks:
```yaml
enabled_sources: [openmeteo, envcanada]

nwp:
  product: hrdps
  live_connector: openmeteo                # Open-Meteo /v1/forecast — inference (full leads)
  historical_connector: openmeteo          # Open-Meteo Historical Forecast API — seed backfill
  sampling: land                           # Open-Meteo cell_selection (elevation-aware land cell)
  # Same Open-Meteo request spec on both feeds (ADR-0019). Deep seed is short-lead-stitched;
  # live serves full leads — accepted lead-time skew (ADR-0019 §1b).

# ... (target / neighbors / horizon_hours / lag_hours / feature_groups / label unchanged) ...

training:
  seed:
    source: openmeteo                      # Historical Forecast API deep archive
    start: "2024-01-01"                    # Open-Meteo HRDPS archive start (~2024)
  holdout_months: 12                       # unchanged — full-year seasonal holdout
```
Also update the file's header comment that says "CaSPAr historical seed" → "Open-Meteo seed backfill (ADR-0019)".

- [ ] **Step 2: Write a config-load assertion** (add to `tests/config/test_schema.py`)

```python
def test_lethbridge_uses_openmeteo_for_both_feeds() -> None:
    import microclimate.connectors  # noqa: F401  # populate registry
    from microclimate.config.loader import load_deployment
    from microclimate.connectors.registry import validate_config_sources

    config = load_deployment("lethbridge")
    assert config.nwp.live_connector == "openmeteo"
    assert config.nwp.historical_connector == "openmeteo"
    assert config.training.seed.source == "openmeteo"
    validate_config_sources(config)  # must not raise (openmeteo is a registered NWPSource)
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/config/test_schema.py -v`
Expected: pass (including the new test).

- [ ] **Step 4: Commit**

```bash
git add config/deployments/lethbridge.yml tests/config/test_schema.py
git commit -m "config(lethbridge): source HRDPS from Open-Meteo (both feeds); seed from 2024 (ADR-0019)"
```

---

## Task 8: Make inference stateless (remove the logger)

**Files:**
- Modify: `src/microclimate/pipelines/inference.py`
- Modify: `.github/workflows/inference.yml`
- Test: `tests/pipelines/test_inference.py`

- [ ] **Step 1: Read the inference test, then write/adjust a failing test**

Run: `sed -n '1,80p' tests/pipelines/test_inference.py` to see current usage. Update the test so `run_inference` is called **without** a `store=` argument and assert only that the forecast JSON is written. Target assertion shape:
```python
def test_run_inference_writes_forecast_json(tmp_path, lethbridge_config, fake_nwp, fake_obs):
    out = tmp_path / "forecast.json"
    doc = run_inference(
        lethbridge_config,
        nwp=fake_nwp,
        observations=fake_obs,
        forecast_path=out,
        issue_time=datetime(2026, 6, 2, 6, 0, tzinfo=UTC),
    )
    assert doc is not None and out.exists()
```
(Reuse the existing fixtures/fakes in that test file; just drop the `TrainingStore` fixture and the `store=` kwarg.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_inference.py -v`
Expected: FAIL — `run_inference` still requires `store`.

- [ ] **Step 3: Edit `src/microclimate/pipelines/inference.py`**

- Remove the import `from microclimate.training_store import TrainingStore` and `import os`.
- Change the module docstring's first sentence to: *"Builds the snapshot, produces the raw-HRDPS baseline forecast, and writes the ForecastDocument JSON. Stateless — no snapshot logging (ADR-0019)."*
- Replace `run_inference` with:
```python
def run_inference(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    forecast_path: Path,
    issue_time: datetime,
) -> ForecastDocument:
    """Build a snapshot → baseline forecast → write JSON. Stateless (ADR-0019)."""
    snapshot = build_snapshot(config, issue_time, nwp, observations)
    matrix = build_features(snapshot, config)
    preds = baseline_predictions(matrix, config.label.precip_occurrence_threshold_mm)
    doc = _assemble_forecast(config, preds, snapshot.issue_time, last_updated=snapshot.issue_time)
    write_forecast(doc, forecast_path)
    return doc
```
- In `main()`, delete the `store = TrainingStore(...)` line and the `store=store` kwarg and the `doc is None` branch; update `_ATTRIBUTION`:
```python
_ATTRIBUTION = [
    "Weather data by Open-Meteo.com (CC-BY 4.0)",
    "Data Source: Environment and Climate Change Canada (ECCC)",
]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/pipelines/test_inference.py -v`
Expected: pass.

- [ ] **Step 5: Strip logging from `.github/workflows/inference.yml`**

Remove the "Install eccodes" step, the `cfgrib` import-check step, and the `TRAINING_STORE_ROOT=store` env + any store checkout/force-push of the `training-data` branch. Keep the step that runs `uv run python -m microclimate.pipelines.inference --deployment "$id"` and the forecast-JSON publish.

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/inference.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/pipelines/inference.py .github/workflows/inference.yml tests/pipelines/test_inference.py
git commit -m "refactor(inference): stateless publish-only; drop the logger (ADR-0019)"
```

---

## Task 9: Issue-time generator (pure)

**Files:**
- Create: `src/microclimate/pipelines/backfill.py` (generator only this task)
- Test: `tests/pipelines/test_backfill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/pipelines/test_backfill.py`:
```python
from __future__ import annotations

from datetime import UTC, datetime


def test_hrdps_issue_times_are_6h_cycles_inclusive() -> None:
    from microclimate.pipelines.backfill import hrdps_issue_times

    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    times = hrdps_issue_times(start, end)
    assert times[0] == start
    assert times[-1] == end
    assert all(t.hour in (0, 6, 12, 18) for t in times)
    # 00,06,12,18 on day 1 + 00 on day 2 == 5 runs.
    assert len(times) == 5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_backfill.py -v`
Expected: FAIL — module/function missing.

- [ ] **Step 3: Implement**

Create `src/microclimate/pipelines/backfill.py`:
```python
"""Retrain-time seed backfill: pull deep HRDPS history into the training store (ADR-0019).

Idempotent and additive — re-running coalesces by (issue_time, lead_hour) and never prunes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

_RUN_HOURS: tuple[int, ...] = (0, 6, 12, 18)


def hrdps_issue_times(start: datetime, end: datetime) -> list[datetime]:
    """All HRDPS run init times (00/06/12/18 UTC) in [start, end], ascending."""
    s = start.astimezone(UTC) if start.tzinfo else start.replace(tzinfo=UTC)
    e = end.astimezone(UTC) if end.tzinfo else end.replace(tzinfo=UTC)
    cur = s.replace(minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    while cur <= e:
        if cur.hour in _RUN_HOURS and cur >= s:
            out.append(cur)
        cur += timedelta(hours=6)
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_backfill.py -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/pipelines/backfill.py tests/pipelines/test_backfill.py
git commit -m "feat(backfill): HRDPS issue-time generator"
```

---

## Task 10: Backfill into the training store (idempotent + additive)

**Files:**
- Modify: `src/microclimate/pipelines/backfill.py`
- Test: `tests/pipelines/test_backfill.py`

- [ ] **Step 1: Write the failing test** (fakes for NWP + obs; assert store rows + idempotency)

```python
def test_backfill_populates_store_idempotently(tmp_path) -> None:
    import pandas as pd

    from microclimate.config.loader import load_deployment
    from microclimate.pipelines.backfill import backfill_store, hrdps_issue_times
    from microclimate.training_store.store import TrainingStore

    import microclimate.connectors  # noqa: F401  # registry

    config = load_deployment("lethbridge")
    store = TrainingStore(tmp_path)

    # Minimal fakes: NWP returns a FORECAST_FRAME for any t0; obs returns an empty as-of frame.
    from datetime import timedelta
    from microclimate.contracts.forecast_frame import FORECAST_FRAME

    class FakeNWP:
        is_live = True
        def fetch_forecast(self, issue_time, lat, lon, lead_hours):
            rows = [{
                "issue_time": pd.Timestamp(issue_time), "lead_hour": int(h),
                "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(h)),
                "temp_c": 10.0, "dewpoint_c": 5.0, "surface_pressure_hpa": 900.0,
                "precip_mm": 0.0, "cloud_cover_fraction": 0.5, "solar_radiation_wm2": 100.0,
                "wind_speed_ms": 3.0, "wind_dir_deg": 180.0,
            } for h in lead_hours]
            return FORECAST_FRAME.validate(pd.DataFrame(rows))

    class FakeObs:
        historical_coverage = "deep"
        def fetch_historical(self, station_id, start, end):
            from microclimate.contracts.observation import OBSERVATION_FRAME
            return OBSERVATION_FRAME.validate(pd.DataFrame(columns=OBSERVATION_FRAME.columns.keys()))
        def fetch_live(self, station_id, since):
            return self.fetch_historical(station_id, since, since)

    nwp = FakeNWP()
    obs = {config.target.connector_key: FakeObs()}
    times = hrdps_issue_times(
        load_deployment("lethbridge") and __import__("datetime").datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        __import__("datetime").datetime(2024, 1, 1, 18, 0, tzinfo=UTC),
    )

    n1 = backfill_store(config, nwp=nwp, observations=obs, store=store, issue_times=times)
    snaps_after_1 = store.read_snapshots(config.deployment_id)
    n2 = backfill_store(config, nwp=nwp, observations=obs, store=store, issue_times=times)
    snaps_after_2 = store.read_snapshots(config.deployment_id)

    assert n1 == len(times)            # all four runs written
    assert n2 == 0                     # idempotent: nothing new on re-run
    assert len(snaps_after_2) == len(snaps_after_1)  # additive, no duplicates, no prune
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_backfill.py::test_backfill_populates_store_idempotently -v`
Expected: FAIL — `backfill_store` not defined.

- [ ] **Step 3: Implement `backfill_store`** (append to `backfill.py`)

```python
import time
from collections.abc import Mapping, Sequence

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import ForecastUnavailable, NWPSource, ObservationSource, SourceUnavailable
from microclimate.features.feature_builder import build_features
from microclimate.features.labeler import attach_labels
from microclimate.features.snapshot_builder import build_snapshot


def backfill_store(
    config: DeploymentConfig,
    *,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
    store: object,  # microclimate.training_store.store.TrainingStore
    issue_times: Sequence[datetime],
    pause_s: float = 0.12,  # ~500/min, under the 600/min free limit
) -> int:
    """Build + persist snapshots and labels for each issue_time. Returns count newly written.

    Idempotent: skips issue_times already stored (current schema). Additive: TrainingStore
    coalesces by (issue_time[, lead_hour]) and never prunes.
    """
    written = 0
    fresh: list[datetime] = []
    for t0 in issue_times:
        if store.has_snapshot(config.deployment_id, t0):  # type: ignore[attr-defined]
            continue
        try:
            snapshot = build_snapshot(config, t0, nwp, observations)
        except (ForecastUnavailable, SourceUnavailable):
            # A run absent from the archive — log-and-skip; backfill stays gapless on re-run.
            continue
        store.append_snapshot(snapshot)  # type: ignore[attr-defined]
        fresh.append(t0)
        written += 1
        if pause_s:
            time.sleep(pause_s)

    if fresh:
        matrices = [build_features(build_snapshot(config, t, nwp, observations), config) for t in fresh]
        import pandas as pd

        matrix = pd.concat(matrices, ignore_index=True)
        target = observations[config.target.connector_key]
        start = matrix["valid_time"].min().to_pydatetime()
        end = matrix["valid_time"].max().to_pydatetime()
        target_obs = target.fetch_historical(config.target.station_id, start, end)
        labeled = attach_labels(matrix, target_obs, config.label.precip_occurrence_threshold_mm)
        label_cols = ["issue_time", "lead_hour", "valid_time", "label_temp_c", "label_precip_occurrence"]
        store.write_labels(config.deployment_id, labeled[label_cols])  # type: ignore[attr-defined]
    return written
```
Note: rebuilding snapshots for the label pass keeps this task self-contained; if profiling shows it matters, refactor to cache snapshots from the first loop (out of scope here).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_backfill.py -v`
Expected: pass. (If `attach_labels` requires non-empty target obs, adjust the fake to return one obs row at a covered `valid_time`; read `src/microclimate/features/labeler.py` for the exact requirement.)

- [ ] **Step 5: Type-check + commit**

Run: `uv run pyright src/microclimate/pipelines/backfill.py && uv run ruff check src/microclimate/pipelines/backfill.py`
```bash
git add src/microclimate/pipelines/backfill.py tests/pipelines/test_backfill.py
git commit -m "feat(backfill): idempotent additive store population from Open-Meteo seed"
```

---

## Task 11: Attribution — store notice, DATA_LICENSES, forecast JSON, CI check

**Files:**
- Create: `scripts/training_store_attribution.txt` (the notice text, copied to the branch root by the publish step)
- Modify: `DATA_LICENSES.md`
- Modify: `.github/workflows/ci.yml` (attribution presence check)
- Test: `tests/publication/test_attribution.py`

- [ ] **Step 1: Write the failing test** (forecast JSON carries Open-Meteo + ECCC)

Create `tests/publication/test_attribution.py`:
```python
from __future__ import annotations

from pathlib import Path


def test_attribution_text_mentions_openmeteo_and_eccc() -> None:
    txt = Path("scripts/training_store_attribution.txt").read_text()
    assert "Open-Meteo" in txt and "CC-BY" in txt.replace("CC BY", "CC-BY")
    assert "Environment and Climate Change Canada" in txt


def test_inference_attribution_constant() -> None:
    from microclimate.pipelines.inference import _ATTRIBUTION

    joined = " ".join(_ATTRIBUTION)
    assert "Open-Meteo" in joined and "Environment and Climate Change Canada" in joined
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/publication/test_attribution.py -v`
Expected: FAIL — file/constant missing (the `_ATTRIBUTION` half passes after Task 8; the file half fails).

- [ ] **Step 3: Create the attribution notice + update DATA_LICENSES.md**

Create `scripts/training_store_attribution.txt`:
```
This dataset contains weather data from Open-Meteo.com, licensed under CC-BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Data was modified: normalized,
unit-converted, and resampled to a single target grid cell.

Underlying model: HRDPS (High Resolution Deterministic Prediction System),
© Environment and Climate Change Canada, used under the ECCC open-data licence.
```
In `DATA_LICENSES.md`: add an **Open-Meteo (CC-BY 4.0)** section with the attribution string; mark the **CaSPAr / Mai et al. 2020** section as *"not used — superseded by ADR-0019"* (retain text, like the ACIS section).

- [ ] **Step 4: Add the CI attribution check** — append a step to the lint job in `.github/workflows/ci.yml`

```yaml
      - name: Training-store attribution notice exists
        run: test -f scripts/training_store_attribution.txt
```
(The publish/backfill job copies this file to the `training-data` branch root; the check guarantees it cannot be deleted silently.)

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/publication/test_attribution.py -v`
Expected: pass. `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` → no error.

- [ ] **Step 6: Commit**

```bash
git add scripts/training_store_attribution.txt DATA_LICENSES.md .github/workflows/ci.yml tests/publication/test_attribution.py
git commit -m "feat(attribution): Open-Meteo + ECCC across JSON/store/CI; drop CaSPAr citation (ADR-0019)"
```

---

## Task 12: Update README "Project status"

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit the "Project status" section**

Reflect: HRDPS now sourced via the **Open-Meteo** connector (live `/v1/forecast` + Historical Forecast API seed backfill); native GRIB2 connectors (`hrdps_datamart`, `hrdps_caspar`, `nwp_core`) and `xarray`/`cfgrib` removed; the **inference logger is removed** (inference is stateless); training data comes from the **retrain-time seed backfill**; note the accepted lead-time skew (ADR-0019 §1b). Update any "outstanding stubs" list accordingly.

- [ ] **Step 2: Sanity-check no stale references remain**

Run: `grep -rni "caspar\|datamart\|logger\|cfgrib" README.md`
Expected: only intentional, past-tense/removed mentions (or none).

- [ ] **Step 3: Final full verification**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
Expected: all green (network tests deselected).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): project status — Open-Meteo source, stateless inference, backfill (ADR-0019)"
```

---

## Self-Review

**Spec coverage:** Unit 1 connector → Tasks 2–5; Unit 2 deletions → Task 6; Unit 3 config → Task 7; Unit 4 stateless inference → Task 8; Unit 5 backfill → Tasks 9–10; Unit 6 attribution → Task 11; Unit 7 parity test → Task 5; Unit 8 smoke/fixtures → Task 1; Unit 9 README → Task 12. All covered.

**Open dependencies the executor must resolve by reading code (not placeholders — real edits to existing files):** the exact current contents of `tests/pipelines/test_inference.py` (Task 8), `tests/config/test_schema.py` connector-key literals (Task 6/7), `tests/connectors/conftest.py` (Task 6), and `src/microclimate/features/labeler.py`'s obs requirement (Task 10). Each step says to read the file first and gives the target shape.

**Type consistency:** `_parse_hourly_to_forecast_frame`, `_build_request`, `OpenMeteoSource`, `hrdps_issue_times`, `backfill_store` are referenced with identical signatures across tasks. `_OPENMETEO_VAR_MAP` keys == `PHYSICAL_VARS`. Fetcher signature `fetcher(url, *, params=...)` matches `http_get`.

**Known accepted scope cut:** full training-pipeline/model-training wiring beyond the backfill is deferred (spec "out of scope"); `pipelines/training.py` remains a stub after this plan — the backfill is callable and tested, ready to be wired into the training pipeline in a follow-on.
