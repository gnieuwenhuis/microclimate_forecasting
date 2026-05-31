# features.build_snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `build_snapshot` — the single normalization/IO/as-of boundary that turns the injected NWP + observation connectors into one `FeatureSnapshot` per `issue_time`, holding raw canonicalized values for all 48 lead hours.

**Architecture:** An orchestrator (`build_snapshot`) calls connectors, applies the degradation policy, and assembles four flat feature maps + metadata into a `FeatureSnapshot`. Three pure, network-free helpers do the value-shaping: `_flatten_forecast` (FORECAST_FRAME → `nwp_{var}_h{lead}`), `_align_obs_to_lag_grid` (OBSERVATION_FRAME → fixed hourly lag grid with NaN/mask), `_temporal_features` (t0 → cyclical encodings). Observations are read only via as-of `fetch_historical(..., end=issue_time)`, never `fetch_live`, so the obs path is identical for training and inference — the train/serve skew guarantee, enforced at the snapshot layer.

**Tech Stack:** Python 3.12, pandas, pydantic (FeatureSnapshot), pytest, pyright strict, ruff, import-linter.

**Source PRD:** GitHub issue #12.

---

## Background the engineer needs

**Domain (see `CONTEXT.md`):** A *feature snapshot* is the single canonical model-input object for one prediction, built by the one builder both training and inference use (this is what prevents train/serve skew). *As-of reconstruction* = a snapshot only ever uses observations at or before `t0` (`issue_time`). A *missingness mask* flags whether an obs value was real or absent.

**The 8 physical variables (canonical names, fixed order):**
`temp_c`, `dewpoint_c`, `surface_pressure_hpa`, `precip_mm`, `cloud_cover_fraction`, `solar_radiation_wm2`, `wind_speed_ms`, `wind_dir_deg`.

**`FeatureSnapshot` contract** (`src/microclimate/contracts/snapshot.py`) — pydantic, `extra="forbid"`, fields: `deployment_id: str`, `issue_time: AwareDatetime`, `nwp_features: Mapping[str,float]`, `observation_features: Mapping[str,float]`, `observation_masks: Mapping[str,bool]`, `static_features: Mapping[str,float]`, `temporal_features: Mapping[str,float]`, `lead_hours: tuple[int,...]`, `schema_version: str`. Construction *is* the validation.

**`FORECAST_FRAME`** (`contracts/forecast_frame.py`) — one row per lead hour; columns `issue_time`, `lead_hour` (1–48), `valid_time`, + the 8 vars (all non-null floats). Returned by `NWPSource.fetch_forecast`.

**`OBSERVATION_FRAME`** (`contracts/observation.py`) — columns `station_id`, `timestamp` (`datetime64[ns, UTC]`), + for each of the 8 vars a `<var>` (nullable float) and a `<var>_present` (bool). Returned by `ObservationSource.fetch_historical`. The connector already dedups to ≤1 row per hour and returns an empty schema-valid frame when a valid station has no data in the window.

**Connector interfaces** (`connectors/base.py`):
- `NWPSource.fetch_forecast(issue_time, lat, lon, lead_hours) -> DataFrame`
- `ObservationSource.fetch_historical(station_id, start, end) -> DataFrame`, `.fetch_live(station_id, since) -> DataFrame`, `.historical_coverage -> "deep"|"shallow"|"none"`
- Exceptions: `SourceUnavailable`, `ForecastUnavailable`, `StationNotFound` (all subclass `ConnectorError`).

**Config** (`config/schema.py`): `DeploymentConfig` has `deployment_id`, `target: StationRef`, `neighbors: list[StationRef]`, `horizon_hours` (default 48), `lag_hours` (>=0), `feature_groups: FeatureGroupSwitches` (`.nwp`, `.observations` bools). `StationRef` has `station_id`, `connector_key`, `lat`, `lon`, `elevation_m: float|None`.

**Layering** (`.importlinter`): `features` may import `contracts`, `config`, `connectors` (all lower) — the imports in this plan are legal. Nothing higher.

**pyright-strict / ruff notes:**
- pandas indexing/`.iloc`/`.iterrows()` return `Unknown`/`Any`; add `# type: ignore[reportUnknownMemberType]` (and `reportUnknownArgumentType` / `reportUnknownVariableType` where flagged) on the offending lines, matching the style in `connectors/nwp_core.py`.
- Importing private helpers (`_align_obs_to_lag_grid` etc.) into tests trips ruff `PLC2701`; suppress with `# noqa: PLC2701` and pyright `# type: ignore[reportPrivateUsage]`, exactly as `tests/connectors/test_hrdps_caspar.py` does.
- After every code change, the gate is: `uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run pyright && uv run pytest -q`.

**Prior art for tests:** `tests/connectors/test_hrdps_caspar.py` (hermetic injectable-seam pattern, pinned-value assertions, the `pd.testing.assert_frame_equal` no-skew test), `tests/connectors/conftest.py` (`build_hrdps_dataset` synthetic-frame helper).

**File map:**

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/microclimate/contracts/snapshot.py` | Add `SNAPSHOT_SCHEMA_VERSION` constant |
| Modify | `src/microclimate/features/snapshot_builder.py` | Helpers + `build_snapshot` implementation |
| Create | `tests/features/conftest.py` | Fake connectors + synthetic-frame + `make_config` helpers |
| Modify | `tests/features/test_snapshot_builder.py` | Replace stub test; add end-to-end + skew tests |
| Create | `tests/features/test_snapshot_helpers.py` | Unit tests for the 3 pure helpers |
| Create | `docs/adr/0011-snapshot-normalization-boundary.md` | ADR for the boundary + as-of skew decision |
| Modify | `CONTEXT.md` | Snapshot feature-key conventions + lag-grid definition |

---

## Task 1: Add `SNAPSHOT_SCHEMA_VERSION` constant

**Files:**
- Modify: `src/microclimate/contracts/snapshot.py`
- Test: `tests/contracts/test_snapshot.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/contracts/test_snapshot.py`:

```python
def test_snapshot_schema_version_constant() -> None:
    from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION

    assert isinstance(SNAPSHOT_SCHEMA_VERSION, str)
    assert SNAPSHOT_SCHEMA_VERSION  # non-empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_snapshot.py::test_snapshot_schema_version_constant -v`
Expected: FAIL with `ImportError: cannot import name 'SNAPSHOT_SCHEMA_VERSION'`.

- [ ] **Step 3: Add the constant**

In `src/microclimate/contracts/snapshot.py`, after the imports and before `class FeatureSnapshot`:

```python
# Single source of truth for the snapshot feature contract version. Bumped when the
# set/meaning of feature keys changes. The training store stamps the same value.
SNAPSHOT_SCHEMA_VERSION = "1.0.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_snapshot.py::test_snapshot_schema_version_constant -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/snapshot.py tests/contracts/test_snapshot.py
git commit -m "feat(contracts): add SNAPSHOT_SCHEMA_VERSION constant"
```

---

## Task 2: `_temporal_features` helper

**Files:**
- Modify: `src/microclimate/features/snapshot_builder.py`
- Test: `tests/features/test_snapshot_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/features/test_snapshot_helpers.py`:

```python
"""Unit tests for the pure helpers behind build_snapshot."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from microclimate.features.snapshot_builder import (
    _temporal_features,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)


def test_temporal_features_keys_and_values() -> None:
    # 2026-01-01 06:00 UTC → day-of-year 1, hour 6.
    t0 = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
    feats = _temporal_features(t0)

    assert set(feats) == {"t0_hour_sin", "t0_hour_cos", "t0_doy_sin", "t0_doy_cos"}
    assert feats["t0_hour_sin"] == math.sin(2 * math.pi * 6 / 24.0)
    assert feats["t0_hour_cos"] == math.cos(2 * math.pi * 6 / 24.0)
    assert feats["t0_doy_sin"] == math.sin(2 * math.pi * 1 / 365.25)
    assert feats["t0_doy_cos"] == math.cos(2 * math.pi * 1 / 365.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_snapshot_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name '_temporal_features'`.

- [ ] **Step 3: Write the implementation**

Replace the entire content of `src/microclimate/features/snapshot_builder.py` with the imports + module constants + this helper (`build_snapshot` stays stubbed for now — later tasks fill it in):

```python
"""The single, only path that produces a FeatureSnapshot (L3).

As-of / no-leakage: this is the only entry point, it takes issue_time, and the only obs
access is bounded to timestamp <= issue_time. There is no parameter for future data.

build_snapshot is the normalization/IO/as-of boundary: it holds raw, canonicalized values
only. Derived features (dewpoint depression, tendency, advection, per-lead-hour encodings)
and the explode-to-per-lead-hour-rows transform are downstream pure functions of the snapshot
(ADR-0011). Observations are read only via as-of fetch_historical, never fetch_live, so the
obs path is identical for training and inference (the skew guarantee).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.config.schema import DeploymentConfig, StationRef
from microclimate.connectors.base import NWPSource, ObservationSource, SourceUnavailable
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot

# Canonical physical variables, fixed order. Match FORECAST_FRAME / OBSERVATION_FRAME.
_PHYSICAL_VARS: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


def _temporal_features(issue_time: datetime) -> dict[str, float]:
    """Cyclical encodings of t0 only (hour-of-day period 24, day-of-year period 365.25).

    Per-lead-hour temporal encodings are built downstream, not here.
    """
    hour = issue_time.hour + issue_time.minute / 60.0
    doy = issue_time.timetuple().tm_yday
    return {
        "t0_hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "t0_hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "t0_doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "t0_doy_cos": math.cos(2 * math.pi * doy / 365.25),
    }


def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_snapshot_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_helpers.py
git commit -m "feat(features): _temporal_features cyclical t0 encodings"
```

---

## Task 3: `_flatten_forecast` helper

**Files:**
- Modify: `src/microclimate/features/snapshot_builder.py`
- Test: `tests/features/test_snapshot_helpers.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/features/test_snapshot_helpers.py` (add the import alongside the existing one):

```python
import pandas as pd

from microclimate.features.snapshot_builder import (
    _flatten_forecast,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)

_PHYS = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


def _forecast_frame(lead_hours: list[int]) -> pd.DataFrame:
    """Minimal FORECAST_FRAME-shaped frame; var value encodes (var index + lead)."""
    rows: list[dict[str, object]] = []
    for lh in lead_hours:
        row: dict[str, object] = {"lead_hour": int(lh)}
        for i, var in enumerate(_PHYS):
            row[var] = float(i) + float(lh)
        rows.append(row)
    return pd.DataFrame(rows)


def test_flatten_forecast_cardinality_and_keys() -> None:
    frame = _forecast_frame([1, 2, 3])
    flat = _flatten_forecast(frame)

    assert len(flat) == 8 * 3  # 8 vars x 3 leads
    assert flat["nwp_temp_c_h1"] == 0.0 + 1.0
    assert flat["nwp_temp_c_h3"] == 0.0 + 3.0
    assert flat["nwp_wind_dir_deg_h2"] == 7.0 + 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_snapshot_helpers.py::test_flatten_forecast_cardinality_and_keys -v`
Expected: FAIL with `ImportError: cannot import name '_flatten_forecast'`.

- [ ] **Step 3: Write the implementation**

In `src/microclimate/features/snapshot_builder.py`, add this function below `_temporal_features`:

```python
def _flatten_forecast(frame: pd.DataFrame) -> dict[str, float]:
    """FORECAST_FRAME (one row per lead hour) → {nwp_{var}_h{lead}: value}.

    Target-cell forecast values only; no masks (NWP is complete-or-fail).
    """
    out: dict[str, float] = {}
    for _, row in frame.iterrows():  # type: ignore[reportUnknownVariableType]
        lead = int(row["lead_hour"])  # type: ignore[reportUnknownArgumentType]
        for var in _PHYSICAL_VARS:
            out[f"nwp_{var}_h{lead}"] = float(row[var])  # type: ignore[reportUnknownArgumentType]
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_snapshot_helpers.py::test_flatten_forecast_cardinality_and_keys -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_helpers.py
git commit -m "feat(features): _flatten_forecast → nwp_{var}_h{lead}"
```

---

## Task 4: `_align_obs_to_lag_grid` helper (the deep core)

**Files:**
- Modify: `src/microclimate/features/snapshot_builder.py`
- Test: `tests/features/test_snapshot_helpers.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_snapshot_helpers.py`:

```python
from datetime import timedelta

from microclimate.features.snapshot_builder import (
    _align_obs_to_lag_grid,  # noqa: PLC2701  # type: ignore[reportPrivateUsage]
)

_OBS_VALUES: dict[str, float] = {
    "temp_c": 15.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.5,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 300.0,
    "wind_speed_ms": 5.0,
    "wind_dir_deg": 270.0,
}


def _obs_frame(
    station_id: str,
    timestamps: list[datetime],
    *,
    absent: set[tuple[int, str]] = frozenset(),  # type: ignore[assignment]
) -> pd.DataFrame:
    """OBSERVATION_FRAME-shaped frame. `absent` = {(row_index, var)} → value NaN, present False."""
    data: dict[str, list[object]] = {
        "station_id": [station_id] * len(timestamps),
        "timestamp": list(pd.to_datetime(timestamps, utc=True)),
    }
    for var in _PHYS:
        col_val: list[object] = []
        col_present: list[object] = []
        for idx in range(len(timestamps)):
            if (idx, var) in absent:
                col_val.append(float("nan"))
                col_present.append(False)
            else:
                col_val.append(_OBS_VALUES[var])
                col_present.append(True)
        data[var] = col_val
        data[f"{var}_present"] = col_present
    return pd.DataFrame(data)


_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)


def test_align_present_values_land_in_correct_lag() -> None:
    # rows at t0, t0-1h, t0-2h all present.
    ts = [_T0, _T0 - timedelta(hours=1), _T0 - timedelta(hours=2)]
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", ts), "S1", _T0, lag_hours=2)

    assert len(feats) == 8 * 3 and len(masks) == 8 * 3  # 8 vars x 3 lags (0,1,2)
    assert set(feats) == set(masks)  # mirrored keys
    assert feats["obs_S1_temp_c_lag0"] == 15.0
    assert masks["obs_S1_temp_c_lag0"] is True
    assert feats["obs_S1_surface_pressure_hpa_lag2"] == 900.0


def test_align_missing_hour_is_nan_and_masked() -> None:
    # Only t0 present; lag1 and lag2 hours have no row.
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", [_T0]), "S1", _T0, lag_hours=2)

    assert masks["obs_S1_temp_c_lag0"] is True
    assert masks["obs_S1_temp_c_lag1"] is False
    assert math.isnan(feats["obs_S1_temp_c_lag1"])
    assert masks["obs_S1_temp_c_lag2"] is False
    assert math.isnan(feats["obs_S1_temp_c_lag2"])


def test_align_present_false_is_nan_and_masked() -> None:
    # Row exists at t0 but pressure sensor reports absent (present=False).
    ts = [_T0]
    feats, masks = _align_obs_to_lag_grid(
        _obs_frame("S1", ts, absent={(0, "surface_pressure_hpa")}), "S1", _T0, lag_hours=0
    )

    assert masks["obs_S1_temp_c_lag0"] is True  # other sensors unaffected
    assert masks["obs_S1_surface_pressure_hpa_lag0"] is False
    assert math.isnan(feats["obs_S1_surface_pressure_hpa_lag0"])


def test_align_filters_rows_after_t0() -> None:
    # A row at t0+1h must never enter the grid (defensive as-of filter).
    ts = [_T0 + timedelta(hours=1), _T0]
    feats, masks = _align_obs_to_lag_grid(_obs_frame("S1", ts), "S1", _T0, lag_hours=0)

    assert masks["obs_S1_temp_c_lag0"] is True
    assert feats["obs_S1_temp_c_lag0"] == 15.0  # the t0 row, not t0+1h


def test_align_none_frame_all_absent() -> None:
    feats, masks = _align_obs_to_lag_grid(None, "S1", _T0, lag_hours=2)

    assert len(feats) == 8 * 3
    assert all(m is False for m in masks.values())
    assert all(math.isnan(v) for v in feats.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_snapshot_helpers.py -v -k align`
Expected: FAIL with `ImportError: cannot import name '_align_obs_to_lag_grid'`.

- [ ] **Step 3: Write the implementation**

In `src/microclimate/features/snapshot_builder.py`, add below `_flatten_forecast`:

```python
def _align_obs_to_lag_grid(
    frame: pd.DataFrame | None,
    station_id: str,
    issue_time: datetime,
    lag_hours: int,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Align one station's OBSERVATION_FRAME onto the fixed hourly lag grid.

    Grid: lag0 = issue_time, lag1 = issue_time-1h, … lag{lag_hours}. Rows are matched by
    exact UTC-hour equality. A slot is absent (value NaN, mask False) when no row exists at
    that hour, the value is null, or the row's <var>_present is False. A None/empty frame
    (degraded source) yields an all-absent grid. issue_time is NOT floored: an off-hour t0
    simply matches no rows.

    Defensive as-of filter: rows with timestamp > issue_time are dropped before matching.
    """
    cutoff = pd.Timestamp(issue_time)
    row_by_ts: dict[pd.Timestamp, int] = {}
    in_window: pd.DataFrame | None = None
    if frame is not None and len(frame) > 0:
        in_window = frame[frame["timestamp"] <= cutoff].reset_index(drop=True)  # type: ignore[reportUnknownMemberType]
        for i, ts in enumerate(in_window["timestamp"]):  # type: ignore[reportUnknownVariableType]
            row_by_ts[pd.Timestamp(ts)] = i  # type: ignore[reportUnknownArgumentType]

    features: dict[str, float] = {}
    masks: dict[str, bool] = {}
    for k in range(lag_hours + 1):
        slot_ts = cutoff - pd.Timedelta(hours=k)
        row_idx = row_by_ts.get(slot_ts)
        for var in _PHYSICAL_VARS:
            key = f"obs_{station_id}_{var}_lag{k}"
            value = float("nan")
            present = False
            if row_idx is not None and in_window is not None:
                is_present = bool(in_window[f"{var}_present"].iloc[row_idx])  # type: ignore[reportUnknownArgumentType]
                raw = in_window[var].iloc[row_idx]  # type: ignore[reportUnknownVariableType]
                if is_present and pd.notna(raw):  # type: ignore[reportUnknownArgumentType]
                    value = float(raw)  # type: ignore[reportUnknownArgumentType]
                    present = True
            features[key] = value
            masks[key] = present
    return features, masks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_snapshot_helpers.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the partial gate**

Run: `uv run ruff check src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_helpers.py && uv run pyright src/microclimate/features/snapshot_builder.py`
Expected: 0 errors. If pyright flags an unannotated pandas access, add the matching `# type: ignore[...]` on that line.

- [ ] **Step 6: Commit**

```bash
git add src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_helpers.py
git commit -m "feat(features): _align_obs_to_lag_grid (fixed hourly lag grid + masks)"
```

---

## Task 5: Test infrastructure — fakes + frame builders + `make_config`

**Files:**
- Create: `tests/features/conftest.py`

This task adds no test of its own; it provides the hermetic fixtures the next tasks consume.

- [ ] **Step 1: Create `tests/features/conftest.py`**

```python
"""Hermetic fixtures for build_snapshot tests: fake connectors + synthetic frames."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from microclimate.config.schema import (
    DeploymentConfig,
    FeatureGroupSwitches,
    LabelConfig,
    NwpConfig,
    OutputConfig,
    SeedConfig,
    StationRef,
    TrainingConfig,
)
from microclimate.connectors.base import (
    HistoricalCoverage,
    NWPSource,
    ObservationSource,
)

PHYS = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)

# Physically plausible pinned values, reused for both NWP and obs frames.
PINNED: dict[str, float] = {
    "temp_c": 15.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.5,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 300.0,
    "wind_speed_ms": 5.0,
    "wind_dir_deg": 270.0,
}


def make_forecast_frame(issue_time: datetime, lead_hours: Sequence[int]) -> pd.DataFrame:
    """FORECAST_FRAME-shaped frame with PINNED values at every lead."""
    rows: list[dict[str, object]] = []
    for lh in lead_hours:
        row: dict[str, object] = {
            "issue_time": pd.Timestamp(issue_time),
            "lead_hour": int(lh),
            "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(lh)),
        }
        for var in PHYS:
            row[var] = PINNED[var]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df


def make_obs_frame(
    station_id: str,
    timestamps: Sequence[datetime],
    *,
    absent: set[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """OBSERVATION_FRAME-shaped frame; `absent` marks (row_index, var) NaN/present=False."""
    absent = absent or set()
    data: dict[str, list[object]] = {
        "station_id": [station_id] * len(timestamps),
        "timestamp": list(pd.to_datetime(list(timestamps), utc=True)),
    }
    for var in PHYS:
        vals: list[object] = []
        pres: list[object] = []
        for idx in range(len(timestamps)):
            if (idx, var) in absent:
                vals.append(float("nan"))
                pres.append(False)
            else:
                vals.append(PINNED[var])
                pres.append(True)
        data[var] = vals
        data[f"{var}_present"] = pres
    return pd.DataFrame(data)


class FakeNWP(NWPSource):
    """Injectable NWPSource returning a prebuilt FORECAST_FRAME or raising `exc`."""

    def __init__(self, frame: pd.DataFrame | None = None, exc: Exception | None = None) -> None:
        self._frame = frame
        self._exc = exc

    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        if self._exc is not None:
            raise self._exc
        assert self._frame is not None
        return self._frame


class FakeObs(ObservationSource):
    """Injectable ObservationSource. Returns prebuilt frames keyed by station_id, or raises."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._frames = frames or {}
        self._exc = exc

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        if self._exc is not None:
            raise self._exc
        return self._frames[station_id]

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError("build_snapshot must never call fetch_live")


def make_config(
    *,
    horizon_hours: int = 3,
    lag_hours: int = 2,
    nwp: bool = True,
    observations: bool = True,
    neighbors: list[StationRef] | None = None,
    connector_key: str = "fake",
) -> DeploymentConfig:
    """Minimal valid DeploymentConfig: 1 target + (default) 1 neighbor, both `connector_key`."""
    if neighbors is None:
        neighbors = [
            StationRef(
                station_id="N1", connector_key=connector_key, lat=51.5, lon=-113.5, elevation_m=950.0
            )
        ]
    return DeploymentConfig(
        deployment_id="test",
        target=StationRef(
            station_id="T1", connector_key=connector_key, lat=51.0, lon=-114.0, elevation_m=900.0
        ),
        neighbors=neighbors,
        enabled_sources=[connector_key],
        nwp=NwpConfig(
            product="hrdps",
            live_connector="x",
            historical_connector="y",
            sampling="nearest_grid_cell",
        ),
        horizon_hours=horizon_hours,
        lag_hours=lag_hours,
        feature_groups=FeatureGroupSwitches(nwp=nwp, observations=observations),
        label=LabelConfig(precip_occurrence_threshold_mm=0.2),
        training=TrainingConfig(
            seed=SeedConfig(source="caspar", start="2017-05-22"), holdout_months=12
        ),
        output=OutputConfig(forecast_json="x.json"),
    )


__all__ = [
    "FakeNWP",
    "FakeObs",
    "PHYS",
    "PINNED",
    "make_config",
    "make_forecast_frame",
    "make_obs_frame",
]
```

- [ ] **Step 2: Sanity-check it imports**

Run: `uv run python -c "import tests.features.conftest as c; print(sorted(c.__all__))"`
Expected: prints the names without error.

- [ ] **Step 3: Commit**

```bash
git add tests/features/conftest.py
git commit -m "test(features): hermetic fixtures (fake connectors, synthetic frames, make_config)"
```

---

## Task 6: `build_snapshot` happy path

**Files:**
- Modify: `src/microclimate/features/snapshot_builder.py`
- Modify: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Replace the stub test file**

Replace the entire content of `tests/features/test_snapshot_builder.py` with:

```python
"""End-to-end hermetic tests for build_snapshot (injected fake connectors)."""

from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime, timedelta

from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot
from microclimate.features.snapshot_builder import build_snapshot

from .conftest import FakeNWP, FakeObs, PINNED, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LEADS = [1, 2, 3]


def _obs_source_all_present() -> FakeObs:
    """A FakeObs with a dense window (t0, t0-1h, t0-2h) for T1 and N1."""
    ts = [_T0, _T0 - timedelta(hours=1), _T0 - timedelta(hours=2)]
    return FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})


def test_signature_takes_issue_time() -> None:
    params = inspect.signature(build_snapshot).parameters
    assert "issue_time" in params  # leakage-proof by signature


def test_happy_path_returns_feature_snapshot() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))
    snap = build_snapshot(config, _T0, nwp, {"fake": _obs_source_all_present()})

    assert isinstance(snap, FeatureSnapshot)
    assert snap.deployment_id == "test"
    assert snap.issue_time == _T0
    assert snap.lead_hours == (1, 2, 3)
    assert snap.schema_version == SNAPSHOT_SCHEMA_VERSION


def test_happy_path_cardinality() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)  # 2 stations, 3 lags
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))
    snap = build_snapshot(config, _T0, nwp, {"fake": _obs_source_all_present()})

    assert len(snap.nwp_features) == 8 * 3  # 24
    assert len(snap.observation_features) == 2 * 8 * 3  # 48
    assert len(snap.observation_masks) == 2 * 8 * 3  # 48
    assert set(snap.observation_features) == set(snap.observation_masks)
    assert len(snap.static_features) == 3
    assert len(snap.temporal_features) == 4


def test_happy_path_pinned_values() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))
    snap = build_snapshot(config, _T0, nwp, {"fake": _obs_source_all_present()})

    assert snap.nwp_features["nwp_temp_c_h1"] == PINNED["temp_c"]
    assert snap.nwp_features["nwp_surface_pressure_hpa_h3"] == PINNED["surface_pressure_hpa"]
    assert snap.observation_features["obs_T1_temp_c_lag0"] == PINNED["temp_c"]
    assert snap.observation_masks["obs_N1_wind_dir_deg_lag2"] is True
    assert snap.static_features["static_lat"] == 51.0
    assert snap.static_features["static_lon"] == -114.0
    assert snap.static_features["static_elevation_m"] == 900.0


def test_static_elevation_nan_when_missing() -> None:
    from microclimate.config.schema import StationRef

    config = make_config(horizon_hours=1, lag_hours=0)
    # Override target elevation to None via a fresh config.
    config = config.model_copy(
        update={
            "target": StationRef(
                station_id="T1", connector_key="fake", lat=51.0, lon=-114.0, elevation_m=None
            )
        }
    )
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
    obs = FakeObs(frames={"T1": make_obs_frame("T1", [_T0]), "N1": make_obs_frame("N1", [_T0])})
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert math.isnan(snap.static_features["static_elevation_m"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v`
Expected: FAIL — `build_snapshot` still raises `NotImplementedError`.

- [ ] **Step 3: Implement `build_snapshot`**

In `src/microclimate/features/snapshot_builder.py`, replace the `build_snapshot` stub body with the full implementation:

```python
def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    """Build the one FeatureSnapshot for `issue_time` (see module docstring / ADR-0011)."""
    issue_utc = issue_time if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)
    lead_hours = tuple(range(1, config.horizon_hours + 1))

    # --- NWP (target cell only) — hard fail on connector errors (they propagate). ---
    nwp_features: dict[str, float] = {}
    if config.feature_groups.nwp:
        frame = nwp.fetch_forecast(issue_utc, config.target.lat, config.target.lon, lead_hours)
        nwp_features = _flatten_forecast(frame)

    # --- Observations — degrade per station; StationNotFound propagates. ---
    obs_features: dict[str, float] = {}
    obs_masks: dict[str, bool] = {}
    if config.feature_groups.observations:
        start = issue_utc - timedelta(hours=config.lag_hours)
        refs: list[StationRef] = [config.target, *config.neighbors]
        for ref in refs:
            source = observations[ref.connector_key]
            station_frame: pd.DataFrame | None
            try:
                station_frame = source.fetch_historical(ref.station_id, start, issue_utc)
            except SourceUnavailable:
                # Transient infra failure → degrade this station to all-absent.
                station_frame = None
            feats, masks = _align_obs_to_lag_grid(
                station_frame, ref.station_id, issue_utc, config.lag_hours
            )
            obs_features.update(feats)
            obs_masks.update(masks)

    # --- Static (target only) — NaN elevation when unknown. ---
    elevation = config.target.elevation_m
    static_features: dict[str, float] = {
        "static_lat": float(config.target.lat),
        "static_lon": float(config.target.lon),
        "static_elevation_m": float(elevation) if elevation is not None else float("nan"),
    }

    return FeatureSnapshot(
        deployment_id=config.deployment_id,
        issue_time=issue_utc,
        nwp_features=nwp_features,
        observation_features=obs_features,
        observation_masks=obs_masks,
        static_features=static_features,
        temporal_features=_temporal_features(issue_utc),
        lead_hours=lead_hours,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_builder.py
git commit -m "feat(features): build_snapshot assembles FeatureSnapshot (happy path)"
```

---

## Task 7: As-of boundary + observation degradation

**Files:**
- Modify: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_snapshot_builder.py` (add imports at the top of the file: `import pytest` and `from microclimate.connectors.base import SourceUnavailable`):

```python
def test_as_of_filters_future_obs() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
    # Frame holds a future row (t0+1h) and the t0 row; the future row must be ignored.
    ts = [_T0 + timedelta(hours=1), _T0]
    obs = FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert snap.observation_features["obs_T1_temp_c_lag0"] == PINNED["temp_c"]
    assert snap.observation_masks["obs_T1_temp_c_lag0"] is True


def test_source_unavailable_degrades_only_that_network() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
    obs = FakeObs(exc=SourceUnavailable("network down"))
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    # Both stations on that source degrade to all-absent; NWP still present.
    assert len(snap.observation_features) == 2 * 8 * 1
    assert all(m is False for m in snap.observation_masks.values())
    assert all(math.isnan(v) for v in snap.observation_features.values())
    assert len(snap.nwp_features) == 8 * 1


def test_empty_frame_degrades_to_masked() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
    # Empty (no rows) but schema-valid frame — valid station, no data in window.
    empty = make_obs_frame("T1", [])
    obs = FakeObs(frames={"T1": empty, "N1": make_obs_frame("N1", [])})
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert all(m is False for m in snap.observation_masks.values())


def test_all_obs_fail_still_emits_nwp_only_snapshot() -> None:
    config = make_config(horizon_hours=2, lag_hours=1)
    nwp = FakeNWP(make_forecast_frame(_T0, [1, 2]))
    obs = FakeObs(exc=SourceUnavailable("down"))
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert len(snap.nwp_features) == 8 * 2
    assert all(m is False for m in snap.observation_masks.values())
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v -k "as_of or degrade or empty or all_obs"`
Expected: PASS (the degradation paths are already implemented in Task 6). If any fail, fix `build_snapshot` until green — do not change the tests.

- [ ] **Step 3: Commit**

```bash
git add tests/features/test_snapshot_builder.py
git commit -m "test(features): as-of boundary + obs degradation contract"
```

---

## Task 8: Hard-fail contract (NWP unavailable, StationNotFound)

**Files:**
- Modify: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_snapshot_builder.py` (add `from microclimate.connectors.base import ForecastUnavailable, StationNotFound` to the imports):

```python
def test_nwp_forecast_unavailable_propagates() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(exc=ForecastUnavailable("no run for issue_time"))
    obs = FakeObs(frames={"T1": make_obs_frame("T1", [_T0]), "N1": make_obs_frame("N1", [_T0])})
    with pytest.raises(ForecastUnavailable):
        build_snapshot(config, _T0, nwp, {"fake": obs})


def test_nwp_source_unavailable_propagates() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(exc=SourceUnavailable("datamart down"))
    obs = FakeObs(frames={"T1": make_obs_frame("T1", [_T0]), "N1": make_obs_frame("N1", [_T0])})
    with pytest.raises(SourceUnavailable):
        build_snapshot(config, _T0, nwp, {"fake": obs})


def test_station_not_found_propagates() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
    obs = FakeObs(exc=StationNotFound("bad station id"))
    with pytest.raises(StationNotFound):
        build_snapshot(config, _T0, nwp, {"fake": obs})
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v -k "propagates"`
Expected: `test_nwp_*` PASS immediately. `test_station_not_found_propagates` PASS because `build_snapshot` only catches `SourceUnavailable`, so `StationNotFound` bubbles. If it does not pass, confirm the `except SourceUnavailable` clause in `build_snapshot` is not over-broad (must NOT catch `ConnectorError` or `StationNotFound`).

- [ ] **Step 3: Commit**

```bash
git add tests/features/test_snapshot_builder.py
git commit -m "test(features): hard-fail contract (NWP unavailable, StationNotFound propagate)"
```

---

## Task 9: `feature_groups` switches

**Files:**
- Modify: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_snapshot_builder.py`:

```python
def test_observations_switch_off_empties_obs_maps() -> None:
    config = make_config(horizon_hours=2, lag_hours=1, observations=False)
    nwp = FakeNWP(make_forecast_frame(_T0, [1, 2]))
    # Obs source would raise if called — proves the builder skips it entirely.
    obs = FakeObs(exc=SourceUnavailable("should not be called"))
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert snap.observation_features == {}
    assert snap.observation_masks == {}
    assert len(snap.nwp_features) == 8 * 2
    assert len(snap.temporal_features) == 4  # always populated
    assert len(snap.static_features) == 3


def test_nwp_switch_off_empties_nwp_but_keeps_lead_hours() -> None:
    config = make_config(horizon_hours=3, lag_hours=0, nwp=False)
    # NWP source would raise if called.
    nwp = FakeNWP(exc=SourceUnavailable("should not be called"))
    obs = FakeObs(frames={"T1": make_obs_frame("T1", [_T0]), "N1": make_obs_frame("N1", [_T0])})
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert snap.nwp_features == {}
    assert snap.lead_hours == (1, 2, 3)  # still the full horizon
    assert len(snap.observation_features) == 2 * 8 * 1
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v -k "switch"`
Expected: PASS (switch behavior implemented in Task 6).

- [ ] **Step 3: Commit**

```bash
git add tests/features/test_snapshot_builder.py
git commit -m "test(features): feature_groups switches empty the right maps"
```

---

## Task 10: Skew test (the headline guarantee)

**Files:**
- Modify: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/features/test_snapshot_builder.py`:

```python
def test_no_train_serve_skew() -> None:
    """A 'live' and a 'historical' NWP source returning identical FORECAST_FRAMEs, with
    identical obs, must produce identical FeatureSnapshots — the skew guarantee at the
    snapshot layer. (Fully-present obs so there are no NaNs to defeat == equality.)"""
    config = make_config(horizon_hours=3, lag_hours=2)
    frame = make_forecast_frame(_T0, _LEADS)

    nwp_live = FakeNWP(frame)
    nwp_hist = FakeNWP(frame)
    obs_live = _obs_source_all_present()
    obs_hist = _obs_source_all_present()

    snap_live = build_snapshot(config, _T0, nwp_live, {"fake": obs_live})
    snap_hist = build_snapshot(config, _T0, nwp_hist, {"fake": obs_hist})

    assert snap_live == snap_hist
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/features/test_snapshot_builder.py::test_no_train_serve_skew -v`
Expected: PASS. (If it fails on NaN inequality, confirm the obs frames are fully present — `_obs_source_all_present` uses dense windows with no `absent` entries.)

- [ ] **Step 3: Commit**

```bash
git add tests/features/test_snapshot_builder.py
git commit -m "test(features): no train/serve skew at the snapshot layer"
```

---

## Task 11: Document the decisions — ADR-0011 + CONTEXT.md

**Files:**
- Create: `docs/adr/0011-snapshot-normalization-boundary.md`
- Modify: `CONTEXT.md`

- [ ] **Step 1: Check the ADR format**

Run: `ls docs/adr/ && sed -n '1,12p' docs/adr/0010-*.md`
Expected: confirms the heading/status/date format to mirror.

- [ ] **Step 2: Write ADR-0011**

Create `docs/adr/0011-snapshot-normalization-boundary.md` (match the header style observed in Step 1):

```markdown
# 11. build_snapshot is the normalization boundary; the as-of obs read is the skew guarantee

- **Status:** Accepted
- **Date:** 2026-05-30
- **Informed by:** PRD issue #12; the grilling session that produced it.
- **Relates to:** ADR-0007 ("one HRDPS spec"), the FeatureSnapshot contract.

## Context

`features.build_snapshot` is the single producer of `FeatureSnapshot`, used by both the
training and inference pipelines. Two design questions had to be settled before
implementation: (1) how much feature engineering it performs, and (2) how it reads
observations for past vs present `issue_time` without introducing train/serve skew or
label leakage.

## Decision

**1. build_snapshot is the normalization / IO / as-of boundary — not the feature-engineering
step.** It holds raw, canonicalized values only: NWP forecast values flattened per lead hour
(`nwp_{var}_h{lead}`), lag-windowed observations on a fixed hourly grid
(`obs_{station_id}_{var}_lag{k}`) each with a presence mask, target static values, and `t0`
cyclical encodings. Derived features (dewpoint depression, pressure tendency, advection,
per-lead-hour encodings) and the explode-to-per-lead-hour-rows transform are downstream pure
functions of the snapshot. This keeps the network-touching, skew-critical code small and
hermetically testable, and lets new derived features be added without touching connectors.

**2. The shared builder reads observations only via as-of `fetch_historical(start, end=issue_time)`
— never `fetch_live`.** Because `fetch_historical` guarantees no rows after `end`, the obs path
is byte-identical whether `issue_time` is years in the past (training) or the current hour
(inference). `fetch_live` is `now`-bounded and would leak future rows for any past `issue_time`,
so it is categorically unsafe for the shared builder; it remains in the `ObservationSource`
contract for other callers. A defensive `timestamp <= issue_time` filter backs the guarantee.
The NWP live-vs-historical choice stays the caller's job, made by injecting the right
`NWPSource`.

## Consequences

- Train/serve skew is eliminated by construction at a second point (after ADR-0007's "one HRDPS
  spec"): one obs code path, one normalization function.
- Degradation is a deliberate decision here, not in connectors: a missing NWP backbone hard-fails
  (`ForecastUnavailable`/`SourceUnavailable` propagate); a transient obs `SourceUnavailable` or an
  empty window degrades that station to NaN+masks; a `StationNotFound` hard-fails (loud config
  bug); when every obs source fails, an NWP-only snapshot is still emitted.
- A downstream "build features from the snapshot" step is now required (separate work item) to
  produce the per-lead-hour model-input rows and derived features.
```

- [ ] **Step 3: Update CONTEXT.md**

In `CONTEXT.md`, under the **Data contract** section, extend the **Feature snapshot** bullet by appending these sentences to it (do not delete the existing text):

```markdown
  The snapshot is built by `features.build_snapshot`, which is the **normalization / as-of
  boundary**: it stores *raw canonicalized values only* — one object per **issue time** spanning
  all lead hours — and never `fetch_live` (only as-of `fetch_historical` bounded to `t0`), which
  is the train/serve skew guarantee (ADR-0011). Derived features (dewpoint depression, pressure
  tendency, advection, per-lead-hour encodings) and the explode-to-per-lead-hour rows are
  *downstream* pure functions of the snapshot. **Feature-key conventions:** NWP →
  `nwp_{var}_h{lead}` (8 variables × leads `1…horizon_hours`, target cell only); observations →
  `obs_{station_id}_{var}_lag{k}` on a fixed hourly **lag grid** `lag0`(=`t0`) … `lag{lag_hours}`
  (absent slot → `NaN`, mask `False`); static → `static_lat`/`static_lon`/`static_elevation_m`
  (target only); temporal → `t0_hour_sin`/`t0_hour_cos`/`t0_doy_sin`/`t0_doy_cos`.
```

- [ ] **Step 4: Verify docs render and don't break any doc test**

Run: `uv run pytest -q -k "context or adr or docs" || true`
Then: `git diff --stat`
Expected: the two doc files changed; no test references a fixed ADR count that would now break (if one does, update it).

- [ ] **Step 5: Commit**

```bash
git add docs/adr/0011-snapshot-normalization-boundary.md CONTEXT.md
git commit -m "docs(adr): ADR-0011 snapshot normalization boundary + as-of skew guarantee"
```

---

## Task 12: Full gate suite + final verification

**Files:** (no new files — verification only)

- [ ] **Step 1: Run the complete gate**

Run:
```bash
uv run ruff check . && uv run ruff format --check . && uv run lint-imports && uv run pyright && uv run pytest -q
```
Expected: `All checks passed!` (ruff ×2), import-linter contracts kept, `0 errors` (pyright), and a pytest summary with **more passing tests than the pre-task baseline** and no new failures. The previously-present `tests/features/test_snapshot_builder.py::test_builder_is_stubbed` is gone (replaced) — that is expected.

- [ ] **Step 2: If any gate fails, fix and re-run**

Common issues:
- pyright `reportUnknownMemberType` / `reportUnknownArgumentType` on pandas `.iloc`/`.iterrows()`/indexing → add the specific `# type: ignore[...]` on that line.
- ruff `PLC2701` on private-helper imports in tests → ensure each has `# noqa: PLC2701`.
- ruff format → `uv run ruff format .`.
- import-linter failure → confirm `snapshot_builder.py` imports only from `contracts`, `config`, `connectors` (no `models`/`evaluation`/`publication`/`pipelines`).

- [ ] **Step 3: Final confirmation — no `fetch_live` on the builder path**

Run: `grep -n "fetch_live" src/microclimate/features/snapshot_builder.py`
Expected: **no output** — the builder must never reference `fetch_live`.

- [ ] **Step 4: (No commit needed if Steps 1–10 each committed.)** If you batched, create a single final commit:

```bash
git add -A
git commit -m "feat(features): build_snapshot (NWP+obs → FeatureSnapshot, as-of, single normalization boundary)"
```

---

## Self-Review Notes

**Spec coverage (PRD #12 → tasks):**
- ✅ One snapshot per issue_time, all leads, raw values (Tasks 2–6; ADR-0011 in Task 11)
- ✅ `nwp_{var}_h{lead}`, target cell only, leads `1…horizon` (Task 3, Task 6)
- ✅ Fixed lag grid `lag0…lag{lag_hours}`, `obs_{station_id}_{var}_lag{k}`, per-(station,var,lag) mask, NaN absent (Task 4)
- ✅ As-of via `fetch_historical(end=issue_time)`, never `fetch_live`, defensive filter (Task 4, Task 7, Task 12 Step 3)
- ✅ Static (target lat/lon/elev, NaN if None), temporal (4 t0 encodings) (Tasks 2, 6)
- ✅ `lead_hours` always full horizon; feature_groups switches (Task 9)
- ✅ Degradation: NWP propagate; obs SourceUnavailable/empty degrade; StationNotFound propagate; all-obs-fail → NWP-only (Tasks 7, 8)
- ✅ `FeatureSnapshot` construction validates; `schema_version`/`deployment_id`/`issue_time` passthrough (Tasks 1, 6)
- ✅ Deep-module decomposition; all helpers + orchestrator + skew test (Tasks 2–6, 10)
- ✅ ADR + CONTEXT (Task 11); gates stay green (Task 12)

**Out of scope (not in any task, by design):** derived features, explode-to-rows, label attachment, the calling pipelines, any `fetch_live` strategy, model work.

**Type consistency check:** `_PHYSICAL_VARS` (src) and `PHYS` (test conftest) hold the same 8 names in the same order; `_align_obs_to_lag_grid` signature `(frame|None, station_id, issue_time, lag_hours)` matches its callers in `build_snapshot` and tests; `make_config`/`FakeNWP`/`FakeObs` signatures match every call site.
