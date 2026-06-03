# HRDPS Far-Lead Null Truncation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop inference going dark on live HRDPS far-lead nulls — truncate the forecast to the available contiguous lead prefix, publish a shorter forecast marked `stale`, and retry only when coverage is below a minimum floor.

**Architecture:** The openmeteo connector returns the contiguous non-null lead prefix instead of raising on the first null; `build_snapshot` stores the *actual* returned leads and raises `ForecastUnavailable` below `config.min_horizon_hours`; `run_inference` marks a truncated forecast `stale` (champion-`degraded` wins).

**Tech Stack:** Python 3.12, pandas, pydantic, pandera (`FORECAST_FRAME`), pytest, `uv`.

**Authoritative docs:** `docs/superpowers/specs/2026-06-03-hrdps-far-lead-null-truncation-design.md`; ADR-0019/0016/0011; CONTEXT.md.

**Verified current code:**
- `openmeteo._parse_hourly_to_forecast_frame(payload, *, issue_time, lead_hours)` loops leads; raises `ForecastUnavailable` on a missing time slot or any null var; builds rows; `return FORECAST_FRAME.validate(pd.DataFrame(rows))`. `_OPENMETEO_VAR_MAP`, `PHYSICAL_VARS`, `_PCT_TO_FRACTION`, `_OM_TIME_FMT` exist; `ForecastUnavailable` already imported.
- `snapshot_builder.build_snapshot`: `lead_hours = tuple(range(1, config.horizon_hours + 1))`; `if config.feature_groups.nwp: frame = nwp.fetch_forecast(...); nwp_features = _flatten_forecast(frame)`; later `FeatureSnapshot(..., lead_hours=lead_hours, ...)` ← stores the **requested** tuple (bug). Imports `SourceUnavailable` from `microclimate.connectors.base` (NOT `ForecastUnavailable` yet).
- `config/schema.py` `DeploymentConfig`: fields incl. `horizon_hours: int = Field(default=48, ge=1, le=48)`, `lag_hours`. `model_config = ConfigDict(extra="forbid")`. No model validator yet. (`from pydantic import BaseModel, ConfigDict, Field` — add `model_validator`.)
- `inference.run_inference`: sets `status` in TWO places — the `if registry_path is None:` early-return (`status="ok"`) and the main path (`status: Literal["ok","degraded"] = "degraded" if (tdeg or pdeg) else "ok"`). `_assemble_forecast` takes a `status` param.
- `FeatureSnapshot.lead_hours: tuple[int, ...]`; `feature_builder` builds rows from `snapshot.lead_hours` (any count); `ForecastDocument.status: Literal["ok","stale","degraded"]`, `series` no min length, `ForecastStep.lead_hour: Field(ge=1, le=48)`.

**Conventions:** UTC everywhere; full gate before each commit: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`.

---

## File Structure
- Modify: `src/microclimate/connectors/sources/openmeteo.py` — truncating parser.
- Modify: `src/microclimate/config/schema.py` — `min_horizon_hours` + validator.
- Modify: `config/deployments/lethbridge.yml` — `min_horizon_hours: 12`.
- Modify: `src/microclimate/features/snapshot_builder.py` — actual leads + floor.
- Modify: `src/microclimate/pipelines/inference.py` — `stale` status (both paths).
- Modify: `CONTEXT.md` — `min_horizon_hours` term + `stale` meaning.
- Tests: `tests/connectors/test_openmeteo.py` (extend), `tests/config/test_schema.py` (extend), `tests/features/test_snapshot_builder.py` (extend or create), `tests/pipelines/test_inference.py` (extend).

---

## Task 1: Connector returns the contiguous non-null prefix

**Files:**
- Modify: `src/microclimate/connectors/sources/openmeteo.py`
- Test: `tests/connectors/test_openmeteo.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/connectors/test_openmeteo.py`)

```python
def _payload(n_present: int, n_total: int) -> dict:
    """hourly payload with n_total slots; vars non-null for the first n_present, null after."""
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2024, 6, 1, 0, tzinfo=UTC)
    times = [(t0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n_total + 1)]
    hourly: dict[str, list] = {"time": times}
    om_vars = [
        "temperature_2m", "dew_point_2m", "surface_pressure", "precipitation",
        "cloud_cover", "shortwave_radiation", "wind_speed_10m", "wind_direction_10m",
    ]
    vals = {"surface_pressure": 900.0, "wind_direction_10m": 180.0, "cloud_cover": 50.0}
    for v in om_vars:
        base = vals.get(v, 1.0)
        # index 0 = t0 (lead 0, unused); indices 1..n_present present, rest null
        hourly[v] = [base] + [base if i <= n_present else None for i in range(1, n_total + 1)]
    return {"hourly": hourly}


def test_parse_truncates_at_first_null_returns_prefix() -> None:
    from datetime import UTC, datetime
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    t0 = datetime(2024, 6, 1, 0, tzinfo=UTC)
    payload = _payload(n_present=3, n_total=6)  # leads 1..3 present, 4..6 null
    df = _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1, 2, 3, 4, 5, 6])
    assert list(df["lead_hour"]) == [1, 2, 3]
    FORECAST_FRAME.validate(df)


def test_parse_returns_full_when_complete() -> None:
    from datetime import UTC, datetime
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    t0 = datetime(2024, 6, 1, 0, tzinfo=UTC)
    payload = _payload(n_present=6, n_total=6)
    df = _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1, 2, 3, 4, 5, 6])
    assert list(df["lead_hour"]) == [1, 2, 3, 4, 5, 6]


def test_parse_raises_when_first_lead_null() -> None:
    from datetime import UTC, datetime
    from microclimate.connectors.base import ForecastUnavailable
    from microclimate.connectors.sources.openmeteo import _parse_hourly_to_forecast_frame

    t0 = datetime(2024, 6, 1, 0, tzinfo=UTC)
    payload = _payload(n_present=0, n_total=6)  # lead 1 already null
    import pytest

    with pytest.raises(ForecastUnavailable):
        _parse_hourly_to_forecast_frame(payload, issue_time=t0, lead_hours=[1, 2, 3])
```
(`FORECAST_FRAME` is already imported at the top of this test module from prior tasks; if not, add `from microclimate.contracts.forecast_frame import FORECAST_FRAME`.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/connectors/test_openmeteo.py -k "truncates or returns_full or first_lead_null" -v`
Expected: `test_parse_truncates_...` FAILS (currently raises on the null at lead 4); the others may pass.

- [ ] **Step 3: Rewrite the loop in `_parse_hourly_to_forecast_frame`** to truncate

Replace the per-lead loop (the `for h in lead_hours:` block that currently raises on missing/null) with:
```python
    rows: list[dict[str, object]] = []
    for h in lead_hours:
        valid = issue_utc + timedelta(hours=int(h))
        key = valid.strftime(_OM_TIME_FMT)
        idx = index_by_time.get(key)
        if idx is None:
            break  # series doesn't reach this lead — truncate to the prefix so far
        row: dict[str, object] = {
            "issue_time": pd.Timestamp(issue_utc),
            "lead_hour": int(h),
            "valid_time": pd.Timestamp(valid),
        }
        incomplete = False
        for canon in PHYSICAL_VARS:
            raw = hourly[_OPENMETEO_VAR_MAP[canon]][idx]  # type: ignore[index]
            if raw is None:
                incomplete = True
                break  # null var (beyond the model's reach) — stop before adding this lead
            value = float(raw)  # type: ignore[arg-type]
            if canon == "cloud_cover_fraction":
                value = max(0.0, min(1.0, value / _PCT_TO_FRACTION))
            elif canon in ("precip_mm", "solar_radiation_wm2"):
                value = max(0.0, value)
            row[canon] = value
        if incomplete:
            break
        rows.append(row)

    if not rows:
        raise ForecastUnavailable(
            f"Open-Meteo returned no usable leads for issue_time {issue_utc.isoformat()} "
            "(lead 1 missing or null)."
        )
    df = pd.DataFrame(rows)
    return FORECAST_FRAME.validate(df)
```
Update the function docstring to: "Map an Open-Meteo `hourly` payload to a FORECAST_FRAME-valid DataFrame, returning the **contiguous non-null lead prefix** (the live series goes null past the freshest run's reach). Raises ForecastUnavailable only if no lead is available."

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/connectors/test_openmeteo.py -v`
Expected: all pass (the new 3 + existing).

- [ ] **Step 5: FULL gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add src/microclimate/connectors/sources/openmeteo.py tests/connectors/test_openmeteo.py
git commit -m "feat(openmeteo): return the contiguous non-null lead prefix (truncate far-lead nulls)"
```

---

## Task 2: `min_horizon_hours` config field + validator

**Files:**
- Modify: `src/microclimate/config/schema.py`
- Modify: `config/deployments/lethbridge.yml`
- Modify: `tests/fakes.py` (`make_config` must produce a valid `min_horizon_hours`)
- Test: `tests/config/test_schema.py`

**⚠ Regression to prevent:** adding the validator with `min_horizon_hours` default **12** will
reject any config built with `horizon_hours < 12` — and `tests/fakes.make_config` is called with
`horizon_hours=3` across the suite (smoke/inference/snapshot tests). So `make_config` MUST set a
valid `min_horizon_hours` (≤ its `horizon_hours`) or the whole suite breaks the moment the
validator lands. Update `make_config` in this task (Step 3b), before running the gate.

- [ ] **Step 1: Write the failing tests** (append to `tests/config/test_schema.py`)

```python
def test_min_horizon_defaults_to_12_and_loads_from_lethbridge() -> None:
    from microclimate.config.loader import load_deployment

    config = load_deployment("lethbridge")
    assert config.min_horizon_hours == 12
    assert config.min_horizon_hours <= config.horizon_hours


def test_min_horizon_greater_than_horizon_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from microclimate.config.schema import DeploymentConfig

    # build the smallest valid kwargs by reusing an existing helper if present; otherwise load
    # lethbridge and re-validate with an override.
    from microclimate.config.loader import load_deployment

    base = load_deployment("lethbridge").model_dump()
    base["min_horizon_hours"] = base["horizon_hours"] + 1
    with pytest.raises(ValidationError):
        DeploymentConfig.model_validate(base)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/config/test_schema.py -k min_horizon -v`
Expected: FAIL — `min_horizon_hours` attribute doesn't exist / not rejected.

- [ ] **Step 3: Add the field + validator** to `DeploymentConfig` in `src/microclimate/config/schema.py`

Ensure the import line includes `model_validator`: change `from pydantic import BaseModel, ConfigDict, Field` to `from pydantic import BaseModel, ConfigDict, Field, model_validator`. Add the field after `horizon_hours`:
```python
    horizon_hours: int = Field(default=48, ge=1, le=48)  # HRDPS lead-time ceiling (ADR-0007)
    min_horizon_hours: int = Field(default=12, ge=1)  # min available leads to still publish; else retry
```
And add the validator inside the class (after the fields):
```python
    @model_validator(mode="after")
    def _min_horizon_within_horizon(self) -> "DeploymentConfig":
        if self.min_horizon_hours > self.horizon_hours:
            raise ValueError(
                f"min_horizon_hours ({self.min_horizon_hours}) must be <= "
                f"horizon_hours ({self.horizon_hours})"
            )
        return self
```

- [ ] **Step 4: Set it in `config/deployments/lethbridge.yml`**

Add under the top-level keys (next to `horizon_hours`/`lag_hours`):
```yaml
min_horizon_hours: 12                      # publish a shorter (stale) forecast above this; else retry
```

- [ ] **Step 3b: Update `tests/fakes.make_config`** so it never violates the new validator

READ `tests/fakes.py`. `make_config` builds a `DeploymentConfig` (used with `horizon_hours=3`).
Add a `min_horizon_hours` parameter that defaults to a value ≤ `horizon_hours`, and pass it into
the `DeploymentConfig(...)` construction. Use:
```python
def make_config(
    *,
    horizon_hours: int = 48,
    lag_hours: int = 6,
    min_horizon_hours: int | None = None,
    # ... keep the rest of make_config's existing params ...
) -> DeploymentConfig:
    ...
    return DeploymentConfig(
        ...,
        horizon_hours=horizon_hours,
        min_horizon_hours=min_horizon_hours if min_horizon_hours is not None else min(12, horizon_hours),
        ...,
    )
```
(Match `make_config`'s actual signature/keyword style; the key point is `DeploymentConfig` gets a
`min_horizon_hours ≤ horizon_hours`. Defaulting to `min(12, horizon_hours)` keeps existing
`horizon_hours=3` callers valid while honoring the real default elsewhere.)

- [ ] **Step 5: Run the FULL suite to verify nothing broke + new tests pass**

Run: `uv run pytest -q`
Expected: all pass — confirm the validator didn't break the `make_config(horizon_hours=3)` callers
(smoke/inference/snapshot tests) and that `tests/config/test_schema.py`'s new tests pass.

- [ ] **Step 6: FULL gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright`
```bash
git add src/microclimate/config/schema.py config/deployments/lethbridge.yml tests/fakes.py tests/config/test_schema.py
git commit -m "feat(config): min_horizon_hours (default 12) + validator <= horizon_hours"
```

---

## Task 3: `build_snapshot` stores actual leads + enforces the floor

**Files:**
- Modify: `src/microclimate/features/snapshot_builder.py`
- Test: `tests/features/test_snapshot_builder.py`

- [ ] **Step 1: Write the failing tests** — add to `tests/features/test_snapshot_builder.py` (read the file first to reuse its fakes/config helper; it already tests build_snapshot. Reuse `tests.fakes` for `make_config`, and a fake NWP returning a chosen-length FORECAST_FRAME.)

```python
def test_build_snapshot_stores_actual_returned_leads(tmp_path=None) -> None:
    from datetime import UTC, datetime
    import pandas as pd
    from microclimate.connectors.base import NWPSource
    from microclimate.contracts.forecast_frame import FORECAST_FRAME
    from microclimate.features.snapshot_builder import build_snapshot
    from tests.fakes import PINNED, make_config

    class _TruncNWP(NWPSource):
        @property
        def is_live(self) -> bool:
            return True

        def fetch_forecast(self, issue_time, lat, lon, lead_hours):  # type: ignore[override]
            leads = [1, 2, 3]  # connector truncated to 3 even though more were requested
            rows = []
            for h in leads:
                r = {"issue_time": pd.Timestamp(issue_time), "lead_hour": h,
                     "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=h)}
                r.update(PINNED)
                rows.append(r)
            return FORECAST_FRAME.validate(pd.DataFrame(rows))

    config = make_config(horizon_hours=6, lag_hours=2, min_horizon_hours=2)
    snap = build_snapshot(config, datetime(2026, 6, 1, tzinfo=UTC), _TruncNWP(), {})
    assert snap.lead_hours == (1, 2, 3)  # actual, NOT the requested 1..6


def test_build_snapshot_raises_below_min_horizon() -> None:
    from datetime import UTC, datetime
    import pandas as pd
    import pytest
    from microclimate.connectors.base import ForecastUnavailable, NWPSource
    from microclimate.contracts.forecast_frame import FORECAST_FRAME
    from microclimate.features.snapshot_builder import build_snapshot
    from tests.fakes import PINNED, make_config

    class _ShortNWP(NWPSource):
        @property
        def is_live(self) -> bool:
            return True

        def fetch_forecast(self, issue_time, lat, lon, lead_hours):  # type: ignore[override]
            r = {"issue_time": pd.Timestamp(issue_time), "lead_hour": 1,
                 "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=1)}
            r.update(PINNED)
            return FORECAST_FRAME.validate(pd.DataFrame([r]))

    config = make_config(horizon_hours=6, lag_hours=2, min_horizon_hours=3)
    with pytest.raises(ForecastUnavailable):
        build_snapshot(config, datetime(2026, 6, 1, tzinfo=UTC), _ShortNWP(), {})
```
NOTE: `make_config` must accept `min_horizon_hours`. If `tests/fakes.make_config` doesn't yet pass it through, update `make_config` to accept `min_horizon_hours: int = 12` and set it on the built `DeploymentConfig` (read `tests/fakes.py`). The fakes config uses `feature_groups.observations` — pass empty `observations={}`; with observations on, build_snapshot reads obs. Set the fake config's `feature_groups.observations=False` (via make_config if it supports it) OR pass `observations={}` and ensure the target/neighbor connector_keys aren't looked up — simplest: use `make_config` with no neighbors and observations off so only NWP runs. Adjust make_config call accordingly (read the helper).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/features/test_snapshot_builder.py -k "actual_returned or below_min" -v`
Expected: FAIL — `lead_hours` is the requested tuple; no floor raise.

- [ ] **Step 3: Edit `build_snapshot`** in `src/microclimate/features/snapshot_builder.py`

Add `ForecastUnavailable` to the import: change `from microclimate.connectors.base import NWPSource, ObservationSource, SourceUnavailable` to `from microclimate.connectors.base import ForecastUnavailable, NWPSource, ObservationSource, SourceUnavailable`. Then in the NWP block, derive actual leads + enforce the floor, and use the actual leads in the snapshot:
```python
    lead_hours = tuple(range(1, config.horizon_hours + 1))
    actual_lead_hours = lead_hours  # may shrink if the NWP source truncated (far-lead nulls)

    nwp_features: dict[str, float] = {}
    if config.feature_groups.nwp:
        frame = nwp.fetch_forecast(issue_utc, config.target.lat, config.target.lon, lead_hours)
        nwp_features = _flatten_forecast(frame)
        actual_lead_hours = tuple(int(h) for h in frame["lead_hour"])
        if len(actual_lead_hours) < config.min_horizon_hours:
            raise ForecastUnavailable(
                f"only {len(actual_lead_hours)} HRDPS leads available for "
                f"{issue_utc.isoformat()} (< min_horizon_hours={config.min_horizon_hours})"
            )
```
and change the `FeatureSnapshot(...)` construction to use `lead_hours=actual_lead_hours` (instead of `lead_hours=lead_hours`).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/features/test_snapshot_builder.py -v`
Expected: pass (existing + new).

- [ ] **Step 5: FULL gate + commit**

```bash
git add src/microclimate/features/snapshot_builder.py tests/features/test_snapshot_builder.py tests/fakes.py
git commit -m "feat(snapshot): store actual returned leads + enforce min_horizon_hours floor"
```

---

## Task 4: Inference marks a truncated forecast `stale`

**Files:**
- Modify: `src/microclimate/pipelines/inference.py`
- Test: `tests/pipelines/test_inference.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/pipelines/test_inference.py`; reuse `_make_fakes`)

```python
def test_truncated_forecast_is_stale(tmp_path: Path) -> None:
    """A snapshot shorter than horizon_hours (no champion) publishes status='stale'."""
    config, nwp, obs = _make_fakes()  # make_config sets horizon_hours=3
    # Build a fake NWP that truncates to 2 leads (< horizon 3, >= a low floor)
    import pandas as pd
    from microclimate.connectors.base import NWPSource
    from microclimate.contracts.forecast_frame import FORECAST_FRAME
    from tests.fakes import PINNED

    class _Trunc(NWPSource):
        @property
        def is_live(self) -> bool:
            return True

        def fetch_forecast(self, issue_time, lat, lon, lead_hours):  # type: ignore[override]
            rows = []
            for h in (1, 2):
                r = {"issue_time": pd.Timestamp(issue_time), "lead_hour": h,
                     "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=h)}
                r.update(PINNED)
                rows.append(r)
            return FORECAST_FRAME.validate(pd.DataFrame(rows))

    it = datetime(2026, 6, 1, 0, tzinfo=UTC)
    doc = run_inference(config, nwp=_Trunc(), observations=obs, forecast_path=tmp_path / "f.json",
                        issue_time=it, registry_path=tmp_path / "absent.json", work_dir=tmp_path / "wd")
    assert doc.status == "stale"
    assert len(doc.series) == 2  # truncated
```
NOTE: `_make_fakes` builds `make_config(horizon_hours=3, lag_hours=2)` — confirm `min_horizon_hours` for that config is ≤ 2 so the floor passes (default 12 would reject 2 leads). Update the `_make_fakes`/`make_config` call to pass `min_horizon_hours=1` for these short-horizon tests, OR have `_Trunc` return 2 leads with the test config's `min_horizon_hours` set low. Adjust so the floor (Task 3) doesn't pre-empt the stale path. Also add a combined case if cheap: a truncated snapshot AND an expected-champion download failure → `status == "degraded"` (degraded wins over stale).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipelines/test_inference.py -k "stale" -v`
Expected: FAIL — status is `"ok"` for the truncated baseline path.

- [ ] **Step 3: Add the `stale` logic in `run_inference`** (`src/microclimate/pipelines/inference.py`)

Right after `snapshot = build_snapshot(...)` (or before assembling), compute:
```python
    truncated = len(snapshot.lead_hours) < config.horizon_hours
```
In the `if registry_path is None:` early-return branch, change `status="ok"` to:
```python
            status="stale" if truncated else "ok",
```
In the main path, widen the status type and precedence (degraded wins, then stale):
```python
    status: Literal["ok", "stale", "degraded"]
    if tdeg or pdeg:
        status = "degraded"
    elif truncated:
        status = "stale"
    else:
        status = "ok"
```
(`Literal` is already imported from `typing` in this module.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipelines/test_inference.py -v`
Expected: pass (existing + new; the existing full-coverage baseline tests stay `"ok"` since their fakes return the full horizon).

- [ ] **Step 5: FULL gate + commit**

```bash
git add src/microclimate/pipelines/inference.py tests/pipelines/test_inference.py
git commit -m "feat(inference): mark a truncated (< horizon) forecast stale (degraded wins)"
```

---

## Task 5: CONTEXT.md — `min_horizon_hours` + `stale` meaning

**Files:**
- Modify: `CONTEXT.md`

- [ ] **Step 1: Add the domain terms** (per CLAUDE.md: update CONTEXT when introducing a concept)

In `CONTEXT.md`, near the HRDPS / forecast-JSON terms, add:
- **`min_horizon_hours`** — the minimum number of contiguous available HRDPS leads for a run to still publish (default 12). The live HRDPS series goes null past the freshest run's reach; inference truncates the forecast to the available prefix `1…k` and publishes a **shorter** forecast, but if `k < min_horizon_hours` the run is treated as unavailable and retries (ADR-0019 §1b).
- Extend the **Forecast JSON** / status description: `status="stale"` now means **the forecast horizon was truncated below `horizon_hours`** (fewer than the target leads were available); `degraded` (a champion fell back) takes precedence over `stale`.

Keep it glossary-style (no implementation detail). If a "status" enumeration is already described, update it in place.

- [ ] **Step 2: Final FULL gate + commit**

Run: `uv run ruff format --check . && uv run ruff check . && uv run lint-imports && uv run pyright && uv run pytest`
```bash
git add CONTEXT.md
git commit -m "docs(context): min_horizon_hours + stale = truncated horizon"
```

---

## Self-Review

**Spec coverage:** connector truncation (Task 1 ↔ spec §1); `min_horizon_hours` config + validator (Task 2 ↔ §3); `build_snapshot` actual-leads + floor (Task 3 ↔ §2); inference `stale` precedence (Task 4 ↔ §4); CONTEXT glossary (Task 5 ↔ project convention). All spec components + error-handling cases covered.

**Placeholder scan:** every code step shows the actual code. Two tests carry explicit "read `tests/fakes.py` and make `make_config` accept `min_horizon_hours` / set a low floor" instructions — these are real read-then-edit dependencies (the fakes helper must thread the new field), not placeholders. Task 3 Step 1 notes the fakes-config must run NWP-only (observations off / no neighbors) — concrete.

**Type consistency:** `min_horizon_hours` (config field) used consistently in Tasks 2/3; `build_snapshot` stores `actual_lead_hours` (Task 3) which Task 4 reads as `snapshot.lead_hours`; `status` Literal widened to `["ok","stale","degraded"]` in Task 4 matches `ForecastDocument.status`. `ForecastUnavailable` imported where raised (connector already; snapshot_builder import added in Task 3).

**Cross-task dependency:** Task 3's tests require `tests/fakes.make_config` to accept `min_horizon_hours` — that helper edit is part of Task 3 (committed with it). Task 4 reuses it.

**Out of scope (per spec):** issue-time selection changes; §1b forward-capture.
