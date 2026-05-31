# Feature Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `features.build_features`, the pure read-time transform that explodes one `FeatureSnapshot` into the long-format **feature matrix** (one row per `(issue_time, lead_hour)`, no labels) with all derived features.

**Architecture:** A single deterministic, side-effect-free function `build_features(snapshot, config) -> pd.DataFrame` in the `features` layer (below `models`, so models and pipelines may call it). It assembles columns into a dict and builds one DataFrame: identity columns, own-lead NWP (+ derived `dpd`/`ptend_3h`), passthrough observations (+ masks, + derived `dpd`/tendencies), per-neighbor advection (gradients + geometry-aware upwind alignment), static, and temporal (t0 passthrough + per-lead `valid_hour`). Output validated by a new `FEATURE_ROW` Pandera schema. The column set is deterministic from `config`, giving train/serve column parity by construction. Purity is enforced by an import-linter forbidden contract.

**Tech Stack:** Python 3.12, pandas, Pandera (`pandera.pandas`), Pydantic v2 (existing `FeatureSnapshot`), pytest, uv, ruff, pyright, import-linter.

**Spec:** `docs/superpowers/specs/2026-05-31-feature-builder-design.md`

---

## File Structure

- **Create** `src/microclimate/contracts/feature_matrix.py` — `FEATURE_SCHEMA_VERSION` constant + `FEATURE_ROW` Pandera schema (identity columns + `strict=False` feature columns). No imports from `config` (contracts is the lowest layer).
- **Create** `src/microclimate/features/feature_builder.py` — `build_features` + private helpers (`_bearing_deg`, `_upwind_align`). Imports `config` + `contracts` (allowed: features > config > contracts).
- **Create** `tests/contracts/test_feature_matrix.py` — schema acceptance/rejection.
- **Create** `tests/features/test_feature_builder.py` — behavior tests; reuses `tests/features/conftest.py` fixtures (`make_config`, `make_forecast_frame`, `make_obs_frame`, `FakeNWP`, `FakeObs`, `PINNED`, `PHYS`) and `build_snapshot` to produce real snapshots.
- **Modify** `.importlinter` — add a forbidden contract: `feature_builder` must not import `connectors`.
- **Modify** `CONTEXT.md` — add *feature matrix*, *derived feature*, *feature schema version*.
- **Create** `docs/adr/0012-feature-builder-read-time-transform.md` — records the three decisions.
- **Modify** `README.md` — "Project status" section.

### Conventions reused from `build_snapshot`

- Physical vars, fixed order: `tests/features/conftest.py::PHYS` (= `snapshot_builder._PHYSICAL_VARS`): `temp_c, dewpoint_c, surface_pressure_hpa, precip_mm, cloud_cover_fraction, solar_radiation_wm2, wind_speed_ms, wind_dir_deg`.
- Snapshot key formats: NWP `nwp_{var}_h{lead}`; obs value/mask `obs_{station_id}_{var}_lag{k}`; static `static_lat/lon/elevation_m`; temporal `t0_hour_sin/cos`, `t0_doy_sin/cos`.
- Fixtures use `connector_key="fake"`, target `T1` at `(51.0, -114.0)`, neighbor `N1` at `(51.5, -113.5)`; `PINNED` gives constant values across all vars/leads/lags (so all tendencies and gradients evaluate to `0.0` when present).

---

## Task 1: `FEATURE_ROW` contract + `FEATURE_SCHEMA_VERSION`

**Files:**
- Create: `src/microclimate/contracts/feature_matrix.py`
- Test: `tests/contracts/test_feature_matrix.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/contracts/test_feature_matrix.py
from __future__ import annotations

import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION, FEATURE_ROW


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_schema_version": [FEATURE_SCHEMA_VERSION],
            "deployment_id": ["lethbridge"],
            "issue_time": pd.to_datetime(["2026-05-30T00:00:00Z"]),
            "lead_hour": [1],
            "valid_time": pd.to_datetime(["2026-05-30T01:00:00Z"]),
            "nwp_temp_c": [11.2],  # dynamic feature column — allowed
        }
    )


def test_feature_schema_version_is_a_string() -> None:
    assert isinstance(FEATURE_SCHEMA_VERSION, str)
    assert FEATURE_SCHEMA_VERSION


def test_accepts_identity_plus_dynamic_feature_columns() -> None:
    FEATURE_ROW.validate(_valid_frame())


def test_rejects_lead_hour_out_of_range() -> None:
    frame = _valid_frame()
    frame["lead_hour"] = [49]
    with pytest.raises(Exception):  # pandera SchemaError
        FEATURE_ROW.validate(frame)


def test_has_no_label_columns() -> None:
    # Feature matrix is label-free (scope A); labels are attached downstream.
    assert "label_temp_c" not in FEATURE_ROW.columns
    assert "label_precip_occurrence" not in FEATURE_ROW.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_feature_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: ... contracts.feature_matrix`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/microclimate/contracts/feature_matrix.py
"""Schema of the label-free feature matrix (L0). strict=False — feature columns vary.

Produced by features.build_features from a FeatureSnapshot. Distinct from TRAINING_ROW
(training_store.py), which is this plus labels; the two version independently.
"""

from __future__ import annotations

import pandera.pandas as pa

# Version of the DERIVED-feature set, distinct from SNAPSHOT_SCHEMA_VERSION (the raw-snapshot
# contract). Bump when the set/meaning of derived feature columns changes, so a model trained
# on a stale feature set is refused rather than silently misread (champion/challenger, ADR-0006).
FEATURE_SCHEMA_VERSION = "1.0.0"

FEATURE_ROW = pa.DataFrameSchema(
    {
        "feature_schema_version": pa.Column(str),
        "deployment_id": pa.Column(str),
        "issue_time": pa.Column("datetime64[ns, UTC]"),
        "lead_hour": pa.Column(int, pa.Check.in_range(1, 48)),
        "valid_time": pa.Column("datetime64[ns, UTC]"),
    },
    strict=False,
    coerce=True,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_feature_matrix.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/contracts/feature_matrix.py tests/contracts/test_feature_matrix.py
git commit -m "feat(contracts): FEATURE_ROW schema + FEATURE_SCHEMA_VERSION (label-free feature matrix)"
```

---

## Task 2: `build_features` skeleton — identity + own-lead NWP passthrough

**Files:**
- Create: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/features/test_feature_builder.py
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION
from microclimate.features.feature_builder import build_features
from microclimate.features.snapshot_builder import build_snapshot

from .conftest import PINNED, FakeNWP, FakeObs, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)


def _snapshot(*, horizon_hours: int = 5, lag_hours: int = 3, neighbors=None):
    """Real snapshot via build_snapshot + dense fake feeds (all PINNED, all present)."""
    config = make_config(horizon_hours=horizon_hours, lag_hours=lag_hours, neighbors=neighbors)
    leads = list(range(1, horizon_hours + 1))
    ts = [_T0 - timedelta(hours=k) for k in range(lag_hours + 1)]
    obs = FakeObs(
        frames={ref.station_id: make_obs_frame(ref.station_id, ts) for ref in [config.target, *config.neighbors]}
    )
    nwp = FakeNWP(make_forecast_frame(_T0, leads))
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})
    return snap, config


def test_returns_one_row_per_lead_hour() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    assert list(df["lead_hour"]) == [1, 2, 3, 4, 5]
    assert len(df) == 5


def test_identity_columns() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    assert (df["deployment_id"] == "test").all()
    assert (df["feature_schema_version"] == FEATURE_SCHEMA_VERSION).all()
    assert (df["issue_time"] == pd.Timestamp(_T0)).all()
    # valid_time = issue_time + lead_hour
    assert df.loc[df["lead_hour"] == 3, "valid_time"].iloc[0] == pd.Timestamp(_T0) + pd.Timedelta(hours=3)


def test_nwp_own_lead_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    # PINNED is constant across leads, so every row carries the pinned value.
    for var in ("temp_c", "dewpoint_c", "surface_pressure_hpa", "wind_speed_ms"):
        assert (df[f"nwp_{var}"] == PINNED[var]).all()


def test_rejects_snapshot_schema_mismatch() -> None:
    snap, config = _snapshot()
    bad = snap.model_copy(update={"schema_version": "0.0.0-bogus"})
    with pytest.raises(ValueError, match="schema_version"):
        build_features(bad, config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: ... features.feature_builder`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/microclimate/features/feature_builder.py
"""The single shared transform: FeatureSnapshot -> feature matrix (L3).

Pure, deterministic, no fitted state, no network. Explodes one FeatureSnapshot (raw,
canonicalized values for one issue time spanning all leads) into long-format rows, one per
(issue_time, lead_hour), with derived features. No labels are attached (ADR-0011 / the
feature-builder spec). Run identically at training-read time and inference; the column set is
deterministic from config, giving train/serve column parity by construction.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot

# Canonical physical variables, fixed order — must match snapshot_builder._PHYSICAL_VARS.
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


def build_features(snapshot: FeatureSnapshot, config: DeploymentConfig) -> pd.DataFrame:
    """Explode one FeatureSnapshot into the per-(issue_time, lead_hour) feature matrix."""
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"snapshot.schema_version {snapshot.schema_version!r} != expected "
            f"{SNAPSHOT_SCHEMA_VERSION!r}; refusing to build features from an incompatible snapshot."
        )

    leads = list(snapshot.lead_hours)
    n = len(leads)
    issue = snapshot.issue_time

    # Idiomatic pandas: lead_hour establishes the row count, then scalars broadcast and
    # per-lead lists assign directly. Avoids a mixed-type dict (pyright invariance) entirely.
    df = pd.DataFrame({"lead_hour": leads})
    df["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    df["deployment_id"] = snapshot.deployment_id
    df["issue_time"] = pd.to_datetime([issue] * n, utc=True)
    df["valid_time"] = pd.to_datetime([issue + timedelta(hours=h) for h in leads], utc=True)

    # --- NWP (own lead; _h{lead} suffix dropped — lead_hour is a column). ---
    if snapshot.nwp_features:
        nwp = snapshot.nwp_features
        for var in _PHYSICAL_VARS:
            df[f"nwp_{var}"] = [nwp[f"nwp_{var}_h{h}"] for h in leads]

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): build_features skeleton — identity columns + own-lead NWP passthrough"
```

---

## Task 3: NWP derived features (`nwp_dpd`, `nwp_ptend_3h`)

**Files:**
- Modify: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_nwp_dewpoint_depression() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    # PINNED: temp 15.0, dewpoint 5.0 -> dpd 10.0 on every row.
    assert (df["nwp_dpd"] == PINNED["temp_c"] - PINNED["dewpoint_c"]).all()


def test_nwp_pressure_tendency_nan_before_lead_4() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    # ptend_3h = pressure_h - pressure_{h-3}; needs h-3 >= 1, so NaN for leads 1,2,3.
    early = df.loc[df["lead_hour"].isin([1, 2, 3]), "nwp_ptend_3h"]
    assert early.isna().all()
    # PINNED pressure is constant -> tendency 0.0 once computable (lead 4, 5).
    late = df.loc[df["lead_hour"].isin([4, 5]), "nwp_ptend_3h"]
    assert (late == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -k "dewpoint_depression or pressure_tendency_nan" -v`
Expected: FAIL with `KeyError: 'nwp_dpd'`.

- [ ] **Step 3: Add to the NWP block (immediately after the passthrough loop, inside `if snapshot.nwp_features:`)**

```python
        df["nwp_dpd"] = [nwp[f"nwp_temp_c_h{h}"] - nwp[f"nwp_dewpoint_c_h{h}"] for h in leads]
        df["nwp_ptend_3h"] = [
            nwp[f"nwp_surface_pressure_hpa_h{h}"] - nwp[f"nwp_surface_pressure_hpa_h{h - 3}"]
            if h - 3 >= 1
            else math.nan
            for h in leads
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): NWP derived features (dewpoint depression, 3h pressure tendency)"
```

---

## Task 4: Observation passthrough (values + masks)

**Files:**
- Modify: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_obs_values_and_masks_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    # All obs present -> value == pinned, mask True, broadcast across all 5 rows.
    assert (df["obs_T1_temp_c_lag0"] == PINNED["temp_c"]).all()
    assert df["obs_T1_temp_c_lag0_mask"].all()
    assert (df["obs_N1_precip_mm_lag2"] == PINNED["precip_mm"]).all()
    assert df["obs_N1_precip_mm_lag2_mask"].all()


def test_absent_obs_is_nan_and_mask_false() -> None:
    # Build a snapshot whose target lag0 temp is absent.
    config = make_config(horizon_hours=5, lag_hours=3)
    ts = [_T0 - timedelta(hours=k) for k in range(4)]
    frames = {
        "T1": make_obs_frame("T1", ts, absent={(0, "temp_c")}),  # row 0 == lag0
        "N1": make_obs_frame("N1", ts),
    }
    nwp = FakeNWP(make_forecast_frame(_T0, list(range(1, 6))))
    snap = build_snapshot(config, _T0, nwp, {"fake": FakeObs(frames=frames)})
    df = build_features(snap, config)
    assert df["obs_T1_temp_c_lag0"].isna().all()
    assert (~df["obs_T1_temp_c_lag0_mask"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -k "passthrough or absent_obs" -v`
Expected: FAIL with `KeyError: 'obs_T1_temp_c_lag0'`.

- [ ] **Step 3: Add an observations block after the NWP block (before `return df`)**

```python
    # --- Observations (passthrough values + masks; scalars broadcast across all lead rows). ---
    if snapshot.observation_features:
        obs = snapshot.observation_features
        masks = snapshot.observation_masks
        for key, value in obs.items():
            df[key] = value
            df[f"{key}_mask"] = masks[key]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): passthrough observation values + missingness masks"
```

---

## Task 5: Observation derived features (`dpd` per lag + target tendencies)

**Files:**
- Modify: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_obs_dewpoint_depression_per_station_lag() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    expected = PINNED["temp_c"] - PINNED["dewpoint_c"]
    assert (df["obs_T1_dpd_lag0"] == expected).all()
    assert (df["obs_N1_dpd_lag3"] == expected).all()


def test_target_tendencies_present_are_zero() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    # PINNED constant across lags -> tendencies 0.0.
    assert (df["obs_T1_ptend_3h"] == 0.0).all()
    assert (df["obs_T1_dpd_tend_3h"] == 0.0).all()


def test_target_tendencies_nan_when_lag3_missing() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=2)  # no lag3 in the grid
    df = build_features(snap, config)
    assert df["obs_T1_ptend_3h"].isna().all()
    assert df["obs_T1_dpd_tend_3h"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -k "obs_dewpoint or tendencies" -v`
Expected: FAIL with `KeyError: 'obs_T1_dpd_lag0'`.

- [ ] **Step 3: Add to the observations block (after the passthrough loop, still inside `if snapshot.observation_features:`)**

```python
        station_ids = [config.target.station_id, *[ref.station_id for ref in config.neighbors]]
        for sid in station_ids:
            for k in range(config.lag_hours + 1):
                t = obs.get(f"obs_{sid}_temp_c_lag{k}", math.nan)
                d = obs.get(f"obs_{sid}_dewpoint_c_lag{k}", math.nan)
                df[f"obs_{sid}_dpd_lag{k}"] = t - d  # scalar broadcast

        tgt = config.target.station_id
        p0 = obs.get(f"obs_{tgt}_surface_pressure_hpa_lag0", math.nan)
        p3 = obs.get(f"obs_{tgt}_surface_pressure_hpa_lag3", math.nan)
        df[f"obs_{tgt}_ptend_3h"] = p0 - p3
        dpd0 = obs.get(f"obs_{tgt}_temp_c_lag0", math.nan) - obs.get(f"obs_{tgt}_dewpoint_c_lag0", math.nan)
        dpd3 = obs.get(f"obs_{tgt}_temp_c_lag3", math.nan) - obs.get(f"obs_{tgt}_dewpoint_c_lag3", math.nan)
        df[f"obs_{tgt}_dpd_tend_3h"] = dpd0 - dpd3
```

Note: `obs.get(..., math.nan)` yields NaN when a lag slot is outside the grid (`lag_hours < 3`)
or absent; the subtractions then propagate NaN. Tendencies recompute their endpoints directly
from `obs` (no DataFrame read-back), so they are NaN-safe regardless of column order.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): obs derived features (dewpoint depression per lag + target 3h tendencies)"
```

---

## Task 6: Static + temporal features

**Files:**
- Modify: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_static_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    assert (df["static_lat"] == 51.0).all()
    assert (df["static_lon"] == -114.0).all()
    assert (df["static_elevation_m"] == 900.0).all()


def test_t0_temporal_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    for key in ("t0_hour_sin", "t0_hour_cos", "t0_doy_sin", "t0_doy_cos"):
        assert (df[key] == snap.temporal_features[key]).all()


def test_valid_hour_encoding_per_lead() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    # _T0 is 00:00Z; lead 3 -> valid hour 3.
    row = df.loc[df["lead_hour"] == 3].iloc[0]
    assert row["valid_hour_sin"] == pytest.approx(math.sin(2 * math.pi * 3 / 24.0))
    assert row["valid_hour_cos"] == pytest.approx(math.cos(2 * math.pi * 3 / 24.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -k "static_passthrough or temporal_passthrough or valid_hour" -v`
Expected: FAIL with `KeyError: 'static_lat'`.

- [ ] **Step 3: Add a static + temporal block after the observations block (before `return df`)**

```python
    # --- Static (target only; broadcast). ---
    for key, value in snapshot.static_features.items():
        df[key] = value

    # --- Temporal: t0 passthrough (broadcast) + per-lead valid-time hour encoding. ---
    for key, value in snapshot.temporal_features.items():
        df[key] = value
    valid_hours = [(issue + timedelta(hours=h)).hour for h in leads]
    df["valid_hour_sin"] = [math.sin(2 * math.pi * vh / 24.0) for vh in valid_hours]
    df["valid_hour_cos"] = [math.cos(2 * math.pi * vh / 24.0) for vh in valid_hours]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): static passthrough + temporal (t0 passthrough + per-lead valid_hour)"
```

---

## Task 7: Advection (bearing helper + gradients + upwind alignment)

**Files:**
- Modify: `src/microclimate/features/feature_builder.py`
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
from microclimate.features.feature_builder import _bearing_deg  # noqa: PLC2701


def test_bearing_due_east_is_90() -> None:
    # Same latitude, neighbor to the east -> bearing ~90 degrees.
    b = _bearing_deg(51.0, -114.0, 51.0, -113.0)
    assert b == pytest.approx(90.0, abs=0.5)


def test_advection_gradients_zero_when_neighbor_equals_target() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    # PINNED identical for target and neighbor -> gradient 0.0.
    assert (df["adv_N1_temp_grad_lag0"] == 0.0).all()
    assert (df["adv_N1_dpd_grad_lag0"] == 0.0).all()
    assert (df["adv_N1_precip_grad_lag0"] == 0.0).all()


def test_upwind_alignment_matches_formula() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    bearing = _bearing_deg(51.0, -114.0, 51.5, -113.5)  # T1 -> N1
    expected = math.cos(math.radians(bearing - PINNED["wind_dir_deg"])) * PINNED["wind_speed_ms"]
    assert (df["adv_N1_upwind_align"] == pytest.approx(expected)).all()


def test_upwind_alignment_nan_when_wind_absent() -> None:
    config = make_config(horizon_hours=5, lag_hours=3)
    ts = [_T0 - timedelta(hours=k) for k in range(4)]
    frames = {
        "T1": make_obs_frame("T1", ts, absent={(0, "wind_dir_deg"), (0, "wind_speed_ms")}),
        "N1": make_obs_frame("N1", ts),
    }
    nwp = FakeNWP(make_forecast_frame(_T0, list(range(1, 6))))
    snap = build_snapshot(config, _T0, nwp, {"fake": FakeObs(frames=frames)})
    df = build_features(snap, config)
    assert df["adv_N1_upwind_align"].isna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/features/test_feature_builder.py -k "bearing or advection or upwind" -v`
Expected: FAIL with `ImportError: cannot import name '_bearing_deg'`.

- [ ] **Step 3a: Add the bearing helper at module level (below `_PHYSICAL_VARS`; `math` already imported at top from Task 2)**

```python
def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing in degrees [0, 360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
```

- [ ] **Step 3b: Add an advection block after the static/temporal block (before `return df`)**

```python
    # --- Advection (per neighbor): neighbor-target gradients at lag0 + upwind alignment. ---
    if snapshot.observation_features and config.neighbors:
        obs = snapshot.observation_features  # re-bind (guard ensures non-empty; avoids unbound)
        tgt = config.target.station_id
        wind_from = obs.get(f"obs_{tgt}_wind_dir_deg_lag0", math.nan)
        wind_speed = obs.get(f"obs_{tgt}_wind_speed_ms_lag0", math.nan)
        t_temp = obs.get(f"obs_{tgt}_temp_c_lag0", math.nan)
        t_precip = obs.get(f"obs_{tgt}_precip_mm_lag0", math.nan)
        t_dpd = t_temp - obs.get(f"obs_{tgt}_dewpoint_c_lag0", math.nan)
        for ref in config.neighbors:
            nid = ref.station_id
            n_temp = obs.get(f"obs_{nid}_temp_c_lag0", math.nan)
            n_precip = obs.get(f"obs_{nid}_precip_mm_lag0", math.nan)
            n_dpd = n_temp - obs.get(f"obs_{nid}_dewpoint_c_lag0", math.nan)
            df[f"adv_{nid}_temp_grad_lag0"] = n_temp - t_temp
            df[f"adv_{nid}_dpd_grad_lag0"] = n_dpd - t_dpd
            df[f"adv_{nid}_precip_grad_lag0"] = n_precip - t_precip
            bearing = _bearing_deg(config.target.lat, config.target.lon, ref.lat, ref.lon)
            df[f"adv_{nid}_upwind_align"] = math.cos(math.radians(bearing - wind_from)) * wind_speed
```

Note: when `wind_from` or `wind_speed` is NaN (target wind absent at lag0), the alignment is NaN and propagates — no branch needed. Scalars broadcast across the lead rows.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/microclimate/features/feature_builder.py tests/features/test_feature_builder.py
git commit -m "feat(features): advection — neighbor-target gradients + geometry-aware upwind alignment"
```

---

## Task 8: Column-set determinism, train/serve parity, and schema validation

**Files:**
- Test: `tests/features/test_feature_builder.py`

- [ ] **Step 1: Write the failing test**

```python
def test_output_validates_against_feature_row() -> None:
    from microclimate.contracts.feature_matrix import FEATURE_ROW

    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    FEATURE_ROW.validate(df)  # raises on failure


def test_train_serve_column_parity() -> None:
    # Same config -> identical column set & order regardless of issue_time (train vs serve).
    config = make_config(horizon_hours=5, lag_hours=3)
    leads = list(range(1, 6))

    def snap_at(t0: datetime):
        ts = [t0 - timedelta(hours=k) for k in range(4)]
        obs = FakeObs(frames={s: make_obs_frame(s, ts) for s in ("T1", "N1")})
        return build_snapshot(config, t0, FakeNWP(make_forecast_frame(t0, leads)), {"fake": obs})

    train = build_features(snap_at(datetime(2019, 1, 1, 6, tzinfo=UTC)), config)
    serve = build_features(snap_at(datetime(2026, 5, 30, 18, tzinfo=UTC)), config)
    assert list(train.columns) == list(serve.columns)


def test_neighbor_count_changes_columns_deterministically() -> None:
    from microclimate.config.schema import StationRef

    cfg1 = make_config(horizon_hours=5, lag_hours=3)
    extra = [
        StationRef(station_id="N1", connector_key="fake", lat=51.5, lon=-113.5, elevation_m=950.0),
        StationRef(station_id="N2", connector_key="fake", lat=50.5, lon=-114.5, elevation_m=920.0),
    ]
    snap1, _ = _snapshot(horizon_hours=5, lag_hours=3)
    snap2, cfg2 = _snapshot(horizon_hours=5, lag_hours=3, neighbors=extra)
    c1 = set(build_features(snap1, cfg1).columns)
    c2 = set(build_features(snap2, cfg2).columns)
    assert "adv_N2_upwind_align" in c2
    assert "adv_N2_upwind_align" not in c1


def test_obs_off_emits_no_obs_or_adv_columns() -> None:
    config = make_config(horizon_hours=5, lag_hours=3, observations=False)
    snap = build_snapshot(config, _T0, FakeNWP(make_forecast_frame(_T0, list(range(1, 6)))), {})
    df = build_features(snap, config)
    assert not [c for c in df.columns if c.startswith(("obs_", "adv_"))]
    assert "nwp_temp_c" in df.columns  # NWP still present
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `uv run pytest tests/features/test_feature_builder.py -k "feature_row or parity or neighbor_count or obs_off" -v`
Expected: These should PASS already if Tasks 2–7 are correct (this task locks the invariants). If `test_output_validates_against_feature_row` fails on a dtype, fix by ensuring `issue_time`/`valid_time` are `datetime64[ns, UTC]` (already coerced) — no further code expected.

- [ ] **Step 3: No new implementation expected**

If all pass, proceed. If `test_obs_off_emits_no_obs_or_adv_columns` fails, confirm the observations and advection blocks are both guarded by `if snapshot.observation_features:` / `and config.neighbors`.

- [ ] **Step 4: Run the full feature-builder suite**

Run: `uv run pytest tests/features/test_feature_builder.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add tests/features/test_feature_builder.py
git commit -m "test(features): lock column determinism, train/serve parity, FEATURE_ROW validation"
```

---

## Task 9: Enforce purity via import-linter forbidden contract

**Files:**
- Modify: `.importlinter`
- Test: `tests/architecture/test_layering.py` (already runs `lint-imports`)

- [ ] **Step 1: Add a forbidden contract to `.importlinter`** (append after the `independence` contract)

```ini
[importlinter:contract:feature_builder_purity]
name = build_features is pure — must not import connectors
type = forbidden
source_modules =
    microclimate.features.feature_builder
forbidden_modules =
    microclimate.connectors
```

- [ ] **Step 2: Run the import-linter check**

Run: `uv run lint-imports`
Expected: all contracts KEPT (including `feature_builder_purity`).

- [ ] **Step 3: Run the architecture test**

Run: `uv run pytest tests/architecture/test_layering.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add .importlinter
git commit -m "test(arch): forbid feature_builder importing connectors (purity guardrail)"
```

---

## Task 10: Documentation — CONTEXT.md terms, ADR-0012, README status

**Files:**
- Modify: `CONTEXT.md`
- Create: `docs/adr/0012-feature-builder-read-time-transform.md`
- Modify: `README.md`

- [ ] **Step 1: Add three terms to `CONTEXT.md`** under the `### Data contract` section, immediately after the **Feature snapshot** bullet

```markdown
- **Feature matrix** — the long-format, per-`(issue_time, lead_hour)` model-input rows
  produced by `features.build_features` from a **feature snapshot**. One row per lead hour;
  carries **derived features** plus the as-of-`t0` snapshot values broadcast across rows;
  **label-free** (labels are attached downstream). Built at training-read time and at
  inference by the **same** function, so its column set is identical for train and serve.
- **Derived feature** — a feature computed from raw snapshot values (dewpoint depression,
  pressure tendency, advection, per-lead-hour `valid_hour` encoding), as distinct from a
  passthrough of a raw snapshot value. Derived features are pure functions of the snapshot
  (ADR-0011, ADR-0012).
- **Feature schema version** — `FEATURE_SCHEMA_VERSION`, the version of the **derived
  feature** set, distinct from `SNAPSHOT_SCHEMA_VERSION` (the raw-snapshot contract). A model
  records the feature version it trained on so a stale-feature champion is refused.
```

- [ ] **Step 2: Create the ADR**

```markdown
# 12. The feature builder is a read-time transform; the training store holds raw snapshots

- **Status:** Accepted
- **Date:** 2026-05-31
- **Relates to:** ADR-0011 (snapshot is the normalization boundary), ADR-0004 (two LightGBM
  models, long-format rows), ADR-0006 (champion/challenger).

## Context

ADR-0011 made `build_snapshot` the normalization/as-of boundary holding raw canonicalized
values, and named a follow-on work item: a downstream step that produces the per-lead-hour
model-input rows and derived features. Three questions had to be settled before building it:
when the transform runs, how the derived-feature set is versioned, and whether values are
statistically scaled.

## Decision

**1. The feature builder runs at read time; the training store holds raw snapshots.**
`features.build_features(snapshot, config)` is a pure function executed at training-read time
and at inference — never at write time. The training store persists raw snapshot values (+
labels); derived features are recomputed on read. This keeps a single shared feature code
path (the reason `build_snapshot` exists), makes feature iteration cheap (retrain only — no
re-log or backfill), keeps the store feature-version-independent and smaller, and forces the
transform to be deterministic and self-contained.

**2. The derived-feature set is versioned independently** via `FEATURE_SCHEMA_VERSION`,
distinct from `SNAPSHOT_SCHEMA_VERSION`. Models record the feature version they trained on so
a champion built against a stale feature set is refused rather than silently misread.
`build_features` also rejects a snapshot whose `schema_version` it does not recognise.

**3. No statistical scaling.** LightGBM is tree-based; the builder adds derived columns and
explodes to rows but does not standardize values. "Normalization" in ADR-0011 means
canonicalization (units, variable order), already done upstream.

## Consequences

- Train/serve skew is eliminated at this layer too: the column set is deterministic from
  config, so a training-`t0` and an inference-`t0` snapshot yield identical columns (a guarded
  parity invariant).
- The transform is import-pure (no `connectors`, no I/O), enforced by an import-linter
  forbidden contract.
- A label-attachment step (join observed labels onto the feature matrix to form `TRAINING_ROW`)
  remains a separate downstream work item.
- `model.predict` needs `config` to build rows; the recommended resolution (deferred to the
  models work) is for the pipeline to own the `build_features` call and pass rows to both
  `fit` and `predict`.
```

Save as `docs/adr/0012-feature-builder-read-time-transform.md`.

- [ ] **Step 3: Update the README "Project status" section**

Run: `grep -n "Project status" README.md` and read that section, then mark the downstream
feature step as implemented. Add a line such as:

```markdown
- `features.build_features` — read-time transform from `FeatureSnapshot` to the feature
  matrix (derived features + explode-to-per-lead-hour rows). **Done** (ADR-0012).
```

Keep the surrounding list style consistent with what is already there.

- [ ] **Step 4: Verify docs reference real symbols**

Run: `grep -rn "build_features\|FEATURE_SCHEMA_VERSION\|feature matrix" CONTEXT.md docs/adr/0012-feature-builder-read-time-transform.md README.md`
Expected: matches in all three files; names match the code (`build_features`, `FEATURE_SCHEMA_VERSION`).

- [ ] **Step 5: Commit**

```bash
git add CONTEXT.md docs/adr/0012-feature-builder-read-time-transform.md README.md
git commit -m "docs: add feature matrix terms, ADR-0012 (read-time transform), README status"
```

---

## Task 11: Full verification gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full CI gate locally**

```bash
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run pyright
uv run pytest
```

Expected: all pass. If `ruff format --check` reports diffs, run `uv run ruff format .` and amend the relevant commit (long lines in the advection/tendency blocks may rewrap). If `pyright` flags `obs.get(..., math.nan)` returning `float | None`, confirm `math.nan` (a `float`) is passed as the default so the overload resolves to `float`; helper `_bearing_deg` has an explicit `-> float` return.

- [ ] **Step 2: Confirm the new tests are collected and green**

Run: `uv run pytest tests/features/test_feature_builder.py tests/contracts/test_feature_matrix.py tests/architecture/test_layering.py -v`
Expected: PASS.

- [ ] **Step 3: Final commit if formatting changed**

```bash
git add -A
git commit -m "chore: ruff format feature builder" || echo "nothing to format"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** placement/signature (T2), contract+versioning (T1), NWP own-lead+derived (T2/T3), obs passthrough+masks+derived+tendencies (T4/T5), advection geometry (T7), static+temporal (T6), error handling/edge cases (T2 schema-mismatch, T4 absent obs, T5 tendency-NaN, T7 wind-absent, T8 obs-off), determinism/parity (T8), purity (T9), CONTEXT/ADR/README (T10). The models-integration item is documented (ADR-0012 consequence), not implemented — matches spec scope.
- **Naming consistency:** `build_features`, `FEATURE_ROW`, `FEATURE_SCHEMA_VERSION`, `_bearing_deg`, column names (`nwp_{var}`, `nwp_dpd`, `nwp_ptend_3h`, `obs_{sid}_{var}_lag{k}(+_mask)`, `obs_{sid}_dpd_lag{k}`, `obs_{tgt}_ptend_3h`, `obs_{tgt}_dpd_tend_3h`, `adv_{nid}_*`, `valid_hour_sin/cos`) are consistent across tasks and match the spec catalog.
- **`ptend` boundary:** corrected to NaN for `lead < 4` (needs `h−3 ≥ 1`); spec updated to match.
