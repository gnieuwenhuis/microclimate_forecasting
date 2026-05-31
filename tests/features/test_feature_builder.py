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
    assert df.loc[df["lead_hour"] == 3, "valid_time"].iloc[0] == pd.Timestamp(_T0) + pd.Timedelta(hours=3)


def test_nwp_own_lead_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    for var in ("temp_c", "dewpoint_c", "surface_pressure_hpa", "wind_speed_ms"):
        assert (df[f"nwp_{var}"] == PINNED[var]).all()


def test_rejects_snapshot_schema_mismatch() -> None:
    snap, config = _snapshot()
    bad = snap.model_copy(update={"schema_version": "0.0.0-bogus"})
    with pytest.raises(ValueError, match="schema_version"):
        build_features(bad, config)


def test_nwp_dewpoint_depression() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    assert (df["nwp_dpd"] == PINNED["temp_c"] - PINNED["dewpoint_c"]).all()


def test_nwp_pressure_tendency_nan_before_lead_4() -> None:
    snap, config = _snapshot(horizon_hours=5)
    df = build_features(snap, config)
    early = df.loc[df["lead_hour"].isin([1, 2, 3]), "nwp_ptend_3h"]
    assert early.isna().all()
    late = df.loc[df["lead_hour"].isin([4, 5]), "nwp_ptend_3h"]
    assert (late == 0.0).all()
