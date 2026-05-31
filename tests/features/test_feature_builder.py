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


def test_obs_values_and_masks_passthrough() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    assert (df["obs_T1_temp_c_lag0"] == PINNED["temp_c"]).all()
    assert df["obs_T1_temp_c_lag0_mask"].all()
    assert (df["obs_N1_precip_mm_lag2"] == PINNED["precip_mm"]).all()
    assert df["obs_N1_precip_mm_lag2_mask"].all()


def test_absent_obs_is_nan_and_mask_false() -> None:
    config = make_config(horizon_hours=5, lag_hours=3)
    ts = [_T0 - timedelta(hours=k) for k in range(4)]
    frames = {
        "T1": make_obs_frame("T1", ts, absent={(0, "temp_c")}),
        "N1": make_obs_frame("N1", ts),
    }
    nwp = FakeNWP(make_forecast_frame(_T0, list(range(1, 6))))
    snap = build_snapshot(config, _T0, nwp, {"fake": FakeObs(frames=frames)})
    df = build_features(snap, config)
    assert df["obs_T1_temp_c_lag0"].isna().all()
    assert (~df["obs_T1_temp_c_lag0_mask"]).all()


def test_obs_dewpoint_depression_per_station_lag() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    expected = PINNED["temp_c"] - PINNED["dewpoint_c"]
    assert (df["obs_T1_dpd_lag0"] == expected).all()
    assert (df["obs_N1_dpd_lag3"] == expected).all()


def test_target_tendencies_present_are_zero() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    assert (df["obs_T1_ptend_3h"] == 0.0).all()
    assert (df["obs_T1_dpd_tend_3h"] == 0.0).all()


def test_target_tendencies_nan_when_lag3_missing() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=2)
    df = build_features(snap, config)
    assert df["obs_T1_ptend_3h"].isna().all()
    assert df["obs_T1_dpd_tend_3h"].isna().all()


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
    row = df.loc[df["lead_hour"] == 3].iloc[0]
    assert row["valid_hour_sin"] == pytest.approx(math.sin(2 * math.pi * 3 / 24.0))
    assert row["valid_hour_cos"] == pytest.approx(math.cos(2 * math.pi * 3 / 24.0))


from microclimate.features.feature_builder import _bearing_deg  # noqa: PLC2701


def test_bearing_due_east_is_90() -> None:
    b = _bearing_deg(51.0, -114.0, 51.0, -113.0)
    assert b == pytest.approx(90.0, abs=0.5)


def test_advection_gradients_zero_when_neighbor_equals_target() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    assert (df["adv_N1_temp_grad_lag0"] == 0.0).all()
    assert (df["adv_N1_dpd_grad_lag0"] == 0.0).all()
    assert (df["adv_N1_precip_grad_lag0"] == 0.0).all()


def test_upwind_alignment_matches_formula() -> None:
    snap, config = _snapshot(horizon_hours=5, lag_hours=3)
    df = build_features(snap, config)
    bearing = _bearing_deg(51.0, -114.0, 51.5, -113.5)
    expected = math.cos(math.radians(bearing - PINNED["wind_dir_deg"])) * PINNED["wind_speed_ms"]
    assert df["adv_N1_upwind_align"].to_numpy() == pytest.approx(expected)


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
