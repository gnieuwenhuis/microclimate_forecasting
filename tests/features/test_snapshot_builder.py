"""End-to-end hermetic tests for build_snapshot (injected fake connectors)."""

from __future__ import annotations

import inspect
import math
from datetime import UTC, datetime, timedelta

import pytest

from microclimate.connectors.base import ForecastUnavailable, SourceUnavailable, StationNotFound
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


def test_as_of_filters_future_obs() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
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

    assert len(snap.observation_features) == 2 * 8 * 1
    assert all(m is False for m in snap.observation_masks.values())
    assert all(math.isnan(v) for v in snap.observation_features.values())
    assert len(snap.nwp_features) == 8 * 1


def test_empty_frame_degrades_to_masked() -> None:
    config = make_config(horizon_hours=1, lag_hours=0)
    nwp = FakeNWP(make_forecast_frame(_T0, [1]))
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


def test_observations_switch_off_empties_obs_maps() -> None:
    config = make_config(horizon_hours=2, lag_hours=1, observations=False)
    nwp = FakeNWP(make_forecast_frame(_T0, [1, 2]))
    obs = FakeObs(exc=SourceUnavailable("should not be called"))
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert snap.observation_features == {}
    assert snap.observation_masks == {}
    assert len(snap.nwp_features) == 8 * 2
    assert len(snap.temporal_features) == 4
    assert len(snap.static_features) == 3


def test_nwp_switch_off_empties_nwp_but_keeps_lead_hours() -> None:
    config = make_config(horizon_hours=3, lag_hours=0, nwp=False)
    nwp = FakeNWP(exc=SourceUnavailable("should not be called"))
    obs = FakeObs(frames={"T1": make_obs_frame("T1", [_T0]), "N1": make_obs_frame("N1", [_T0])})
    snap = build_snapshot(config, _T0, nwp, {"fake": obs})

    assert snap.nwp_features == {}
    assert snap.lead_hours == (1, 2, 3)
    assert len(snap.observation_features) == 2 * 8 * 1
