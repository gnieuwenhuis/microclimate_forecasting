from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.pipelines.training_data import assemble_training_rows
from tests.fakes import FakeNWP, FakeObs, make_config, make_forecast_frame, make_obs_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LEADS = [1, 2, 3]


def _obs_window() -> list[datetime]:
    # lag window (t0, t0-1, t0-2) + future label window (t0+1..t0+3)
    return [
        _T0 - timedelta(hours=2),
        _T0 - timedelta(hours=1),
        _T0,
        _T0 + timedelta(hours=1),
        _T0 + timedelta(hours=2),
        _T0 + timedelta(hours=3),
    ]


def _sources() -> tuple[FakeNWP, dict[str, FakeObs]]:
    ts = _obs_window()
    obs = FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})
    return FakeNWP(make_forecast_frame(_T0, _LEADS)), {"fake": obs}


def test_assembles_labeled_rows_with_cardinality() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)  # threshold default 0.2
    nwp, obs = _sources()

    rows = assemble_training_rows(config, nwp, obs, [_T0])

    assert len(rows) == 3  # one row per lead
    assert list(rows["lead_hour"]) == [1, 2, 3]
    assert "label_temp_c" in rows.columns
    # PINNED precip 0.5 >= 0.2 -> all occurrence 1
    assert list(rows["label_precip_occurrence"].astype("Int64")) == [1, 1, 1]
    assert rows["label_temp_c"].iloc[0] == 15.0  # PINNED temp_c


def test_threshold_drives_occurrence() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    config = config.model_copy(update={"label": config.label.model_copy(update={"precip_occurrence_threshold_mm": 0.6})})
    nwp, obs = _sources()

    rows = assemble_training_rows(config, nwp, obs, [_T0])
    assert list(rows["label_precip_occurrence"].astype("Int64")) == [0, 0, 0]  # 0.5 < 0.6


def test_multiple_issue_times_concatenate() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    t1 = _T0 + timedelta(hours=6)
    ts = _obs_window() + [t1 + timedelta(hours=h) for h in (-2, -1, 0, 1, 2, 3)]
    obs = {"fake": FakeObs(frames={"T1": make_obs_frame("T1", ts), "N1": make_obs_frame("N1", ts)})}
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))  # frame shape reused per issue_time

    rows = assemble_training_rows(config, nwp, obs, [_T0, t1])
    assert len(rows) == 6
    assert set(rows["issue_time"]) == {pd.Timestamp(_T0), pd.Timestamp(t1)}


def test_assemble_or_load_uses_cache(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from microclimate.connectors.base import SourceUnavailable
    from microclimate.pipelines.training_data import assemble_or_load

    config = make_config(horizon_hours=3, lag_hours=2)
    nwp, obs = _sources()
    cache = tmp_path / "rows.parquet"

    first = assemble_or_load(config, nwp, obs, [_T0], cache_path=cache)
    assert cache.exists()

    # Second call with exploding sources must still succeed -> proves it read the cache.
    boom_nwp = FakeNWP(exc=SourceUnavailable("should not be called"))
    boom_obs = {"fake": FakeObs(exc=SourceUnavailable("should not be called"))}
    second = assemble_or_load(config, boom_nwp, boom_obs, [_T0], cache_path=cache)

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True), check_like=True
    )
