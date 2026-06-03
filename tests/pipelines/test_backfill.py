from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, NWPSource, ObservationSource
from microclimate.contracts.physical_vars import PHYSICAL_VARS


def test_hrdps_issue_times_are_6h_cycles_inclusive() -> None:
    from microclimate.pipelines.backfill import hrdps_issue_times

    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    times = hrdps_issue_times(start, end)
    assert times[0] == start
    assert times[-1] == end
    assert all(t.hour in (0, 6, 12, 18) for t in times)
    assert len(times) == 5


# ---------------------------------------------------------------------------
# Typed fakes scoped to this test module (real lethbridge config + 48-lead horizon)
# ---------------------------------------------------------------------------

_PINNED: dict[str, float] = {
    "temp_c": 10.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.0,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 100.0,
    "wind_speed_ms": 3.0,
    "wind_dir_deg": 180.0,
}


class _FakeNWP(NWPSource):
    """Returns a valid FORECAST_FRAME for any issue_time / lead_hours."""

    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self,
        issue_time: datetime,
        lat: float,
        lon: float,
        lead_hours: Sequence[int],
    ) -> pd.DataFrame:
        from microclimate.contracts.forecast_frame import FORECAST_FRAME

        rows = [
            {
                "issue_time": pd.Timestamp(issue_time),
                "lead_hour": int(h),
                "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(h)),
                **_PINNED,
            }
            for h in lead_hours
        ]
        return FORECAST_FRAME.validate(pd.DataFrame(rows))


class _FakeObs(ObservationSource):
    """Returns an OBSERVATION_FRAME-valid frame with hourly rows covering [start, end].

    attach_labels joins on timestamp == valid_time (UTC-aware), so we cover the full
    valid-time span produced by each issue's 48-lead-hour forecast as well as the lag
    window before each issue_time needed by build_snapshot.
    """

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        from microclimate.contracts.observation import OBSERVATION_FRAME

        s = (
            pd.Timestamp(start).tz_localize("UTC")
            if pd.Timestamp(start).tzinfo is None
            else pd.Timestamp(start).tz_convert("UTC")
        )
        e = (
            pd.Timestamp(end).tz_localize("UTC")
            if pd.Timestamp(end).tzinfo is None
            else pd.Timestamp(end).tz_convert("UTC")
        )
        ts_range = pd.date_range(s, e, freq="1h")
        row: dict[str, object] = {"station_id": station_id, "timestamp": ts_range}
        for var in PHYSICAL_VARS:
            row[var] = _PINNED[var]
            row[f"{var}_present"] = True
        return OBSERVATION_FRAME.validate(pd.DataFrame(row))

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError("fetch_live must not be called during backfill")


def test_backfill_populates_store_idempotently(tmp_path: Path) -> None:
    from microclimate.config.loader import load_deployment
    from microclimate.pipelines.backfill import backfill_store, hrdps_issue_times
    from microclimate.training_store.store import TrainingStore

    config = load_deployment("lethbridge")
    store = TrainingStore(tmp_path)

    # All stations in lethbridge use connector_key="envcanada"; one fake source handles all.
    obs_map: dict[str, ObservationSource] = {
        config.target.connector_key: _FakeObs(),
    }

    times = hrdps_issue_times(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 18, 0, tzinfo=UTC),
    )  # 4 runs: 00, 06, 12, 18
    assert len(times) == 4

    n1 = backfill_store(
        config,
        nwp=_FakeNWP(),
        observations=obs_map,
        store=store,
        issue_times=times,
        pause_s=0.0,
    )
    snaps1 = store.read_snapshots(config.deployment_id)

    n2 = backfill_store(
        config,
        nwp=_FakeNWP(),
        observations=obs_map,
        store=store,
        issue_times=times,
        pause_s=0.0,
    )
    snaps2 = store.read_snapshots(config.deployment_id)

    assert n1 == len(times)
    assert n2 == 0  # idempotent: nothing new
    assert len(snaps2) == len(snaps1)  # additive, no dupes, no prune
    labels = store.read_labels(config.deployment_id)
    assert len(labels) == len(times) * config.horizon_hours


def test_hrdps_issue_times_off_boundary_start() -> None:
    from microclimate.pipelines.backfill import hrdps_issue_times

    times = hrdps_issue_times(
        datetime(2024, 1, 1, 3, 0, tzinfo=UTC), datetime(2024, 1, 1, 18, 0, tzinfo=UTC)
    )
    assert times == [
        datetime(2024, 1, 1, 6, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 18, 0, tzinfo=UTC),
    ]


def test_backfill_obs_prefetched_once_per_station(tmp_path: Path) -> None:
    """Inner fetch_historical must be called exactly once per station, not once per issue_time."""
    from microclimate.config.loader import load_deployment
    from microclimate.pipelines.backfill import backfill_store, hrdps_issue_times
    from microclimate.training_store.store import TrainingStore

    config = load_deployment("lethbridge")
    store = TrainingStore(tmp_path)

    # Count calls to fetch_historical per station_id.
    call_counts: dict[str, int] = {}

    class _CountingObs(_FakeObs):
        def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
            call_counts[station_id] = call_counts.get(station_id, 0) + 1
            return super().fetch_historical(station_id, start, end)

    obs_map: dict[str, ObservationSource] = {
        config.target.connector_key: _CountingObs(),
    }

    times = hrdps_issue_times(
        datetime(2024, 1, 1, 0, 0, tzinfo=UTC),
        datetime(2024, 1, 1, 18, 0, tzinfo=UTC),
    )  # 4 issue_times
    assert len(times) == 4

    backfill_store(
        config,
        nwp=_FakeNWP(),
        observations=obs_map,
        store=store,
        issue_times=times,
        pause_s=0.0,
    )

    # lethbridge: target + 4 neighbors = 5 distinct stations, all under "envcanada".
    all_station_ids = [config.target.station_id, *(n.station_id for n in config.neighbors)]
    assert len(all_station_ids) == 5

    # Each station should be fetched exactly once (the prefetch), NOT once per issue_time.
    for sid in all_station_ids:
        assert call_counts.get(sid, 0) == 1, (
            f"station {sid!r} was fetched {call_counts.get(sid, 0)} times; expected 1"
        )
