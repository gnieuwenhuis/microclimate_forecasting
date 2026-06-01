from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot
from microclimate.training_store import TrainingStore

_T0 = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)


def _snap(deployment_id: str = "lethbridge", issue_time: datetime = _T0) -> FeatureSnapshot:
    return FeatureSnapshot(
        deployment_id=deployment_id,
        issue_time=issue_time,
        nwp_features={"nwp_temp_c_h1": 10.0, "nwp_temp_c_h2": 11.0},
        observation_features={"obs_T1_temp_c_lag0": 9.5},
        observation_masks={"obs_T1_temp_c_lag0": True},
        static_features={"static_lat": 49.7, "static_lon": -112.77},
        temporal_features={"t0_hour_sin": 0.0, "t0_hour_cos": 1.0},
        lead_hours=(1, 2, 3),
        schema_version=SNAPSHOT_SCHEMA_VERSION,
    )


def test_append_then_read_round_trips(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    snap = _snap()
    store.append_snapshot(snap)
    out = store.read_snapshots("lethbridge")
    assert len(out) == 1
    assert out[0] == snap  # Pydantic value-equality across the JSON round-trip


def test_read_unknown_deployment_is_empty(tmp_path: Path) -> None:
    assert TrainingStore(tmp_path).read_snapshots("nope") == []


def test_dedupe_keeps_latest_write_per_issue_time(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    first = _snap()
    second = _snap()
    second = second.model_copy(update={"nwp_features": {"nwp_temp_c_h1": 99.0}})
    store.append_snapshot(first, written_at=_T0)
    store.append_snapshot(second, written_at=_T0 + timedelta(hours=1))  # later write wins
    out = store.read_snapshots("lethbridge")
    assert len(out) == 1
    assert out[0].nwp_features["nwp_temp_c_h1"] == 99.0


def test_start_end_filter_by_issue_time(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    for i in range(0, 90, 30):  # three issue_times spanning ~3 months
        store.append_snapshot(_snap(issue_time=_T0 + timedelta(days=i)))
    mid = store.read_snapshots(
        "lethbridge", start=_T0 + timedelta(days=20), end=_T0 + timedelta(days=40)
    )
    assert [s.issue_time for s in mid] == [_T0 + timedelta(days=30)]


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.append_snapshot(_snap().model_copy(update={"schema_version": "9.9.9"}))
    with pytest.raises(ValueError, match="schema_version"):
        store.read_snapshots("lethbridge")
