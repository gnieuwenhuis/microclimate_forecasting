from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
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


def _labels(issue_time: datetime = _T0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issue_time": pd.to_datetime([issue_time] * 3, utc=True),
            "lead_hour": [1, 2, 3],
            "valid_time": pd.to_datetime(
                [issue_time + timedelta(hours=h) for h in (1, 2, 3)], utc=True
            ),
            "label_temp_c": [10.0, 11.0, 12.0],
            "label_precip_occurrence": pd.array([1, 0, 1], dtype="Int64"),
        }
    )


def test_write_then_read_labels_round_trips(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels())
    out = store.read_labels("lethbridge")
    assert list(out["lead_hour"]) == [1, 2, 3]
    assert list(out["label_temp_c"]) == [10.0, 11.0, 12.0]
    assert list(out["label_precip_occurrence"].astype("Int64")) == [1, 0, 1]
    assert "deployment_id" in out.columns
    assert "written_at" not in out.columns  # internal bookkeeping not surfaced


def test_read_labels_unknown_deployment_is_empty(tmp_path: Path) -> None:
    out = TrainingStore(tmp_path).read_labels("nope")
    assert out.empty
    assert "lead_hour" in out.columns


def test_labels_dedupe_keeps_latest(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels(), written_at=_T0)
    revised = _labels()
    revised["label_temp_c"] = [20.0, 21.0, 22.0]
    store.write_labels("lethbridge", revised, written_at=_T0 + timedelta(hours=1))
    out = store.read_labels("lethbridge")
    assert list(out["label_temp_c"]) == [20.0, 21.0, 22.0]


def test_read_accepts_naive_and_non_utc_bounds(tmp_path: Path) -> None:
    from datetime import timezone

    store = TrainingStore(tmp_path)
    store.append_snapshot(_snap())  # issue_time = _T0 (2026-06-01 00:00 UTC)
    naive = datetime(2026, 5, 31, 23, 0)  # tz-naive → assumed UTC
    plus5 = datetime(2026, 6, 1, 6, 0, tzinfo=timezone(timedelta(hours=5)))  # = 01:00 UTC
    assert len(store.read_snapshots("lethbridge", start=naive, end=plus5)) == 1


def test_read_rejects_start_after_end(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    with pytest.raises(ValueError, match="after end"):
        store.read_snapshots("lethbridge", start=_T0 + timedelta(days=1), end=_T0)


def test_write_labels_missing_column_raises(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    bad = _labels().drop(columns="valid_time")
    with pytest.raises(ValueError, match="missing required column"):
        store.write_labels("lethbridge", bad)


def test_one_data_file_per_month_with_both_rows(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.append_snapshot(_snap(issue_time=_T0))
    store.append_snapshot(_snap(issue_time=_T0 + timedelta(days=2)))  # same month (June)
    ym_dir = tmp_path / "snapshots" / "deployment_id=lethbridge" / "ym=202606"
    assert [p.name for p in ym_dir.glob("*.parquet")] == ["data.parquet"]  # exactly one file
    assert len(store.read_snapshots("lethbridge")) == 2


def test_has_snapshot(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    assert store.has_snapshot("lethbridge", _T0) is False
    store.append_snapshot(_snap(issue_time=_T0))
    assert store.has_snapshot("lethbridge", _T0) is True
    assert store.has_snapshot("lethbridge", _T0 + timedelta(hours=1)) is False
    assert store.has_snapshot("other", _T0) is False


def test_one_label_data_file_per_month(tmp_path: Path) -> None:
    store = TrainingStore(tmp_path)
    store.write_labels("lethbridge", _labels(_T0), written_at=_T0)
    store.write_labels("lethbridge", _labels(_T0 + timedelta(days=1)), written_at=_T0)  # same month
    ym_dir = tmp_path / "labels" / "deployment_id=lethbridge" / "ym=202606"
    assert [p.name for p in ym_dir.glob("*.parquet")] == ["data.parquet"]
    out = store.read_labels("lethbridge")
    assert len(out) == 6  # two issue_times × 3 leads
