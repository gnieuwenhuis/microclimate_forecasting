from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.features.labeler import attach_labels

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)


def _matrix(valid_times: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.to_datetime([_T0] * len(valid_times), utc=True),
            "lead_hour": list(range(1, len(valid_times) + 1)),
            "valid_time": pd.to_datetime(valid_times, utc=True),
            "nwp_temp_c": 10.0,
        }
    )


def _target_obs(rows: list[tuple[pd.Timestamp, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": "T1",
            "timestamp": pd.to_datetime([r[0] for r in rows], utc=True),
            "temp_c": [r[1] for r in rows],
            "precip_mm": [r[2] for r in rows],
        }
    )


def test_labels_join_at_valid_time_and_threshold() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    v2 = pd.Timestamp(_T0) + pd.Timedelta(hours=2)
    matrix = _matrix([v1, v2])
    obs = _target_obs([(v1, 12.5, 0.5), (v2, 9.0, 0.0)])

    out = attach_labels(matrix, obs, threshold_mm=0.2)

    assert list(out["label_temp_c"]) == [12.5, 9.0]
    assert list(out["label_precip_occurrence"]) == [1, 0]
    # original columns preserved (not converted to TRAINING_ROW)
    assert "feature_schema_version" in out.columns
    assert "nwp_temp_c" in out.columns


def test_threshold_is_inclusive() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    out = attach_labels(_matrix([v1]), _target_obs([(v1, 5.0, 0.2)]), threshold_mm=0.2)
    assert int(out["label_precip_occurrence"].iloc[0]) == 1


def test_missing_obs_yields_null_labels() -> None:
    v1 = pd.Timestamp(_T0) + pd.Timedelta(hours=1)
    v2 = pd.Timestamp(_T0) + pd.Timedelta(hours=2)
    matrix = _matrix([v1, v2])
    obs = _target_obs([(v1, 12.5, 0.5)])  # nothing for v2

    out = attach_labels(matrix, obs, threshold_mm=0.2)

    assert out["label_temp_c"].iloc[0] == 12.5
    assert pd.isna(out["label_temp_c"].iloc[1])
    assert int(out["label_precip_occurrence"].iloc[0]) == 1
    assert pd.isna(out["label_precip_occurrence"].iloc[1])
