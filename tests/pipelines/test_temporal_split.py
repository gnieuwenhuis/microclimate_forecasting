from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest


def _rows(n_issue: int, step_h: int = 6, leads: int = 2) -> pd.DataFrame:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    recs: list[dict[str, object]] = []
    for i in range(n_issue):
        it = start + timedelta(hours=step_h * i)
        for lh in range(1, leads + 1):
            recs.append({"issue_time": pd.Timestamp(it), "lead_hour": lh})
    return pd.DataFrame(recs)


def test_temporal_split_test_is_recent_holdout_calib_disjoint() -> None:
    from microclimate.pipelines.training_data import temporal_split

    rows = _rows(n_issue=4 * 365)
    train, calib, test = temporal_split(rows, holdout_months=3, calib_months=1)

    last = rows["issue_time"].max()
    test_lo = test["issue_time"].min()
    calib_hi, calib_lo = calib["issue_time"].max(), calib["issue_time"].min()
    assert test["issue_time"].max() == last
    assert calib_hi < test_lo
    assert train["issue_time"].max() < calib_lo

    def its(d: pd.DataFrame) -> set[object]:
        return set(d["issue_time"].unique())

    assert its(train).isdisjoint(its(calib)) and its(calib).isdisjoint(its(test))
    assert len(train) and len(calib) and len(test)


def test_temporal_split_raises_when_a_slice_is_empty() -> None:
    from microclimate.pipelines.training_data import temporal_split

    rows = _rows(n_issue=4)
    with pytest.raises(ValueError, match="too little history|empty"):
        temporal_split(rows, holdout_months=3, calib_months=1)
