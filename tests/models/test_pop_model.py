# tests/models/test_pop_model.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models.pop_model import PrecipOccurrenceClassifier


def _rows(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    prob = 1.0 / (1.0 + np.exp(-x))
    y = (rng.uniform(size=n) < prob).astype(int)  # both classes present
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "valid_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "lead_hour": rng.integers(1, 49, size=n),
            "nwp_precip_mm": x,
            "label_precip_occurrence": y,
        }
    )


def test_fit_calibrate_predict_in_unit_interval() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    preds = model.predict(rows)
    assert len(preds) == len(rows)
    assert preds.index.equals(rows.index)
    arr = preds.to_numpy()
    assert ((arr >= 0.0) & (arr <= 1.0)).all()


def test_predict_requires_calibration() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict(rows)


def test_predict_rejects_mismatched_feature_version() -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    bad = rows.copy()
    bad["feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="feature_schema_version"):
        model.predict(bad)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rows = _rows()
    model = PrecipOccurrenceClassifier()
    model.fit(rows)
    model.calibrate(rows)
    before = model.predict(rows)
    path = tmp_path / "pop.joblib"
    model.save(path)
    after = PrecipOccurrenceClassifier.load(path).predict(rows)
    pd.testing.assert_series_equal(before, after)


def test_empty_rows_raise_clear_error() -> None:
    model = PrecipOccurrenceClassifier()
    with pytest.raises(ValueError, match="empty"):
        model.fit(_rows(0))
    model.fit(_rows())
    with pytest.raises(ValueError, match="empty"):
        model.calibrate(_rows(0))


def test_fit_rejects_single_class() -> None:
    rows = _rows()
    rows["label_precip_occurrence"] = 0  # all one class
    with pytest.raises(ValueError, match="single class"):
        PrecipOccurrenceClassifier().fit(rows)


def test_calibrate_rejects_single_class_slice() -> None:
    model = PrecipOccurrenceClassifier()
    model.fit(_rows())
    calib = _rows()
    calib["label_precip_occurrence"] = 1  # calib slice is single-class
    with pytest.raises(ValueError, match="single class"):
        model.calibrate(calib)


def test_fit_rejects_mixed_feature_versions() -> None:
    rows = _rows()
    rows.loc[rows.index[0], "feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="mix feature_schema_versions"):
        PrecipOccurrenceClassifier().fit(rows)
