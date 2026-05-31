# tests/models/test_temp_model.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.models.temp_model import TemperatureRegressor


def _rows(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    lead = rng.integers(1, 49, size=n)
    return pd.DataFrame(
        {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "deployment_id": "test",
            "issue_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "valid_time": pd.Timestamp("2026-05-30", tz="UTC"),
            "lead_hour": lead,
            "nwp_temp_c": x,
            "label_temp_c": 2.0 * x + 1.0,  # learnable signal
        }
    )


def test_fit_predict_returns_aligned_finite_series() -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    preds = model.predict(rows)
    assert len(preds) == len(rows)
    assert preds.index.equals(rows.index)
    assert np.isfinite(preds.to_numpy()).all()
    # learns the signal: beats predicting the mean
    mae_model = (preds - rows["label_temp_c"]).abs().mean()
    mae_mean = (rows["label_temp_c"].mean() - rows["label_temp_c"]).abs().mean()
    assert mae_model < mae_mean


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        TemperatureRegressor().predict(_rows(5))


def test_predict_rejects_mismatched_feature_version() -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    bad = rows.copy()
    bad["feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="feature_schema_version"):
        model.predict(bad)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    before = model.predict(rows)
    path = tmp_path / "temp.joblib"
    model.save(path)
    reloaded = TemperatureRegressor.load(path)
    after = reloaded.predict(rows)
    pd.testing.assert_series_equal(before, after)


def test_empty_rows_raise_clear_error() -> None:
    model = TemperatureRegressor()
    with pytest.raises(ValueError, match="empty"):
        model.fit(_rows(0))
    model.fit(_rows())
    with pytest.raises(ValueError, match="empty"):
        model.predict(_rows(0))


def test_fit_rejects_all_missing_labels() -> None:
    rows = _rows()
    rows["label_temp_c"] = float("nan")
    with pytest.raises(ValueError, match="no rows have a label_temp_c"):
        TemperatureRegressor().fit(rows)


def test_fit_rejects_mixed_feature_versions() -> None:
    rows = _rows()
    rows.loc[rows.index[0], "feature_schema_version"] = "9.9.9"
    with pytest.raises(ValueError, match="mix feature_schema_versions"):
        TemperatureRegressor().fit(rows)


def test_predict_uses_fit_time_columns() -> None:
    rows = _rows()
    model = TemperatureRegressor()
    model.fit(rows)
    # Reordered columns still predict (stored fit-time order is used).
    reordered = rows[rows.columns[::-1]]
    pd.testing.assert_series_equal(model.predict(rows), model.predict(reordered))
    # A dropped feature column fails loud.
    with pytest.raises(KeyError):
        model.predict(rows.drop(columns=["nwp_temp_c"]))
