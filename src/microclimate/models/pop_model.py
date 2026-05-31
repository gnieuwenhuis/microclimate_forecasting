"""Precipitation-occurrence classifier wrapper with isotonic calibration (L4).

Row-based like the temp model. Calibration (ADR-0004) is required: fit() learns the booster,
calibrate() fits an isotonic map on a disjoint slice, predict() returns the calibrated
probability per row. The fitted calibrator is persisted alongside the booster.
"""

from __future__ import annotations

from pathlib import Path

import joblib  # pyright: ignore[reportMissingTypeStubs]
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # pyright: ignore[reportMissingTypeStubs]

from microclimate.models._columns import feature_columns, single_feature_schema_version


class PrecipOccurrenceClassifier:
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._model: lgb.LGBMClassifier | None = None
        self._calibrator: IsotonicRegression | None = None
        self._features: list[str] | None = None
        self._feature_schema_version: str | None = None

    def fit(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            raise ValueError("rows is empty; nothing to fit")
        version = single_feature_schema_version(rows)
        labeled = rows.dropna(subset=["label_precip_occurrence"])
        if labeled.empty:
            raise ValueError("no rows have a label_precip_occurrence; nothing to fit")
        y = labeled["label_precip_occurrence"].astype(int)
        if y.nunique() < 2:
            raise ValueError(
                "label_precip_occurrence has a single class; need both 0 and 1 to fit a "
                "classifier (widen the training window)."
            )
        feats = feature_columns(labeled)
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
        )
        model.fit(labeled[feats], y)  # pyright: ignore[reportUnknownMemberType]
        self._model = model
        self._features = feats
        self._feature_schema_version = version
        self._calibrator = None

    def calibrate(self, rows: pd.DataFrame) -> None:
        raw = self._raw_proba(rows)
        labeled_idx = rows["label_precip_occurrence"].notna()
        y = rows.loc[labeled_idx, "label_precip_occurrence"].astype(int)
        if y.empty:
            raise ValueError("no rows have a label_precip_occurrence; nothing to calibrate")
        if y.nunique() < 2:
            raise ValueError(
                "calibration slice has a single class; isotonic calibration needs both 0 and "
                "1 (widen the calibration slice)."
            )
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw[labeled_idx.to_numpy()], y)  # pyright: ignore[reportUnknownMemberType]
        self._calibrator = calibrator

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self._calibrator is None:
            raise RuntimeError("call calibrate() before predict()")
        raw = self._raw_proba(rows)
        calibrated: np.ndarray = np.asarray(
            self._calibrator.predict(raw)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        return pd.Series(np.clip(calibrated, 0.0, 1.0), index=rows.index, name="pred_pop")

    def _raw_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self._model is None or self._features is None:
            raise RuntimeError("call fit() before calibrate()/predict()")
        if rows.empty:
            raise ValueError("rows is empty; nothing to predict")
        got = str(rows["feature_schema_version"].iloc[0])
        if got != self._feature_schema_version:
            raise ValueError(
                f"rows feature_schema_version {got!r} != model's "
                f"{self._feature_schema_version!r}; refusing to predict."
            )
        proba = np.asarray(
            self._model.predict_proba(rows[self._features])  # pyright: ignore[reportUnknownMemberType]
        )
        return proba[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(  # pyright: ignore[reportUnknownMemberType]
            {
                "model": self._model,
                "calibrator": self._calibrator,
                "features": self._features,
                "feature_schema_version": self._feature_schema_version,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> PrecipOccurrenceClassifier:
        state = joblib.load(path)  # pyright: ignore[reportUnknownMemberType]
        obj = cls()
        obj._model = state["model"]
        obj._calibrator = state["calibrator"]
        obj._features = state["features"]
        obj._feature_schema_version = state["feature_schema_version"]
        return obj
