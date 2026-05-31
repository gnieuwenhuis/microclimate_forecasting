"""Temperature regressor wrapper (L4). Row-based: fit/predict over the feature matrix.

lead_hour is a feature (ADR-0004). predict takes feature-matrix rows and returns one
prediction per (issue_time, lead_hour); the inference pipeline owns build_features and
reshapes per-row predictions into the published {lead_hour: value} forecast (ADR-0012).
"""

from __future__ import annotations

from pathlib import Path

import joblib  # pyright: ignore[reportMissingTypeStubs]
import lightgbm as lgb
import pandas as pd

from microclimate.models._columns import feature_columns, single_feature_schema_version


class TemperatureRegressor:
    version: str = "0.1.0"

    def __init__(self) -> None:
        self._model: lgb.LGBMRegressor | None = None
        self._features: list[str] | None = None
        self._feature_schema_version: str | None = None

    def fit(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            raise ValueError("rows is empty; nothing to fit")
        version = single_feature_schema_version(rows)
        labeled = rows.dropna(subset=["label_temp_c"])
        if labeled.empty:
            raise ValueError("no rows have a label_temp_c; nothing to fit")
        feats = feature_columns(labeled)
        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=0, verbose=-1
        )
        model.fit(labeled[feats], labeled["label_temp_c"])  # pyright: ignore[reportUnknownMemberType]
        self._model = model
        self._features = feats
        self._feature_schema_version = version

    def predict(self, rows: pd.DataFrame) -> pd.Series:
        if self._model is None or self._features is None:
            raise RuntimeError("call fit() before predict()")
        if rows.empty:
            raise ValueError("rows is empty; nothing to predict")
        got = str(rows["feature_schema_version"].iloc[0])
        if got != self._feature_schema_version:
            raise ValueError(
                f"rows feature_schema_version {got!r} != model's "
                f"{self._feature_schema_version!r}; refusing to predict."
            )
        preds = self._model.predict(rows[self._features])  # pyright: ignore[reportUnknownMemberType]
        return pd.Series(preds, index=rows.index, name="pred_temp_c")  # pyright: ignore[reportCallIssue, reportArgumentType]

    def save(self, path: Path) -> None:
        joblib.dump(  # pyright: ignore[reportUnknownMemberType]
            {
                "model": self._model,
                "features": self._features,
                "feature_schema_version": self._feature_schema_version,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> TemperatureRegressor:
        state = joblib.load(path)  # pyright: ignore[reportUnknownMemberType]
        obj = cls()
        obj._model = state["model"]
        obj._features = state["features"]
        obj._feature_schema_version = state["feature_schema_version"]
        return obj
