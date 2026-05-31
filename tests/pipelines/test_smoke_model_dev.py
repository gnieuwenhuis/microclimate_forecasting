"""Fast smoke of the model-dev path: assemble -> split -> fit -> predict -> metrics.

Uses fake sources (no network). Exercises the SAME shared functions the notebook calls, so
notebook bitrot surfaces here without executing the .ipynb.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource
from microclimate.evaluation.metrics import (
    nwp_pop_baseline,
    pop_skill_by_lead,
    reliability_table,
    temp_skill_by_lead,
)
from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor
from microclimate.pipelines.training_data import assemble_training_rows, chronological_split
from tests.fakes import PHYS, PINNED, FakeNWP, make_config, make_forecast_frame

_T0 = datetime(2026, 5, 30, 0, 0, tzinfo=UTC)
_LEADS = [1, 2, 3]


def _varying_obs_frame(station_id: str, timestamps: list[datetime]) -> pd.DataFrame:
    """OBSERVATION_FRAME with precip alternating by hour parity so PoP has both classes."""
    ts = pd.to_datetime(timestamps, utc=True)
    data: dict[str, object] = {"station_id": [station_id] * len(ts), "timestamp": list(ts)}
    for var in PHYS:
        data[var] = [PINNED[var]] * len(ts)
        data[f"{var}_present"] = [True] * len(ts)
    data["precip_mm"] = [0.5 if t.hour % 2 == 0 else 0.0 for t in ts]
    return pd.DataFrame(data)


class _VaryingObs(ObservationSource):
    def __init__(self, station_ids: list[str], timestamps: list[datetime]) -> None:
        self._frames = {s: _varying_obs_frame(s, timestamps) for s in station_ids}

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._frames[station_id]

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError


def test_model_dev_path_runs_end_to_end() -> None:
    config = make_config(horizon_hours=3, lag_hours=2)
    issue_times = [_T0 + timedelta(hours=i) for i in range(40)]
    span = [_T0 - timedelta(hours=2) + timedelta(hours=i) for i in range(40 + 3 + 3)]
    obs = {"fake": _VaryingObs(["T1", "N1"], span)}
    nwp = FakeNWP(make_forecast_frame(_T0, _LEADS))

    rows = assemble_training_rows(config, nwp, obs, issue_times)
    assert len(rows) == 40 * 3

    train, calib, test = chronological_split(rows)
    assert len(train) and len(calib) and len(test)

    temp = TemperatureRegressor()
    temp.fit(pd.concat([train, calib], ignore_index=True))
    test = test.copy()
    test["pred_temp_c"] = temp.predict(test).to_numpy()

    pop = PrecipOccurrenceClassifier()
    pop.fit(train)
    pop.calibrate(calib)
    test["pred_pop"] = pop.predict(test).to_numpy()
    test["baseline_pop"] = nwp_pop_baseline(
        test, config.label.precip_occurrence_threshold_mm
    ).to_numpy()

    temp_skill = temp_skill_by_lead(test)
    pop_skill = pop_skill_by_lead(test)
    rel = reliability_table(test)

    assert {"lead_hour", "rmse", "skill"}.issubset(temp_skill.columns)
    assert {"lead_hour", "brier", "bss"}.issubset(pop_skill.columns)
    assert len(rel) == 10
    assert np.isfinite(test["pred_temp_c"].to_numpy()).all()
    assert ((test["pred_pop"] >= 0) & (test["pred_pop"] <= 1)).all()
