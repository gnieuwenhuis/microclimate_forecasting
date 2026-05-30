from __future__ import annotations

import pandas as pd
import pytest

from microclimate.models.pop_model import PrecipOccurrenceClassifier
from microclimate.models.temp_model import TemperatureRegressor


def test_temp_fit_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        TemperatureRegressor().fit(pd.DataFrame())


def test_pop_calibrate_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        PrecipOccurrenceClassifier().calibrate(pd.DataFrame())
