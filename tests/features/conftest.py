"""Hermetic fixtures for build_snapshot tests: fake connectors + synthetic frames."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from microclimate.config.schema import (
    DeploymentConfig,
    FeatureGroupSwitches,
    LabelConfig,
    NwpConfig,
    OutputConfig,
    SeedConfig,
    StationRef,
    TrainingConfig,
)
from microclimate.connectors.base import (
    HistoricalCoverage,
    NWPSource,
    ObservationSource,
)
from microclimate.contracts.physical_vars import PHYSICAL_VARS

PHYS = PHYSICAL_VARS

PINNED: dict[str, float] = {
    "temp_c": 15.0,
    "dewpoint_c": 5.0,
    "surface_pressure_hpa": 900.0,
    "precip_mm": 0.5,
    "cloud_cover_fraction": 0.5,
    "solar_radiation_wm2": 300.0,
    "wind_speed_ms": 5.0,
    "wind_dir_deg": 270.0,
}


def make_forecast_frame(issue_time: datetime, lead_hours: Sequence[int]) -> pd.DataFrame:
    """FORECAST_FRAME-shaped frame with PINNED values at every lead."""
    rows: list[dict[str, object]] = []
    for lh in lead_hours:
        row: dict[str, object] = {
            "issue_time": pd.Timestamp(issue_time),
            "lead_hour": int(lh),
            "valid_time": pd.Timestamp(issue_time) + pd.Timedelta(hours=int(lh)),
        }
        for var in PHYS:
            row[var] = PINNED[var]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["issue_time"] = pd.to_datetime(df["issue_time"], utc=True)
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    return df


def make_obs_frame(
    station_id: str,
    timestamps: Sequence[datetime],
    *,
    absent: set[tuple[int, str]] | None = None,
) -> pd.DataFrame:
    """OBSERVATION_FRAME-shaped frame; `absent` marks (row_index, var) NaN/present=False."""
    absent = absent or set()
    data: dict[str, list[object]] = {
        "station_id": [station_id] * len(timestamps),
        "timestamp": list(pd.to_datetime(list(timestamps), utc=True)),
    }
    for var in PHYS:
        vals: list[object] = []
        pres: list[object] = []
        for idx in range(len(timestamps)):
            if (idx, var) in absent:
                vals.append(float("nan"))
                pres.append(False)
            else:
                vals.append(PINNED[var])
                pres.append(True)
        data[var] = vals
        data[f"{var}_present"] = pres
    return pd.DataFrame(data)


class FakeNWP(NWPSource):
    """Injectable NWPSource returning a prebuilt FORECAST_FRAME or raising `exc`."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        exc: Exception | None = None,
        *,
        is_live: bool = True,
    ) -> None:
        self._frame = frame
        self._exc = exc
        self._is_live = is_live

    @property
    def is_live(self) -> bool:
        return self._is_live

    def fetch_forecast(
        self, issue_time: datetime, lat: float, lon: float, lead_hours: Sequence[int]
    ) -> pd.DataFrame:
        if self._exc is not None:
            raise self._exc
        assert self._frame is not None
        return self._frame


class FakeObs(ObservationSource):
    """Injectable ObservationSource. Returns prebuilt frames keyed by station_id, or raises."""

    def __init__(
        self,
        frames: dict[str, pd.DataFrame] | None = None,
        exc: Exception | None = None,
    ) -> None:
        self._frames = frames or {}
        self._exc = exc

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        if self._exc is not None:
            raise self._exc
        return self._frames[station_id]

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        raise NotImplementedError("build_snapshot must never call fetch_live")


def make_config(
    *,
    horizon_hours: int = 3,
    lag_hours: int = 2,
    nwp: bool = True,
    observations: bool = True,
    neighbors: list[StationRef] | None = None,
    connector_key: str = "fake",
) -> DeploymentConfig:
    """Minimal valid DeploymentConfig: 1 target + (default) 1 neighbor, both `connector_key`."""
    if neighbors is None:
        neighbors = [
            StationRef(
                station_id="N1",
                connector_key=connector_key,
                lat=51.5,
                lon=-113.5,
                elevation_m=950.0,
            )
        ]
    return DeploymentConfig(
        deployment_id="test",
        target=StationRef(
            station_id="T1", connector_key=connector_key, lat=51.0, lon=-114.0, elevation_m=900.0
        ),
        neighbors=neighbors,
        enabled_sources=[connector_key],
        nwp=NwpConfig(
            product="hrdps",
            live_connector="x",
            historical_connector="y",
            sampling="nearest_grid_cell",
        ),
        horizon_hours=horizon_hours,
        lag_hours=lag_hours,
        feature_groups=FeatureGroupSwitches(nwp=nwp, observations=observations),
        label=LabelConfig(precip_occurrence_threshold_mm=0.2),
        training=TrainingConfig(
            seed=SeedConfig(source="caspar", start="2017-05-22"), holdout_months=12
        ),
        output=OutputConfig(forecast_json="x.json"),
    )
