"""The single, only path that produces a FeatureSnapshot (L3).

As-of / no-leakage: this is the only entry point, it takes issue_time, and the only obs
access is bounded to timestamp <= issue_time. There is no parameter for future data.

build_snapshot is the normalization/IO/as-of boundary: it holds raw, canonicalized values
only. Derived features (dewpoint depression, tendency, advection, per-lead-hour encodings)
and the explode-to-per-lead-hour-rows transform are downstream pure functions of the snapshot
(ADR-0011). Observations are read only via as-of fetch_historical, never fetch_live, so the
obs path is identical for training and inference (the skew guarantee).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot

# Canonical physical variables, fixed order. Match FORECAST_FRAME / OBSERVATION_FRAME.
_PHYSICAL_VARS: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


def _temporal_features(issue_time: datetime) -> dict[str, float]:
    """Cyclical encodings of t0 only (hour-of-day period 24, day-of-year period 365.25).

    Per-lead-hour temporal encodings are built downstream, not here.
    """
    hour = issue_time.hour + issue_time.minute / 60.0
    doy = issue_time.timetuple().tm_yday
    return {
        "t0_hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "t0_hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "t0_doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "t0_doy_cos": math.cos(2 * math.pi * doy / 365.25),
    }


def _flatten_forecast(frame: pd.DataFrame) -> dict[str, float]:
    """FORECAST_FRAME (one row per lead hour) → {nwp_{var}_h{lead}: value}.

    Target-cell forecast values only; no masks (NWP is complete-or-fail).
    """
    out: dict[str, float] = {}
    for _, row in frame.iterrows():  # type: ignore[reportUnknownVariableType]
        lead = int(row["lead_hour"])  # type: ignore[reportUnknownArgumentType]
        for var in _PHYSICAL_VARS:
            out[f"nwp_{var}_h{lead}"] = float(row[var])  # type: ignore[reportUnknownArgumentType]
    return out


def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    raise NotImplementedError
