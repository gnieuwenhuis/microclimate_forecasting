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
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.config.schema import DeploymentConfig, StationRef
from microclimate.connectors.base import NWPSource, ObservationSource, SourceUnavailable
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


def _align_obs_to_lag_grid(
    frame: pd.DataFrame | None,
    station_id: str,
    issue_time: datetime,
    lag_hours: int,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Align one station's OBSERVATION_FRAME onto the fixed hourly lag grid.

    Grid: lag0 = issue_time, lag1 = issue_time-1h, … lag{lag_hours}. Rows are matched by
    exact UTC-hour equality. A slot is absent (value NaN, mask False) when no row exists at
    that hour, the value is null, or the row's <var>_present is False. A None/empty frame
    (degraded source) yields an all-absent grid. issue_time is NOT floored: an off-hour t0
    simply matches no rows.

    Defensive as-of filter: rows with timestamp > issue_time are dropped before matching.
    """
    cutoff = pd.Timestamp(issue_time)
    row_by_ts: dict[pd.Timestamp, int] = {}
    in_window: pd.DataFrame | None = None
    if frame is not None and len(frame) > 0:
        in_window = frame[frame["timestamp"] <= cutoff].reset_index(drop=True)  # type: ignore[reportUnknownMemberType]
        for i, ts in enumerate(in_window["timestamp"]):  # type: ignore[reportUnknownVariableType]
            row_by_ts[pd.Timestamp(ts)] = i  # type: ignore[reportUnknownArgumentType]

    features: dict[str, float] = {}
    masks: dict[str, bool] = {}
    for k in range(lag_hours + 1):
        slot_ts = cutoff - pd.Timedelta(hours=k)
        row_idx = row_by_ts.get(slot_ts)
        for var in _PHYSICAL_VARS:
            key = f"obs_{station_id}_{var}_lag{k}"
            value = float("nan")
            present = False
            if row_idx is not None and in_window is not None:
                is_present = bool(in_window[f"{var}_present"].iloc[row_idx])  # type: ignore[reportUnknownArgumentType]
                raw = in_window[var].iloc[row_idx]  # type: ignore[reportUnknownVariableType]
                if is_present and pd.notna(raw):  # type: ignore[reportUnknownArgumentType]
                    value = float(raw)  # type: ignore[reportUnknownArgumentType]
                    present = True
            features[key] = value
            masks[key] = present
    return features, masks


def build_snapshot(
    config: DeploymentConfig,
    issue_time: datetime,
    nwp: NWPSource,
    observations: Mapping[str, ObservationSource],
) -> FeatureSnapshot:
    """Build the one FeatureSnapshot for `issue_time` (see module docstring / ADR-0011)."""
    # Normalise to UTC: naive datetimes are assumed UTC (the project's "UTC everywhere"
    # convention); aware datetimes are converted, so an aware-but-non-UTC issue_time can't
    # skew the t0 temporal encodings or the UTC-keyed lag-grid lookup.
    issue_utc = (
        issue_time.replace(tzinfo=UTC) if issue_time.tzinfo is None else issue_time.astimezone(UTC)
    )
    lead_hours = tuple(range(1, config.horizon_hours + 1))

    # --- NWP (target cell only) — hard fail on connector errors (they propagate). ---
    nwp_features: dict[str, float] = {}
    if config.feature_groups.nwp:
        frame = nwp.fetch_forecast(issue_utc, config.target.lat, config.target.lon, lead_hours)
        nwp_features = _flatten_forecast(frame)

    # --- Observations — degrade per station; StationNotFound propagates. ---
    obs_features: dict[str, float] = {}
    obs_masks: dict[str, bool] = {}
    if config.feature_groups.observations:
        start = issue_utc - timedelta(hours=config.lag_hours)
        refs: list[StationRef] = [config.target, *config.neighbors]
        for ref in refs:
            try:
                source = observations[ref.connector_key]
            except KeyError as exc:
                raise KeyError(
                    f"No observation source provided for connector_key "
                    f"{ref.connector_key!r} (required by station {ref.station_id!r}). "
                    f"Available: {sorted(observations)}."
                ) from exc
            station_frame: pd.DataFrame | None
            try:
                station_frame = source.fetch_historical(ref.station_id, start, issue_utc)
            except SourceUnavailable:
                # Transient infra failure → degrade this station to all-absent.
                station_frame = None
            feats, masks = _align_obs_to_lag_grid(
                station_frame, ref.station_id, issue_utc, config.lag_hours
            )
            obs_features.update(feats)
            obs_masks.update(masks)

    # --- Static (target only) — NaN elevation when unknown. ---
    elevation = config.target.elevation_m
    static_features: dict[str, float] = {
        "static_lat": float(config.target.lat),
        "static_lon": float(config.target.lon),
        "static_elevation_m": float(elevation) if elevation is not None else float("nan"),
    }

    return FeatureSnapshot(
        deployment_id=config.deployment_id,
        issue_time=issue_utc,
        nwp_features=nwp_features,
        observation_features=obs_features,
        observation_masks=obs_masks,
        static_features=static_features,
        temporal_features=_temporal_features(issue_utc),
        lead_hours=lead_hours,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
    )
