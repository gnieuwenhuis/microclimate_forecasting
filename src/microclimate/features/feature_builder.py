"""The single shared transform: FeatureSnapshot -> feature matrix (L3).

Pure, deterministic, no fitted state, no network. Explodes one FeatureSnapshot (raw,
canonicalized values for one issue time spanning all leads) into long-format rows, one per
(issue_time, lead_hour), with derived features. No labels are attached (ADR-0011 / the
feature-builder spec). Run identically at training-read time and inference; the column set is
deterministic from config, giving train/serve column parity by construction.
"""

from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from microclimate.config.schema import DeploymentConfig
from microclimate.contracts.feature_matrix import FEATURE_SCHEMA_VERSION
from microclimate.contracts.physical_vars import PHYSICAL_VARS
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing in degrees [0, 360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def build_features(snapshot: FeatureSnapshot, config: DeploymentConfig) -> pd.DataFrame:
    """Explode one FeatureSnapshot into the per-(issue_time, lead_hour) feature matrix."""
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"snapshot.schema_version {snapshot.schema_version!r} != expected "
            f"{SNAPSHOT_SCHEMA_VERSION!r}; refusing to build features from an "
            "incompatible snapshot."
        )

    leads = list(snapshot.lead_hours)
    n = len(leads)
    issue = snapshot.issue_time

    # Accumulate all columns in a dict, then construct the DataFrame once at the end.
    # lead_hour (a list) establishes the row count; scalars broadcast automatically.
    cols: dict[str, object] = {"lead_hour": leads}
    cols["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    cols["deployment_id"] = snapshot.deployment_id
    cols["issue_time"] = pd.to_datetime([issue] * n, utc=True)
    cols["valid_time"] = pd.to_datetime([issue + timedelta(hours=h) for h in leads], utc=True)

    # --- NWP (own lead; _h{lead} suffix dropped — lead_hour is a column). ---
    if snapshot.nwp_features:
        nwp = snapshot.nwp_features
        for var in PHYSICAL_VARS:
            cols[f"nwp_{var}"] = [nwp[f"nwp_{var}_h{h}"] for h in leads]

        cols["nwp_dpd"] = [nwp[f"nwp_temp_c_h{h}"] - nwp[f"nwp_dewpoint_c_h{h}"] for h in leads]
        cols["nwp_ptend_3h"] = [
            nwp[f"nwp_surface_pressure_hpa_h{h}"] - nwp[f"nwp_surface_pressure_hpa_h{h - 3}"]
            if h - 3 >= 1
            else math.nan
            for h in leads
        ]

    # --- Observations (passthrough values + masks; scalars broadcast across all lead rows). ---
    if snapshot.observation_features:
        obs = snapshot.observation_features
        masks = snapshot.observation_masks
        for key, value in obs.items():
            cols[key] = value
            cols[f"{key}_mask"] = masks[key]

        station_ids = [config.target.station_id, *[ref.station_id for ref in config.neighbors]]
        for sid in station_ids:
            for k in range(config.lag_hours + 1):
                t = obs.get(f"obs_{sid}_temp_c_lag{k}", math.nan)
                d = obs.get(f"obs_{sid}_dewpoint_c_lag{k}", math.nan)
                cols[f"obs_{sid}_dpd_lag{k}"] = t - d  # scalar broadcast

        tgt = config.target.station_id
        p0 = obs.get(f"obs_{tgt}_surface_pressure_hpa_lag0", math.nan)
        p3 = obs.get(f"obs_{tgt}_surface_pressure_hpa_lag3", math.nan)
        cols[f"obs_{tgt}_ptend_3h"] = p0 - p3
        dpd0 = obs.get(f"obs_{tgt}_temp_c_lag0", math.nan) - obs.get(
            f"obs_{tgt}_dewpoint_c_lag0", math.nan
        )
        dpd3 = obs.get(f"obs_{tgt}_temp_c_lag3", math.nan) - obs.get(
            f"obs_{tgt}_dewpoint_c_lag3", math.nan
        )
        cols[f"obs_{tgt}_dpd_tend_3h"] = dpd0 - dpd3

    # --- Advection (per neighbor): neighbor-target gradients at lag0 + upwind alignment. ---
    if snapshot.observation_features and config.neighbors:
        obs = snapshot.observation_features  # local alias (advection is a separate guarded block)
        tgt = config.target.station_id
        wind_from = obs.get(f"obs_{tgt}_wind_dir_deg_lag0", math.nan)
        wind_speed = obs.get(f"obs_{tgt}_wind_speed_ms_lag0", math.nan)
        t_temp = obs.get(f"obs_{tgt}_temp_c_lag0", math.nan)
        t_precip = obs.get(f"obs_{tgt}_precip_mm_lag0", math.nan)
        t_dpd = t_temp - obs.get(f"obs_{tgt}_dewpoint_c_lag0", math.nan)
        for ref in config.neighbors:
            nid = ref.station_id
            n_temp = obs.get(f"obs_{nid}_temp_c_lag0", math.nan)
            n_precip = obs.get(f"obs_{nid}_precip_mm_lag0", math.nan)
            n_dpd = n_temp - obs.get(f"obs_{nid}_dewpoint_c_lag0", math.nan)
            cols[f"adv_{nid}_temp_grad_lag0"] = n_temp - t_temp
            cols[f"adv_{nid}_dpd_grad_lag0"] = n_dpd - t_dpd
            cols[f"adv_{nid}_precip_grad_lag0"] = n_precip - t_precip
            bearing = _bearing_deg(config.target.lat, config.target.lon, ref.lat, ref.lon)
            cols[f"adv_{nid}_upwind_align"] = (
                math.cos(math.radians(bearing - wind_from)) * wind_speed
            )

    # --- Static (target only; broadcast). ---
    for key, value in snapshot.static_features.items():
        cols[key] = value

    # --- Temporal: t0 passthrough (broadcast) + per-lead valid-time hour encoding. ---
    for key, value in snapshot.temporal_features.items():
        cols[key] = value
    valid_hours = [(issue + timedelta(hours=h)).hour for h in leads]
    cols["valid_hour_sin"] = [math.sin(2 * math.pi * vh / 24.0) for vh in valid_hours]
    cols["valid_hour_cos"] = [math.cos(2 * math.pi * vh / 24.0) for vh in valid_hours]

    df = pd.DataFrame(cols)
    return df
