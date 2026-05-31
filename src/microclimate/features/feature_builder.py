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
from microclimate.contracts.snapshot import SNAPSHOT_SCHEMA_VERSION, FeatureSnapshot

# Canonical physical variables, fixed order — must match snapshot_builder._PHYSICAL_VARS.
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


def build_features(snapshot: FeatureSnapshot, config: DeploymentConfig) -> pd.DataFrame:
    """Explode one FeatureSnapshot into the per-(issue_time, lead_hour) feature matrix."""
    if snapshot.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"snapshot.schema_version {snapshot.schema_version!r} != expected "
            f"{SNAPSHOT_SCHEMA_VERSION!r}; refusing to build features from an incompatible snapshot."
        )

    leads = list(snapshot.lead_hours)
    n = len(leads)
    issue = snapshot.issue_time

    # Idiomatic pandas: lead_hour establishes the row count, then scalars broadcast and
    # per-lead lists assign directly. Avoids a mixed-type dict (pyright invariance) entirely.
    df = pd.DataFrame({"lead_hour": leads})
    df["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    df["deployment_id"] = snapshot.deployment_id
    df["issue_time"] = pd.to_datetime([issue] * n, utc=True)
    df["valid_time"] = pd.to_datetime([issue + timedelta(hours=h) for h in leads], utc=True)

    # --- NWP (own lead; _h{lead} suffix dropped — lead_hour is a column). ---
    if snapshot.nwp_features:
        nwp = snapshot.nwp_features
        for var in _PHYSICAL_VARS:
            df[f"nwp_{var}"] = [nwp[f"nwp_{var}_h{h}"] for h in leads]

        df["nwp_dpd"] = [nwp[f"nwp_temp_c_h{h}"] - nwp[f"nwp_dewpoint_c_h{h}"] for h in leads]
        df["nwp_ptend_3h"] = [
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
            df[key] = value
            df[f"{key}_mask"] = masks[key]

    return df
