"""The canonical physical-variable list, fixed order (L0).

Single source of truth shared by FORECAST_FRAME / OBSERVATION_FRAME, snapshot_builder, and
feature_builder, so the train/serve feature-column set cannot silently diverge.
"""

from __future__ import annotations

PHYSICAL_VARS: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)
