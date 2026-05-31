"""The canonical physical-variable list, fixed order (L0).

The single source of truth for the feature path: snapshot_builder (which flattens these into
the snapshot) and feature_builder (which reads them back) both import this, so the train/serve
feature-column set cannot silently diverge. The FORECAST_FRAME / OBSERVATION_FRAME contract
schemas and the connector readers (e.g. envcanada `_PHYS_VARS`) still declare their own column
lists — they carry per-column dtype/range checks, not just names; unifying those onto this
constant is a separate, larger refactor.
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
