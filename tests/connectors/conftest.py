"""Shared test helpers for connector tests."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

# Identity var_map: the connector produces canonical-named dataset variables and
# hands nwp_core an identity map (no ECMWF-shortName indirection).  The synthetic
# fixture mirrors that — its data vars are named by these canonical keys.
VAR_MAP: dict[str, str] = {
    "temp_c": "temp_c",
    "dewpoint_c": "dewpoint_c",
    "surface_pressure_hpa": "surface_pressure_hpa",
    "precip_mm": "precip_mm",
    "cloud_cover_fraction": "cloud_cover_fraction",
    "solar_radiation_wm2": "solar_radiation_wm2",
    "wind_speed_ms": "wind_speed_ms",
    "wind_dir_deg": "wind_dir_deg",
}

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "envcanada"


def build_hrdps_dataset(
    *,
    grid_size: tuple[int, int] = (2, 2),
    lead_hours: Sequence[int] = (0, 1, 2, 3),
    var_map: Mapping[str, str] = VAR_MAP,
) -> xr.Dataset:
    """Build a small synthetic xr.Dataset that matches the nwp_core Dataset contract.

    Args:
        grid_size:   (ny, nx) shape of the 2-D lat/lon grid.  Default (2, 2)
                     gives the fixed 4-cell grid used by all existing tests.
        lead_hours:  Lead hours to include as a coordinate.  Default
                     ``(0, 1, 2, 3)`` covers the original fixture; Task-B
                     (issue-6) can pass more hours without duplicating the
                     fixture builder.

    Grid layout (default grid_size=(2,2)):
        Cell (iy=0, ix=0): lat=51.0, lon=-114.0  ← TARGET cell
        Cell (iy=0, ix=1): lat=51.0, lon=-113.0
        Cell (iy=1, ix=0): lat=52.0, lon=-114.0
        Cell (iy=1, ix=1): lat=52.0, lon=-113.0  ← alternate target

    Data variables are named by their CANONICAL column names (the connector
    produces canonical-named vars and hands nwp_core an identity map).

    The target cell (0,0) has KNOWN values so unit-conversion tests are predictable:
        temp_c                = 288.15 K  → 15.0 °C after conversion
        dewpoint_c            = 278.15 K  → 5.0 °C after conversion
        surface_pressure_hpa  = 90000 Pa  → 900.0 hPa after conversion
        precip_mm  (accum)    = [0.0, 0.5, 2.0, 2.0, …]  accumulated kg/m²
                                → per-hour at leads [1,2,3]: [0.5, 1.5, 0.0]
        cloud_cover_fraction  = 50 %      → 0.5 after conversion
        solar_radiation_wm2 (accum J/m²) = [0.0, 3_600_000, 7_200_000, 7_200_000]
                                → per-hour mean flux at leads [1,2,3]: [1000.0, 1000.0, 0.0] W/m²
        wind_speed_ms         = 5.0 m/s   → 5.0 (pass-through)
        wind_dir_deg          = 270 deg   → 270.0 (pass-through)

    Alternate cell (1,1) has distinct values (e.g. temp=293.15 K → 20.0 °C) so a
    wrong-cell selection causes test failures.
    """
    ny, nx = grid_size
    if ny < 2 or nx < 2:
        raise ValueError("grid_size must be at least (2, 2) for the alternate-cell tests.")

    lead_hours_arr = np.array(list(lead_hours), dtype=np.int64)
    n_lh = len(lead_hours_arr)

    # 2-D lat/lon arrays over dims (y, x): first row lat=51, second lat=52;
    # first col lon=-114, second col lon=-113.  Extra cells follow the same
    # stride so cell (0,0) always has the canonical (51, -114) position.
    lat_vals = np.array([51.0 + i for i in range(ny)], dtype=np.float64)
    lon_vals = np.array([-114.0 + j for j in range(nx)], dtype=np.float64)
    lat_data = np.tile(lat_vals[:, np.newaxis], (1, nx))
    lon_data = np.tile(lon_vals[np.newaxis, :], (ny, 1))

    dims_lh_yx = ("lead_hour", "y", "x")
    dims_yx = ("y", "x")

    # -----------------------------------------------------------------------
    # Per-variable data: shape (lead_hour, y, x)
    # Values at (0,0) match the KNOWN expectations above; other cells differ.
    # -----------------------------------------------------------------------

    def _fill(target_val: float, other_val: float) -> np.ndarray:
        """Fill array: target cell (0,0)=target_val, all others=other_val."""
        arr = np.full((n_lh, ny, nx), other_val, dtype=np.float64)
        arr[:, 0, 0] = target_val
        return arr

    # Temperature (K): 288.15 @ target, 293.15 @ others
    t2m_data = _fill(288.15, 293.15)

    # Dewpoint (K): 278.15 @ target, 283.15 @ others
    d2m_data = _fill(278.15, 283.15)

    # Surface pressure (Pa): 90000 @ target, 95000 @ others
    sp_data = _fill(90000.0, 95000.0)

    # Accumulated precip (kg/m²): target=[0,0.5,2.0,2.0,…], others=0.0.
    # The target cell gets a linearly increasing accumulation so that any
    # number of lead hours yields a positive de-accumulated value; the
    # standard first-four values are pinned for backward compatibility.
    tp_data = np.zeros((n_lh, ny, nx), dtype=np.float64)
    pinned = [0.0, 0.5, 2.0, 2.0]
    for lh_i in range(n_lh):
        tp_data[lh_i, 0, 0] = pinned[lh_i] if lh_i < len(pinned) else float(lh_i)
    # Distinct values for other corner cells (2×2 minimum guaranteed above)
    tp_data[:, 0, 1] = [float(i) for i in range(n_lh)]  # distinct from target
    tp_data[:, 1, 0] = [float(i * 2) for i in range(n_lh)]  # distinct from target
    tp_data[:, 1, 1] = [float(i) * 0.5 + float(i) * 0.5 * i for i in range(n_lh)]  # distinct

    # Cloud cover (%): 50 @ target, 75 @ others
    tcc_data = _fill(50.0, 75.0)

    # Accumulated downward shortwave (J/m²): run-total at the target cell so it
    # de-accumulates (÷3600) to a known per-hour mean flux.  The standard
    # first-four values are pinned: [0.0, 3_600_000, 7_200_000, 7_200_000]
    # → per-hour mean flux at leads [1,2,3]: [1000.0, 1000.0, 0.0] W/m².  Other
    # cells get distinct accumulated values so wrong-cell selection fails.
    solar_pinned = [0.0, 3_600_000.0, 7_200_000.0, 7_200_000.0]
    dswrf_data = np.full((n_lh, ny, nx), 1_000_000.0, dtype=np.float64)
    for lh_i in range(n_lh):
        dswrf_data[lh_i, 0, 0] = (
            solar_pinned[lh_i] if lh_i < len(solar_pinned) else float(lh_i) * 3_600_000.0
        )
    # Distinct accumulated ramps for the other corner cells.
    dswrf_data[:, 0, 1] = [float(i) * 1_800_000.0 for i in range(n_lh)]
    dswrf_data[:, 1, 0] = [float(i) * 900_000.0 for i in range(n_lh)]
    dswrf_data[:, 1, 1] = [float(i) * 450_000.0 for i in range(n_lh)]

    # Wind speed (m/s): 5.0 @ target, 8.0 @ others
    si10_data = _fill(5.0, 8.0)

    # Wind direction (deg): 270 @ target, 180 @ others
    wdir10_data = _fill(270.0, 180.0)

    coords: dict[str, object] = {
        "lead_hour": lead_hours_arr,
        "latitude": xr.DataArray(lat_data, dims=dims_yx),
        "longitude": xr.DataArray(lon_data, dims=dims_yx),
    }

    # Map each per-variable numpy array to the dataset-variable name that
    # var_map assigns (the identity map → canonical names).  Building the dict
    # from var_map ensures this fixture never silently drifts from the mapping.
    _data_by_var_name: dict[str, np.ndarray] = {
        var_map["temp_c"]: t2m_data,
        var_map["dewpoint_c"]: d2m_data,
        var_map["surface_pressure_hpa"]: sp_data,
        var_map["precip_mm"]: tp_data,
        var_map["cloud_cover_fraction"]: tcc_data,
        var_map["solar_radiation_wm2"]: dswrf_data,
        var_map["wind_speed_ms"]: si10_data,
        var_map["wind_dir_deg"]: wdir10_data,
    }

    ds = xr.Dataset(
        {k: xr.DataArray(v, dims=dims_lh_yx) for k, v in _data_by_var_name.items()},
        coords=coords,
    )
    return ds


# ---------------------------------------------------------------------------
# EnvCanada helpers (pre-existing)
# ---------------------------------------------------------------------------


def load_fixture(name: str) -> str:
    """Read a fixture file from the envcanada fixture directory."""
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8-sig")


def make_fetcher(csv_text: str):
    """Return a fetcher callable that always returns the given CSV text."""

    def fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        return csv_text

    return fetcher
