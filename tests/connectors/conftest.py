"""Shared test helpers for connector tests."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

import numpy as np
import xarray as xr

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "envcanada"

# ---------------------------------------------------------------------------
# var_map: canonical name → HRDPS-ish dataset variable name
# Exposed here so both test_nwp_core.py and future connector tests share it.
# ---------------------------------------------------------------------------

VAR_MAP: Mapping[str, str] = {
    "temp_c": "t2m",
    "dewpoint_c": "d2m",
    "surface_pressure_hpa": "sp",
    "precip_mm": "tp",
    "cloud_cover_fraction": "tcc",
    "solar_radiation_wm2": "dswrf",
    "wind_speed_ms": "si10",
    "wind_dir_deg": "wdir10",
}


def build_hrdps_dataset() -> xr.Dataset:
    """Build a small synthetic xr.Dataset that matches the nwp_core Dataset contract.

    Grid layout (y=2, x=2):
        Cell (iy=0, ix=0): lat=51.0, lon=-114.0  ← TARGET cell
        Cell (iy=0, ix=1): lat=51.0, lon=-113.0
        Cell (iy=1, ix=0): lat=52.0, lon=-114.0
        Cell (iy=1, ix=1): lat=52.0, lon=-113.0  ← alternate target

    The target cell (0,0) has KNOWN values so unit-conversion tests are predictable:
        t2m   = 288.15 K  → 15.0 °C after conversion
        d2m   = 278.15 K  → 5.0 °C after conversion
        sp    = 90000 Pa  → 900.0 hPa after conversion
        tp    = [0.0, 0.5, 2.0, 2.0]  accumulated kg/m²
                → per-hour at leads [1,2,3]: [0.5, 1.5, 0.0]
        tcc   = 50 %      → 0.5 after conversion
        dswrf = 300 W/m²  → 300.0 (pass-through)
        si10  = 5.0 m/s   → 5.0 (pass-through)
        wdir10 = 270 deg  → 270.0 (pass-through)

    Alternate cell (1,1) has distinct values (e.g. temp=293.15 K → 20.0 °C) so a
    wrong-cell selection causes test failures.

    lead_hour coordinate: [0, 1, 2, 3] (ascending, includes hour-before-first-requested).
    """
    lead_hours = np.array([0, 1, 2, 3], dtype=np.int64)
    n_lh = len(lead_hours)

    # 2-D lat/lon arrays over dims (y, x)
    lat_data = np.array([[51.0, 51.0], [52.0, 52.0]], dtype=np.float64)
    lon_data = np.array([[-114.0, -113.0], [-114.0, -113.0]], dtype=np.float64)

    dims_lh_yx = ("lead_hour", "y", "x")
    dims_yx = ("y", "x")

    # -----------------------------------------------------------------------
    # Per-variable data: shape (lead_hour, y, x)
    # Values at (0,0) match the KNOWN expectations above; other cells differ.
    # -----------------------------------------------------------------------

    def _fill(target_val: float, other_val: float) -> np.ndarray:
        """Fill array: target cell (0,0)=target_val, all others=other_val."""
        arr = np.full((n_lh, 2, 2), other_val, dtype=np.float64)
        arr[:, 0, 0] = target_val
        return arr

    # Temperature (K): 288.15 @ target, 293.15 @ others
    t2m_data = _fill(288.15, 293.15)

    # Dewpoint (K): 278.15 @ target, 283.15 @ others
    d2m_data = _fill(278.15, 283.15)

    # Surface pressure (Pa): 90000 @ target, 95000 @ others
    sp_data = _fill(90000.0, 95000.0)

    # Accumulated precip (kg/m²): target=[0,0.5,2.0,2.0], others=0.0
    tp_data = np.zeros((n_lh, 2, 2), dtype=np.float64)
    tp_data[:, 0, 0] = [0.0, 0.5, 2.0, 2.0]
    tp_data[:, 0, 1] = [0.0, 1.0, 1.0, 1.0]  # distinct from target
    tp_data[:, 1, 0] = [0.0, 2.0, 2.0, 2.0]  # distinct from target
    tp_data[:, 1, 1] = [0.0, 3.0, 3.5, 4.0]  # distinct from target

    # Cloud cover (%): 50 @ target, 75 @ others
    tcc_data = _fill(50.0, 75.0)

    # Solar radiation (W/m²): 300 @ target, 400 @ others
    dswrf_data = _fill(300.0, 400.0)

    # Wind speed (m/s): 5.0 @ target, 8.0 @ others
    si10_data = _fill(5.0, 8.0)

    # Wind direction (deg): 270 @ target, 180 @ others
    wdir10_data = _fill(270.0, 180.0)

    coords: dict[str, object] = {
        "lead_hour": lead_hours,
        "latitude": xr.DataArray(lat_data, dims=dims_yx),
        "longitude": xr.DataArray(lon_data, dims=dims_yx),
    }

    ds = xr.Dataset(
        {
            "t2m": xr.DataArray(t2m_data, dims=dims_lh_yx),
            "d2m": xr.DataArray(d2m_data, dims=dims_lh_yx),
            "sp": xr.DataArray(sp_data, dims=dims_lh_yx),
            "tp": xr.DataArray(tp_data, dims=dims_lh_yx),
            "tcc": xr.DataArray(tcc_data, dims=dims_lh_yx),
            "dswrf": xr.DataArray(dswrf_data, dims=dims_lh_yx),
            "si10": xr.DataArray(si10_data, dims=dims_lh_yx),
            "wdir10": xr.DataArray(wdir10_data, dims=dims_lh_yx),
        },
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
