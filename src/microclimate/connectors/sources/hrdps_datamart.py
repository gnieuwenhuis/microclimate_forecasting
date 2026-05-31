"""HRDPS via MSC Datamart — live NWP source (GRIB2 seam/core split).

Architecture:
  Seam  — ``_open_latest_run(issue_time, lead_hours)`` builds MSC Datamart GRIB2
           URLs, downloads each file via ``http_get_bytes``, writes to temp files,
           opens with cfgrib, and normalises to the nwp_core Dataset contract.
           ALL network/cfgrib I/O is confined here.  cfgrib is imported LAZILY
           inside this function (eccodes binary unavailable in CI/local envs).
  Core  — ``dataset_to_forecast_frame`` (in nwp_core) is a pure xarray/pandas
           function that consumes the normalised Dataset.  No network; no cfgrib.

Injectable opener:
  ``HrdpsDatamartSource(opener=...)`` accepts an optional
  ``Callable[[datetime, Sequence[int]], xr.Dataset]`` for hermetic unit testing.
  The default (``opener=None``) falls through to the real seam.  The registry
  calls ``HrdpsDatamartSource()`` with no args, so the default MUST work
  argument-free.

IMPORTANT — seam caveats (unverified against live data):
  * The Datamart GRIB2 URL pattern used in ``_open_latest_run`` is a best-effort
    reconstruction from public MSC documentation.  The exact path components
    (sub-directory names, filename conventions, product codes) MUST be verified
    against a live Datamart mirror before relying on the default seam in
    production.
  * The GRIB shortNames assumed in ``HRDPS_VAR_MAP`` match the synthetic dataset
    in the test fixtures (``build_hrdps_dataset``).  Whether real HRDPS GRIB2
    files encode these variables under the same shortNames (vs GRIB2 paramId /
    eccodes table) has NOT been verified against live data.
  * cfgrib's ``filter_by_keys`` arguments used in ``_open_latest_run`` may need
    tuning for the actual file layout of Datamart HRDPS continental products.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import xarray as xr

from microclimate.connectors.base import ForecastUnavailable, NWPSource, SourceUnavailable
from microclimate.connectors.http import http_get_bytes
from microclimate.connectors.nwp_core import dataset_to_forecast_frame
from microclimate.connectors.registry import register_source

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Datamart base URL for HRDPS continental 2.5 km products.
# NOTE: This URL pattern is unverified — see module docstring.
_DATAMART_BASE: str = "https://dd.weather.gc.ca/model_hrdps/continental/2.5km"

# Mapping: canonical column name → HRDPS GRIB shortName.
# Exported as a public name so tests can import it directly and stay in sync.
# NOTE: Real HRDPS GRIB shortNames are unverified — see module docstring.
HRDPS_VAR_MAP: dict[str, str] = {
    "temp_c": "t2m",
    "dewpoint_c": "d2m",
    "surface_pressure_hpa": "sp",
    "precip_mm": "tp",
    "cloud_cover_fraction": "tcc",
    "solar_radiation_wm2": "dswrf",
    "wind_speed_ms": "si10",
    "wind_dir_deg": "wdir10",
}

# Private alias retained for readability within this module.
_HRDPS_VAR_MAP: dict[str, str] = HRDPS_VAR_MAP

# GRIB shortNames that cfgrib needs to load from individual GRIB2 files.
_GRIB_SHORT_NAMES: tuple[str, ...] = tuple(_HRDPS_VAR_MAP.values())


# ---------------------------------------------------------------------------
# Seam — network + cfgrib I/O lives here exclusively
# ---------------------------------------------------------------------------


def _build_datamart_url(issue_time: datetime, lead_hour: int, short_name: str) -> str:
    """Build a best-effort MSC Datamart HRDPS continental GRIB2 URL.

    IMPORTANT: This URL pattern is unverified against live Datamart data.
    The actual path structure, filename convention, and product codes must be
    confirmed before using in production.  See module docstring.

    Args:
        issue_time: UTC model run initialisation time.
        lead_hour:  Forecast lead hour (0-based from run start).
        short_name: GRIB shortName for the variable (e.g. ``"t2m"``).

    Returns:
        Best-effort absolute URL string.
    """
    run_str = issue_time.strftime("%Y%m%dT%HZ")
    hh = f"{lead_hour:03d}"
    # Best-effort filename pattern:
    # CMC_hrdps_continental_<shortName>_SFC_0_ps2.5km_YYYYMMDDTHHUTZ_Phhh.grib2
    filename = f"CMC_hrdps_continental_{short_name}_SFC_0_ps2.5km_{run_str}_P{hh}.grib2"
    return f"{_DATAMART_BASE}/{run_str}/{hh}/{filename}"


def _open_latest_run(
    issue_time: datetime,
    lead_hours: Sequence[int],
) -> xr.Dataset:
    """Download HRDPS GRIB2 files from MSC Datamart and return a normalised Dataset.

    This is the network/cfgrib seam — it is NOT hermetically testable here because
    eccodes binary is unavailable and network access is not guaranteed.  All
    correctness testing is done via the injectable ``opener`` in
    ``HrdpsDatamartSource``.

    The Dataset returned conforms to the nwp_core contract:
        - 1-D integer ``lead_hour`` coordinate (ascending, includes the hour
          before the smallest requested hour for precip de-accumulation).
        - 2-D ``latitude`` / ``longitude`` coordinates over dims (y, x).
        - Data variables named by HRDPS GRIB shortNames (see ``HRDPS_VAR_MAP``).

    IMPORTANT: URL patterns, cfgrib filter_by_keys arguments, and shortName
    mappings are unverified against live Datamart data.  See module docstring.

    Args:
        issue_time:  UTC model run initialisation time.
        lead_hours:  Requested forecast lead hours.

    Returns:
        Normalised xr.Dataset matching the nwp_core Dataset contract, fully
        loaded into memory.

    Raises:
        SourceUnavailable: If cfgrib/eccodes is not importable, if a GRIB2
            download fails (HTTP/network error), or if a disk I/O error occurs
            while writing temp files.
        ForecastUnavailable: If the run is absent, truncated, or GRIB2 decoding
            fails (e.g. missing expected variables, corrupt data, cannot
            normalise to Dataset contract).
    """
    # cfgrib is imported lazily: eccodes binary may be absent at import time.
    try:
        import cfgrib  # type: ignore[reportMissingTypeStubs]
    except ImportError as exc:
        raise SourceUnavailable("cfgrib is not importable (eccodes binary unavailable)") from exc

    # Build the full set of lead hours (include h-1 for precip de-accumulation).
    min_lh = min(lead_hours)
    all_lead_hours = sorted({max(0, min_lh - 1), *lead_hours})

    # Download one GRIB2 file per (lead_hour, shortName) combination.
    # Each shortName lives in its own Datamart file.
    # Results collected as {lead_hour: {short_name: xr.DataArray}}.
    # All temp files are written inside a single TemporaryDirectory that is
    # automatically removed (along with its contents) when the context exits.
    # After combining into the final Dataset we call .load() to materialise the
    # data into memory BEFORE the directory is deleted — cfgrib reads lazily.
    with tempfile.TemporaryDirectory() as tmpdir:
        per_lh: dict[int, dict[str, xr.DataArray]] = {}

        for lh in all_lead_hours:
            per_lh[lh] = {}
            for short_name in _GRIB_SHORT_NAMES:
                url = _build_datamart_url(issue_time, lh, short_name)
                # SourceUnavailable from http_get_bytes propagates unchanged.
                data_bytes = http_get_bytes(url)

                # Write bytes to a temp file so cfgrib can read it.
                tmp_path = f"{tmpdir}/{lh}_{short_name}.grib2"
                try:
                    with open(tmp_path, "wb") as fh:
                        fh.write(data_bytes)
                except OSError as exc:
                    raise SourceUnavailable(
                        f"Disk I/O error writing temp GRIB2 for shortName={short_name!r}, "
                        f"lead_hour={lh}: {exc}"
                    ) from exc

                # Decode with cfgrib — failures here mean the run is bad/absent.
                try:
                    ds_single: xr.Dataset = cfgrib.open_dataset(  # type: ignore[reportUnknownMemberType]
                        tmp_path,
                        filter_by_keys={"shortName": short_name},
                        indexpath="",  # avoid creating .idx sidecar files
                    )
                except Exception as exc:
                    raise ForecastUnavailable(
                        f"Failed to decode GRIB2 for shortName={short_name!r}, "
                        f"lead_hour={lh}: {exc}"
                    ) from exc

                # Extract the DataArray for this variable.
                if short_name not in ds_single:
                    raise ForecastUnavailable(
                        f"Variable {short_name!r} not found in GRIB2 at lead_hour={lh}. "
                        f"Available: {list(ds_single.data_vars)}"
                    )
                per_lh[lh][short_name] = ds_single[short_name]

        # Combine into a single Dataset with a lead_hour dimension.
        # Extract spatial shape from the first DataArray.
        first_da = next(iter(per_lh[all_lead_hours[0]].values()))
        spatial_dims: tuple[str, ...] = tuple(str(d) for d in first_da.dims)  # type: ignore[reportUnknownMemberType]
        spatial_shape: tuple[int, ...] = tuple(int(s) for s in first_da.shape)  # type: ignore[reportUnknownMemberType]

        # Build combined arrays of shape (lead_hour, y, x).
        data_vars: dict[str, xr.DataArray] = {}
        for short_name in _GRIB_SHORT_NAMES:
            stacked: np.ndarray = np.stack(  # type: ignore[reportUnknownMemberType]
                [per_lh[lh][short_name].values for lh in all_lead_hours],  # type: ignore[reportUnknownMemberType]
                axis=0,
            )
            data_vars[short_name] = xr.DataArray(
                stacked,
                dims=("lead_hour", *spatial_dims),
            )

        # Extract latitude/longitude from the first DataArray's coordinates.
        first_var_da = per_lh[all_lead_hours[0]][_GRIB_SHORT_NAMES[0]]
        lat_vals: xr.DataArray
        lon_vals: xr.DataArray
        if "latitude" in first_var_da.coords and "longitude" in first_var_da.coords:  # type: ignore[reportUnknownMemberType]
            lat_raw: xr.DataArray = first_var_da.coords["latitude"]  # type: ignore[reportUnknownMemberType]
            lon_raw: xr.DataArray = first_var_da.coords["longitude"]  # type: ignore[reportUnknownMemberType]
            if int(lat_raw.ndim) == 1:  # type: ignore[reportUnknownMemberType]
                # 1-D → broadcast to 2-D over (y, x) to satisfy nwp_core contract.
                lat_2d: np.ndarray = np.broadcast_to(  # type: ignore[reportUnknownMemberType]
                    np.array(lat_raw.values)[:, np.newaxis],  # type: ignore[reportUnknownMemberType]
                    spatial_shape,
                ).copy()
                lon_2d: np.ndarray = np.broadcast_to(  # type: ignore[reportUnknownMemberType]
                    np.array(lon_raw.values)[np.newaxis, :],  # type: ignore[reportUnknownMemberType]
                    spatial_shape,
                ).copy()
                lat_vals = xr.DataArray(lat_2d, dims=spatial_dims)
                lon_vals = xr.DataArray(lon_2d, dims=spatial_dims)
            else:
                lat_vals = xr.DataArray(np.array(lat_raw.values), dims=spatial_dims)  # type: ignore[reportUnknownMemberType]
                lon_vals = xr.DataArray(np.array(lon_raw.values), dims=spatial_dims)  # type: ignore[reportUnknownMemberType]
        else:
            raise ForecastUnavailable(
                "GRIB2 Dataset does not contain 'latitude'/'longitude' coordinates — "
                "cannot normalise to nwp_core Dataset contract."
            )

        lh_coord = xr.DataArray(np.array(all_lead_hours, dtype=np.int64), dims=("lead_hour",))
        coords: dict[str, object] = {
            "lead_hour": lh_coord,
            "latitude": lat_vals,
            "longitude": lon_vals,
        }

        ds = xr.Dataset(data_vars, coords=coords)
        # Materialise into memory BEFORE the TemporaryDirectory context exits —
        # cfgrib reads lazily from the file path; after cleanup the paths are gone.
        return ds.load()  # type: ignore[reportUnknownMemberType]


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------


@register_source("hrdps_datamart")
class HrdpsDatamartSource(NWPSource):
    """HRDPS live connector via MSC Datamart GRIB2 files.

    Fetches HRDPS continental 2.5 km GRIB2 files from the MSC Datamart,
    decodes them with cfgrib, and normalises the result to a FORECAST_FRAME-
    valid DataFrame via ``dataset_to_forecast_frame``.

    Optionally accepts an injectable ``opener`` callable for hermetic unit
    testing (see module docstring).  The registry calls
    ``HrdpsDatamartSource()`` with no args, so the default MUST work
    argument-free.
    """

    def __init__(
        self,
        opener: Callable[[datetime, Sequence[int]], xr.Dataset] | None = None,
    ) -> None:
        # Default to the real network/cfgrib seam; allows zero-arg instantiation
        # by the registry.
        self._opener: Callable[[datetime, Sequence[int]], xr.Dataset] = (
            opener if opener is not None else _open_latest_run
        )

    @property
    def is_live(self) -> bool:
        return True

    def fetch_forecast(
        self,
        issue_time: datetime,
        lat: float,
        lon: float,
        lead_hours: Sequence[int],
    ) -> pd.DataFrame:
        """Return a FORECAST_FRAME-valid DataFrame for the requested lead hours.

        Args:
            issue_time:  UTC model run initialisation time.
            lat:         Target latitude (decimal degrees).
            lon:         Target longitude (decimal degrees).
            lead_hours:  Ordered sequence of forecast lead hours to return
                         (each >= 1; must be present in the downloaded run).

        Returns:
            DataFrame validated against ``FORECAST_FRAME``, one row per
            requested lead hour.

        Raises:
            SourceUnavailable: If the opener encounters an infra/network failure
                (cfgrib not importable, HTTP error, disk I/O failure).
            ForecastUnavailable: If the run is absent, truncated, or corrupt, or
                if the core rejects the dataset (e.g. missing lead hour needed
                for precip de-accumulation).
        """
        issue_utc = issue_time if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)
        # The opener raises SourceUnavailable (infra) or ForecastUnavailable
        # (absent/truncated run); BOTH propagate unchanged — do NOT re-wrap
        # SourceUnavailable here.
        ds = self._opener(issue_utc, lead_hours)
        try:
            return dataset_to_forecast_frame(
                ds,
                _HRDPS_VAR_MAP,
                issue_time=issue_utc,
                lat=lat,
                lon=lon,
                lead_hours=lead_hours,
            )
        except ValueError as exc:
            # Core rejects an incomplete/malformed dataset (e.g. a truncated run
            # missing the lead hour needed for precip de-accumulation) → that is
            # a forecast-unavailable condition.
            raise ForecastUnavailable(
                f"HRDPS run at {issue_utc.isoformat()} produced an unusable dataset: {exc}"
            ) from exc
