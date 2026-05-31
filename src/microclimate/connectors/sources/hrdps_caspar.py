"""HRDPS via the CaSPAr archive — historical NWP seed (local-archive reader).

Architecture:
  Seam  — ``_open_caspar_file(path)`` opens a pre-staged CaSPAr archive file
           (GRIB2 or netCDF) and normalises it to the nwp_core Dataset contract.
           ALL file I/O and cfgrib decoding is confined here.  cfgrib is imported
           LAZILY inside the GRIB branch only (eccodes binary may be absent in
           CI/local envs).
  Core  — ``dataset_to_forecast_frame`` (in nwp_core) is a pure xarray/pandas
           function that consumes the normalised Dataset.  No file I/O; no cfgrib.

Pinned archive layout (contract with the out-of-scope bulk-acquisition step):
  {archive_root}/{YYYY}/{MM}/hrdps_{YYYYMMDDHH}.{ext}

  where YYYYMMDDHH is derived from the UTC issue_time (zero-padded).  Supported
  extensions (tried in priority order): .grib2, .grib, .nc.

  Example: issue_time=2026-05-30 00:00 UTC →
    {archive_root}/2026/05/hrdps_2026053000.grib2

Configuration:
  archive_root can be passed as a constructor arg (Path | str) or read from the
  ``CASPAR_ARCHIVE_ROOT`` environment variable at construction time.  If neither
  is set, ``archive_root`` is stored as ``None`` and a clear error is raised only
  when ``fetch_forecast`` is first called (so zero-arg registry construction works).

Injectable opener:
  ``HrdpsCasparSource(opener=...)`` accepts an optional
  ``Callable[[Path], xr.Dataset]`` for hermetic unit testing.
  The default (``opener=None``) falls through to the real seam.  The registry
  calls ``HrdpsCasparSource()`` with no args, so the default MUST work
  argument-free.

IMPORTANT — seam caveats (unverified against real CaSPAr files):
  * The archive layout assumed by ``_archive_path`` is a best-effort convention.
    Actual CaSPAr bulk-download scripts may produce a different directory
    structure or filename pattern.
  * The variable names in ``CASPAR_VAR_MAP`` are plausible CaSPAr netCDF names
    but have NOT been verified against real CaSPAr files.  Whether real CaSPAr
    GRIB2 files encode these variables under the same shortNames has not been
    confirmed.
  * cfgrib ``filter_by_keys`` arguments in ``_open_caspar_file`` may need tuning
    for the actual file layout of CaSPAr HRDPS products.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import xarray as xr

from microclimate.connectors.base import ForecastUnavailable, NWPSource, SourceUnavailable
from microclimate.connectors.nwp_core import dataset_to_forecast_frame
from microclimate.connectors.registry import register_source

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Supported archive file extensions tried in priority order.
_SUPPORTED_EXTS: tuple[str, ...] = (".grib2", ".grib", ".nc")

# Mapping: canonical column name → CaSPAr variable name.
# Exported as a public name so tests can import it directly and stay in sync.
# NOTE: Real CaSPAr variable names are unverified — see module docstring.
CASPAR_VAR_MAP: dict[str, str] = {
    "temp_c": "TT",
    "dewpoint_c": "TD",
    "surface_pressure_hpa": "PN",
    "precip_mm": "PR",
    "cloud_cover_fraction": "NT",
    "solar_radiation_wm2": "FB",
    "wind_speed_ms": "UV",
    "wind_dir_deg": "WD",
}

# Private alias retained for readability within this module.
_CASPAR_VAR_MAP: dict[str, str] = CASPAR_VAR_MAP


# ---------------------------------------------------------------------------
# Pure path helpers
# ---------------------------------------------------------------------------


def _archive_path(root: Path, issue_time: datetime, ext: str) -> Path:
    """Return the pinned archive path for a given issue_time and extension.

    Layout: {root}/{YYYY}/{MM}/hrdps_{YYYYMMDDHH}{ext}

    Args:
        root:        Archive root directory.
        issue_time:  UTC model run initialisation time (must be UTC-aware or naive-UTC).
        ext:         File extension including leading dot (e.g. ``".grib2"``).

    Returns:
        Absolute Path to the expected archive file.
    """
    year = issue_time.strftime("%Y")
    month = issue_time.strftime("%m")
    stem = issue_time.strftime("%Y%m%d%H")
    return root / year / month / f"hrdps_{stem}{ext}"


def _resolve_existing_archive(root: Path, issue_time: datetime) -> Path:
    """Return the path of the first existing archive file for issue_time.

    Tries ``_SUPPORTED_EXTS`` in priority order.

    Args:
        root:        Archive root directory.
        issue_time:  UTC model run initialisation time.

    Returns:
        Path to the first existing file.

    Raises:
        ForecastUnavailable: If no file exists at any supported extension.
    """
    for ext in _SUPPORTED_EXTS:
        candidate = _archive_path(root, issue_time, ext)
        if candidate.is_file():
            return candidate
    stem = issue_time.strftime("%Y%m%d%H")
    raise ForecastUnavailable(
        f"No CaSPAr archive file found for issue_time={issue_time.isoformat()}. "
        f"Expected stem: {root}/{issue_time.strftime('%Y/%m')}/hrdps_{stem}"
        f" with extension in {list(_SUPPORTED_EXTS)}."
    )


# ---------------------------------------------------------------------------
# Seam — file I/O + cfgrib lives here exclusively
# ---------------------------------------------------------------------------


def _open_caspar_file(path: Path) -> xr.Dataset:
    """Open a CaSPAr archive file and return a normalised Dataset.

    This is the file-I/O seam — it is NOT hermetically testable here because
    eccodes binary is unavailable and no real archive is present in CI.  All
    correctness testing is done via the injectable ``opener`` in
    ``HrdpsCasparSource``.

    Chooses engine by extension:
      * ``.nc``          → ``xr.open_dataset`` (standard netCDF; no cfgrib)
      * ``.grib2/.grib`` → lazily ``import cfgrib``; ``cfgrib.open_dataset``

    The Dataset returned conforms to the nwp_core contract:
        - 1-D integer ``lead_hour`` coordinate (ascending, includes the hour
          before the smallest requested hour for precip de-accumulation).
        - 2-D ``latitude`` / ``longitude`` coordinates over dims (y, x).
        - Data variables named by CaSPAr variable names (see ``CASPAR_VAR_MAP``).

    IMPORTANT: Normalisation logic here is best-effort (unverified against real
    CaSPAr files).  The cfgrib filter_by_keys arguments and netCDF variable
    layout assumptions must be confirmed against actual CaSPAr output.

    Args:
        path: Path to the CaSPAr archive file.

    Returns:
        Normalised xr.Dataset matching the nwp_core Dataset contract, fully
        loaded into memory.

    Raises:
        SourceUnavailable: If cfgrib/eccodes is not importable (GRIB branch),
            or if a disk I/O error occurs while reading the file.
        ForecastUnavailable: If the file cannot be decoded, is missing expected
            variables, or cannot be normalised to the Dataset contract.
    """
    ext = path.suffix.lower()
    if ext == ".nc":
        try:
            ds: xr.Dataset = xr.open_dataset(path)  # type: ignore[reportUnknownMemberType]
            return ds.load()  # type: ignore[reportUnknownMemberType]
        except (OSError, PermissionError) as exc:
            raise SourceUnavailable(
                f"Disk I/O error reading CaSPAr archive at {path}: {exc}"
            ) from exc
        except Exception as exc:
            raise ForecastUnavailable(f"Failed to decode netCDF archive at {path}: {exc}") from exc
    else:
        # GRIB2/GRIB: import cfgrib lazily — eccodes binary may be absent.
        try:
            import cfgrib  # type: ignore[reportMissingTypeStubs]
        except ImportError as exc:
            raise SourceUnavailable(
                "cfgrib is not importable (eccodes binary unavailable)"
            ) from exc
        try:
            ds = cfgrib.open_dataset(  # type: ignore[reportUnknownMemberType]
                str(path),
                indexpath="",  # avoid creating .idx sidecar files
            )
            return ds.load()  # type: ignore[reportUnknownMemberType]
        except (OSError, PermissionError) as exc:
            raise SourceUnavailable(
                f"Disk I/O error reading CaSPAr archive at {path}: {exc}"
            ) from exc
        except Exception as exc:
            raise ForecastUnavailable(f"Failed to decode GRIB archive at {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------


@register_source("hrdps_caspar")
class HrdpsCasparSource(NWPSource):
    """HRDPS historical connector via a pre-staged CaSPAr local archive.

    Reads pre-staged CaSPAr HRDPS files from a local archive directory,
    decodes them (GRIB2 via cfgrib, or netCDF via xarray), and normalises
    the result to a FORECAST_FRAME-valid DataFrame via
    ``dataset_to_forecast_frame``.

    Optionally accepts an injectable ``opener`` callable for hermetic unit
    testing (see module docstring).  The registry calls
    ``HrdpsCasparSource()`` with no args, so the default MUST work
    argument-free.
    """

    def __init__(
        self,
        archive_root: Path | str | None = None,
        opener: Callable[[Path], xr.Dataset] | None = None,
    ) -> None:
        # Resolve archive_root: constructor arg > env var > None.
        # If None, we defer the error to fetch_forecast so the registry's
        # zero-arg HrdpsCasparSource() construction still works.
        if archive_root is not None:
            self._archive_root: Path | None = Path(archive_root)
        else:
            env_val = os.environ.get("CASPAR_ARCHIVE_ROOT")
            self._archive_root = Path(env_val) if env_val else None

        # Default to the real file-I/O seam; allows zero-arg instantiation.
        self._opener: Callable[[Path], xr.Dataset] = (
            opener if opener is not None else _open_caspar_file
        )

    @property
    def is_live(self) -> bool:
        return False

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
                         (each >= 1; must be present in the archive file).

        Returns:
            DataFrame validated against ``FORECAST_FRAME``, one row per
            requested lead hour.

        Raises:
            ForecastUnavailable: If no archive_root is configured, if no file
                exists at the expected path, or if the core rejects the dataset.
            SourceUnavailable: If the opener encounters an infra/I/O failure
                (cfgrib not importable, disk I/O error).
        """
        if self._archive_root is None:
            raise ForecastUnavailable(
                "CaSPAr archive root is not configured. "
                "Pass archive_root to HrdpsCasparSource() or set the "
                "CASPAR_ARCHIVE_ROOT environment variable."
            )

        issue_utc = issue_time if issue_time.tzinfo is not None else issue_time.replace(tzinfo=UTC)

        archive_path = _resolve_existing_archive(self._archive_root, issue_utc)

        # The opener raises SourceUnavailable (infra) or ForecastUnavailable
        # (absent/truncated run); BOTH propagate unchanged — do NOT re-wrap
        # SourceUnavailable here.
        ds = self._opener(archive_path)

        try:
            return dataset_to_forecast_frame(
                ds,
                _CASPAR_VAR_MAP,
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
                f"CaSPAr archive for {issue_utc.isoformat()} produced an unusable dataset: {exc}"
            ) from exc
