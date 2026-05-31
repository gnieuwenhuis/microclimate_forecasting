"""Environment Canada ECCC bulk hourly CSV observation connector (seam/core split).

Architecture:
  Seam  — ``_fetch_eccc_csv(station_id, year, month)`` builds the URL + params and
           calls ``http_get``; ALL network I/O is confined here.
  Core  — ``_parse_eccc_csv(csv_text, station_id)`` is a pure function: raw CSV text
           → OBSERVATION_FRAME-valid DataFrame. No network; hammered by unit tests.

Injectable fetcher
  ``EnvCanadaSource(fetcher=...)`` accepts an optional ``Callable[[str, int, int], str]``
  (station_id, year, month) → csv_text.  The default (``fetcher=None``) falls through to
  the real seam.  The registry calls ``EnvCanadaSource()`` with no args, so the default
  MUST work argument-free.

v1 limitation — LST / UTC offset
  All v1 ``envcanada`` stations are in Alberta = Mountain Standard Time = UTC−7
  year-round (no DST).  If ``envcanada`` ever serves non-Alberta stations this
  module-level offset MUST be revisited.  See ``_LST_UTC_OFFSET`` below.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd

from microclimate.connectors.base import HistoricalCoverage, ObservationSource, StationNotFound
from microclimate.connectors.http import http_get
from microclimate.connectors.registry import register_source
from microclimate.contracts.observation import OBSERVATION_FRAME

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_ECCC_URL: str = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

# v1 assumption: all envcanada stations are in Alberta (MST = UTC−7, no DST).
# REVISIT if the envcanada connector is ever extended to non-Alberta (non-MST) stations.
_LST_UTC_OFFSET: timedelta = timedelta(hours=7)

# Magnus-Tetens coefficients (August-Roche Magnus approximation)
_MT_A: float = 17.625
_MT_B: float = 243.04  # °C

# The sentinel column we check to verify the response is actually an ECCC station CSV.
_REQUIRED_COLUMN_PREFIX: str = "Date/Time"

# Ordered list of physical variable names (drives output column construction).
_PHYS_VARS: tuple[str, ...] = (
    "temp_c",
    "dewpoint_c",
    "surface_pressure_hpa",
    "precip_mm",
    "cloud_cover_fraction",
    "solar_radiation_wm2",
    "wind_speed_ms",
    "wind_dir_deg",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_float(value: object) -> float | None:
    """Parse *value* to float; return None if blank, non-numeric, or the ECCC ``"M"``
    missing-data sentinel (which ECCC uses to mark officially missing readings)."""
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _magnus_tetens(t: float, rh: float) -> float:
    """Derive dewpoint (°C) from temperature *t* (°C) and relative humidity *rh* (%)."""
    gamma = math.log(rh / 100.0) + _MT_A * t / (_MT_B + t)
    return _MT_B * gamma / (_MT_A - gamma)


def _find_col(columns: pd.Index, prefix: str) -> str | None:  # type: ignore[reportUnknownMemberType]
    """Return the first column name whose text starts with *prefix*, or None."""
    for col in columns:
        if str(col).startswith(prefix):
            return str(col)
    return None


def _get_cell(series: pd.Series, col: str | None) -> float | None:  # type: ignore[reportUnknownMemberType]
    """Extract a float from *series[col]*; return None if col is None or value is non-numeric."""
    if col is None:
        return None
    return _try_float(series.get(col, ""))


def _empty_obs_frame() -> pd.DataFrame:
    """Return a zero-row DataFrame that conforms to OBSERVATION_FRAME.

    Used as the canonical empty result whenever a valid ECCC station CSV contains
    no reportable observation rows (e.g. an entirely future/unfilled month).
    The frame is validated here so callers get a guaranteed-conformant result.
    """
    columns: dict[str, object] = {
        "station_id": pd.Series([], dtype="object"),
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
    }
    for var in _PHYS_VARS:
        columns[var] = pd.Series([], dtype="float64")
        columns[f"{var}_present"] = pd.Series([], dtype="bool")

    df: pd.DataFrame = pd.DataFrame(columns)  # type: ignore[reportUnknownMemberType]
    OBSERVATION_FRAME.validate(df)
    return df


# ---------------------------------------------------------------------------
# Seam — network I/O lives here exclusively
# ---------------------------------------------------------------------------


def _fetch_eccc_csv(station_id: str, year: int, month: int) -> str:
    """Fetch one month of ECCC bulk hourly CSV for *station_id*.

    Returns the raw CSV text (UTF-8 BOM, ECCC format).
    Raises SourceUnavailable on any network / HTTP error (via http_get).
    """
    params: dict[str, str | int] = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    }
    return http_get(_ECCC_URL, params=params)


# ---------------------------------------------------------------------------
# Core — pure CSV → DataFrame parsing (no network)
# ---------------------------------------------------------------------------


def _parse_eccc_csv(csv_text: str, station_id: str) -> pd.DataFrame:
    """Parse raw ECCC bulk hourly CSV text into an OBSERVATION_FRAME-compatible DataFrame.

    Args:
        csv_text:   Raw CSV returned by the ECCC bulk endpoint (UTF-8 BOM encoded).
        station_id: The MSC station identifier to populate the ``station_id`` column.

    Returns:
        DataFrame conforming to OBSERVATION_FRAME (not yet Pandera-validated; caller
        decides when to validate).  Returns a zero-row schema-valid frame when the
        CSV is a legitimate ECCC station CSV that happens to have no data rows yet
        (e.g. a future/unfilled month).

    Raises:
        StationNotFound: If *csv_text* is not a recognisable ECCC station CSV (i.e.
            the required ``Date/Time (LST)`` column is absent).  A valid-header CSV
            with zero data rows does NOT raise; it returns an empty schema-valid frame.
    """
    # ------------------------------------------------------------------
    # 1. Parse raw CSV; handle UTF-8 BOM via encoding_errors-safe read.
    # ------------------------------------------------------------------
    try:
        raw = pd.read_csv(  # type: ignore[reportUnknownMemberType]
            io.StringIO(csv_text),
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except Exception as exc:
        raise StationNotFound(f"Cannot parse ECCC CSV for station {station_id!r}: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Validate the response looks like an ECCC station CSV.
    #    A non-station body (HTML, empty, wrong format) → StationNotFound.
    #    A valid-header CSV with zero rows → return empty schema-valid frame.
    # ------------------------------------------------------------------
    dt_col = _find_col(raw.columns, _REQUIRED_COLUMN_PREFIX)
    if dt_col is None:
        raise StationNotFound(
            f"Response for station {station_id!r} is not an ECCC bulk hourly CSV "
            f"(missing '{_REQUIRED_COLUMN_PREFIX}...' column)."
        )

    # ------------------------------------------------------------------
    # 3. Locate source columns by prefix (immune to degree-sign encoding).
    # ------------------------------------------------------------------
    temp_col = _find_col(raw.columns, "Temp (")
    dp_col = _find_col(raw.columns, "Dew Point Temp (")
    rh_col = _find_col(raw.columns, "Rel Hum (")
    press_col = _find_col(raw.columns, "Stn Press (")
    precip_col = _find_col(raw.columns, "Precip. Amount (")
    wdir_col = _find_col(raw.columns, "Wind Dir (")
    wspd_col = _find_col(raw.columns, "Wind Spd (")

    # ------------------------------------------------------------------
    # 4. Iterate rows: parse timestamp, convert LST → UTC, build output.
    # ------------------------------------------------------------------
    rows: list[dict[str, object]] = []

    for _, series in raw.iterrows():
        dt_str = str(series.get(dt_col, "")).strip()
        if not dt_str:
            continue
        try:
            lst = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except ValueError:
            continue

        utc_ts = pd.Timestamp(lst + _LST_UTC_OFFSET, tz="UTC")

        # ------------------------------------------------------------------
        # 5. Parse each physical variable; apply unit conversions.
        # ------------------------------------------------------------------
        temp_v = _get_cell(series, temp_col)
        rh_v = _get_cell(series, rh_col)

        # Dewpoint: use cell value if present; else derive from T+RH; else absent.
        dp_v_raw = _get_cell(series, dp_col)
        if dp_v_raw is not None:
            dp_v: float | None = dp_v_raw
        elif temp_v is not None and rh_v is not None and rh_v > 0.0:
            dp_v = _magnus_tetens(temp_v, rh_v)
        else:
            dp_v = None

        press_raw = _get_cell(series, press_col)
        press_v = press_raw * 10.0 if press_raw is not None else None  # kPa → hPa

        precip_v = _get_cell(series, precip_col)

        wdir_raw = _get_cell(series, wdir_col)
        wdir_v = wdir_raw * 10.0 if wdir_raw is not None else None  # tens-of-deg → deg

        wspd_raw = _get_cell(series, wspd_col)
        wspd_v = wspd_raw / 3.6 if wspd_raw is not None else None  # km/h → m/s

        # cloud and solar are not in the ECCC bulk CSV → always absent.
        cloud_v: float | None = None
        solar_v: float | None = None

        var_values: dict[str, float | None] = {
            "temp_c": temp_v,
            "dewpoint_c": dp_v,
            "surface_pressure_hpa": press_v,
            "precip_mm": precip_v,
            "cloud_cover_fraction": cloud_v,
            "solar_radiation_wm2": solar_v,
            "wind_speed_ms": wspd_v,
            "wind_dir_deg": wdir_v,
        }

        # ------------------------------------------------------------------
        # 6. Drop rows where ALL 8 physical variables are absent
        #    (not-yet-reported trailing rows in live current-month CSV).
        # ------------------------------------------------------------------
        if all(v is None for v in var_values.values()):
            continue

        # ------------------------------------------------------------------
        # 7. Build the output row with _present masks.
        # ------------------------------------------------------------------
        out: dict[str, object] = {
            "station_id": station_id,
            "timestamp": utc_ts,
        }
        for var, val in var_values.items():
            out[var] = val if val is not None else float("nan")
            out[f"{var}_present"] = val is not None

        rows.append(out)

    if not rows:
        # Valid ECCC station CSV but no reportable rows (e.g. future/unfilled month).
        # Degrade gracefully: return empty schema-valid frame (ADR-0002).
        return _empty_obs_frame()

    df: pd.DataFrame = pd.DataFrame(rows)  # type: ignore[reportUnknownMemberType]
    return df


# ---------------------------------------------------------------------------
# Connector implementation
# ---------------------------------------------------------------------------


@register_source("envcanada")
class EnvCanadaSource(ObservationSource):
    """Environment Canada ECCC bulk hourly CSV observation connector.

    Fetches monthly bulk CSVs from the ECCC climate data portal and normalises them
    to ``OBSERVATION_FRAME``.  Optionally accepts an injectable ``fetcher`` callable
    for hermetic unit testing (see module docstring).
    """

    def __init__(
        self,
        fetcher: Callable[[str, int, int], str] | None = None,
    ) -> None:
        # Default to the real seam; allows zero-arg instantiation by the registry.
        self._fetcher: Callable[[str, int, int], str] = (
            fetcher if fetcher is not None else _fetch_eccc_csv
        )

    @property
    def historical_coverage(self) -> HistoricalCoverage:
        return "deep"

    def fetch_historical(self, station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return observations for *station_id* in [*start*, *end*] (inclusive).

        Args:
            station_id: MSC station identifier (e.g. ``"49268"``).
            start:      Window start (UTC-aware; if naive, treated as UTC).
            end:        Window end   (UTC-aware; if naive, treated as UTC).

        Returns:
            OBSERVATION_FRAME-conformant DataFrame, sorted ascending by timestamp.
            Returns an empty schema-valid frame when the station is valid but has
            no observations in the requested window (ADR-0002 graceful degradation).

        Raises:
            SourceUnavailable: If any underlying HTTP request fails.
            StationNotFound:   If the response is not a recognisable ECCC station CSV
                               (i.e. the station does not exist at the source).
        """
        start_utc = _ensure_utc(start)
        end_utc = _ensure_utc(end)

        # Derive the CSV month-keys from LST bounds, not UTC bounds.
        # ECCC bulk CSVs are keyed by LST calendar month (LST = UTC − 7h).
        # A row whose UTC timestamp is T lives in the CSV for LST month of (T − offset).
        start_lst = start_utc - _LST_UTC_OFFSET
        end_lst = end_utc - _LST_UTC_OFFSET
        months = _month_range(start_lst, end_lst)
        frames: list[pd.DataFrame] = []
        for year, month in months:
            csv_text = self._fetcher(station_id, year, month)
            # _parse_eccc_csv raises StationNotFound for a non-station body;
            # returns an empty schema-valid frame for a valid-header-but-no-data month.
            df = _parse_eccc_csv(csv_text, station_id)
            frames.append(df)

        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)  # type: ignore[reportUnknownMemberType]
        combined = combined.drop_duplicates(subset="timestamp")  # type: ignore[reportUnknownMemberType]
        # Filter to the requested window (inclusive); filtering stays in UTC (correct).
        mask = (combined["timestamp"] >= pd.Timestamp(start_utc)) & (
            combined["timestamp"] <= pd.Timestamp(end_utc)
        )
        result: pd.DataFrame = (
            combined.loc[mask].sort_values("timestamp").reset_index(drop=True)  # type: ignore[reportUnknownMemberType]
        )

        # An empty result is valid: the station exists but has no data in the window.
        # Return the canonical empty frame (already schema-valid) rather than raising.
        if len(result) == 0:
            return _empty_obs_frame()

        return result

    def fetch_live(self, station_id: str, since: datetime) -> pd.DataFrame:
        """Return observations for *station_id* from *since* to now (current month).

        Args:
            station_id: MSC station identifier.
            since:      Lower bound (UTC-aware; if naive, treated as UTC).

        Returns:
            OBSERVATION_FRAME-conformant DataFrame, sorted ascending by timestamp.
            Returns an empty schema-valid frame when the current month has no
            observations yet (ADR-0002 graceful degradation).

        Raises:
            SourceUnavailable: If the underlying HTTP request fails.
            StationNotFound:   If the response is not a recognisable ECCC station CSV.
        """
        since_utc = _ensure_utc(since)
        now_utc = datetime.now(tz=UTC)

        # Derive the CSV month-keys from LST bounds, not UTC bounds.
        # ECCC bulk CSVs are keyed by LST calendar month (LST = UTC − 7h).
        now_lst = now_utc - _LST_UTC_OFFSET
        since_lst = since_utc - _LST_UTC_OFFSET
        months = _month_range(since_lst, now_lst)

        frames: list[pd.DataFrame] = []
        for year, month in months:
            csv_text = self._fetcher(station_id, year, month)
            df_month = _parse_eccc_csv(csv_text, station_id)
            frames.append(df_month)

        combined: pd.DataFrame = pd.concat(frames, ignore_index=True)  # type: ignore[reportUnknownMemberType]
        combined = combined.drop_duplicates(subset="timestamp")  # type: ignore[reportUnknownMemberType]

        if len(combined) == 0:
            return _empty_obs_frame()

        # Filter to rows >= since; filtering stays in UTC (correct).
        mask = combined["timestamp"] >= pd.Timestamp(since_utc)
        result: pd.DataFrame = (
            combined.loc[mask].sort_values("timestamp").reset_index(drop=True)  # type: ignore[reportUnknownMemberType]
        )

        # An empty result is valid: the station exists but has no data since the cutoff.
        if len(result) == 0:
            return _empty_obs_frame()

        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime; naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _month_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples spanning *start* to *end* (inclusive)."""
    months: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months
