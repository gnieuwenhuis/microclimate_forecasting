"""Shared HTTP GET helper for L2 connectors (timeouts, explicit bounded backoff, descriptive UA)."""

from __future__ import annotations

import time
from collections.abc import Mapping

import requests

from microclimate.connectors.base import SourceUnavailable

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_USER_AGENT: str = (
    "microclimate-forecasting (+https://github.com/gnieuwenhuis/microclimate_forecasting)"
)

# Explicit (connect_timeout, read_timeout) in seconds.
_TIMEOUT: tuple[float, float] = (10.0, 30.0)

# Sleeps between attempts on transient failures (HTTP 5xx, connection error, timeout):
# 6 attempts over ~155 s of backoff (~2.6 min budget). Rides out short upstream blips
# (e.g. an Open-Meteo 502); for longer outages the next hourly Action run is the
# fallback and the caller warns-and-skips (ADR-0020).
_BACKOFF_SCHEDULE: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)

# Module-level indirection so tests can patch sleeping away.
_sleep = time.sleep

# ---------------------------------------------------------------------------
# Module-level session (created once, shared across calls)
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = _USER_AGENT


# ---------------------------------------------------------------------------
# Internal shared request helper
# ---------------------------------------------------------------------------


def _is_transient(exc: requests.RequestException) -> bool:
    """True when the failure class can heal on retry (HTTP 5xx, connection, timeout)."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _do_get(
    url: str, *, params: Mapping[str, str | int | float] | None = None
) -> requests.Response:
    """Perform an HTTP GET with bounded backoff on transient failures; return the Response.

    Transient failures (HTTP 5xx, connection errors, timeouts) are retried through
    ``_BACKOFF_SCHEDULE``; anything else (4xx, malformed requests) fails immediately.

    Raises:
        SourceUnavailable: On a non-transient failure, or once retries are exhausted —
            the exhaustion message names the URL, attempt count, and elapsed retry time.
    """
    start = time.monotonic()
    for attempt in range(1, len(_BACKOFF_SCHEDULE) + 2):
        try:
            response = _SESSION.get(url, params=params, timeout=_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if not _is_transient(exc):
                kind = "HTTP error" if isinstance(exc, requests.HTTPError) else "request failed"
                raise SourceUnavailable(f"{kind} fetching {url!r}: {exc}") from exc
            if attempt > len(_BACKOFF_SCHEDULE):
                elapsed = time.monotonic() - start
                raise SourceUnavailable(
                    f"gave up after {attempt} attempts over {elapsed:.0f}s fetching {url!r}: {exc}"
                ) from exc
            _sleep(_BACKOFF_SCHEDULE[attempt - 1])
    raise AssertionError("unreachable: loop exits via return or raise")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def http_get(url: str, *, params: Mapping[str, str | int | float] | None = None) -> str:
    """Perform an HTTP GET and return the response body as text.

    Args:
        url:    Absolute URL to fetch.
        params: Optional query parameters to append to the URL.

    Returns:
        The response body as a decoded string.

    Raises:
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status
            (transient failures are retried first — see ``_do_get``).
    """
    return _do_get(url, params=params).text


def http_get_bytes(url: str, *, params: Mapping[str, str | int | float] | None = None) -> bytes:
    """Perform an HTTP GET and return the response body as bytes.

    Useful for binary formats such as GRIB2.

    Args:
        url:    Absolute URL to fetch.
        params: Optional query parameters to append to the URL.

    Returns:
        The response body as raw bytes (``response.content``).

    Raises:
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status
            (transient failures are retried first — see ``_do_get``).
    """
    return _do_get(url, params=params).content
