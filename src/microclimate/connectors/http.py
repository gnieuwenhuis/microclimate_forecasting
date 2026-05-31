"""Shared HTTP GET helper for L2 connectors (timeouts, bounded retries, descriptive UA)."""

from __future__ import annotations

from collections.abc import Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from microclimate.connectors.base import SourceUnavailable

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_USER_AGENT: str = (
    "microclimate-forecasting (+https://github.com/gnieuwenhuis/microclimate_forecasting)"
)

# Explicit (connect_timeout, read_timeout) in seconds.
_TIMEOUT: tuple[float, float] = (10.0, 30.0)

_RETRY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

# ---------------------------------------------------------------------------
# Module-level session (created once, shared across calls)
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = _USER_AGENT
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
_SESSION.mount("http://", HTTPAdapter(max_retries=_RETRY))


# ---------------------------------------------------------------------------
# Internal shared request helper
# ---------------------------------------------------------------------------


def _do_get(
    url: str, *, params: Mapping[str, str | int | float] | None = None
) -> requests.Response:
    """Perform an HTTP GET with shared error-handling; return the Response.

    Raises:
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status.
    """
    try:
        response = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SourceUnavailable(f"HTTP error fetching {url!r}: {exc}") from exc
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise SourceUnavailable(f"Network error fetching {url!r}: {exc}") from exc
    except requests.RequestException as exc:
        raise SourceUnavailable(f"Request failed for {url!r}: {exc}") from exc
    return response


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
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status.
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
        SourceUnavailable: On any network failure, timeout, or non-2xx HTTP status.
    """
    return _do_get(url, params=params).content
