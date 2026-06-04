"""Tests for the shared HTTP GET helper (L2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from microclimate.connectors.base import SourceUnavailable
from microclimate.connectors.http import http_get, http_get_bytes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    text: str = "",
    status_code: int = 200,
    content: bytes = b"",
) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.text = text
    resp.content = content or text.encode()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp  # type: ignore[call-arg]
        )
    return resp


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Patch backoff sleeping away — no test in this module may sleep for real."""

    def _no_sleep(_s: float) -> None:
        pass

    monkeypatch.setattr("microclimate.connectors.http._sleep", _no_sleep)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_success_returns_body_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """http_get returns the response body text on a 200 OK."""
    mock_get = MagicMock(return_value=_make_response("hello world"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    result = http_get("https://example.com/data")

    assert result == "hello world"


def test_success_sends_descriptive_user_agent() -> None:
    """Session is initialised with the project User-Agent header."""
    import microclimate.connectors.http as http_mod

    ua = http_mod._SESSION.headers["User-Agent"]  # type: ignore[reportPrivateUsage]
    assert isinstance(ua, str)
    assert "microclimate-forecasting" in ua


def test_success_passes_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """http_get calls Session.get with an explicit (connect, read) timeout tuple."""
    mock_get = MagicMock(return_value=_make_response("ok"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    http_get("https://example.com/data")

    _, call_kwargs = mock_get.call_args
    timeout: object = call_kwargs.get("timeout")
    assert isinstance(timeout, tuple), "timeout must be a (connect, read) tuple"
    assert len(timeout) == 2  # type: ignore[arg-type]


def test_success_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query params are forwarded to Session.get."""
    mock_get = MagicMock(return_value=_make_response("data"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    http_get("https://example.com/data", params={"station": "AAA", "limit": 10})

    _, call_kwargs = mock_get.call_args
    assert call_kwargs.get("params") == {"station": "AAA", "limit": 10}


# ---------------------------------------------------------------------------
# Transient failures → SourceUnavailable
# ---------------------------------------------------------------------------


def test_connection_error_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requests.ConnectionError propagates as SourceUnavailable."""
    mock_get = MagicMock(side_effect=requests.ConnectionError("refused"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get("https://example.com/data")

    assert exc_info.value.__cause__ is not None


def test_timeout_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timeout is transient: retries the full backoff schedule then raises SourceUnavailable."""
    import microclimate.connectors.http as http_mod

    mock_get = MagicMock(side_effect=requests.Timeout("timed out"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert mock_get.call_count == len(http_mod._BACKOFF_SCHEDULE) + 1  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# HTTP error status → SourceUnavailable
# ---------------------------------------------------------------------------


def test_http_500_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx response (after raise_for_status) propagates as SourceUnavailable."""
    mock_get = MagicMock(return_value=_make_response("server error", status_code=500))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


def test_http_404_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx response (after raise_for_status) propagates as SourceUnavailable."""
    mock_get = MagicMock(return_value=_make_response("not found", status_code=404))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


# ---------------------------------------------------------------------------
# Non-transient RequestException subclasses → immediate SourceUnavailable
# ---------------------------------------------------------------------------


def test_other_request_exception_fails_immediately_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RequestException that is neither HTTPError, ConnectionError, nor Timeout is
    non-transient and must fail immediately (exactly 1 attempt, zero sleeps, __cause__ set).

    RetryError is used as the concrete exemplar here — it stands in for any
    RequestException subclass not matched by _is_transient.
    """
    from requests.exceptions import RetryError

    sleeps: list[float] = []
    monkeypatch.setattr("microclimate.connectors.http._sleep", sleeps.append)
    mock_get = MagicMock(side_effect=RetryError("Max retries exceeded"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get("https://example.com/data")

    assert mock_get.call_count == 1
    assert sleeps == []
    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Float params (geographic connectors pass lat/lon)
# ---------------------------------------------------------------------------


def test_success_passes_float_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Float query params (e.g. lat/lon) are forwarded to Session.get."""
    mock_get = MagicMock(return_value=_make_response("data"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    http_get("https://example.com/data", params={"lat": 51.5, "lon": -0.12})

    _, call_kwargs = mock_get.call_args
    assert call_kwargs.get("params") == {"lat": 51.5, "lon": -0.12}


# ---------------------------------------------------------------------------
# http_get_bytes — success path
# ---------------------------------------------------------------------------


def test_get_bytes_success_returns_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """http_get_bytes returns response.content (bytes) on a 200 OK."""
    binary_body = b"\x1f\x8b\x08\x00GRIB binary payload"
    mock_get = MagicMock(return_value=_make_response(content=binary_body))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    result = http_get_bytes("https://example.com/file.grib2")

    assert result == binary_body
    assert isinstance(result, bytes)


def test_get_bytes_success_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query params are forwarded to Session.get by http_get_bytes."""
    mock_get = MagicMock(return_value=_make_response(content=b"data"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    http_get_bytes("https://example.com/file.grib2", params={"run": "00", "step": 3})

    _, call_kwargs = mock_get.call_args
    assert call_kwargs.get("params") == {"run": "00", "step": 3}


# ---------------------------------------------------------------------------
# http_get_bytes — failure path → SourceUnavailable
# ---------------------------------------------------------------------------


def test_get_bytes_http_error_raises_source_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-2xx HTTP response from http_get_bytes raises SourceUnavailable."""
    mock_get = MagicMock(return_value=_make_response(status_code=404))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get_bytes("https://example.com/missing.grib2")

    assert exc_info.value.__cause__ is not None


def test_get_bytes_connection_error_raises_source_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requests.ConnectionError from http_get_bytes raises SourceUnavailable."""
    mock_get = MagicMock(side_effect=requests.ConnectionError("refused"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get_bytes("https://example.com/file.grib2")

    assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Explicit backoff loop (ADR-0020)
# ---------------------------------------------------------------------------


def test_transient_502_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 502 is retried; the next attempt's body is returned."""
    mock_get = MagicMock(
        side_effect=[_make_response("bad", status_code=502), _make_response("recovered")]
    )
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    assert http_get("https://example.com/data") == "recovered"
    assert mock_get.call_count == 2


def test_backoff_sleeps_follow_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A persistent 502 sleeps through the whole schedule, one attempt per slot + 1."""
    import microclimate.connectors.http as http_mod

    sleeps: list[float] = []
    monkeypatch.setattr("microclimate.connectors.http._sleep", sleeps.append)
    mock_get = MagicMock(return_value=_make_response("bad", status_code=502))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert sleeps == list(http_mod._BACKOFF_SCHEDULE)  # pyright: ignore[reportPrivateUsage]
    assert mock_get.call_count == len(http_mod._BACKOFF_SCHEDULE) + 1  # pyright: ignore[reportPrivateUsage]


def test_exhaustion_message_names_url_attempts_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The give-up error carries the URL, attempt count, elapsed time, and the cause."""
    import re

    # _make_response builds a message-less HTTPError (str(exc) == ""), so attach a
    # realistic message here — the assertion below needs the cause text to survive.
    resp = _make_response("bad", status_code=502)
    resp.raise_for_status.side_effect = requests.HTTPError(
        "502 Server Error: Bad Gateway",
        response=resp,  # type: ignore[call-arg]
    )
    mock_get = MagicMock(return_value=resp)
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get("https://example.com/data")

    msg = str(exc_info.value)
    assert "https://example.com/data" in msg
    assert "6 attempts" in msg
    assert re.search(r"over \d+s", msg), msg
    assert "502" in msg


def test_http_404_fails_immediately_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx won't heal: exactly one attempt, no sleeping."""
    sleeps: list[float] = []
    monkeypatch.setattr("microclimate.connectors.http._sleep", sleeps.append)
    mock_get = MagicMock(return_value=_make_response("nope", status_code=404))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert mock_get.call_count == 1
    assert sleeps == []


def test_connection_error_retries_like_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection errors are transient: the full schedule is consumed before giving up."""
    import microclimate.connectors.http as http_mod

    mock_get = MagicMock(side_effect=requests.ConnectionError("refused"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")

    assert mock_get.call_count == len(http_mod._BACKOFF_SCHEDULE) + 1  # pyright: ignore[reportPrivateUsage]


def test_backoff_schedule_is_bounded() -> None:
    """Total backoff stays within the ~2-3 min budget (spec) — bounded, not unbounded."""
    import microclimate.connectors.http as http_mod

    total = sum(http_mod._BACKOFF_SCHEDULE)  # pyright: ignore[reportPrivateUsage]
    assert 60 <= total <= 300
