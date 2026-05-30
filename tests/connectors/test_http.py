"""Tests for the shared HTTP GET helper (L2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from microclimate.connectors.base import SourceUnavailable
from microclimate.connectors.http import http_get

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp  # type: ignore[call-arg]
        )
    return resp


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
    """A requests.Timeout propagates as SourceUnavailable."""
    mock_get = MagicMock(side_effect=requests.Timeout("timed out"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


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
# Retry configuration (structural — no real sleeps)
# ---------------------------------------------------------------------------


def test_retry_adapter_is_mounted() -> None:
    """The session has an HTTPAdapter with bounded Retry mounted on both https:// and http://."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    import microclimate.connectors.http as http_mod

    https_adapter = http_mod._SESSION.get_adapter("https://example.com")  # type: ignore[reportPrivateUsage]
    assert isinstance(https_adapter, HTTPAdapter)

    https_retry: Retry = https_adapter.max_retries  # type: ignore[assignment]
    assert isinstance(https_retry, Retry)
    assert https_retry.total is not None
    assert https_retry.total <= 5, "retries must be bounded (<=5)"

    http_adapter = http_mod._SESSION.get_adapter("http://example.com")  # type: ignore[reportPrivateUsage]
    assert isinstance(http_adapter, HTTPAdapter)

    http_retry: Retry = http_adapter.max_retries  # type: ignore[assignment]
    assert isinstance(http_retry, Retry)


# ---------------------------------------------------------------------------
# RetryError (exhausted retries) → SourceUnavailable
# ---------------------------------------------------------------------------


def test_retry_error_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exhausted-retry requests.RetryError propagates as SourceUnavailable."""
    from requests.exceptions import RetryError

    mock_get = MagicMock(side_effect=RetryError("Max retries exceeded"))
    monkeypatch.setattr("microclimate.connectors.http._SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable) as exc_info:
        http_get("https://example.com/data")

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
