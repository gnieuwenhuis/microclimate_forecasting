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
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    result = http_get("https://example.com/data")

    assert result == "hello world"


def test_success_sends_descriptive_user_agent() -> None:
    """Session is initialised with the project User-Agent header."""
    import microclimate.connectors.http as http_mod

    ua = http_mod.SESSION.headers["User-Agent"]
    assert isinstance(ua, str)
    assert "microclimate-forecasting" in ua


def test_success_passes_explicit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """http_get calls Session.get with an explicit (connect, read) timeout tuple."""
    mock_get = MagicMock(return_value=_make_response("ok"))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    http_get("https://example.com/data")

    _, call_kwargs = mock_get.call_args
    timeout: object = call_kwargs.get("timeout")
    assert isinstance(timeout, tuple), "timeout must be a (connect, read) tuple"
    assert len(timeout) == 2  # type: ignore[arg-type]


def test_success_passes_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query params are forwarded to Session.get."""
    mock_get = MagicMock(return_value=_make_response("data"))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    http_get("https://example.com/data", params={"station": "AAA", "limit": 10})

    _, call_kwargs = mock_get.call_args
    assert call_kwargs.get("params") == {"station": "AAA", "limit": 10}


# ---------------------------------------------------------------------------
# Transient failures → SourceUnavailable
# ---------------------------------------------------------------------------


def test_connection_error_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requests.ConnectionError propagates as SourceUnavailable."""
    mock_get = MagicMock(side_effect=requests.ConnectionError("refused"))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


def test_timeout_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A requests.Timeout propagates as SourceUnavailable."""
    mock_get = MagicMock(side_effect=requests.Timeout("timed out"))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


# ---------------------------------------------------------------------------
# HTTP error status → SourceUnavailable
# ---------------------------------------------------------------------------


def test_http_500_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx response (after raise_for_status) propagates as SourceUnavailable."""
    mock_get = MagicMock(return_value=_make_response("server error", status_code=500))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


def test_http_404_raises_source_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx response (after raise_for_status) propagates as SourceUnavailable."""
    mock_get = MagicMock(return_value=_make_response("not found", status_code=404))
    monkeypatch.setattr("microclimate.connectors.http.SESSION.get", mock_get)

    with pytest.raises(SourceUnavailable):
        http_get("https://example.com/data")


# ---------------------------------------------------------------------------
# Retry configuration (structural — no real sleeps)
# ---------------------------------------------------------------------------


def test_retry_adapter_is_mounted() -> None:
    """The session has an HTTPAdapter with bounded Retry mounted on https://."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    import microclimate.connectors.http as http_mod

    adapter = http_mod.SESSION.get_adapter("https://example.com")
    assert isinstance(adapter, HTTPAdapter)

    retry: Retry = adapter.max_retries  # type: ignore[assignment]
    assert isinstance(retry, Retry)
    assert retry.total is not None
    assert retry.total <= 5, "retries must be bounded (<=5)"
