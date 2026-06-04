from __future__ import annotations

from pathlib import Path
from typing import NoReturn
from unittest.mock import MagicMock

import pytest

from microclimate.connectors.base import ForecastUnavailable, SourceUnavailable
from microclimate.pipelines import inference


def test_inference_cli_requires_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])  # no --deployment
    with pytest.raises(SystemExit):
        inference.main()


# ---------------------------------------------------------------------------
# Upstream unavailability is not failure (ADR-0020): warn and exit 0
# ---------------------------------------------------------------------------


def _wire_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point main() at a mocked config/sources so only run_inference's outcome matters."""
    monkeypatch.setattr("sys.argv", ["prog", "--deployment", "testdep"])
    monkeypatch.setenv("FORECAST_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    config = MagicMock()
    config.output.forecast_json = "forecast.json"
    config.target.connector_key = "obs_key"
    config.neighbors = []
    config.nwp.live_connector = "nwp_key"
    monkeypatch.setattr(inference, "load_deployment", MagicMock(return_value=config))
    monkeypatch.setattr(inference, "get_source", MagicMock())


def test_cli_source_unavailable_exits_zero_with_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SourceUnavailable → no exception, warning names the deployment and the cause."""
    _wire_cli(monkeypatch, tmp_path)
    cause = "gave up after 6 attempts over 155s fetching 'https://api.example': 502"

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise SourceUnavailable(cause)

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()  # returning (no SystemExit, no exception) IS exit 0

    out = capsys.readouterr().out
    assert "testdep" in out
    assert cause in out
    assert "upstream unavailable" in out


def test_cli_forecast_unavailable_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ForecastUnavailable (HRDPS run not published yet) gets the same graceful skip."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise ForecastUnavailable("no leads available for issue_time")

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()

    out = capsys.readouterr().out
    assert "upstream unavailable" in out
    assert "no leads available for issue_time" in out


def test_cli_emits_github_warning_annotation_under_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under GITHUB_ACTIONS a ::warning:: annotation is emitted; locally it is not."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise SourceUnavailable("502")

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()
    assert "::warning" not in capsys.readouterr().out  # local: no annotation noise

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    inference.main()
    assert "::warning" in capsys.readouterr().out


def test_cli_github_warning_annotation_escapes_newlines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Multiline exception messages are %-escaped so the ::warning:: annotation is not truncated."""
    _wire_cli(monkeypatch, tmp_path)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise SourceUnavailable("line one\nline two")

    monkeypatch.setattr(inference, "run_inference", boom)

    inference.main()

    out = capsys.readouterr().out
    warning_line = next(line for line in out.splitlines() if line.startswith("::warning"))
    assert "line one%0Aline two" in warning_line
    assert "line one\nline two" not in warning_line


def test_cli_bug_exceptions_still_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anything outside the expected-unavailability pair fails loudly (red run wanted)."""
    _wire_cli(monkeypatch, tmp_path)

    def boom(*args: object, **kwargs: object) -> NoReturn:
        raise KeyError("missing feature column")

    monkeypatch.setattr(inference, "run_inference", boom)

    with pytest.raises(KeyError):
        inference.main()
