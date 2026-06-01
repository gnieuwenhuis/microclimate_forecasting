# tests/publication/test_forecast_writer.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from microclimate.contracts.forecast import (
    FORECAST_SCHEMA_VERSION,
    ForecastDocument,
    ForecastStep,
)
from microclimate.publication.forecast_writer import write_forecast


def _doc() -> ForecastDocument:
    t0 = datetime(2026, 6, 1, 0, tzinfo=UTC)
    return ForecastDocument(
        schema_version=FORECAST_SCHEMA_VERSION,
        deployment_id="lethbridge",
        issue_time=t0,
        last_updated=t0,
        status="ok",
        model_versions={"temp": "baseline", "pop": "baseline"},
        attribution=["Data Source: Environment and Climate Change Canada (ECCC)"],
        series=[ForecastStep(lead_hour=1, valid_time=t0, temp_c=10.0, pop=0.0)],
    )


def test_write_forecast_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "lethbridge.json"  # parent dir does not exist yet
    doc = _doc()
    write_forecast(doc, path)
    assert path.exists()
    assert ForecastDocument.model_validate_json(path.read_text()) == doc


def test_write_forecast_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "lethbridge.json"
    write_forecast(_doc(), path)
    assert list(tmp_path.glob(".*.tmp")) == []
    assert path.exists()
