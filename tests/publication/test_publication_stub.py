from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from microclimate.contracts.forecast import ForecastDocument, ForecastStep
from microclimate.publication.forecast_writer import write_forecast
from microclimate.publication.registry_store import read_registry


def test_write_forecast_stubbed(tmp_path: Path) -> None:
    doc = ForecastDocument(
        schema_version="1",
        deployment_id="lethbridge",
        issue_time=datetime(2026, 5, 30, tzinfo=UTC),
        last_updated=datetime(2026, 5, 30, tzinfo=UTC),
        status="ok",
        model_versions={"temp": "1.0.0", "pop": "1.0.0"},
        attribution=["Data Source: Environment and Climate Change Canada"],
        series=[
            ForecastStep(
                lead_hour=1,
                valid_time=datetime(2026, 5, 30, 1, tzinfo=UTC),
                temp_c=11.0,
                pop=0.1,
            )
        ],
    )
    with pytest.raises(NotImplementedError):
        write_forecast(doc, tmp_path / "out.json")


def test_read_registry_stubbed(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        read_registry(tmp_path / "registry.json")
