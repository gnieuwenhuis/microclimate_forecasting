from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from microclimate.contracts.forecast import ForecastDocument, ForecastStep


def _doc(**overrides: object) -> ForecastDocument:
    base: dict[str, object] = dict(
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
                pop=0.2,
            )
        ],
    )
    base.update(overrides)
    return ForecastDocument(**base)  # type: ignore[arg-type]


def test_valid_document() -> None:
    assert _doc().status == "ok"


def test_pop_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ForecastStep(
            lead_hour=1,
            valid_time=datetime(2026, 5, 30, 1, tzinfo=UTC),
            temp_c=11.0,
            pop=1.5,
        )


def test_empty_attribution_rejected() -> None:
    with pytest.raises(ValidationError):
        _doc(attribution=[])
