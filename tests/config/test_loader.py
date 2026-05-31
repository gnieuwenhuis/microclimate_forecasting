from __future__ import annotations

import pytest

from microclimate.config.loader import list_deployments, load_deployment


def test_lethbridge_is_listed() -> None:
    assert "lethbridge" in list_deployments()


def test_load_lethbridge_returns_config() -> None:
    config = load_deployment("lethbridge")
    assert config.deployment_id == "lethbridge"
    assert config.target.connector_key == "envcanada"
    assert config.target.station_id == "2265"


def test_missing_deployment_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_deployment("does-not-exist")
