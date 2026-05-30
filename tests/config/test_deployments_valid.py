from __future__ import annotations

import pytest

import microclimate.connectors  # noqa: F401  # type: ignore[reportUnusedImport]  (populates the registry)
from microclimate.config.loader import list_deployments, load_deployment
from microclimate.connectors.registry import validate_config_sources


@pytest.mark.parametrize("deployment_id", list_deployments())
def test_committed_deployment_is_valid(deployment_id: str) -> None:
    config = load_deployment(deployment_id)
    validate_config_sources(config)  # raises if any source is unregistered or non-deep
