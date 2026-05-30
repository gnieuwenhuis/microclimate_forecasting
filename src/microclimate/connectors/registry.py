"""Source registry + strategy-aware eligibility (L2). Imports config (downward) + base."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from microclimate.config.schema import DeploymentConfig
from microclimate.connectors.base import NWPSource, ObservationSource, Source

_REGISTRY: dict[str, type[Source]] = {}

S = TypeVar("S", bound=Source)


def register_source(key: str) -> Callable[[type[S]], type[S]]:
    def decorator(cls: type[S]) -> type[S]:
        if key in _REGISTRY:
            raise ValueError(f"Duplicate source key: {key!r}")
        _REGISTRY[key] = cls
        return cls

    return decorator


def is_registered(key: str) -> bool:
    return key in _REGISTRY


def registered_keys() -> set[str]:
    return set(_REGISTRY)


def get_source(key: str) -> Source:
    if key not in _REGISTRY:
        raise KeyError(f"Unregistered source: {key!r}")
    return _REGISTRY[key]()


def validate_config_sources(config: DeploymentConfig) -> None:
    """Raise unless every named source is registered and every station source is deep.

    v1 requires deep historical coverage for all observation sources (ADR-0008).
    """
    for key in config.enabled_sources:
        if not is_registered(key):
            raise ValueError(f"enabled_sources names unregistered source: {key!r}")

    for key in (config.nwp.live_connector, config.nwp.historical_connector):
        source = get_source(key)
        if not isinstance(source, NWPSource):
            raise ValueError(f"nwp connector {key!r} is not an NWPSource")

    for station in [config.target, *config.neighbors]:
        source = get_source(station.connector_key)
        if not isinstance(source, ObservationSource):
            raise ValueError(
                f"station connector {station.connector_key!r} is not an ObservationSource"
            )
        if source.historical_coverage != "deep":
            raise ValueError(
                f"source {station.connector_key!r} coverage "
                f"{source.historical_coverage!r} != 'deep'"
            )
