from __future__ import annotations

from datetime import UTC, datetime

from microclimate.contracts.registry import (
    RegistryEntry,
    RegistryManifest,
    manifest_key,
)


def test_manifest_roundtrip() -> None:
    entry = RegistryEntry(
        version="1.0.0",
        release_asset_url="https://example/asset",
        promoted_at=datetime(2026, 5, 30, tzinfo=UTC),
        holdout_metrics={"mae_skill": 0.12},
    )
    manifest = RegistryManifest(entries={manifest_key("lethbridge", "temp"): entry})
    assert manifest.entries["lethbridge/temp"].version == "1.0.0"
