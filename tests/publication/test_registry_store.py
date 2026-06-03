from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from microclimate.contracts.registry import RegistryEntry, RegistryManifest, manifest_key
from microclimate.publication.registry_store import promote, read_registry, write_registry


def _entry(v: str) -> RegistryEntry:
    return RegistryEntry(
        version=v,
        release_asset_url=f"https://example/{v}.joblib",
        promoted_at=datetime(2026, 6, 3, tzinfo=UTC),
        holdout_metrics={"mae": 1.0},
    )


def test_read_missing_returns_empty_manifest(tmp_path: Path) -> None:
    assert read_registry(tmp_path / "nope.json").entries == {}


def test_promote_then_roundtrip(tmp_path: Path) -> None:
    m = RegistryManifest()
    m2 = promote(m, "temp", "lethbridge", _entry("v1"))
    assert m.entries == {}  # original unchanged (immutable update)
    assert m2.entries[manifest_key("lethbridge", "temp")].version == "v1"

    path = tmp_path / "registry.json"
    write_registry(m2, path)
    back = read_registry(path)
    assert back.entries[manifest_key("lethbridge", "temp")].version == "v1"

    m3 = promote(back, "pop", "lethbridge", _entry("p1"))
    assert set(m3.entries) == {
        manifest_key("lethbridge", "temp"),
        manifest_key("lethbridge", "pop"),
    }
