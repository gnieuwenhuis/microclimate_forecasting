from __future__ import annotations

from pathlib import Path

import pytest

from microclimate.publication.registry_store import read_registry


def test_read_registry_stubbed(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        read_registry(tmp_path / "registry.json")
