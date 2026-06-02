from __future__ import annotations

from pathlib import Path


def test_attribution_text_mentions_openmeteo_and_eccc() -> None:
    txt = Path("scripts/training_store_attribution.txt").read_text()
    assert "Open-Meteo" in txt and "CC-BY" in txt.replace("CC BY", "CC-BY")
    assert "Environment and Climate Change Canada" in txt


def test_inference_attribution_constant() -> None:
    from microclimate.pipelines.inference import _ATTRIBUTION  # type: ignore[reportPrivateUsage]

    joined = " ".join(_ATTRIBUTION)
    assert "Open-Meteo" in joined and "Environment and Climate Change Canada" in joined
