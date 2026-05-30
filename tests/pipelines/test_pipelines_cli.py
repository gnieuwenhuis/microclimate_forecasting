from __future__ import annotations

import pytest

from microclimate.pipelines import inference, training


def test_run_inference_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        inference.run_inference("lethbridge")


def test_run_training_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        training.run_training("lethbridge")


def test_inference_cli_requires_deployment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["prog"])  # no --deployment
    with pytest.raises(SystemExit):
        inference.main()
