"""Shared test helpers for connector tests."""

from __future__ import annotations

import pathlib

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "envcanada"


def load_fixture(name: str) -> str:
    """Read a fixture file from the envcanada fixture directory."""
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8-sig")


def make_fetcher(csv_text: str):
    """Return a fetcher callable that always returns the given CSV text."""

    def fetcher(station_id: str, year: int, month: int) -> str:  # noqa: ARG001
        return csv_text

    return fetcher
