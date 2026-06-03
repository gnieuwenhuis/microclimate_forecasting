"""Regression guards for the static dashboard thin client (no JS runner in CI)."""

import re
from pathlib import Path

from microclimate.contracts.forecast import FORECAST_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "dashboard" / "app.js"


def test_app_js_supported_major_matches_contract():
    text = APP_JS.read_text()
    match = re.search(r'SUPPORTED_SCHEMA_MAJOR\s*=\s*"(\d+)"', text)
    assert match, "dashboard/app.js must define SUPPORTED_SCHEMA_MAJOR"
    assert match.group(1) == FORECAST_SCHEMA_VERSION.split(".")[0]
