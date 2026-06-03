"""Regression guards for the static dashboard thin client (no JS runner in CI)."""

import re
from pathlib import Path

from microclimate.contracts.forecast import FORECAST_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_JS = REPO_ROOT / "dashboard" / "app.js"


def test_app_js_supported_major_matches_contract():
    text = APP_JS.read_text(encoding="utf-8")
    match = re.search(r'SUPPORTED_SCHEMA_MAJOR\s*=\s*"(\d+)"', text)
    assert match, "dashboard/app.js must define SUPPORTED_SCHEMA_MAJOR"
    assert match.group(1) == FORECAST_SCHEMA_VERSION.split(".")[0]


INFERENCE_YML = REPO_ROOT / ".github" / "workflows" / "inference.yml"


def test_inference_workflow_publishes_dashboard():
    text = INFERENCE_YML.read_text(encoding="utf-8")
    for asset in ("dashboard/index.html", "dashboard/app.js", "dashboard/styles.css"):
        assert asset in text, f"inference.yml must copy {asset} into the gh-pages worktree"
    assert "gp/.nojekyll" in text, "inference.yml must create gp/.nojekyll on gh-pages"
