from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "microclimate"


def test_import_linter_contracts_pass() -> None:
    exe = shutil.which("lint-imports")
    assert exe is not None, "import-linter not installed (uv sync)"
    result = subprocess.run([exe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_only_snapshot_builder_defines_build_snapshot() -> None:
    offenders = [
        str(path)
        for path in SRC.rglob("*.py")
        if path.name != "snapshot_builder.py" and "def build_snapshot" in path.read_text()
    ]
    assert not offenders, f"build_snapshot defined outside snapshot_builder: {offenders}"
