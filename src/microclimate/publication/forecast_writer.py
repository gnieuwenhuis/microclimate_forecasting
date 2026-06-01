"""Write a ForecastDocument to JSON — only through the validated model (L5)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from microclimate.contracts.forecast import ForecastDocument


def write_forecast(doc: ForecastDocument, path: Path) -> None:
    """Atomically write the forecast document as JSON (temp file + os.replace).

    The ForecastDocument is schema-valid by construction (Pydantic), so dumping the model is
    the validation boundary. Atomic so a crashed run never leaves a half-written JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{uuid.uuid4().hex}.tmp"
    tmp.write_text(doc.model_dump_json(indent=2))
    os.replace(tmp, path)
