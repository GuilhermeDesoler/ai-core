from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def create_run_dir(base: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, data: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
