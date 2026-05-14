"""Preset save/load."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _sanitize(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name)
    return name or "preset"


def list_presets(presets_dir: str | Path) -> list[str]:
    presets_dir = Path(presets_dir)
    if not presets_dir.exists():
        return []
    return sorted(p.stem for p in presets_dir.glob("*.json"))


def load_preset(presets_dir: str | Path, name: str) -> dict[str, Any]:
    path = Path(presets_dir) / f"{_sanitize(name)}.json"
    if not path.exists():
        raise FileNotFoundError(f"preset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_preset(presets_dir: str | Path, name: str, data: dict[str, Any]) -> Path:
    presets_dir = Path(presets_dir)
    presets_dir.mkdir(parents=True, exist_ok=True)
    path = presets_dir / f"{_sanitize(name)}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved preset %s", path)
    return path
