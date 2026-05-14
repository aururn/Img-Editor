"""Application configuration loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def app(self) -> dict[str, Any]:
        return self.raw.get("app", {})

    @property
    def paths(self) -> dict[str, str]:
        return self.raw.get("paths", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def detection(self) -> dict[str, Any]:
        return self.raw.get("detection", {})

    @property
    def defaults(self) -> dict[str, Any]:
        return self.raw.get("defaults", {})

    @property
    def limits(self) -> dict[str, Any]:
        return self.raw.get("limits", {})

    @property
    def history(self) -> dict[str, Any]:
        return self.raw.get("history", {})

    def path(self, key: str) -> Path:
        return Path(self.paths.get(key, f"./{key}"))


def load_config(path: str | Path = "./config.yaml") -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(raw=raw)
