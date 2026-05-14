"""In-memory generation history."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class HistoryItem:
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)
    saved_path: Path | None = None


class History:
    def __init__(self, max_items: int = 20) -> None:
        self._items: deque[HistoryItem] = deque(maxlen=max_items)

    def add(self, item: HistoryItem) -> None:
        self._items.appendleft(item)

    def items(self) -> list[HistoryItem]:
        return list(self._items)

    def gallery(self) -> list[Image.Image]:
        return [item.image for item in self._items]

    def __len__(self) -> int:
        return len(self._items)
