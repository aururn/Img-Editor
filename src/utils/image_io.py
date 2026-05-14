"""Image I/O helpers."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from .logger import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    if img.format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {img.format}. Expected PNG/JPEG/WebP."
        )
    return img.convert("RGB")


def resize_if_needed(
    image: Image.Image, max_size: int = 2048
) -> tuple[Image.Image, bool]:
    """Resize image so its longest side does not exceed max_size.

    Returns (image, was_resized).
    """
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image, False
    scale = max_size / longest
    new_size = (int(w * scale), int(h * scale))
    logger.info("Resizing image %s -> %s", (w, h), new_size)
    return image.resize(new_size, Image.LANCZOS), True


def save_image(
    image: Image.Image,
    outputs_dir: str | Path,
    suffix: str = "",
    pnginfo=None,
) -> Path:
    outputs_dir = Path(outputs_dir)
    today = datetime.now().strftime("%Y%m%d")
    target_dir = outputs_dir / today
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%H%M%S_%f")[:-3]
    filename = f"{stamp}{suffix}.png"
    path = target_dir / filename
    image.save(path, format="PNG", pnginfo=pnginfo)
    logger.info("Saved image to %s", path)
    return path
