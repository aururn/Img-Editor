"""Mask construction from detection boxes or user canvas."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

from .eye_detector import EyeBox


class MaskBuilder:
    @staticmethod
    def from_boxes(
        image_size: tuple[int, int],
        boxes: list[EyeBox],
        padding: int = 8,
        feather: int = 12,
    ) -> Image.Image:
        mask = Image.new("L", image_size, 0)
        if not boxes:
            return mask
        draw = ImageDraw.Draw(mask)
        for box in boxes:
            x1 = max(0, box.x - padding)
            y1 = max(0, box.y - padding)
            x2 = min(image_size[0], box.x + box.w + padding)
            y2 = min(image_size[1], box.y + box.h + padding)
            draw.ellipse((x1, y1, x2, y2), fill=255)
        if feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
        return mask

    @staticmethod
    def from_user_canvas(canvas_image) -> Image.Image:
        """Extract a binary-ish mask from a Gradio ImageEditor result.

        Gradio ``ImageEditor`` returns a dict like::

            {"background": ..., "layers": [...], "composite": ...}

        Layers contain alpha-channel painted strokes; we accumulate them.
        Falls back to the alpha channel of ``composite`` if no layers found.
        """
        if isinstance(canvas_image, dict):
            layers = canvas_image.get("layers") or []
            background = canvas_image.get("background")
            composite = canvas_image.get("composite")

            base_size = None
            if background is not None:
                base_size = background.size
            elif composite is not None:
                base_size = composite.size
            elif layers:
                base_size = layers[0].size

            if base_size is None:
                raise ValueError("ImageEditor returned no usable layers")

            mask = Image.new("L", base_size, 0)
            for layer in layers:
                if layer.mode == "RGBA":
                    alpha = layer.split()[-1]
                else:
                    alpha = layer.convert("L")
                if alpha.size != base_size:
                    alpha = alpha.resize(base_size, Image.BILINEAR)
                mask = Image.composite(
                    Image.new("L", base_size, 255), mask, alpha
                )

            if not layers and composite is not None and composite.mode == "RGBA":
                mask = composite.split()[-1]
            return mask

        # Raw PIL Image with alpha
        if isinstance(canvas_image, Image.Image):
            if canvas_image.mode == "RGBA":
                return canvas_image.split()[-1]
            return canvas_image.convert("L")
        raise TypeError(f"Unsupported canvas type: {type(canvas_image)!r}")

    @staticmethod
    def feather(mask: Image.Image, radius: int) -> Image.Image:
        if radius <= 0:
            return mask
        return mask.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def overlay_preview(
        image: Image.Image, mask: Image.Image, color: tuple[int, int, int] = (255, 80, 80)
    ) -> Image.Image:
        """Return ``image`` with the masked region tinted in ``color``."""
        rgba = image.convert("RGBA")
        if mask.size != rgba.size:
            mask = mask.resize(rgba.size, Image.BILINEAR)
        overlay = Image.new("RGBA", rgba.size, (*color, 0))
        red = Image.new("RGBA", rgba.size, (*color, 130))
        overlay = Image.composite(red, overlay, mask)
        return Image.alpha_composite(rgba, overlay).convert("RGB")
