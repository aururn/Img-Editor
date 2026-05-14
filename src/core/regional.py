"""A1111-style regional prompt parsing and mask generation.

The regional generation strategy implemented here is **staged inpaint with
per-region LoRA swap**:

1. A base image is generated (or supplied) using ``common_loras`` + ``base_prompt``.
2. For each region, the LoRA set is swapped to ``common_loras + region_loras[i]``
   and the region mask is inpainted with the region's prompt.

This avoids the "LoRA cross-contamination" problem (e.g. character A LoRA and
character B LoRA mixing into a third face when both are applied globally),
because each region is generated with only its own character LoRA active.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageDraw

from .lora_loader import LoRASpec


RegionalLayout = Literal["horizontal", "vertical", "grid", "mask"]

MAX_REGIONS = 4


@dataclass
class RegionalPromptSpec:
    enabled: bool = False
    layout: RegionalLayout = "horizontal"
    ratios: str = ""
    common_prompt: str = ""
    base_prompt: str = ""
    region_prompts: list[str] = field(default_factory=list)
    region_negative_prompts: list[str] = field(default_factory=list)
    use_base_pass: bool = False
    mask: Image.Image | None = None
    # Per-region staging
    common_loras: list[LoRASpec] = field(default_factory=list)
    region_loras: list[list[LoRASpec]] = field(default_factory=list)
    base_strength: float | None = None
    region_strength: float | None = None

    def to_metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "layout": self.layout,
            "ratios": self.ratios,
            "common_prompt": self.common_prompt,
            "base_prompt": self.base_prompt,
            "region_prompts": self.region_prompts,
            "region_negative_prompts": self.region_negative_prompts,
            "use_base_pass": self.use_base_pass,
            "has_mask": self.mask is not None,
            "common_loras": [
                {"name": s.name, "weight": s.weight} for s in self.common_loras
            ],
            "region_loras": [
                [{"name": s.name, "weight": s.weight} for s in row]
                for row in self.region_loras
            ],
            "base_strength": self.base_strength,
            "region_strength": self.region_strength,
        }


def split_breaks(text: str) -> list[str]:
    """Split A1111 Regional Prompter style BREAK blocks."""
    if not text:
        return []
    parts = re.split(r"(?:^|\n)\s*BREAK\s*(?:\n|$)|\s+BREAK\s+", text)
    return [p.strip(" ,\n\t") for p in parts if p.strip(" ,\n\t")]


def parse_region_lines(text: str) -> list[str]:
    if not text:
        return []
    break_parts = split_breaks(text)
    if break_parts:
        return break_parts
    return [line.strip() for line in text.splitlines() if line.strip()]


def combine_prompt(common: str, region: str) -> str:
    parts = [p.strip(" ,") for p in (common, region) if p and p.strip(" ,")]
    return ", ".join(parts)


def _parse_ratios(ratios: str, count: int) -> list[float]:
    if count <= 0:
        return []
    tokens = [t for t in re.split(r"[,:\s]+", ratios or "") if t]
    values: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    if len(values) < count:
        values.extend([1.0] * (count - len(values)))
    return values[:count]


def _cumulative_edges(total: int, ratios: list[float]) -> list[int]:
    denom = sum(ratios) or 1.0
    edges = [0]
    acc = 0.0
    for ratio in ratios[:-1]:
        acc += ratio
        edges.append(int(round(total * acc / denom)))
    edges.append(total)
    return edges


def _blank_mask(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def build_region_masks(spec: RegionalPromptSpec, size: tuple[int, int]) -> list[Image.Image]:
    """Create region masks in image pixel space."""
    count = max(1, len(spec.region_prompts))
    width, height = size

    if spec.layout == "mask" and spec.mask is not None:
        mask = spec.mask.convert("L")
        if mask.size != size:
            mask = mask.resize(size, Image.BILINEAR)
        return [mask]

    masks: list[Image.Image] = []
    if spec.layout == "vertical":
        ratios = _parse_ratios(spec.ratios, count)
        edges = _cumulative_edges(height, ratios)
        for i in range(count):
            mask = _blank_mask(size)
            ImageDraw.Draw(mask).rectangle((0, edges[i], width, edges[i + 1]), fill=255)
            masks.append(mask)
        return masks

    if spec.layout == "grid":
        cols = int(round(count ** 0.5))
        cols = max(1, cols)
        while cols * ((count + cols - 1) // cols) < count:
            cols += 1
        rows = (count + cols - 1) // cols
        cell_w = width / cols
        cell_h = height / rows
        for i in range(count):
            col = i % cols
            row = i // cols
            x1 = int(round(col * cell_w))
            y1 = int(round(row * cell_h))
            x2 = int(round((col + 1) * cell_w))
            y2 = int(round((row + 1) * cell_h))
            mask = _blank_mask(size)
            ImageDraw.Draw(mask).rectangle((x1, y1, x2, y2), fill=255)
            masks.append(mask)
        return masks

    ratios = _parse_ratios(spec.ratios, count)
    edges = _cumulative_edges(width, ratios)
    for i in range(count):
        mask = _blank_mask(size)
        ImageDraw.Draw(mask).rectangle((edges[i], 0, edges[i + 1], height), fill=255)
        masks.append(mask)
    return masks


def normalize_spec(
    enabled: bool,
    layout: str,
    ratios: str,
    common_prompt: str,
    base_prompt: str,
    region_prompt_text: str,
    region_negative_text: str,
    fallback_prompt: str,
    use_base_pass: bool,
    mask: Image.Image | None = None,
    common_loras: list[LoRASpec] | None = None,
    region_loras: list[list[LoRASpec]] | None = None,
    base_strength: float | None = None,
    region_strength: float | None = None,
) -> RegionalPromptSpec:
    prompts = parse_region_lines(region_prompt_text)
    if not prompts:
        prompts = split_breaks(fallback_prompt)
    if len(prompts) > MAX_REGIONS:
        prompts = prompts[:MAX_REGIONS]
    negatives = parse_region_lines(region_negative_text)
    if negatives and len(negatives) < len(prompts):
        negatives.extend([""] * (len(prompts) - len(negatives)))
    layout_key = (layout or "horizontal").strip().lower()
    if layout_key not in {"horizontal", "vertical", "grid", "mask"}:
        layout_key = "horizontal"

    region_loras = region_loras or []
    if len(region_loras) > len(prompts):
        region_loras = region_loras[: len(prompts)]
    elif len(region_loras) < len(prompts):
        region_loras = list(region_loras) + [[]] * (len(prompts) - len(region_loras))

    return RegionalPromptSpec(
        enabled=bool(enabled and prompts),
        layout=layout_key,  # type: ignore[arg-type]
        ratios=ratios or "",
        common_prompt=common_prompt or "",
        base_prompt=base_prompt or "",
        region_prompts=prompts,
        region_negative_prompts=negatives,
        use_base_pass=bool(use_base_pass),
        mask=mask,
        common_loras=list(common_loras or []),
        region_loras=region_loras,
        base_strength=base_strength,
        region_strength=region_strength,
    )

