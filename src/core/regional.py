"""Regional prompt parsing and mask generation for Regional Prompter.

This module implements the app-local, Diffusers-compatible subset of
hako-mikan's Regional Prompter syntax:
  - BREAK separated regions.
  - ADDROW / ADDCOL two-dimensional region prompts.
  - ADDBASE / ADDCOMM markers.
  - Divide ratios and base ratios.
  - Painted and color-split mask regional prompting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageDraw

from .lora_loader import LoRASpec


RegionalLayout = Literal["horizontal", "vertical", "grid", "mask"]

MAX_REGIONS = 4
DEFAULT_BASE_RATIO = 0.2
_LORA_TAG_RE = re.compile(
    r"<(?:lora|lyco|locon|loha):([^:>]+?)(?::([-+]?\d*\.?\d+))?(?::([-+]?\d*\.?\d+))?\s*>",
    flags=re.I,
)


@dataclass(frozen=True)
class RegionBox:
    """Region rectangle in normalized image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class RegionalPromptSpec:
    enabled: bool = False
    layout: RegionalLayout = "horizontal"
    ratios: str = ""
    base_ratios: str = ""
    overlay_ratio: float = 0.0
    use_base_prompt: bool = False
    use_common_prompt: bool = False
    use_common_negative: bool = False
    lora_negative_text_encoder_ratios: str = ""
    lora_negative_unet_ratios: str = ""
    lora_stop_step: int = 0
    common_prompt: str = ""
    base_prompt: str = ""
    region_prompts: list[str] = field(default_factory=list)
    region_negative_prompts: list[str] = field(default_factory=list)
    common_loras: list[LoRASpec] = field(default_factory=list)
    base_loras: list[LoRASpec] = field(default_factory=list)
    region_loras: list[list[LoRASpec]] = field(default_factory=list)
    region_base_ratios: list[float] = field(default_factory=list)
    region_boxes: list[RegionBox] = field(default_factory=list)
    mask: Image.Image | None = None

    def to_metadata(self) -> dict:
        return {
            "enabled": self.enabled,
            "layout": self.layout,
            "ratios": self.ratios,
            "base_ratios": self.base_ratios,
            "overlay_ratio": self.overlay_ratio,
            "use_base_prompt": self.use_base_prompt,
            "use_common_prompt": self.use_common_prompt,
            "use_common_negative": self.use_common_negative,
            "lora_negative_text_encoder_ratios": self.lora_negative_text_encoder_ratios,
            "lora_negative_unet_ratios": self.lora_negative_unet_ratios,
            "lora_stop_step": self.lora_stop_step,
            "common_prompt": self.common_prompt,
            "base_prompt": self.base_prompt,
            "region_prompts": self.region_prompts,
            "region_negative_prompts": self.region_negative_prompts,
            "common_loras": _lora_specs_to_metadata(self.common_loras),
            "base_loras": _lora_specs_to_metadata(self.base_loras),
            "region_loras": [
                _lora_specs_to_metadata(group) for group in self.region_loras
            ],
            "region_base_ratios": self.region_base_ratios,
            "region_boxes": [box.__dict__ for box in self.region_boxes],
            "has_mask": self.mask is not None,
        }


def _lora_specs_to_metadata(specs: list[LoRASpec]) -> list[dict]:
    return [
        {
            "name": spec.name,
            "weight": spec.weight,
            "text_weight": spec.text_weight,
            "unet_weight": spec.unet_weight,
        }
        for spec in specs
    ]


def split_breaks(text: str) -> list[str]:
    """Split A1111-style BREAK / AND blocks."""
    if not text:
        return []
    parts = re.split(
        r"(?:^|\n)\s*(?:BREAK|AND)\s*(?:\n|$)|\s+(?:BREAK|AND)\s+",
        text,
        flags=re.I,
    )
    return [p.strip(" ,\n\t") for p in parts if p.strip(" ,\n\t")]


def split_regional_clauses(text: str) -> list[str]:
    """Split region clauses on all Regional Prompter separators."""
    if not text:
        return []
    parts = re.split(
        r"(?:^|\n)\s*(?:BREAK|AND|ADDBASE|ADDCOMM)\s*(?:\n|$)|\s+(?:BREAK|AND|ADDBASE|ADDCOMM)\s+",
        text,
        flags=re.I,
    )
    return [p.strip(" ,\n\t") for p in parts if p.strip(" ,\n\t")]


def _split_marker(text: str, marker: str) -> list[str]:
    if not text:
        return []
    parts = re.split(rf"\s*{marker}\s*", text, flags=re.I)
    return [p.strip(" ,\n\t") for p in parts if p.strip(" ,\n\t")]


def _strip_markers(text: str) -> str:
    return re.sub(r"\b(?:ADDBASE|ADDCOMM)\b", "", text, flags=re.I).strip(" ,\n\t")


def _consume_marker_prefix(text: str) -> tuple[str, str, str]:
    """Extract ADDCOMM and ADDBASE prefixes.

    Returns (remaining_text, common_prefix, base_prefix). Marker order follows
    the common Regional Prompter form:
      common ADDCOMM base ADDBASE region1 BREAK region2
    """
    remaining = text or ""
    common_prefix = ""
    base_prefix = ""

    common_match = re.search(r"\bADDCOMM\b", remaining, flags=re.I)
    if common_match:
        common_prefix = remaining[: common_match.start()].strip(" ,\n\t")
        remaining = remaining[common_match.end() :].strip(" ,\n\t")

    base_match = re.search(r"\bADDBASE\b", remaining, flags=re.I)
    if base_match:
        base_prefix = remaining[: base_match.start()].strip(" ,\n\t")
        remaining = remaining[base_match.end() :].strip(" ,\n\t")

    return remaining, common_prefix, base_prefix


def parse_region_lines(text: str) -> list[str]:
    if not text:
        return []
    text = _strip_markers(text)
    break_parts = split_breaks(text)
    if break_parts:
        return break_parts
    return [line.strip() for line in text.splitlines() if line.strip()]


def combine_prompt(common: str, region: str) -> str:
    parts = [p.strip(" ,") for p in (common, region) if p and p.strip(" ,")]
    return ", ".join(parts)


def _safe_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _clean_after_lora_removal(text: str) -> str:
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,\n\t")


def extract_lora_tags(text: str) -> tuple[str, list[LoRASpec]]:
    """Remove A1111-style LoRA tags and return their adapter specs.

    Supported forms:
      <lora:name:0.8>        -> text encoder 0.8, UNet 0.8
      <lora:name:0.6:0.9>    -> text encoder 0.6, UNet 0.9
    """
    if not text:
        return "", []

    specs: list[LoRASpec] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if not name:
            return ""
        first = match.group(2)
        second = match.group(3)
        if second is None:
            weight = _safe_float(first, 0.8)
            specs.append(LoRASpec(name, weight))
        else:
            text_weight = _safe_float(first, 0.8)
            unet_weight = _safe_float(second, text_weight)
            specs.append(LoRASpec(name, unet_weight, text_weight, unet_weight))
        return ""

    cleaned = _LORA_TAG_RE.sub(_replace, text)
    return _clean_after_lora_removal(cleaned), specs


def parse_lora_specs(text: str) -> list[LoRASpec]:
    """Parse loose LoRA specs for the regional LoRA textbox.

    Besides ``<lora:name:weight>`` tags, this accepts:
      name@0.8
      name@0.6/0.9      (text encoder / UNet)
      name:0.8
      name:0.6:0.9      (text encoder / UNet)
    """
    cleaned, specs = extract_lora_tags(text or "")
    for raw in re.split(r"[\n,]+", cleaned):
        token = raw.strip(" \t,")
        if not token or token.upper() in {"BREAK", "AND", "ADDROW", "ADDCOL"}:
            continue

        match = re.match(
            r"^(.+?)\s*@\s*([-+]?\d*\.?\d+)(?:\s*/\s*([-+]?\d*\.?\d+))?$",
            token,
        )
        if match is None:
            match = re.match(
                r"^(.+?)\s*:\s*([-+]?\d*\.?\d+)(?:\s*:\s*([-+]?\d*\.?\d+))?$",
                token,
            )

        if match is None:
            specs.append(LoRASpec(token, 0.8))
            continue

        name = match.group(1).strip()
        first = _safe_float(match.group(2), 0.8)
        second = match.group(3)
        if second is None:
            specs.append(LoRASpec(name, first))
        else:
            unet_weight = _safe_float(second, first)
            specs.append(LoRASpec(name, unet_weight, first, unet_weight))
    return specs


def parse_region_lora_groups(text: str, count: int) -> list[list[LoRASpec]]:
    groups: list[list[LoRASpec]] = [[] for _ in range(max(0, count))]
    if not text or count <= 0:
        return groups
    parts = parse_region_lines(text)
    for index, part in enumerate(parts[:count]):
        groups[index].extend(parse_lora_specs(part))
    return groups


def parse_lora_negative_ratios(text: str, count: int) -> list[float]:
    """Parse LoRA leakage ratios for non-target regions.

    A blank value means 0.0 for every LoRA. A single value applies to all
    region LoRAs; shorter lists repeat their final value.
    """
    if count <= 0:
        return []
    values = _parse_ratio_values(text)
    if not values:
        return [0.0] * count
    if len(values) == 1:
        values = values * count
    elif len(values) < count:
        values.extend([values[-1]] * (count - len(values)))
    return [min(1.0, max(0.0, value)) for value in values[:count]]


def _parse_ratios(ratios: str, count: int) -> list[float]:
    if count <= 0:
        return []
    values = _parse_ratio_values(ratios)
    if len(values) < count:
        values.extend([1.0] * (count - len(values)))
    return values[:count]


def _parse_ratio_values(ratios: str) -> list[float]:
    tokens = [t for t in re.split(r"[,:\s]+", ratios or "") if t]
    values: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


def _parse_ratio_groups(ratios: str) -> list[list[float]]:
    groups: list[list[float]] = []
    for group in (ratios or "").split(";"):
        values = _parse_ratio_values(group)
        if values:
            groups.append(values)
    return groups


def _parse_base_ratios(ratios: str, count: int) -> list[float]:
    values = _parse_ratios(ratios, count)
    if not values:
        values = [DEFAULT_BASE_RATIO] * count
    if len(values) < count:
        values.extend([values[-1] if values else DEFAULT_BASE_RATIO] * (count - len(values)))
    return [min(1.0, max(0.0, value)) for value in values[:count]]


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


def _is_grayscale_mask(mask: Image.Image) -> bool:
    if mask.mode in {"1", "L"}:
        return True
    try:
        rgb = mask.convert("RGB")
        extrema = rgb.getextrema()
    except Exception:
        return False
    return all(channel[0] == channel[1] for channel in extrema)


def _split_mask_regions(mask: Image.Image, size: tuple[int, int], count: int) -> list[Image.Image]:
    """Split L/RGB/RGBA masks into one L mask per region."""
    if count <= 0:
        return []
    if mask.size != size:
        resample = Image.BILINEAR if _is_grayscale_mask(mask) else Image.NEAREST
        mask = mask.resize(size, resample)

    if _is_grayscale_mask(mask):
        gray = mask.convert("L")
        if count == 1:
            return [gray]
        values: list[int] = []
        seen: set[int] = set()
        for value in gray.getdata():
            if value <= 0 or value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) >= count:
                break
        if len(values) < count:
            raise ValueError("Mask layout needs one grayscale value or color per region prompt")
        masks: list[Image.Image] = []
        gray_data = list(gray.getdata())
        for value in values[:count]:
            region = Image.new("L", size, 0)
            region.putdata([255 if pixel == value else 0 for pixel in gray_data])
            masks.append(region)
        return masks

    rgba = mask.convert("RGBA")
    colors: list[tuple[int, int, int]] = []
    seen_colors: set[tuple[int, int, int]] = set()
    for r, g, b, a in rgba.getdata():
        color = (r, g, b)
        if a <= 0 or color == (0, 0, 0) or color in seen_colors:
            continue
        seen_colors.add(color)
        colors.append(color)
        if len(colors) >= count:
            break
    if len(colors) < count:
        if count == 1:
            return [rgba.split()[-1]]
        raise ValueError("Mask layout needs one visible color per region prompt")

    rgb_data = list(rgba.convert("RGB").getdata())
    masks = []
    for color in colors[:count]:
        region = Image.new("L", size, 0)
        region.putdata([255 if pixel == color else 0 for pixel in rgb_data])
        masks.append(region)
    return masks


def _overlay_ratio(value: float) -> float:
    return min(0.5, max(0.0, float(value or 0.0)))


def _expanded_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    overlay: float,
) -> tuple[int, int, int, int]:
    ow = (x2 - x1) * width * overlay
    oh = (y2 - y1) * height * overlay
    return (
        max(0, int(round(x1 * width - ow))),
        max(0, int(round(y1 * height - oh))),
        min(width, int(round(x2 * width + ow))),
        min(height, int(round(y2 * height + oh))),
    )


def _normalized_edges(ratios: list[float]) -> list[float]:
    denom = sum(ratios) or 1.0
    edges = [0.0]
    acc = 0.0
    for ratio in ratios[:-1]:
        acc += ratio
        edges.append(acc / denom)
    edges.append(1.0)
    return edges


def _parse_2d_regions(text: str, ratios: str) -> tuple[list[str], list[RegionBox]]:
    """Parse ADDROW/ADDCOL prompts into row-major region boxes.

    The implementation follows the common Regional Prompter mental model:
    ADDROW creates a new row, ADDCOL creates a new column inside that row.
    Ratio groups separated by semicolons map to row groups. If no ratio group
    matches the prompt shape, equal splits are used.
    """
    rows: list[list[str]] = []
    for row_text in _split_marker(_strip_markers(text), "ADDROW"):
        cols = _split_marker(row_text, "ADDCOL")
        rows.append(cols or [row_text.strip(" ,\n\t")])
    rows = [[cell for cell in row if cell] for row in rows if any(row)]
    if not rows:
        return [], []

    ratio_groups = _parse_ratio_groups(ratios)
    row_ratios = [1.0] * len(rows)
    col_ratios_by_row: list[list[float]] = []

    if len(ratio_groups) == len(rows) and all(
        len(group) in {1, len(rows[i])} for i, group in enumerate(ratio_groups)
    ):
        for i, row in enumerate(rows):
            group = ratio_groups[i]
            if len(group) == len(row):
                col_ratios_by_row.append(group)
            else:
                row_ratios[i] = group[0]
                col_ratios_by_row.append([1.0] * len(row))
    elif ratio_groups:
        first = ratio_groups[0]
        if len(first) == len(rows):
            row_ratios = first
            extra_groups = ratio_groups[1:]
            for i, row in enumerate(rows):
                group = extra_groups[i] if i < len(extra_groups) else []
                col_ratios_by_row.append(group if len(group) == len(row) else [1.0] * len(row))
        else:
            for row in rows:
                col_ratios_by_row.append(first if len(first) == len(row) else [1.0] * len(row))
    else:
        col_ratios_by_row = [[1.0] * len(row) for row in rows]

    prompts: list[str] = []
    boxes: list[RegionBox] = []
    y_edges = _normalized_edges(row_ratios)
    for row_index, row in enumerate(rows):
        x_edges = _normalized_edges(col_ratios_by_row[row_index])
        for col_index, prompt in enumerate(row):
            prompts.append(prompt)
            boxes.append(
                RegionBox(
                    x1=x_edges[col_index],
                    y1=y_edges[row_index],
                    x2=x_edges[col_index + 1],
                    y2=y_edges[row_index + 1],
                )
            )
    return prompts, boxes


def build_region_masks(spec: RegionalPromptSpec, size: tuple[int, int]) -> list[Image.Image]:
    """Create region masks in image pixel space."""
    count = max(1, len(spec.region_prompts))
    width, height = size

    if spec.layout == "mask" and spec.mask is not None:
        return _split_mask_regions(spec.mask, size, count)

    masks: list[Image.Image] = []
    overlay = _overlay_ratio(spec.overlay_ratio)
    if spec.region_boxes:
        for box in spec.region_boxes[:count]:
            mask = _blank_mask(size)
            ImageDraw.Draw(mask).rectangle(
                _expanded_rect(box.x1, box.y1, box.x2, box.y2, width, height, overlay),
                fill=255,
            )
            masks.append(mask)
        return masks

    if spec.layout == "vertical":
        ratios = _parse_ratios(spec.ratios, count)
        edges = _cumulative_edges(height, ratios)
        for i in range(count):
            mask = _blank_mask(size)
            region_h = edges[i + 1] - edges[i]
            pad = int(round(region_h * overlay))
            ImageDraw.Draw(mask).rectangle(
                (0, max(0, edges[i] - pad), width, min(height, edges[i + 1] + pad)),
                fill=255,
            )
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
            padx = int(round(cell_w * overlay))
            pady = int(round(cell_h * overlay))
            mask = _blank_mask(size)
            ImageDraw.Draw(mask).rectangle(
                (
                    max(0, x1 - padx),
                    max(0, y1 - pady),
                    min(width, x2 + padx),
                    min(height, y2 + pady),
                ),
                fill=255,
            )
            masks.append(mask)
        return masks

    # Default: horizontal
    ratios = _parse_ratios(spec.ratios, count)
    edges = _cumulative_edges(width, ratios)
    for i in range(count):
        mask = _blank_mask(size)
        region_w = edges[i + 1] - edges[i]
        pad = int(round(region_w * overlay))
        ImageDraw.Draw(mask).rectangle(
            (max(0, edges[i] - pad), 0, min(width, edges[i + 1] + pad), height),
            fill=255,
        )
        masks.append(mask)
    return masks


def parse_region_negatives(
    text: str,
    count: int,
    *,
    use_common_negative: bool = False,
) -> list[str]:
    if count <= 0 or not text or not text.strip():
        return []

    has_break = bool(re.search(r"\b(?:BREAK|AND)\b", text, flags=re.I))
    parts = parse_region_lines(text)
    if not parts:
        return []

    if not has_break and len(parts) == 1:
        return [parts[0]] * count

    common = ""
    if use_common_negative and parts:
        common = parts.pop(0)

    if len(parts) < count:
        parts.extend([""] * (count - len(parts)))

    out: list[str] = []
    for part in parts[:count]:
        out.append(combine_prompt(common, part) if common else part)
    return out


def normalize_spec(
    enabled: bool,
    layout: str,
    ratios: str,
    base_ratios: str = "",
    overlay_ratio: float = 0.0,
    use_base_prompt: bool = False,
    use_common_prompt: bool = False,
    use_common_negative: bool = False,
    lora_negative_text_encoder_ratios: str = "",
    lora_negative_unet_ratios: str = "",
    lora_stop_step: int = 0,
    common_prompt: str = "",
    region_prompt_text: str = "",
    region_negative_text: str = "",
    region_lora_text: str = "",
    fallback_prompt: str = "",
    mask: Image.Image | None = None,
) -> RegionalPromptSpec:
    original_text = region_prompt_text or ""
    source_text, common_prefix, base_prefix = _consume_marker_prefix(original_text)
    layout_key = (layout or "horizontal").strip().lower()
    has_2d_markers = bool(re.search(r"\b(?:ADDROW|ADDCOL)\b", source_text, flags=re.I))
    if has_2d_markers or (";" in (ratios or "") and layout_key in {"grid", "horizontal", "vertical"}):
        prompts, boxes = _parse_2d_regions(source_text, ratios)
        layout_key = "grid"
    else:
        prompts = (
            split_regional_clauses(source_text)
            if re.search(r"\b(?:ADDBASE|ADDCOMM)\b", original_text, flags=re.I)
            else parse_region_lines(source_text)
        )
        boxes = []

    if not prompts:
        prompts = split_breaks(fallback_prompt)
        boxes = []

    base_prompt = fallback_prompt or ""
    if common_prefix:
        common_prompt = combine_prompt(common_prompt, common_prefix)
        use_common_prompt = True

    if base_prefix:
        base_prompt = base_prefix
        use_base_prompt = True

    if len(prompts) > MAX_REGIONS:
        prompts = prompts[:MAX_REGIONS]
        boxes = boxes[:MAX_REGIONS]

    common_prompt, common_loras = extract_lora_tags(common_prompt)
    base_prompt, base_loras = extract_lora_tags(base_prompt)
    explicit_region_loras = parse_region_lora_groups(region_lora_text, len(prompts))
    cleaned_prompts: list[str] = []
    region_loras: list[list[LoRASpec]] = []
    for i, prompt in enumerate(prompts):
        cleaned_prompt, inline_loras = extract_lora_tags(prompt)
        cleaned_prompts.append(cleaned_prompt)
        group = list(inline_loras)
        if i < len(explicit_region_loras):
            group.extend(explicit_region_loras[i])
        region_loras.append(group)
    prompts = cleaned_prompts

    negatives = parse_region_negatives(
        region_negative_text,
        len(prompts),
        use_common_negative=use_common_negative,
    )
    if layout_key not in {"horizontal", "vertical", "grid", "mask"}:
        layout_key = "horizontal"

    region_base_ratios = _parse_base_ratios(base_ratios, len(prompts))

    return RegionalPromptSpec(
        enabled=bool(enabled and prompts),
        layout=layout_key,  # type: ignore[arg-type]
        ratios=ratios or "",
        base_ratios=base_ratios or "",
        overlay_ratio=_overlay_ratio(overlay_ratio),
        use_base_prompt=bool(use_base_prompt),
        use_common_prompt=bool(use_common_prompt),
        use_common_negative=bool(use_common_negative),
        lora_negative_text_encoder_ratios=lora_negative_text_encoder_ratios or "",
        lora_negative_unet_ratios=lora_negative_unet_ratios or "",
        lora_stop_step=max(0, int(lora_stop_step or 0)),
        common_prompt=common_prompt or "",
        base_prompt=base_prompt,
        region_prompts=prompts,
        region_negative_prompts=negatives,
        common_loras=common_loras,
        base_loras=base_loras,
        region_loras=region_loras,
        region_base_ratios=region_base_ratios,
        region_boxes=boxes,
        mask=mask,
    )
