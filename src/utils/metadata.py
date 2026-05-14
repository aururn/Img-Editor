"""PNG metadata embedding (A1111 compatible-ish)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .logger import get_logger

logger = get_logger(__name__)


def build_parameters_string(meta: dict[str, Any]) -> str:
    """Build an A1111 style ``parameters`` string.

    Expected keys: prompt, negative_prompt, steps, sampler, cfg_scale,
    seed, width, height, model, loras (list of {"name", "weight"}),
    mode, mask_blur.
    """
    prompt = meta.get("prompt", "")
    negative = meta.get("negative_prompt", "")

    parts = [
        f"Steps: {meta.get('steps', '')}",
        f"Sampler: {meta.get('sampler', '')}",
        f"CFG scale: {meta.get('cfg_scale', '')}",
        f"Seed: {meta.get('seed', '')}",
        f"Size: {meta.get('width', '')}x{meta.get('height', '')}",
        f"Model: {meta.get('model', '')}",
    ]

    loras = meta.get("loras") or []
    if loras:
        lora_str = ", ".join(
            f"{spec['name']}:{spec['weight']}" for spec in loras
        )
        parts.append(f"LoRAs: {lora_str}")

    embeddings = meta.get("embeddings") or []
    if embeddings:
        emb_str = ", ".join(
            f"{spec.get('name', '')}<{spec.get('token', '')}>" for spec in embeddings
        )
        parts.append(f"Embeddings: {emb_str}")

    if meta.get("mode"):
        parts.append(f"Mode: {meta['mode']}")

    if meta.get("mask_blur") is not None and meta.get("mode", "").startswith(
        "inpaint"
    ):
        parts.append(f"Mask blur: {meta['mask_blur']}")

    controlnet = meta.get("controlnet") or {}
    if controlnet:
        parts.append(
            "ControlNet: "
            f"{controlnet.get('model', '')}:{controlnet.get('scale', '')}"
        )

    ip_adapter = meta.get("ip_adapter") or {}
    if ip_adapter:
        parts.append(
            f"IP-Adapter: {ip_adapter.get('name', '')}:{ip_adapter.get('scale', '')}"
        )

    regional = meta.get("regional") or {}
    if regional:
        parts.append(
            f"Regional: {regional.get('layout', '')}; ratios={regional.get('ratios', '')}"
        )
        if regional.get("lora_stop_step"):
            parts.append(f"Regional LoRA stop step: {regional.get('lora_stop_step')}")
        if regional.get("lora_negative_text_encoder_ratios"):
            parts.append(
                "Regional LoRA negative TE: "
                + str(regional.get("lora_negative_text_encoder_ratios"))
            )
        if regional.get("lora_negative_unet_ratios"):
            parts.append(
                "Regional LoRA negative U-Net: "
                + str(regional.get("lora_negative_unet_ratios"))
            )
        def _fmt_loras(items):
            return ", ".join(
                f"{s['name']}:{s.get('weight', '')}" for s in items
            )

        common_loras = regional.get("common_loras") or []
        if common_loras:
            parts.append(
                "Regional common LoRAs: "
                + _fmt_loras(common_loras)
            )
        base_loras = regional.get("base_loras") or []
        if base_loras:
            parts.append(
                "Regional base LoRAs: "
                + _fmt_loras(base_loras)
            )
        region_loras = regional.get("region_loras") or []
        for idx, row in enumerate(region_loras):
            if not row:
                continue
            parts.append(
                f"Regional region {idx + 1} LoRAs: "
                + _fmt_loras(row)
            )

    lines = [prompt]
    if negative:
        lines.append(f"Negative prompt: {negative}")
    lines.append(", ".join(parts))
    return "\n".join(lines)


def save_image_with_metadata(
    image: Image.Image, path: str | Path, meta: dict[str, Any]
) -> Path:
    info = PngInfo()
    info.add_text("parameters", build_parameters_string(meta))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", pnginfo=info)
    logger.debug("Wrote PNG with metadata: %s", path)
    return path
