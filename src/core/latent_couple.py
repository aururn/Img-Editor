"""Latent Couple (hako-mikan method) for regional prompting.

At each UNet denoising step:
  1. Run the base/common UNet call with CFG.
  2. For each region, run a CFG UNet call with region-specific positive and
     negative embeddings.
  3. Blend negative and positive noise predictions separately using latent
     masks.
  4. Return the blended CFG pair; the pipeline applies guidance as usual.

LoRAs can be switched per region during UNet calls. Single-pass denoising; no seams.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable

import torch
from PIL import Image


@dataclass
class RegionCondition:
    """Per-region CFG text embeddings and pixel-space mask."""
    prompt_embeds: torch.Tensor            # [1, seq_len, 2048]
    negative_prompt_embeds: torch.Tensor   # [1, seq_len, 2048]
    pooled_embeds: torch.Tensor            # [1, 1280]
    negative_pooled_embeds: torch.Tensor   # [1, 1280]
    pil_mask: Image.Image                  # L-mode PIL mask in image pixel space
    base_ratio: float = 0.2                # 0.0 region-only, 1.0 base-only
    unet_loras: tuple[Any, ...] = ()       # Active LoRAs for this region's UNet call


def _pil_mask_to_latent_tensor(
    pil_mask: Image.Image,
    latent_h: int,
    latent_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    import numpy as np
    resized = pil_mask.convert("L").resize((latent_w, latent_h), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return t.unsqueeze(0).unsqueeze(0)  # [1, 1, H_lat, W_lat]


@contextlib.contextmanager
def latent_couple_context(
    unet,
    conditions: list[RegionCondition],
    *,
    base_weight: float = 0.2,
    set_loras: Callable[[tuple[Any, ...]], None] | None = None,
    base_loras: tuple[Any, ...] = (),
    lora_stop_step: int = 0,
):
    """Context manager patching unet.forward with Latent Couple blending.

    base_weight is used only as a fallback for conditions without their own
    base_ratio. A base ratio of 0.2 means 20% base prompt and 80% regional
    prompt inside that region.
    """
    if not conditions:
        yield
        return

    original_forward = unet.forward
    step_index = -1

    def _set_loras(specs: tuple[Any, ...]) -> None:
        if set_loras is not None:
            set_loras(specs)

    def _region_loras_for_step(specs: tuple[Any, ...]) -> tuple[Any, ...]:
        if lora_stop_step > 0 and step_index >= lora_stop_step:
            return base_loras
        return specs

    def _patched(
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict | None = None,
        **kwargs,
    ):
        batch_size = sample.shape[0]
        acw = added_cond_kwargs or {}

        if batch_size != 2:
            # Non-CFG mode: just run base unchanged
            _set_loras(base_loras)
            return original_forward(
                sample, timestep, encoder_hidden_states,
                added_cond_kwargs=acw, **kwargs,
            )

        nonlocal step_index
        step_index += 1

        # --- Base CFG call (neg=[:1] + pos=[1:]) ----------------------------
        _set_loras(base_loras)
        base_out = original_forward(
            sample, timestep, encoder_hidden_states,
            added_cond_kwargs=acw, **kwargs,
        )
        # Handle return_dict=True (ModelOutput) and return_dict=False (tuple)
        base_noise = base_out[0] if isinstance(base_out, (tuple, list)) else base_out.sample
        neg_noise = base_noise[:1]
        pos_noise = base_noise[1:]

        latent_h, latent_w = sample.shape[2], sample.shape[3]
        dev, dtype = sample.device, sample.dtype

        blended_neg = torch.zeros_like(neg_noise)
        blended_pos = torch.zeros_like(pos_noise)
        coverage = torch.zeros((1, 1, latent_h, latent_w), device=dev, dtype=dtype)

        # --- Per-region CFG UNet calls --------------------------------------
        for cond in conditions:
            _set_loras(_region_loras_for_step(tuple(getattr(cond, "unet_loras", ()))))
            mask = _pil_mask_to_latent_tensor(cond.pil_mask, latent_h, latent_w, dev, dtype)
            region_acw = dict(acw)
            region_acw["text_embeds"] = torch.cat(
                [
                    cond.negative_pooled_embeds.to(device=dev, dtype=dtype),
                    cond.pooled_embeds.to(device=dev, dtype=dtype),
                ],
                dim=0,
            )
            region_hidden = torch.cat(
                [
                    cond.negative_prompt_embeds.to(device=dev, dtype=dtype),
                    cond.prompt_embeds.to(device=dev, dtype=dtype),
                ],
                dim=0,
            )
            region_out = original_forward(
                sample, timestep,
                region_hidden,
                added_cond_kwargs=region_acw, **kwargs,
            )
            region_noise = region_out[0] if isinstance(region_out, (tuple, list)) else region_out.sample
            ratio = float(getattr(cond, "base_ratio", base_weight))
            ratio = min(1.0, max(0.0, ratio))
            blended_neg = blended_neg + mask * (ratio * neg_noise + (1.0 - ratio) * region_noise[:1])
            blended_pos = blended_pos + mask * (ratio * pos_noise + (1.0 - ratio) * region_noise[1:])
            coverage = coverage + mask

        _set_loras(base_loras)
        safe_coverage = coverage.clamp(min=1e-6)
        regional_neg = blended_neg / safe_coverage
        regional_pos = blended_pos / safe_coverage
        clamped_coverage = coverage.clamp(0.0, 1.0)
        final_neg = regional_neg * clamped_coverage + neg_noise * (1.0 - clamped_coverage)
        final_pos = regional_pos * clamped_coverage + pos_noise * (1.0 - clamped_coverage)
        combined = torch.cat([final_neg, final_pos], dim=0)

        if isinstance(base_out, (tuple, list)):
            return (combined,) + tuple(base_out[1:])
        return base_out.__class__(sample=combined)

    unet.forward = _patched
    try:
        yield
    finally:
        unet.forward = original_forward
        if set_loras is not None:
            set_loras(base_loras)
