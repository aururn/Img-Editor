"""Gradio event handlers for Img Editor."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from ..core import (
    AppConfig,
    GenerationRequest,
    InferenceService,
    LoRASpec,
    PipelineManager,
    build_diagnostics,
    discover_checkpoints,
    discover_controlnets,
    discover_embeddings,
    discover_ip_adapters,
)
from ..core.assets import EmbeddingSpec, IPAdapterSpec
from ..core.regional import build_region_masks, normalize_spec
from ..detection import EyeDetector, MaskBuilder
from ..utils import resize_if_needed, save_image_with_metadata
from ..utils.logger import get_logger
from . import presets
from .history import History, HistoryItem

logger = get_logger(__name__)


TXT2IMG = "Text to image"
IMG2IMG = "Image to image"
INPAINT_MANUAL = "Inpaint manual"
INPAINT_AUTO = "Inpaint auto"
MODE_MAP = {
    TXT2IMG: "txt2img",
    IMG2IMG: "img2img",
    INPAINT_MANUAL: "inpaint_manual",
    INPAINT_AUTO: "inpaint_auto",
}


@dataclass
class HandlerContext:
    config: AppConfig
    pipeline_manager: PipelineManager
    inference: InferenceService
    detector: EyeDetector
    history: History


def build_context(config: AppConfig) -> HandlerContext:
    pm = PipelineManager(config)
    inference = InferenceService(pm)
    det_cfg = config.detection
    detector = EyeDetector(
        model_path=config.path("detection_dir") / det_cfg.get("eye_model", ""),
        device=pm.device,
        confidence_threshold=float(det_cfg.get("confidence_threshold", 0.35)),
    )
    history = History(max_items=int(config.history.get("max_items", 20)))
    return HandlerContext(config, pm, inference, detector, history)


# ---------------------------------------------------------------------- discovery


def diagnostics_markdown(ctx: HandlerContext) -> str:
    return build_diagnostics(ctx.config).as_markdown()


def _as_rows(data: Any) -> list[list[Any]]:
    if data is None:
        return []
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.values.tolist()
    except Exception:
        pass
    if isinstance(data, list):
        return data
    return []


def lora_choices(ctx: HandlerContext, query: str | None = None) -> list[tuple[str, str]]:
    q = (query or "").strip().lower()
    choices: list[tuple[str, str]] = []
    for entry in ctx.pipeline_manager.lora_loader.discover():
        badges = []
        if entry.nsfw:
            badges.append("18+")
        badges.extend(entry.warnings)
        label = entry.name + (f"  [{', '.join(badges)}]" if badges else "")
        haystack = " ".join(
            [
                entry.name,
                entry.file,
                entry.title or "",
                entry.base_model or "",
                " ".join(badges),
            ]
        ).lower()
        if q and q not in haystack:
            continue
        choices.append((label, entry.name))
    return choices


def checkpoint_choices(ctx: HandlerContext) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in discover_checkpoints(ctx.config.path("checkpoints_dir")):
        label = item.name
        if item.path != item.name:
            label = f"{item.name} ({item.path})"
        if item.warnings:
            label = f"{label} [{'; '.join(item.warnings)}]"
        choices.append((label, item.path))
    return choices


def default_checkpoint(ctx: HandlerContext) -> str:
    configured = str(ctx.config.model.get("base_checkpoint", "") or "")
    values = [value for _, value in checkpoint_choices(ctx)]
    if configured in values:
        return configured
    return values[0] if values else configured


def parse_lora_table(names: list[str] | None, ctx: HandlerContext) -> list[LoRASpec]:
    if not names:
        return []
    known = {e.name: e for e in ctx.pipeline_manager.lora_loader.discover()}
    specs: list[LoRASpec] = []
    for name in names:
        name = str(name)
        if not name:
            continue
        entry = known.get(name)
        weight = entry.recommended_weight if entry else 0.8
        specs.append(LoRASpec(name, weight))
    return specs


def embedding_choices(ctx: HandlerContext, query: str | None = None) -> list[tuple[str, str]]:
    q = (query or "").strip().lower()
    choices: list[tuple[str, str]] = []
    for item in discover_embeddings(ctx.config.path("embeddings_dir")):
        warn = f" [{'; '.join(item.warnings)}]" if item.warnings else ""
        label = f"{item.name} <{item.token}>{warn}"
        if q and q not in label.lower():
            continue
        choices.append((label, item.name))
    return choices


def _embedding_map(ctx: HandlerContext) -> dict[str, EmbeddingSpec]:
    return {item.name: item for item in discover_embeddings(ctx.config.path("embeddings_dir"))}


def _ip_adapter_map(ctx: HandlerContext) -> dict[str, IPAdapterSpec]:
    return {item.name: item for item in discover_ip_adapters(ctx.config.path("ip_adapter_dir"))}


def controlnet_choices(ctx: HandlerContext) -> list[tuple[str, str]]:
    choices = [("None", "")]
    for item in discover_controlnets(ctx.config.path("controlnet_dir")):
        label = item.name
        if item.path != item.name:
            label = f"{item.name} ({item.path})"
        if item.warnings:
            label = f"{label} [{'; '.join(item.warnings)}]"
        choices.append((label, item.path))
    return choices


def _controlnet_value(ctx: HandlerContext, value: str | None) -> str:
    if not value:
        return ""
    value = str(value)
    for item in discover_controlnets(ctx.config.path("controlnet_dir")):
        aliases = {item.path, item.name, Path(item.path).stem}
        if value in aliases:
            return item.path
    return value


def ip_adapter_choices(ctx: HandlerContext) -> list[str]:
    return [""] + [item.name for item in discover_ip_adapters(ctx.config.path("ip_adapter_dir"))]


def list_presets_choices(ctx: HandlerContext) -> list[str]:
    return presets.list_presets(ctx.config.path("presets_dir"))


# ---------------------------------------------------------------------- asset uploads


def _max_upload_size_bytes(ctx: HandlerContext) -> int | None:
    try:
        mb = float(ctx.config.limits.get("max_file_size_mb", 0))
    except (TypeError, ValueError):
        return None
    return int(mb * 1024 * 1024) if mb > 0 else None


def _available_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _copy_uploads(
    filepaths: list[str] | str | None,
    target_dir: Path,
    exts: set[str],
    max_bytes: int | None = None,
) -> None:
    if not filepaths:
        return
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    target_dir.mkdir(parents=True, exist_ok=True)
    rejected: list[str] = []
    copied: list[str] = []
    for fp in filepaths:
        if not fp:
            continue
        src = Path(fp)
        if not src.is_file():
            rejected.append(f"{src.name}: not a file")
            continue
        if src.suffix.lower() not in exts:
            rejected.append(f"{src.name}: unsupported extension")
            continue
        if max_bytes is not None and src.stat().st_size > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            rejected.append(f"{src.name}: exceeds {limit_mb:g} MB")
            continue
        target = _available_path(target_dir / src.name)
        shutil.copy2(src, target)
        copied.append(target.name)
    if copied:
        logger.info("Uploaded assets to %s: %s", target_dir, ", ".join(copied))
    if rejected:
        raise gr.Error(f"Skipped files: {', '.join(rejected)}")


def upload_lora(ctx: HandlerContext, filepaths, query: str | None):
    _copy_uploads(
        filepaths,
        ctx.config.path("loras_dir"),
        {".safetensors"},
        _max_upload_size_bytes(ctx),
    )
    return gr.update(choices=lora_choices(ctx, query))


def upload_embedding(ctx: HandlerContext, filepaths, query: str | None):
    _copy_uploads(
        filepaths,
        ctx.config.path("embeddings_dir"),
        {".safetensors", ".pt", ".bin"},
        _max_upload_size_bytes(ctx),
    )
    return gr.update(choices=embedding_choices(ctx, query))


def upload_controlnet(ctx: HandlerContext, filepaths):
    _copy_uploads(
        filepaths,
        ctx.config.path("controlnet_dir"),
        {".safetensors", ".bin"},
        _max_upload_size_bytes(ctx),
    )
    return gr.update(choices=controlnet_choices(ctx))


def upload_ip_adapter(ctx: HandlerContext, filepaths):
    _copy_uploads(
        filepaths,
        ctx.config.path("ip_adapter_dir"),
        {".safetensors", ".bin"},
        _max_upload_size_bytes(ctx),
    )
    return gr.update(choices=ip_adapter_choices(ctx))


# ---------------------------------------------------------------------- prompt helpers


def insert_lora_triggers(ctx: HandlerContext, names: list[str] | None, prompt: str) -> str:
    entries = {e.name: e for e in ctx.pipeline_manager.lora_loader.discover()}
    triggers: list[str] = []
    for name in (names or []):
        entry = entries.get(str(name))
        if entry:
            triggers.extend(entry.trigger_words)
    return _append_unique(prompt, triggers)


def insert_embedding_tokens(
    ctx: HandlerContext, selected_names: list[str] | None, prompt: str
) -> str:
    entries = _embedding_map(ctx)
    tokens = [entries[name].token for name in (selected_names or []) if name in entries]
    return _append_unique(prompt, tokens)


def _append_unique(prompt: str | None, additions: list[str]) -> str:
    text = prompt or ""
    existing = {part.strip().lower() for part in text.split(",")}
    new_items = [item for item in additions if item and item.strip().lower() not in existing]
    if not new_items:
        return text
    if text.strip():
        return text.rstrip(" ,") + ", " + ", ".join(new_items)
    return ", ".join(new_items)


def send_to_inpaint(image):
    if image is None:
        raise gr.Error("No result image")
    return {"background": image, "layers": [], "composite": image}


def cancel_generation(ctx: HandlerContext) -> None:
    ctx.inference.request_stop()


# ---------------------------------------------------------------------- generation


def _canvas_image(canvas_state: Any) -> Image.Image | None:
    if canvas_state is None:
        return None
    if isinstance(canvas_state, dict):
        return canvas_state.get("background") or canvas_state.get("composite")
    return canvas_state


def _prepare_input_image(image: Any, max_size: int) -> Image.Image | None:
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image, _ = resize_if_needed(image.convert("RGB"), max_size=max_size)
    return image


def _mode_key(mode_label: str) -> str:
    return MODE_MAP.get(mode_label, "txt2img")


def _selected_embeddings(ctx: HandlerContext, names: list[str] | None) -> list[EmbeddingSpec]:
    mapping = _embedding_map(ctx)
    return [mapping[name] for name in (names or []) if name in mapping]


def _selected_ip_adapter(ctx: HandlerContext, name: str | None) -> IPAdapterSpec | None:
    if not name:
        return None
    return _ip_adapter_map(ctx).get(name)


def _regional_mask(layout: str, canvas_state: Any, image: Image.Image | None) -> Image.Image | None:
    if (layout or "").lower() != "mask":
        return None
    if canvas_state is None:
        return None
    try:
        mask = MaskBuilder.from_user_canvas_regions(canvas_state)
    except Exception:
        return None
    if image is not None and mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    return mask


def _regional_preview_base(
    mode_label: str,
    img2img_image,
    canvas_state,
    width: int,
    height: int,
    max_size: int,
) -> Image.Image:
    mode_key = _mode_key(mode_label)
    image: Image.Image | None = None
    if mode_key == "img2img":
        image = _prepare_input_image(img2img_image, max_size)
    elif mode_key.startswith("inpaint"):
        image = _prepare_input_image(_canvas_image(canvas_state), max_size)
    if image is not None:
        return image
    return Image.new("RGB", (max(8, int(width)), max(8, int(height))), (246, 248, 252))


def _render_mask_preview(base: Image.Image, masks: list[Image.Image]) -> Image.Image:
    colors = [
        (239, 68, 68),
        (34, 197, 94),
        (59, 130, 246),
        (245, 158, 11),
        (168, 85, 247),
        (20, 184, 166),
    ]
    out = base.convert("RGBA")
    size = out.size
    for index, mask in enumerate(masks):
        if mask.size != size:
            mask = mask.resize(size, Image.BILINEAR)
        color = colors[index % len(colors)]
        transparent = Image.new("RGBA", size, (*color, 0))
        tinted = Image.new("RGBA", size, (*color, 125))
        out = Image.alpha_composite(out, Image.composite(tinted, transparent, mask.convert("L")))
    return out.convert("RGB")


def _simple_ratio_count(ratios: str) -> int:
    if not ratios or ";" in ratios:
        return 0
    count = 0
    for token in re.split(r"[,:\s]+", ratios):
        if not token:
            continue
        try:
            if float(token) > 0:
                count += 1
        except ValueError:
            continue
    return count


def _validate_regional_ratio_count(spec, ratios: str) -> None:
    ratio_count = _simple_ratio_count(ratios)
    region_count = len(spec.region_prompts)
    if ratio_count > 1 and region_count < ratio_count and not spec.region_boxes:
        raise gr.Error(
            f"Ratios has {ratio_count} values, but Region prompts has only "
            f"{region_count} region(s). Add BREAK-separated region prompts."
        )


def preview_regional_masks(
    ctx: HandlerContext,
    mode_label: str,
    img2img_image,
    canvas_state,
    prompt: str,
    width: int,
    height: int,
    regional_enabled: bool,
    regional_layout: str,
    regional_ratios: str,
    regional_base_ratios: str,
    regional_overlay_ratio: float,
    regional_use_base_prompt: bool,
    regional_use_common_prompt: bool,
    regional_use_common_negative: bool,
    regional_common_prompt: str,
    regional_prompt_text: str,
    regional_negative_text: str,
    regional_lora_text: str,
    regional_lora_negative_te: str,
    regional_lora_negative_unet: str,
    regional_lora_stop_step: int,
):
    max_size = int(ctx.config.limits.get("max_image_size", 2048))
    base = _regional_preview_base(mode_label, img2img_image, canvas_state, width, height, max_size)
    regional_mask = _regional_mask(regional_layout, canvas_state, base)
    if (regional_layout or "").lower() == "mask" and regional_mask is None:
        raise gr.Error("Mask layout needs a painted color mask on the canvas")
    spec = normalize_spec(
        enabled=True,
        layout=regional_layout,
        ratios=regional_ratios,
        base_ratios=regional_base_ratios,
        overlay_ratio=float(regional_overlay_ratio or 0.0),
        use_base_prompt=bool(regional_use_base_prompt),
        use_common_prompt=bool(regional_use_common_prompt),
        use_common_negative=bool(regional_use_common_negative),
        lora_negative_text_encoder_ratios=regional_lora_negative_te,
        lora_negative_unet_ratios=regional_lora_negative_unet,
        lora_stop_step=int(regional_lora_stop_step or 0),
        common_prompt=regional_common_prompt,
        region_prompt_text=regional_prompt_text,
        region_negative_text=regional_negative_text,
        region_lora_text=regional_lora_text,
        fallback_prompt=prompt or "region",
        mask=regional_mask,
    )
    if not spec.enabled:
        raise gr.Error("Add at least one region prompt")
    _validate_regional_ratio_count(spec, regional_ratios)
    masks = build_region_masks(spec, base.size)
    return _render_mask_preview(base, masks)


def generate_v2(
    ctx: HandlerContext,
    mode_label: str,
    checkpoint: str,
    img2img_image,
    canvas_state,
    prompt: str,
    negative_prompt: str,
    lora_rows,
    embedding_names: list[str],
    cfg_scale: float,
    steps: int,
    sampler: str,
    seed: int,
    mask_blur: int,
    strength: float,
    width: int,
    height: int,
    controlnet_model: str,
    controlnet_image,
    controlnet_preprocess: str,
    controlnet_scale: float,
    controlnet_start: float,
    controlnet_end: float,
    ip_adapter_name: str,
    ip_adapter_image,
    ip_adapter_scale: float,
    regional_enabled: bool,
    regional_layout: str,
    regional_ratios: str,
    regional_base_ratios: str,
    regional_overlay_ratio: float,
    regional_use_base_prompt: bool,
    regional_use_common_prompt: bool,
    regional_use_common_negative: bool,
    regional_common_prompt: str,
    regional_prompt_text: str,
    regional_negative_text: str,
    regional_lora_text: str,
    regional_lora_negative_te: str,
    regional_lora_negative_unet: str,
    regional_lora_stop_step: int,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is required")

    mode_key = _mode_key(mode_label)
    max_size = int(ctx.config.limits.get("max_image_size", 2048))
    image: Image.Image | None = None
    mask: Image.Image | None = None

    if mode_key == "img2img":
        image = _prepare_input_image(img2img_image, max_size)
        if image is None:
            raise gr.Error("Upload an img2img source image")
    elif mode_key.startswith("inpaint"):
        image = _prepare_input_image(_canvas_image(canvas_state), max_size)
        if image is None:
            raise gr.Error("Upload an inpaint source image")

    if mode_key == "inpaint_manual":
        mask = MaskBuilder.from_user_canvas(canvas_state)
        if image is not None and mask.size != image.size:
            mask = mask.resize(image.size, Image.BILINEAR)
        if not mask.getbbox():
            raise gr.Error("Mask is empty")
    elif mode_key == "inpaint_auto":
        boxes = ctx.detector.detect(image)
        if not boxes:
            raise gr.Error("Eyes were not detected. Try manual inpaint.")
        det_cfg = ctx.config.detection
        mask = MaskBuilder.from_boxes(
            image.size,
            boxes,
            padding=int(det_cfg.get("padding_px", 8)),
            feather=int(det_cfg.get("feather_px", 12)),
        )

    regional_mask = _regional_mask(regional_layout, canvas_state, image)
    if regional_enabled and (regional_layout or "").lower() == "mask" and regional_mask is None:
        raise gr.Error("Regional mask layout needs a painted mask on the canvas")
    regional = normalize_spec(
        enabled=regional_enabled,
        layout=regional_layout,
        ratios=regional_ratios,
        base_ratios=regional_base_ratios,
        overlay_ratio=float(regional_overlay_ratio or 0.0),
        use_base_prompt=bool(regional_use_base_prompt),
        use_common_prompt=bool(regional_use_common_prompt),
        use_common_negative=bool(regional_use_common_negative),
        lora_negative_text_encoder_ratios=regional_lora_negative_te,
        lora_negative_unet_ratios=regional_lora_negative_unet,
        lora_stop_step=int(regional_lora_stop_step or 0),
        common_prompt=regional_common_prompt,
        region_prompt_text=regional_prompt_text,
        region_negative_text=regional_negative_text,
        region_lora_text=regional_lora_text,
        fallback_prompt=prompt,
        mask=regional_mask,
    )
    if regional_enabled:
        _validate_regional_ratio_count(regional, regional_ratios)

    req = GenerationRequest(
        mode=mode_key,  # type: ignore[arg-type]
        checkpoint=checkpoint or default_checkpoint(ctx),
        image=image,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        strength=float(strength),
        cfg_scale=float(cfg_scale),
        steps=int(steps),
        sampler=sampler,
        seed=int(seed),
        mask_blur=int(mask_blur),
        width=int(width),
        height=int(height),
        loras=parse_lora_table(lora_rows, ctx),
        embeddings=_selected_embeddings(ctx, embedding_names),
        controlnet_model=_controlnet_value(ctx, controlnet_model),
        controlnet_image=_prepare_input_image(controlnet_image, max_size),
        controlnet_preprocess=(controlnet_preprocess or "none").lower(),
        controlnet_scale=float(controlnet_scale),
        controlnet_start=float(controlnet_start),
        controlnet_end=float(controlnet_end),
        ip_adapter=_selected_ip_adapter(ctx, ip_adapter_name),
        ip_adapter_image=_prepare_input_image(ip_adapter_image, max_size),
        ip_adapter_scale=float(ip_adapter_scale),
        regional=regional,
    )

    logger.info(
        "Run checkpoint=%s mode=%s steps=%d loras=%d embeddings=%d controlnet=%s ip=%s regional=%s",
        req.checkpoint,
        mode_key,
        req.steps,
        len(req.loras),
        len(req.embeddings),
        req.controlnet_model or "-",
        req.ip_adapter.name if req.ip_adapter else "-",
        bool(req.regional and req.regional.enabled),
    )
    result = ctx.inference.run(req)

    outputs_dir = Path(ctx.config.path("outputs_dir"))
    now = datetime.now()
    target = (
        outputs_dir
        / now.strftime("%Y%m%d")
        / f"{now.strftime('%H%M%S_%f')[:-3]}_seed{result.seed_used}.png"
    )
    target = _available_path(target)
    save_image_with_metadata(result.image, target, result.metadata)

    ctx.history.add(
        HistoryItem(image=result.image, metadata=result.metadata, saved_path=target)
    )
    info = f"seed={result.seed_used} / {result.elapsed_ms} ms / {target}"
    return result.image, info, result.seed_used, ctx.history.gallery()


# ---------------------------------------------------------------------- presets


def save_preset(
    ctx: HandlerContext,
    name: str,
    mode: str,
    checkpoint: str,
    prompt: str,
    negative_prompt: str,
    lora_rows,
    embedding_names: list[str],
    cfg_scale: float,
    steps: int,
    sampler: str,
    seed: int,
    mask_blur: int,
    strength: float,
    width: int,
    height: int,
    controlnet_model: str,
    controlnet_preprocess: str,
    controlnet_scale: float,
    controlnet_start: float,
    controlnet_end: float,
    ip_adapter_name: str,
    ip_adapter_scale: float,
    regional_enabled: bool,
    regional_layout: str,
    regional_ratios: str,
    regional_base_ratios: str,
    regional_overlay_ratio: float,
    regional_use_base_prompt: bool,
    regional_use_common_prompt: bool,
    regional_use_common_negative: bool,
    regional_common_prompt: str,
    regional_prompt_text: str,
    regional_negative_text: str,
    regional_lora_text: str,
    regional_lora_negative_te: str,
    regional_lora_negative_unet: str,
    regional_lora_stop_step: int,
):
    if not name or not name.strip():
        raise gr.Error("Preset name is required")
    data = {
        "mode": mode,
        "checkpoint": checkpoint or default_checkpoint(ctx),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "loras": [{"name": s.name, "weight": s.weight} for s in parse_lora_table(lora_rows, ctx)],
        "embeddings": embedding_names or [],
        "strength": float(strength),
        "cfg_scale": float(cfg_scale),
        "steps": int(steps),
        "sampler": sampler,
        "seed": int(seed),
        "mask_blur": int(mask_blur),
        "width": int(width),
        "height": int(height),
        "controlnet": {
            "model": _controlnet_value(ctx, controlnet_model),
            "preprocess": controlnet_preprocess or "none",
            "scale": float(controlnet_scale),
            "start": float(controlnet_start),
            "end": float(controlnet_end),
        },
        "ip_adapter": {"name": ip_adapter_name or "", "scale": float(ip_adapter_scale)},
        "regional": {
            "enabled": bool(regional_enabled),
            "layout": regional_layout,
            "ratios": regional_ratios,
            "base_ratios": regional_base_ratios,
            "overlay_ratio": float(regional_overlay_ratio or 0.0),
            "use_base_prompt": bool(regional_use_base_prompt),
            "use_common_prompt": bool(regional_use_common_prompt),
            "use_common_negative": bool(regional_use_common_negative),
            "common_prompt": regional_common_prompt,
            "region_prompt_text": regional_prompt_text,
            "region_negative_text": regional_negative_text,
            "region_lora_text": regional_lora_text,
            "lora_negative_text_encoder_ratios": regional_lora_negative_te,
            "lora_negative_unet_ratios": regional_lora_negative_unet,
            "lora_stop_step": int(regional_lora_stop_step or 0),
        },
    }
    presets.save_preset(ctx.config.path("presets_dir"), name, data)
    return gr.update(choices=list_presets_choices(ctx), value=name)


def load_preset(ctx: HandlerContext, name: str):
    if not name:
        raise gr.Error("Select a preset")
    data = presets.load_preset(ctx.config.path("presets_dir"), name)

    known = {entry.name: entry for entry in ctx.pipeline_manager.lora_loader.discover()}
    preset_lora_names = {item["name"] for item in data.get("loras", [])}
    selected_names = [n for n in preset_lora_names if n in known]

    control = data.get("controlnet", {})
    ip = data.get("ip_adapter", {})
    regional = data.get("regional", {})

    return (
        data.get("mode", TXT2IMG),
        data.get("checkpoint", default_checkpoint(ctx)),
        data.get("prompt", ""),
        data.get("negative_prompt", ""),
        selected_names,
        data.get("embeddings", []),
        data.get("cfg_scale", 7.0),
        data.get("steps", 28),
        data.get("sampler", "DPM++ 2M Karras"),
        data.get("seed", -1),
        data.get("mask_blur", 8),
        data.get("strength", 0.55),
        data.get("width", 1024),
        data.get("height", 1024),
        _controlnet_value(ctx, control.get("model", "")),
        control.get("preprocess", "none"),
        control.get("scale", 1.0),
        control.get("start", 0.0),
        control.get("end", 1.0),
        ip.get("name", ""),
        ip.get("scale", 1.0),
        regional.get("enabled", False),
        regional.get("layout", "horizontal"),
        regional.get("ratios", ""),
        regional.get("base_ratios", ""),
        regional.get("overlay_ratio", 0.0),
        regional.get("use_base_prompt", False),
        regional.get("use_common_prompt", False),
        regional.get("use_common_negative", False),
        regional.get("common_prompt", ""),
        regional.get("region_prompt_text", ""),
        regional.get("region_negative_text", ""),
        regional.get("region_lora_text", ""),
        regional.get("lora_negative_text_encoder_ratios", ""),
        regional.get("lora_negative_unet_ratios", ""),
        regional.get("lora_stop_step", 0),
    )
