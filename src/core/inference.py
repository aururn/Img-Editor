"""Image-generation orchestration."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Literal

from PIL import Image

from ..utils.logger import get_logger
from .assets import EmbeddingSpec, IPAdapterSpec
from .lora_loader import LoRASpec
from .pipeline_manager import PipelineManager
from .regional import RegionalPromptSpec, build_region_masks, combine_prompt

logger = get_logger(__name__)


@dataclass
class GenerationRequest:
    mode: Literal["txt2img", "img2img", "inpaint_manual", "inpaint_auto"]
    checkpoint: str
    image: Image.Image | None
    mask: Image.Image | None
    prompt: str
    negative_prompt: str = ""
    strength: float = 0.55
    cfg_scale: float = 7.0
    steps: int = 28
    sampler: str = "DPM++ 2M Karras"
    seed: int = -1
    mask_blur: int = 8
    width: int = 1024
    height: int = 1024
    loras: list[LoRASpec] = field(default_factory=list)
    embeddings: list[EmbeddingSpec] = field(default_factory=list)
    controlnet_model: str = ""
    controlnet_image: Image.Image | None = None
    controlnet_preprocess: str = "none"
    controlnet_scale: float = 1.0
    controlnet_start: float = 0.0
    controlnet_end: float = 1.0
    ip_adapter: IPAdapterSpec | None = None
    ip_adapter_image: Image.Image | None = None
    ip_adapter_scale: float = 1.0
    regional: RegionalPromptSpec | None = None


@dataclass
class GenerationResult:
    image: Image.Image
    seed_used: int
    elapsed_ms: int
    metadata: dict


class InferenceService:
    def __init__(self, pipeline_manager: PipelineManager) -> None:
        self.pm = pipeline_manager
        self._stop_event = threading.Event()
        self._run_lock = threading.Lock()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _reset_stop(self) -> None:
        self._stop_event.clear()

    def _step_callback(self, pipe, step: int, timestep, callback_kwargs: dict) -> dict:
        if self._stop_event.is_set():
            pipe._interrupt = True
        return callback_kwargs

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        if seed is None or int(seed) < 0:
            return random.randint(0, 2**31 - 1)
        return int(seed)

    def _generator(self, seed: int):
        import torch

        device = self.pm.device
        gen = torch.Generator(device=device if device != "mps" else "cpu")
        gen.manual_seed(seed)
        return gen

    @staticmethod
    def _round_to_8(value: int) -> int:
        return max(8, (int(value) // 8) * 8)

    @staticmethod
    def _resize_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
        image = image.convert("RGB")
        if image.size != size:
            image = image.resize(size, Image.LANCZOS)
        return image

    def _prepare_inpaint_inputs(
        self, image: Image.Image, mask: Image.Image, mask_blur: int
    ) -> tuple[Image.Image, Image.Image]:
        from PIL import ImageFilter

        if mask.mode != "L":
            mask = mask.convert("L")
        if mask_blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.BILINEAR)
        return image.convert("RGB"), mask

    @staticmethod
    def _has_controlnet(req: GenerationRequest) -> bool:
        return bool(req.controlnet_model and req.controlnet_model.strip())

    def _prepare_pipe(self, pipe, req: GenerationRequest) -> None:
        self.pm.set_sampler(req.sampler)
        self.pm.apply_loras(req.loras)
        self.pm.apply_textual_inversions(pipe, req.embeddings)
        if req.ip_adapter and req.ip_adapter_image is not None:
            self.pm.apply_ip_adapter(pipe, req.ip_adapter, req.ip_adapter_scale)
        else:
            self.pm.apply_ip_adapter(pipe, None)

    def _control_image(
        self,
        req: GenerationRequest,
        size: tuple[int, int],
        source_image: Image.Image | None = None,
    ) -> Image.Image | None:
        if not self._has_controlnet(req):
            return None
        if req.controlnet_image is not None:
            return self._resize_rgb(req.controlnet_image, size)
        if (req.controlnet_preprocess or "").lower() != "canny":
            return None
        if source_image is None:
            raise ValueError("ControlNet Canny preprocessing needs an input image")
        try:
            import cv2
            import numpy as np
        except Exception as exc:
            raise RuntimeError("opencv-python is required for Canny ControlNet") from exc
        src = self._resize_rgb(source_image, size)
        arr = np.array(src)
        edges = cv2.Canny(arr, 100, 200)
        edges = np.stack([edges, edges, edges], axis=2)
        return Image.fromarray(edges)

    # ------------------------------------------------------------------ run

    def run(self, req: GenerationRequest) -> GenerationResult:
        with self._run_lock:
            return self._run_locked(req)

    def _run_locked(self, req: GenerationRequest) -> GenerationResult:
        self._reset_stop()
        seed = self._resolve_seed(req.seed)
        self.pm.set_checkpoint(req.checkpoint)
        self.pm.set_sampler(req.sampler)

        start = time.perf_counter()
        try:
            if req.regional and req.regional.enabled:
                image = self._run_regional(req, seed)
            else:
                image = self._run_standard(req, seed)
        except Exception as exc:
            if self._is_oom(exc):
                logger.warning("CUDA OOM detected, applying fallbacks")
                image = self._run_with_fallback(req, seed)
            else:
                raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        meta = self._metadata(req, image, seed)
        return GenerationResult(
            image=image, seed_used=seed, elapsed_ms=elapsed_ms, metadata=meta
        )

    def _metadata(self, req: GenerationRequest, image: Image.Image, seed: int) -> dict:
        return {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "steps": req.steps,
            "sampler": req.sampler,
            "cfg_scale": req.cfg_scale,
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "model": self.pm.current_checkpoint(),
            "loras": [{"name": s.name, "weight": s.weight} for s in req.loras],
            "embeddings": [{"name": e.name, "token": e.token} for e in req.embeddings],
            "mode": req.mode,
            "mask_blur": req.mask_blur if req.mode.startswith("inpaint") else None,
            "controlnet": {
                "model": req.controlnet_model,
                "preprocess": req.controlnet_preprocess,
                "scale": req.controlnet_scale,
                "start": req.controlnet_start,
                "end": req.controlnet_end,
            }
            if self._has_controlnet(req)
            else None,
            "ip_adapter": {
                "name": req.ip_adapter.name,
                "scale": req.ip_adapter_scale,
            }
            if req.ip_adapter and req.ip_adapter_image is not None
            else None,
            "regional": req.regional.to_metadata()
            if req.regional and req.regional.enabled
            else None,
        }

    # ------------------------------------------------------------------ prompt encoding

    @staticmethod
    def _encode_long_prompt(pipe, prompt: str, negative_prompt: str):
        """Encode prompts with 75-token chunking to bypass the 77-token CLIP limit."""
        import torch

        def _encoder_device(encoder):
            try:
                return next(encoder.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        def _chunk_encode(tokenizer, encoder, text: str, is_clip_g: bool = False):
            ids = tokenizer(text, truncation=False, return_tensors="pt").input_ids[0]
            bos, eos = ids[0].item(), ids[-1].item()
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
            body = ids[1:-1]
            dev = _encoder_device(encoder)
            chunks, pooled = [], None
            for start in range(0, max(len(body), 1), 75):
                seg = body[start : start + 75]
                n_pad = 75 - len(seg)
                inp = torch.cat(
                    [
                        torch.tensor([bos], dtype=torch.long),
                        seg,
                        torch.tensor([eos], dtype=torch.long),
                        torch.tensor([pad] * n_pad, dtype=torch.long),
                    ]
                ).unsqueeze(0).to(dev)
                with torch.no_grad():
                    out = encoder(inp, output_hidden_states=True)
                chunks.append(out.hidden_states[-2])
                if is_clip_g and pooled is None:
                    pooled = out.text_embeds
            embs = torch.cat(chunks, dim=1)
            return (embs, pooled) if is_clip_g else embs

        neg = negative_prompt or ""
        prompt_l = _chunk_encode(pipe.tokenizer, pipe.text_encoder, prompt)
        neg_l = _chunk_encode(pipe.tokenizer, pipe.text_encoder, neg)
        prompt_g, pooled_pos = _chunk_encode(
            pipe.tokenizer_2, pipe.text_encoder_2, prompt, is_clip_g=True
        )
        neg_g, pooled_neg = _chunk_encode(
            pipe.tokenizer_2, pipe.text_encoder_2, neg, is_clip_g=True
        )

        max_len = max(t.shape[1] for t in (prompt_l, neg_l, prompt_g, neg_g))

        def _pad(t, target):
            gap = target - t.shape[1]
            if gap > 0:
                t = torch.cat(
                    [
                        t,
                        torch.zeros(
                            *t.shape[:1],
                            gap,
                            t.shape[2],
                            dtype=t.dtype,
                            device=t.device,
                        ),
                    ],
                    dim=1,
                )
            return t

        prompt_embeds = torch.cat(
            [_pad(prompt_l, max_len), _pad(prompt_g, max_len)], dim=-1
        )
        neg_embeds = torch.cat(
            [_pad(neg_l, max_len), _pad(neg_g, max_len)], dim=-1
        )
        return prompt_embeds, neg_embeds, pooled_pos, pooled_neg

    # ------------------------------------------------------------------ dispatch

    def _run_standard(self, req: GenerationRequest, seed: int) -> Image.Image:
        if req.mode == "txt2img":
            return self._run_txt2img(req, seed)
        if req.mode == "img2img":
            return self._run_img2img(req, seed)
        return self._run_inpaint(req, seed)

    def _common_kwargs(self, req: GenerationRequest, seed: int) -> dict:
        kwargs: dict = {
            "guidance_scale": float(req.cfg_scale),
            "num_inference_steps": int(req.steps),
            "generator": self._generator(seed),
            "callback_on_step_end": self._step_callback,
        }
        if req.ip_adapter and req.ip_adapter_image is not None:
            kwargs["ip_adapter_image"] = req.ip_adapter_image.convert("RGB")
        return kwargs

    def _run_txt2img(self, req: GenerationRequest, seed: int) -> Image.Image:
        width = self._round_to_8(req.width)
        height = self._round_to_8(req.height)
        if self._has_controlnet(req):
            pipe = self.pm.get_controlnet("txt2img", req.controlnet_model)
        else:
            pipe = self.pm.get_txt2img()
        self._prepare_pipe(pipe, req)
        pe, ne, pp, np_ = self._encode_long_prompt(pipe, req.prompt, req.negative_prompt)
        kwargs = self._common_kwargs(req, seed)
        kwargs.update(
            {
                "prompt_embeds": pe,
                "negative_prompt_embeds": ne,
                "pooled_prompt_embeds": pp,
                "negative_pooled_prompt_embeds": np_,
                "width": width,
                "height": height,
            }
        )
        if self._has_controlnet(req):
            control = self._control_image(req, (width, height), req.controlnet_image)
            if control is None:
                raise ValueError("ControlNet txt2img needs a control image")
            kwargs.update(
                {
                    "image": control,
                    "controlnet_conditioning_scale": float(req.controlnet_scale),
                    "control_guidance_start": float(req.controlnet_start),
                    "control_guidance_end": float(req.controlnet_end),
                }
            )
        out = pipe(**kwargs)
        return out.images[0]

    def _run_img2img(self, req: GenerationRequest, seed: int) -> Image.Image:
        if req.image is None:
            raise ValueError("img2img mode requires an image")
        image = req.image.convert("RGB")
        width = self._round_to_8(image.width)
        height = self._round_to_8(image.height)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)
        if self._has_controlnet(req):
            pipe = self.pm.get_controlnet("img2img", req.controlnet_model)
        else:
            pipe = self.pm.get_img2img()
        self._prepare_pipe(pipe, req)
        pe, ne, pp, np_ = self._encode_long_prompt(pipe, req.prompt, req.negative_prompt)
        kwargs = self._common_kwargs(req, seed)
        kwargs.update(
            {
                "prompt_embeds": pe,
                "negative_prompt_embeds": ne,
                "pooled_prompt_embeds": pp,
                "negative_pooled_prompt_embeds": np_,
                "image": image,
                "strength": float(req.strength),
            }
        )
        if self._has_controlnet(req):
            control = self._control_image(req, image.size, image)
            if control is None:
                raise ValueError("ControlNet img2img needs a control image or Canny preprocess")
            kwargs.update(
                {
                    "control_image": control,
                    "controlnet_conditioning_scale": float(req.controlnet_scale),
                    "control_guidance_start": float(req.controlnet_start),
                    "control_guidance_end": float(req.controlnet_end),
                }
            )
        out = pipe(**kwargs)
        return out.images[0]

    def _run_inpaint(self, req: GenerationRequest, seed: int) -> Image.Image:
        if req.image is None or req.mask is None:
            raise ValueError("inpaint mode requires an image and a mask")
        image, mask = self._prepare_inpaint_inputs(req.image, req.mask, req.mask_blur)
        width = self._round_to_8(image.width)
        height = self._round_to_8(image.height)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)
            mask = mask.resize((width, height), Image.BILINEAR)
        if self._has_controlnet(req):
            pipe = self.pm.get_controlnet("inpaint", req.controlnet_model)
        else:
            pipe = self.pm.get_inpaint()
        self._prepare_pipe(pipe, req)
        pe, ne, pp, np_ = self._encode_long_prompt(pipe, req.prompt, req.negative_prompt)
        kwargs = self._common_kwargs(req, seed)
        kwargs.update(
            {
                "prompt_embeds": pe,
                "negative_prompt_embeds": ne,
                "pooled_prompt_embeds": pp,
                "negative_pooled_prompt_embeds": np_,
                "image": image,
                "mask_image": mask,
                "strength": float(req.strength),
                "width": width,
                "height": height,
            }
        )
        if self._has_controlnet(req):
            control = self._control_image(req, image.size, image)
            if control is None:
                raise ValueError("ControlNet inpaint needs a control image or Canny preprocess")
            kwargs.update(
                {
                    "control_image": control,
                    "controlnet_conditioning_scale": float(req.controlnet_scale),
                    "control_guidance_start": float(req.controlnet_start),
                    "control_guidance_end": float(req.controlnet_end),
                }
            )
        out = pipe(**kwargs)
        return out.images[0]

    # ------------------------------------------------------------------ regional

    def _run_regional(self, req: GenerationRequest, seed: int) -> Image.Image:
        """Staged inpaint with per-region LoRA swap.

        1. Build a base image using ``common_loras`` and ``base_prompt`` (or use
           the user-supplied image for img2img/inpaint when ``use_base_pass`` is
           off).
        2. For each region, swap LoRAs to ``common_loras + region_loras[i]`` and
           inpaint with the region mask + region prompt.

        Per-region LoRA swap is what prevents character LoRA cross-contamination
        — the actual LoRA application happens inside ``_run_inpaint`` via
        ``_prepare_pipe`` → ``pm.apply_loras``, which is no-op when unchanged
        and unload+reload when changed.
        """
        spec = req.regional
        if spec is None or not spec.enabled:
            return self._run_standard(req, seed)

        # --- 1. Base pass ---------------------------------------------------
        base_prompt = spec.base_prompt.strip() or req.prompt
        # common_loras (e.g. situation LoRA) provide the base layout; fall back
        # to req.loras so the legacy code path still works when the new UI
        # fields are empty.
        base_loras = list(spec.common_loras) if spec.common_loras else list(req.loras)
        base_strength = (
            float(spec.base_strength) if spec.base_strength is not None else req.strength
        )
        base_req = replace(
            req,
            prompt=base_prompt,
            strength=base_strength,
            loras=base_loras,
            regional=None,
        )

        if req.mode == "txt2img" or spec.use_base_pass:
            current = self._run_standard(base_req, seed)
        else:
            if req.image is None:
                current = self._run_standard(base_req, seed)
            else:
                current = req.image.convert("RGB")

        # --- 2. Region masks ------------------------------------------------
        masks = build_region_masks(spec, current.size)
        if not masks:
            return current

        # --- 3. Per-region inpaint with swapped LoRAs ----------------------
        negatives = spec.region_negative_prompts or []
        region_strength = (
            float(spec.region_strength)
            if spec.region_strength is not None
            else req.strength
        )
        for index, (mask, region_prompt) in enumerate(zip(masks, spec.region_prompts)):
            if self._stop_event.is_set():
                logger.info("Regional generation cancelled at region %d", index)
                break
            region_negative = negatives[index] if index < len(negatives) else ""
            negative = combine_prompt(req.negative_prompt, region_negative)
            prompt = combine_prompt(spec.common_prompt, region_prompt)
            per_region = (
                spec.region_loras[index] if index < len(spec.region_loras) else []
            )
            merged_loras = list(spec.common_loras) + list(per_region)
            region_req = replace(
                req,
                mode="inpaint_manual",
                image=current,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative,
                strength=region_strength,
                loras=merged_loras,
                seed=seed + index + 1,
                regional=None,
            )
            current = self._run_inpaint(region_req, seed + index + 1)
        return current

    # ------------------------------------------------------------------ OOM

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(exc).lower()

    def _run_with_fallback(self, req: GenerationRequest, seed: int) -> Image.Image:
        self.pm.free_memory()
        for pipe in (
            self.pm._txt2img_pipe,
            self.pm._img2img_pipe,
            self.pm._inpaint_pipe,
            *self.pm._controlnet_pipes.values(),
        ):
            if pipe is None:
                continue
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass

        try:
            return self._run_standard(replace(req, regional=None), seed)
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            logger.warning("Retry OOM: downscaling to 0.75x")
            self.pm.free_memory()
            if req.mode == "txt2img":
                scaled = replace(
                    req,
                    width=int(req.width * 0.75),
                    height=int(req.height * 0.75),
                    regional=None,
                )
            else:
                if req.image is None:
                    raise
                scaled_image = req.image.resize(
                    (int(req.image.width * 0.75), int(req.image.height * 0.75)),
                    Image.LANCZOS,
                )
                scaled_mask = (
                    req.mask.resize(scaled_image.size, Image.BILINEAR)
                    if req.mask is not None
                    else None
                )
                scaled = replace(
                    req,
                    image=scaled_image,
                    mask=scaled_mask,
                    regional=None,
                )
            return self._run_standard(scaled, seed)
