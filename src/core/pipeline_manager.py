"""SDXL pipeline construction, caching, and adapter application."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger
from .assets import EmbeddingSpec, IPAdapterSpec
from .config import AppConfig
from .lora_loader import LoRALoader, LoRASpec

logger = get_logger(__name__)


_SAMPLER_MAP: dict[str, str] = {
    "Euler a": "EulerAncestralDiscreteScheduler",
    "DPM++ 2M Karras": "DPMSolverMultistepScheduler",
    "DDIM": "DDIMScheduler",
}


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _dtype(precision: str):
    import torch

    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(precision, torch.float16)


class PipelineManager:
    """Lazily constructs SDXL pipelines and shares weights where possible."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._txt2img_pipe: Any | None = None
        self._img2img_pipe: Any | None = None
        self._inpaint_pipe: Any | None = None
        self._controlnet_pipes: dict[tuple[str, str], Any] = {}
        self._controlnet_models: dict[str, Any] = {}
        self._loaded_embeddings: set[str] = set()
        self._loaded_ip_adapters: dict[int, tuple[str, float] | None] = {}
        self._active_checkpoint: str | None = None
        self.lora_loader = LoRALoader(config.path("loras_dir"))
        self.device = _device()
        self.dtype = None

    # ------------------------------------------------------------------ helpers

    def _resolve_checkpoint(self, checkpoint_name: str | None = None) -> Path:
        ckpt_name = checkpoint_name or self.config.model.get("base_checkpoint")
        if not ckpt_name:
            raise FileNotFoundError("config.model.base_checkpoint is empty")
        path = self.config.path("checkpoints_dir") / ckpt_name
        if not path.exists():
            raise FileNotFoundError(
                f"Base checkpoint not found: {path}. "
                "Place a .safetensors model in models/checkpoints/."
            )
        return path

    def _torch_dtype(self):
        import torch

        return self.dtype if self.device != "cpu" else torch.float32

    def _apply_runtime_optimizations(self, pipe) -> None:
        model_cfg = self.config.model
        try:
            pipe.enable_attention_slicing()
        except Exception as exc:
            logger.debug("attention slicing unavailable: %s", exc)
        try:
            pipe.vae.enable_slicing()
        except Exception as exc:
            logger.debug("vae slicing unavailable: %s", exc)
        if model_cfg.get("enable_vae_tiling", True):
            try:
                pipe.vae.enable_tiling()
            except Exception as exc:
                logger.debug("vae tiling unavailable: %s", exc)
        if model_cfg.get("enable_xformers", True) and self.device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                logger.debug("xformers unavailable: %s", exc)
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

    def _load_vae_override(self):
        """Load an external VAE if ``model.vae`` is configured."""
        vae_setting = self.config.model.get("vae")
        if not vae_setting:
            return None

        from diffusers import AutoencoderKL

        vae_str = str(vae_setting)
        local_path = self.config.path("vae_dir") / vae_str
        if local_path.exists() and local_path.is_file():
            logger.info("Loading VAE from local file %s", local_path)
            return AutoencoderKL.from_single_file(
                str(local_path), torch_dtype=self._torch_dtype()
            )

        logger.info("Loading VAE from Hugging Face: %s", vae_str)
        return AutoencoderKL.from_pretrained(vae_str, torch_dtype=self._torch_dtype())

    def _base_components(self, base) -> dict[str, Any]:
        return {
            "vae": base.vae,
            "text_encoder": base.text_encoder,
            "text_encoder_2": base.text_encoder_2,
            "tokenizer": base.tokenizer,
            "tokenizer_2": base.tokenizer_2,
            "unet": base.unet,
            "scheduler": base.scheduler,
        }

    def _maybe_to_device(self, pipe):
        if not (self.config.model.get("cpu_offload") and self.device == "cuda"):
            pipe = pipe.to(self.device)
        return pipe

    def _all_pipes(self) -> list[Any]:
        pipes = [self._txt2img_pipe, self._img2img_pipe, self._inpaint_pipe]
        pipes.extend(self._controlnet_pipes.values())
        return [p for p in pipes if p is not None]

    def current_checkpoint(self) -> str:
        return self._active_checkpoint or str(self.config.model.get("base_checkpoint", ""))

    def set_checkpoint(self, checkpoint_name: str | None = None) -> None:
        target = str(checkpoint_name or self.config.model.get("base_checkpoint", "")).strip()
        if not target:
            raise FileNotFoundError("No checkpoint selected")

        self._resolve_checkpoint(target)
        if self._active_checkpoint == target:
            return

        if self._active_checkpoint is not None:
            logger.info("Switching checkpoint: %s -> %s", self._active_checkpoint, target)
            self._txt2img_pipe = None
            self._img2img_pipe = None
            self._inpaint_pipe = None
            self._controlnet_pipes.clear()
            self._loaded_embeddings.clear()
            self._loaded_ip_adapters.clear()
            self.lora_loader = LoRALoader(self.config.path("loras_dir"))
            self.free_memory()

        self.config.raw.setdefault("model", {})["base_checkpoint"] = target
        self._active_checkpoint = target

    # ------------------------------------------------------------------ builders

    def _build_txt2img(self):
        from diffusers import StableDiffusionXLPipeline

        self.dtype = _dtype(self.config.model.get("precision", "fp16"))
        self.set_checkpoint()
        ckpt = self._resolve_checkpoint()
        logger.info("Loading SDXL txt2img from %s (device=%s)", ckpt, self.device)

        kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype(),
            "use_safetensors": True,
        }
        vae = self._load_vae_override()
        if vae is not None:
            kwargs["vae"] = vae

        pipe = StableDiffusionXLPipeline.from_single_file(str(ckpt), **kwargs)
        if self.config.model.get("cpu_offload") and self.device == "cuda":
            pipe.enable_model_cpu_offload()
            logger.info("CPU offload enabled")
        else:
            pipe = pipe.to(self.device)
        self._apply_runtime_optimizations(pipe)
        return pipe

    def _build_img2img_from(self, base):
        from diffusers import StableDiffusionXLImg2ImgPipeline

        logger.info("Building img2img pipeline sharing weights with txt2img")
        pipe = StableDiffusionXLImg2ImgPipeline(**self._base_components(base))
        pipe = self._maybe_to_device(pipe)
        self._apply_runtime_optimizations(pipe)
        return pipe

    def _build_inpaint_from(self, base):
        from diffusers import StableDiffusionXLInpaintPipeline

        logger.info("Building inpaint pipeline sharing weights with txt2img")
        pipe = StableDiffusionXLInpaintPipeline(**self._base_components(base))
        pipe = self._maybe_to_device(pipe)
        self._apply_runtime_optimizations(pipe)
        return pipe

    def _controlnet_path_for(self, name: str) -> Path:
        root = self.config.path("controlnet_dir")
        direct = root / name
        if direct.exists():
            return direct
        for path in (root.iterdir() if root.exists() else []):
            if path.stem == name or path.name == name:
                return path
        return direct

    def _load_controlnet_model(self, name: str):
        if name in self._controlnet_models:
            return self._controlnet_models[name]

        from diffusers import ControlNetModel

        path = self._controlnet_path_for(name)
        if not path.exists():
            # Allow a Hugging Face repo id for advanced users.
            source = name
        else:
            source = str(path)

        logger.info("Loading ControlNet %s", source)
        if path.exists() and path.is_file():
            model = ControlNetModel.from_single_file(source, torch_dtype=self._torch_dtype())
        else:
            model = ControlNetModel.from_pretrained(source, torch_dtype=self._torch_dtype())
        if not (self.config.model.get("cpu_offload") and self.device == "cuda"):
            model = model.to(self.device)
        self._controlnet_models[name] = model
        return model

    def _build_controlnet_pipe(self, mode: str, name: str):
        base = self.get_txt2img()
        controlnet = self._load_controlnet_model(name)
        components = self._base_components(base)
        components["controlnet"] = controlnet

        if mode == "txt2img":
            from diffusers import StableDiffusionXLControlNetPipeline

            pipe = StableDiffusionXLControlNetPipeline(**components)
        elif mode == "img2img":
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline

            pipe = StableDiffusionXLControlNetImg2ImgPipeline(**components)
        else:
            from diffusers import StableDiffusionXLControlNetInpaintPipeline

            pipe = StableDiffusionXLControlNetInpaintPipeline(**components)

        pipe = self._maybe_to_device(pipe)
        self._apply_runtime_optimizations(pipe)
        return pipe

    # ------------------------------------------------------------------ public

    def get_txt2img(self):
        self.set_checkpoint()
        if self._txt2img_pipe is None:
            self._txt2img_pipe = self._build_txt2img()
        return self._txt2img_pipe

    def get_img2img(self):
        if self._img2img_pipe is None:
            self._img2img_pipe = self._build_img2img_from(self.get_txt2img())
        return self._img2img_pipe

    def get_inpaint(self):
        if self._inpaint_pipe is None:
            self._inpaint_pipe = self._build_inpaint_from(self.get_txt2img())
        return self._inpaint_pipe

    def get_controlnet(self, mode: str, name: str):
        key = (mode, name)
        if key not in self._controlnet_pipes:
            self._controlnet_pipes[key] = self._build_controlnet_pipe(mode, name)
        return self._controlnet_pipes[key]

    def set_sampler(self, name: str) -> None:
        try:
            import diffusers

            klass_name = _SAMPLER_MAP.get(name)
            if not klass_name:
                logger.warning("Unknown sampler: %s", name)
                return
            scheduler_cls = getattr(diffusers, klass_name)
            for pipe in self._all_pipes():
                kwargs: dict[str, Any] = {}
                if name == "DPM++ 2M Karras":
                    kwargs = {"use_karras_sigmas": True, "algorithm_type": "dpmsolver++"}
                pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
        except Exception as exc:
            logger.exception("Failed to set sampler %s: %s", name, exc)

    def apply_loras(self, specs: list[LoRASpec]) -> None:
        primary = self._txt2img_pipe or self._img2img_pipe or self._inpaint_pipe
        if primary is None:
            return
        self.lora_loader.apply(primary, specs)

    def apply_textual_inversions(self, pipe, specs: list[EmbeddingSpec]) -> None:
        if not specs:
            return
        root = self.config.path("embeddings_dir")
        for spec in specs:
            if spec.token in self._loaded_embeddings:
                continue
            path = root / spec.file
            if not path.exists():
                logger.warning("Textual inversion file not found: %s", path)
                continue
            try:
                pipe.load_textual_inversion(str(path), token=spec.token)
                self._loaded_embeddings.add(spec.token)
                logger.info("Loaded textual inversion %s as %s", path.name, spec.token)
            except Exception as exc:
                logger.exception("Failed to load textual inversion %s: %s", spec.name, exc)
                raise

    def apply_ip_adapter(
        self, pipe, spec: IPAdapterSpec | None, scale: float = 1.0
    ) -> None:
        key = (spec.name, round(float(scale), 4)) if spec else None
        pipe_id = id(pipe)
        if self._loaded_ip_adapters.get(pipe_id) == key:
            return

        if spec is None:
            try:
                pipe.unload_ip_adapter()
            except Exception:
                try:
                    pipe.set_ip_adapter_scale(0.0)
                except Exception:
                    pass
            self._loaded_ip_adapters[pipe_id] = None
            return

        root = self.config.path("ip_adapter_dir")
        path = root / spec.path
        kwargs: dict[str, Any] = {}
        if path.is_file():
            source = str(path.parent)
            kwargs["weight_name"] = path.name
        else:
            source = str(path) if path.exists() else spec.path
            if spec.weight_name:
                kwargs["weight_name"] = spec.weight_name
        try:
            pipe.load_ip_adapter(source, **kwargs)
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(float(scale))
            self._loaded_ip_adapters[pipe_id] = key
            logger.info("Loaded IP-Adapter %s scale=%.2f", spec.name, scale)
        except Exception as exc:
            logger.exception("Failed to load IP-Adapter %s: %s", spec.name, exc)
            raise

    def free_memory(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
