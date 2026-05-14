"""LoRA discovery and (un)loading."""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Common NSFW indicator words (substring match on filename, case-insensitive)
_NSFW_HINTS = ("nsfw", "r18", "r-18", "hentai", "lewd")
_LYCORIS_HINTS = ("locon", "loha", "lokr", "dylora", "ia3", "lycoris")


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    """Read the JSON metadata header embedded in a .safetensors file.

    The format is: 8-byte little-endian header length + JSON header.
    The JSON object has a "__metadata__" key with string-string pairs.
    """
    try:
        with path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return {}
            header_len = int.from_bytes(header_len_bytes, "little")
            if header_len <= 0 or header_len > 100 * 1024 * 1024:
                return {}
            header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8", errors="replace"))
        meta = header.get("__metadata__", {}) or {}
        return {str(k): str(v) for k, v in meta.items()}
    except Exception as exc:
        logger.debug("safetensors metadata read failed for %s: %s", path, exc)
        return {}


def _extract_triggers(meta: dict[str, str], top_n: int = 8) -> list[str]:
    """Pull likely trigger words from various LoRA training metadata fields."""
    triggers: list[str] = []

    # explicit fields used by some training tools
    for key in ("modelspec.trigger_text", "ss_trigger_text", "trigger_text"):
        v = meta.get(key)
        if v:
            triggers.extend(t.strip() for t in v.split(",") if t.strip())

    # ss_tag_frequency: nested dict of folder -> tag -> count
    raw = meta.get("ss_tag_frequency")
    if raw:
        try:
            tag_map = json.loads(raw)
            counts: dict[str, int] = {}
            for tags in tag_map.values():
                if not isinstance(tags, dict):
                    continue
                for tag, count in tags.items():
                    counts[tag] = counts.get(tag, 0) + int(count)
            ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
            for tag, _ in ranked[: top_n * 3]:
                if len(tag) < 2 or tag.isdigit():
                    continue
                triggers.append(tag)
        except Exception as exc:
            logger.debug("ss_tag_frequency parse failed: %s", exc)

    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in triggers:
        t = t.strip(", ")
        key = t.lower()
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= top_n:
            break
    return out


@dataclass
class LoRASpec:
    name: str
    weight: float = 0.8
    text_weight: float | None = None
    unet_weight: float | None = None

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        self.text_weight = self.weight if self.text_weight is None else float(self.text_weight)
        self.unet_weight = self.weight if self.unet_weight is None else float(self.unet_weight)

    def key(self) -> tuple[str, float, float, float]:
        return (
            self.name,
            round(self.weight, 4),
            round(float(self.text_weight), 4),
            round(float(self.unet_weight), 4),
        )

    def text_encoder_spec(self) -> "LoRASpec":
        return LoRASpec(self.name, float(self.text_weight), self.text_weight, self.unet_weight)

    def unet_spec(self) -> "LoRASpec":
        return LoRASpec(self.name, float(self.unet_weight), self.text_weight, self.unet_weight)


def _adapter_name(name: str) -> str:
    stem = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or "adapter"
    digest = hashlib.sha1(name.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"lora_{stem[:48]}_{digest}"


def _dedupe_specs(specs: list[LoRASpec]) -> list[LoRASpec]:
    deduped: dict[str, LoRASpec] = {}
    for spec in specs:
        if not spec.name:
            continue
        deduped[spec.name] = spec
    return list(deduped.values())


@dataclass
class LoRAEntry:
    name: str
    file: str
    base_model: str | None = None
    trigger_words: list[str] = field(default_factory=list)
    recommended_weight: float = 0.8
    min_weight: float = 0.0
    max_weight: float = 1.5
    nsfw: bool = False
    title: str | None = None
    network_module: str | None = None
    warnings: list[str] = field(default_factory=list)


class LoRALoader:
    """Discovers LoRA files and applies them to a diffusers pipeline."""

    def __init__(self, loras_dir: str | Path) -> None:
        self.loras_dir = Path(loras_dir)
        self._applied: list[LoRASpec] = []
        self._loaded_adapters: dict[str, str] = {}

    def discover(self) -> list[LoRAEntry]:
        entries: list[LoRAEntry] = []
        if not self.loras_dir.exists():
            logger.warning("LoRAs dir not found: %s", self.loras_dir)
            return entries
        for path in sorted(self.loras_dir.glob("*.safetensors")):
            # external json overrides
            meta_path = path.with_suffix(".json")
            external_meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    external_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(
                        "Failed to parse LoRA metadata %s: %s", meta_path, exc
                    )

            # safetensors header metadata
            st_meta = _read_safetensors_metadata(path)
            auto_triggers = _extract_triggers(st_meta)
            title = st_meta.get("modelspec.title") or st_meta.get("ss_output_name")
            base_model = (
                external_meta.get("base_model")
                or st_meta.get("ss_base_model_version")
                or st_meta.get("modelspec.architecture")
                or st_meta.get("modelspec.base_model")
            )
            network_module = (
                external_meta.get("network_module")
                or st_meta.get("ss_network_module")
                or st_meta.get("modelspec.implementation")
            )
            warnings: list[str] = []
            haystack = " ".join(
                str(v).lower()
                for v in (path.name, title, base_model, network_module)
                if v
            )
            if any(hint in haystack for hint in _LYCORIS_HINTS):
                warnings.append("LyCORIS/LoKr-like LoRA may not load in diffusers")
            if base_model and "sdxl" not in str(base_model).lower() and "xl" not in str(base_model).lower():
                warnings.append(f"base model metadata: {base_model}")

            # NSFW heuristic: filename or external override
            nsfw_flag = bool(external_meta.get("nsfw"))
            if not nsfw_flag:
                name_lower = path.stem.lower()
                nsfw_flag = any(h in name_lower for h in _NSFW_HINTS)

            entry = LoRAEntry(
                name=external_meta.get("name", path.stem),
                file=path.name,
                base_model=base_model,
                trigger_words=external_meta.get("trigger_words") or auto_triggers,
                recommended_weight=float(external_meta.get("recommended_weight", 0.8)),
                min_weight=float(external_meta.get("min_weight", 0.0)),
                max_weight=float(external_meta.get("max_weight", 1.5)),
                nsfw=nsfw_flag,
                title=external_meta.get("title") or title,
                network_module=network_module,
                warnings=external_meta.get("warnings") or warnings,
            )
            entries.append(entry)
        return entries

    def file_for(self, name: str) -> Path | None:
        for entry in self.discover():
            if entry.name == name or entry.file == name:
                return self.loras_dir / entry.file
        candidate = self.loras_dir / f"{name}.safetensors"
        return candidate if candidate.exists() else None

    def applied(self) -> list[LoRASpec]:
        return list(self._applied)

    def ensure_loaded(self, pipeline, specs: list[LoRASpec]) -> None:
        """Load all LoRA adapters needed by ``specs`` if they are not loaded."""
        for spec in specs:
            if spec.name in self._loaded_adapters:
                continue
            file_path = self.file_for(spec.name)
            if not file_path:
                logger.warning("LoRA file not found: %s", spec.name)
                continue
            adapter = _adapter_name(spec.name)
            try:
                pipeline.load_lora_weights(
                    str(file_path.parent),
                    weight_name=file_path.name,
                    adapter_name=adapter,
                )
                self._loaded_adapters[spec.name] = adapter
                logger.info("Loaded LoRA adapter %s as %s", spec.name, adapter)
            except Exception as exc:
                logger.exception("Failed to load LoRA %s: %s", spec.name, exc)

    def set_active(self, pipeline, specs: list[LoRASpec]) -> None:
        """Set active LoRA weights without unloading already loaded adapters."""
        specs = _dedupe_specs(specs)
        current_keys = [s.key() for s in self._applied]
        new_keys = [s.key() for s in specs]
        if current_keys == new_keys:
            return

        self.ensure_loaded(pipeline, specs)
        if not self._loaded_adapters:
            self._applied = []
            return

        requested = {
            self._loaded_adapters[spec.name]: float(spec.weight)
            for spec in specs
            if spec.name in self._loaded_adapters
        }
        adapter_names = list(self._loaded_adapters.values())
        weights = [requested.get(adapter, 0.0) for adapter in adapter_names]
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=weights)
        except Exception as exc:
            logger.warning("set_adapters failed: %s", exc)
        self._applied = [
            spec for spec in specs if spec.name in self._loaded_adapters
        ]
        logger.debug("Active LoRAs: %s", [s.key() for s in self._applied])

    def apply(self, pipeline, specs: list[LoRASpec]) -> None:
        """Apply the LoRA spec list to ``pipeline``. No-op if unchanged."""
        self.set_active(pipeline, specs)

    def set_active_for_text_encoder(self, pipeline, specs: list[LoRASpec]) -> None:
        self.set_active(pipeline, [spec.text_encoder_spec() for spec in specs])

    def set_active_for_unet(self, pipeline, specs: list[LoRASpec]) -> None:
        self.set_active(pipeline, [spec.unet_spec() for spec in specs])

    def disable(self, pipeline) -> None:
        if not self._loaded_adapters:
            self._applied = []
            return
        adapter_names = list(self._loaded_adapters.values())
        try:
            pipeline.set_adapters(adapter_names, adapter_weights=[0.0] * len(adapter_names))
        except Exception as exc:
            logger.warning("disable LoRA adapters failed: %s", exc)
        self._applied = []

    def unload_all(self, pipeline) -> None:
        try:
            pipeline.unload_lora_weights()
        finally:
            self._applied = []
            self._loaded_adapters = {}
