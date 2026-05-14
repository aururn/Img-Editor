"""Discovery helpers for Stable Diffusion sidecar assets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


_MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
_EMBEDDING_EXTS = {".safetensors", ".pt", ".bin"}
_CONTROL_EXTS = {".safetensors", ".bin"}
_IP_EXTS = {".safetensors", ".bin"}


@dataclass
class CheckpointSpec:
    name: str
    path: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class EmbeddingSpec:
    name: str
    token: str
    file: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ControlNetSpec:
    name: str
    path: str
    source_type: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class IPAdapterSpec:
    name: str
    path: str
    weight_name: str | None = None
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse asset metadata %s: %s", path, exc)
        return {}


def _iter_asset_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return files


def discover_checkpoints(root: str | Path) -> list[CheckpointSpec]:
    entries: list[CheckpointSpec] = []
    for path in _iter_asset_files(Path(root), _MODEL_EXTS):
        warnings: list[str] = []
        if path.suffix.lower() != ".safetensors":
            warnings.append("non-safetensors checkpoint")
        entries.append(
            CheckpointSpec(
                name=path.stem,
                path=path.name,
                warnings=warnings,
            )
        )
    return entries


def discover_embeddings(root: str | Path) -> list[EmbeddingSpec]:
    root = Path(root)
    entries: list[EmbeddingSpec] = []
    for path in _iter_asset_files(root, _EMBEDDING_EXTS):
        meta = _read_json(path.with_suffix(".json"))
        token = str(meta.get("token") or path.stem)
        warnings: list[str] = []
        if path.suffix.lower() == ".safetensors":
            # SDXL textual inversion embeddings often need clip_l and clip_g.
            # We do a cheap header-only check and warn if this is ambiguous.
            try:
                from safetensors import safe_open

                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                if not ({"clip_l", "clip_g"} & keys):
                    warnings.append("SDXL TI may need clip_l/clip_g tensors")
            except Exception:
                warnings.append("metadata unavailable")
        entries.append(
            EmbeddingSpec(
                name=str(meta.get("name") or path.stem),
                token=token,
                file=path.name,
                warnings=warnings,
            )
        )
    return entries


def discover_controlnets(root: str | Path) -> list[ControlNetSpec]:
    root = Path(root)
    entries: list[ControlNetSpec] = []
    if not root.exists():
        return entries

    for path in sorted(root.iterdir()):
        if path.is_dir():
            meta = _read_json(path / "asset.json")
            if (path / "config.json").exists() or meta:
                entries.append(
                    ControlNetSpec(
                        name=str(meta.get("name") or path.name),
                        path=path.name,
                        source_type="directory",
                    )
                )
        elif path.is_file() and path.suffix.lower() in _CONTROL_EXTS:
            meta = _read_json(path.with_suffix(".json"))
            entries.append(
                ControlNetSpec(
                    name=str(meta.get("name") or path.stem),
                    path=path.name,
                    source_type="single_file",
                    warnings=["single-file ControlNet support depends on diffusers version"],
                )
            )
    return entries


def discover_ip_adapters(root: str | Path) -> list[IPAdapterSpec]:
    root = Path(root)
    entries: list[IPAdapterSpec] = []
    if not root.exists():
        return entries

    for path in sorted(root.iterdir()):
        if path.is_dir():
            meta = _read_json(path / "asset.json")
            weight_name = meta.get("weight_name")
            if not weight_name:
                candidates = _iter_asset_files(path, _IP_EXTS)
                weight_name = candidates[0].name if candidates else None
            entries.append(
                IPAdapterSpec(
                    name=str(meta.get("name") or path.name),
                    path=path.name,
                    weight_name=str(weight_name) if weight_name else None,
                )
            )
        elif path.is_file() and path.suffix.lower() in _IP_EXTS:
            meta = _read_json(path.with_suffix(".json"))
            entries.append(
                IPAdapterSpec(
                    name=str(meta.get("name") or path.stem),
                    path=path.name,
                    weight_name=path.name,
                )
            )
    return entries
