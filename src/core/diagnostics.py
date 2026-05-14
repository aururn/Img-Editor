"""Runtime dependency and asset diagnostics."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig


@dataclass
class DiagnosticReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_markdown(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append("### Errors")
            lines.extend(f"- {item}" for item in self.errors)
        if self.warnings:
            lines.append("### Warnings")
            lines.extend(f"- {item}" for item in self.warnings)
        if self.info:
            lines.append("### Info")
            lines.extend(f"- {item}" for item in self.info)
        return "\n".join(lines) if lines else "Diagnostics: OK"


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_diagnostics(config: AppConfig) -> DiagnosticReport:
    report = DiagnosticReport()

    required = ("diffusers", "transformers", "accelerate", "torch", "PIL", "gradio")
    for name in required:
        if not _module_available(name):
            report.errors.append(f"Missing required Python package: {name}")

    if not _module_available("peft"):
        report.errors.append("PEFT is missing; LoRA loading will fail.")

    if not _module_available("safetensors"):
        report.errors.append("safetensors is missing; .safetensors assets cannot be read.")

    if not _module_available("cv2"):
        report.warnings.append("opencv-python is missing; ControlNet Canny preprocessing is disabled.")

    checkpoint = config.path("checkpoints_dir") / str(config.model.get("base_checkpoint", ""))
    if not checkpoint.exists():
        report.errors.append(f"Base checkpoint not found: {checkpoint}")

    for key in (
        "loras_dir",
        "embeddings_dir",
        "controlnet_dir",
        "ip_adapter_dir",
        "vae_dir",
        "detection_dir",
        "outputs_dir",
        "presets_dir",
        "logs_dir",
    ):
        path = Path(config.path(key))
        if not path.exists():
            report.warnings.append(f"Directory does not exist yet: {path}")

    return report

