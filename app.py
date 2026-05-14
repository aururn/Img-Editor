"""Img Editor — Gradio entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from src.core import load_config
from src.ui import build_ui
from src.utils import setup_logging
from src.utils.logger import get_logger


def _launch_max_file_size(config) -> int | None:
    raw_limit = config.limits.get("max_file_size_mb")
    if raw_limit is None:
        return None

    try:
        size_mb = float(raw_limit)
    except (TypeError, ValueError):
        return None

    if size_mb <= 0:
        return None

    return int(size_mb * 1024 * 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Img Editor")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--share", action="store_true", help="Expose via Gradio share link"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    setup_logging(
        log_dir=config.path("logs_dir") if "logs_dir" in config.paths else "./logs",
        level=config.app.get("log_level", "INFO"),
    )
    logger = get_logger("img_editor")
    logger.info("Starting Img Editor")

    host = args.host or config.app.get("host", "127.0.0.1")
    port = args.port or int(config.app.get("port", 7860))
    max_file_size = _launch_max_file_size(config)
    if max_file_size is not None:
        logger.info("Max UI upload size: %.0f MB", max_file_size / 1024 / 1024)

    demo = build_ui(config)
    demo.queue().launch(
        server_name=host,
        server_port=port,
        share=args.share,
        inbrowser=True,
        max_file_size=max_file_size,
    )


if __name__ == "__main__":
    main()
