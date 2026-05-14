"""Compatibility wrapper for scripts/fetch-civitai-model.py."""

from pathlib import Path
import runpy


SCRIPT_PATH = Path(__file__).resolve().parent / "scripts" / "fetch-civitai-model.py"
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
