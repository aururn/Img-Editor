#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/workspace/ImgEditor}"
PYTHON_BIN="${PYTHON_BIN:-}"
DOWNLOAD_CHECKPOINTS=1
START_APP=0
SKIP_INSTALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-checkpoints)
      DOWNLOAD_CHECKPOINTS=1
      ;;
    --skip-checkpoints)
      DOWNLOAD_CHECKPOINTS=0
      ;;
    --start)
      START_APP=1
      ;;
    --no-install)
      SKIP_INSTALL=1
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

detect_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return
  fi
  if [[ -x "/venv/main/bin/python3" ]]; then
    PYTHON_BIN="/venv/main/bin/python3"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
}

if [[ ! -d "$APP_DIR" ]]; then
  echo "App dir not found: $APP_DIR" >&2
  exit 1
fi

cd "$APP_DIR"
detect_python

log "Create directories"
mkdir -p \
  models/checkpoints \
  models/loras \
  models/detection \
  models/vae \
  models/embeddings \
  models/controlnet \
  models/ip_adapter \
  outputs \
  presets \
  logs

log "Environment check"
df -h /workspace || true
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import sys
print("python", sys.executable)
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
  log "Install Python dependencies"
  "$PYTHON_BIN" -m pip install --no-cache-dir -r requirements.txt

  log "Upgrade Gradio stack for Vast.ai"
  "$PYTHON_BIN" -m pip install --no-cache-dir --upgrade 'gradio<6' fastapi starlette 'pydantic>=2.12,<3'

  log "Installed package versions"
  "$PYTHON_BIN" -m pip show gradio gradio_client diffusers transformers accelerate peft | grep -E "^Name|^Version" || true
else
  log "Skip Python dependency installation"
fi

install_aria2() {
  if command -v aria2c >/dev/null 2>&1; then
    return
  fi
  log "Install aria2"
  apt-get install -y aria2 || {
    apt-get update
    apt-get install -y aria2
  }
}

download_checkpoint() {
  local url="$1"
  local final_url="$url"
  if [[ -n "${CIVITAI_TOKEN:-}" ]]; then
    final_url="${url}&token=${CIVITAI_TOKEN}"
  fi
  aria2c \
    -x 16 \
    -s 16 \
    --continue=true \
    --max-tries=5 \
    --retry-wait=10 \
    --auto-file-renaming=false \
    --content-disposition=true \
    "$final_url"
}

if [[ "$DOWNLOAD_CHECKPOINTS" -eq 1 ]]; then
  install_aria2
  log "Download checkpoints"
  cd "$APP_DIR/models/checkpoints"
  if [[ -z "${CIVITAI_TOKEN:-}" ]]; then
    echo "CIVITAI_TOKEN is empty. Public downloads may work, but restricted models can fail." >&2
  fi
  download_checkpoint "https://civitai.com/api/download/models/889818?type=Model&format=SafeTensor&size=pruned&fp=fp16"
  download_checkpoint "https://civitai.red/api/download/models/2883731?type=Model&format=SafeTensor&size=pruned&fp=fp16"
  download_checkpoint "https://civitai.com/api/download/models/1190596?type=Model&format=SafeTensor&size=full&fp=bf16"
  ls -lh ./*.safetensors || true
  unset CIVITAI_TOKEN
  cd "$APP_DIR"
else
  log "Skip checkpoint downloads"
fi

log "Update config base_checkpoint if needed"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

try:
    import yaml
except Exception as exc:
    print(f"PyYAML unavailable; skip config update: {exc}")
    raise SystemExit(0)

root = Path.cwd()
cfg_path = root / "config.yaml"
ckpt_dir = root / "models" / "checkpoints"
checkpoints = sorted(
    p for p in ckpt_dir.glob("*.safetensors")
    if p.is_file() and p.stat().st_size > 1024 * 1024
)
if not checkpoints:
    print("No checkpoint files found; config.yaml unchanged")
    raise SystemExit(0)

data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
model = data.setdefault("model", {})
current = str(model.get("base_checkpoint") or "").strip()
if current and (ckpt_dir / current).exists():
    print(f"base_checkpoint already valid: {current}")
    raise SystemExit(0)

model["base_checkpoint"] = checkpoints[0].name
cfg_path.write_text(
    yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
print(f"base_checkpoint set to: {checkpoints[0].name}")
PY

if [[ "$START_APP" -eq 1 ]]; then
  log "Start Img Editor"
  exec "$PYTHON_BIN" app.py
fi

log "Setup complete"
echo "Start app with:"
echo "  cd $APP_DIR && $PYTHON_BIN app.py"
