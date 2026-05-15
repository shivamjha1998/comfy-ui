#!/usr/bin/env bash
# One-time install for the face-swap demo on Mac (Apple Silicon).
# Idempotent — re-runs are no-ops once everything is in place.
#
# Prerequisites the user must provide:
#   - macOS arm64 (Apple Silicon)
#   - Homebrew + python@3.11   (brew install python@3.11)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"
HOMEBREW_PY="/opt/homebrew/opt/python@3.11/bin/python3.11"

# ─── Sanity checks ────────────────────────────────────────────────────────
if [ "$(uname)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "✗ This setup targets macOS on Apple Silicon. Detected: $(uname) $(uname -m)" >&2
  exit 1
fi

if [ ! -x "$HOMEBREW_PY" ]; then
  echo "✗ Python 3.11 not found at $HOMEBREW_PY" >&2
  echo "  Install with:  brew install python@3.11" >&2
  exit 1
fi

# ─── venv ─────────────────────────────────────────────────────────────────
if [ ! -x "$VENV_PY" ]; then
  echo "[setup] creating Python 3.11 venv at .venv"
  "$HOMEBREW_PY" -m venv "$VENV_DIR"
fi
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel

# ─── ComfyUI itself ───────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/ComfyUI" ]; then
  echo "[setup] cloning ComfyUI"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$REPO_DIR/ComfyUI"
fi
echo "[setup] installing ComfyUI requirements"
"$VENV_PY" -m pip install --quiet -r "$REPO_DIR/ComfyUI/requirements.txt"

# ─── Custom nodes ─────────────────────────────────────────────────────────
NODE_DIR="$REPO_DIR/ComfyUI/custom_nodes"
mkdir -p "$NODE_DIR"

clone_if_missing() {
  local name="$1"
  local url="$2"
  if [ ! -d "$NODE_DIR/$name" ]; then
    echo "[setup] cloning $name"
    git clone --depth 1 "$url" "$NODE_DIR/$name"
  fi
}

clone_if_missing ComfyUI-ReActor              https://github.com/Gourieff/ComfyUI-ReActor.git
clone_if_missing ComfyUI-VideoHelperSuite     https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
clone_if_missing ComfyUI-Frame-Interpolation  https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# ReActor's install.py handles its own deps + the inswapper_128.onnx download
# (~530 MB on first run; idempotent after that).
echo "[setup] running ReActor install.py"
( cd "$NODE_DIR/ComfyUI-ReActor" && "$VENV_PY" install.py )

echo "[setup] installing VideoHelperSuite requirements"
"$VENV_PY" -m pip install --quiet -r "$NODE_DIR/ComfyUI-VideoHelperSuite/requirements.txt"

echo "[setup] installing Frame-Interpolation requirements (no-cupy: Mac has no CUDA)"
"$VENV_PY" -m pip install --quiet -r "$NODE_DIR/ComfyUI-Frame-Interpolation/requirements-no-cupy.txt"

# ─── gradio_app deps ──────────────────────────────────────────────────────
echo "[setup] installing gradio_app requirements"
"$VENV_PY" -m pip install --quiet -r "$REPO_DIR/gradio_app/requirements.txt"

# ─── Models not auto-fetched by anything ──────────────────────────────────
MODELS="$REPO_DIR/ComfyUI/models"

GFPGAN_PATH="$MODELS/facerestore_models/GFPGANv1.4.pth"
if [ ! -f "$GFPGAN_PATH" ]; then
  echo "[setup] downloading GFPGANv1.4 (~332 MB)"
  mkdir -p "$(dirname "$GFPGAN_PATH")"
  curl -L --fail --progress-bar -o "$GFPGAN_PATH" \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
fi

YOLO_PATH="$MODELS/ultralytics/bbox/face_yolov8m.pt"
if [ ! -f "$YOLO_PATH" ]; then
  echo "[setup] downloading face_yolov8m (~50 MB)"
  mkdir -p "$(dirname "$YOLO_PATH")"
  curl -L --fail --progress-bar -o "$YOLO_PATH" \
    https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt
fi

# ─── Mac-specific ReActor patches ─────────────────────────────────────────
echo "[setup] applying Mac ReActor patches"
bash "$REPO_DIR/scripts/patch_reactor_mac.sh"

# ─── Done ─────────────────────────────────────────────────────────────────
cat <<'EOF'

────────────────────────────────────────────────────
  ✓ Setup complete

  Start the demo:   ./run.sh
  Open in browser:  http://127.0.0.1:7860
────────────────────────────────────────────────────

EOF
