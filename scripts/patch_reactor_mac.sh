#!/usr/bin/env bash
# Apply Mac-specific patches to ComfyUI-ReActor.
#
# ComfyUI-ReActor (Gourieff fork) makes two assumptions that fail on Apple
# Silicon. This script patches them idempotently so the fixes survive a fresh
# clone of ComfyUI/custom_nodes/. Run on every ./run.sh start; a no-op if the
# files have already been patched.
#
# 1. PROVIDERS: defaults to CoreMLExecutionProvider when MPS is available, but
#    CoreML mis-handles buffalo_l's RetinaFace output shapes — face detection
#    silently returns empty results ("No faces found"). Force CPU.
#
# 2. INPUT BULK TRANSFER: blindly does `input_image.to(device)` at the top of
#    execute(), then immediately calls .cpu().numpy() inside batch_tensor_to_pil.
#    The pointless GPU allocation OOMs on long clips on unified-memory Macs.
#    Skip it on MPS.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REACTOR_DIR="$REPO_DIR/ComfyUI/custom_nodes/ComfyUI-ReActor"

if [ ! -d "$REACTOR_DIR" ]; then
  echo "[patch_reactor_mac] ComfyUI-ReActor not installed at $REACTOR_DIR — skipping"
  exit 0
fi

SWAPPER="$REACTOR_DIR/scripts/reactor_swapper.py"
NODES="$REACTOR_DIR/nodes.py"

# Patch 1: CoreML -> CPU on MPS
if grep -q 'elif torch.backends.mps.is_available():\s*$' "$SWAPPER" 2>/dev/null \
   && grep -A1 'elif torch.backends.mps.is_available():' "$SWAPPER" | grep -q '"CoreMLExecutionProvider"'; then
  echo "[patch_reactor_mac] patch 1/2: forcing CPU provider on MPS"
  python3 - "$SWAPPER" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]'''
new = '''    elif torch.backends.mps.is_available():
        # CoreMLExecutionProvider mis-handles buffalo_l's RetinaFace output shapes,
        # silently returning empty detections ("No faces found"). CPU is reliable
        # and fast enough for face-analysis at our throughput.
        providers = ["CPUExecutionProvider"]'''
if old in src:
    p.write_text(src.replace(old, new))
PY
else
  echo "[patch_reactor_mac] patch 1/2: already applied"
fi

# Patch 2: skip input_image.to(device) on MPS
if ! grep -q "not torch.backends.mps.is_available()" "$NODES" 2>/dev/null; then
  echo "[patch_reactor_mac] patch 2/2: skipping wasteful MPS bulk transfer"
  python3 - "$NODES" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = '''        if isinstance(input_image, torch.Tensor) and input_image.device != device:
            input_image = input_image.to(device)'''
new = '''        # On MPS, skip the bulk GPU transfer: the next call (batch_tensor_to_pil)
        # immediately moves the tensor back to CPU via .cpu().numpy(), so paying
        # for a multi-GB MPS allocation here just OOMs long clips on 16 GB Macs.
        if isinstance(input_image, torch.Tensor) and input_image.device != device:
            if not torch.backends.mps.is_available():
                input_image = input_image.to(device)'''
if old in src:
    p.write_text(src.replace(old, new))
PY
else
  echo "[patch_reactor_mac] patch 2/2: already applied"
fi
