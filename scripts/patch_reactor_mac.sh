#!/usr/bin/env bash
# Apply Mac-specific patches to ComfyUI-ReActor.
#
# ComfyUI-ReActor (Gourieff fork) makes two assumptions that fail on Apple
# Silicon. This script patches them idempotently so the fixes survive a fresh
# clone of ComfyUI/custom_nodes/. Run on every ./run.sh start; a no-op if the
# files have already been patched.
#
# 1. PROVIDERS: defaults to CoreMLExecutionProvider when MPS is available, used
#    for BOTH the buffalo_l face analyzer and the actual face swap. CoreML
#    mis-handles buffalo_l's RetinaFace output shapes (silent empty detections),
#    so we split providers into two lists:
#      - analysis_providers = CPU only          (buffalo_l face detect)
#      - providers          = CoreML + CPU fallback  (inswapper / hyperswap)
#    The swap models compile cleanly on CoreML and use GPU/ANE; the analyzer
#    stays on the reliable CPU path. Empirically faster end-to-end than CPU-only.
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

# Patch 1: split providers (analysis_providers for buffalo_l, providers for swap)
if grep -q "analysis_providers" "$SWAPPER" 2>/dev/null; then
  echo "[patch_reactor_mac] patch 1/2: providers split already applied"
else
  echo "[patch_reactor_mac] patch 1/2: installing split CoreML+CPU / CPU-only providers"
  python3 - "$SWAPPER" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()

# (a) Rewrite the entire `# PROVIDERS` try/except block.
start = src.find("# PROVIDERS\ntry:")
if start == -1:
    print("[patch_reactor_mac]   '# PROVIDERS' block not found, leaving file untouched", file=sys.stderr)
    sys.exit(0)

except_pos = src.find("except Exception as e:", start)
if except_pos == -1:
    print("[patch_reactor_mac]   end of try/except not found, leaving file untouched", file=sys.stderr)
    sys.exit(0)

end_marker = 'providers = ["CPUExecutionProvider"]\n'
end_pos = src.find(end_marker, except_pos)
if end_pos == -1:
    print("[patch_reactor_mac]   end-of-except marker not found, leaving file untouched", file=sys.stderr)
    sys.exit(0)
end_pos += len(end_marker)

new_block = '''# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        analysis_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif torch.backends.mps.is_available():
        # CoreML mis-handles buffalo_l's RetinaFace output shapes (silent empty
        # detections), so face analysis stays on CPU. The actual swap step
        # (inswapper / hyperswap) compiles cleanly on CoreML and runs on GPU/ANE.
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        analysis_providers = ["CPUExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        analysis_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
        analysis_providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]
    analysis_providers = ["CPUExecutionProvider"]
'''

src = src[:start] + new_block + src[end_pos:]

# (b) Route the buffalo_l ReActorFaceAnalysis call to analysis_providers.
src = src.replace(
    'name="buffalo_l", providers=providers',
    'name="buffalo_l", providers=analysis_providers',
)

p.write_text(src)
PY
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
