#!/bin/bash
#
# launch.sbatch — Slurm batch script that brings ComfyUI + Gradio online
# ─────────────────────────────────────────────────────────────────────────
#
# Submit from the SoftBank login server with:
#   sbatch scripts/launch.sbatch
#
# Once running, find the GPU node hostname:
#   squeue -h -o %N -j <jobid>
#
# Then from your Mac mini, open the second-hop tunnel via:
#   scripts/connect.sh --gpu-node <hostname>
#
# What this job does:
#   1. Allocates 8× A100 on 082-partition exclusively (the whole DGX node)
#   2. Loads the Slurm module (so module-based env is set up)
#   3. Starts the container via Pyxis with /lustre/comfyui bind-mounted
#   4. Inside the container: launches ComfyUI in the background, then Gradio
#      in the foreground (so the job stays alive as long as Gradio is up)
#
# ─── Slurm directives ────────────────────────────────────────────────────
#SBATCH -J comfyui-faceswap
#SBATCH -p 082-partition
#SBATCH -N 1
#SBATCH --gpus=8
#SBATCH --exclusive
#SBATCH -t 0-12:00:00
#SBATCH -o /lustre/comfyui/logs/launch-%j.out
#SBATCH -e /lustre/comfyui/logs/launch-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sales@abckk.dev

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────
# /lustre is 100 GB total — too small for the WAN 2.2 stack (~36 GB) alongside
# Qwen (~47 GB). Models and user-uploaded inputs now live on /home (1 PB).
# Output, logs, and the container .sqsh stay on /lustre (transient / faster fs).
LUSTRE_BASE="/lustre/comfyui"
HOME_BASE="${HOME}/comfyui"
IMAGE_TAG="${IMAGE_TAG:-$(ls -t "${LUSTRE_BASE}"/comfyui-*.sqsh 2>/dev/null | head -1 | sed 's|.*/comfyui-||;s|\.sqsh$||')}"
SQSH_FILE="${LUSTRE_BASE}/comfyui-${IMAGE_TAG}.sqsh"

COMFYUI_PORT="${COMFYUI_PORT:-8188}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

echo "── launch.sbatch ───────────────────────────────────────"
echo "  Job ID       : ${SLURM_JOB_ID}"
echo "  Node         : ${SLURM_JOB_NODELIST}"
echo "  Partition    : ${SLURM_JOB_PARTITION}"
echo "  Image        : ${SQSH_FILE}"
echo "  ComfyUI port : ${COMFYUI_PORT}"
echo "  Gradio port  : ${GRADIO_PORT}"
echo "  Start time   : $(date)"
echo "────────────────────────────────────────────────────────"

# ─── Pre-flight ──────────────────────────────────────────────────────────
[[ -f "${SQSH_FILE}" ]] || { echo "ERROR: ${SQSH_FILE} not found. Run scripts/login_server_setup.sh first."; exit 1; }

module load slurm/slurm/23.02.7 || true

# ─── Bind mounts ─────────────────────────────────────────────────────────
# Models + input on /home (capacity); output + logs on /lustre (perf, transient).
# /home has a 61 GB user quota so the big WF-B files (umt5 + 14B, ~23 GB) live
# on /lustre/comfyui/models-staging and are stitched into /home via symlink.
# That dir must be bind-mounted at the same path so the absolute symlinks resolve.
MOUNTS="${HOME_BASE}/models:/workspace/ComfyUI/models"
MOUNTS+=",${LUSTRE_BASE}/models-staging:${LUSTRE_BASE}/models-staging"
MOUNTS+=",${LUSTRE_BASE}/output:/workspace/ComfyUI/output"
MOUNTS+=",${HOME_BASE}/input:/workspace/ComfyUI/input"
MOUNTS+=",${HOME}/comfyui-faceswap/workflows:/workspace/workflows"
MOUNTS+=",${HOME}/comfyui-faceswap/gradio_app:/workspace/app"
MOUNTS+=",${LUSTRE_BASE}/logs:/workspace/logs"

# ─── In-container launcher ───────────────────────────────────────────────
# This is the script that actually runs inside the container. ComfyUI is
# started in the background; Gradio runs in the foreground. The job ends
# when Gradio exits (Ctrl-C from `srun --pty`, or the time limit hits).
read -r -d '' IN_CONTAINER_CMD <<'INNER' || true
set -euo pipefail

export INSIGHTFACE_HOME=/workspace/ComfyUI/models/insightface
export HF_HOME=/workspace/ComfyUI/models/huggingface

# ─── Container-stability env ─────────────────────────────────────────────
# The CUDA 13 NGC base segfaults inside libtorch_cpu.so during multi-threaded
# TensorIterator ops (signal 11 from libgomp parallel region). Layered
# mitigation: kill multi-threading at every layer (OMP/MKL/torch), disable
# UCX memory-type cache (a common cause of "invalid permissions" segfaults
# in containers), and try generic aimdo-disable env vars.
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export OMP_NESTED=FALSE
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export TORCH_NUM_THREADS=1
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=^cuda
# Best-effort aimdo disables — the package ignores unknown env vars so this
# is harmless if none of these are the actual flag name.
export AIMDO_DISABLE=1
export COMFY_AIMDO_DISABLE=1
export NVIDIA_AIMDO_DISABLE=1

# ONNXRuntime in this container was built against CUDA 12 (libcublasLt.so.12)
# but the base image ships CUDA 13. Symlink so ORT can find a usable lib;
# this is best-effort, ORT falls back to CPU if the symlink fails.
CUBLASLT_13=$(ls /usr/local/cuda/lib64/libcublasLt.so.13* 2>/dev/null | head -1)
if [ -n "${CUBLASLT_13}" ]; then
    ln -sf "${CUBLASLT_13}" /usr/local/cuda/lib64/libcublasLt.so.12 || true
    echo "[launcher] symlinked libcublasLt.so.12 -> ${CUBLASLT_13}"
fi

# Fix cv2 conflict: remove ALL cv2 files (pip and non-pip) then reinstall clean
echo "[launcher] Patching opencv…"
pip uninstall -y opencv-python opencv-python-headless \
    opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true
# Also nuke any cv2 the NGC base dropped outside pip's registry
find /usr/local/lib/python3.10/dist-packages \
     /usr/lib/python3/dist-packages \
     -maxdepth 2 \( -name "cv2" -type d -o -name "cv2*.so" \) \
     -exec rm -rf {} + 2>/dev/null || true
pip install --no-cache-dir "opencv-python-headless==4.10.0.84" 2>&1 | tail -5
echo "[launcher] numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[launcher] cv2:   $(python3 -c 'import cv2; print(cv2.__version__, cv2.__file__)' 2>&1)"

# Install ComfyUI-KJNodes — kijai's general-utility nodes that the WanAnimate
# example workflows depend on (GetImageSizeAndCount, BlockifyMask, ImageResizeKJv2,
# SetNode/GetNode reroute helpers, etc.). Not in the container by default.
KJ_DIR=/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes
if [ ! -d "${KJ_DIR}" ]; then
    echo "[launcher] Cloning ComfyUI-KJNodes…"
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git "${KJ_DIR}"
    if [ -f "${KJ_DIR}/requirements.txt" ]; then
        pip install --no-cache-dir -r "${KJ_DIR}/requirements.txt" 2>&1 | tail -5 || \
            echo "[launcher] WARN: KJNodes requirements install had issues"
    fi
else
    echo "[launcher] ComfyUI-KJNodes already present"
fi

# NSFW checker: Pre-download model directly into the Lustre model path
echo "[launcher] Wiring up NSFW checker model…"
rm -f /workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector 2>/dev/null || true
mkdir -p /workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector

# Download into the bind-mounted dir if model is missing or truncated (< 100 MB)
NSFW_MODEL=/workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector/model.safetensors
NSFW_SIZE=$(stat -c%s "${NSFW_MODEL}" 2>/dev/null || echo 0)
if [ ! -f "${NSFW_MODEL}" ] || [ "${NSFW_SIZE}" -lt 104857600 ]; then
    echo "[launcher] NSFW model missing or truncated (${NSFW_SIZE} bytes), re-downloading…"
    rm -f "${NSFW_MODEL}" 2>/dev/null || true
    echo "[launcher] Downloading NSFW checker model via wget…"
    wget -q -O /workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector/config.json https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json || true
    wget -q -O /workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector/preprocessor_config.json https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json || true
    wget -q --show-progress -O /workspace/ComfyUI/models/nsfw_detector/vit-base-nsfw-detector/model.safetensors https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors || echo "[launcher] WARN: NSFW model download failed"
else
    echo "[launcher] NSFW checker model OK (${NSFW_SIZE} bytes)."
fi

# Wan 2.1 VAE for Wan 2.2 Animate pose control
echo "[launcher] Checking Wan 2.1 VAE on Lustre..."
LUSTRE_VAE_DIR="/lustre/comfyui/models-staging/vae"
mkdir -p "${LUSTRE_VAE_DIR}"
VAE_PATH="${LUSTRE_VAE_DIR}/Wan2.1_VAE.pth"
VAE_SIZE=$(stat -c%s "${VAE_PATH}" 2>/dev/null || echo 0)
if [ ! -f "${VAE_PATH}" ] || [ "${VAE_SIZE}" -lt 1000000000 ]; then
    echo "[launcher] Wan 2.1 VAE missing on Lustre, downloading..."
    rm -f "${VAE_PATH}" 2>/dev/null || true
    wget -q --show-progress -O "${VAE_PATH}" https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/Wan2.1_VAE.pth || echo "[launcher] WARN: Wan 2.1 VAE download failed"
else
    echo "[launcher] Wan 2.1 VAE OK."
fi
# Symlink into models dir
mkdir -p /workspace/ComfyUI/models/vae
ln -sf "${VAE_PATH}" /workspace/ComfyUI/models/vae/Wan2.1_VAE.pth


cd /workspace/ComfyUI

# ─── Fix WAN 2.2 Animate 64ch → 36ch patch_embedding channel mismatch ────────
# Older container builds of model_base.py lack WAN21.concat_cond, causing the
# sampling code to pass 64 channels into patch_embedding (noise=16 + wrong
# c_concat=48) when the model expects 36 (noise=16 + ref=16 + mask=4).
#
# Rather than replacing model_base.py (risks API incompatibility), we write a
# ComfyUI custom node that monkey-patches WAN21.concat_cond at load time with
# the correct weight-shape-aware implementation from current ComfyUI main.
echo "[launcher] Writing WAN concat_cond fix (custom node)…"
python3 - <<'PYEOF'
fix_code = r'''
"""
WAN 2.2 Animate 64ch->36ch fix.

Strategy: patch extra_conds on WAN22_Animate so c_concat is always built via
concat_cond (20ch = mask4 + ref16) instead of whatever the native extra_conds
assembles. Also print extra_conds source and catch concat_cond exceptions.
"""
import inspect
import traceback
import torch
import comfy.model_base
import comfy.conds
import comfy.utils


def _build_c_concat(self, **kwargs):
    """Build c_concat tensor: mask(4ch) + ref_image(16ch) = 20ch for 36ch patch_embedding."""
    try:
        noise = kwargs.get("noise", None)
        if noise is None:
            print("[fix_wan22_animate] concat_cond: noise missing from kwargs")
            return None
        device = noise.device

        extra_channels = self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
        print(f"[fix_wan22_animate] concat_cond: noise={noise.shape} patch_emb_in={self.diffusion_model.patch_embedding.weight.shape[1]} extra_ch={extra_channels}")
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        if image is None:
            s = list(noise.shape)
            s[1] = extra_channels
            result = torch.zeros(s, dtype=noise.dtype, layout=noise.layout, device=noise.device)
            print(f"[fix_wan22_animate] concat_cond: no ref image, returning zeros {result.shape}")
            return result

        print(f"[fix_wan22_animate] concat_cond: concat_latent_image.shape={image.shape}")
        image = image.to(device)

        # Handle 5D video latent [B, C, T, H, W]
        if image.ndim == 5:
            B, C, T, H, W = image.shape
            tgt_h, tgt_w = noise.shape[-2], noise.shape[-1]
            tgt_t = noise.shape[-3]
            if H != tgt_h or W != tgt_w:
                flat = image.reshape(B * T, C, H, W)
                flat = comfy.utils.common_upscale(flat, tgt_w, tgt_h, "bilinear", "center")
                image = flat.reshape(B, C, T, tgt_h, tgt_w)
            if T < tgt_t:
                # Repeat the last frame across the missing time slots — zero-pad
                # would leave frames 2-tgt_t with no reference info, causing the
                # diffusion to default to the original target video appearance.
                last = image[:, :, -1:].expand(-1, -1, tgt_t - T, -1, -1).contiguous()
                image = torch.cat([image, last], dim=2)
                print(f"[fix_wan22_animate] image: repeated frame {T-1} to fill {tgt_t-T} timesteps -> {image.shape}")
            elif T > tgt_t:
                image = image[:, :, :tgt_t]
        else:
            image = comfy.utils.common_upscale(image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])

        # Trim to extra_channels - 4 (leaving room for mask)
        max_img_ch = extra_channels - 4
        if max_img_ch <= 0:
            # No room for mask+image, just return image channels
            if image.shape[1] > extra_channels:
                image = image[:, :extra_channels]
            print(f"[fix_wan22_animate] concat_cond: returning image-only {image.shape}")
            return image
        if image.shape[1] > max_img_ch:
            image = image[:, :max_img_ch]

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            if image.ndim == 5:
                mask = torch.zeros(noise.shape[0], 4, noise.shape[-3], noise.shape[-2], noise.shape[-1],
                                   dtype=noise.dtype, device=noise.device)
            else:
                mask = torch.zeros_like(noise)[:, :4]
        else:
            mask = mask.to(device)
            if mask.ndim == 5:
                B, C, T, H, W = mask.shape
                tgt_h, tgt_w, tgt_t = noise.shape[-2], noise.shape[-1], noise.shape[-3]
                if C != 4:
                    mask = torch.mean(mask, dim=1, keepdim=True).expand(-1, 4, -1, -1, -1)
                mask = 1.0 - mask
                if H != tgt_h or W != tgt_w:
                    flat = mask.reshape(B * T, 4, H, W)
                    flat = comfy.utils.common_upscale(flat, tgt_w, tgt_h, "bilinear", "center")
                    mask = flat.reshape(B, 4, T, tgt_h, tgt_w)
                if T < tgt_t:
                    # Repeat last frame instead of zero-padding (same reasoning
                    # as the image branch above).
                    last = mask[:, :, -1:].expand(-1, -1, tgt_t - T, -1, -1).contiguous()
                    mask = torch.cat([mask, last], dim=2)
                    print(f"[fix_wan22_animate] mask: repeated frame {T-1} to fill {tgt_t-T} timesteps -> {mask.shape}")
                elif T > tgt_t:
                    mask = mask[:, :, :tgt_t]
            else:
                if mask.shape[1] != 4:
                    mask = torch.mean(mask, dim=1, keepdim=True)
                mask = 1.0 - mask
                mask = comfy.utils.common_upscale(mask, noise.shape[-1], noise.shape[-2], "bilinear", "center")
                if mask.shape[1] == 1:
                    mask = mask.repeat(1, 4, 1, 1)
            mask = comfy.utils.resize_to_batch_size(mask, noise.shape[0])

        result = torch.cat((mask, image), dim=1)
        print(f"[fix_wan22_animate] concat_cond: returning {result.shape}")
        return result

    except Exception as e:
        print(f"[fix_wan22_animate] concat_cond EXCEPTION: {e}")
        traceback.print_exc()
        return None


def _print_extra_conds_source():
    """Print extra_conds source for all WAN classes to see how c_concat is built."""
    for name in sorted(dir(comfy.model_base)):
        if not name.upper().startswith("WAN"):
            continue
        cls = getattr(comfy.model_base, name)
        if not isinstance(cls, type):
            continue
        ec = getattr(cls, "extra_conds", None)
        if ec is None:
            continue
        try:
            src = inspect.getsource(ec)
        except Exception:
            continue
        lines = src.split("\n")
        print(f"[diag4] {name}.extra_conds ({len(lines)} lines):")
        for i, line in enumerate(lines[:60]):
            print(f"[diag4]  {i+1:3d}: {line}")
        break  # first WAN class is enough


def _apply_patch():
    patched_any = False
    for name in dir(comfy.model_base):
        if not name.upper().startswith("WAN"):
            continue
        cls = getattr(comfy.model_base, name)
        if not isinstance(cls, type):
            continue
        existing = getattr(cls, "concat_cond", None)
        had_correct = False
        if existing is not None:
            try:
                src = inspect.getsource(existing)
                had_correct = "patch_embedding.weight.shape" in src
            except (OSError, TypeError):
                pass
        # Patch concat_cond
        cls.concat_cond = _build_c_concat
        status = "force-replaced (native had weight.shape but still wrong)" if had_correct else "patched (was missing)"
        print(f"[fix_wan22_animate] {name}.concat_cond {status}")

        # Also patch extra_conds to guarantee c_concat uses our concat_cond
        _original_extra_conds = getattr(cls, "extra_conds", None)
        if _original_extra_conds is not None:
            def _make_fixed_extra_conds(orig):
                def _fixed_extra_conds(self, **kwargs):
                    out = orig(self, **kwargs)
                    try:
                        c_concat = _build_c_concat(self, **kwargs)
                        if c_concat is not None:
                            out["c_concat"] = comfy.conds.CONDRegular(c_concat)
                        elif "c_concat" in out:
                            del out["c_concat"]
                    except Exception as e:
                        print(f"[fix_wan22_animate] extra_conds override EXCEPTION: {e}")
                        traceback.print_exc()
                    return out
                return _fixed_extra_conds
            cls.extra_conds = _make_fixed_extra_conds(_original_extra_conds)
            print(f"[fix_wan22_animate] {name}.extra_conds overridden to force c_concat via concat_cond")
        patched_any = True

    if not patched_any:
        print("[fix_wan22_animate] WARNING: No WAN classes found in model_base")


_print_extra_conds_source()
_apply_patch()
'''
dest = '/workspace/ComfyUI/custom_nodes/fix_wan22_animate.py'
with open(dest, 'w') as f:
    f.write(fix_code)
print(f'[launcher] {dest} written OK')
PYEOF



# Patch ComfyUI model_base.py: WAN Animate pose_latents normalisation crash.
# Bug: extra_conds() calls self.process_latent_in(pose_latents), which does
# (x - latents_mean) / latents_std with 16-channel mean/std. WAN Animate's
# pose_latents have 48 channels (3 frames × 16 ch concatenated) so broadcasting
# fails with "size of tensor a (48) must match the size of tensor b (16)".
# Fix: skip process_latent_in for pose_latents — pose has its own embedding path
# in the WAN model and doesn't need VAE-space normalization.
echo "[launcher] Patching model_base.py pose_latents normalisation…"
MB=/workspace/ComfyUI/comfy/model_base.py
if grep -q "CONDRegular(self.process_latent_in(pose_latents))" "${MB}"; then
    sed -i 's|CONDRegular(self\.process_latent_in(pose_latents))|CONDRegular(pose_latents)|' "${MB}"
    echo "[launcher]   patched ${MB}"
else
    echo "[launcher]   pattern not found in ${MB} — already patched or upstream changed"
fi


# Patch ComfyUI sd.py to bypass torchaudio audio_vae import.
# The CUDA 13 NGC base ships a torchaudio whose ABI doesn't match the bundled
# torch ("undefined symbol: torch_library_impl"), but ComfyUI's comfy/sd.py
# does an unconditional `import comfy.ldm.lightricks.vae.audio_vae` which
# transitively pulls torchaudio and crashes ComfyUI on startup. We don't use
# audio models, so just remove that import line.
echo "[launcher] Patching ComfyUI sd.py to bypass torchaudio audio_vae import…"
SD_PY=/workspace/ComfyUI/comfy/sd.py
if grep -qE '^import comfy\.ldm\.lightricks\.vae\.audio_vae' "${SD_PY}"; then
    sed -i 's|^import comfy\.ldm\.lightricks\.vae\.audio_vae$|# patched out: torchaudio ABI mismatch in CUDA 13 NGC base|' "${SD_PY}"
    echo "[launcher]   patched ${SD_PY}"
else
    echo "[launcher]   audio_vae import not found in ${SD_PY} — already patched or moved"
fi

# Same defensive patch for ANY ComfyUI module (core + custom nodes) that
# hard-imports torchaudio. Replace `import torchaudio` (top-level statement
# only) with a safe try/except. We also handle `from torchaudio import …`.
echo "[launcher] Wrapping torchaudio imports in try/except across ComfyUI…"
python3 - <<'PYEOF'
import os, re
root = '/workspace/ComfyUI'
patched = 0
for dirpath, _, files in os.walk(root):
    if '/.git/' in dirpath:
        continue
    for fname in files:
        if not fname.endswith('.py'):
            continue
        path = os.path.join(dirpath, fname)
        try:
            with open(path) as f:
                src = f.read()
        except Exception:
            continue
        if 'torchaudio' not in src:
            continue
        new = src
        # Wrap bare `import torchaudio` and `import torchaudio.X` and `import torchaudio as X`
        new = re.sub(
            r'^(import torchaudio[^\n]*)$',
            r'try:\n    \1\nexcept Exception:\n    torchaudio = None  # patched: torchaudio ABI mismatch',
            new, flags=re.M)
        # Wrap `from torchaudio[…] import …`
        new = re.sub(
            r'^(from torchaudio[^\n]+)$',
            r'try:\n    \1\nexcept Exception:\n    pass  # patched: torchaudio ABI mismatch',
            new, flags=re.M)
        if new != src:
            with open(path, 'w') as f:
                f.write(new)
            patched += 1
            print(f'[launcher]   wrapped torchaudio in {path}')
print(f'[launcher] torchaudio import wrap: {patched} files patched')
PYEOF


# Diagnostic: log nodes_wan.py version so we know if pose_video_latent routing
# is correct (new approach) or needs a follow-up fix (old channel-concat approach).
echo "[launcher] Checking nodes_wan.py WanAnimateToVideo version…"
python3 - <<'PYEOF'
path = '/workspace/ComfyUI/comfy_extras/nodes_wan.py'
try:
    content = open(path).read()
    if '"pose_video_latent"' in content or "'pose_video_latent'" in content:
        print('[diag] nodes_wan.py: pose_video_latent key found — WanAnimateToVideo routes pose correctly (new version)')
    else:
        print('[diag] nodes_wan.py: pose_video_latent key NOT found — WanAnimateToVideo may use old channel-concat approach')
        print('[diag]   → pose conditioning may not work; a follow-up nodes_wan.py fix may be needed')
except Exception as e:
    print(f'[diag] Could not read nodes_wan.py: {e}')
PYEOF

# Start ComfyUI in the background.
# --listen 0.0.0.0 so the SSH tunnel from the Mac (Mac→login→fcdgx_NNN:8188)
# can reach it. The tunnel forwards to the GPU node's external IP, not loopback.
echo "[launcher] Starting ComfyUI on port ${COMFYUI_PORT}…"
python main.py \
    --listen 0.0.0.0 \
    --port "${COMFYUI_PORT}" \
    --output-directory /workspace/ComfyUI/output \
    --input-directory /workspace/ComfyUI/input \
    > /workspace/logs/comfyui-${SLURM_JOB_ID}.log 2>&1 &
COMFYUI_PID=$!

# Trap so Ctrl-C / job kill cleans up the background ComfyUI process
cleanup() {
    echo "[launcher] Cleaning up (ComfyUI PID=${COMFYUI_PID})"
    kill "${COMFYUI_PID}" 2>/dev/null || true
    wait "${COMFYUI_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait briefly for ComfyUI to bind its port
echo "[launcher] Waiting for ComfyUI to come up…"
for i in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:${COMFYUI_PORT}/system_stats" >/dev/null 2>&1; then
        echo "[launcher] ComfyUI is up after ${i}s."
        break
    fi
    sleep 1
done

# Start Gradio in the foreground — this is what keeps the job alive
echo "[launcher] Starting Gradio on port ${GRADIO_PORT}…"
cd /workspace
COMFYUI_HOST=127.0.0.1 \
COMFYUI_PORT="${COMFYUI_PORT}" \
GRADIO_HOST=0.0.0.0 \
GRADIO_PORT="${GRADIO_PORT}" \
WORKFLOWS_DIR=/workspace/workflows \
python -m app
INNER

# ─── Pre-create bind-mount targets ──────────────────────────────────────────
mkdir -p "${LUSTRE_BASE}/output" "${LUSTRE_BASE}/logs"
mkdir -p "${LUSTRE_BASE}/models-staging/diffusion_models" "${LUSTRE_BASE}/models-staging/text_encoders"
mkdir -p "${HOME_BASE}/models" "${HOME_BASE}/input"

# ─── Run inside the container via Pyxis ──────────────────────────────────
srun -l \
    --container-image="${SQSH_FILE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir=/workspace \
    --export=ALL,COMFYUI_PORT="${COMFYUI_PORT}",GRADIO_PORT="${GRADIO_PORT}",SLURM_JOB_ID \
    bash -c "${IN_CONTAINER_CMD}"

echo "── launch.sbatch finished at $(date) ───────────────────"