#!/usr/bin/env bash
#
# login_server_setup.sh — one-time setup on the SoftBank login server
# ─────────────────────────────────────────────────────────────────────
#
# Runs on : SoftBank AI Data Center login server (fcpv00314 / fcpv00315)
# Run as  : user82001 (or user82002)
# When    : Once, after the container has been pushed to NGC and security-scanned.
#
# What it does:
#   1. Creates the /lustre/comfyui project layout
#   2. Configures enroot credentials so we can pull from NGC Private Registry
#   3. Imports the container image as a .sqsh file via enroot
#   4. Downloads all model files into /lustre/comfyui/models
#   5. Stages the workflow JSON templates
#   6. Verifies everything looks right
#
# Usage:
#   ./login_server_setup.sh                     # full setup
#   ./login_server_setup.sh --skip-models       # skip 35GB model downloads
#   ./login_server_setup.sh --models-only       # re-download / fix truncated models only (skips container import)
#   ./login_server_setup.sh --image-tag <tag>   # pull a specific image tag
#   ./login_server_setup.sh --help

set -euo pipefail

# ─── Constants ───────────────────────────────────────────────────────────
# /lustre is 100 GB total — models live on /home (1 PB) so they don't fight
# Qwen for space. Container .sqsh and runtime output/logs stay on /lustre.
readonly LUSTRE_BASE="/lustre/comfyui"
readonly HOME_BASE="${HOME}/comfyui"
readonly REGISTRY="nvcr.io"
readonly TEAM="abc1"
readonly IMAGE_NAME="comfyui"
readonly ENROOT_CRED_FILE="${HOME}/.config/enroot/.credentials"

# ─── Colors ──────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_RED=$'\033[0;31m'; C_GRN=$'\033[0;32m'; C_YLW=$'\033[0;33m'
    C_BLU=$'\033[0;34m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
    C_RED=''; C_GRN=''; C_YLW=''; C_BLU=''; C_DIM=''; C_RST=''
fi

log()  { printf '%s[%s]%s %s\n' "${C_BLU}" "$(date +%H:%M:%S)" "${C_RST}" "$*"; }
ok()   { printf '%s ✓ %s%s\n' "${C_GRN}" "$*" "${C_RST}"; }
warn() { printf '%s ⚠ %s%s\n' "${C_YLW}" "$*" "${C_RST}" >&2; }
die()  { printf '%s ✗ %s%s\n' "${C_RED}" "$*" "${C_RST}" >&2; exit 1; }

# ─── Args ────────────────────────────────────────────────────────────────
SKIP_MODELS=false
MODELS_ONLY=false
IMAGE_TAG="latest"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-models)  SKIP_MODELS=true; shift ;;
        --models-only)  MODELS_ONLY=true; shift ;;   # skip container import, only (re-)download models
        --image-tag)    IMAGE_TAG="${2:?--image-tag requires value}"; shift 2 ;;
        -h|--help)      sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              die "Unknown arg: $1" ;;
    esac
done

# ─── Pre-flight ──────────────────────────────────────────────────────────
log "Pre-flight checks…"

[[ -d /lustre ]] || die "/lustre not mounted. Are you on the SoftBank login server?"

if [[ "${MODELS_ONLY}" == false ]]; then
    command -v enroot >/dev/null 2>&1 || die "enroot not in PATH. Source /etc/profile or contact SoftBank support."
    : "${NGC_ORG:?Set NGC_ORG before running (export NGC_ORG=...).}"
    : "${NGC_API_KEY:?Set NGC_API_KEY before running (export NGC_API_KEY=...).}"
fi

readonly IMAGE="${REGISTRY}/${NGC_ORG:-unset}/${TEAM}/${IMAGE_NAME}:${IMAGE_TAG}"
readonly SQSH_FILE="${LUSTRE_BASE}/comfyui-${IMAGE_TAG}.sqsh"

ok "Pre-flight passed"

# ─── 1. Directory layout ─────────────────────────────────────────────────
log "Creating layout (models on ${HOME_BASE}, runtime on ${LUSTRE_BASE})…"
mkdir -p \
    "${HOME_BASE}/models/insightface" \
    "${HOME_BASE}/models/facerestore_models" \
    "${HOME_BASE}/models/diffusion_models" \
    "${HOME_BASE}/models/vae" \
    "${HOME_BASE}/models/clip_vision" \
    "${HOME_BASE}/models/text_encoders" \
    "${HOME_BASE}/models/loras" \
    "${HOME_BASE}/models/sam2" \
    "${HOME_BASE}/models/yolo" \
    "${HOME_BASE}/models/detection" \
    "${HOME_BASE}/models/upscale_models" \
    "${HOME_BASE}/models/rife" \
    "${HOME_BASE}/input" \
    "${HOME_BASE}/workflows" \
    "${LUSTRE_BASE}/output" \
    "${LUSTRE_BASE}/logs"
ok "Directories created"

# ─── 2. enroot credentials ───────────────────────────────────────────────
if [[ "${MODELS_ONLY}" == false ]]; then
    log "Configuring enroot credentials at ${ENROOT_CRED_FILE}…"
    mkdir -p "$(dirname "${ENROOT_CRED_FILE}")"
    chmod 700 "$(dirname "${ENROOT_CRED_FILE}")"

    # Idempotent write — overwrite cleanly so re-runs don't append duplicates.
    cat > "${ENROOT_CRED_FILE}" <<EOF
machine ${REGISTRY} login \$oauthtoken password ${NGC_API_KEY}
machine authn.nvidia.com login \$oauthtoken password ${NGC_API_KEY}
EOF
    chmod 600 "${ENROOT_CRED_FILE}"
    ok "enroot credentials written"
else
    log "Skipping enroot credentials (--models-only)"
fi

# ─── 3. enroot import ────────────────────────────────────────────────────
if [[ "${MODELS_ONLY}" == false ]]; then
    if [[ -f "${SQSH_FILE}" ]]; then
        warn "${SQSH_FILE} already exists. Delete it and re-run if you want a fresh pull."
    else
        log "Importing container from NGC (this can take 10–20 min for large images)…"
        cd "${LUSTRE_BASE}"
        enroot import --output "comfyui-${IMAGE_TAG}.sqsh" "docker://${IMAGE}"
        ok "Container imported as ${SQSH_FILE}"
    fi
else
    log "Skipping container import (--models-only)"
fi

# ─── 4. Model downloads ──────────────────────────────────────────────────
if [[ "${SKIP_MODELS}" == true ]]; then
    warn "Skipping model downloads (--skip-models). You'll need to run download_models.sh later."
else
    log "Downloading models (~35 GB total)…"
    warn "Model URLs are PLACEHOLDERS — verify each source before running. Many of these"
    warn "models change distribution channels frequently. Edit this section before use."

    # ── Disk space pre-check ─────────────────────────────────────────────────
    # WF-A models: ~1 GB.  WF-B models: ~36 GB.  Total: ~37 GB on /home.
    # /home has 1 PB so this is virtually always fine — kept as a smoke check.
    AVAIL_KB=$(df -k "${HOME_BASE}" | awk 'NR==2 {print $4}')
    AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
    if [[ "${AVAIL_GB}" -lt 50 ]]; then
        warn "Only ${AVAIL_GB} GB free on ${HOME_BASE} — need at least 50 GB."
        warn "Continuing anyway; incomplete downloads will be re-attempted on re-run."
    else
        log "Disk space OK: ${AVAIL_GB} GB free on ${HOME_BASE}"
    fi

    # ── download_model <dest_dir> <filename> <url> <min_bytes> ──────────────
    # • min_bytes: minimum acceptable file size. Files smaller than this are
    #   considered truncated and will be deleted and re-downloaded.
    #   Pass 0 (or omit) to skip the size check (not recommended).
    # • Uses an atomic .tmp write: a failed/interrupted download leaves a .tmp
    #   file rather than a silently-corrupt target.
    download_model() {
        local dest="$1" name="$2" url="$3" min_bytes="${4:-0}"
        local out="${HOME_BASE}/models/${dest}/${name}"
        local tmp="${out}.tmp"
        local size
        size=$(stat -c%s "${out}" 2>/dev/null || echo 0)

        # Already present and big enough — skip.
        if [[ -f "${out}" ]] && { [[ "${min_bytes}" -eq 0 ]] || [[ "${size}" -ge "${min_bytes}" ]]; }; then
            log "  · ${name} OK ($(numfmt --to=iec-i --suffix=B --format='%.1f' "${size}"))"
            return
        fi

        # Present but too small — truncated download from a previous run.
        if [[ -f "${out}" && "${min_bytes}" -gt 0 && "${size}" -lt "${min_bytes}" ]]; then
            warn "  · ${name} truncated ($(numfmt --to=iec-i --suffix=B --format='%.1f' "${size}") < $(numfmt --to=iec-i --suffix=B --format='%.1f' "${min_bytes}") min) — re-downloading"
            rm -f "${out}"
        fi

        # Clean up any leftover .tmp from a prior interrupted run.
        rm -f "${tmp}"

        log "  · downloading ${name}…"
        if wget --quiet --show-progress -O "${tmp}" "${url}"; then
            mv "${tmp}" "${out}"
            local final_size
            final_size=$(stat -c%s "${out}" 2>/dev/null || echo 0)
            ok "  · ${name} saved ($(numfmt --to=iec-i --suffix=B --format='%.1f' "${final_size}"))"
        else
            rm -f "${tmp}"
            warn "  · FAILED to download ${name} from ${url}"
        fi
    }

    # ── ReActor / face swap ─────────────────────────────────────────────────
    download_model insightface        inswapper_128.onnx      "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"    480000000
    download_model insightface        retinaface_resnet50.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"  100000000
    download_model facerestore_models GFPGANv1.4.pth          "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"               330000000

    # ── WAN 2.2 ─────────────────────────────────────────────────────────────
    # Main model: fp8-quantised single-file build by kijai (~17.3 GB).
    # Placed in diffusion_models/ — the standard ComfyUI path for diffusion
    # transformers that kijai's WanVideoWrapper expects.
    download_model diffusion_models \
        "Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors" \
        "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors" \
        16000000000

    # VAE (Comfy-Org repackaged, ~1.4 GB)
    download_model vae \
        "wan2.2_vae.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors" \
        1200000000

    # CLIP Vision encoder (~1.3 GB)
    download_model clip_vision \
        "clip_vision_h.safetensors" \
        "https://huggingface.co/AtelierDarren/Wan2.2_Animate/resolve/main/clip_vision_h.safetensors" \
        1200000000

    # UMT5-XXL text encoder — fp8 scaled safetensors (~6.7 GB).
    # CLIPLoader (type "wan") searches models/text_encoders/.
    # Filename matches the one referenced in workflows/wf_b_wan22.json (node 15).
    download_model text_encoders \
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
        "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
        6500000000

    # Relight LoRA — fp16 (~1.4 GB)
    download_model loras \
        "WanAnimate_relight_lora_fp16.safetensors" \
        "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors" \
        1200000000

# ── Pose / segmentation / interpolation ─────────────────────────────────
    #
    # Detection models go in models/detection/ — that is the folder path
    # that ComfyUI-WanAnimatePreprocess's OnnxDetectionModelLoader indexes.
    #
    # yolov10m.onnx: YOLOv10-M person detector — ONNX export (~25 MB).
    # OnnxDetectionModelLoader uses onnxruntime — must be .onnx, not .pt.
    download_model detection  yolov10m.onnx         "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.onnx"                                                         22000000
    #
    # vitpose-l-wholebody.onnx: ViTPose-L wholebody keypoint estimator (ONNX, ~270 MB).
    # Workflow uses the large wholebody variant — not vitpose_base.onnx.
    # ⚠️  VERIFY URL before first run — kijai may update the repo path.
    download_model detection  vitpose-l-wholebody.onnx "https://huggingface.co/Kijai/WanAnimatePreprocess_models/resolve/main/vitpose-l-wholebody.onnx"                               250000000
    #
    # SAM2 — kijai's safetensors conversion (the DownloadAndLoadSAM2Model node
    # enumerates *.safetensors in models/sam2/, not the original Facebook .pt).
    # ⚠️  VERIFY URL before first run.
    download_model sam2       sam2_hiera_large.safetensors "https://huggingface.co/Kijai/sam2-safetensors/resolve/main/sam2_hiera_large.safetensors"               500000000
    download_model rife       rife47.pth                   "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/rife47.pth"          90000000

    ok "Model download phase complete (check warnings above for any failures)"
fi

# ─── 5. Stage workflow templates ─────────────────────────────────────────
log "Staging workflow templates…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -d "${PROJECT_ROOT}/workflows" ]]; then
    cp -v "${PROJECT_ROOT}/workflows/"*.json "${HOME_BASE}/workflows/"
    ok "Workflows staged"
else
    warn "No workflows/ directory found alongside script. Stage them manually:"
    warn "    scp workflows/*.json user82001@localhost:${HOME_BASE}/workflows/"
fi

# ─── 6. Verification ─────────────────────────────────────────────────────
log "Final verification…"
echo
echo "  ${LUSTRE_BASE}:"
ls -lh "${LUSTRE_BASE}/" | sed 's/^/    /'
echo
echo "  ${HOME_BASE}:"
ls -lh "${HOME_BASE}/" | sed 's/^/    /'
echo
echo "  Models:"
du -sh "${HOME_BASE}/models/"* 2>/dev/null | sed 's/^/    /' || echo "    (empty)"
echo
echo "  Container:"
ls -lh "${SQSH_FILE}" 2>/dev/null | sed 's/^/    /' || warn "  ${SQSH_FILE} not present"
echo

cat <<EOF
${C_GRN}┌─ Setup complete ────────────────────────────────────────────${C_RST}
  Container: ${SQSH_FILE}
  Models:    ${HOME_BASE}/models/
  Workflows: ${HOME_BASE}/workflows/
  Output:    ${LUSTRE_BASE}/output/

  Next: submit your first job:
    sbatch ${PROJECT_ROOT:-~/comfyui-faceswap}/scripts/launch.sbatch
${C_GRN}└─────────────────────────────────────────────────────────────${C_RST}
EOF