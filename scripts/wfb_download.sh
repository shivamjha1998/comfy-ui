#!/usr/bin/env bash
# Download the four WF-B models. Big files (umt5, 14B) land on /lustre because
# /home has a 61 GB user quota — symlinks from /home stitch them back in.
# Small files (vae, clip_vision) stay on /home directly.
set -uo pipefail

DEST=/home/user82001/comfyui/models                  # canonical (bind-mounted into container)
STAGING=/lustre/comfyui/models-staging               # big files live here, also bind-mounted
LOG=/home/user82001/wfb_download.log

mkdir -p "$DEST/diffusion_models" "$DEST/vae" "$DEST/clip_vision" "$DEST/text_encoders"
mkdir -p "$STAGING/diffusion_models" "$STAGING/text_encoders"

echo "=== START $(date -Is) ===" > "$LOG"

# dl <sub> <name> <url> <expected_bytes> <storage>
#   storage = "home" → write directly to $DEST/$sub/$name
#   storage = "lustre" → write to $STAGING/$sub/$name and create symlink at $DEST/$sub/$name
dl() {
    local sub="$1" name="$2" url="$3" expected="$4" storage="${5:-home}"
    local target_dir
    if [[ "$storage" == "lustre" ]]; then
        target_dir="$STAGING/$sub"
    else
        target_dir="$DEST/$sub"
    fi
    local out="$target_dir/$name"
    local tmp="$out.tmp"
    local link="$DEST/$sub/$name"

    # Already complete?
    if [[ -f "$out" ]]; then
        local sz
        sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [[ $sz -ge $expected ]]; then
            echo "[skip] $name already present ($sz bytes)" | tee -a "$LOG"
            # Make sure the symlink in /home points at it (idempotent).
            if [[ "$storage" == "lustre" && ! -L "$link" ]]; then
                ln -sfn "$out" "$link"
                echo "[link] $link -> $out" | tee -a "$LOG"
            fi
            return 0
        fi
        echo "[redo] $name truncated ($sz < $expected), re-downloading" | tee -a "$LOG"
        rm -f "$out"
    fi

    echo "[get ] $name -> $out (resumable, storage=$storage)" | tee -a "$LOG"
    if wget -c --tries=10 --waitretry=30 --timeout=120 \
            --retry-connrefused -q -O "$tmp" "$url"; then
        mv "$tmp" "$out"
        local sz
        sz=$(stat -c%s "$out" 2>/dev/null || echo 0)
        echo "[done] $name ($sz bytes) at $(date -Is)" | tee -a "$LOG"
        if [[ "$storage" == "lustre" ]]; then
            ln -sfn "$out" "$link"
            echo "[link] $link -> $out" | tee -a "$LOG"
        fi
    else
        local rc=$?
        echo "[FAIL] $name (wget rc=$rc) at $(date -Is)" | tee -a "$LOG"
    fi
}

dl vae           wan2.2_vae.safetensors    'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors'                                            1409400960  home
dl clip_vision   clip_vision_h.safetensors 'https://huggingface.co/AtelierDarren/Wan2.2_Animate/resolve/main/clip_vision_h.safetensors'                                                                  1264219396  home
dl text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors           'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors' 6735906897  lustre
dl diffusion_models Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors 'https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors' 17317143060 lustre

echo "=== END   $(date -Is) ===" | tee -a "$LOG"
df -h /home /lustre | tee -a "$LOG"
echo "Models on /home:"   | tee -a "$LOG"; du -sh "$DEST"    2>/dev/null | tee -a "$LOG"
echo "Models on /lustre:" | tee -a "$LOG"; du -sh "$STAGING" 2>/dev/null | tee -a "$LOG"
