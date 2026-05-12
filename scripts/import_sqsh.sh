#!/usr/bin/env bash
# Re-import the comfyui container .sqsh from NGC into /lustre/comfyui/.
# Uses persisted creds at ~/.config/enroot/.credentials.
set -uo pipefail

LUSTRE_BASE=/lustre/comfyui
NGC_ORG="${NGC_ORG:-z4pymjpdqjsj}"
NGC_TEAM="${NGC_TEAM:-abc1}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE="nvcr.io/${NGC_ORG}/${NGC_TEAM}/comfyui:${IMAGE_TAG}"
OUT="${LUSTRE_BASE}/comfyui-${IMAGE_TAG}.sqsh"
LOG=/home/user82001/sqsh_import.log

echo "=== START $(date -Is) ===" >"$LOG"
echo "image: $IMAGE" >>"$LOG"
echo "out:   $OUT" >>"$LOG"

if test -f "$OUT"; then
    echo "[skip] $OUT already exists ($(stat -c%s "$OUT") bytes)" | tee -a "$LOG"
    exit 0
fi

cd "$LUSTRE_BASE"
echo "[run ] enroot import -> $OUT" | tee -a "$LOG"
if enroot import --output "comfyui-${IMAGE_TAG}.sqsh" "docker://${IMAGE}" >>"$LOG" 2>&1; then
    echo "[done] $(stat -c%s "$OUT") bytes at $(date -Is)" | tee -a "$LOG"
    df -h /lustre | tee -a "$LOG"
else
    rc=$?
    echo "[FAIL] enroot import rc=$rc" | tee -a "$LOG"
    tail -40 "$LOG"
    exit $rc
fi
