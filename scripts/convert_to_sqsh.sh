#!/usr/bin/env bash
#
# convert_to_sqsh.sh — convert a local Docker image into a .sqsh file
# ─────────────────────────────────────────────────────────────────────
#
# Workaround for personal NGC accounts that lack Private Registry access.
# Runs a Linux helper container (with enroot installed) on the Mac that
# reads from the host's Docker daemon and produces a SquashFS .sqsh file
# we can scp to the SoftBank login server.
#
# Runs on : Mac mini M4
# Output  : ./build/comfyui-<tag>.sqsh
#
# Usage:
#   ./scripts/convert_to_sqsh.sh                     # use default tag
#   ./scripts/convert_to_sqsh.sh --tag dev-...       # specific tag
#   ./scripts/convert_to_sqsh.sh --image-id <id>     # by docker image ID
#   ./scripts/convert_to_sqsh.sh --rebuild-helper    # force rebuild of converter image

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly CONVERTER_DOCKERFILE="${SCRIPT_DIR}/dockerfiles/enroot-converter.Dockerfile"
readonly HELPER_IMAGE="comfyui-faceswap/enroot-converter:local"
readonly OUTPUT_DIR="${PROJECT_ROOT}/build"

# ─── Colors ──────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_GRN=$'\033[0;32m'; C_YLW=$'\033[0;33m'; C_BLU=$'\033[0;34m'
    C_RED=$'\033[0;31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
    C_GRN=''; C_YLW=''; C_BLU=''; C_RED=''; C_DIM=''; C_RST=''
fi

log()  { printf '%s[%s]%s %s\n' "${C_BLU}" "$(date +%H:%M:%S)" "${C_RST}" "$*"; }
ok()   { printf '%s ✓ %s%s\n' "${C_GRN}" "$*" "${C_RST}"; }
warn() { printf '%s ⚠ %s%s\n' "${C_YLW}" "$*" "${C_RST}" >&2; }
die()  { printf '%s ✗ %s%s\n' "${C_RED}" "$*" "${C_RST}" >&2; exit 1; }

# ─── Args ────────────────────────────────────────────────────────────────
TAG=""
IMAGE_ID=""
REBUILD_HELPER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)            TAG="${2:?--tag requires a value}"; shift 2 ;;
        --image-id)       IMAGE_ID="${2:?--image-id requires a value}"; shift 2 ;;
        --rebuild-helper) REBUILD_HELPER=true; shift ;;
        -h|--help)        sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                die "Unknown arg: $1 (try --help)" ;;
    esac
done

# ─── Pre-flight ──────────────────────────────────────────────────────────
log "Pre-flight checks…"
docker info >/dev/null 2>&1 || die "Docker Desktop is not running."
[[ -f "${CONVERTER_DOCKERFILE}" ]] || die "Helper Dockerfile not found at ${CONVERTER_DOCKERFILE}"
mkdir -p "${OUTPUT_DIR}"

# Auto-discover the most recent comfyui image if no tag/id given
if [[ -z "${TAG}" && -z "${IMAGE_ID}" ]]; then
    log "Auto-discovering most recent comfyui image…"
    TAG="$(docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep -E '/comfyui:dev-' | head -1 || true)"
    [[ -n "${TAG}" ]] || die "No comfyui image found locally. Build one first with scripts/build_and_push.sh --no-push"
fi

if [[ -n "${IMAGE_ID}" ]]; then
    SOURCE_REF="${IMAGE_ID}"
    OUTPUT_NAME="comfyui-$(echo "${IMAGE_ID}" | cut -c1-12).sqsh"
else
    SOURCE_REF="${TAG}"
    OUTPUT_NAME="comfyui-$(echo "${TAG}" | sed 's|.*:||' | tr '/' '-').sqsh"
fi

ok "Pre-flight passed (source=${SOURCE_REF})"

# ─── Build the helper image (cached after first run) ─────────────────────
if [[ "${REBUILD_HELPER}" == true ]] || ! docker image inspect "${HELPER_IMAGE}" >/dev/null 2>&1; then
    log "Building enroot helper image (first run only, ~3 min)…"
    docker build \
        --platform linux/amd64 \
        --file "${CONVERTER_DOCKERFILE}" \
        --tag "${HELPER_IMAGE}" \
        "${SCRIPT_DIR}/dockerfiles"
    ok "Helper image built"
else
    log "Reusing cached helper image ${C_DIM}${HELPER_IMAGE}${C_RST}"
fi

# ─── Convert ─────────────────────────────────────────────────────────────
log "Converting ${SOURCE_REF} → ${OUTPUT_NAME}"
log "${C_DIM}This will take 5–10 minutes for a 12 GB image. Be patient.${C_RST}"
echo

docker run --rm \
    --platform linux/amd64 \
    --privileged \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "${OUTPUT_DIR}:/output" \
    -w /output \
    "${HELPER_IMAGE}" \
    import \
        --output "${OUTPUT_NAME}" \
        "dockerd://${SOURCE_REF}"

# ─── Verify ──────────────────────────────────────────────────────────────
SQSH_PATH="${OUTPUT_DIR}/${OUTPUT_NAME}"
if [[ ! -f "${SQSH_PATH}" ]]; then
    die "Conversion appeared to succeed but no .sqsh file was produced."
fi

SQSH_SIZE="$(du -h "${SQSH_PATH}" | awk '{print $1}')"
ok "Conversion complete: ${SQSH_PATH} (${SQSH_SIZE})"

# ─── Print upload instructions ───────────────────────────────────────────
cat <<EOF

${C_GRN}┌─ Next: upload to SoftBank login server ─────────────────────${C_RST}
  Make sure your SSH tunnel is open in another terminal:
    ./scripts/connect.sh

  Then upload (this will take 15–60 min depending on your upload speed):
    scp -P 2222 \\
        ${SQSH_PATH} \\
        user82001@localhost:/lustre/comfyui/comfyui.sqsh

  Once uploaded, on the login server:
    df -h /lustre
    ls -lh /lustre/comfyui/comfyui.sqsh

  Then submit the job:
    sbatch scripts/launch.sbatch
${C_GRN}└─────────────────────────────────────────────────────────────${C_RST}
EOF