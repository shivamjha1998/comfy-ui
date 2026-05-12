#!/usr/bin/env bash
#
# build_and_push.sh — build the ComfyUI face-swap container and push to NGC
# ─────────────────────────────────────────────────────────────────────────
#
# Runs on : Mac mini M4 (Apple Silicon, arm64)
# Targets : linux/amd64 (cross-compile via docker buildx)
# Pushes  : nvcr.io/${NGC_ORG}/abc1/comfyui:<tag>
# Then    : the image must be security-scanned on the NGC web UI before
#           it can be pulled onto the SoftBank login server with `enroot import`.
#
# ─── Required environment ────────────────────────────────────────────────
# Either export these in your shell, or create a .env file beside this
# script (loaded automatically). NEVER commit .env to git.
#
#   NGC_ORG       NGC organization slug (visible in nvcr.io URLs).
#                 The placeholder in the spec was 'z4pymjpdqjsj' — replace
#                 with the real value from https://ngc.nvidia.com after
#                 logging in to the abc1 team.
#   NGC_API_KEY   Personal API key generated at NGC → Setup → Generate API Key.
#                 Treated as a password — never log, never echo.
#
# ─── Usage ───────────────────────────────────────────────────────────────
#   ./scripts/build_and_push.sh                  # build + push :dev-<ts>-<sha> and :latest
#   ./scripts/build_and_push.sh --no-push        # build only, skip push
#   ./scripts/build_and_push.sh --tag v0.1.0     # build + push :v0.1.0 (release)
#   ./scripts/build_and_push.sh --comfyui-ref <sha>  # pin ComfyUI to a commit
#   ./scripts/build_and_push.sh --help

set -euo pipefail

# ─── Constants ───────────────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly DOCKERFILE="${PROJECT_ROOT}/Dockerfile"
readonly REGISTRY="nvcr.io"
readonly IMAGE_NAME="comfyui"
# NGC_TEAM is read from .env later (cannot be declared readonly here because
# .env is sourced AFTER the constants block). Empty string = personal account
# (path: nvcr.io/<org>/<image>). "abc1" or similar = team account
# (path: nvcr.io/<org>/<team>/<image>).
readonly BUILDER_NAME="comfyui-faceswap-builder"
readonly BUILD_LOG="${PROJECT_ROOT}/.build-logs/build-$(date +%Y%m%d-%H%M%S).log"
readonly TARGET_PLATFORM="linux/amd64"

# ─── Colors (suppressed when stdout is not a TTY) ────────────────────────
if [[ -t 1 ]]; then
    readonly C_RED=$'\033[0;31m'
    readonly C_GRN=$'\033[0;32m'
    readonly C_YLW=$'\033[0;33m'
    readonly C_BLU=$'\033[0;34m'
    readonly C_DIM=$'\033[2m'
    readonly C_RST=$'\033[0m'
else
    readonly C_RED='' C_GRN='' C_YLW='' C_BLU='' C_DIM='' C_RST=''
fi

log()  { printf '%s[%s]%s %s\n' "${C_BLU}" "$(date +%H:%M:%S)" "${C_RST}" "$*"; }
ok()   { printf '%s ✓ %s%s\n' "${C_GRN}" "$*" "${C_RST}"; }
warn() { printf '%s ⚠ %s%s\n' "${C_YLW}" "$*" "${C_RST}" >&2; }
err()  { printf '%s ✗ %s%s\n' "${C_RED}" "$*" "${C_RST}" >&2; }
die()  { err "$*"; exit 1; }

# ─── Argument parsing ────────────────────────────────────────────────────
DO_PUSH=true
EXPLICIT_TAG=""
COMFYUI_REF="master"

usage() {
    sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push)        DO_PUSH=false; shift ;;
        --tag)            EXPLICIT_TAG="${2:?--tag requires a value}"; shift 2 ;;
        --comfyui-ref)    COMFYUI_REF="${2:?--comfyui-ref requires a value}"; shift 2 ;;
        -h|--help)        usage ;;
        *)                die "Unknown argument: $1 (try --help)" ;;
    esac
done

# ─── Load .env if present ────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log "Loading ${C_DIM}${PROJECT_ROOT}/.env${C_RST}"
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────
log "Running pre-flight checks…"

[[ -f "${DOCKERFILE}" ]] || die "Dockerfile not found at ${DOCKERFILE}"

command -v docker >/dev/null 2>&1 \
    || die "docker not installed. Install Docker Desktop for Mac."

docker info >/dev/null 2>&1 \
    || die "docker daemon not running. Start Docker Desktop and retry."

docker buildx version >/dev/null 2>&1 \
    || die "docker buildx not available. Update Docker Desktop."

: "${NGC_ORG:?NGC_ORG not set. Export it or add to .env (see header).}"
if [[ "${DO_PUSH}" == true ]]; then
    : "${NGC_API_KEY:?NGC_API_KEY not set. Export it or add to .env (see header).}"
fi

# Detect host architecture (informational)
HOST_ARCH="$(uname -m)"
if [[ "${HOST_ARCH}" != "arm64" && "${HOST_ARCH}" != "x86_64" ]]; then
    warn "Unexpected host arch: ${HOST_ARCH}"
fi

ok "Pre-flight passed (host=${HOST_ARCH}, target=${TARGET_PLATFORM})"

# ─── Compute tags ────────────────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
GIT_SHA="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"

if [[ -n "${EXPLICIT_TAG}" ]]; then
    PRIMARY_TAG="${EXPLICIT_TAG}"
else
    PRIMARY_TAG="dev-${TIMESTAMP}-${GIT_SHA}"
fi

NGC_TEAM="${NGC_TEAM:-}"   # default to empty if .env didn't set it
if [[ -n "${NGC_TEAM}" ]]; then
    readonly IMAGE_BASE="${REGISTRY}/${NGC_ORG}/${NGC_TEAM}/${IMAGE_NAME}"
else
    readonly IMAGE_BASE="${REGISTRY}/${NGC_ORG}/${IMAGE_NAME}"
fi
readonly IMAGE_PRIMARY="${IMAGE_BASE}:${PRIMARY_TAG}"
readonly IMAGE_LATEST="${IMAGE_BASE}:latest"

# ─── Plan summary ────────────────────────────────────────────────────────
cat <<EOF

${C_BLU}┌─ Build plan ─────────────────────────────────────────────────${C_RST}
  Dockerfile      ${DOCKERFILE}
  Platform        ${TARGET_PLATFORM}
  ComfyUI ref     ${COMFYUI_REF}
  Primary tag     ${C_GRN}${IMAGE_PRIMARY}${C_RST}
  Also tag        ${C_GRN}${IMAGE_LATEST}${C_RST}
  Push to NGC     $([[ "${DO_PUSH}" == true ]] && echo "${C_GRN}YES${C_RST}" || echo "${C_YLW}NO (--no-push)${C_RST}")
  Build log       ${BUILD_LOG}
${C_BLU}└──────────────────────────────────────────────────────────────${C_RST}

EOF

# Confirm if pushing — pushes are visible to others on the NGC org
if [[ "${DO_PUSH}" == true ]]; then
    read -r -p "Push to ${REGISTRY}? [y/N] " confirm
    [[ "${confirm}" =~ ^[Yy]$ ]] || die "Aborted by user."
fi

# ─── Docker login (only when pushing) ────────────────────────────────────
if [[ "${DO_PUSH}" == true ]]; then
    log "Logging in to ${REGISTRY}…"
    # NGC uses literal '$oauthtoken' as the username and the API key as the password.
    # --password-stdin keeps the key out of `ps` and shell history.
    printf '%s' "${NGC_API_KEY}" | docker login "${REGISTRY}" \
        --username '$oauthtoken' \
        --password-stdin >/dev/null
    ok "Logged in to ${REGISTRY}"
fi

# ─── Buildx builder ──────────────────────────────────────────────────────
if ! docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1; then
    log "Creating buildx builder ${C_DIM}${BUILDER_NAME}${C_RST}"
    docker buildx create \
        --name "${BUILDER_NAME}" \
        --driver docker-container \
        --bootstrap >/dev/null
    ok "Builder ${BUILDER_NAME} created"
else
    log "Reusing existing buildx builder ${C_DIM}${BUILDER_NAME}${C_RST}"
fi

# ─── Build (and optionally push) ─────────────────────────────────────────
mkdir -p "$(dirname "${BUILD_LOG}")"

BUILDX_ARGS=(
    buildx build
    --builder "${BUILDER_NAME}"
    --platform "${TARGET_PLATFORM}"
    --file "${DOCKERFILE}"
    --build-arg "COMFYUI_REF=${COMFYUI_REF}"
    --tag "${IMAGE_PRIMARY}"
    --tag "${IMAGE_LATEST}"
    --progress plain
)

if [[ "${DO_PUSH}" == true ]]; then
    BUILDX_ARGS+=( --push )
else
    # --load only works for single-platform builds (which is fine for us;
    # linux/amd64 images CAN be loaded into Docker Desktop on Apple Silicon
    # via Rosetta emulation — slow but useful for inspection).
    BUILDX_ARGS+=( --load )
fi

BUILDX_ARGS+=( "${PROJECT_ROOT}" )

log "Starting build (this can take 15–30 min on first run; subsequent builds use cache)"
log "Streaming output to ${C_DIM}${BUILD_LOG}${C_RST}"
echo

# Tee the output so the user can watch progress live AND keep a log file
if docker "${BUILDX_ARGS[@]}" 2>&1 | tee "${BUILD_LOG}"; then
    echo
    ok "Build succeeded"
else
    echo
    die "Build failed. Full log: ${BUILD_LOG}"
fi

# ─── Post-build instructions ─────────────────────────────────────────────
if [[ "${DO_PUSH}" == true ]]; then
    cat <<EOF

${C_GRN}┌─ Pushed ────────────────────────────────────────────────────${C_RST}
  ${IMAGE_PRIMARY}
  ${IMAGE_LATEST}
${C_GRN}└─────────────────────────────────────────────────────────────${C_RST}

${C_YLW}Next steps (manual, on the NGC web UI):${C_RST}
  1. Open https://ngc.nvidia.com/ and switch to the ${NGC_TEAM} team
  2. Navigate to: Private Registry → Containers → ${IMAGE_NAME}
     (under org "${NGC_ORG}"$([[ -n "${NGC_TEAM}" ]] && echo " / team \"${NGC_TEAM}\""))
  3. Click the new tag → "Security Scanning" tab → Begin Scan
  4. Wait for the completion email (typically a few minutes)
  5. Confirm the scan grade is ${C_GRN}B or above${C_RST} — anything lower
     will be rejected by enroot import on the login server
  6. Once scanned, on the SoftBank login server (fcpv00314):
       cd /lustre/comfyui
       enroot import "docker://${IMAGE_PRIMARY}"
EOF
else
    cat <<EOF

${C_YLW}Build complete (not pushed).${C_RST}
Inspect locally with:
  docker run --rm -it ${IMAGE_PRIMARY} bash
EOF
fi