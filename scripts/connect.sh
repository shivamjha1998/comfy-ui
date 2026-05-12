#!/usr/bin/env bash
#
# connect.sh — open the SSH tunnel(s) needed to reach the SoftBank GPU
# ─────────────────────────────────────────────────────────────────────────
#
# Runs on : Mac mini M4
# What it does:
#   • Opens the access-server tunnel: localhost:2222 → login server :22
#   • If --gpu-node is given, spawns a second SSH session that routes
#     Gradio (7860) and ComfyUI (8188) through the login server to the GPU node.
#     (The access server cannot reach compute nodes directly — the login
#      server is the required intermediate hop.)
#
# Usage:
#   ./scripts/connect.sh                           # tunnel only
#   ./scripts/connect.sh --gpu-node dgxa100-001    # tunnel + port forwards
#   ./scripts/connect.sh --secondary               # use access server #2
#   ./scripts/connect.sh --help

set -euo pipefail

# ─── Locate project root and load .env ───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ─── Defaults (can be overridden in .env) ────────────────────────────────
SOFTBANK_USER="${SOFTBANK_USER:-user82001}"
SOFTBANK_ACCESS_HOST="${SOFTBANK_ACCESS_HOST:-221.111.218.60}"
SOFTBANK_ACCESS_HOST_2="${SOFTBANK_ACCESS_HOST_2:-221.111.218.61}"
SOFTBANK_ACCESS_PORT="${SOFTBANK_ACCESS_PORT:-50001}"
SOFTBANK_LOGIN_HOST="${SOFTBANK_LOGIN_HOST:-10.120.58.68}"
SOFTBANK_LOGIN_HOST_2="${SOFTBANK_LOGIN_HOST_2:-10.120.58.69}"
SOFTBANK_LOGIN_PORT="${SOFTBANK_LOGIN_PORT:-22}"
SOFTBANK_LOCAL_TUNNEL_PORT="${SOFTBANK_LOCAL_TUNNEL_PORT:-2222}"

GRADIO_PORT="${GRADIO_PORT:-7860}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
QWEN_PORT="${QWEN_PORT:-8000}"
KEY_FILE="${KEY_FILE:-${HOME}/.ssh/id_rsa}"

# ─── Colors ──────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_GRN=$'\033[0;32m'; C_YLW=$'\033[0;33m'; C_BLU=$'\033[0;34m'; C_RST=$'\033[0m'
else
    C_GRN=''; C_YLW=''; C_BLU=''; C_RST=''
fi

# ─── Args ────────────────────────────────────────────────────────────────
USE_SECONDARY=false
GPU_NODE="${GPU_NODE_HOST:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-node)   GPU_NODE="${2:?--gpu-node requires hostname}"; shift 2 ;;
        --secondary)  USE_SECONDARY=true; shift ;;
        -h|--help)    sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "${USE_SECONDARY}" == true ]]; then
    ACCESS_HOST="${SOFTBANK_ACCESS_HOST_2}"
    LOGIN_HOST="${SOFTBANK_LOGIN_HOST_2}"
else
    ACCESS_HOST="${SOFTBANK_ACCESS_HOST}"
    LOGIN_HOST="${SOFTBANK_LOGIN_HOST}"
fi

[[ -f "${KEY_FILE}" ]] || { echo "SSH key not found at ${KEY_FILE}" >&2; exit 1; }

# ─── Tunnel 1: access server → login server ──────────────────────────────
cat <<EOF
${C_BLU}┌─ Opening tunnel ────────────────────────────────────────────${C_RST}
  Local port    ${SOFTBANK_LOCAL_TUNNEL_PORT}
  → forwards to ${LOGIN_HOST}:${SOFTBANK_LOGIN_PORT}
  via           ${SOFTBANK_USER}@${ACCESS_HOST}:${SOFTBANK_ACCESS_PORT}
EOF

if [[ -n "${GPU_NODE}" ]]; then
    cat <<EOF
  ${C_GRN}+ GPU port forwards${C_RST}
  Local ${GRADIO_PORT}    → ${GPU_NODE}:${GRADIO_PORT}    (Gradio UI)
  Local ${COMFYUI_PORT}    → ${GPU_NODE}:${COMFYUI_PORT}    (ComfyUI API)
  Local ${QWEN_PORT}    → ${GPU_NODE}:${QWEN_PORT}    (Qwen API)
EOF
fi
echo "${C_BLU}└─────────────────────────────────────────────────────────────${C_RST}"
echo
echo "${C_YLW}Keep this terminal open. Ctrl-C to disconnect.${C_RST}"
echo

# ─── GPU port forwards (background, through login server) ────────────────────
# Routes: Mac:PORT → Access Server → Login Server → GPU Node:PORT
# ProxyJump (-J) lands the SSH session on the login server, which can reach
# the compute node on the internal network. The access server cannot.
GPU_FWD_PID=""
if [[ -n "${GPU_NODE}" ]]; then
    ssh -N \
        -J "${SOFTBANK_USER}@${ACCESS_HOST}:${SOFTBANK_ACCESS_PORT}" \
        -i "${KEY_FILE}" \
        -o IdentitiesOnly=yes \
        -o ServerAliveInterval=30 \
        -o ExitOnForwardFailure=yes \
        -L "${GRADIO_PORT}:${GPU_NODE}:${GRADIO_PORT}" \
        -L "${COMFYUI_PORT}:${GPU_NODE}:${COMFYUI_PORT}" \
        -L "${QWEN_PORT}:${GPU_NODE}:${QWEN_PORT}" \
        "${SOFTBANK_USER}@${LOGIN_HOST}" &
    GPU_FWD_PID=$!
    trap 'kill "${GPU_FWD_PID}" 2>/dev/null; wait "${GPU_FWD_PID}" 2>/dev/null' EXIT INT TERM
fi

# ─── Main tunnel (access server → login server, foreground) ──────────────────
ssh -N \
    -i "${KEY_FILE}" \
    -o IdentitiesOnly=yes \
    -o ServerAliveInterval=30 \
    -o ExitOnForwardFailure=yes \
    -L "${SOFTBANK_LOCAL_TUNNEL_PORT}:${LOGIN_HOST}:${SOFTBANK_LOGIN_PORT}" \
    "${SOFTBANK_USER}@${ACCESS_HOST}" \
    -p "${SOFTBANK_ACCESS_PORT}"