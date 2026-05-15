#!/usr/bin/env bash
# Start ComfyUI (port 8188) and the Gradio front-end (port 7860) together.
# Both stop when you press Ctrl+C.
#
# Flags:
#   --public              also start a Cloudflare quick tunnel and print a
#                         https://<random>.trycloudflare.com URL anyone with the
#                         link can reach. Requires `brew install cloudflared`.
#                         Strongly recommended to set GRADIO_AUTH_USER and
#                         GRADIO_AUTH_PASS in .env before using this.
#
# Optional env vars:
#   GRADIO_HOST=0.0.0.0   expose the UI on the LAN (default: 127.0.0.1)
#   COMFYUI_PORT=8188     change the ComfyUI port
#   GRADIO_PORT=7860      change the Gradio port
#   GRADIO_AUTH_USER      basic-auth username for the Gradio UI (recommended with --public)
#   GRADIO_AUTH_PASS      basic-auth password (matched with GRADIO_AUTH_USER)

set -euo pipefail

PUBLIC=false
for arg in "$@"; do
  case "$arg" in
    --public) PUBLIC=true ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/.logs"
COMFY_LOG="$LOG_DIR/comfyui.log"
GRADIO_LOG="$LOG_DIR/gradio.log"
COMFY_PORT="${COMFYUI_PORT:-8188}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

mkdir -p "$LOG_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "venv missing at $VENV_PY" >&2
  echo "  bootstrap: /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv \\" >&2
  echo "             && .venv/bin/pip install -r ComfyUI/requirements.txt -r gradio_app/requirements.txt" >&2
  exit 1
fi

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1; }

if port_busy "$COMFY_PORT"; then
  echo "port $COMFY_PORT already in use — ComfyUI may already be running." >&2
  echo "  stop it with:  kill \$(lsof -ti TCP:$COMFY_PORT)" >&2
  exit 1
fi
if port_busy "$GRADIO_PORT"; then
  echo "port $GRADIO_PORT already in use — Gradio may already be running." >&2
  echo "  stop it with:  kill \$(lsof -ti TCP:$GRADIO_PORT)" >&2
  exit 1
fi

COMFY_PID=""
GRADIO_PID=""
TAIL_PID=""
CF_PID=""

cleanup() {
  echo
  echo "[run.sh] stopping..."
  for pid in "$TAIL_PID" "$CF_PID" "$GRADIO_PID" "$COMFY_PID"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --public preflight: cloudflared installed + auth recommendation
if $PUBLIC; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "✗ --public needs cloudflared. Install with:  brew install cloudflared" >&2
    exit 1
  fi
  if [ -z "${GRADIO_AUTH_USER:-}" ] || [ -z "${GRADIO_AUTH_PASS:-}" ]; then
    echo "⚠ --public without GRADIO_AUTH_USER / GRADIO_AUTH_PASS — anyone with the URL can use the demo." >&2
    echo "   Strongly recommended: add them to .env before sharing the link." >&2
  fi
fi

# Apply Mac-only patches to ComfyUI-ReActor (no-op if already patched, or if
# the custom_node isn't installed yet).
if [ "$(uname)" = "Darwin" ]; then
  bash "$REPO_DIR/scripts/patch_reactor_mac.sh"
fi

echo "[run.sh] starting ComfyUI on :$COMFY_PORT  -> $COMFY_LOG"
( cd "$REPO_DIR/ComfyUI" && exec "$VENV_PY" main.py --listen 127.0.0.1 --port "$COMFY_PORT" ) >"$COMFY_LOG" 2>&1 &
COMFY_PID=$!

echo -n "[run.sh] waiting for ComfyUI"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
    echo " ok"
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo
    echo "ComfyUI exited before becoming ready. Last log lines:" >&2
    tail -30 "$COMFY_LOG" >&2
    exit 1
  fi
  echo -n "."
  sleep 1
done

if ! curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
  echo
  echo "ComfyUI did not respond within 120s" >&2
  tail -30 "$COMFY_LOG" >&2
  exit 1
fi

echo "[run.sh] starting Gradio on :$GRADIO_PORT  -> $GRADIO_LOG"
( cd "$REPO_DIR" && exec "$VENV_PY" -m gradio_app ) >"$GRADIO_LOG" 2>&1 &
GRADIO_PID=$!

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$GRADIO_PORT/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
    echo "Gradio exited before becoming ready" >&2
    tail -30 "$GRADIO_LOG" >&2
    exit 1
  fi
  sleep 1
done

HOST_DISPLAY="${GRADIO_HOST:-127.0.0.1}"
PUBLIC_URL=""

if $PUBLIC; then
  CF_LOG="$LOG_DIR/cloudflared.log"
  echo "[run.sh] starting Cloudflare quick tunnel  -> $CF_LOG"
  cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$GRADIO_PORT" >"$CF_LOG" 2>&1 &
  CF_PID=$!
  # cloudflared prints "Your quick Tunnel has been created! ... https://<rand>.trycloudflare.com"
  # within a few seconds of starting. Poll the log.
  for _ in $(seq 1 30); do
    PUBLIC_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 || true)
    [ -n "$PUBLIC_URL" ] && break
    if ! kill -0 "$CF_PID" 2>/dev/null; then
      echo "cloudflared exited before printing a URL. Last log lines:" >&2
      tail -20 "$CF_LOG" >&2
      exit 1
    fi
    sleep 1
  done
fi

cat <<EOF

----------------------------------------------------
  Face-Swap demo is ready.

  Browser:    http://$HOST_DISPLAY:$GRADIO_PORT
  ComfyUI:    http://127.0.0.1:$COMFY_PORT
  Logs:       $LOG_DIR/
EOF

if [ -n "$PUBLIC_URL" ]; then
  cat <<EOF

  🌍 Public URL: $PUBLIC_URL
EOF
  if [ -n "${GRADIO_AUTH_USER:-}" ]; then
    echo "     (basic auth required: user=$GRADIO_AUTH_USER)"
  else
    echo "     ⚠ no basic auth set — anyone with this URL can use the demo"
  fi
fi

cat <<EOF

  Press Ctrl+C to stop everything.
----------------------------------------------------

EOF

tail -f "$COMFY_LOG" "$GRADIO_LOG" &
TAIL_PID=$!

# Block until any tracked child exits; trap will tear the others down.
while kill -0 "$COMFY_PID" 2>/dev/null && kill -0 "$GRADIO_PID" 2>/dev/null; do
  if [ -n "$CF_PID" ] && ! kill -0 "$CF_PID" 2>/dev/null; then
    echo
    echo "[run.sh] cloudflared exited; shutting down"
    break
  fi
  sleep 2
done

echo
echo "[run.sh] a service exited unexpectedly; shutting down the other"
exit 1
