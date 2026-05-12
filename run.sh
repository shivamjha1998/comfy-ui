#!/usr/bin/env bash
# Start ComfyUI (port 8188) and the Gradio front-end (port 7860) together.
# Both stop when you press Ctrl+C.
#
# Optional env vars:
#   GRADIO_HOST=0.0.0.0   expose the UI on the LAN (default: 127.0.0.1)
#   COMFYUI_PORT=8188     change the ComfyUI port
#   GRADIO_PORT=7860      change the Gradio port

set -euo pipefail

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

cleanup() {
  echo
  echo "[run.sh] stopping..."
  for pid in "$TAIL_PID" "$GRADIO_PID" "$COMFY_PID"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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

cat <<EOF

----------------------------------------------------
  Face-Swap demo is ready.

  Browser:    http://$HOST_DISPLAY:$GRADIO_PORT
  ComfyUI:    http://127.0.0.1:$COMFY_PORT
  Logs:       $LOG_DIR/

  Press Ctrl+C to stop both services.
----------------------------------------------------

EOF

tail -f "$COMFY_LOG" "$GRADIO_LOG" &
TAIL_PID=$!

# Block until either child exits; trap will tear the other down.
while kill -0 "$COMFY_PID" 2>/dev/null && kill -0 "$GRADIO_PID" 2>/dev/null; do
  sleep 2
done

echo
echo "[run.sh] a service exited unexpectedly; shutting down the other"
exit 1
