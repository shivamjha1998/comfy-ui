"""Entrypoint: `python -m gradio_app` from inside the container."""
from __future__ import annotations

import sys

from .comfyui_client import ComfyUIClient
from .config import CONFIG
from .ui import build_ui


def main() -> int:
    print(f"[gradio_app] Waiting for ComfyUI at {CONFIG.comfyui_http} …", flush=True)
    client = ComfyUIClient()
    try:
        client.wait_until_ready(timeout_s=120)
    except Exception as exc:  # noqa: BLE001
        print(f"[gradio_app] ComfyUI did not come up: {exc}", file=sys.stderr)
        return 1
    print("[gradio_app] ComfyUI is ready. Starting Gradio…", flush=True)

    demo = build_ui()
    demo.queue().launch(
        server_name=CONFIG.gradio_host,
        server_port=CONFIG.gradio_port,
        auth=CONFIG.gradio_auth,
        share=False,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())