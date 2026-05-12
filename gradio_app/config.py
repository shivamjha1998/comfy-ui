"""Configuration via env vars with sane defaults for local Mac use."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # ─── ComfyUI back-end ────────────────────────────────────────────────
    comfyui_host: str = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    comfyui_port: int = int(os.environ.get("COMFYUI_PORT", "8188"))

    # ─── Gradio front-end ────────────────────────────────────────────────
    gradio_host: str = os.environ.get("GRADIO_HOST", "127.0.0.1")
    gradio_port: int = int(os.environ.get("GRADIO_PORT", "7860"))
    gradio_auth_user: str | None = os.environ.get("GRADIO_AUTH_USER")
    gradio_auth_pass: str | None = os.environ.get("GRADIO_AUTH_PASS")

    # ─── Filesystem (paths relative to the repo root) ────────────────────
    workflows_dir: Path = Path(os.environ.get("WORKFLOWS_DIR", str(_REPO_ROOT / "workflows")))
    input_dir: Path = Path(os.environ.get("INPUT_DIR", str(_REPO_ROOT / "ComfyUI" / "input")))
    output_dir: Path = Path(os.environ.get("OUTPUT_DIR", str(_REPO_ROOT / "ComfyUI" / "output")))

    # ─── Behavior ────────────────────────────────────────────────────────
    request_timeout_s: int = int(os.environ.get("REQUEST_TIMEOUT_S", "30"))

    @property
    def comfyui_http(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def comfyui_ws(self) -> str:
        return f"ws://{self.comfyui_host}:{self.comfyui_port}/ws"

    @property
    def gradio_auth(self) -> tuple[str, str] | None:
        if self.gradio_auth_user and self.gradio_auth_pass:
            return (self.gradio_auth_user, self.gradio_auth_pass)
        return None


CONFIG = Config()