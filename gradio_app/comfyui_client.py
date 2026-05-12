"""Thin wrapper around the ComfyUI HTTP + WebSocket API.

ComfyUI exposes:
  POST /upload/image          — upload a file (image or video — yes, despite the name)
  POST /prompt                — submit a workflow graph for execution
  GET  /history/{prompt_id}   — fetch outputs after completion
  GET  /view?filename=...     — download an output file
  WS   /ws?clientId=<uuid>    — live execution events

This module is deliberately tiny: no retries, no caching, no thread pool.
The Gradio handler in ui.py is the only consumer; if we need more later,
we add it there.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
import websocket  # from websocket-client

from .config import CONFIG


class ComfyUIError(RuntimeError):
    """Anything ComfyUI returns that we can't make sense of."""


@dataclass
class UploadedFile:
    name: str        # ComfyUI's stored filename (may be renamed if collision)
    subfolder: str   # usually "" for the default input dir
    type: str        # "input" / "temp" / "output"

    @property
    def reference(self) -> str:
        """The string ComfyUI expects when referring back to this file in a graph."""
        return f"{self.subfolder}/{self.name}" if self.subfolder else self.name


class ComfyUIClient:
    def __init__(self, http_url: str | None = None, ws_url: str | None = None) -> None:
        self.http_url = http_url or CONFIG.comfyui_http
        self.ws_url = ws_url or CONFIG.comfyui_ws
        self.client_id = str(uuid.uuid4())

    # ─── Health ──────────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.http_url}/system_stats", timeout=2)
            return r.ok
        except requests.RequestException:
            return False

    def wait_until_ready(self, timeout_s: int = 60) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_alive():
                return
            time.sleep(1)
        raise ComfyUIError(f"ComfyUI not reachable at {self.http_url} after {timeout_s}s")

    # ─── Upload ──────────────────────────────────────────────────────────
    def upload(self, file_path: str | Path, *, overwrite: bool = True) -> UploadedFile:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as fh:
            resp = requests.post(
                f"{self.http_url}/upload/image",
                files={"image": (path.name, fh)},
                data={"overwrite": "true" if overwrite else "false"},
                timeout=CONFIG.request_timeout_s,
            )
        if not resp.ok:
            raise ComfyUIError(f"upload failed ({resp.status_code}): {resp.text[:200]}")
        body = resp.json()
        return UploadedFile(
            name=body["name"],
            subfolder=body.get("subfolder", ""),
            type=body.get("type", "input"),
        )

    # ─── Submit workflow ─────────────────────────────────────────────────
    def submit(self, workflow_graph: dict[str, Any]) -> str:
        payload = {"prompt": workflow_graph, "client_id": self.client_id}
        resp = requests.post(
            f"{self.http_url}/prompt",
            json=payload,
            timeout=CONFIG.request_timeout_s,
        )
        if not resp.ok:
            raise ComfyUIError(f"submit failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if "prompt_id" not in body:
            raise ComfyUIError(f"submit returned no prompt_id: {body}")
        return body["prompt_id"]

    # ─── History / outputs ───────────────────────────────────────────────
    def history(self, prompt_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.http_url}/history/{prompt_id}", timeout=CONFIG.request_timeout_s)
        if not resp.ok:
            raise ComfyUIError(f"history failed ({resp.status_code})")
        body = resp.json()
        return body.get(prompt_id, {})

    def download_output(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        resp = requests.get(f"{self.http_url}/view", params=params, timeout=CONFIG.request_timeout_s * 4)
        if not resp.ok:
            raise ComfyUIError(f"download failed ({resp.status_code})")
        return resp.content

    # ─── Progress stream (WebSocket) ─────────────────────────────────────
    def stream_progress(self, prompt_id: str) -> Iterator[dict[str, Any]]:
        """Yield ComfyUI execution events for a single prompt until completion.

        Events are dicts with at least a 'type' key. We yield raw events and let
        the caller decide what to render. The iterator terminates when ComfyUI
        signals 'execution_success' or 'execution_error' for our prompt_id.
        """
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}")
        try:
            while True:
                raw = ws.recv()
                if isinstance(raw, bytes):
                    # Binary frames are intermediate previews — skip
                    continue
                event = json.loads(raw)
                etype = event.get("type")
                data = event.get("data", {})

                # Filter to our prompt only (ComfyUI broadcasts to all clients)
                if data.get("prompt_id") not in (None, prompt_id):
                    continue

                yield event

                if etype == "execution_success" and data.get("prompt_id") == prompt_id:
                    return
                if etype == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise ComfyUIError(f"execution failed: {data}")
        finally:
            ws.close()

    def run_and_collect_outputs(
        self,
        workflow_graph: dict[str, Any],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[tuple[str, str, str]]:
        """Submit a workflow, stream progress, and return [(filename, subfolder, type)]."""
        prompt_id = self.submit(workflow_graph)
        for event in self.stream_progress(prompt_id):
            if on_progress is not None:
                on_progress(event)

        history = self.history(prompt_id)
        outputs = history.get("outputs", {})
        results: list[tuple[str, str, str]] = []
        for node_outputs in outputs.values():
            # VHS_VideoCombine puts results under 'gifs' (yes, even mp4s) or 'videos'
            for key in ("gifs", "videos", "images"):
                for item in node_outputs.get(key, []):
                    results.append((item["filename"], item.get("subfolder", ""), item.get("type", "output")))
        return results