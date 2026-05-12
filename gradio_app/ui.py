"""Gradio interface — the only thing agency staff ever sees.

Flow:
    1. Upload source image (face reference)
    2. Upload target video
    3. Adjust parameters
    4. Click Execute → progress → preview → download
"""
from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
from typing import Generator

import gradio as gr

from .comfyui_client import ComfyUIClient, ComfyUIError, UploadedFile
from .workflow import WfAParams, WorkflowError, load_template, patch_wf_a


_client: ComfyUIClient | None = None


def _get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client


def _run(
    source_file: str | None,
    target_video: str | None,
    face_index: int,
    face_restore_strength: float,
    detect_gender: str,
    enable_interp: bool,
) -> Generator[tuple[str, str | None], None, None]:
    """Generator yielding (status_text, output_video_path_or_None)."""
    try:
        if not source_file or not target_video:
            yield ("⚠ Please upload both a source image and a target video.", None)
            return

        client = _get_client()
        if not client.is_alive():
            yield (f"⚠ ComfyUI is not reachable at {client.http_url}. Is the back-end up?", None)
            return

        yield ("⏳ Uploading source image…", None)
        src: UploadedFile = client.upload(source_file)

        yield ("⏳ Uploading target video…", None)
        tgt: UploadedFile = client.upload(target_video)

        yield ("⏳ Building workflow…", None)
        template = load_template("wf_a_reactor")
        graph = patch_wf_a(
            template,
            WfAParams(
                source_image=src.reference,
                target_video=tgt.reference,
                face_index=int(face_index),
                face_restore_strength=float(face_restore_strength),
                detect_gender=detect_gender,
                enable_frame_interp=bool(enable_interp),
            ),
        )

        yield ("⏳ Submitting to ComfyUI…", None)
        prompt_id = client.submit(graph)

        last_node: str | None = None
        for event in client.stream_progress(prompt_id):
            etype = event.get("type")
            data = event.get("data", {})
            if etype == "executing":
                node = data.get("node")
                if node and node != last_node:
                    last_node = node
                    yield (f"⚙ Running node {node}…", None)
            elif etype == "progress":
                cur = data.get("value", 0)
                total = data.get("max", 1)
                pct = int(100 * cur / max(total, 1))
                yield (f"⚙ {pct}% (step {cur}/{total})", None)

        yield ("⏳ Fetching result…", None)
        history = client.history(prompt_id)
        outputs = history.get("outputs", {})

        result_path: str | None = None
        for node_outputs in outputs.values():
            for key in ("gifs", "videos", "images"):
                for item in node_outputs.get(key, []):
                    blob = client.download_output(
                        item["filename"],
                        item.get("subfolder", ""),
                        item.get("type", "output"),
                    )
                    suffix = Path(item["filename"]).suffix or ".mp4"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(blob)
                    tmp.close()
                    result_path = tmp.name
                    break
                if result_path:
                    break
            if result_path:
                break

        if result_path is None:
            yield ("⚠ Workflow finished but no output file was returned.", None)
            return

        yield ("✓ Done.", result_path)

    except (ComfyUIError, WorkflowError) as exc:
        yield (f"✗ {exc}", None)
    except Exception as exc:  # noqa: BLE001 — UI surface
        traceback.print_exc()
        yield (f"✗ Unexpected error: {exc}", None)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Face-Swap Demo") as demo:
        gr.Markdown("# Face-Swap Demo\n*Powered by ComfyUI + ReActor*")

        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Image(label="① Source face", type="filepath", sources=["upload"])
                target = gr.Video(label="② Target video", sources=["upload"])

            with gr.Column(scale=1):
                gr.Markdown("### ③ Parameters")
                face_index = gr.Number(value=0, precision=0, label="Face index (0 = first detected)")
                face_restore = gr.Slider(0.0, 1.0, value=0.85, step=0.05,
                                         label="Face restore strength (higher = smoother)")
                detect_gender = gr.Dropdown(["no", "male", "female"], value="no", label="Gender filter")
                enable_interp = gr.Checkbox(value=False,
                                            label="Enable RIFE frame interpolation (2×, smoother motion)")
                run_btn = gr.Button("④ Execute", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False)
        result = gr.Video(label="Result", interactive=False)

        run_btn.click(
            _run,
            inputs=[source, target, face_index, face_restore, detect_gender, enable_interp],
            outputs=[status, result],
        )

    return demo
