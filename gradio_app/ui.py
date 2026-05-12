"""Gradio interface — the only thing agency staff ever sees.

The four-step flow from spec §6.1:
    1. Upload source image (face / character reference)
    2. Upload target video
    3. Select workflow + tune parameters
    4. Click Execute → progress bar → preview → download

Authentication is provided either by Gradio's built-in auth (env vars) or
by Nginx Basic Auth in front of us — see scripts/launch.sbatch + nginx
config (added in a later task).
"""
from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Generator

import gradio as gr

from .comfyui_client import ComfyUIClient, ComfyUIError, UploadedFile
from .config import CONFIG
from .workflow import (
    WfAParams,
    WfBParams,
    WorkflowError,
    load_template,
    patch_wf_a,
    patch_wf_b,
)


_client: ComfyUIClient | None = None


def _get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client


# ─── Workflow runner (yields progress messages for the UI) ───────────────
def _run(
    workflow_choice: str,
    source_file: str | None,
    target_video: str | None,
    face_index: int,
    face_restore_strength: float,
    detect_gender: str,
    enable_interp: bool,
    prompt: str,
    relight_w: float,
    steps: int,
    segment_frames: int,
) -> Generator[tuple[str, str | None], None, None]:
    """Generator that yields (status_text, output_video_path_or_None)."""
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

        yield (f"⏳ Building {workflow_choice} workflow…", None)
        if workflow_choice.startswith("wf_a"):
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
        else:
            template = load_template("wf_b_wan22")
            # Auto-pick resolution preset from the *local* video file's aspect.
            # Done before upload so we still have direct fs access.
            from .workflow import video_dims_for_target
            tw, th = video_dims_for_target(target_video)
            yield (f"⏳ Building wf_b_wan22 workflow ({tw}×{th})…", None)
            graph = patch_wf_b(
                template,
                WfBParams(
                    reference_image=src.reference,
                    target_video=tgt.reference,
                    prompt=prompt or "",
                    relight_lora_weight=float(relight_w),
                    steps=int(steps),
                    segment_length_frames=int(segment_frames),
                    target_width=tw,
                    target_height=th,
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


# ─── UI definition ───────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VTuber Face-Swap System") as demo:
        gr.Markdown("# VTuber Face-Swap System\n*Powered by ComfyUI · ReActor · WAN 2.2*")

        with gr.Row():
            with gr.Column(scale=1):
                workflow = gr.Radio(
                    choices=[
                        ("Face Swap (fast, long videos)", "wf_a_reactor"),
                        ("Full-Body Replace (short clips, lip-sync)", "wf_b_wan22"),
                    ],
                    value="wf_a_reactor",
                    label="① Workflow",
                )
                source = gr.Image(label="② Source face / character", type="filepath", sources=["upload"])
                target = gr.Video(label="③ Target video", sources=["upload"])

            with gr.Column(scale=1):
                gr.Markdown("### ④ Parameters")

                with gr.Group(visible=True) as wf_a_panel:
                    gr.Markdown("**WF-A: ReActor settings**")
                    face_index = gr.Number(value=0, precision=0, label="Face index (0 = first detected)")
                    face_restore = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Face restore strength (higher = smoother)")
                    detect_gender = gr.Dropdown(["no", "male", "female"], value="no", label="Gender filter")
                    enable_interp = gr.Checkbox(value=True, label="Enable RIFE frame interpolation (2×, smoother motion)")

                with gr.Group(visible=False) as wf_b_panel:
                    gr.Markdown("**WF-B: WAN 2.2 settings**")
                    prompt = gr.Textbox(label="Scene description (optional)", lines=2)
                    relight_w = gr.Slider(0.0, 1.5, value=0.6, step=0.05, label="Relight LoRA weight")
                    steps = gr.Slider(20, 80, value=40, step=1, label="Diffusion steps")
                    segment_frames = gr.Slider(40, 81, value=81, step=1, label="Segment length (frames)")

                run_btn = gr.Button("⑤ Execute", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False)
        result = gr.Video(label="Result", interactive=False)

        # Toggle parameter panels based on workflow choice
        def _toggle_panel(choice: str):
            return (
                gr.update(visible=choice == "wf_a_reactor"),
                gr.update(visible=choice == "wf_b_wan22"),
            )
        workflow.change(_toggle_panel, inputs=workflow, outputs=[wf_a_panel, wf_b_panel])

        run_btn.click(
            _run,
            inputs=[workflow, source, target, face_index, face_restore, detect_gender, enable_interp,
                    prompt, relight_w, steps, segment_frames],
            outputs=[status, result],
        )

    return demo