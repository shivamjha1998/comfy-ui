"""Gradio interface — the only thing agency staff ever sees.

Flow:
    1. Upload source image (face reference)
    2. Upload target video
    3. Adjust parameters
    4. Click Execute → progress → preview → download

Videos always go through a chunk-and-stitch pipeline: ffmpeg splits the
(possibly downscaled) target into N-second pieces, each piece is sent through
the swap workflow independently, and the results are concatenated back with no
re-encoding. This keeps the per-job memory footprint bounded so even long
clips don't OOM on 16 GB Macs, while preserving quality end-to-end.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import gradio as gr
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image

from .comfyui_client import ComfyUIClient, ComfyUIError, UploadedFile
from .workflow import WfAParams, WorkflowError, load_template, patch_wf_a


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FACE_DETECTOR_PATH = _REPO_ROOT / "ComfyUI" / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"

_client: ComfyUIClient | None = None
_MAX_WIDTH = 1280
_MAX_FPS = 30
_MAX_DURATION_S = 15 * 60         # refuse anything longer than 15 min
_CHUNK_S_WITH_RESTORE = 10        # GFPGAN ceiling on 16 GB M4 is ~10–15 s
_CHUNK_S_NO_RESTORE = 30          # plenty of headroom without restoration
_TARGET_SCAN_SECONDS = 5          # how far into the target to scan for a preview face frame
_TARGET_SCAN_SAMPLE_EVERY = 5     # check every Nth frame to keep the scan fast


def _get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client


def _annotate_faces(frame: np.ndarray) -> Image.Image | None:
    """Run YOLO face detection on a BGR frame, sort by area (largest to smallest),
    draw bounding boxes with index numbers, and return as a PIL Image.
    Returns None if no faces found or detector missing.
    """
    if not _FACE_DETECTOR_PATH.is_file():
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model = YOLO(str(_FACE_DETECTOR_PATH))
    results = model(frame, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort boxes by area descending
    sorted_indices = np.argsort(areas)[::-1]
    sorted_boxes = boxes[sorted_indices]

    annotated = frame.copy()
    for idx, (x1, y1, x2, y2) in enumerate(sorted_boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # Draw label background
        label = f"[{idx}]"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _preview_source_faces(source_path: str | None) -> Image.Image | None:
    if not source_path:
        return None
    try:
        frame = cv2.imread(source_path)
        if frame is None:
            return None
        return _annotate_faces(frame)
    except Exception:
        return None


def _preview_target_faces(video_path: str | None) -> Image.Image | None:
    if not video_path:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        max_frames = int(fps * _TARGET_SCAN_SECONDS)
        idx = 0
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % _TARGET_SCAN_SAMPLE_EVERY == 0:
                res = _annotate_faces(frame)
                if res is not None:
                    cap.release()
                    return res
            idx += 1
        cap.release()
        return None
    except Exception:
        return None


def _video_duration_s(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps > 0 else 0.0


def _prepare_video(path: str) -> tuple[str, str | None]:
    """Single-pass re-encode: normalize to <=1280px / 30 fps with dense keyframes.

    Dense keyframes (~every 1 s, GOP=30) are required by `_chunk_video` so the
    `-c copy` segment muxer can split close to the requested boundary. Also
    sidesteps the macOS pyav/cv2 dylib clash that black-screens 1080p inputs.

    Returns (path_to_use, status_message_or_None).
    """
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # Only apply scale filter if needed — otherwise just normalize fps + GOP.
    vf = []
    if w > _MAX_WIDTH:
        vf.append(f"scale={_MAX_WIDTH}:-2")
    if fps > _MAX_FPS + 0.5:
        vf.append(f"fps={_MAX_FPS}")

    cmd = [get_ffmpeg_exe(), "-y", "-i", path]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-g", str(_MAX_FPS), "-keyint_min", str(_MAX_FPS), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "128k", "-map", "0",
        out,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if vf:
        msg = f"Normalized {w}×{h}@{fps:.0f}fps → ≤{_MAX_WIDTH}×?@{_MAX_FPS}fps"
    else:
        msg = None  # quiet pass-through when already within limits
    return out, msg


def _chunk_video(path: str, seconds_per_chunk: int) -> tuple[Path, list[str]]:
    """Split a prepared video into ~N-second chunks with ffmpeg's segment muxer.

    Lossless `-c copy` — assumes the input has been through `_prepare_video`
    so keyframes are ~1 s apart, which makes splits land within ~1 s of the
    requested boundary. Returns (chunk_dir, sorted_chunk_paths); caller is
    responsible for cleanup.
    """
    chunk_dir = Path(tempfile.mkdtemp(prefix="fs_chunks_"))
    pattern = str(chunk_dir / "chunk_%04d.mp4")
    cmd = [
        get_ffmpeg_exe(), "-y", "-i", path,
        "-c", "copy", "-map", "0",
        "-f", "segment",
        "-segment_time", str(seconds_per_chunk),
        "-reset_timestamps", "1",
        pattern,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    chunks = sorted(chunk_dir.glob("chunk_*.mp4"))
    return chunk_dir, [str(p) for p in chunks]


def _concat_videos(chunk_paths: list[str], output_path: str) -> None:
    """Concatenate chunks via ffmpeg concat demuxer — lossless, no re-encode."""
    list_file = Path(output_path).with_suffix(".concat.txt")
    with list_file.open("w") as fh:
        for p in chunk_paths:
            # ffmpeg concat list format requires single-quoted paths
            fh.write(f"file '{p}'\n")
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    list_file.unlink(missing_ok=True)


def _process_chunk(
    client: ComfyUIClient,
    chunk_path: str,
    src_ref: str,
    params: WfAParams,
    template: dict,
) -> str:
    """Run one chunk through the swap workflow and return a local temp file path."""
    # Rename the chunk to a unique name before upload so concurrent / repeat
    # runs don't collide in ComfyUI's input/ directory.
    unique_name = f"fs_{os.getpid()}_{Path(chunk_path).stem}.mp4"
    staged = Path(tempfile.gettempdir()) / unique_name
    shutil.copy(chunk_path, staged)
    uploaded = client.upload(str(staged))
    staged.unlink(missing_ok=True)

    params = WfAParams(**{**params.__dict__, "target_video": uploaded.reference, "source_image": src_ref})
    graph = patch_wf_a(template, params)

    prompt_id = client.submit(graph)
    for _ in client.stream_progress(prompt_id):
        pass  # we only care that it finished; outer loop reports chunk-level progress

    history = client.history(prompt_id)
    for node_outputs in history.get("outputs", {}).values():
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
                return tmp.name
    raise ComfyUIError("workflow finished but returned no output file")


def _run(
    source_file: str | None,
    target_video: str | None,
    target_face_index: int,
    source_face_index: int,
    face_restore_strength: float,
    detect_gender: str,
    enable_interp: bool,
    face_boost_strength: float,
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

        yield ("⏳ Preparing target video…", None)
        target_path, prep_msg = _prepare_video(target_video)
        if prep_msg:
            yield (f"⏳ {prep_msg}", None)

        duration = _video_duration_s(target_path)
        if duration > _MAX_DURATION_S:
            yield (f"⚠ Video too long ({duration/60:.1f} min). Max {_MAX_DURATION_S//60} min.", None)
            return

        seconds_per_chunk = (
            _CHUNK_S_WITH_RESTORE if face_restore_strength > 0 else _CHUNK_S_NO_RESTORE
        )

        yield (f"⏳ Splitting into ~{seconds_per_chunk}s chunks…", None)
        chunk_dir, chunks = _chunk_video(target_path, seconds_per_chunk)
        yield (f"⏳ {len(chunks)} chunk{'s' if len(chunks)!=1 else ''} to process.", None)

        template = load_template("wf_a_reactor")
        base_params = WfAParams(
            source_image=src.reference,
            target_video="<set-per-chunk>",
            target_face_index=int(target_face_index),
            source_face_index=int(source_face_index),
            face_restore_strength=float(face_restore_strength),
            detect_gender=detect_gender,
            enable_frame_interp=bool(enable_interp),
            face_boost_strength=float(face_boost_strength),
        )

        result_chunk_paths: list[str] = []
        for i, chunk_path in enumerate(chunks, 1):
            yield (f"⚙ Chunk {i}/{len(chunks)} — processing…", None)
            result_chunk_paths.append(
                _process_chunk(client, chunk_path, src.reference, base_params, template)
            )

        yield (f"⏳ Concatenating {len(result_chunk_paths)} chunk{'s' if len(result_chunk_paths)!=1 else ''}…", None)
        final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        _concat_videos(result_chunk_paths, final)

        # Cleanup intermediates (best-effort)
        for p in result_chunk_paths:
            Path(p).unlink(missing_ok=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        if target_path != target_video:
            Path(target_path).unlink(missing_ok=True)

        yield ("✓ Done.", final)

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
                with gr.Accordion("Source Face Index Preview", open=False):
                    source_faces_preview = gr.Image(interactive=False, show_label=False)
                target = gr.Video(label="② Target video", sources=["upload"])
                with gr.Accordion("Target Face Index Preview", open=False):
                    target_faces_preview = gr.Image(interactive=False, show_label=False)

            with gr.Column(scale=1):
                gr.Markdown("### ③ Parameters")
                with gr.Row():
                    target_face_index = gr.Number(value=0, precision=0, label="Target face index (0 = first detected)")
                    source_face_index = gr.Number(value=0, precision=0, label="Source face index (0 = first detected)")
                face_restore = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                         label="Face restore strength (0 = off; > 0 = GFPGAN restoration per chunk)")
                detect_gender = gr.Dropdown(["no", "male", "female"], value="no", label="Gender filter")
                enable_interp = gr.Checkbox(value=False,
                                            label="Enable RIFE frame interpolation (2×, smoother motion)")
                face_boost_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                label="Face boost strength (0 = off; higher = sharper face but can warp edges)")
                run_btn = gr.Button("④ Execute", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False)
        result = gr.Video(label="Result", interactive=False)

        run_btn.click(
            _run,
            inputs=[source, target, target_face_index, source_face_index, face_restore, detect_gender,
                    enable_interp, face_boost_strength],
            outputs=[status, result],
        )

        source.change(
            fn=_preview_source_faces,
            inputs=[source],
            outputs=[source_faces_preview]
        )
        target.change(
            fn=_preview_target_faces,
            inputs=[target],
            outputs=[target_faces_preview]
        )

    return demo
