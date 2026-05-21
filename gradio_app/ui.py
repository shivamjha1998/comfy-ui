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
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import gradio as gr
import requests
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image

from .comfyui_client import ComfyUIClient, ComfyUIError, UploadedFile
from .workflow import WfAParams, WorkflowError, load_template, patch_wf_a


_REPO_ROOT = Path(__file__).resolve().parent.parent
_FACE_DETECTOR_PATH = _REPO_ROOT / "ComfyUI" / "models" / "ultralytics" / "bbox" / "face_yolov8m.pt"
_OUTPUTS_DIR = _REPO_ROOT / ".outputs"

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


def _list_past_jobs() -> list[tuple[str, str]]:
    """Return [(display_label, filepath)] for past saved results, newest first."""
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    files = sorted(_OUTPUTS_DIR.glob("*.mp4"), reverse=True)
    out: list[tuple[str, str]] = []
    for f in files:
        # Stem looks like 2026-05-17_14-32-15 — show as "2026-05-17 14:32:15".
        stem = f.stem
        if len(stem) == 19 and stem[10] == "_":
            label = stem[:10] + "  " + stem[11:].replace("-", ":")
        else:
            label = stem
        out.append((label, str(f)))
    return out


def _persist_result(temp_path: str) -> str:
    """Copy a finished result mp4 into .outputs/ under a timestamped name so it
    survives a page refresh / restart. Returns the new path."""
    _OUTPUTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest = _OUTPUTS_DIR / f"{stamp}.mp4"
    n = 1
    while dest.exists():
        dest = _OUTPUTS_DIR / f"{stamp}_{n}.mp4"
        n += 1
    shutil.copy(temp_path, dest)
    return str(dest)


def _cancel_running_job() -> str:
    """Tell ComfyUI to interrupt whatever it's executing. Best-effort."""
    try:
        client = _get_client()
        requests.post(f"{client.http_url}/interrupt", timeout=3)
    except Exception:  # noqa: BLE001 — never let cancel fail loudly
        pass
    return "✗ Cancelled by user."


def _format_duration(seconds: float) -> str:
    """Render an ETA as e.g. '45s', '3m 20s', or '1h 12m'. Rounds whole seconds."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s}s" if s else f"{minutes}m"
    hours, m = divmod(minutes, 60)
    return f"{hours}h {m}m" if m else f"{hours}h"


def _prepare_source(path: str, face_index: int) -> tuple[str, bool]:
    """Pre-crop the chosen face out of a source image before sending it to ReActor.

    Returns (path_to_use, was_cropped). On success — face_yolov8m detected at
    least one face — returns a temp PNG containing the face at `face_index`
    (sorted largest-first) with ~60 % padding for hair/jaw/neck context, plus
    True. The caller should then force ReActor's source_faces_index to "0"
    because the cropped image only has one face.

    On any failure (detector missing, ultralytics import error, no face found),
    returns the original path + False; the caller should fall back to letting
    ReActor's own buffalo_l pick the face via the user-supplied index.
    """
    if not _FACE_DETECTOR_PATH.is_file():
        return path, False
    try:
        from ultralytics import YOLO
    except ImportError:
        return path, False

    img = cv2.imread(path)
    if img is None:
        return path, False

    model = YOLO(str(_FACE_DETECTOR_PATH))
    results = model(img, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return path, False

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = np.argsort(areas)[::-1]
    sorted_boxes = boxes[sorted_indices]

    idx = face_index if 0 <= face_index < len(sorted_boxes) else 0
    x1, y1, x2, y2 = sorted_boxes[idx]
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * 0.6, bh * 0.6
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    cropped = img[cy1:cy2, cx1:cx2]

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    cv2.imwrite(out, cropped)
    return out, True


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
    source_extra_files: list | None,
    target_video: str | None,
    target_face_index: int,
    source_face_index: int,
    face_restore_strength: float,
    enable_interp: bool,
    face_boost_strength: float,
) -> Generator[tuple, None, None]:
    """Yields (status_text, target_preview_or_skip, result_video_or_skip).

    Cleanup of chunk intermediates lives in `finally` so cancelling the run
    (GeneratorExit) or any unexpected exception still leaves the working
    directory tidy.
    """
    chunk_dir: Path | None = None
    result_chunk_paths: list[str] = []
    target_path: str | None = None
    try:
        if not source_file or not target_video:
            yield ("⚠ Please upload both a source image and a target video.", gr.skip(), gr.skip())
            return

        client = _get_client()
        if not client.is_alive():
            yield (f"⚠ ComfyUI is not reachable at {client.http_url}. Is the back-end up?", gr.skip(), gr.skip())
            return

        # Show the input target alongside the eventual result so the user can compare
        # side-by-side as it runs.
        # Gradio's File(file_count="multiple") returns either a list of NamedString
        # objects (.name attribute) or filepath strings depending on version; handle both.
        extras_raw = source_extra_files or []
        extras_paths = [getattr(f, "name", f) for f in extras_raw]
        n_total = 1 + len(extras_paths)

        if extras_paths:
            yield (f"⏳ Pre-cropping faces from {n_total} source photos…", target_video, gr.skip())
        else:
            yield ("⏳ Pre-cropping face from source image…", target_video, gr.skip())
        # Pre-crop each source to just the chosen face (primary uses the user's
        # source_face_index; extras always take the largest detected face).
        # Cropped sources tighten ReActor's embedding by removing background context.
        prepared_primary, primary_cropped = _prepare_source(source_file, int(source_face_index))
        prepared_extras: list[str] = []
        for p in extras_paths:
            pp, _ = _prepare_source(p, 0)
            prepared_extras.append(pp)

        yield ("⏳ Uploading source(s)…", gr.skip(), gr.skip())
        src: UploadedFile = client.upload(prepared_primary)
        extra_refs: list[str] = []
        for p in prepared_extras:
            extra_refs.append(client.upload(p).reference)

        # When we pre-cropped, the uploaded source has exactly one face — index 0.
        # When YOLO didn't detect anything, fall back to the user's chosen index
        # so ReActor's own face detection still picks the right face.
        effective_source_face_index = 0 if primary_cropped else int(source_face_index)

        yield ("⏳ Preparing target video…", gr.skip(), gr.skip())
        target_path, prep_msg = _prepare_video(target_video)
        if prep_msg:
            yield (f"⏳ {prep_msg}", gr.skip(), gr.skip())

        duration = _video_duration_s(target_path)
        if duration > _MAX_DURATION_S:
            yield (f"⚠ Video too long ({duration/60:.1f} min). Max {_MAX_DURATION_S//60} min.", gr.skip(), gr.skip())
            return

        seconds_per_chunk = (
            _CHUNK_S_WITH_RESTORE if face_restore_strength > 0 else _CHUNK_S_NO_RESTORE
        )

        yield (f"⏳ Splitting into ~{seconds_per_chunk}s chunks…", gr.skip(), gr.skip())
        chunk_dir, chunks = _chunk_video(target_path, seconds_per_chunk)
        yield (f"⏳ {len(chunks)} chunk{'s' if len(chunks)!=1 else ''} to process.", gr.skip(), gr.skip())

        template = load_template("wf_a_reactor")
        base_params = WfAParams(
            source_image=src.reference,
            extra_source_images=tuple(extra_refs),
            target_video="<set-per-chunk>",
            target_face_index=int(target_face_index),
            source_face_index=effective_source_face_index,
            face_restore_strength=float(face_restore_strength),
            enable_frame_interp=bool(enable_interp),
            face_boost_strength=float(face_boost_strength),
        )

        chunk_durations: list[float] = []
        total_chunks = len(chunks)
        for i, chunk_path in enumerate(chunks, 1):
            # ETA: start showing once we have at least one chunk's actual duration.
            if chunk_durations:
                avg = sum(chunk_durations) / len(chunk_durations)
                eta = avg * (total_chunks - (i - 1))
                eta_str = f" · ~{_format_duration(eta)} remaining"
            else:
                eta_str = ""
            yield (f"⚙ Chunk {i}/{total_chunks} — processing{eta_str}", gr.skip(), gr.skip())
            t0 = time.time()
            result_chunk_paths.append(
                _process_chunk(client, chunk_path, src.reference, base_params, template)
            )
            chunk_durations.append(time.time() - t0)

        yield (f"⏳ Concatenating {len(result_chunk_paths)} chunk{'s' if len(result_chunk_paths)!=1 else ''}…", gr.skip(), gr.skip())
        final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        _concat_videos(result_chunk_paths, final)

        # Persist to .outputs/ so the user can re-download from the Past Jobs
        # list even after a browser refresh.
        persisted = _persist_result(final)
        Path(final).unlink(missing_ok=True)

        yield ("✓ Done.", gr.skip(), persisted)

    except (ComfyUIError, WorkflowError) as exc:
        yield (f"✗ {exc}", gr.skip(), gr.skip())
    except Exception as exc:  # noqa: BLE001 — UI surface
        traceback.print_exc()
        yield (f"✗ Unexpected error: {exc}", gr.skip(), gr.skip())
    finally:
        # Runs on success, error, AND cancel (GeneratorExit).
        for p in result_chunk_paths:
            Path(p).unlink(missing_ok=True)
        if chunk_dir is not None:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        if target_path is not None and target_path != target_video:
            Path(target_path).unlink(missing_ok=True)


THEME = gr.themes.Soft(
    primary_hue="orange",
    neutral_hue="stone",
    spacing_size="lg",
    radius_size="lg",
).set(
    # Background tokens sampled from the styleguide reference: warm cream page,
    # crisp white surfaces, soft warm-gray borders, no harsh shadows.
    body_background_fill="#FAF7F2",
    body_background_fill_dark="#1A1815",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#262421",
    background_fill_secondary="#FBF8F3",
    border_color_primary="#E5E2DC",
    block_border_width="1px",
    block_shadow="0 2px 10px rgba(50, 40, 30, 0.05)",
    panel_background_fill="#FFFFFF",
    # Coral-orange primary (lifted off the styleguide buttons).
    button_primary_background_fill="#F76C3F",
    button_primary_background_fill_hover="#E55A2D",
    button_primary_text_color="#FFFFFF",
)

CSS = """
/* Full-width container — was capped at 1200px; styleguide layout wants the
   whole screen, with generous side padding so content doesn't kiss the edge. */
.gradio-container { max-width: none !important; padding: 24px 40px !important; }

/* Hero title — generous space, big bold heading like the styleguide */
#hero { padding: 12px 0 28px; }
#hero h1 { font-size: 2.6em; font-weight: 800; margin: 0; line-height: 1.15; letter-spacing: -0.01em; }
#hero p  { color: var(--body-text-color-subdued); margin-top: 6px; font-size: 1.05em; }

/* Headings — breathing room top/bottom so sections don't crowd each other */
h2, h3, h4 { margin-top: 16px !important; margin-bottom: 12px !important; }

/* Tab nav — minimal text-style tabs with a coral underline on active.
   Mirrors the "General · Analytics · Settings" tab pattern in the styleguide. */
.tabs > .tab-nav {
  border-bottom: 1px solid var(--border-color-primary);
  margin-bottom: 16px;
  gap: 4px;
}
.tabs > .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 18px !important;
  color: var(--body-text-color-subdued);
  font-weight: 600;
}
.tabs > .tab-nav button.selected {
  border-bottom-color: #F76C3F !important;
  color: var(--body-text-color) !important;
  background: transparent !important;
}

/* Tab content padding so controls aren't pressed to the edge */
.tabitem { padding: 8px 4px !important; }
.tabitem > .markdown { margin-bottom: 12px; }

/* Card surfaces — white bg, soft elevation, warm-gray border. Wrap major
   sections (inputs, parameters, output, history) in gr.Group(elem_classes="section-card")
   to lift them off the cream page background like the styleguide's card examples. */
.section-card {
  background: #FFFFFF !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 14px !important;
  box-shadow: 0 2px 10px rgba(50, 40, 30, 0.05) !important;
  padding: 20px !important;
}
.section-card + .section-card { margin-top: 20px; }

/* Status textbox styled like the styleguide's alert pills */
#status-alert textarea {
  background: #FBF8F3 !important;
  border-left: 3px solid #F76C3F !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  font-weight: 500;
}
"""


def build_ui() -> gr.Blocks:
    # Gradio 6.0 moved theme/css to launch() — see __main__.py for where they're applied.
    with gr.Blocks(title="Face-Swap Demo") as demo:
        with gr.Group(elem_id="hero"):
            gr.Markdown("# Face-Swap Demo\nPowered by ComfyUI + ReActor")

        with gr.Tabs():
            # ─── Tab 1: Swap ─────────────────────────────────────────────
            with gr.Tab("Swap"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="section-card"):
                            source = gr.Image(label="① Source face", type="filepath", sources=["upload"])
                            with gr.Accordion("Source Face Index Preview", open=False):
                                source_faces_preview = gr.Image(interactive=False, show_label=False)
                            with gr.Accordion("Additional source photos (optional)", open=False):
                                gr.Markdown(
                                    "Add more photos of the same person here to average their face "
                                    "embeddings — gives a stronger, more stable identity."
                                )
                                source_extra = gr.File(
                                    file_count="multiple",
                                    file_types=["image"],
                                    type="filepath",
                                    show_label=False,
                                )
                            target = gr.Video(label="② Target video", sources=["upload"])
                            with gr.Accordion("Target Face Index Preview", open=False):
                                target_faces_preview = gr.Image(interactive=False, show_label=False)

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes="section-card"):
                            gr.Markdown("### ③ Parameters")
                            with gr.Row():
                                target_face_index = gr.Number(value=0, precision=0, label="Target face index (0 = first detected)")
                                source_face_index = gr.Number(value=0, precision=0, label="Source face index (0 = first detected)")
                            face_restore = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                     label="Face restore strength (0 = off; > 0 = GFPGAN restoration per chunk)")
                            enable_interp = gr.Checkbox(value=False,
                                                        label="Enable RIFE frame interpolation (2×, smoother motion)")
                            face_boost_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                            label="Face boost strength (0 = off; higher = sharper face but can warp edges)")
                            with gr.Row():
                                run_btn = gr.Button("④ Execute", variant="primary", size="lg")
                                cancel_btn = gr.Button("✗ Cancel", variant="stop", size="lg")

                with gr.Group(elem_classes="section-card"):
                    status = gr.Textbox(label="Status", interactive=False, elem_id="status-alert")
                    with gr.Row():
                        target_preview = gr.Video(label="Target (original)", interactive=False)
                        result = gr.Video(label="Result (swapped)", interactive=False)

            # ─── Tab 2: History ──────────────────────────────────────────
            with gr.Tab("History"):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown(
                        "Completed swaps are saved under `.outputs/` and listed here even after a page refresh. "
                        "Pick one to replay or right-click the player to download."
                    )
                    with gr.Row():
                        past_dropdown = gr.Dropdown(
                            choices=_list_past_jobs(),
                            label="Past results",
                            interactive=True,
                        )
                        past_refresh_btn = gr.Button("🔄 Refresh", scale=0)
                    past_video = gr.Video(label="Past result playback", interactive=False)

        # ─── Event handlers (cross-tab references work since all components are in the same Blocks) ─
        run_event = run_btn.click(
            _run,
            inputs=[source, source_extra, target, target_face_index, source_face_index, face_restore,
                    enable_interp, face_boost_strength],
            outputs=[status, target_preview, result],
        )
        # Re-list past jobs once the run finishes so the new result appears in the History tab.
        run_event.then(
            fn=lambda: gr.update(choices=_list_past_jobs()),
            outputs=[past_dropdown],
        )

        # Cancel: tell ComfyUI to interrupt + cancel the Gradio event (raises
        # GeneratorExit inside _run, which our finally-block handles).
        cancel_btn.click(
            fn=_cancel_running_job,
            outputs=[status],
            cancels=[run_event],
        )

        past_dropdown.change(
            fn=lambda p: p,
            inputs=[past_dropdown],
            outputs=[past_video],
        )
        past_refresh_btn.click(
            fn=lambda: gr.update(choices=_list_past_jobs()),
            outputs=[past_dropdown],
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
