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
_STYLE_MATCH_SCAN_SECONDS = 5     # only scan first N s of target for a face frame
_STYLE_MATCH_SAMPLE_EVERY = 5     # check every Nth frame to keep scan fast
# Gemini image-generation model. Default to Nano Banana Pro (gemini-3-pro-image)
# for best instruction-following on the style-match prompt. Override via env var
# if you want the cheaper / faster gemini-2.5-flash-image or gemini-3.1-flash-image-preview.
_GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

_STYLE_MATCH_PROMPT = (
    "Two images are provided. Image 1 is a frame from a video showing a person in a scene. "
    "Image 2 is a reference photograph of a different person whose face we want to use. "
    "Generate a new photographic portrait of the person from Image 2 — same identity and face "
    "features — but pose, framing, head angle, eye gaze, facial expression, lighting direction, "
    "color palette, and background composition must exactly match Image 1. The output must look "
    "like a still photograph taken in the same moment and place as Image 1, just with the "
    "person's identity replaced with the one from Image 2. Output only the new portrait, no text."
)


def _get_client() -> ComfyUIClient:
    global _client
    if _client is None:
        _client = ComfyUIClient()
    return _client


def _find_first_face_frame(video_path: str) -> Image.Image | None:
    """Scan up to _STYLE_MATCH_SCAN_SECONDS of the video for a frame with a
    detectable face. Returns the PIL image of the first match, or None if no
    detector is available or no face is found in the scan window.
    """
    if not _FACE_DETECTOR_PATH.is_file():
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    model = YOLO(str(_FACE_DETECTOR_PATH))
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * _STYLE_MATCH_SCAN_SECONDS)
    idx = 0
    found: Image.Image | None = None
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % _STYLE_MATCH_SAMPLE_EVERY == 0:
            results = model(frame, verbose=False)
            if results and len(results[0].boxes) > 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                found = Image.fromarray(rgb)
                break
        idx += 1
    cap.release()
    return found


def _nano_banana_match(source_path: str, target_frame: Image.Image) -> str:
    """Call Gemini 2.5 Flash Image to render the source face in the style of the
    target frame. Returns the path to a temp PNG. Raises on any failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise WorkflowError("GEMINI_API_KEY not set (Nano Banana style-match disabled)")

    from google import genai
    client = genai.Client(api_key=api_key)

    source_img = Image.open(source_path)
    response = client.models.generate_content(
        model=_GEMINI_IMAGE_MODEL,
        contents=[_STYLE_MATCH_PROMPT, target_frame, source_img],
    )

    # Newer SDK shape: response.candidates[0].content.parts; older: response.parts
    parts = []
    cands = getattr(response, "candidates", None)
    if cands:
        parts = list(cands[0].content.parts)
    elif hasattr(response, "parts"):
        parts = list(response.parts)

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            with open(out, "wb") as fh:
                fh.write(inline.data)
            return out

    raise WorkflowError("Gemini returned no image in the response")


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
    face_index: int,
    face_restore_strength: float,
    detect_gender: str,
    enable_interp: bool,
    enable_face_boost: bool,
    enable_style_match: bool,
) -> Generator[tuple, None, None]:
    """Generator yielding (status_text, stylized_preview_path_or_skip, output_video_path_or_skip).

    Use gr.skip() to leave an output untouched on a given yield.
    """
    try:
        if not source_file or not target_video:
            yield ("⚠ Please upload both a source image and a target video.", gr.skip(), gr.skip())
            return

        client = _get_client()
        if not client.is_alive():
            yield (f"⚠ ComfyUI is not reachable at {client.http_url}. Is the back-end up?", gr.skip(), gr.skip())
            return

        effective_source = source_file
        if enable_style_match:
            if not os.environ.get("GEMINI_API_KEY"):
                yield ("⚠ GEMINI_API_KEY not set — skipping style-match, using raw source.", gr.skip(), gr.skip())
            else:
                yield ("⏳ Scanning target for a frame with a face…", gr.skip(), gr.skip())
                frame = _find_first_face_frame(target_video)
                if frame is None:
                    yield ("⚠ No face found in first 5 s of target — using raw source.", gr.skip(), gr.skip())
                else:
                    yield ("⏳ Calling Gemini Nano Banana to style-match the source…", gr.skip(), gr.skip())
                    try:
                        effective_source = _nano_banana_match(source_file, frame)
                        yield ("✓ Style-matched source ready.", effective_source, gr.skip())
                    except Exception as exc:  # noqa: BLE001 — surface any Gemini issue and fall back
                        yield (f"⚠ Style-match failed ({exc}); using raw source.", gr.skip(), gr.skip())

        yield ("⏳ Uploading source image…", gr.skip(), gr.skip())
        src: UploadedFile = client.upload(effective_source)

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
            target_video="<set-per-chunk>",
            face_index=int(face_index),
            face_restore_strength=float(face_restore_strength),
            detect_gender=detect_gender,
            enable_frame_interp=bool(enable_interp),
            enable_face_boost=bool(enable_face_boost),
        )

        result_chunk_paths: list[str] = []
        for i, chunk_path in enumerate(chunks, 1):
            yield (f"⚙ Chunk {i}/{len(chunks)} — processing…", gr.skip(), gr.skip())
            result_chunk_paths.append(
                _process_chunk(client, chunk_path, src.reference, base_params, template)
            )

        yield (f"⏳ Concatenating {len(result_chunk_paths)} chunk{'s' if len(result_chunk_paths)!=1 else ''}…", gr.skip(), gr.skip())
        final = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        _concat_videos(result_chunk_paths, final)

        # Cleanup intermediates (best-effort)
        for p in result_chunk_paths:
            Path(p).unlink(missing_ok=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        if target_path != target_video:
            Path(target_path).unlink(missing_ok=True)

        yield ("✓ Done.", gr.skip(), final)

    except (ComfyUIError, WorkflowError) as exc:
        yield (f"✗ {exc}", gr.skip(), gr.skip())
    except Exception as exc:  # noqa: BLE001 — UI surface
        traceback.print_exc()
        yield (f"✗ Unexpected error: {exc}", gr.skip(), gr.skip())


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
                face_restore = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                         label="Face restore strength (0 = off; > 0 = GFPGAN restoration per chunk)")
                detect_gender = gr.Dropdown(["no", "male", "female"], value="no", label="Gender filter")
                enable_interp = gr.Checkbox(value=False,
                                            label="Enable RIFE frame interpolation (2×, smoother motion)")
                enable_face_boost = gr.Checkbox(value=False,
                                                label="Face boost (extra face crop upscale — can cause edge warping)")
                enable_style_match = gr.Checkbox(value=bool(os.environ.get("GEMINI_API_KEY")),
                                                 label="Style-match source to scene (Gemini Nano Banana — needs GEMINI_API_KEY)")
                run_btn = gr.Button("④ Execute", variant="primary", size="lg")

        status = gr.Textbox(label="Status", interactive=False)
        stylized_preview = gr.Image(label="Style-matched source (Gemini output)", interactive=False, type="filepath")
        result = gr.Video(label="Result", interactive=False)

        run_btn.click(
            _run,
            inputs=[source, target, face_index, face_restore, detect_gender,
                    enable_interp, enable_face_boost, enable_style_match],
            outputs=[status, stylized_preview, result],
        )

    return demo
