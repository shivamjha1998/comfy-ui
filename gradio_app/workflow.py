"""Load workflow JSON templates and patch them with user parameters.

Templates live as JSON files under WORKFLOWS_DIR. Each file has a top-level
'_meta' block (which we strip before sending to ComfyUI) containing a
'patch_targets' map for documentation, plus '__PATCH_*__' sentinels in the
node graph that get substituted by the patcher.

The patcher is intentionally explicit (one method per workflow) rather than
generic — workflows have very different parameter sets and the spec already
treats them as separate first-class citizens.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .config import CONFIG


WORKFLOW_REGISTRY = {
    "wf_a_reactor": "WF-A: ReActor Face Swap (face only, fast, long videos)",
    "wf_b_wan22":   "WF-B: WAN 2.2 Animate (full body, short clips, lip-sync)",
}


class WorkflowError(RuntimeError):
    pass


@dataclass
class WfAParams:
    source_image: str           # uploaded filename ref
    target_video: str
    face_index: int = 0
    face_restore_strength: float = 0.85
    detect_gender: str = "no"   # "no" | "male" | "female"
    output_codec: str = "h264_nvenc"
    enable_frame_interp: bool = True


@dataclass
class WfBParams:
    reference_image: str
    target_video: str
    prompt: str = ""
    relight_lora_weight: float = 0.6
    steps: int = 40
    segment_length_frames: int = 81  # ~5 sec @ 16fps; max recommended
    target_width: int = 1280   # set by ui.py based on uploaded video aspect ratio
    target_height: int = 720


def video_dims_for_target(local_path: str, prefer_720p: bool = True) -> tuple[int, int]:
    """Pick a WAN-trained resolution preset matching the uploaded video's aspect.

    WAN 2.2 Animate is trained at multi-resolution; landscape 1280x720 / portrait
    720x1280 give best quality, 832x480 / 480x832 are faster but slightly softer.
    Returns (width, height) snapped to one of these. Falls back to 1280x720 if
    the file can't be probed.
    """
    try:
        import cv2  # opencv-python-headless ships in the container
        cap = cv2.VideoCapture(local_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        cap.release()
    except Exception:
        w, h = 0, 0

    if w == 0 or h == 0:
        return (1280, 720) if prefer_720p else (832, 480)

    aspect = w / h
    if prefer_720p:
        if aspect >= 1.3:
            return (1280, 720)
        if aspect <= 0.77:
            return (720, 1280)
        return (768, 768)
    else:
        if aspect >= 1.3:
            return (832, 480)
        if aspect <= 0.77:
            return (480, 832)
        return (640, 640)


def load_template(workflow_id: str) -> dict[str, Any]:
    if workflow_id not in WORKFLOW_REGISTRY:
        raise WorkflowError(f"unknown workflow id: {workflow_id}")

    path = CONFIG.workflows_dir / f"{workflow_id}.json"
    if not path.is_file():
        raise WorkflowError(f"workflow template not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _strip_meta(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the graph with the top-level _meta block removed.

    ComfyUI accepts top-level keys other than node IDs in some versions but
    rejects them in others — easier to always strip.
    """
    out = copy.deepcopy(graph)
    out.pop("_meta", None)
    # Also strip per-node _meta titles — purely cosmetic for the UI graph view
    # and ComfyUI's API mode doesn't care about them either way.
    for node in out.values():
        if isinstance(node, dict):
            node.pop("_meta", None)
    return out


def _set(graph: dict[str, Any], node_id: str, field: str, value: Any) -> None:
    """Helper that fails loudly if a node or input field is missing."""
    if node_id not in graph:
        raise WorkflowError(f"node {node_id} not in graph")
    if "inputs" not in graph[node_id]:
        raise WorkflowError(f"node {node_id} has no 'inputs' block")
    if field not in graph[node_id]["inputs"]:
        raise WorkflowError(f"node {node_id} has no input '{field}' (did the template change?)")
    graph[node_id]["inputs"][field] = value


def patch_wf_a(template: dict[str, Any], params: WfAParams) -> dict[str, Any]:
    g = _strip_meta(template)
    _set(g, "1", "video", params.target_video)
    _set(g, "2", "image", params.source_image)
    _set(g, "3", "input_faces_index", str(params.face_index))
    _set(g, "3", "detect_gender_input", params.detect_gender)
    # ReActor's codeformer_weight is inverted from intuition: 0 = max restoration,
    # 1 = preserve raw inswapper output. Invert so the UI slider reads naturally
    # ("higher = smoother face").
    _set(g, "3", "codeformer_weight", 1.0 - float(params.face_restore_strength))
    # If frame interp is on, RIFE 2× doubles the frame count → bump the output
    # frame rate accordingly so playback is real-time, not slow-motion.
    # If interp is disabled, rewire VideoCombine to read directly from the
    # ReActor output (node 3) instead of RIFE (node 5), and drop node 5.
    if params.enable_frame_interp:
        g["6"]["inputs"]["frame_rate"] = 60
    else:
        g["6"]["inputs"]["images"] = ["3", 0]
        g["6"]["inputs"]["frame_rate"] = 30
        g.pop("5", None)
    return g


def patch_wf_b(template: dict[str, Any], params: WfBParams) -> dict[str, Any]:
    g = _strip_meta(template)
    # Video + segment length — keep VHS_LoadVideo and WanVideoAnimateEmbeds in sync.
    _set(g, "1", "video", params.target_video)
    _set(g, "1", "frame_load_cap", int(params.segment_length_frames))
    _set(g, "14", "num_frames", int(params.segment_length_frames))
    # Reference character image.
    _set(g, "2", "image", params.reference_image)
    # Positive prompt → WanVideoTextEncodeCached (node 10, kijai wrapper).
    _set(g, "10", "positive_prompt", params.prompt or "")
    # Relight LoRA weight — slot 0 of WanVideoLoraSelectMulti (node 12).
    _set(g, "12", "strength_0", float(params.relight_lora_weight))
    # Diffusion steps — WanVideoSampler (node 15).
    _set(g, "15", "steps", int(params.steps))
    # Resolution — patch every node that has explicit width/height so the whole
    # pipeline (load → pose detection → pose draw → animate embeds) uses the
    # same canvas. Auto-derived from input video aspect by ui.py.
    w, h = int(params.target_width), int(params.target_height)
    _set(g, "1",  "custom_width",  w)
    _set(g, "1",  "custom_height", h)
    _set(g, "4",  "width",  w)
    _set(g, "4",  "height", h)
    _set(g, "5",  "width",  w)
    _set(g, "5",  "height", h)
    _set(g, "14", "width",  w)
    _set(g, "14", "height", h)
    return g