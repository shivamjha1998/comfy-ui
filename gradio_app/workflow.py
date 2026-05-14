"""Load workflow JSON templates and patch them with user parameters.

Templates live as JSON files under WORKFLOWS_DIR. Each file has a top-level
'_meta' block (which we strip before sending to ComfyUI) containing a
'patch_targets' map for documentation, plus '__PATCH_*__' sentinels in the
node graph that get substituted by the patcher.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .config import CONFIG


WORKFLOW_REGISTRY = {
    "wf_a_reactor": "WF-A: ReActor Face Swap (face only, fast, long videos)",
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
    enable_frame_interp: bool = False
    # ReActor's per-face-crop upscale pass. 0.0 = off entirely; >0 = enabled at
    # that blend visibility (1.0 = full boost replace, 0.1 = barely-perceptible
    # sharpening blended over the raw swap). 0.5 is a safe default.
    face_boost_strength: float = 0.5


def load_template(workflow_id: str) -> dict[str, Any]:
    if workflow_id not in WORKFLOW_REGISTRY:
        raise WorkflowError(f"unknown workflow id: {workflow_id}")

    path = CONFIG.workflows_dir / f"{workflow_id}.json"
    if not path.is_file():
        raise WorkflowError(f"workflow template not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _strip_meta(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the graph with top-level + per-node _meta blocks removed."""
    out = copy.deepcopy(graph)
    out.pop("_meta", None)
    for node in out.values():
        if isinstance(node, dict):
            node.pop("_meta", None)
    return out


def _set(graph: dict[str, Any], node_id: str, field: str, value: Any) -> None:
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
    # Face restoration toggle: strength <= 0 means "off". On a 16 GB Mac, GFPGAN
    # restoration is fine for short clips (~10 s) but eats too much unified memory
    # for longer clips and either silently writes zeros or OOM-kills ComfyUI.
    # Users who upload longer clips should slide this to 0.
    if float(params.face_restore_strength) <= 0.0:
        _set(g, "3", "face_restore_model", "none")
    else:
        _set(g, "3", "codeformer_weight", 1.0 - float(params.face_restore_strength))
    # Face boost: 0 = off; >0 = enabled, with the slider value as the blend
    # visibility (1.0 = full boost replace, lower = blend with raw inswapper
    # to keep edges natural).
    boost_strength = max(0.0, min(1.0, float(params.face_boost_strength)))
    if boost_strength <= 0.0:
        _set(g, "10", "enabled", False)
    else:
        _set(g, "10", "enabled", True)
        # Node's visibility field has min=0.1; clamp accordingly.
        _set(g, "10", "visibility", max(0.1, boost_strength))
    # If frame interp is on, RIFE 2× doubles frame count → bump output frame rate
    # so playback stays real-time. If interp is disabled, rewire VideoCombine to
    # read directly from ReActor (node 3) and drop the RIFE node entirely.
    if params.enable_frame_interp:
        g["6"]["inputs"]["frame_rate"] = 60
    else:
        g["6"]["inputs"]["images"] = ["3", 0]
        g["6"]["inputs"]["frame_rate"] = 30
        g.pop("5", None)
    return g
