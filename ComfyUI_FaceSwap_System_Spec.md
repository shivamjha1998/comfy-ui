# ComfyUI Video Face-Swap System
**Task Specification & Technical Reference**

| Field | Value |
|---|---|
| **Client** | VTuber Talent Agency |
| **Date** | 2026-04-07 |
| **Status** | Draft |
| **Infrastructure** | NVIDIA H100 GPU Server (80 GB VRAM, CUDA-enabled, Linux) |
| **Primary UI** | ComfyUI + Gradio Simplified Front-End |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Rationale for ComfyUI](#2-rationale-for-comfyui)
3. [System Architecture](#3-system-architecture)
4. [Environment Setup](#4-environment-setup)
5. [Workflow Build Tasks](#5-workflow-build-tasks)
6. [Simplified Front-End Build](#6-simplified-front-end-build)
7. [Operations & Maintenance](#7-operations--maintenance)
8. [Technical Concerns & Constraints](#8-technical-concerns--constraints)
9. [Definition of Done](#9-definition-of-done)
10. [Reference Resources](#10-reference-resources)

---

## 1. Project Overview

### Objective

Build a system that replaces performers' faces (or full-body characters) in video footage with designated alternatives, operable by agency staff through a Web UI — without requiring knowledge of AI tooling.

### Deliverables

- Pre-built face-swap and character-replacement workflows running on ComfyUI
- A simplified Gradio-based front-end for non-technical agency staff

### Stakeholder Summary

| Stakeholder | Role |
|---|---|
| VTuber Talent Agency | Client — end user of the system |
| Agency Staff | Day-to-day operators via Gradio UI |
| Developer | Builds and maintains the system |

---

## 2. Rationale for ComfyUI

ComfyUI is a node-based generative-AI workflow engine. The table below justifies its selection as the primary platform.

| Advantage | Details |
|---|---|
| **Unified Platform** | ReActor (face swap) and WAN 2.2 Animate (full-body replacement) both run in the same UI — no need to maintain separate interfaces per engine. |
| **Rich Custom-Node Ecosystem** | Hundreds of custom nodes installable via ComfyUI Manager in one click: face detection, segmentation, super-resolution, frame interpolation, and more. |
| **Workflow Portability** | An entire workflow is stored as a single JSON file, enabling easy sharing, version control, and reproducibility. |
| **API-First Architecture** | ComfyUI exposes an HTTP/WebSocket API server internally. Submitting a JSON payload to `/prompt` triggers execution — simple to wrap with a lightweight front-end. |
| **Real-Time Progress** | Execution events are streamed via WebSocket (`/ws`), enabling per-node status visualization in real time. |
| **Community Track Record** | Proven face-swap workflows already exist: WAN 2.2 Animate Face Swap, ReActor + RIFE Face Swap, etc. |
| **Extensibility** | Future features (LoRA application, style transfer, lip-sync) can be added simply by inserting new nodes — no code changes required. |

---

## 3. System Architecture

### 3-1. Overall Structure

The system follows a three-layer architecture. ComfyUI serves as the back-end inference engine, with a thin Gradio layer for agency staff.

```
┌─────────────────────────────────────────────────────────┐
│                    NGINX (HTTPS + Auth)                  │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │   Gradio Front-End  │    │    ComfyUI Server    │
    │      port 7860      │◄──►│      port 8188       │
    │  (Python wrapper)   │    │   (Python core)      │
    └─────────────────────┘    └──────────┬───────────┘
                                          │
                               ┌──────────▼───────────┐
                               │     GPU Worker        │
                               │  H100 80 GB VRAM      │
                               │  PyTorch + ONNX RT    │
                               │  + TensorRT           │
                               └──────────────────────┘
```

| Layer | Component | Technology Stack |
|---|---|---|
| **Front-End** | Simplified Operation UI | Gradio 4.x (Python). 4-step flow: upload → select workflow → execute → download. |
| **Back-End** | ComfyUI Server | ComfyUI core (Python). Listens on HTTP port 8188. Custom nodes pre-installed. |
| **Inference Engine** | GPU Worker | PyTorch + ONNX Runtime + TensorRT on H100 80 GB. ReActor / WAN 2.2 Animate nodes perform actual processing. |

---

### 3-2. Workflow Configuration (Dual-Track)

Two workflows are built and selectable from the front-end depending on the use case.

| Workflow | Use Case | Core Node Pipeline |
|---|---|---|
| **WF-A: ReActor Face Swap** | Face-only replacement. Handles long videos. Fast processing. | `LoadVideo` → `ReActorFaceSwap (inswapper_128)` → `GFPGAN` → `RIFE` → `MergeAudio` → `SaveVideo` |
| **WF-B: WAN 2.2 Animate Replace** | Full-body character replacement. Auto scene-lighting adaptation. Lip-sync capable. | `LoadVideo` → `PoseAndFaceDetection (YOLO+ViTPose)` → `Sam2Segmentation` → `WanVideoAnimateEmbeds` → `WanVideoSampler` → `WanVideoDecode` → `SaveVideo` |

---

### 3-3. Processing Flow

```
Staff                   Gradio UI               ComfyUI              H100 GPU
  │                        │                       │                     │
  │── Upload face image ──►│                       │                     │
  │── Upload target video ►│                       │                     │
  │                        │── POST /upload/image ►│                     │
  │── Select workflow ─────►│                       │                     │
  │── Set parameters ──────►│                       │                     │
  │── Click Execute ───────►│                       │                     │
  │                        │── POST /prompt ───────►│                     │
  │                        │                       │── Inference ────────►│
  │◄── Progress (WebSocket)─┤◄── /ws events ────────┤◄─── Progress ───────│
  │                        │                       │                     │
  │                        │◄── /history/{id} ─────┤◄─── Done ───────────│
  │◄── Download result ────┤── GET /view?filename ─►│                     │
```

**Step-by-step:**

1. Agency staff uploads a source face image and target video via the simplified front-end.
2. Staff selects a workflow (WF-A: Face Swap / WF-B: Full-Body Replacement).
3. The front-end uploads files via ComfyUI's `/upload/image` API.
4. Workflow JSON template parameters (filenames, thresholds, etc.) are dynamically rewritten and POSTed to `/prompt`.
5. Processing progress is received in real time via WebSocket (`/ws`) and displayed on the front-end.
6. Upon completion, output file info is retrieved from `/history/{prompt_id}`, and the result video is made downloadable via `/view`.

---

## 4. Environment Setup

### 4-1. Base OS & Drivers

| Task | Details / Commands |
|---|---|
| **OS Verification** | Ubuntu 22.04 LTS or 24.04 LTS recommended |
| **NVIDIA Driver** | `nvidia-driver-550` or later. Verify H100 recognition and 80 GB VRAM: `nvidia-smi` |
| **CUDA Toolkit** | CUDA 12.4+. Verify with: `nvcc --version` |
| **cuDNN** | Install cuDNN 9.x via conda or apt |
| **FFmpeg** | `apt install ffmpeg` — NVENC-enabled build recommended for fast video encoding |
| **Python** | Python 3.10 or 3.11 (ComfyUI recommended versions) |

---

### 4-2. ComfyUI Core

```bash
# 1. Clone repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2. Create virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt

# 5. Start and verify
python main.py --listen 0.0.0.0 --port 8188
# → Confirm Web UI loads in browser at http://<server-ip>:8188

# 6. Install ComfyUI Manager
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

---

### 4-3. Custom Node Installation

Install via ComfyUI Manager or manual `git clone` into the `custom_nodes/` directory.

| Custom Node | Purpose | Repository |
|---|---|---|
| `comfyui-reactor-node` | ReActor face swap (inswapper_128-based) | `github.com/Gourieff/comfyui-reactor-node` |
| `ComfyUI-WanVideoWrapper` | WAN 2.2 Animate/Replace inference nodes | `github.com/kijai/ComfyUI-WanVideoWrapper` |
| `ComfyUI-WanAnimatePreprocess` | Pose detection & face cropping pre-processing | `github.com/kijai/ComfyUI-WanAnimatePreprocess` |
| `ComfyUI-segment-anything-2` | SAM2-based person segmentation | `github.com/kijai/ComfyUI-segment-anything-2` |
| `ComfyUI-VideoHelperSuite` | Video loading, export & audio merging | `github.com/Kosinkadink/ComfyUI-VideoHelperSuite` |
| `ComfyUI-Frame-Interpolation` | RIFE frame interpolation (smoothing) | `github.com/Fannovel16/ComfyUI-Frame-Interpolation` |
| `ComfyUI-GGUF` | GGUF model loading (for VRAM-saving runs) | `github.com/city96/ComfyUI-GGUF` |

```bash
# Example: manual clone of all custom nodes
cd ComfyUI/custom_nodes

git clone https://github.com/Gourieff/comfyui-reactor-node
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess
git clone https://github.com/kijai/ComfyUI-segment-anything-2
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
git clone https://github.com/city96/ComfyUI-GGUF
```

---

### 4-4. Model Downloads

| Model | Approx. Size | Destination Path |
|---|---|---|
| `inswapper_128.onnx` | ~500 MB | `ComfyUI/models/insightface/` |
| `GFPGANv1.4.pth` | ~350 MB | `ComfyUI/models/facerestore_models/` |
| `retinaface_resnet50.pth` | ~100 MB | `ComfyUI/models/insightface/` |
| `Wan2.2-Animate-14B` (diffusion model) | ~30 GB | `ComfyUI/models/wan/` |
| `Wan2.2 VAE` | ~300 MB | `ComfyUI/models/vae/` |
| `Wan2.2 CLIP Vision` | ~1.5 GB | `ComfyUI/models/clip_vision/` |
| `Wan2.2 Relight LoRA` | ~1 GB | `ComfyUI/models/loras/` |
| `Lightx2v Acceleration LoRA` *(optional)* | ~500 MB | `ComfyUI/models/loras/` |
| `YOLOv10m` (person detection) | ~50 MB | `ComfyUI/models/yolo/` |
| `RealESRGAN_x4plus.pth` *(optional)* | ~64 MB | `ComfyUI/models/upscale_models/` |
| `RIFE` (frame interpolation) | ~100 MB | `ComfyUI/models/rife/` |

> **Total:** ~35+ GB of model data. Ensure adequate disk space before downloading.

---

## 5. Workflow Build Tasks

### 5-1. WF-A: ReActor Face Swap

Primary workflow for **face-only replacement** in existing videos. Best suited for long-form content and batch processing.

#### Node Configuration

| # | Node | Settings / Notes |
|---|---|---|
| 1 | **Load Video (VHS)** | Loads target video. Auto-detects frame rate and resolution. |
| 2 | **Load Image** | Loads source face image (the replacement face). |
| 3 | **ReActorFaceSwap** | Executes face swap using `inswapper_128` model. Configurable `detect_gender` setting. `face_index` parameter selects target among multiple people. |
| 4 | **FaceRestore (GFPGAN)** | Applies GFPGANv1.4 to restore/enhance post-swap face quality. Adjustable `strength` parameter (0.5–1.0 recommended). |
| 5 | **RIFE VFI** *(optional)* | Frame interpolation for smoother output. Default 2× multiplier. Trade-off with processing time. |
| 6 | **Video Combine (VHS)** | Re-combines frames and merges original audio track. Codec: H.264/H.265 selectable. NVENC recommended. |
| 7 | **Save Output** | Saves to `ComfyUI/output/` with timestamped filename. |

#### Exposed Parameters (modifiable via API)

| Parameter | Type | Description |
|---|---|---|
| `source_image` | string (path) | File path of the replacement face image |
| `target_video` | string (path) | File path of the target video to process |
| `face_index` | int (0-based) | Index of the target person among multiple detected faces |
| `face_restore_strength` | float (0.0–1.0) | GFPGAN face restoration intensity |
| `detect_gender` | enum (`no` / `male` / `female`) | Gender filter for face selection |
| `output_codec` | enum (`h264_nvenc` / `hevc_nvenc`) | Output video codec |
| `enable_frame_interp` | bool | RIFE frame interpolation ON/OFF toggle |

---

### 5-2. WF-B: WAN 2.2 Animate Replace

Full-body character replacement workflow. Supports scene-lighting adaptation and lip-sync.
Optimized for **short clips (max ~30 seconds)**.

#### Node Configuration

| # | Node | Settings / Notes |
|---|---|---|
| 1 | **Load Video (VHS)** | Loads target video. Recommended: segment into 1280×720, 81 frames (~5 sec + 1 frame) units. |
| 2 | **Load Image (Reference Character)** | Loads replacement character image. Use front-facing, clearly visible face. |
| 3 | **PoseAndFaceDetection** | Extracts body keypoints and face crops via YOLO + ViTPose. Foundation for lip-sync tracking. |
| 4 | **Sam2Segmentation** | Generates foreground (person) mask via SAM2. Essential for background preservation. |
| 5 | **WanVideoLoraSelectMulti** | Mixes Lightx2v (acceleration) + Wan22 Relight (lighting) LoRAs. Adjust weights for quality/speed balance. |
| 6 | **WanVideoAnimateEmbeds** | Fuses reference image + pose + face crops + background + mask. Core node for identity preservation. |
| 7 | **WanVideoSampler** | Executes diffusion inference. Adjust step count and scheduler for quality. 30–50 steps recommended on H100. |
| 8 | **WanVideoDecode** | Decodes latent representations into video frames. |
| 9 | **Video Combine (VHS)** | Combines frames and merges audio. If segmented, concatenates multiple outputs. |

#### Exposed Parameters (modifiable via API)

| Parameter | Type | Description |
|---|---|---|
| `reference_image` | string (path) | File path of replacement character image |
| `target_video` | string (path) | File path of the target video to process |
| `prompt` | string | Text description of video content (auto-generated via VLM or manual input) |
| `relight_lora_weight` | float | Relighting LoRA strength (degree of lighting adaptation) |
| `accel_lora_weight` | float | Lightx2v acceleration LoRA strength (speed vs. quality trade-off) |
| `steps` | int (20–50) | Diffusion sampling step count |
| `segment_length` | int (seconds) | Video segment split length. Max 30 sec recommended. |

---

## 6. Simplified Front-End Build

ComfyUI's node-based UI is designed for technical users. A Gradio-based front-end wraps the ComfyUI API for agency staff.

### 6-1. Screen Layout

| Screen / Component | Functional Requirements |
|---|---|
| **Workflow Selector** | Dropdown to switch between 'Face Swap (WF-A)' and 'Full-Body Replacement (WF-B)'. Parameter panel updates dynamically based on selection. |
| **Source Image Upload** | Drag-and-drop upload for the replacement face/character image. Preview display. |
| **Target Video Upload** | Upload target video. Includes preview playback functionality. |
| **Parameter Panel** | Exposes each workflow's parameters as sliders/dropdowns (see Section 5 for parameter definitions). |
| **Execute Button** | Submits job to ComfyUI `/prompt` API. Loading indicator to prevent double-submission. |
| **Progress Display** | Real-time progress bar via WebSocket. Shows currently executing node name and step count. |
| **Result Preview** | In-browser video playback of processed output. Before/after comparison view. |
| **Download** | Download button for processed video. |
| **Job History** | List of past processing results fetched from ComfyUI `/history` API. Re-download and parameter review available. |

---

### 6-2. Technical Implementation Guidelines

| Item | Approach |
|---|---|
| **Framework** | Gradio 4.x (Python-based, minimal setup). Future migration to Streamlit or Next.js possible. |
| **ComfyUI API Integration** | Hold workflow JSON templates on the Python side. Dynamically rewrite parameters based on user input and POST to `/prompt`. |
| **File Upload** | Use ComfyUI `/upload/image` API. Inject uploaded filenames into workflow JSON. |
| **Progress Monitoring** | Subscribe to WebSocket (`ws://localhost:8188/ws`). Parse `executing` / `progress` / `complete` events. |
| **Result Retrieval** | `/history/{prompt_id}` → `/view?filename=...` to retrieve output video. |
| **Authentication** | Gradio auth (username/password) + Nginx Basic Auth. HTTPS required for external access. |
| **Deployment** | Gradio front-end: port 7860. ComfyUI back-end: port 8188. Both run on the same server. |

```python
# Example: Gradio <-> ComfyUI integration sketch
import gradio as gr
import requests, json, websocket

COMFYUI_URL = "http://localhost:8188"

def run_workflow(source_image, target_video, workflow_choice, **params):
    # 1. Upload files
    with open(source_image, "rb") as f:
        r = requests.post(f"{COMFYUI_URL}/upload/image", files={"image": f})
    uploaded_name = r.json()["name"]

    # 2. Load and patch workflow JSON template
    with open(f"templates/{workflow_choice}.json") as f:
        workflow = json.load(f)
    # ... patch node params ...

    # 3. Submit to /prompt
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    prompt_id = resp.json()["prompt_id"]

    # 4. Poll WebSocket for progress (simplified)
    # ws = websocket.WebSocket()
    # ws.connect(f"ws://localhost:8188/ws?clientId=...")
    # ... handle progress events ...

    # 5. Retrieve result
    history = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
    output_filename = history[prompt_id]["outputs"]["..."]["filename"]
    return f"{COMFYUI_URL}/view?filename={output_filename}"
```

---

## 7. Operations & Maintenance

### systemd Service Registration

Both services must auto-start on boot and auto-restart on crash.

```ini
# /etc/systemd/system/comfyui.service
[Unit]
Description=ComfyUI Inference Server
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/opt/ComfyUI
ExecStart=/opt/ComfyUI/venv/bin/python main.py --listen 0.0.0.0 --port 8188
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable comfyui gradio-frontend
sudo systemctl start comfyui gradio-frontend
```

---

### Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name <your-domain>;

    # HTTPS termination
    ssl_certificate     /etc/letsencrypt/live/<your-domain>/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/<your-domain>/privkey.pem;

    # Basic Auth
    auth_basic           "VTuber Agency Face-Swap System";
    auth_basic_user_file /etc/nginx/.htpasswd;

    # Gradio front-end
    location / {
        proxy_pass http://localhost:7860;
    }

    # WebSocket proxy for ComfyUI progress events
    location /ws {
        proxy_pass http://localhost:8188/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

### Operational Tasks Summary

| Item | Implementation Details |
|---|---|
| **GPU Monitoring** | `nvidia-smi`-based GPU utilization, VRAM usage, and temperature monitoring. Prometheus + Grafana dashboard recommended. |
| **Log Management** | ComfyUI logs (`--log-stdout`) + Gradio logs → `/var/log/`. `logrotate` with 7-day rotation. |
| **Storage Management** | Cron job to auto-delete processed files under `ComfyUI/output/` after 7 days. |
| **Backup** | Backup script for workflow JSON files, custom node configs, and model file list (paths + hashes). |
| **Workflow Updates** | Adding new workflows requires only placing a JSON file on the Gradio side. Zero code changes required. |
| **Dockerization (Future)** | Design with future Docker Compose migration in mind: `comfyui` + `gradio-frontend` + `nginx` containers. |

```bash
# Cron job: auto-delete output files older than 7 days
# Add to crontab with: crontab -e
0 3 * * * find /opt/ComfyUI/output -type f -mtime +7 -delete
```

---

## 8. Technical Concerns & Constraints

### 8-1. Quality Constraints

| Workflow | Known Limitations | Mitigation |
|---|---|---|
| **WF-A (ReActor)** | Quality degradation on profile views and occlusions (masks, hair, hands). Cannot guarantee perfect results. | Tune `face_restore_strength`; GFPGAN post-processing. Frames below threshold may require manual review. |
| **WF-B (WAN 2.2)** | Optimal output is 720p at ~5-second (81 frame) segments. Longer videos require segmentation → flicker at boundaries. | RIFE frame interpolation reduces but does not eliminate flicker. |
| **Long-form video (5+ min)** | WF-B is not designed for long videos. | Use WF-A for long-form content. Prepare a usage guideline document for agency staff. |

---

### 8-2. Performance Estimates (H100)

| Process | Estimated Speed | Notes |
|---|---|---|
| WF-A: ReActor face swap | 1080p / 60s → ~30–60 sec | Frame-level parallel processing |
| WF-A: + GFPGAN post-processing | Add ~15–30 sec | Face region only |
| WF-A: + RIFE interpolation | Add ~20–40 sec | At 2× interpolation |
| WF-B: WAN 2.2 Replace | 720p / 5s → ~2–5 min | 14B model, 30–50 steps |
| WF-B: + Lightx2v accel LoRA | ~50% reduction from above | Enables 4-step inference |

---

### 8-3. VRAM Usage Estimates

| Workflow | Est. VRAM Usage | H100 (80 GB) Headroom |
|---|---|---|
| WF-A: ReActor standalone | ~4–8 GB | ✅ Ample headroom |
| WF-A: + GFPGAN + RIFE | ~8–12 GB | ✅ Ample headroom |
| WF-B: WAN 2.2 14B model | ~40–60 GB | ⚠️ Operable. Moderate headroom. |
| WF-A + WF-B simultaneous | Not recommended | ❌ VRAM overflow risk. Use queue-based mutual exclusion. |

> **Important:** Implement a job queue with mutual exclusion between WF-A and WF-B. Do not allow concurrent workflow execution on a single H100.

---

### 8-4. ⚠️ Legal & Ethical Considerations

The following items **must be confirmed and actioned by the client** before development begins.

| Item | Details |
|---|---|
| **Performer Consent** | Establish a process for obtaining **written consent** from performers whose faces will be swapped. |
| **Terms of Use** | Draft system usage terms: abuse prevention, external sharing prohibition, etc. |
| **Output Management** | Define responsibilities, retention periods, and disposal rules for processed videos. |
| **Watermarking** | Determine whether invisible watermarks indicating AI-generated content are required. |
| **inswapper License** ⚠️ | The InsightFace `inswapper_128` model used by ReActor is licensed for **non-commercial use only**. A commercial license agreement with InsightFace may be required. **Confirm before deployment.** |
| **WAN 2.2 License** | Apache 2.0 — commercial use permitted. Confirm terms regarding usage of generated output content separately. |

---

## 9. Definition of Done

This task is considered complete when **all** of the following criteria are met.

| # | Acceptance Criterion |
|---|---|
| 1 | ComfyUI runs persistently on the H100 server at HTTP port 8188, and the node UI is accessible via browser. |
| 2 | **WF-A:** A test video (1080p / 30fps / 60s) face replacement completes successfully and outputs a video with audio. |
| 3 | **WF-B:** A test video (720p / 24fps / 10s) full-body character replacement completes successfully. |
| 4 | From the Gradio simplified front-end, the complete workflow of image/video upload → workflow selection → parameter configuration → execution → download functions correctly. |
| 5 | In a video with multiple people, a specific individual can be selected and swapped using the `face_index` parameter. |
| 6 | Processing progress is displayed in real time on the front-end via WebSocket. |
| 7 | Authentication is functional and unauthenticated users cannot access the system. |
| 8 | Both ComfyUI and Gradio are registered as systemd services and auto-start after server reboot. |
| 9 | Operational documentation is complete: setup procedures, user manual, troubleshooting guide, and workflow addition instructions. |

---

## 10. Reference Resources

| Resource | URL |
|---|---|
| ComfyUI Official | https://github.com/comfyanonymous/ComfyUI |
| ComfyUI API Documentation | https://docs.comfy.org/ |
| comfyui-reactor-node | https://github.com/Gourieff/comfyui-reactor-node |
| ComfyUI-WanVideoWrapper | https://github.com/kijai/ComfyUI-WanVideoWrapper |
| WAN 2.2 Official | https://github.com/Wan-Video/Wan2.2 |
| WAN 2.2 Face Swap Workflow Example | https://www.runninghub.ai/post/1953306162369392642 |
| WAN 2.2 Animate Face Swap WF Example | https://www.runninghub.ai/post/1970320770746490881 |
| ReActor + RIFE Workflow Example | https://comfyui.org/en/face-swap-revolution-with-reactor-and-rife |
| ComfyUI Production API Guide | https://www.viewcomfy.com/blog/building-a-production-ready-comfyui-api |
| InsightFace (inswapper model) | https://www.insightface.ai/ |
| GFPGAN | https://github.com/TencentARC/GFPGAN |
| comfy-pack (BentoML integration) | https://github.com/bentoml/comfy-pack |

---

*End of specification — Draft v1.0 · 2026-04-07*
