# Face-Swap Demo

Single-clip face swap on a Mac Mini M4 16 GB. Upload a source face + a target video, get back the video with the face replaced. Web UI via Gradio, pipeline via ComfyUI + [ReActor](https://github.com/Gourieff/ComfyUI-ReActor).

---

## Hardware

- Apple Silicon Mac (tested on M4 16 GB)
- macOS 14+ (Sonoma or later)
- ~15 GB free disk for models

Linux / CUDA deploy was built earlier; it's archived on the [`softbank-archive`](https://github.com/shivamjha1998/comfy-ui/tree/softbank-archive) branch and not maintained on `main`.

## Prerequisites

```bash
brew install python@3.11
```

That's all. macOS's default `python3` is 3.14, which is too new for the PyTorch / ComfyUI stack. `ffmpeg` is bundled by `imageio-ffmpeg`, and `git` ships with Xcode Command Line Tools.

## Install

```bash
git clone git@github.com:shivamjha1998/comfy-ui.git
cd comfy-ui
./setup.sh
```

`setup.sh` is idempotent — re-runs are no-ops once everything's in place. First run pulls ~2 GB of models and takes ~10 min on a decent connection.

## Run

```bash
./run.sh
```

Starts ComfyUI on `:8188` and Gradio on `http://127.0.0.1:7860`. Press `Ctrl+C` to stop both cleanly. To expose Gradio on the LAN (e.g. for a demo from a laptop):

```bash
GRADIO_HOST=0.0.0.0 ./run.sh
```

## How to use it

1. **Upload a source face image.** Any clear photo with a visible face. The collapsible *Source Face Index Preview* draws numbered boxes on every face it detects, so you can pick which one with the *Source face index* number input.
2. **Upload a target video** (up to 15 min). 1080p+ inputs are auto-downscaled to 720 p / 30 fps. *Target Face Index Preview* lets you pick which face in the scene gets replaced.
3. **Tune the sliders** (sweet spots below).
4. **Click Execute.** The pipeline chunks the video, swaps each piece, stitches the results losslessly.

### Slider sweet spots (M4 16 GB)

| Knob | Recommended | Notes |
|---|---|---|
| Face restore strength | **0.55–0.65** | 1.0 = porcelain-doll skin; 0 = soft inswapper output |
| Face boost strength | **0.5–0.7** | 1.0 = sharper but can warp face edges; 0 = blurry |
| RIFE frame interp | off by default | Doubles output framerate (smoother motion); no per-frame quality boost |

### Throughput

Wall-clock numbers measured on M4 16 GB, 720p / 30 fps inputs:

| Restoration | Chunk length | Time per chunk |
|---|---|---|
| off | 30 s | ~2 min |
| on  | 10 s | ~6 min |

- A 30-second clip with restoration ≈ 18 min
- A 1-minute clip with restoration ≈ 36 min
- A 10-minute clip with restoration ≈ 6 h (use restoration off for long clips)

## Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │  gradio_app (web UI on :7860)                                │
        │  ├── chunk-and-stitch loop (gradio_app/ui.py)                │
        │  ├── ffmpeg pre-process + lossless concat                    │
        │  └── HTTP+WS client to ComfyUI                               │
        └─────────────────────────────┬────────────────────────────────┘
                                      │
        ┌─────────────────────────────▼────────────────────────────────┐
        │  ComfyUI (on :8188)                                          │
        │  ├── workflows/wf_a_reactor.json                             │
        │  ├── VHS_LoadVideo  →  ReActorFaceSwap  →  ReActorFaceBoost  │
        │  │                        (inswapper_128                     │
        │  │                         + GFPGAN restoration              │
        │  │                         + buffalo_l face detect)          │
        │  ├── [optional] RIFE VFI                                     │
        │  └── VHS_VideoCombine                                        │
        └──────────────────────────────────────────────────────────────┘
```

Each upload goes through:

```
prepare    → ffmpeg re-encode to ≤1280×?@30 fps with dense keyframes
chunk      → ffmpeg segment, -c copy (no re-encode), ~10 s or ~30 s pieces
per-chunk  → upload to ComfyUI, run wf_a_reactor.json, download result
concat     → ffmpeg concat demuxer, -c copy (no re-encode)
```

Why chunking: the full IMAGE tensor for a long clip blows past 16 GB unified memory; chunking bounds per-job memory. Lossless concat means no extra quality cost.

## Mac-specific patches

`scripts/patch_reactor_mac.sh` runs on every `./run.sh` start. It enforces two patches on the `ComfyUI-ReActor` install that are needed on Apple Silicon:

1. **Split ONNX execution providers.** Upstream defaults to `CoreMLExecutionProvider` for everything; we split:
   - `analysis_providers = ["CPUExecutionProvider"]` — used by `buffalo_l` face detection, which CoreML mis-handles (silent empty detections).
   - `providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]` — used by `inswapper` / `hyperswap`, which run cleanly on CoreML and benefit from GPU/ANE.
2. **Skip `input_image.to(device)` on MPS.** Upstream blindly moves the whole video tensor to MPS at the top of `execute()`, but the very next call moves it back to CPU. The pointless allocation OOMs long clips on a 16 GB Mac.

Both patches are idempotent. No need to re-apply manually.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Mac M4 demo (this) |
| `softbank-archive` | Original SoftBank A100 setup — WAN 2.2 Animate full-body workflow, Docker, Slurm. Frozen; restore from here if redeploying on a Linux GPU box. |

## Known limits

- **Max input length: 15 min.** Hard cap in `gradio_app/ui.py`; longer uploads are rejected up front.
- **1080p+ inputs are auto-downscaled to 720 p.** On Mac, `pyav` and `cv2` each bundle their own FFmpeg dylibs; the clash silently produces zero-frame decode at 1080p inside ComfyUI. Downscaling sidesteps this.
- **GFPGAN restoration ≤ ~10 s chunks** on 16 GB Mac. Longer chunks OOM during restoration. The pipeline picks a safe chunk size automatically based on the restoration setting.
- **`inswapper_128` is non-commercial.** Fine for a demo; needs license review before any paid deployment.
- **Docker on Mac is not viable for this stack.** Docker Desktop on macOS runs Linux containers in a VM; Metal / CoreML / MPS do not cross the VM boundary, so the GPU acceleration we rely on disappears. If you ever need a containerized deploy, target a Linux/CUDA host using the `softbank-archive` Dockerfile as a starting point.

## Layout

```
.
├── setup.sh                         # one-time installer (this Mac)
├── run.sh                           # start ComfyUI + Gradio
├── README.md
├── gradio_app/                      # web UI + chunk-and-stitch orchestration
│   ├── __main__.py
│   ├── comfyui_client.py            # thin HTTP+WS wrapper
│   ├── config.py
│   ├── ui.py                        # the bulk of the app
│   └── workflow.py                  # WfAParams + patch_wf_a (workflow JSON patcher)
├── workflows/
│   └── wf_a_reactor.json            # ComfyUI graph template
├── scripts/
│   └── patch_reactor_mac.sh         # Mac-only ReActor fixes (idempotent)
├── ComfyUI/                         # cloned by setup.sh, gitignored
└── .venv/                           # gitignored
```
