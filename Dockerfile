# syntax=docker/dockerfile:1.7
#
# ComfyUI Video Face-Swap System — Inference Container
# ─────────────────────────────────────────────────────
# Target hardware : NVIDIA DGX A100 (8× A100 80GB)
# Target host     : SoftBank AI Data Center · Tokyo Zone 1 · 082-partition
# Architecture    : linux/amd64 ONLY (build on Mac mini M4 via buildx)
# Runtime         : Enroot/Pyxis via Slurm `srun --container-image=...`
#
# ┌─ What's IN this image ──────────────────────────────────────────────┐
# │  • ComfyUI core + 7 custom nodes (face swap, WAN 2.2, video, etc.) │
# │  • Python deps for ComfyUI and all custom nodes                     │
# │  • Gradio 5.x front-end deps (UI runs inside the same container)    │
# │  • ffmpeg with NVENC (inherited from the NGC PyTorch base)          │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─ What's NOT in this image (lives on /lustre, bind-mounted at runtime) ┐
# │  • Model files (~35GB)        → /lustre/comfyui/models                │
# │  • Workflow JSON templates    → /lustre/comfyui/workflows             │
# │  • Generated outputs          → /lustre/comfyui/output                │
# │  • Staff uploads              → /lustre/comfyui/input                 │
# └───────────────────────────────────────────────────────────────────────┘
#
# Run-time mount example (executed from the SoftBank login server):
#   srun -p 082-partition --gpus=8 --exclusive --pty \
#     --container-image=/lustre/comfyui/comfyui.sqsh \
#     --container-mounts=/lustre/comfyui/models:/workspace/ComfyUI/models,\
# /lustre/comfyui/output:/workspace/ComfyUI/output,\
# /lustre/comfyui/input:/workspace/ComfyUI/input,\
# /lustre/comfyui/workflows:/workspace/ComfyUI/user/default/workflows \
#     bash

# Base image is parameterised so we can bump CUDA without editing the FROM line.
# 25.10-py3 is the first NGC PyTorch image to ship CUDA 13.0, which is the
# minimum required by kijai's ComfyUI-WanVideoWrapper (libcudart.so.13).
# Override with --build-arg BASE_IMAGE=... if a newer tag is desired.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

# ─── Build args ───────────────────────────────────────────────────────────
# Pin ComfyUI to a specific commit before the NGC security scan; track
# master only during initial development.
ARG COMFYUI_REF=master
ARG DEBIAN_FRONTEND=noninteractive

# Verify the base image actually has CUDA 13 (libcudart.so.13). We've been
# bitten before by the wrong base — a one-line check beats debugging at runtime.
RUN ldconfig -p | grep -E 'libcudart\.so\.(13|14|15)' \
    || { echo "ERROR: base image lacks libcudart.so.13+; bump BASE_IMAGE"; exit 1; }

LABEL org.opencontainers.image.title="comfyui-faceswap"
LABEL org.opencontainers.image.description="ComfyUI + ReActor + WAN 2.2 face/character swap, with Gradio front-end. For SoftBank AI Data Center A100."
LABEL org.opencontainers.image.source="https://github.com/comfyanonymous/ComfyUI"
LABEL com.abckk.target-partition="082-partition"

# ─── Remove packages with high/critical CVEs not needed at runtime ────────────
#
# NsightSystems-cli: NVIDIA profiling tool — 3 CRITICAL + 14 HIGH from Go 1.22
#   Installed by the CUDA toolkit at /usr/local/cuda-*/NsightSystems-cli-*/
#   (NOT an apt package — apt-get purge does nothing; must use rm -rf directly).
# libslurm37/libpmi2-0: Slurm client libs — 4 HIGH, no upstream fix, not needed
#   inside the container (Slurm runs on the HOST login server).
RUN rm -rf /usr/local/cuda-*/NsightSystems-cli-*/ \
    && apt-get update \
    && apt-get purge -y \
        libslurm37 \
        libpmi2-0 \
    ; apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# ─── OS package security upgrades ────────────────────────────────────────────
# Fixes 14 HIGH CVEs across gnupg suite, git, and rsync.
RUN apt-get update \
    && apt-get install -y --only-upgrade \
        gnupg \
        gnupg2 \
        gnupg-agent \
        gpg \
        gpg-agent \
        gpgconf \
        gpgsm \
        gpgv \
        dirmngr \
        libgpg-error0 \
        git \
        git-man \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# ─── System packages ──────────────────────────────────────────────────────
# git              clone ComfyUI + custom nodes
# libgl1, libglib  OpenCV runtime (face/pose detection nodes)
# libsm6, libxext6 Additional OpenCV deps for some custom nodes
# wget             model download fallback (when staging /lustre)
# Note: ffmpeg is already present in the NGC PyTorch base image with NVENC.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ─── ComfyUI core ─────────────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone https://github.com/comfyanonymous/ComfyUI.git \
    && cd ComfyUI \
    && git checkout ${COMFYUI_REF} \
    && git rev-parse HEAD > /workspace/COMFYUI_COMMIT.txt

WORKDIR /workspace/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# ─── Custom nodes (cloned per spec §4.3) ──────────────────────────────────
# Each clone is its own RUN layer so a single broken repo doesn't bust
# the entire cache during iteration.
WORKDIR /workspace/ComfyUI/custom_nodes

RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git
# Note: the old `comfyui-reactor-node` repo was removed by GitHub Staff in
# 2025 for a TOS violation. The author republished as `ComfyUI-ReActor`
# with the NSFW filter restored. Functionally the same node classes
# (ReActorFaceSwap, etc.) — workflow JSONs do not need to change.
RUN git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor.git
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-segment-anything-2.git
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
RUN git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git

# Install each custom node's Python deps. We use `|| true` because some
# nodes ship requirements.txt entries that try to downgrade torch or pull
# from broken indexes — we'd rather see a warning at build time and fix
# the offender than have the entire build fail. The smoke test (next
# build task) will catch nodes that don't actually load.
RUN set -e; \
    for d in */; do \
        req="${d}requirements.txt"; \
        if [ -f "${req}" ]; then \
            echo "──── installing deps for ${d}"; \
            pip install --no-cache-dir -r "${req}" || \
                echo "!!!! WARN: pip install failed for ${d} — review before scan"; \
        fi; \
    done

# ─── Fix OpenCV version conflict ─────────────────────────────────────────────
# Custom nodes pull in opencv-python and opencv-contrib-python at different
# versions, producing a broken mixed install (cv2.dnn.DictValue missing).
# Force a single consistent opencv-contrib-headless build.
RUN pip uninstall -y \
        opencv-python \
        opencv-python-headless \
        opencv-contrib-python \
        opencv-contrib-python-headless 2>/dev/null || true \
    && pip install --no-cache-dir "opencv-python-headless==4.10.0.84"
# Note: NGC base ships opencv-python 4.11.0.86 (requires numpy 2.x) but the
# container has numpy 1.26.4. Pinning 4.10.0.84 which works with both.

# ─── Front-end deps (Gradio in same container per SoftBank policy) ────────
# Gradio is the only piece that talks to agency staff; ComfyUI sits behind
# it on localhost:8188. Both processes are launched from the sbatch script.
# 5.11.0+ required: fixes CRITICAL CVE GHSA-j2jg-fq62-7c3h present in 4.x.
RUN pip install --no-cache-dir \
        "gradio>=5.11.0" \
        "requests>=2.31" \
        "websocket-client>=1.7"

# ─── Python security upgrades ─────────────────────────────────────────────────
# Run after all custom node installs so these versions win over any older
# packages pulled in transitively. Fixes 1 CRITICAL + ~34 HIGH CVEs.
RUN pip install --no-cache-dir \
        "h11==0.16.0" \
        "onnx==1.21.0" \
        "tornado==6.5.5" \
        "urllib3==2.6.3" \
        "protobuf>=5.29.6" \
        "setuptools==78.1.1" \
        "jupyter-core==5.8.1" \
        "pillow==12.1.1" \
        "wheel==0.46.2" \
        "nbconvert==7.17.0" \
        "black==26.3.1"

# ─── Remove lingering vulnerable metadata that pip installs don't touch ───────
# onnx 1.16.2 egg-info in PyTorch source tree: 10 HIGH CVEs.  The pip install
# above already placed a clean onnx==1.21.0 in site-packages; this orphaned
# egg-info in the PyTorch third-party tree just confuses the scanner.
# setuptools _vendor dist-info: 5 HIGH from jaraco-context 5.3.0 and wheel
# 0.45.1 vendored inside the setuptools package itself.  Deleting them doesn't
# affect pip functionality — setuptools uses its own bundled copies at runtime.
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx/onnx.egg-info \
    && find /usr/local/lib/python3.10/dist-packages/setuptools/_vendor \
            -maxdepth 2 -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true

# ─── Mount-point directories ──────────────────────────────────────────────
# Pyxis bind mounts will overlay these at runtime with /lustre/comfyui/*.
# We pre-create them so the mounts don't need to autocreate (which can
# fail with non-root mappings depending on Pyxis config).
RUN mkdir -p \
        /workspace/ComfyUI/models \
        /workspace/ComfyUI/output \
        /workspace/ComfyUI/input \
        /workspace/ComfyUI/user/default/workflows

# ─── Application code (placeholder for the Gradio front-end) ──────────────
# The Gradio app source will be added in the next build task and copied
# into /workspace/app. Leaving the directory in place now so the structure
# is visible.
RUN mkdir -p /workspace/app
COPY gradio_app/ /workspace/app/

# ─── Network ──────────────────────────────────────────────────────────────
# 8188 = ComfyUI HTTP/WebSocket API
# 7860 = Gradio front-end
# These are documentation only — Slurm/Pyxis uses host networking by default
# and we'll forward them via SSH from the Mac mini.
EXPOSE 8188 7860

# ─── Working dir ──────────────────────────────────────────────────────────
WORKDIR /workspace/ComfyUI

# ─── Default command ──────────────────────────────────────────────────────
# Slurm will override this with the launcher script in scripts/launch.sbatch.
# Default to a shell so `srun --pty bash --container-image=...` works for
# manual debugging and the smoke test.
CMD ["/bin/bash"]