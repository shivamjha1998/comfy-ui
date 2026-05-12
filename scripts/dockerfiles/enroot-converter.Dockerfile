# enroot-converter — a tiny Linux helper image with enroot installed
# ──────────────────────────────────────────────────────────────────
# Purpose: convert a Docker image (read from the host's Docker daemon
# via /var/run/docker.sock bind mount) into a SquashFS .sqsh file that
# can be scp'd to the SoftBank login server.
#
# Used by: scripts/convert_to_sqsh.sh
#
# This is a one-off workaround until proper NGC team registry access
# is available. It exists because:
#   - enroot is Linux-only (no macOS port)
#   - Personal NGC accounts have no Private Registry to push to
#   - The SoftBank login server can't import docker save tarballs
#
# We force linux/amd64 so the enroot binaries match the architecture
# of the SoftBank GPU server (also linux/amd64). On Apple Silicon
# this runs under Rosetta — slow, but only happens during conversion.

FROM --platform=linux/amd64 ubuntu:22.04

ARG ENROOT_VERSION=3.4.1
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        curl \
        ca-certificates \
        squashfs-tools \
        jq \
        parallel \
        zstd \
        pigz \
    && rm -rf /var/lib/apt/lists/*

# Pin to the same enroot version SoftBank's login server runs (we verified
# /usr/bin/enroot 3.4.1 there) so the .sqsh format is fully compatible.
RUN ARCH="amd64" && \
    curl -fSsL -o /tmp/enroot.deb \
        "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${ARCH}.deb" && \
    curl -fSsL -o /tmp/enroot+caps.deb \
        "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot+caps_${ENROOT_VERSION}-1_${ARCH}.deb" && \
    apt-get update -qq && \
    apt-get install -y -qq /tmp/enroot.deb /tmp/enroot+caps.deb && \
    rm -f /tmp/*.deb && \
    rm -rf /var/lib/apt/lists/*

# Sanity check
RUN enroot version

WORKDIR /work
ENTRYPOINT ["enroot"]