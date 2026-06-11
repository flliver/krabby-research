# Depth Anything 3 (DA3) container — feed-forward multi-view reconstruction
#   Story: STO-SCN-060 (EPI-SCN-FEEDFORWARD-RECON)
#   Build (on dbeeprz):  docker build -t krabby-da3:0.1 .
#   Push:                docker tag krabby-da3:0.1 j.pski.org:5000/krabby-da3:0.1
#                        docker push j.pski.org:5000/krabby-da3:0.1
#
# Self-contained: code (pinned), gsplat gaussian head (compiled for
# sm_89 Ada + sm_120 Blackwell), and the DA3NESTED-GIANT-LARGE-1.1
# checkpoint are all baked in.
#
# LICENSE NOTE (recorded in STO-SCN-059/060): the GIANT/LARGE
# checkpoints are CC BY-NC 4.0 — internal research evaluation only.
# Apache-2.0 tiers: DA3-BASE, DA3-SMALL, DA3METRIC-LARGE.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Pinned upstream (main @ 2026-03-21)
ARG DA3_SHA=41736238f5bced4debf3f2a12375d2466874866d
# gsplat commit pinned by DA3's own README
ARG GSPLAT_SHA=0b4dddf04cb687367602c01196913cde6a743d70

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV FORCE_CUDA=1

RUN apt-get update && apt-get install -y \
        git wget curl ffmpeg cmake ninja-build \
        libgl1 libglib2.0-0 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/da3env
ENV PATH="/opt/da3env/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel setuptools ninja

# Torch stack — same known-good cu128 wheels as krabby-matcha
RUN pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128
# xformers AFTER torch — pinned to the torch-2.7.0 build from the SAME
# cu128 index (unpinned xformers pulls a newer torch; the assert below
# caught exactly that on the first build attempt, 2026-06-10)
RUN pip install --no-cache-dir xformers==0.0.30 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    python -c "import torch; assert torch.__version__.startswith('2.7'), torch.__version__"

# DA3, pinned
RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /opt/DA3 \
    && cd /opt/DA3 && git checkout ${DA3_SHA}
RUN cd /opt/DA3 && pip install --no-cache-dir -e .

# gsplat gaussian head (source build; needs the CUDA archs above)
RUN pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_SHA}

# Bake the checkpoint (CC BY-NC tier — see license note above)
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --local-dir /opt/checkpoints/DA3NESTED-GIANT-LARGE-1.1

ENV DA3_MODEL_DIR=/opt/checkpoints/DA3NESTED-GIANT-LARGE-1.1

# ── krabby pipeline tools — BAKED, not mounted ──────────────────────────
# Operator policy (2026-06-10): results delivered via /tmp-copied tools
# are INVALID — the image digest in results.json must cover the tools
# that produced the result. Sync krabby-tools/ from real2sim/ before
# building (see README); TOOLS_GIT_SHA records the source commit.
ARG TOOLS_GIT_SHA=unknown
COPY krabby-tools/ /opt/krabby-tools/
ENV PYTHONPATH=/opt/krabby-tools:${PYTHONPATH}

LABEL org.opencontainers.image.title="krabby-da3" \
      org.opencontainers.image.source="https://github.com/ByteDance-Seed/Depth-Anything-3" \
      io.krabby.da3.upstream_sha="${DA3_SHA}" \
      io.krabby.da3.checkpoint="DA3NESTED-GIANT-LARGE-1.1" \
      io.krabby.da3.checkpoint_license="CC-BY-NC-4.0 (research evaluation only)" \
      io.krabby.da3.selfcontained="true" \
      io.krabby.da3.story="STO-SCN-060" \
      io.krabby.da3.tools_git_sha="${TOOLS_GIT_SHA}"

WORKDIR /workspace
CMD ["/bin/bash"]
