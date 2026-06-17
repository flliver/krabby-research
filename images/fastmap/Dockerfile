# FastMap GPU SfM container (STO-SCN-101, EPI-SCN-AUTO-SUBSET-SELECT)
#
# A GPU-accelerated SfM that yields camera poses + a sparse point cloud
# (the co-visibility / track graph STO-SCN-093 needs). FastMap (pals-ttic)
# does GPU pose estimation + sparse triangulation, REPLACING COLMAP's
# CPU-bound incremental mapper. COLMAP is built WITH CUDA but used ONLY for
# GPU feature extraction + matching (no CPU mapper anywhere — operator
# directive: GPU-accelerated only).
#
# Base: krabby-da3 (proven CUDA 12.8 / torch 2.7.0 cu128 / sm_89 Ada +
#       sm_120 Blackwell on the RTX 5080).
#
# Build (on a GPU x86 host — cannot build CUDA images on the Mac):
#   rsync -a images/fastmap/ <host>:~/build/fastmap/
#   ssh <host> 'cd ~/build/fastmap && docker build -t krabby-fastmap:0.1 .'
# Push:
#   docker tag krabby-fastmap:0.1 j.pski.org:5000/krabby-fastmap:0.1
#   docker push j.pski.org:5000/krabby-fastmap:0.1
#
# Pipeline (GPU end-to-end):
#   colmap feature_extractor (SiftExtraction.use_gpu=1)
#     -> colmap exhaustive_matcher (SiftMatching.use_gpu=1)  -> database.db
#     -> python run.py --database ... --headless              -> sparse/0
#       (COLMAP format: cameras + images poses + points3D tracks)

FROM j.pski.org:5000/krabby-da3:0.4

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV FORCE_CUDA=1
ENV PATH="/opt/da3env/bin:$PATH"

# ── COLMAP 4.0.4 with CUDA — GPU feature extraction + matching only ──────
# GUI/CGAL off (headless server; no Qt, no mesher needed — FastMap does the
# pose/triangulation). CUDA archs match the da3 base (Ada + Blackwell).
# Deps per COLMAP 4.0.4 doc/install.rst (COLMAP 4.x uses OpenImageIO, not
# FreeImage). GUI off -> no Qt. BLAS via OpenBLAS (not the ~2GB MKL default).
ARG COLMAP_TAG=4.0.4
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libmetis-dev \
        libopenimageio-dev openimageio-tools \
        libgoogle-glog-dev libgtest-dev libgmock-dev \
        libsqlite3-dev libglew-dev \
        libceres-dev libsuitesparse-dev libopenblas-dev liblapack-dev \
        libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Ubuntu's OpenImageIO cmake config references an OpenCV include dir that may
# not exist; COLMAP doesn't use OIIO's OpenCV path (doc/install.rst workaround).
RUN mkdir -p /usr/include/opencv4
RUN git clone --branch ${COLMAP_TAG} --depth 1 \
        https://github.com/colmap/colmap.git /opt/colmap-src \
    && cmake -S /opt/colmap-src -B /opt/colmap-build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89;120" \
        -DGUI_ENABLED=OFF -DCGAL_ENABLED=OFF \
    && ninja -C /opt/colmap-build install \
    && rm -rf /opt/colmap-src /opt/colmap-build
RUN colmap -h >/dev/null 2>&1 && echo "colmap CLI OK"

# ── FastMap — GPU pose estimation + sparse triangulation ────────────────
# Installed into the da3 torch env (torch 2.7.0 cu128). Custom CUDA kernels
# compiled in-place (build_ext) for sm_89 + sm_120.
ARG FASTMAP_SHA=dafd165121036746e32a270a0a4e252fafb41ad7
RUN pip install --no-cache-dir \
        trimesh "pyglet<2" pyyaml dacite loguru prettytable psutil \
    && pip install --no-cache-dir "git+https://github.com/jiahaoli95/pyrender.git"
RUN git clone https://github.com/pals-ttic/fastmap.git /opt/fastmap \
    && cd /opt/fastmap && git checkout ${FASTMAP_SHA} \
    && python setup.py build_ext --inplace
ENV PYTHONPATH=/opt/fastmap:${PYTHONPATH}

# Sanity: torch CUDA + fastmap import resolve at build time.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && python -c "import sys; sys.path.insert(0,'/opt/fastmap'); import fastmap; print('fastmap import OK')"

# ── krabby pipeline tools — BAKED (STO-SCN-093 decision D; operator policy:
#    results.json provenance must cover the tools that produced them, not a
#    mounted /tmp copy). krabby-tools/ is a build-time MIRROR of canonical
#    real2sim/ sources — keep it in lockstep with `images/fastmap/sync-tools.sh`
#    and gate the build with `sync-tools.sh --check` (STO-SCN-157). Do NOT
#    hand-edit files under krabby-tools/; edit real2sim/ and re-sync. ──
ARG TOOLS_GIT_SHA=unknown
COPY krabby-tools/ /opt/krabby-tools/
ENV PYTHONPATH=/opt/krabby-tools:${PYTHONPATH}
ENV PATH=/opt/krabby-tools:${PATH}
RUN chmod +x /opt/krabby-tools/*.sh 2>/dev/null || true \
    && python -c "import sys; sys.path.insert(0,'/opt/krabby-tools'); \
import solve_plan, undistort_fisheye, covis_graph, validity_gate, capture_profile, precull_frames; \
print('krabby-tools import OK')"

LABEL org.opencontainers.image.title="krabby-fastmap" \
      io.krabby.fastmap.tools_git_sha="${TOOLS_GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/pals-ttic/fastmap" \
      io.krabby.fastmap.upstream_sha="${FASTMAP_SHA}" \
      io.krabby.fastmap.colmap_tag="${COLMAP_TAG}" \
      io.krabby.fastmap.base="krabby-da3:0.4" \
      io.krabby.fastmap.gpu_only="true (colmap=GPU match only; fastmap=GPU pose)" \
      io.krabby.fastmap.story="STO-SCN-101"

WORKDIR /workspace
CMD ["/bin/bash"]
