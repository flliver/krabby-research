# Docker Base Images and Dependencies

This document specifies the base images and dependencies required for each Dockerfile.

## Prerequisites

### GPU Support Setup

All containers that use GPU acceleration require NVIDIA Container Toolkit to be installed and configured. Use the provided setup script:

```bash
./scripts/jetson/setup-docker-gpu.sh
```

This script will:
- Detect your Linux distribution
- Install nvidia-container-toolkit
- Configure Docker to use the NVIDIA runtime
- Restart the Docker daemon
- Verify GPU access works

**Note**: You may need to restart your terminal session or log out and back in after running the script for changes to take effect.

For manual installation, see the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Required `docker run` flags for PyTorch containers

Any container running PyTorch with multiprocessing (which includes the IsaacSim,
locomotion, testing, and any future ML containers) **must** be started with
`--shm-size=8g`:

```bash
docker run --rm --gpus all --shm-size=8g ...
```

Docker's default `/dev/shm` is **64 MB**, which is insufficient for PyTorch's
shared-memory CUDA tensors used by `DataLoader` workers and SLAM/pose-estimation
backends. **Without `--shm-size`, the container starts, prints its config, and
silently deadlocks at 0% GPU.** No error message, no crash — just no progress.
This wasted multiple debugging cycles before being identified.

Recommended flag set for any PyTorch + CUDA container:

```bash
docker run --rm --gpus all \
    --shm-size=8g \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ...
```

The latter three are NVIDIA's official recommendations and are echoed by the NGC
PyTorch container itself at startup.

## Overview

The project requires the following container images:

**Locomotion stack:**
1. **Locomotion container** (Jetson/ARM) - **Production** - Combined inference logic + HAL server for robot deployment (inproc ZMQ)
2. **IsaacSim container** (x86) - Combined inference logic + IsaacSim HAL server for simulation testing (inproc ZMQ)
3. **Testing container (x86)** - **Testing/Development only** - Combined inference logic + game loop test script (inproc ZMQ)
4. **Testing container (ARM)** - **Testing/Development only** - Combined inference logic + game loop test script for ARM-specific testing (inproc ZMQ)

**M11 scene-reconstruction stack** (real-to-sim pipeline; see grant
[Milestone 11](https://github.com/flliver/patina-foundation-grants/blob/main/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md)):
5. **Scene-reconstruction-base** (x86) — COLMAP (source-built CUDA) + Open3D base for the canonical T0/T1 pipeline.
6. **Matcha** (x86) — MAtCha sparse-view 3D reconstruction; **primary** T1 pipeline (watertight TSDF + tetra meshes).
7. **MASt3R-SLAM** (x86) — neural SLAM as T0 alternative per grant Appendix A.
8. **SLAM3R** (x86) — Real-time dense reconstruction; T0/T1 alternative per grant Appendix A.
9. **VGGT** (x86) — Feed-forward 3D reconstruction; T0 alternative per grant Appendix A.

## Container Images

### 1. Locomotion Container (`images/locomotion/`)

**Purpose**: **Production container** for Jetson Orin robot deployment. Combines inference logic (`compute/parkour/`) and HAL server (`locomotion/`) in the same process (inproc ZMQ). This is the container that runs on the actual robot in production.

#### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:25.10-py3-igpu` (for JetPack 6+)
  - **JetPack 7.0**: Latest version, uses Jetson Linux 38.2 (Ubuntu 24.04) - Use `nvcr.io/nvidia/pytorch:25.10-py3-igpu` or latest available
  - **JetPack 6.x**: Uses Jetson Linux 36.x (Ubuntu 22.04) - Use `nvcr.io/nvidia/pytorch:25.10-py3-igpu` or latest available
  - **Alternative (JetPack 5.x)**: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (latest available for JetPack 5)
  - **Alternative (Community)**: `dustynv/pytorch:r36.2.0-pth2.1-py3` (community maintained)
- **Architecture**: ARM64 (aarch64)
- **Note**: For JetPack 6 and later, NVIDIA moved PyTorch containers from `l4t-pytorch` to `pytorch` with `igpu` tag. JetPack 7.0 is the latest version. See [NVIDIA PyTorch Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for latest available images, [NVIDIA Forums](https://forums.developer.nvidia.com/t/orin-nano-l4t-r36-4-3-docker-pull-fails-manifest-unknown-on-nvcr-io/335413/5) for migration details, and [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads) for JetPack version information.

**Base Image Contents** (what's already included in `nvcr.io/nvidia/pytorch:25.10-py3-igpu`):
- **OS**: Ubuntu 24.04
- **CUDA**: 9.0
- **cuDNN**: 9.10.2.21
- **TensorRT**: 10.11.0.33+jp6
- **PyTorch**: 2.9.0a0+145a3a7
- **NumPy**: 1.26.4?
- **NVIDIA drivers**: Pre-installed
- **Python**: 3.11?
- **Python packages**: Common ML dependencies that come with PyTorch
- **System libraries**: Standard Ubuntu packages, hardware access libraries

**Why PyTorch with igpu tag instead of a separate JetPack image?**
- The PyTorch image (with `igpu` tag for Jetson) already includes all necessary runtime components (CUDA, cuDNN, TensorRT, OS) directly in the image
- We don't need the full JetPack SDK development tools (like sample code, documentation, additional dev tools) for production deployment
- The PyTorch image is optimized for ML inference workloads and includes pre-built PyTorch optimized for Jetson
- Using PyTorch with `igpu` tag gives us everything we need (OS, CUDA, cuDNN, TensorRT, PyTorch) without the overhead of the full JetPack SDK

#### System Dependencies
- `python3.10`
- `python3-pip`
- `git`
- `build-essential`
- `libzmq3-dev`

#### Python Dependencies (`requirements.txt`)
```
# Core ML/AI (already in base image - specify versions for consistency)
# torch>=2.1.0  # Pre-installed in l4t-pytorch base image
# numpy>=1.24.0  # Usually pre-installed with PyTorch

# ZMQ communication (inproc for internal, TCP for external if needed)
pyzmq>=25.0.0

# ZED SDK (if using ZED camera)
# Note: ZED SDK must be installed separately via NVIDIA's installer
# or included in base image if available

# Data handling
dataclasses-json>=0.6.0

# Utilities
typing-extensions>=4.8.0

# Note: This container includes both:
# - compute/parkour/ (inference logic)
# - locomotion/jetson/ (HAL server with real sensors)

# HAL data collector (optional `--data-collector-output-dir <path>`; settings in data_collection/collector_settings.py)
PyYAML>=6.0.0
rosbags>=0.10.0
```

#### ZED SDK Installation

The ZED SDK is installed automatically during the Docker build. The Dockerfile uses **ZED SDK 5.1.1 for L4T 36.4 (JetPack 6.1/6.2)** (static configuration).

**Current Configuration:**
- **ZED SDK Version**: 5.1.1
- **L4T Version**: 36.4
- **JetPack Version**: 6.1/6.2
- **Target**: Jetson Orin
- **CUDA**: 12.6
- **Status**: Stable, recommended for Jetson Orin
- **Download URL Format**: `https://download.stereolabs.com/zedsdk/{version}/l4t{l4t_version}/jetsons`
- **Installer Type**: `.zstd.run` (self-extracting installer script)

**Installation Process:**
The Dockerfile downloads the ZED SDK installer from Stereolabs, extracts it, and runs the installation script in **silent mode** (`-- silent`). The license is accepted non-interactively; no manual prompts during build. The installer is a self-extracting `.run` file that contains the installation scripts and binaries. AI models (NEURAL depth, etc.) are not included in the installer and are downloaded/optimized on first use unless pre-optimized; see [JETSON_DEPLOYMENT.md — Pre-optimizing ZED NEURAL depth models](JETSON_DEPLOYMENT.md#pre-optimizing-zed-neural-depth-models) to bake them into your workflow.

**Available ZED SDK versions by JetPack** (for reference):

- **JetPack 7.0 (L4T 38.2) - ZED SDK 5.1** ⚠️ **BETA**
  - **Target**: Jetson Thor
  - **CUDA**: 13.0
  - **Download URL**: `https://download.stereolabs.com/zedsdk/5.1.1/l4t38.2/jetsons`
  - **Limitations**: 
    - Video encoding/decoding is **not functional** with this version of JetPack
    - GMSL Camera support requires drivers and a compatible partner carrier board (compatible setup will be released soon)
  - **Status**: Beta release - use with caution for production
  - **Note**: Not currently configured in Dockerfile. To use JetPack 7.0, modify the Dockerfile ZED SDK installation section to use `l4t38.2` instead of `l4t36.4`.

- **JetPack 6.1 and 6.2 (L4T 36.4) - ZED SDK 5.1** ✅ **Currently Used**
  - **Target**: Jetson Orin
  - **CUDA**: 12.6
  - **Download URL**: `https://download.stereolabs.com/zedsdk/5.1.1/l4t36.4/jetsons`
  - **Status**: Stable, recommended for Jetson Orin

**Note**: ZED SDK downloads from Stereolabs are publicly accessible (no authentication required). The download URLs use the format `/zedsdk/{version}/l4t{l4t_version}/jetsons` and redirect to the actual installer file. If the automatic download fails, you may need to:
1. Download the ZED SDK `.run` installer manually from [Stereolabs Developer Portal](https://www.stereolabs.com/en-fr/developers/release)
2. Copy it into the build context
3. Modify the Dockerfile to use `COPY` instead of `wget`

**References:**
- [ZED SDK Release Notes](https://www.stereolabs.com/en-fr/developers/release)
- [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)

#### Special Considerations
- **Production container** - This is what runs on the robot
- **ARM64 architecture only** - Cannot run on x86
- Requires JetPack/L4T base image for hardware access (JetPack 7.0 is latest, see [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads))
- Combines inference logic and HAL server in single container
- Uses inproc ZMQ for communication between inference and HAL (same process)
- **ZED SDK installed automatically** during Docker build (see ZED SDK Installation section above)
- PyTorch version may differ from x86 (use Jetson-optimized builds)
- TensorRT may be needed for optimized inference (included in JetPack)
- ZED 2i camera access: ZED SDK uses USB (libusb). Use `-v /dev:/dev` and `--privileged` (see [JETSON_DEPLOYMENT.md](JETSON_DEPLOYMENT.md)); `--device=/dev/video0` alone is usually insufficient.
- Real sensors (camera, IMU, encoders) connected via HAL server
- **JetPack 7.0 users**: Be aware that ZED SDK 5.1 for JetPack 7.0 is in beta and has limitations (video encoding/decoding not functional)

#### Runtime Requirements
- Jetson Orin hardware
- ZED 2i camera (if using): `-v /dev:/dev` and `--privileged` (USB access)
- GPU access: `--gpus all` (if using nvidia-docker)

---

### 2. IsaacSim Container (`images/simulation/isaac/`)

**Purpose**: Combined inference logic + IsaacSim HAL server container. Runs Isaac Sim with HAL server integration and inference logic in the same process (inproc ZMQ). Publishes simulation state, applies joint commands, and runs policy inference.

#### Base Image
- **Image**: `nvcr.io/nvidia/isaac-sim:5.1.0` (or latest Isaac Sim container)
  - **NGC Catalog**: [Isaac Sim 5.1.0 Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim?version=5.1.0)
  - **Alternative**: Custom base with Isaac Sim installed
- **OS**: Ubuntu 22.04 (Isaac Sim requirement)
- **Isaac Sim**: 5.1.0 (Kit 107.3.3)
- **Architecture**: Multi-architecture support
  - Linux (x86_64) - Primary platform
  - Linux (aarch64) - ARM64 support available
  - **Note**: Live-streaming not yet supported on aarch64
- **Container Security**: Runs as rootless user by default

#### System Dependencies
- Most dependencies included in Isaac Sim base image
- **`images/isaacsim/Dockerfile`** installs a **full in-container Gst + GI stack** for HAL tests and tooling: **`gstreamer1.0-tools`**, **`gstreamer1.0-plugins-{base,good,bad,ugly}`** (e.g. **`videoconvert`**, **`x264enc`**, **`h264parse`**), **`gir1.2-gstreamer-1.0`**, **`gir1.2-gst-plugins-base-1.0`**, and dev packages (**`libgstreamer1.0-dev`**, **`libgstreamer-plugins-base1.0-dev`**, **`libgirepository1.0-dev`**, **`libglib2.0-dev`**, **`pkg-config`**, **`gobject-introspection`**, **`meson`**, **`ninja-build`**, **`libcairo2-dev`**) so **`pip install PyGObject`** into **`/workspace/testenv`** can compile against system typelibs and **`import gi.repository.Gst`** matches the distro GStreamer. **`pytest tests/unit/hal/test_gstreamer_runtime.py`** is expected to run in **`krabby-testing-x86`** / Isaac CI images built from this Dockerfile. The **locomotion** image carries runtime GStreamer plugins for ZED/Jetson paths but does not duplicate this full venv+GI build (see **`images/locomotion/Dockerfile`** comments).
- Additional packages if needed:
  - `python3.10`
  - `python3-pip`
  - `libzmq3-dev`

#### Python Dependencies (`requirements.txt`)
```
# Isaac Sim dependencies (usually pre-installed in base image)
# isaac-sim
# omni

# Core ML/AI (for inference)
# torch - Usually pre-installed in Isaac Sim base image
# numpy - Usually pre-installed

# ZMQ communication (inproc for internal communication)
pyzmq>=25.0.0

# Data handling
dataclasses-json>=0.6.0

# Utilities
typing-extensions>=4.8.0

# Note: This container includes both:
# - compute/parkour/ (inference logic)
# - hal/server/isaac/ (HAL server with simulation)

# HAL data collector (optional `--data-collector-output-dir <path>`; rosbag2 mcap via rosbags)
PyYAML>=6.0.0
rosbags>=0.10.0
```

#### Special Considerations
- **Combines inference and HAL server** - Both run in same process using inproc ZMQ
- **Requires Isaac Sim license** - Base image requires NVIDIA NGC account and authentication
- **Multi-architecture support** - Container supports both x86_64 and aarch64 architectures
- **Rootless container** - Runs as non-root user by default for improved security
- GPU required for Isaac Sim rendering
- Large image size (~20GB+) due to Isaac Sim
- May need to mount Isaac Sim assets/workspace
- Display access may be needed for GUI (use `--env DISPLAY` and X11 forwarding)
- **Live-streaming limitation** - Live-streaming features not yet supported on aarch64 architecture

#### Runtime Requirements
- NVIDIA GPU with CUDA support
- Isaac Sim license/access
- Display (optional, for GUI): `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`
- GPU access: `--gpus all`

#### X11 Forwarding for Visual Display
To enable visual display when running Isaac Sim containers, you need to configure X11 forwarding:

1. **Allow X11 access** (run once per session):
   ```bash
   xhost +local:docker
   # Or more securely (only for root):
   xhost +local:root
   ```
   Note: This setting is session-specific and will reset when you log out, restart your X server, or start a new X session.

2. **Test X11 forwarding** (optional but recommended):
   ```bash
   docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --entrypoint /bin/bash krabby-isaacsim:latest -c "apt-get update && apt-get install -y x11-apps && xeyes"
   ```
   If you see the xeyes window appear on your host, X11 forwarding is configured correctly.

3. **Make it persistent** (optional):
   Add `xhost +local:docker` to one of:
   - `~/.xprofile` (runs when X session starts)
   - `~/.xsessionrc` (runs when X session starts)
   - `~/.bashrc` or `~/.zshrc` (runs when shell starts, but only if DISPLAY is set)

---

### 3. Testing Container - x86 (`images/testing/x86/`)

**Purpose**: **Testing/Development only** - Container for x86 workstations that combines inference logic and game loop test script in the same process (inproc ZMQ). The game loop simulates sensor messages by sending test data over the HAL interface to test inference logic. This is NOT used in production.

#### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:25.10-py3`
  - **NGC Catalog**: [NVIDIA PyTorch 25.10-py3 Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=25.10-py3)
  - **Alternative**: Custom base with PyTorch installed
- **Architecture**: x86_64 (amd64)
- **Note**: NVIDIA PyTorch containers are optimized for ML workloads and include pre-built PyTorch with CUDA support. See [NVIDIA PyTorch Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for latest available images and version information.

**Base Image Contents** (what's already included in `nvcr.io/nvidia/pytorch:25.10-py3`):
- **OS**: Ubuntu 24.04.3 LTS
- **CUDA**: 9.0
- **cuDNN**: 9.14.0.64
- **TensorRT**: 10.13.3.9
- **PyTorch**: 2.9.0a0+145a3a7
- **torchvision**: 0.24.0a0+094e7af5
- **torchaudio**: Usually included (version depends on PyTorch build)
- **NumPy**: Usually included (version depends on PyTorch build)
- **Python**: 3.12.3
- **NVIDIA drivers**: Pre-installed
- **System libraries**: Standard Ubuntu packages

#### System Dependencies
- `python3.12`
- `python3.12-dev`
- `python3-pip`
- `git`
- `build-essential`
- `libzmq3-dev`

#### Python Dependencies (`requirements.txt`)
```
# Core ML/AI (already in base image - specify versions for consistency)
# torch==2.9.0a0  # Pre-installed in NVIDIA PyTorch base image
# torchvision==0.24.0a0  # Pre-installed with PyTorch
# torchaudio  # Usually pre-installed with PyTorch (version depends on PyTorch build)
# numpy>=1.24.0  # Usually pre-installed with PyTorch

# ZMQ communication
pyzmq>=25.0.0

# Data handling
dataclasses-json>=0.6.0  # For version-aware serialization

# Utilities
typing-extensions>=4.8.0
```

#### Special Considerations
- **Testing/Development only** - Not deployed to production robots
- **Combines inference and game loop** - Both run in same process using inproc ZMQ
- **NVIDIA PyTorch container** - Optimized for ML workloads with pre-built PyTorch and CUDA support
- Requires NVIDIA GPU support (nvidia-docker runtime)
- CUDA 9.0 compatibility required
- PyTorch must match training environment version
- No Isaac Sim dependencies (decoupled from simulation)
- Game loop script simulates sensor messages for testing inference logic
- **Requires NGC authentication** - Base image requires NVIDIA NGC account: `docker login nvcr.io`

---

### 4. Testing Container - ARM (`images/testing/arm/`)

**Purpose**: **Testing/Development only** - Container for ARM workstations (e.g., Jetson) that combines inference logic and game loop test script in the same process (inproc ZMQ). Used to test ARM-specific nuances and catch platform-specific issues before production deployment. This is NOT used in production.

#### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:25.10-py3-igpu` (for JetPack 6+)
  - **JetPack 7.0**: Latest version, uses Jetson Linux 38.2 (Ubuntu 24.04) - Use `nvcr.io/nvidia/pytorch:25.10-py3-igpu` or latest available
  - **JetPack 6.x**: Uses Jetson Linux 36.x (Ubuntu 22.04) - Use `nvcr.io/nvidia/pytorch:25.10-py3-igpu` or latest available
  - **Alternative (JetPack 5.x)**: `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` (latest available for JetPack 5)
  - **Alternative (Community)**: `dustynv/pytorch:r36.2.0-pth2.1-py3` (community maintained)
- **Architecture**: ARM64 (aarch64)
- **Note**: For JetPack 6 and later, NVIDIA moved PyTorch containers from `l4t-pytorch` to `pytorch` with `igpu` tag. JetPack 7.0 is the latest version. See [NVIDIA PyTorch Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for latest available images, [NVIDIA Forums](https://forums.developer.nvidia.com/t/orin-nano-l4t-r36-4-3-docker-pull-fails-manifest-unknown-on-nvcr-io/335413/5) for migration details, and [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads) for JetPack version information.

**Base Image Contents** (what's already included in `nvcr.io/nvidia/pytorch:25.10-py3-igpu`):
- **OS**: Ubuntu 24.04
- **CUDA**: 9.0
- **cuDNN**: 9.10.2.21
- **TensorRT**: 10.11.0.33+jp6
- **PyTorch**: 2.9.0a0+145a3a7
- **NumPy**: 1.26.4?
- **NVIDIA drivers**: Pre-installed
- **Python**: 3.11?
- **Python packages**: Common ML dependencies that come with PyTorch
- **System libraries**: Standard Ubuntu packages, hardware access libraries

**Why PyTorch with igpu tag instead of a separate JetPack image?**
- The PyTorch image (with `igpu` tag for Jetson) already includes all necessary JetPack runtime components (CUDA, cuDNN, TensorRT) from the L4T Base layer
- We don't need the full JetPack SDK development tools (like sample code, documentation, additional dev tools) for testing
- The PyTorch image is optimized for ML inference workloads and includes pre-built PyTorch optimized for Jetson
- Using PyTorch with `igpu` tag gives us everything we need (OS, CUDA, cuDNN, TensorRT, PyTorch) without the overhead of the full JetPack SDK

#### System Dependencies
- `python3.10`
- `python3-pip`
- `git`
- `build-essential`
- `libzmq3-dev`

#### Python Dependencies (`requirements.txt`)
```
# Core ML/AI (already in base image - specify versions for consistency)
# torch>=2.1.0  # Pre-installed in l4t-pytorch base image
# numpy>=1.24.0  # Usually pre-installed with PyTorch

# ZMQ communication (inproc for internal communication)
pyzmq>=25.0.0

# Data handling
dataclasses-json>=0.6.0

# Utilities
typing-extensions>=4.8.0

# Note: This container includes both:
# - compute/parkour/ (inference logic)
# - tests/game_loops/ (game loop test scripts)
```

#### Special Considerations
- **Testing/Development only** - Not deployed to production robots
- **Combines inference and game loop** - Both run in same process using inproc ZMQ
- **ARM64 architecture only** - Cannot run on x86
- Requires JetPack/L4T base image for hardware access (JetPack 7.0 is latest, see [NVIDIA JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads))
- PyTorch version may differ from x86 (use Jetson-optimized builds)
- Used to catch ARM-specific issues (numerical precision, performance differences, etc.)
- Can run on Jetson hardware or ARM emulation environments

#### Runtime Requirements
- Jetson Orin hardware (or ARM64 compatible system)
- GPU access: `--gpus all` (if using nvidia-docker)

---

### 5. Scene-Reconstruction-Base Container (`images/scene-reconstruction-base/`)

**Purpose**: Base container for the M11 grant-canonical T0/T1 pipeline.
COLMAP source-built with CUDA support for sm_89 (Ada) and sm_120
(Blackwell), plus Open3D / pymeshlab / trimesh for downstream mesh
processing. Used directly for the COLMAP MVS + Poisson workflow that
the grant overview specifies.

#### Base Image
- **Image**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **OS**: Ubuntu 24.04
- **CUDA**: 12.8 (devel; required for sm_120 source compilation)
- **Architecture**: x86_64

#### System Dependencies
- COLMAP build chain: `python3 python3-pip python3-venv git wget curl ffmpeg cmake ninja-build libboost-all-dev libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev`
- COLMAP itself is source-built in the Dockerfile (3.11.1, with `CMAKE_CUDA_ARCHITECTURES="89;100"`).

#### Python Dependencies (`requirements.txt`)
```
open3d
pymeshlab
trimesh
numpy
pillow
```

System-Python install via `pip3 install --break-system-packages` (PEP 668).

#### Special Considerations
- Build is heavy (~10–20 min) due to COLMAP source compilation.
- HF_TOKEN is supplied at run time via `--env-file`, never baked in.

#### Runtime Requirements
- NVIDIA GPU with CUDA 12.8+ (Ada sm_89 or Blackwell sm_120 verified)
- NVIDIA Container Toolkit (run `scripts/setup-docker-gpu.sh` once)
- See **Required `docker run` flags for PyTorch containers** above for
  the mandatory flag set.

---

### 6. Matcha Container (`images/matcha/`)

**Purpose**: **Primary M11 T1 pipeline.** MAtCha (Atlas of Charts)
sparse-view 3D reconstruction producing watertight TSDF and adaptive-
tetrahedralization meshes natively. Selected over the grant-canonical
COLMAP MVS + Poisson because it was the only candidate that reliably
produced watertight meshes end-to-end during M11 evaluation. Tool
substitution is defensible per grant Appendix A; disclosure tracked.

See `images/matcha/NOTES.md` for the full backstory of each build choice
(8 patches discovered while porting from PyTorch 2.0.1+cu118 to
2.7.0+cu128 for sm_120 Blackwell support).

#### Base Image
- **Image**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **OS**: Ubuntu 24.04
- **CUDA**: 12.8 (devel)
- **Architecture**: x86_64
- **TORCH_CUDA_ARCH_LIST**: `"8.9;12.0"` (Ada + Blackwell)

#### System Dependencies
- `python3.11` via deadsnakes PPA (venv at `/opt/matcha`)
- Build chain: `git wget curl ffmpeg cmake ninja-build libgl1 libglib2.0-0 libeigen3-dev libcgal-dev libboost-all-dev`

#### Python Dependencies (`requirements.txt`)
See `images/matcha/requirements.txt`. Highlights:
- PyTorch 2.7.0 + torchvision 0.22.0 + cu128
- pytorch3d 0.7.8 (built from source)
- 6 native CUDA extensions built from source: `curope`, `ASMK`,
  `diff-surfel-rasterization`, `simple-knn`, `tetra-triangulation`
- `faiss-cpu` (faiss-gpu-cu12 1.14.1 lacks sm_120 kernels)
- Full conda-replacement runtime: plyfile, open3d, trimesh, tqdm,
  matplotlib, roma, einops, huggingface-hub, safetensors, transformers,
  timm, opencv-python==4.11.0.86, imageio + ffmpeg, natsort,
  scikit-learn, scikit-image, pyyaml, gradio
- **Excluded**: `xformers` (pulls torch 2.11 nightly → breaks pytorch3d)

#### Special Considerations
- ~30-min build on RTX 5080
- Pre-downloads ~4.3 GB of model checkpoints at build time (Depth-Anything-V2 + MAtCha)
- 8 build-time patches under `images/matcha/patches/`
- See `images/matcha/NOTES.md` for the lesson catalog

#### Runtime Requirements
- NVIDIA GPU with CUDA 12.8+, ≥ 16 GB VRAM (validated on RTX 5080)
- ≥ 32 GB RAM recommended
- HF_TOKEN in `--env-file`

---

### 7. MASt3R-SLAM Container (`images/mast3r/`)

**Purpose**: T0 alternative pipeline per grant Appendix A. Real-time
dense SLAM via two-hierarchy neural network (I2P local + L2W global).
Generates COLMAP-compatible camera poses + dense pointmap from
monocular RGB video at 20+ FPS. Used for M11's primary T0 path
(scaling MASt3R-SfM to 350+ frames).

See `images/mast3r/NOTES.md` for the 11 distinct lessons during port to
NGC PyTorch 25.10 + CUDA 13.

#### Base Image
- **Image**: `nvcr.io/nvidia/pytorch:25.10-py3` *(shared with `images/testing/x86/`)*
- **OS**: Ubuntu 24.04.3 LTS
- **CUDA**: 13.0
- **PyTorch**: 2.9.0a0+145a3a7 (prebuilt sm_89 + sm_120 kernels)
- **Architecture**: x86_64
- **TORCH_CUDA_ARCH_LIST**: `"8.9;12.0"`

#### System Dependencies
- Build chain: `git wget curl ffmpeg cmake ninja-build libgl1 libglib2.0-0 libeigen3-dev`

#### Python Dependencies (`requirements.txt`)
See `images/mast3r/requirements.txt`. Base image already provides
PyTorch + numpy + pillow; minimal additions:
- `wheel`, `setuptools` (build-time)
- `natsort`, `plyfile` (runtime)
- Source-installed: thirdparty/mast3r, thirdparty/in3d, lietorch,
  mast3r_slam_backends — all built from cloned source

#### Special Considerations
- Image is large (~36 GB)
- Multi-host distribution: `docker save | docker load` over LAN
  (~2.5 min on gigabit per `images/mast3r/NOTES.md`)
- 4 build-time patches under `images/mast3r/patches/`:
  - `patch_curope.py` — PyTorch 2.6+ `.type → .scalar_type`
  - `patch_mast3r_setup.py` — modern arch list (sm_75+sm_80+sm_86+sm_89+sm_120) + build-time CUDA check
  - `patch_dataloader.py` — `.jpg` accept + soft pyrealsense2 import
  - `patch_torch_load.py` — PyTorch 2.6+ `weights_only=True` rejects MASt3R checkpoints
- `verify_imports.py` smoke test runs at build time

#### Runtime Requirements
- NVIDIA GPU with CUDA 12.8+ (sm_89 or sm_120 verified)
- ≥ 16 GB VRAM
- HF_TOKEN in `--env-file` (MASt3R checkpoints downloaded at build time
  from Naver Labs Europe)

---

### 8. SLAM3R Container (`images/slam3r/`)

**Purpose**: Alternative T0/T1 pipeline per grant Appendix A. SLAM3R
(CVPR 2025 Highlight) — real-time dense scene reconstruction via
two-hierarchy neural network at 20+ FPS. Built as a fallback; not the
primary M11 path.

#### Base Image
- **Image**: `nvidia/cuda:12.8.0-devel-ubuntu24.04`
- **OS**: Ubuntu 24.04
- **CUDA**: 12.8 (devel)
- **Architecture**: x86_64

#### System Dependencies
- `python3.11` via deadsnakes PPA (venv at `/opt/slam3r`)
- Build chain: `git wget curl ffmpeg cmake`

#### Python Dependencies (`requirements.txt`)
See `images/slam3r/requirements.txt`. Highlights:
- PyTorch 2.5.0 + torchvision 0.20.0 + torchaudio 2.5.0 + cu124
- Upstream SLAM3R `requirements.txt` and `requirements_optional.txt`
  pulled from cloned repo at build time (**reproducibility gap** —
  not pinned to a SHA)

#### Special Considerations
- CuRoPE extension fails-soft (`|| true`) since some platforms hit
  build issues
- HF_TOKEN in `--env-file` (model weights HuggingFace-hosted)

#### Runtime Requirements
- NVIDIA GPU with CUDA 12.4+ (cu124 wheel)
- ≥ 16 GB VRAM

---

### 9. VGGT Container (`images/vggt/`)

**Purpose**: Alternative T0 pipeline per grant Appendix A. VGGT
(CVPR 2025 Best Paper) — feed-forward 3D reconstruction. Outputs land
in COLMAP-format `sparse/` directories. Built as a fallback; not the
primary M11 path.

#### Base Image
- **Image**: `nvidia/cuda:12.4.0-devel-ubuntu22.04`
- **OS**: Ubuntu 22.04
- **CUDA**: 12.4 (devel)
- **Architecture**: x86_64
- **TORCH_CUDA_ARCH_LIST**: `"8.9;12.0"`

#### System Dependencies
- `python3.11` (venv at `/opt/vggt`)
- Build chain: `python3-pip git wget curl ffmpeg`

#### Python Dependencies (`requirements.txt`)
See `images/vggt/requirements.txt`. Highlights:
- PyTorch 2.5.1 + torchvision 0.20.1 + cu124
- Upstream VGGT `requirements.txt` and `requirements_demo.txt`
  pulled from cloned repo at build time (same reproducibility caveat
  as slam3r)
- `open3d`, `trimesh` for point-cloud → mesh

#### Special Considerations
- Pre-downloads `facebook/VGGT-1B` HuggingFace model at build time
  (~2 GB)
- For 16 GB GPUs, runtime uses reduced query params:
  `--max_query_pts 2048 --query_frame_num 5`
- HF_TOKEN in `--env-file` if the gated model requires authentication

#### Runtime Requirements
- NVIDIA GPU with CUDA 12.4+
- ≥ 16 GB VRAM

---

## Shared Dependencies

### Common Python Packages
All containers share these dependencies:
- `pyzmq>=25.0.0` - ZMQ communication
- `dataclasses-json>=0.6.0` - Version-aware JSON serialization
- `typing-extensions>=4.8.0` - Type hints support
- `numpy>=1.24.0` - Array operations

### Version Compatibility Matrix

|Component|Locomotion (Jetson)|IsaacSim|Testing (x86)|Testing (ARM)|
|---------|-------------------|--------|-------------|-------------|
|Python|3.10|3.10|3.12.3|3.10|
|PyTorch|2.9.0a0+145a3a7 (Jetson)|Included|2.9.0a0+145a3a7|2.9.0a0+145a3a7 (Jetson)|
|CUDA|9.0 (JetPack)|Included|9.0|9.0 (JetPack)|
|cuDNN|9.10.2.21|Included|9.14.0.64|9.10.2.21|
|TensorRT|10.11.0.33+jp6|N/A|10.13.3.9|10.11.0.33+jp6|
|ZMQ|25.0.0+|25.0.0+|25.0.0+|25.0.0+|
|OS|Ubuntu 24.04 (L4T)|Ubuntu 22.04|Ubuntu 24.04|Ubuntu 24.04 (L4T)|

### Version Compatibility Matrix — M11 Scene-Reconstruction Stack

|Component|Scene-Recon Base|Matcha|MASt3R-SLAM|SLAM3R|VGGT|
|---------|----------------|------|-----------|------|----|
|Python|3.12 (system)|3.11 (venv)|3.12 (base)|3.11 (venv)|3.11 (venv)|
|PyTorch|N/A (no PyTorch)|2.7.0+cu128|2.9.0a0+cu130 (NGC)|2.5.0+cu124|2.5.1+cu124|
|CUDA (devel)|12.8|12.8|13.0|12.8|12.4|
|sm_89 (Ada)|✅|✅|✅|✅|✅|
|sm_120 (Blackwell)|✅|✅|✅|✅|✅|
|OS|Ubuntu 24.04|Ubuntu 24.04|Ubuntu 24.04|Ubuntu 24.04|Ubuntu 22.04|

## Build Order

**Locomotion stack:**
1. **Locomotion container** - Production container, highest priority. Can be built independently, requires Jetson hardware for testing
2. **IsaacSim container** - Can be built independently, requires Isaac Sim access
3. **Testing container (x86)** - Testing/development only. Can be built independently, no dependencies
4. **Testing container (ARM)** - Testing/development only. Can be built independently, requires Jetson hardware or ARM emulation

**M11 scene-reconstruction stack** (independent of locomotion stack):
5. **Scene-Reconstruction-Base** - COLMAP source-built; ~10–20 min build
6. **Matcha** - Primary T1 pipeline; ~30 min build (heaviest)
7. **MASt3R-SLAM** - T0 alternative; ~36 GB image
8. **SLAM3R** - Alternative; depends on upstream `requirements.txt` (pin recommended)
9. **VGGT** - Alternative; pre-downloads ~2 GB model at build time

Run `make build-m11-images` to build all five M11 images in order. Or
build individually with `make build-<name>-image`.

## Version Pinning Strategy

- **Major versions**: Pin major versions (e.g., `torch==2.7.0`)
- **Minor versions**: Allow patch updates (e.g., `pyzmq>=25.0.0,<26.0.0`)
- **System packages**: Use specific versions in apt (e.g., `python3.11=3.11.0-1`)
- **Base images**: Pin to specific tags (avoid `latest` in production)

## Testing Dependencies

### Unit Tests
- `pytest>=7.4.0`
- `pytest-asyncio>=0.21.0` (if using async)
- `pytest-timeout>=2.1.0`

### Integration Tests
- Same as runtime dependencies
- Additional test frameworks as needed

## Security Considerations

- Use official base images from trusted sources (NVIDIA, PyTorch)
- Regularly update base images for security patches
- Scan images for vulnerabilities (use `docker scan` or similar)
- Minimize attack surface (use `-runtime` variants when possible)
- Don't include secrets or API keys in images

## Troubleshooting

### Common Issues

1. **CUDA version mismatch**: Ensure base image CUDA version matches PyTorch CUDA version
2. **Architecture mismatch**: Jetson containers must use ARM64 base images
3. **ZED SDK not found**: Install ZED SDK separately or use pre-built image with ZED
4. **Isaac Sim license**: Ensure valid Isaac Sim access/license
5. **Memory limits**: Isaac Sim requires significant RAM (16GB+ recommended)

## References

- PyTorch Docker Hub: https://hub.docker.com/r/pytorch/pytorch
- NVIDIA PyTorch Container Catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
- NVIDIA L4T Containers: https://catalog.ngc.nvidia.com/containers
- NVIDIA JetPack Downloads: https://developer.nvidia.com/embedded/jetpack/downloads
- Isaac Sim Documentation: https://docs.omniverse.nvidia.com/app_isaacsim/
- ZMQ Python: https://pyzmq.readthedocs.io/

