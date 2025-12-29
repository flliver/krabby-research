# Docker Base Images and Dependencies

This document specifies the base images and dependencies required for each Dockerfile.

## Prerequisites

### GPU Support Setup

All containers that use GPU acceleration require NVIDIA Container Toolkit to be installed and configured. Use the provided setup script:

```bash
./scripts/setup-docker-gpu.sh
```

This script will:
- Detect your Linux distribution
- Install nvidia-container-toolkit
- Configure Docker to use the NVIDIA runtime
- Restart the Docker daemon
- Verify GPU access works

**Note**: You may need to restart your terminal session or log out and back in after running the script for changes to take effect.

For manual installation, see the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Overview

The project requires four container images:
1. **Locomotion container** (Jetson/ARM) - **Production** - Combined inference logic + HAL server for robot deployment (inproc ZMQ)
2. **IsaacSim container** (x86) - Combined inference logic + IsaacSim HAL server for simulation testing (inproc ZMQ)
3. **Testing container (x86)** - **Testing/Development only** - Combined inference logic + game loop test script (inproc ZMQ)
4. **Testing container (ARM)** - **Testing/Development only** - Combined inference logic + game loop test script for ARM-specific testing (inproc ZMQ)

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
The Dockerfile downloads the ZED SDK installer from Stereolabs, extracts it, and runs the installation script in silent mode. The installer is a self-extracting `.run` file that contains the installation scripts and binaries.

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
- Camera access requires device mounting: `--device=/dev/video0`
- Real sensors (camera, IMU, encoders) connected via HAL server
- **JetPack 7.0 users**: Be aware that ZED SDK 5.1 for JetPack 7.0 is in beta and has limitations (video encoding/decoding not functional)

#### Runtime Requirements
- Jetson Orin hardware
- ZED Mini camera (if using): `--device=/dev/video0`
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

## Build Order

1. **Locomotion container** - Production container, highest priority. Can be built independently, requires Jetson hardware for testing
2. **IsaacSim container** - Can be built independently, requires Isaac Sim access
3. **Testing container (x86)** - Testing/development only. Can be built independently, no dependencies
4. **Testing container (ARM)** - Testing/development only. Can be built independently, requires Jetson hardware or ARM emulation

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

