# Locomotion Container

## Canonical install path

```bash
pip install krabby-launcher
krabby install            # pull mainline-latest, set up udev + dialout
krabby firmware show      # verify boards
krabby run                # start the locomotion stack
```

`krabby run` wires GPU flags, serial + input device passthrough, and ZMQ ports automatically. See [krabby/README.md](../../krabby/README.md) for the full CLI reference.

---

This directory contains two Dockerfiles for the Jetson Orin robot deployment.

| Image | Dockerfile | Source | Pushed to ECR |
|-------|-----------|--------|---------------|
| **Dev** | `Dockerfile` | Monorepo `COPY` (local wheels) | No |
| **Production** | `Dockerfile.release` | PyPI packages (pinned versions) | Yes |

---

## Dev Image

For developers iterating on HAL or compute code locally. Builds directly from the monorepo — no PyPI version bump required to test changes.

```bash
# Build (from repo root)
docker build -f images/locomotion/Dockerfile -t krabby-locomotion:dev .

# Run
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-locomotion:dev \
    --checkpoint /workspace/checkpoints/checkpoint.pt
```

---

## Production Image

Installs Krabby packages from PyPI with pinned versions. Bundled with `avrdude`, `arduino-cli` (Mega 2560 core, same pin as firmware CI), and `krabby-firmware` so MCU flashing works from inside the container without host-side flash tools.

### Pulling from ECR Public

No AWS credentials required — ECR Public allows anonymous pulls.

```bash
ECR=public.ecr.aws/t7t7b3i3/krabby-locomotion

# Latest mainline build
docker pull ${ECR}:mainline-latest

# Specific commit
docker pull ${ECR}:<sha7>

# Semver release (after a locomotion-v* tag)
docker pull ${ECR}:0.2.9
```

### Running

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    ${ECR}:mainline-latest \
    --checkpoint /workspace/checkpoints/checkpoint.pt
```

With MCU flashing:
```bash
docker run --rm --gpus all \
    --device /dev/ttyACM0 \
    ${ECR}:mainline-latest \
    krabby-firmware show
```

### Tag Scheme

| Tag | Updated on |
|-----|-----------|
| `<sha7>` | Every push to a tracked branch |
| `mainline-latest` | Every push to `mainline` |
| `release-<x-y-z>-latest` | Every push to `release/x.y.z` |
| `<semver>` (e.g. `0.2.9`) | Push of a `locomotion-v*` tag |

### PyPI Packages

Pinned in `requirements.release.txt`:

```
krabby-hal-client==0.1.0
krabby-hal-server==0.1.1
krabby-hal-server-jetson==0.1.1
krabby-hal-tools==0.1.0
krabby-compute-parkour==0.1.0
krabby-controller==0.1.2
krabby-firmware==0.2.9
```

`krabby-data-collection` and `krabby-teleop-edge` are not yet published to PyPI and are excluded from the production image.

### Bumping Pins

1. Edit the `==` versions in `requirements.release.txt`
2. Push to `mainline` or a `release/*` branch
3. CI builds and pushes the new image automatically

### Bundled Flash Tooling

The production image includes:
- `avrdude` (via apt) — for direct AVR flashing
- `arduino-cli` 1.1.1 (ARM64) + `arduino:avr` Mega 2560 core — same pin as firmware CI
- `krabby-firmware` — `show`, `update`, and `install` subcommands

Host-side `udev` rules and `dialout` group membership are not handled by the container; run `krabby --install` on the host (Task 3).

---

## Overview

Both images combine:
- **Policy inference** (`krabby-compute-parkour`) — policy wrapper and model inference
- **HAL server** (`krabby-hal-server-jetson`) — Jetson HAL server (catalog-driven RGB-D: ZED, MaixSense, etc.)
- **Process entrypoint** — `python -m hal.server.jetson.main`; control loop and HAL share inproc ZMQ in the same process

All components communicate via **inproc ZMQ** (same process, zero-copy).

## Configuration

- **Base image**: `nvcr.io/nvidia/pytorch:25.10-py3-igpu` (JetPack 6+)
- **Architecture**: ARM64 (aarch64) for Jetson
- **Communication**: inproc ZMQ (same process)
- **Control rate**: 100+ Hz

## Notes

- ZED SDK Python bindings (`pyzed`) require ZED SDK installed on the system (via NVIDIA installer). The Dockerfile installs it automatically.
- If ZED SDK is not available, the code gracefully falls back to mock camera mode.
- Container entrypoint: `hal.server.jetson.main` (`hal/server/jetson/main.py`)
