# Developer setup

Two halves of development in this repo:

- **Real robot (Jetson Orin)** — kit path and editable installs below
- **Simulation / training (x86)** — Isaac Sim / Isaac Lab and `make test` below

Package layout: [`docs/FOLDER_LAYOUT.md`](docs/FOLDER_LAYOUT.md).

## Real robot (Orin)

Kit cold-start, firmware, gamepad, and fleet enroll are owned by the root
[README Start here](README.md#start-here) and the docs it links (Software
quick-start, `firmware/SETUP.md`, Pro Controller / E2E guides, `fleet/ENROLL.md`,
`fleet/FIELD-TELEOP.md`, `krabby/README.md`).

### Developing code on Orin

1. Bring the host up with the [README Software quick-start](README.md#software-quick-start)
   (user venv, `krabby install`, flash as needed).
2. Clone this repo on the Orin (or sync your working tree).
3. Activate that venv and editable-install only the packages you change:

   ```bash
   source ~/.venv-krabby/bin/activate
   pip install -e ./krabby
   pip install -e ./firmware          # firmware CLI / GUI
   pip install -e ./controller        # gamepad / krabby-uno client
   ```

   For HAL packages under `hal/krabby-hal-*/`, use `make install-editable`
   (see [HAL Development Setup](#hal-development-setup)) with the venv active.
4. Re-run with the venv `krabby` on `PATH` (same sudo pattern as the README when
   host setup needs it). Reflash after Arduino sketch changes
   (`krabby firmware update` / [`firmware/SETUP.md`](firmware/SETUP.md)).

## Simulation and training (x86)

The sections below are for Isaac Sim / Isaac Lab and x86 test workflows — not
the Orin kit install path.

### Install CUDA on Ubuntu 24.04

The recommended configuration using a RTX 5080 on Ubuntu 24.04
* Kernel: 6.14.0
* Nvidia Driver: `nvidia-driver-580-open`
* Cuda Toolkit 13.0: `cuda-toolkit-13-0`

### Install Python3.11 on Ubuntu 24.04

Python3.11 is the recommended version for IsaacLab, but Python3.12 comes standard with Ubuntu.  These steps help to install Python with venv.
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### Create a venv and install PyTorch

```
python3.11 -m venv testenv
source testenv/bin/activate
pip3 install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu130
```

### Install IsaacSim

```
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### Install Isaaclab

```
git clone git@github.com:isaac-sim/IsaacLab.git
./isaaclab.sh --install
```

### HAL Development Setup

HAL packages are organized as Python packages in `hal/krabby-hal-*/` directories. For development, install packages in editable mode:

**Note:** `make install-editable` runs `pip install` commands, so ensure you have a virtual environment activated before running it.

```bash
# Install build tool (required for make build-wheels)
pip install build

# Install all HAL packages in editable mode (one-time setup)
make install-editable
```

### Docker GPU Setup

Tests require GPU access in Docker. If you see "could not select device driver" errors, configure Docker for GPU support:

```bash
./scripts/jetson/setup-docker-gpu.sh
```

This installs `nvidia-container-toolkit` and configures Docker to use it. Requires sudo access.

### Running Tests

Tests run inside the x86 testing Docker container. Run all tests (excluding Jetson tests):

```bash
make test
```
