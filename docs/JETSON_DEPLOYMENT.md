# Jetson Deployment Guide

This guide explains how to build and deploy the parkour policy runtime on Jetson Orin hardware.

## System Requirements

- **Device**: NVIDIA Jetson Orin
- **OS**: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
- **Kernel**: 5.15.148-tegra (NVIDIA Tegra)
- **Architecture**: aarch64 (ARM64)
- **JetPack**: 6.1/6.2 (L4T 36.4) or later
- **ZED 2i Camera**: Optional RGB-D (ZED SDK 5.1.1+ for L4T 36.4)
- **MaixSense-A075V**: Optional RGB-D over USB RNDIS + HTTP (no ZED SDK); see [MaixSense-A075V (optional host bring-up)](#maixsense-a075v-optional-host-bring-up) below
- **Model checkpoint file**: `.pt` format (e.g., `unitree_go2_parkour_teacher.pt`)

### Seeed reComputer Jetson Robotics J401 (reference carrier)

If you deploy on **Seeed Studio**’s [reComputer Jetson Robotics J401](https://wiki.seeedstudio.com/recomputer_jetson_robotics_J401_getting_started/), use their documentation alongside this guide:

- **Getting started:** [reComputer Jetson Robotics J401 — Seeed Wiki](https://wiki.seeedstudio.com/recomputer_jetson_robotics_J401_getting_started/)
- **Hardware reference:** [reComputer Jetson Robotics J401 datasheet (PDF)](https://files.seeedstudio.com/products/NVIDIA-Jetson/reComputer_robotics_J401_datasheet.pdf)

This guide’s reference deployment uses **reComputer J4012**, which pairs the J401 carrier with a **Jetson Orin NX 16GB** module. Seeed’s product code (e.g. J4012) and the exact NVIDIA module SKU are printed on a **sticker on the bottom** of the Jetson module—use that sticker to confirm which hardware you have if the model number is unclear.

## Prerequisites: System Setup

Before building and deploying, ensure the Jetson system is properly configured.

For **flashing** and **initial setup** (before SSH and networking are dialed in), it helps to have a **keyboard**, **mouse**, and **display** connected directly to the Jetson. On the **reComputer J401** carrier, connect the monitor with a **USB-C to HDMI** adapter or cable to the **USB 3.2 / DisplayPort 1.4** USB-C port.

Optionally, you can skip local peripherals and connect from the **flashing host** using **SSH over USB** (Jetson USB device mode), then complete **system configuration** using the **console interface** (terminal session over that link).

### 1. SSH Access

**When it applies:** SSH from a **development machine** is the usual way to copy images, run commands, and deploy over the network. For a **field deployment**, remote SSH may be **unnecessary** if the Jetson is provisioned and operated without ongoing shell access from another host (for example, fully onboard workflows or non-SSH operational interfaces).

**If you want SSH for remote access**, ensure the SSH server is installed and running **on the Jetson**:

```bash
# Install OpenSSH server if not present
sudo apt-get update
sudo apt-get install -y openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

**On your local machine**

These steps use **OpenSSH** (`ssh`, `ssh-keygen`). On **Linux and macOS** it is usually preinstalled. On **Windows 10/11**, install **OpenSSH Client** if `ssh` is not found: *Settings → Apps → Optional features → Add a feature → OpenSSH Client* (or `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0` in an elevated PowerShell). **WSL** and **Git for Windows** shells behave like Linux for paths (`~/.ssh`).

**Key-based authentication (recommended):** Use a key pair so you are not prompted for the account password on every login.

1. Create a key pair if you do not already have one:

   ```bash
   ssh-keygen -t ed25519 -C "jetson"
   ```

   Accept the default path or choose another; you can set a passphrase or leave it empty. If you press Enter at the file prompt, OpenSSH names the key pair **`id_ed25519`** / **`id_ed25519.pub`** by convention for Ed25519 keys (that name is not special to this project—it is what `ssh-keygen` suggests). Typical full paths: **`~/.ssh/id_ed25519`** on Linux, macOS, and WSL; on **native Windows**, **`%USERPROFILE%\.ssh\id_ed25519`** (for example `C:\Users\<YourUsername>\.ssh\id_ed25519`).

2. Install your **public** key on the Jetson (one-time; you will enter the Jetson user’s password when asked):

   **Linux, macOS, or WSL** (has `ssh-copy-id`):

   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <username>@<hostname-or-ip>
   ```

   **Windows PowerShell** (no `ssh-copy-id` in the base OpenSSH package—pipe the public key over SSH):

   ```powershell
   Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh <username>@<hostname-or-ip> "mkdir -p .ssh && chmod 700 .ssh && cat >> .ssh/authorized_keys && chmod 600 .ssh/authorized_keys"
   ```

   **Manual (any OS):** Append the contents of your `id_ed25519.pub` file to `~/.ssh/authorized_keys` on the Jetson (create the file and directory if needed), then on the Jetson run `chmod 700 ~/.ssh` and `chmod 600 ~/.ssh/authorized_keys`. On Windows you can copy the public key to the clipboard with `Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | Set-Clipboard`, then paste into an editor after SSHing in with a password.

3. **SSH config** so you can use a short alias (`ssh jetson`). Create or edit:

   - **Linux / macOS / WSL:** `~/.ssh/config`
   - **Windows:** `C:\Users\<YourUsername>\.ssh\config` (create the `.ssh` folder if it does not exist)

```
Host jetson
    HostName <hostname-or-ip>
    User <username>
    IdentityFile ~/.ssh/id_ed25519
```

Replace `<hostname-or-ip>` with the Jetson’s hostname (if it resolves on your network) or its IP address. Replace `<username>` with your Jetson user account.

**`IdentityFile` on Windows:** OpenSSH for Windows expands `~` to your user profile, so `~/.ssh/id_ed25519` in `config` often works. If not, use a full path with forward slashes, e.g. `IdentityFile C:/Users/<YourUsername>/.ssh/id_ed25519`.

If you use **password authentication only**, omit the `IdentityFile` line (or leave it commented). Older tooling sometimes expects RSA keys (`ssh-keygen -t rsa -b 4096` and `IdentityFile ~/.ssh/id_rsa`).

**Test the connection:**

```bash
ssh jetson
# Or, if not using the config alias:
ssh <username>@<hostname-or-ip>
```

**Troubleshooting:**

- If the connection fails, ensure the Jetson is on the same network and reachable (e.g., `ping <hostname-or-ip>`).
- Ensure the SSH server is running on the Jetson: `sudo systemctl status ssh`.
- Check that the firewall allows SSH (port 22).

### 1.1 DNS note for `.local` hostnames on Linux

Use this when the Jetson needs to connect to an external host by DNS name (for example, a teleop/relay service) and your network uses `*.local` records. This may be unnecessary for field deployments that do not depend on external hostname-based connectivity.

If your network DNS authority serves names like `host.local` but the Jetson still cannot resolve them, Linux NSS may be stopping at mDNS before trying normal DNS. A common default is:

`hosts: files mdns4_minimal [NOTFOUND=return] dns`

The `[NOTFOUND=return]` part can prevent DNS lookups for `.local` when mDNS does not answer. On Jetson/Linux, you can make DNS fallback work by removing that early return:

```bash
sudo sed -i 's/^hosts:.*/hosts:          files mdns4_minimal dns/' /etc/nsswitch.conf
```

Why this is needed:
- It keeps mDNS support but allows fallback to configured DNS servers when mDNS has no answer.

Scope of this fix:
- This addresses `*.local` name resolution via DNS fallback.
- It does **not** make bare hostnames (for example `host`) resolve by itself; bare names still require a DNS search domain or using the FQDN directly (`host.local`).

### 2. Docker Installation

Docker is required for running the locomotion container.

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group (to run docker without sudo)
sudo usermod -aG docker $USER

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Verify installation
docker --version
```

**Important**: After adding your user to the docker group, you must either:
- Log out and log back in, OR
- Run `newgrp docker` in your terminal

This is required for the group membership to take effect.

### 3. NVIDIA Container Toolkit Installation

Required for GPU access in Docker containers.

```bash
# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y curl

# Configure repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' |  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install nvidia-container-toolkit packages
sudo apt-get update
sudo apt-get install -y  nvidia-container-toolkit  nvidia-container-toolkit-base  libnvidia-container-tools  libnvidia-container1

# Configure Docker to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

# Verify configuration
sudo docker info | grep -i runtime
```

**Note**: `nvidia` should appear in the `Runtimes:` list. `Default Runtime` may still be `runc` (this is common and OK); on Jetson, run containers with `--runtime=nvidia` as shown below.

### 4. Power and performance mode (optional)

Many guides recommend setting the Jetson to maximum performance before running cameras or inference, to avoid throttling and USB/camera detection issues:

```bash
# Set power mode to max performance (mode 0; mode IDs can vary by board)
sudo nvpmodel -m 0

# Lock CPU/GPU to maximum frequencies (disables dynamic scaling)
sudo jetson_clocks
```

- **nvpmodel -m 0**: Uses the highest power profile (check available modes with `nvpmodel -q`). **Does not persist** across reboots; the board typically boots in its default (often lower) power mode.
- **jetson_clocks**: Keeps clocks at max until reboot. **Does not persist** across reboots. Omit if you prefer power saving or don’t need sustained peak performance.

Use these when you need consistent performance for the ZED camera or the locomotion pipeline; they are optional for initial testing.

**Apply on every boot (optional)** — Neither setting survives a reboot. To run both automatically at startup, create a systemd service:

```bash
sudo tee /etc/systemd/system/jetson-maxperf.service > /dev/null <<'EOF'
[Unit]
Description=Set Jetson to max performance (nvpmodel + jetson_clocks)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jetson-maxperf.service
```

After the next reboot, the service will set max power mode and lock clocks. To run it once without rebooting: `sudo systemctl start jetson-maxperf.service`. To disable: `sudo systemctl disable jetson-maxperf.service`. Adjust the `-m 0` or paths if your board uses different mode IDs or install locations (`which nvpmodel jetson_clocks`).

### 5. ZED 2i Camera (Optional — Host Verification)

**Required once per Jetson before production runs (recommended order):**

1. **Verify USB**: `lsusb | grep -i zed` — device must appear on a USB 3.0 port.
2. **Run initial diagnostics in container**: `sudo docker run --rm --runtime=nvidia --network host -v /dev:/dev --privileged -v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings --entrypoint ZED_Diagnostic krabby-locomotion:latest -c -d`

   You should see **OK** for: ZED SDK Diagnostic, Processor, Graphics Card, and CUDA Operations. Under **AI Models Diagnostic**, detection models (MULTI CLASS, HUMAN BODY, PERSON HEAD, REID) may show "not optimized" - that is normal and not used by the HAL RGB/depth pipeline. On this initial diagnostic pass, depth models may still show "not optimized".

3. **One-time optimization of all neural depth models**:

```bash
yes | sudo docker run --rm --runtime=nvidia --network host -v /dev:/dev --privileged -v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings --entrypoint ZED_Diagnostic krabby-locomotion:latest -nrlo_all
```

4. **Rerun diagnostics**: `sudo docker run --rm --runtime=nvidia --network host -v /dev:/dev --privileged -v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings --entrypoint ZED_Diagnostic krabby-locomotion:latest -c -d`

   After optimization, verify **NEURAL LIGHT DEPTH**, **NEURAL DEPTH**, and **NEURAL PLUS DEPTH** show "optimized".

Then keep the same mounts on subsequent runs:

```bash
sudo docker run --rm --runtime=nvidia --network host -v /path/to/checkpoints:/workspace/checkpoints -v /dev:/dev --privileged -v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings krabby-locomotion:latest --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt --robot go2
```

Optional camera validation using the Docker image:

1. **Validate camera output with ZED_Explorer (local desktop/X11 only)**:
   - `xhost +local:root`
   - `sudo docker run --rm --runtime=nvidia --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --privileged -v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings --entrypoint ZED_Explorer -it krabby-locomotion:latest`

**If the SDK can’t open the camera** (appears in `lsusb` but diagnostic fails), install udev rules:

```bash
echo ‘SUBSYSTEM==”usb”, ATTR{idVendor}==”2b03”, MODE=”0666”’ | sudo tee /etc/udev/rules.d/99-slabs.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Unplug and replug the ZED, then retry.

**Front + side RGB-D:** the repo catalog uses **`front_rgbd`** (primary, **`camera_driver="zed"`**) and **`side_rgbd`** (**`maixsense_a075v`**, `hal_open_rgbd=True`, `policy_scan_slot="side"`). Set **`zed_usb_serial_env`** on the front row (or rely on first detected ZED). For side MaixSense, set **`maixsense_host_env`** / **`maixsense_port_env`** (defaults in catalog: `KRABBY_JETSON_MAIXSENSE_SIDE_HOST` / `KRABBY_JETSON_MAIXSENSE_SIDE_PORT`). To use a second ZED instead of MaixSense on **`side_rgbd`**, change that row’s **`camera_driver`** to **`zed`** and set **`KRABBY_SIDE_ZED_USB_SERIAL`**.

**Container requirements for ZED on Jetson:** ZED uses USB/libusb access, so run the locomotion container with `-v /dev:/dev` and `--privileged` (using only `--device /dev/video0` is insufficient).

### MaixSense-A075V (optional host bring-up)

Optional RGB-D over **HTTP** (no Stereolabs SDK). Use as the primary `front_rgbd` driver or as extra `rgbd` catalog rows. Official docs: [MaixSense-A075V – Sipeed Wiki](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/maixsense-a075v.html).

1. **USB / link**: Often **`0525:a4a2`** (Linux RNDIS). The module is **`192.168.233.1`** on the USB Ethernet link by default.
2. **Jetson IP / routing pitfall**: If multiple host interfaces are attached to **`192.168.233.0/24`**, routing to the MaixSense default address **`192.168.233.1`** can become ambiguous and cause intermittent timeouts / unreachable camera behavior.
3. **Check reachability**: Run `ip route get 192.168.233.1` and confirm that traffic to the camera goes out through the USB network interface used by that camera and uses the expected source IP on that link. Then verify connectivity with `ping -c 3 192.168.233.1` and `curl -sS -o /dev/null -w '%{http_code}\n' http://192.168.233.1/`.
4. **Host network config**: Ensure the USB network interface for the camera is configured persistently on the host (for example via NetworkManager) with an address in the camera subnet, and that this interface comes up automatically after reboot.

5. **Web UI**: `http://192.168.233.1` (~10–15 s after power-on). Remote browser via Jetson: `ssh -N -L 8080:192.168.233.1:80 USER@JETSON` then open `http://127.0.0.1:8080`.
6. **If needed (troubleshooting only)**: Driver install notes from Sipeed are available here: [Sipeed install / driver notes](https://wiki.sipeed.com/hardware/en/maixsense/maixsense-a075v/install_drivers.html). For this Jetson production deployment, camera access uses USB networking; use the driver steps only if that USB network path is not detected or reachable.

**HAL wiring**

- Python extras: `pip install "krabby-hal-server-jetson[maixsense]"` (`requests`, `opencv-python-headless`).
- Catalog: set **`camera_driver="maixsense_a075v"`** on each **`rgbd`** row that uses MaixSense (primary **`front_rgbd`** and/or extra rows with **`hal_open_rgbd=True`**). Each such row must set **`maixsense_host_env`** and **`maixsense_port_env`** (or literal `maixsense_host` + `maixsense_port`) to explicit HTTP endpoint values—**you choose those names** in **`JETSON_SENSOR_CATALOG`** (distinct per module). Deployment passes **one `-e`** host and one **`-e`** port per env-based row.
- **Policy** uses **`camera_*`** / **`scan_features`** from the primary row; optional **`side_*`** when one row has **`policy_scan_slot="side"`** and the checkpoint uses **`num_side_scan`**. **Collision / extra streams**: read **`HardwareObservations.rgbd_by_catalog_id[id].rgb`** / **`.depth`** (each row’s own resolution). Implementation: `hal/server/jetson/maixsense_a075v.py`, `maixsense_rgb_depth_camera.py`, **`JETSON_SENSOR_CATALOG`** in `sensor_backend_jetson.py`.
- **Hardware smoke test**: `scripts/run_jetson_maixsense_hal_hw_test.sh` (expects Docker image `krabby-locomotion:latest`, **`--network host`**, and **`KRABBY_MAIXSENSE_LIVE_TEST_HOST`** set to the module IP; optional **`KRABBY_MAIXSENSE_LIVE_TEST_PORT`**).

**Docker**: No ZED-style USB passthrough for HTTP; the container must reach each module’s IP (**`--network host`** on Jetson is typical). For every MaixSense row, the catalog’s **`maixsense_host_env`** / **`maixsense_port_env`** name the variables you set at **`docker run`** (**`-e NAME=value`**, one pair per module)—same pattern as multiple ZED serial envs.

**Runtime example (MaixSense over HTTP):**

```bash
sudo docker run --rm --runtime=nvidia --network host -v /path/to/checkpoints:/workspace/checkpoints -e KRABBY_JETSON_MAIXSENSE_SIDE_HOST=192.168.233.1 krabby-locomotion:latest --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt --robot go2
```

Add more `-e ...` entries (and optional `-e ..._PORT=...`) for each additional MaixSense catalog row.

**HAL front camera (camera_rgb / camera_depth)**  
The Jetson HAL fills `HardwareObservations.camera_rgb` and `camera_depth` from the **front RGB-D observation camera** defined in **`JETSON_SENSOR_CATALOG`**: the row with **`id="front_rgbd"`** and **`is_primary=True`** sets **`camera_driver`** (e.g. **`zed`** or **`maixsense_a075v`**), **resolution**, **depth_resolution**, **fps**, and **depth_mode**. **`depth_mode`** applies to **ZED** only. **Policy scan** (`scan_features`, optional **`side_*`**) comes from the configured depth streams; **every opened RGB-D row** also appears under **`rgbd_by_catalog_id`**. GStreamer IDs, ZED install, and MaixSense networking: **SENSOR_INTERFACE.md**, ZED section above, and [MaixSense-A075V (optional host bring-up)](#maixsense-a075v-optional-host-bring-up). Wire format: **HAL_GUIDE.md** and `hal/client/data_structures/hardware.py`.

## Obtaining the Docker Image

The Docker image must be available on the Jetson device. Pull it from your container registry or load it from an archive, then tag it as `krabby-locomotion:latest` (or specify the image name/tag in the `docker run` commands below).

## Running on Jetson

**Important**: Checkpoint files are not included in the Docker image. You must mount your checkpoint directory as a volume using `-v /path/to/checkpoints:/workspace/checkpoints`. All examples below assume checkpoints are mounted at `/workspace/checkpoints` inside the container.

**Camera-specific mounts**:
- If the active front camera driver is `zed`, include both mounts on runs after pre-optimization and initial calibration fetch: `-v ~/zed-resources/resources:/usr/local/zed/resources -v ~/zed-resources/settings:/usr/local/zed/settings` (single host root with hardcoded subfolders).
- If the active front camera driver is `maixsense_a075v`, that ZED resources mount is not required.

**Optional: Enable data collection with host persistence**

Create the host folder first:

```bash
mkdir -p /path/to/krabby_bags
```

Set `--data-collector-output-dir` to enable recording. The mount target must match the same container path:

```bash
sudo docker run --rm --runtime=nvidia  -v /path/to/checkpoints:/workspace/checkpoints  -v /path/to/krabby_bags:/workspace/bags  -v /dev:/dev  --privileged  krabby-locomotion:latest  --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt  --robot go2  --data-collector-output-dir /workspace/bags
```

### Important: GPU Runtime Flag

**On Jetson, use `--runtime=nvidia` instead of `--gpus all`:**

```bash
# Correct for Jetson
sudo docker run --rm --runtime=nvidia <image> <command>

# If you get "unknown or invalid runtime name: nvidia", reconfigure:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Basic Usage

Run the production inference runner:

```bash
sudo docker run --rm --runtime=nvidia  -v /path/to/checkpoints:/workspace/checkpoints  -v /dev:/dev  krabby-locomotion:latest  --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt  --robot go2
```

**Note**: 
- The container entrypoint is `hal.server.jetson.main`, so you can pass arguments directly without specifying the Python module.
- Replace `/path/to/checkpoints` with the actual path to your checkpoint directory on the host.
- Ensure the checkpoint file exists at the specified path (e.g., `/path/to/checkpoints/unitree_go2_parkour_teacher.pt` must exist on the host).
- Optional: set process logging verbosity with `--log-level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`), for example `--log-level DEBUG`.

## Troubleshooting

### Permission Denied Errors

If you see "permission denied while trying to connect to the docker API":
1. Verify you're in the docker group: `groups`
2. If not, run `newgrp docker` or log out/in
3. Verify docker socket permissions: `ls -l /var/run/docker.sock` (should show `docker` group)

### NVIDIA Runtime Not Found

If you get "unknown or invalid runtime name: nvidia":
1. Verify nvidia-container-toolkit is installed: `which nvidia-ctk`
2. Reconfigure: `sudo nvidia-ctk runtime configure --runtime=docker`
3. Restart Docker: `sudo systemctl restart docker`
4. Verify: `sudo docker info | grep -i runtime`

### Docker iptables / networking errors

If containers fail with iptables-related errors on Jetson (some kernels lack modules Docker expects for default bridge networking), run the container with **`--network host`** when that fits your deployment, and adjust how you expose or bind ports accordingly.

### CUDA Out of Memory

If you encounter CUDA OOM errors:
- Reduce batch size if applicable
- Use TensorRT optimization (see `export_to_tensorrt.py`)
- Ensure no other processes are using GPU memory
- Check GPU memory usage: `nvidia-smi`

### ZED Camera Not Detected

- Verify camera is connected: `lsusb | grep ZED`
- The ZED SDK uses USB (not only V4L2). Ensure the container is run with `-v /dev:/dev` and `--privileged` so it can access `/dev/bus/usb`. `--device /dev/video0` alone is often insufficient.

## Production Deployment Notes

This section lists Jetson production runtime readiness checks for hardware deployment:

- **JetPack/L4T**: Use a JetPack release compatible with the container assumptions in this repo (6.1/6.2 with L4T 36.4 is the tested baseline).
- **Container runtime**: Use `--runtime=nvidia`; use `--network host` when required by your Jetson networking setup (see Docker iptables / networking errors above).
- **Permissions**: Ensure the deployment user can run Docker commands without interactive escalation.
- **Camera reachability**: Confirm each required camera endpoint is reachable from the Jetson host before starting the container.
- **Checkpoint + storage**: Ensure checkpoint and bag output mounts exist on the host before launch.


