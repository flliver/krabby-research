# End-to-End Test: Pro Controller → Krabby HAL (Two-Process)

Two-process E2E on Jetson Orin: **Pro Controller** → **ControlLoop (INPUT_CONTROLLER_KRABBY)** → **HalClient (ZMQ TCP)** → **Jetson HAL server** → **KrabbyMCUSDK** → **firmware/krabby_mcu.py**. No camera or inference; command path only.

---

## Canonical path: `krabby` CLI

```bash
pip install krabby-launcher
krabby install              # pull mainline-latest, set up udev + dialout
krabby firmware show        # verify all three boards are visible
krabby run --gamepad-only   # start the locomotion stack in gamepad mode
```

`krabby run --gamepad-only` starts the container with `--privileged -v /dev:/dev`, which passes through the MCU serial ports (`/dev/ttyACM*`, `/dev/ttyUSB*`) **and** all input devices (`/dev/input/*`) in a single flag — no separate `--device` arguments needed.

With the container running, start the control-loop client on the host (or inside the container):

```bash
# host (pip-installed client)
pip install krabby-controller
krabby-uno

# or, if running from the repo
python controller/scripts/jetson/run_gamepad_to_krabby_client.py
```

Pair the Pro Controller over Bluetooth before starting the container (see [CONNECT_PRO_CONTROLLER.md](CONNECT_PRO_CONTROLLER.md)). The paired `/dev/input/js0` is available inside the container automatically because of `-v /dev:/dev`.

### Testing without a checkpoint (gamepad-only mode)

**Terminal 1 — HAL server in container (gamepad-only):**

```bash
krabby run --gamepad-only
```

Wait for: `HAL server started in gamepad-only mode ... Run \`krabby uno\` to connect.`

**Terminal 2 — control-loop client on host:**

```bash
PYTHONPATH=/tmp/krabby-research python3 \
  -m controller.scripts.jetson.run_gamepad_to_krabby_client
```

Or with the pip-installed client: `krabby-uno` (defaults to `tcp://localhost:6001/6002`).

---

## Manual / debug path (raw docker run)

## Prerequisites

- **Pro Controller** connected (USB or Bluetooth). Verify: `python -m controller.input --list`
- **Jetson HAL (MCU)** connected (serial). **Firmware** package: `firmware.krabby_mcu` (see [firmware/](../../../firmware/)).
- Server **exits with error** if firmware or MCU not available.
- When using the helper scripts to start the container, the scripts check that the MCU device (default `/dev/ttyACM0`) exists before starting; if not, they exit with a clear error. For a different port, set `KRABBY_MCU_PORT` (e.g. `export KRABBY_MCU_PORT=/dev/ttyUSB0`).

It is recommended to use a Python virtual environment to isolate dependencies: create with `python -m venv .venv`, then activate with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).

## Communication

- **Server:** Binds ZMQ — PUB observation, PULL command (e.g. `tcp://*:6001`, `tcp://*:6002`).
- **Client:** Connects — SUB observation, PUSH command (e.g. `tcp://localhost:6001`, `tcp://localhost:6002`).

## Steps

### Terminal 1: HAL server

```bash
krabby run --gamepad-only
```

Custom bind ports: `krabby-hal-server-jetson --control-source gamepad --observation-bind tcp://*:6001 --command-bind tcp://*:6002`

Wait for: `HAL server started in gamepad-only mode ... Run \`krabby uno\` to connect.`

### Terminal 2: Control-loop client

Run the client in either of these ways:

**From repo (krabby-research):**

```bash
python controller/scripts/jetson/run_gamepad_to_krabby_client.py
```

**Alternative (pip):** In an activated virtual environment, install and run the client via the published package:

```bash
pip install krabby-controller
krabby-uno
```

Explicit endpoints: `--observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002`. Optional: `--device-id N`, `--rate` (Hz).

## Gamepad mapping

- **Legs:** LT / LB / LS / RS / RT / RB ([gamepad_to_krabby_hal_mapper.py](../../mappers/gamepad_to_krabby_hal_mapper.py)).
- **Axes:** Left stick Y = hip; left stick X = knee; right stick Y = hip yaw.

## Stopping

**Ctrl+C** in either terminal.

## Verification

- Server: Missing firmware/MCU → error log and non-zero exit.
- Client: Pro Controller input → joint commands over ZMQ; server applies via KrabbyMCUSDK → firmware.

---

## Running with the locomotion container

The locomotion image includes the controller and firmware source so you can run the gamepad-only HAL server inside the container and the control-loop client on the host (or in the container with gamepad device access).

### Build on Orin

From **krabby-research** on the Jetson Orin:

1. Build wheels: `make build-wheels` (requires venv; see Makefile).
2. Build the locomotion image: `make build-locomotion-image` or:
   `docker build -f images/locomotion/Dockerfile -t krabby-locomotion:latest .`


### How to run (two options)
To run the HAL server and client, there are 2 options: (1) Run HAL server in container and client in virtual python environment or (2) Run both HAL server and client in the container in detached mode (one command, no second terminal). The section below details both options.

The helper scripts use `--rm` (a **start** option for `docker run`) so the container is removed when it stops. That avoids leaving stopped containers and prevents container logs from accumulating on disk.

### Method 1 - Run Hal Server in container and client in virtual python environment
#### Terminal 1: Gamepad-only HAL server in container

Override the default entrypoint to run the gamepad-only server:

```bash
docker run --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint krabby-hal-server-jetson \
  krabby-locomotion:latest \
  --control-source gamepad \
  --observation-bind tcp://*:6001 --command-bind tcp://*:6002
```

- `--privileged`: required so the container can open the serial device (e.g. `/dev/ttyACM0`) for the MCU; otherwise you may see "Operation not permitted".
- `-v /dev:/dev`: serial access for MCU (e.g. `/dev/ttyACM0`).
- Alternatively: `./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh`

#### Terminal 2: Control-loop client (on host)

On the Orin host (with a venv activated), connect the Pro Controller to the host and run the client. You can use the repo script or the pip-installed `krabby-uno` (after `pip install krabby-controller` in that venv):

```bash
python controller/scripts/jetson/run_gamepad_to_krabby_client.py \
  --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002
```

Or with the pip client: `krabby-uno --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002` (defaults are already these endpoints).

The client talks to the server in the container via localhost.

### Method 2 - Run both Hal Server and Client in the container (detached)
Tip: This is the easiest way to run Hal Server and Client.

Both the HAL server and the control-loop client run in **detached** mode inside one container. No second terminal is needed. The container sees the host’s `/dev/input/*` via `-v /dev:/dev` and `--privileged`, so connect the Pro Controller to the host **before** starting the container.

**Step 1 – Start both server and client (detached)**

```bash
docker run -d --name hal-gamepad --rm --privileged --runtime=nvidia \
  -v /dev:/dev \
  -p 6001:6001 -p 6002:6002 \
  --entrypoint /workspace/controller/scripts/jetson/helper/run_server_and_client_in_container.sh \
  krabby-locomotion:latest
```
Alternatively: `./controller/scripts/jetson/helper/start_gamepad_hal_and_client_detached.sh`

**Stopping.** From the host run: `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh`. This stops the container and thus both server and client. (With `--rm` in start script, the container and logs are removed after it stops.) If you don't have the repo locally, you can stop the container with Docker directly: `docker stop hal-gamepad`. If the container was started without `--rm`, remove it afterward with `docker rm hal-gamepad`.

**Where to see the logs**

- `docker logs hal-gamepad` — view logs from the container (server and client output; may be interleaved).
- `docker logs -f hal-gamepad` — follow logs in real time.

In disk, the container and logs are under /var/lib/docker/containers.

**Alternative: two terminals.** If you prefer to run the server detached and the client interactively in a second terminal, use `./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh`, then in another terminal `./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh`. Stop with `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh` after Ctrl+C in the client terminal.

**Troubleshooting:**
- If the client reports no controller, ensure the Pro Controller is connected to the host before `docker run`, and that the container was started with `-v /dev:/dev` and `--privileged`.
- **Docker permission denied:** If `docker exec` (or `docker run`) fails with "permission denied" connecting to the Docker socket, add your user to the `docker` group on the host: `sudo usermod -aG docker $USER`, then run `newgrp docker` or log out and back in. See [JETSON_DEPLOYMENT.md](../../../docs/JETSON_DEPLOYMENT.md) Troubleshooting.

---

### Using the scripts

Helper scripts in `controller/scripts/jetson/helper/` (run from **krabby-research** repo root):

**Server only (client on host)**  
- **Start server:** `./controller/scripts/jetson/helper/run_gamepad_hal_server_only_in_container.sh` — foreground; Ctrl+C stops the container.  
- **Client:** Run on host (see [Terminal 2: Control-loop client (on host)](#terminal-2-control-loop-client-on-host)).

**Client inside the container**  
- **Start both (detached):** `./controller/scripts/jetson/helper/start_gamepad_hal_and_client_detached.sh` — server and client run in one container; no second terminal.  
- **Stop:** `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh`  
- **Logs:** `docker logs hal-gamepad` or `docker logs -f hal-gamepad`  
- **Alternative (two terminals):** Start server (detached): `./controller/scripts/jetson/helper/start_gamepad_hal_server_container_detached.sh`; run client in second terminal: `./controller/scripts/jetson/helper/run_gamepad_client_in_container.sh`; stop after Ctrl+C in client: `./controller/scripts/jetson/helper/stop_gamepad_hal_container.sh`
