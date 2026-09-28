# krabby-uno and krabby-uno-sim CLIs

These entry points run **`ControlLoop`** as a **separate process** with a **local pygame gamepad** (`InputController`) and ZMQ TCP to the HAL server. They do **not** start WebRTC teleop or read the browser data channel.

> **Running on a Jetson?** Use `krabby run` to start the locomotion stack — it handles GPU flags, device passthrough, and ZMQ ports automatically, and already starts a `krabby-uno` client/controller inside the same container. See [images/locomotion/README.md](../../images/locomotion/README.md). The standalone `krabby-uno` below is for the two-process debug flow (connect a separate client to a server-only run) or running the client on another host.

## ControlLoop modes (what these CLIs use)

| CLI | `ControlLoop` mode | Input | HAL server (typical) |
|-----|-------------------|-------|----------------------|
| `krabby-uno` | `INPUT_CONTROLLER_KRABBY` | Pro Controller / gamepad (pygame) | `--control-source gamepad` |
| `krabby-uno-sim` | `INPUT_CONTROLLER_ISAACSIM` | Pro Controller / gamepad (pygame) | `--joystick` |

## Browser / WebRTC teleop (not these CLIs)

Remote driving from the **teleop portal** uses **`ControlLoop(INPUT_CONTROLLER_WEBRTC)`** inside the HAL process (`hal/server/teleop_portal_signaling.py`), not `krabby-uno` or `krabby-uno-sim`. There is no `--controller webrtc` flag on the CLIs below.

| Goal | Jetson | Isaac Sim |
|------|--------|-----------|
| **Browser gamepad → robot** | `krabby-hal-server-jetson --control-source portal --teleop-ip <portal-host>` | Isaac HAL with `--teleop-ip <portal-host>`; in the portal UI enable **operator_override** and drive |
| **Portal viewer / signaling** | `krabby-teleop-portal --host 0.0.0.0 --port 9000` on the operator host | same |

Chain on the robot: WebRTC `krabby-control-v1` → `WebRTCInputController` → **`ControlLoop` (`INPUT_CONTROLLER_WEBRTC`)** → `GamepadToKrabbyHALMapper` → `HalClient.put_joint_command`. On Isaac, **`operator_override`** gates sends via `ControlLoopConfig.command_send_gate` when local `--joystick` / inference is also active.

See [docs/TELEOP.md](../../docs/TELEOP.md) for protocol, latency, and CLI naming notes.

## Install

When installing from source, install the HAL client first. From the **krabby-research** directory:

```bash
pip install ./hal/client
pip install ./controller
```

Use `pip install -e ./hal/client` and `pip install -e ./controller` for editable installs. With a venv activated, after both are installed, `krabby-uno` and `krabby-uno-sim` are on PATH. A single `pip install .` from the controller directory only works once `krabby-hal-client` is already installed.

## krabby-uno (real HAL)

1. **Start the HAL server** (one terminal, on Jetson):

   ```bash
   krabby run --gamepad-only
   ```

   This runs `krabby-hal-server-jetson --control-source gamepad` inside the locomotion container with all required device passthrough. Server binds observation `tcp://*:6001` and command `tcp://*:6002` by default.

2. **Run the client** (second terminal):

   ```bash
   krabby-uno
   ```

   Defaults: observation `tcp://localhost:6001`, command `tcp://localhost:6002`. Override with `--observation_endpoint` and `--command_endpoint`. Use `--device-id` or `--InputController <id>` for a specific gamepad.

## krabby-uno-sim (IsaacSim)

The Isaac Sim HAL server must run in an environment that has Isaac Sim and Isaac Lab (Docker or native). **Recommended:** run the server inside the Isaac Sim Docker image. See [controller/scripts/isaac/isaacsim_demo_runbook.md](controller/scripts/isaac/isaacsim_demo_runbook.md) for Docker and native commands.

1. Start the Isaac Sim HAL server (joystick mode). Example with Docker (from **krabby-research** after `make build-isaacsim-image`):
   ```bash
   xhost +local:docker 2>/dev/null
   docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     krabby-isaacsim:latest --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
   ```
   For the hexapod (crab_hex_ref.usd), mount assets and use `--usd`; then run the client with `--hex`:
   ```bash
   docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v "$(pwd)/assets:/workspace/assets" \
     krabby-isaacsim:latest --joystick --usd /workspace/assets/crab_hex_ref.usd
   ```
   Or use `./scripts/run_isaac_hal_server.sh --hexapod` (see runbook).

2. Run the client (use `--quad` for Go2, `--hex` for crab hex):
   ```bash
   krabby-uno-sim --quad
   ```
   For hexapod server: `krabby-uno-sim --hex`
   Defaults: observation `tcp://127.0.0.1:5555`, command `tcp://127.0.0.1:5556`. Use `--InputController <id>` for a specific gamepad.

## Gamepad

List devices: `python -m controller.input --list`
