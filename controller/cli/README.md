# krabby-uno and krabby-uno-sim CLIs

> **Running on a Jetson?** Use `krabby run` to start the locomotion stack — it handles GPU flags, device passthrough, and ZMQ ports automatically. See [images/locomotion/README.md](../../images/locomotion/README.md). The `krabby-uno` client below connects to the server started by `krabby run`.

## Install

When installing from source, install the HAL client first. From the **krabby-research** directory:

```bash
pip install ./hal/client
pip install ./controller
```

Use `pip install -e ./hal/client` and `pip install -e ./controller` for editable installs. With a venv activated, after both are installed, `krabby-uno` and `krabby-uno-sim` are on PATH. A single `pip install .` from the controller directory only works once `krabby-hal-client` is already installed.

## krabby-uno (real HAL)

1. **Start the HAL server** (one terminal, from `krabby-research`):
   E.g. 
   ```bash
   python controller/scripts/jetson/main_gamepad_only.py
   ```

   Server binds observation `tcp://*:6001` and command `tcp://*:6002` by default.

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
