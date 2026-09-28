# Isaac Sim joystick demo runbook

**Overview.** This runbook describes how to test Isaac Sim with a Nintendo Pro Controller (or a compatible gamepad). First, run the HAL server in joystick mode, then launch the krabby-uno-sim client on the host.

## Pairing (Ubuntu)

- `bluetoothctl`: power on, scan-on, pair `<MAC>`, trust `<MAC>`
- Verify: `python -m controller.input --list`

## Prerequisites

The Isaac Sim HAL server must run in an environment where **Isaac Sim** and **Isaac Lab** are available. Two options:

- **Docker (recommended):** Build the image with `make build-isaacsim-image` (requires `make isaaclab-cache` and `make build-wheels`). Then run the server inside the container (see Launch). The client (`krabby-uno-sim`) runs on the host and connects to the container via TCP. When using Docker, set up GPU access first: run `./scripts/jetson/setup-docker-gpu.sh` from the repo root, or see [DOCKER_DEPENDENCIES.md](../../../docs/DOCKER_DEPENDENCIES.md) (GPU Support Setup).
- **Native:** Install Isaac Sim and Isaac Lab per [DEVELOPER.md](../../DEVELOPER.md), then run the module with the Isaac Lab Python (e.g. from the Isaac Lab repo: set `PYTHONPATH` to include the krabby-research repo root and run `./isaaclab.sh -p python -m hal.server.isaac.main ...`).

## Launch

1. Start Isaac Sim HAL server in joystick mode (minimal 640×360 window, no Isaac Lab UI). Supported: **quad (12-joint Go2)** with the parkour task, or **hexapod (18-joint)** using the crab hex USD (`assets/crab_hex_ref.usd`).

   **Option A – Docker** (from **krabby-research**, after `make build-isaacsim-image`; publish 5555/5556):

   For a **visible Isaac Sim window**, the container needs the host display. The script below already passes `-e DISPLAY` and `-v /tmp/.X11-unix` to `docker run`—do not pass those to the script (they are Docker options, not app options). Run the script with no arguments, or only app options (e.g. `--seed 0`). If you use a raw `docker run` without those flags, the server runs **headless** and you will see: `DISPLAY environment variable is not set, running in headless mode`.

   **Go2 (quad):**
   ```bash
   ./scripts/run_isaac_hal_server.sh
   ```
   **Hexapod (crab_hex_ref.usd):** The script mounts `assets` and uses `--usd` (task Isaac-CrabHex-Joystick-v0, 18 joints). Start the client with **`krabby-uno-sim --hex`**.
   ```bash
   ./scripts/run_isaac_hal_server.sh --hexapod
   ```
   Add `--debug` for verbose logs (e.g. per-command joint values). Manual **Go2** run (includes display):
   ```bash
   xhost +local:docker 2>/dev/null
   docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     krabby-isaacsim:latest --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
   ```
   Manual **hexapod** run (mount assets, use `--usd`; client must use `krabby-uno-sim --hex`):
   ```bash
   xhost +local:docker 2>/dev/null
   docker run --rm --gpus all -p 5555:5555 -p 5556:5556 \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v "$(pwd)/assets:/workspace/assets" \
     krabby-isaacsim:latest --joystick --usd /workspace/assets/crab_hex_ref.usd
   ```

   **Option B – Native** (Isaac Lab env, `PYTHONPATH` includes krabby-research root):
   ```bash
   ./isaaclab.sh -p python -m hal.server.isaac.main --joystick --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0
   ```
   Hexapod (from repo root, so `assets/crab_hex_ref.usd` is available):
   ```bash
   ./isaaclab.sh -p python -m hal.server.isaac.main --joystick --usd assets/crab_hex_ref.usd
   ```

   Server binds TCP 5555/5556 by default.

2. Start the **client** in a **second terminal** (install once: `pip install -e ./hal/client` and `pip install -e ./controller`). Use **`--quad`** for Go2 (12 joints) or **`--hex`** for the crab hex (18 joints):
   ```bash
   krabby-uno-sim --quad
   ```
   For the hexapod server (e.g. `--hexapod` or `--usd ...`), use:
   ```bash
   krabby-uno-sim --hex
   ```
   Add `--debug` to log every gamepad→joint command and mapper activity.

   The **Pro Controller is used by the client**, not the server. The server only logs when it receives commands from the client (e.g. `Joystick: first command received`, `Joystick: command applied`). If you move the controller but never start the client, you will see no joystick-related logs on the server.

   Client waits up to 10 min for a gamepad by default (`--gamepad-wait 600`). Override endpoints with `--observation_endpoint tcp://127.0.0.1:5555 --command_endpoint tcp://127.0.0.1:5556` if needed.

In joystick mode the viewport camera is set closer to the robot (eye 0, 1.2, 0.8) so the robot is clearly visible. Move the joystick to confirm joints in the sim.

## Where to see logs

- **Server (Docker):** The terminal where you ran `./scripts/run_isaac_hal_server.sh` **is** the Docker container’s logs (stdout/stderr). You don’t need a separate command to “see Docker logs.” Add `--debug` to that script to get per-command debug lines on the server.
- **Client:** The terminal where you run `krabby-uno-sim --quad` shows client logs. Add `--debug` there to see every gamepad→joint command.
- **When you move the Pro Controller:** The controller is read by the **client** (krabby-uno-sim). The client sends commands to the server. So you only see server-side debug logs (and robot movement) when **both** are running: (1) server in one terminal, (2) `krabby-uno-sim --quad` in a second terminal. Then move sticks/buttons—server terminal will show “Client connected…” and every 5 s “Joystick: N steps…”, and with `--debug` on the server you’ll see “Joint command received: …” for each command.

## Stopping

- **HAL server (Docker):** Ctrl+C in the server terminal often does not stop Isaac Sim. **Use `docker stop` from another terminal:** run `docker ps` to get the container ID (e.g. `abc123def456`), then `docker stop <container_id>`. The container will exit and be removed (`--rm`).
- **HAL server (native):** Press **Ctrl+C** in the terminal running `hal.server.isaac.main`. The process logs "Received interrupt signal, stopping..." and shuts down.
- **Client (krabby-uno-sim):** Press **Ctrl+C** in the client terminal. It logs "Received interrupt, stopping..." and exits.

## Leg selection

LT/LB/LS/RS/RT/RB and combos (LT+LB, RT+RB) follow the mapping in **Appendix B** below. Use `krabby-uno-sim --quad` for 12 joints (FL, FR, RL, RR) or `krabby-uno-sim --hex` for 18 joints (FL, FR, ML, MR, RL, RR). Axis scaling matches Jetson by default: full stick (±1) → ±0.3 rad (hip/knee) and ±0.2 rad (hip yaw); override with `--mapper-hip-scale`, `--mapper-knee-scale`, `--mapper-hip-yaw-scale` if needed.

## Troubleshooting

- **No Isaac Sim window (Docker):** Use `./scripts/run_isaac_hal_server.sh` (it sets up the display), or a manual `docker run` with `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix` and `xhost +local:docker`. Without the display, the app runs headless.
- **`unrecognized arguments: -e DISPLAY -v /tmp/.X11-unix`:** You passed Docker flags to the script; those are for `docker run` only. Run `./scripts/run_isaac_hal_server.sh` with no args (or only app args like `--seed 0`).
- **`Robot joint count (18 for --robot hex) does not match the task's action dimension (12)`:** You started the server with a Go2 task but the client is in hex mode (or vice versa). For the hexapod (crab_hex_ref.usd), use `./scripts/run_isaac_hal_server.sh --hexapod` and client `krabby-uno-sim --hex`. For Go2, use the script without `--hexapod` and client `krabby-uno-sim --quad`.
- **No server logs when moving the Pro Controller:** Start `krabby-uno-sim` in a second terminal. The server only logs when it **receives** commands from the client. Normal server messages: "Client connected (joint command received).", then every 5 s "Joystick: N steps in X.Xs (~XX Hz)". If you see none of these, ensure the server was started with `--joystick` and the client is running and connected.
- **Robot doesn’t move when I move the Pro Controller:** The robot moves only when the **client** sends commands to the server. You must run **`krabby-uno-sim --quad`** (Go2) or **`krabby-uno-sim --hex`** (hexapod) in a **second terminal** (on the host, with the controller connected). The server terminal should show "Client connected (joint command received)." when the client connects, then "Joystick: N steps…" every 5 s while you move the sticks. If the server shows **"Joint command received: 12 joints"** (or **18 joints**) and **"all positions zero"**, the client is sending zeros—**select a leg** (e.g. hold **LT** for front-left, or **RT** for front-right) and move the **left stick** (Y/X = hip/knee); with `--debug` on the server you’ll then see "non-zero positions (joint=rad): …". If you don’t select a leg, all joints stay at 0 and the robot doesn’t move.
- **Client sends commands but server shows no "Client connected" or joystick logs:** The server only logs when it **receives** a command. If the client logs "Sent joint command" but the server never shows "Client connected", the commands are not reaching the server. (1) Ensure the server is fully started—wait until you see **"Joystick: main loop ready, waiting for first command on tcp://*:5556"** in the server terminal before assuming connectivity. (2) From the host, verify the command port is reachable: `nc -zv 127.0.0.1 5556` (or `python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 5556)); print('5556 open'); s.close()"`). (3) If you changed HAL or server code, rebuild the Docker image (`make build-isaacsim-image`) so the container runs the latest code.

---

## Appendix B: Gamepad mapping (quad + hexapod)

This appendix defines the Pro Controller → leg/joint mapping used by both the Jetson hexapod HAL path and the IsaacSim joystick path. The same rules are implemented in:

- `controller/mappers/gamepad_to_krabby_hal_mapper.py` (Jetson hexapod, real hardware)
- `controller/mappers/gamepad_to_isaacsim_hal_mapper.py` (IsaacSim quad and hexapod)

and are referenced from the Jetson E2E doc `[controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md](controller/scripts/jetson/E2E_GAMEPAD_KRABBY.md)`.

### B.1 Leg selection (LT/LB/LS/RS/RT/RB)

Legs are selected using the six shoulder/stick buttons:

- **Single-leg selection (no combo pressed):**
  - **LT** (without LB): Front Left (**FL**)
  - **LB** (without LT): Rear Left (**RL**)
  - **LS**: Middle Left (**ML**)
  - **RS**: Middle Right (**MR**)
  - **RT** (without RB): Front Right (**FR**)
  - **RB** (without RT): Rear Right (**RR**)

- **Tripod combos:**
  - **LT + LB**: tripod **left** → {FL, RL, MR}
  - **RT + RB**: tripod **right** → {FR, RR, ML}

If a leg does not exist on the robot (e.g. ML/MR on the quad), it is ignored. For the hexapod, all six legs FL, FR, ML, MR, RL, RR are available.

### B.2 Axes → joints

Stick axes map to joint control axes as:

- **Left stick Y (`LY`)**: hip up/down (hip pitch)
  - Internally mapped as `hip_up_down = -LY` (up on the stick = positive angle).
- **Left stick X (`LX`)**: knee out/in (knee)
  - Internally mapped as `knee_out_in = LX`.
- **Right stick Y (`RY`)**: hip yaw forward/back (hip yaw)
  - Internally mapped as `hip_yaw = RY`.

For every selected leg, the same axis values are applied:

- `hip_pitch` joint ← `hip_up_down * hip_up_down_scale`
- `knee` joint ← `knee_out_in * knee_out_in_scale`
- `hip_yaw` joint ← `hip_yaw * hip_yaw_scale`

The **joint ordering** for the hexapod is:

- Legs: **FL, FR, ML, MR, RL, RR**
- Per leg: `(hip_yaw, hip_pitch, knee)`

This gives 18 joints total, in the order used by `KRABBY_HEX_DEFINITION`:

`FL_hip_yaw, FL_hip_pitch, FL_knee, FR_hip_yaw, FR_hip_pitch, FR_knee, ML_hip_yaw, ML_hip_pitch, ML_knee, MR_hip_yaw, MR_hip_pitch, MR_knee, RL_hip_yaw, RL_hip_pitch, RL_knee, RR_hip_yaw, RR_hip_pitch, RR_knee`.

### B.3 Scaling

Scaling is the same for both Jetson and IsaacSim by default (same feel on hardware and in sim):

- **Hip up/down:** 0.3 rad per unit stick (full stick ±1 → ±0.3 rad)
- **Knee out/in:** 0.3 rad per unit stick
- **Hip yaw:** 0.2 rad per unit stick

Jetson: defined in `controller/mappers/gamepad_to_krabby_hal_mapper.py`. IsaacSim: the mapper receives scale values from the control loop; the client **`krabby-uno-sim`** sets these defaults (0.3, 0.3, 0.2) and passes them via `ControlLoopConfig`. Override with `--mapper-hip-scale`, `--mapper-knee-scale`, `--mapper-hip-yaw-scale` if needed.
