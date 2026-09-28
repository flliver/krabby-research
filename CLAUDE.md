# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Locomotion stack for the Krabby hexapod robot: Arduino firmware for the motor-driver boards, a Hardware Abstraction Layer (HAL) that lets the same policy run against simulation or real hardware, a parkour RL inference runtime, teleop/gamepad control, a data collector, and deployment tooling (`krabby` CLI + bench watchdog).

The robot has three Arduino Mega 2560 boards (roles: `front`, `left`, `right`) each driving up to 6 motors via BTS7960 H-bridges, talked to over USB serial from a Jetson Orin. See `docs/GLOSSARY.md` for domain terms (joint codes like `FLHL`, IS/current-sense, CAL, etc.) — check it before asking the user to define jargon.

## Commands

### Python packages (host-side)

Almost every subtree (`hal/client`, `hal/server`, `hal/server/isaac`, `hal/server/jetson`, `hal/tools`, `compute/parkour`, `controller`, `data_collection`, `teleop/edge`, `teleop/portal`, `krabby`, `bench`) is its own pip package with its own `pyproject.toml`.

```bash
# One-time: create ./testenv (or activate your own venv — an active VIRTUAL_ENV wins)
make venv

# One-time: install all HAL/compute/controller/data_collection packages editable
make install-editable

# Build all wheels (used for Docker images / publishing)
make build-wheels

# Remove all dist/build/egg-info + __pycache__
make clean
```

Every root Makefile target except `venv` errors out unless a venv exists (activated or `./testenv`) — run `make venv` first on a fresh clone. Bare `make` runs `test`. Docker image targets: `build-locomotion-image`, `build-isaacsim-image`, `build-test-image`, `build-test-image-arm` (all depend on `build-wheels`).

### Tests

Full suite runs inside a Docker test image and needs GPU passthrough:

```bash
make test              # builds x86 test image, runs pytest tests/ -m "not jetson and not isaacsim"
make test-coverage     # same, with coverage report under tests/coverage/
make test-isaacsim     # Isaac Sim container, requires a policy checkpoint
```

For fast local iteration without Docker, run pytest directly against `./testenv`:

```bash
testenv/bin/pytest tests/unit/ -v
testenv/bin/pytest tests/unit/firmware/test_cli.py -v   # single file
testenv/bin/pytest tests/unit -k test_name -v           # single test
```

Pytest markers (`pytest.ini`): `jetson` (needs Jetson/HAL/ZED/MaixSense hardware or its mocks) and `isaacsim` (needs a real Isaac Sim env). Both are excluded by default in `make test`.

`tests/unit/` mirrors the top-level package layout (`tests/unit/firmware/`, `tests/unit/hal/`, `tests/unit/controller/`, etc.). `tests/integration/` holds cross-component tests. `tests/helpers.py` has shared fixtures like `create_dummy_hw_obs`.

### Firmware (Arduino sketch)

```bash
make -C firmware compile-firmware                  # compile only
make -C firmware upload-firmware                    # compile + flash local board (auto-detects PORT)
make -C firmware upload-firmware PORT=/dev/ttyACM0 PIN_REV=1

# Flash a board attached to a remote host over SSH (e.g. the Jetson with the USB hub)
make -C firmware flash-remote REMOTE=user@orin PORT=/dev/ttyACM0
make -C firmware flash-remote-all REMOTE=jetson       # flash every discovered board, same hex on all (role lives in EEPROM)

# Push host-side Python (menu/SDK/CLI) to a bench host without reflashing
make -C firmware sync-remote REMOTE=krabby-orin
```

Builds always bake `-DSERIAL_RX_BUFFER_SIZE=256 -DKRABBY_PIN_REV=<n>` (see comments in `firmware/Makefile`) so `make`-built and CI-built binaries are identical regardless of host OS or AVR core version. Requires `arduino-cli` on PATH; `pyserial` for port auto-detect.

The firmware Python SDK/CLI (`firmware/krabby_mcu.py`, `firmware/cli.py`, entry point `python -m firmware`) talks to boards over serial for bring-up: `show`, `update` (OTA-style flash from S3), `set`/`get` (EEPROM config), `calibrate-all`/`calibrate-joint`/`get-calibration`, `validate-current-sense`, `jog`, `observe`. See `firmware/SETUP.md` (wiring, protocol) and `firmware/COMMS_DEBUG.md` (leader/follower serial debugging history) before changing serial protocol or timing-sensitive code.

### Deployment CLIs

```bash
krabby install / update / run / firmware <subcommand>   # on-robot: pulls/runs the locomotion Docker image (krabby/README.md)
krabby-bench install                                     # bench watchdog: polls ECR, smoke-tests firmware, alerts (bench/README.md)
```

No linter/formatter is configured in this repo — don't assume `ruff`/`black`/`mypy` are wired into CI.

## Architecture

### HAL (Hardware Abstraction Layer) — `hal/`

The boundary that lets the *same* policy run against simulation or real hardware. Wheel-based packages, directory structure matches import namespace (`hal/client/` → `from hal.client import ...`):

- `hal/client/` (`krabby-hal-client`) — `HalClient`: polls latest `HardwareObservations`, sends `JointCommand`. Generic types only (`NavigationCommand`, `HardwareObservations`, `JointCommand`).
- `hal/server/` (`krabby-hal-server`) — `HalServerBase`, shared ZMQ server logic.
  - `hal/server/isaac/` (`krabby-hal-server-isaac`) — `IsaacSimHalServer`, entry point `krabby-hal-server-isaac`.
  - `hal/server/jetson/` (`krabby-hal-server-jetson`) — `JetsonHalServer` (real sensors: ZED camera, robot state), entry `python -m hal.server.jetson.main`. Drives the full production loop and can start optional data-collector / teleop-signaling threads.
- `hal/tools/` (`krabby-hal-tools`) — `hal-dump` CLI for debugging.

Model-specific types (`ParkourObservation`, `ParkourModelIO`, `InferenceResponse`) and hardware↔model mappers live in `compute/parkour/`, **not** in `hal/` — HAL stays generic.

Transport is ZMQ, `inproc://` within one process (production/tests default) or `tcp://` for cross-process. Two channels: observation (PUB/SUB, latest-only via `SNDHWM=1`/`RCVHWM=1`+`CONFLATE=1`) and joint commands (PUSH/PULL, ordered with `HWM=5`). Full diagram and timing budget (100 Hz / 10 ms tick) in `docs/RUNTIME_ARCHITECTURE.md`.

### Inference — `compute/parkour/`

Production policy runtime, shared across simulation/testing/production images. `ParkourInferenceClient` polls `HalClient` → maps `HardwareObservations` to model tensors (`mappers/hardware_to_model.py`) → runs `ParkourPolicyModel` → maps the action tensor back to `JointCommand` (`mappers/model_to_hardware.py`).

### `parkour/` — RL training (separate from `compute/parkour/`)

IsaacLab-based training/eval code (`parkour_isaaclab/`, `parkour_tasks/`, `scripts/rsl_rl/`) for the crab-hexapod policy. Not part of the runtime stack; see `docs/crab-hexapod-policy-config.md`.

### Firmware — `firmware/`

`firmware/arduino/` is the actual sketch (`arduino.ino` + headers: `actuator_manager.h`, `board_pins.h`, `command.h`, `eeprom_layout.h`, `hall_hw.cpp/h`). One sketch image flashed identically to all three boards — role (front/left/right) and per-joint calibration live in EEPROM (`eeprom_layout.h`), not in the compiled firmware. A "leader" board forwards follower telemetry over Serial1/Serial2.

Host-side Python (`firmware/krabby_mcu.py` = SDK, `firmware/cli.py` + `firmware/__main__.py` = CLI, `firmware/gui/` = Tk-based bench GUI) drives bring-up over serial: show/update/set/get/calibrate/jog/observe. `firmware/manifest.py` + `firmware/scripts/publish_firmware.py` handle the S3-backed firmware manifest/channel system used by `cmd_update` and the bench watchdog's smoke test.

`firmware/observation_mapping.py` holds the raw-MCU-telemetry → HAL-observation transforms (per-leg current → contact_forces, joint-velocity EMA). It is deliberately dependency-free (no numpy/torch) because both the Jetson HAL server and the lightweight `observe` bench command import it — keep it that way.

### Controller — `controller/`

On-robot gamepad/bring-up scripts. `control_loop.py` is the core poll→map→send loop; `mappers/` and `input/` handle gamepad-to-command mapping; `cli/cli_uno.py` / `cli_uno_sim.py` are console entry points against real hardware vs. simulation.

### Teleop — `teleop/`

WebRTC-based remote drive/viewing. `teleop/edge/` runs on the robot (signaling client, QoS, video/depth capture); `teleop/portal/` is the relay/signaling server (auth, ICE config, static web client).

### Data collection — `data_collection/`

`HalDataCollector` (optional thread started from the Jetson HAL server) records rotating bag files (`rotating_bag.py`) of `HardwareObservations`/commands for offline analysis; `serialization.py` defines the on-disk format.

### `krabby/` and `bench/` — deployment

`krabby/` is the `krabby-launcher` PyPI package: the `krabby` CLI that installs/updates/runs the locomotion Docker image on the Jetson host (`_docker.py`, `_host.py`), including udev rules and a `krabby-locomotion.service` systemd unit for boot autostart.

`bench/` is `krabby-bench`: a systemd watchdog that polls ECR for new locomotion images, updates+smoke-tests the deployed stack, and alerts (email/GitHub issue) on failure. Pulls credentials from AWS SSM Parameter Store by default.

### `images/`

Dockerfiles for the four runtime targets: `images/locomotion/` (Jetson production: `compute/parkour` + `hal/server/jetson`), `images/isaacsim/` (x86 sim), `images/testing/x86/` and `images/testing/arm/` (mock-HAL test containers). All install the wheels built by `make build-wheels`; built via the root `make build-*-image` targets.

### `hardware/` and `scripts/` — not code

`hardware/` is reference material: EasyEDA PCB projects (`.epro`), wiring diagrams, BOM, and motor datasheets for the Uno chassis revisions. `scripts/` holds host setup and demo launchers (`scripts/jetson/setup-docker-gpu.sh` for Docker GPU passthrough, Isaac Sim demo/run scripts). `set-ssm-params.sh` at the repo root seeds the AWS SSM parameters the bench watchdog reads.

## Key docs

| Doc | Covers |
|---|---|
| `docs/RUNTIME_ARCHITECTURE.md` | Container patterns, ZMQ transport/contract layers, data flow, timing |
| `docs/HAL_GUIDE.md` | HAL usage in depth |
| `docs/FOLDER_LAYOUT.md` | HAL wheel package structure, import patterns |
| `docs/GLOSSARY.md` / `docs/TECHNOLOGY_AND_TERMINOLOGY.md` | Domain terms and deeper explanations |
| `docs/SENSOR_INTERFACE.md`, `docs/MAIXSENSE_A075V_SETUP.md` | Jetson sensor backends |
| `docs/JETSON_DEPLOYMENT.md`, `docs/PUBLISHING.md`, `docs/DOCKER_DEPENDENCIES.md` | Deployment/publish pipeline |
| `docs/DATA_COLLECTOR.md`, `docs/TELEOP.md` | Data collection and teleop subsystems |
| `firmware/SETUP.md`, `firmware/COMMS_DEBUG.md`, `firmware/MOTOR_HEADER_PINOUT.md` | Firmware wiring, protocol, debugging history |
| `krabby/README.md`, `bench/README.md` | CLI and bench watchdog reference |
| `DEVELOPER.md` | Host machine setup: CUDA/driver versions, Python 3.11 venv, Isaac Sim/Lab install |
