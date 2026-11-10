# Host integration (`krabby-firmware --stm`)

## Purpose

Single **`krabby-firmware`** PyPI package hosts Mega serial **and** leg-board CAN tools; this doc defines layout, CLI, encoding, fleet hooks, and flash UX.

## Components

- **Package:** `krabby-firmware` (`firmware/pyproject.toml`, entrypoint `firmware.__main__:main`)
- **Serial stack:** `firmware/krabby_mcu.py`, `firmware/mcu_port.py`, `firmware/cli.py`
- **CAN stack:** `firmware/stm/` beside `firmware/sim/` (host plant sim, not leg CAN)
- **DBC:** `firmware/stm/dbc/krabby_leg.dbc`
- **HAL boundary:** [`HAL_MCU_LAYER.md`](./HAL_MCU_LAYER.md) — Jetson / Isaac / GUI use `mcu_can`, not `krabby_mcu`, when **`--stm`** / **`KRABBY_MCU_STM`** is active
- **Scripts:** `firmware/stm/scripts/can_setup_orin.sh`, `firmware/stm/scripts/sweep_18_joints.py`

## Critical points

- **One package, two transports:** default CLI remains **USB serial → three Mega boards**. Flag **`--stm`** selects **SocketCAN-FD → six leg nodes**. Passing **`--stm`** never opens serial for MCU command/telemetry paths.
- **Activation:**
  - CLI: **`krabby-firmware --stm <subcommand>`** (global flag before subcommand).
  - Long-running services: env **`KRABBY_MCU_STM=1`** (same semantics as **`--stm`**).
  - **`krabby run`** / **`krabby-locomotion`** image: set **`KRABBY_MCU_STM=1`** when the robot uses leg CAN boards; leave unset for USB serial Mega controllers.
- **Python dependencies:** base package keeps **`pyserial`**. Optional **`[project.optional-dependencies] stm`** in **`pyproject.toml`** (also pinned in locomotion release requirements):
  - **`python-can`** ≥ 4.3 — SocketCAN, FD frames, dual interface
  - **`cantools`** ≥ 39.0 — load `krabby_leg.dbc`, encode/decode `LegCommand` / `LegTelemetry`
- **Locomotion image:** `images/locomotion/requirements.release.txt` adds **`python-can`**, **`cantools`**, and bumps **`krabby-firmware`** pin when leg CAN support ships; container still carries Mega flash tools for serial-only fleets.

### `firmware/stm/` layout (host Python)

| Module | Role |
|--------|------|
| `ids.py` | Leg index, HAL `mcu_joints` names, CAN ID helpers (`PROVISIONING.md`) |
| `can_transport.py` | Robot: `can0`+`can1`; bench: one iface; FD bitrate from `CAN_PROTOCOL.md` |
| `codec.py` | cantools DB singleton; encode command frames, decode telemetry |
| `mcu_can.py` | **`KrabbyMCUCANSDK`** — joint dict cache, connect/disconnect, jog/position send, reader thread |
| `cli_stm.py` | `show`, `provision`, `jog`, `telemetry`, `update` handlers |
| `flash.py` | OpenBLT streamer over CAN (`FLASH.md`) |
| `provision.py` | `ProvisionSet` TX/RX |
| `scripts/can_setup_orin.sh` | Robot SocketCAN-FD bring-up |
| `scripts/sweep_18_joints.py` | Eighteen-joint verification sweep |

C firmware and **`board_ids.h`** live under the same tree for MCU builds; host imports **`ids.py`** only.

### Shared joint encoding

- **Names:** eighteen strings **`FL_hip_yaw` … `RR_knee`** — same tuple as `RobotDefinition.get_mcu_joints()` and `ROBOT_SENSORS.md`.
- **Position command:** HAL **radians** on the policy path → host normalizes to **0.0–1.0** with the same per-joint limits as `hal/server/jetson/krabby_mcusdk.py` → DBC position fields (`LEG_BOARD_CONTROL.md`).
- **Jog command:** signed PWM **−255…255** per joint → DBC jog fields; CLI interactive keys reuse Mega name→PWM mapping internally, then expand to **`mcu_joints`** before encode.
- **Telemetry decode:** `LegTelemetry` → `firmware.interfaces.joint_telemetry.JointTelemetry` (position, pot, current, connection) for GUI and tests.
- **Routing:** `mcu_can` maps each joint name → `(bus, leg_index, joint_index)` via `ids.py`; one SDK instance fans out to six nodes (`HAL_MCU_LAYER.md`).

### CLI (`krabby-firmware --stm`)

| Subcommand | Serial (`krabby-firmware`) | Leg CAN (`krabby-firmware --stm`) |
|------------|---------------------------|-------------------------------|
| *(none)* | Interactive Mega key menu | Interactive CAN jog menu (same key legend, CAN backend) |
| `show` | USB Mega + S3 branch list | Six-leg discovery table (`PROVISIONING.md`) |
| `provision` | — | Assign `FL`…`RR` |
| `jog` | — | Scripted jog (`--joint`, PWM) for automation |
| `telemetry` | — | Stream decoded joints + `NodeStatus` |
| `update` | avrdude `.hex` → Mega port(s) | OpenBLT `.elf` over CAN (below) |
| `install` | udev / serial permissions | udev rules for serial; CAN via **`can_setup_orin.sh`** + group `dialout`/`netdev` as site requires |

Global flags with **`--stm`:** `--debug` (frame logging), `--interface can0` (bench override), `--leg <code>` (target one node on multi-command ops).

### `update` behavior (`--stm`) — normative UX (wire details in `FLASH.md`)

1. **Preconditions:** CAN interfaces up (`scripts/stm/can_setup_orin.sh` on robot). All targeted nodes respond to **`DiscoveryRequest`** (provisioned legs only for production update; bench may target **`UNK`** after single-board bring-up).
2. **Artifact source:** same S3 bucket family as serial (`krabby-firmware-public`), leg-board keys **`leg_app.elf`** (application) and optional **`leg_bootloader.elf`** in the branch build record. Cache path: **`~/.cache/krabby-firmware/<branch>/<commit>/leg_app.elf`** (mirror serial HEX cache layout).
3. **Channel argument:** `krabby-firmware --stm update` → latest `release/*` build; `update <branch>` → explicit branch; no serial port argument.
4. **Scope flags:**
   - **Default:** flash **all six** provisioned legs (parallel sessions per bus: up to three nodes on `can0`, three on `can1`).
   - **`--leg <code>`:** single leg **`FL` | `FR` | `ML` | `MR` | `RL` | `RR`** only.
5. **Per-node sequence:** host commands **bootloader entry** (OpenBLT CAN handshake, `FLASH.md` IDs) → stream **`.elf`** segments → **verify** → **reset to application** → wait for **`NodeStatus`** with new firmware identity string.
6. **NV preservation:** provisioning byte and joint calibration (`LEG_BOARD_CONTROL.md`) must remain valid after app update; host does not re-provision unless operator runs **`provision`**.
7. **Progress output:** one line per leg: `FL: downloading …`, `FL: flashing 42%`, `FL: done (1.2.3)`; non-zero exit if any leg fails; successful legs stay on new firmware.
8. **Power loss mid-update:** operator re-runs the **same** `update` command; bootloader resumes OpenBLT session per `FLASH.md`. Same retry-until-complete UX as serial **`update`**.
9. **Implementation note:** until `flash.py` is complete, `update` may support **`--dry-run`** only; operator behavior above is the contract either way.

### GUI (`krabby-firmware-gui`)

- Flag **`krabby-firmware-gui --stm`** (and **`KRABBY_MCU_STM=1`**) constructs GUI against **`KrabbyMCUCANSDK`** instead of **`KrabbyMCUSDK`**.
- Joint grid uses **`mcu_joints`** labels; telemetry poll reads CAN cache at the same **50 ms** UI cadence as serial.
- Jog buttons emit CAN jog frames; no USB port picker when **`--stm`** (interface from env; optional **`~/.config/krabby/can.toml`** in implementation).

### `krabby run` / locomotion service

- Jetson **`hal.server.jetson.main`** chooses MCU backend from **`KRABBY_MCU_STM`**: serial **`KrabbyMCUSDK`** path vs CAN **`KrabbyMCUCANSDK`** + dual-bus router (`HAL_MCU_LAYER.md`).
- **`krabby run`** selects the MCU backend via **`KRABBY_MCU_STM=1`** in the locomotion systemd unit / fleet env for CAN leg robots.
- Gamepad / policy **`JointCommand`** uses the same HAL API; serial vs CAN is selected by **`KRABBY_MCU_STM`** (`KrabbyMCUSDK` vs `KrabbyMCUCANSDK`).

### Environment variables (leg CAN)

| Variable | Purpose |
|----------|---------|
| `KRABBY_MCU_STM` | `1` → CAN backend everywhere the serial SDK was used |
| `KRABBY_CAN_IFACES` | Robot: `can0,can1` |
| `KRABBY_CAN_IFACE` | Single iface: `can0` bench hardware; `vcan0` bench sim (`CAN_PROTOCOL.md`) |

## Reference

### S3 artifact fields (leg-board branch build)

| Key | File | Consumer |
|-----|------|----------|
| `leg_app_elf_url` | `leg_app.elf` | `update` default |
| `leg_boot_elf_url` | `leg_bootloader.elf` | factory / recovery flash |
| `leg_app_sha256` | — | cache verify |

Exact JSON field names follow the leg-board fields added to `firmware/manifest.py` in CI publish.

### Dependency pins (target)

```
# pyproject.toml [project.optional-dependencies] stm =
python-can>=4.3.0
cantools>=39.0.0
```

Locomotion **`requirements.release.txt`**: add the same pins alongside **`krabby-firmware==<version>`**.
