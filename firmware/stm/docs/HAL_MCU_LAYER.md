# HAL ↔ MCU boundary

## Purpose

Contract between **`hal/`** and leg CAN transport when **`--stm`** / **`KRABBY_MCU_STM`** is active.

## Components

- **`hal.client`:** `JointCommand`
- **`RobotDefinition.mcu_joints`** — eighteen HAL names
- **Serial:** `KrabbyMCUSDK` / `krabby_mcusdk.py` (radians → Mega PWM)
- **CAN (target):** `KrabbyStmMCUSDK` — duty on wire via **`SET_DUTY`**
- **Names:** `firmware/stm/hal_joint_names.py` — HAL name ↔ `(leg_id, joint_id)`

## Critical points

### Encoding split

| Backend | `JointCommand` on wire |
|---------|------------------------|
| Mega serial | Radians → PWM (**510** scale) |
| **`--stm`** | **Duty −1…1** per joint; **no** radian conversion in SDK |

### `KrabbyStmMCUSDK` (target)

- Build table from **`mcu_joints`** → `(leg_id, joint_id)`.
- **`apply_command`:** clamp to **±1.0**, one **`set_duties()`** → **`SET_DUTY`** each tick.
- **`connect`:** dual bus or `virtual=True` on `vcan0`.
- Reader fills **`joints[name]`** from **`TELEMETRY_*`** decode.

### Dual-bus send

Every **`SET_DUTY`** goes to **both** `can0` and `can1`; boards ignore other legs’ bytes (`CAN_PROTOCOL.md`).

### Isaac fullsim

**`--firmware-transport can`:** attach six sim boards on **`vcan0`**; HAL steps env; boards do not call `env.step()` themselves (`VIRTUAL_CAN.md`).

### Gamepad / model with `--stm`

Mappers pass **duty** (full stick → **1.0**); without `--stm`, Mega serial still uses radian → PWM encoding.

### Regression

Mega path unchanged when **`KRABBY_MCU_STM`** unset. Lazy-import CAN deps only when **`stm=True`**.

## Reference

Jetson init site: `hal/server/jetson/hal_server.py`. Isaac transport: `hal/server/isaac/command_backend_cli.py`.