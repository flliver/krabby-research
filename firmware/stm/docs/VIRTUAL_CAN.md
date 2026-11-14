# Virtual CAN (host simulation)

## Purpose

Exercise **`krabby.dbc`** on **`vcan0`** without hardware — Isaac fullsim and unit tests.

## Components

- **`vcan0`** (kernel `vcan`, MTU **72**)
- Host **`krabby_stm_host`:** `board_pins_isaac.c`, `can_sim.c` (target layout)
- Isaac loop: **`firmware/sim/can_robot_core.py`**, **`run_can_sim`**
- HAL: **`KrabbyStmMCUSDK`** with `virtual=True`

## Critical points

- **CAN sim:** `firmware/sim/can_robot_core.py` drives six leg backends on **`vcan0`** in one Isaac env (`run_can_sim`).
- **`leg_id` 1…6** per board; every node shares the same virtual bus.
- **Same DBC and IDs** as hardware (`CAN_PROTOCOL.md`); pin enforced in tests.
- **FD:** bring-up matches robot (1M/5M intent); `is_fd=True`, `bitrate_switch=True` on send.
- **Acceptance shape:** `krabby-firmware-stm --virtual duty 1 1.0 0 0` moves only leg 1 joint 1 in Isaac; reverse duty reverses hall.

### `vcan0` bring-up

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 mtu 72
sudo ip link set up vcan0
```

### Target operator flow

```bash
python -m firmware.sim.run_can_sim --bus vcan0   # terminal A
krabby-firmware-stm --virtual duty 1 0.5 0 0   # terminal B
```

### Environment

| Variable | Sim value |
|----------|-----------|
| `KRABBY_CAN_IFACE` | `vcan0` |

## Reference

`firmware/stm/sim/` (`board_pins_isaac.c`, `can_sim.c`) and `firmware/sim/can_robot_core.py`.