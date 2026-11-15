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

Requires **sudo** once per machine (or use the lifecycle script below).

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 mtu 72
sudo ip link set up vcan0
```

**Lifecycle script (recommended):** creates the link only when missing, records ownership, and
**deletes `vcan0` on teardown only if this script created it** — pre-existing interfaces are left alone.

```bash
firmware/stm/scripts/vcan0-lifecycle.sh up      # create / fix mtu / up
firmware/stm/scripts/vcan0-lifecycle.sh down    # delete only if we created it
firmware/stm/scripts/vcan0-lifecycle.sh run -- CMD   # up, run CMD, down on exit
```

### CLI on the bus (no Isaac)

`candump vcan0` in one terminal, then `krabby-firmware-stm --virtual duty 1 0.5 0 0` in another.

### Target operator flow (Isaac sim on the bus)

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