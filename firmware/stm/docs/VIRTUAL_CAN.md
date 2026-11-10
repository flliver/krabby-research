# Virtual CAN (host simulation)

## Purpose

Exercise **leg CAN-FD + DBC** on **`vcan0`** without STM32 hardware — Isaac fullsim, bench tests, and CI decode/routing checks.

## Components

- **SocketCAN virtual iface:** **`vcan0`** (kernel `vcan` module)
- **Sim leg cluster:** Python processes speaking **`krabby_leg.dbc`** on one bus (`firmware/stm/sim/`)
- **Host HAL path:** `firmware_transport='can_vcan'` → **`KrabbyMCUCANSDK`** (`HAL_MCU_LAYER.md`)
- **DBC:** identical to hardware (`../dbc/krabby_leg.dbc`, `CAN_PROTOCOL.md` IDs)
- **Not in scope:** **`firmware/sim/robot_core.py`** Mega pin loop (separate **`in_process`** fullsim path)

## Critical points

### Two simulation paths (do not merge)

| Path | MCU model | Bus | Isaac HAL `firmware_transport` |
|------|-----------|-----|--------------------------------|
| **Pin / Mega fullsim** | Host AVR shim + `robot_core.py` | Virtual USB serial | `in_process`, `serial_sdk` |
| **Leg CAN fullsim** | **Virtual leg nodes** on **`vcan0`** | CAN-FD + DBC | **`can_vcan`** |

**`robot_core.py`** stays the pin-sim game loop. **Virtual CAN** replaces USB/serial with **`vcan0`** and does **not** compile STM32 firmware on the host; nodes are **Python DBC speakers** backed by Isaac joint state.

### Virtual leg node count

| Mode | Nodes on `vcan0` | Use |
|------|------------------|-----|
| **Robot parity (default)** | **6** provisioned legs **`FL`…`RR`** | Isaac fullsim, routing tests, `show` with six rows |
| **Single-node bench** | **1** node (**`FL`**, index 0) | CLI jog/telemetry, codec unit tests |
| **Unassigned bench** | **1** node, `LEG_UNASSIGNED` / index 6 | `provision` flow tests (`PROVISIONING.md`) |

Six-node cluster shares **one** `vcan0` (bench sim mode in `CAN_PROTOCOL.md`). Host **`KRABBY_CAN_IFACE=vcan0`**; **`KRABBY_CAN_IFACES`** unset on sim.

### DBC parity

- Encode/decode with the **same** `krabby_leg.dbc` as hardware; tests pin **DBC hash** from the repo pin file.
- Bit timing / FD flags on `vcan0` match robot settings (`CAN_PROTOCOL.md`: FD on, 500k / 2M nominal config in `can_transport.py`).
- Message sets: **`LegCommand`**, **`LegTelemetry`**, **`NodeStatus`**, **`DiscoveryRequest`**, **`ProvisionSet`** — no sim-only IDs.

### Virtual node behavior (per leg)

Each **`VirtualLegNode`** (index 0–5):

- RX **`LegCommand`** for its leg index → update internal jog/position targets for **three** joints.
- TX **`LegTelemetry`** at **100 Hz** with positions from **Isaac articulation** when attached; otherwise integrated stub positions.
- TX **`NodeStatus`** at **1 Hz** + on **`DiscoveryRequest`** (firmware identity string **`sim-<git-sha>`**).
- Honors **100 ms** command stale → safe telemetry (zero PWM, hold position).
- Does **not** implement OpenBLT — **`FLASH.md`** frames are ignored in sim.

Cluster entrypoint **`firmware/stm/sim/can_leg_cluster.py`** owns six nodes; **`--legs 1`** runs single-node mode for CI.

### Isaac single-joint jog rule

When Isaac HAL uses **`can_vcan`**:

1. `CanCommandLoop.apply_hal_command` receives full **`JointCommand`** (18 radians).
2. Encoder sets **non-zero jog PWM only** on joints where **|rad| > `NEUTRAL_RAD_THRESHOLD`**; **all other** joints in every `LegCommand` frame get **jog 0** and unchanged position fields.
3. **At most one** joint is non-neutral per HAL tick during Isaac jog tests (teleop/policy may send one active joint per tick).
4. **`VirtualLegNode`** applies PWM **only** to joints with non-zero jog in the RX frame; Isaac **`advance_sim`** moves **only** the mapped articulation DOFs for that joint (`firmware/stm/sim/isaac_joint_map.py`).
5. Other five legs still **emit telemetry** (held pose); their Isaac joints do not move.

### `vcan0` bring-up

**Bench / Isaac host (Linux):**

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 mtu 72
sudo ip link set up vcan0
```

**Robot mode** uses physical **`can0`/`can1`** — not this section.

Teardown: `sudo ip link delete vcan0` when finished.

### Harness scope

| Layer | Runs without Isaac | Runs in CI |
|-------|--------------------|------------|
| `codec.py` roundtrip | yes | yes |
| `can_leg_cluster` + recorded frames | yes | yes |
| `KrabbyMCUCANSDK` on `vcan0` | yes | yes (vcan required) |
| Isaac `can_vcan` fullsim | no | optional nightly (GPU) |
| Mega `robot_core` / pin sim | separate | separate |

**CI (no Isaac):** start cluster in background, run **`tests/integration/firmware/stm/test_vcan_*.py`**, jog one joint, assert single-leg motion in cluster metrics.

### Attachment through HAL (`HAL_MCU_LAYER.md`)

1. Test or Isaac launcher starts **`can_leg_cluster.py`** on **`vcan0`**.
2. Export **`KRABBY_MCU_STM=1`**, **`KRABBY_CAN_IFACE=vcan0`**.
3. Construct **`IsaacSimHalServer`** with **`firmware_transport='can_vcan'`**.
4. HAL opens **`KrabbyMCUCANSDK`**; Isaac scene registers joint map from virtual nodes to env DOFs.
5. Shutdown: HAL `close()` → cluster SIGTERM.

Jetson production does **not** use virtual CAN; GUI developers may point **`--stm`** at **`vcan0`** with cluster running locally.

### Operator quick start (no Isaac)

```bash
# terminal A
python -m firmware.stm.sim.can_leg_cluster --iface vcan0 --legs 6

# terminal B
export KRABBY_MCU_STM=1 KRABBY_CAN_IFACE=vcan0
krabby-firmware --stm show
krabby-firmware --stm jog --joint FL_hip_yaw --pwm 120
krabby-firmware --stm telemetry
```

## Reference

### Leg index on `vcan0` (six-node mode)

| Leg | Index | Receives ID offset |
|-----|-------|-------------------|
| `FL` | 0 | `+0` |
| `FR` | 1 | `+1` |
| `ML` | 2 | `+2` |
| `MR` | 3 | `+3` |
| `RL` | 4 | `+4` |
| `RR` | 5 | `+5` |

Host routes all six on **`vcan0`** (logical bus); dual physical buses apply only on robot hardware.

### Implementation files

| Path | Role |
|------|------|
| `firmware/stm/sim/virtual_leg_node.py` | One DBC leg |
| `firmware/stm/sim/can_leg_cluster.py` | Cluster launcher (`--legs 1`, `--legs 6`) |
| `firmware/stm/sim/isaac_joint_map.py` | HAL DOF names per `mcu_joints` |
| `hal/server/isaac/firmware_can_loop.py` | `CanCommandLoop` |
| `tests/integration/firmware/stm/test_vcan_*.py` | Harness |

### Environment (sim)

| Variable | Value |
|----------|--------|
| `KRABBY_MCU_STM` | `1` |
| `KRABBY_CAN_IFACE` | `vcan0` |
| `KRABBY_VCAN_LEGS` | `6` (cluster); `1` for minimal |
