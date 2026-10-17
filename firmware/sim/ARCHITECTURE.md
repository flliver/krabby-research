# Firmware simulation architecture

Host-side simulation runs **shipping control logic** (`actuator_manager.h`, `arduino.ino` on
the host build) against a **simulated Arduino hardware API**, then drives Isaac Lab joints
from recorded pin writes. The control loop and serial protocol are the same sources that
flash to the Mega; only GPIO, timing, and bus I/O are swapped for sim backends.

## Flat stack

Only these public layers; everything else is implementation detail inside them.

```text
  shipping firmware (ActuatorManager, arduino.ino main loop on host)
           |  calls Arduino API (digitalWrite, analogWrite, millis, …)
  host Arduino.h shim  →  pin state model  (replaces AVR GPIO/PWM)
           |
  SimulatedRobotConfig   →  table: joint name, Mega pins, Isaac joint
           |
  RobotCore              →  tick firmware, read pins, set Isaac targets, env.step()
```

Isaac HAL **`--backend fullsim`** attaches three-board host firmware to the same scene; it
does **not** insert another tier between firmware and pins.

Later bus changes (e.g. CAN on leg MCUs) should swap **shim + config columns** (pins → bus
IDs) while keeping **firmware + config table + RobotCore** shape — avoid a separate “bus
provider” framework on top of this stack.

## Host build vs AVR build

| Build | Command | What runs |
|-------|---------|-----------|
| **AVR (robot)** | `make -C firmware compile-firmware` | `arduino-cli`, Mega2560; real `Arduino.h`; flash via `upload-firmware`. Unchanged by sim work. |
| **Host (sim)** | `make -C firmware host-actuator` | CMake under `firmware/sim/host/`: host `Arduino.h`, `libkrabby_host`, compiled `actuator_manager` / firmware loop. Native tests: `make -C firmware test-native`. |
| **Python loop** | `firmware/sim/robot_core.py`, tests | Pybind/C ABI via `host_bridge.py`; Isaac fixtures in `firmware/sim/isaac_scene.py`. |

The host target links shipping firmware against `firmware/sim/host/` shims. Extend shims when
new `arduino.ino` paths join the host CMake target. The AVR target does not use the host
library.

## Repo map (main types)

| Layer | Location |
|-------|----------|
| Pin state | `host/pin_state.{hpp,cpp}`, `pin_state.py` |
| Arduino shim | `host/Arduino.h`, `arduino_io.hpp` |
| Config + actuators | `simulated_robot_config.py`, `simulated_actuator.py` |
| Game loop | `robot_core.py` |
| Full firmware on host | `host/host_firmware_loop.cpp`, `host_multi_board.cpp` |
| Three-board + SDK | `three_board_*.py`, [ROLE_BOOT.md](ROLE_BOOT.md) |
| Operator pin/joint table | [README.md](README.md) |
| Isaac HAL fullsim | [hal/server/isaac/README.md](../../hal/server/isaac/README.md) |

Tests refer to **joint names** (`FLHY`) and config slots, not scattered pin integers.

## Deliberately not added

- Plugin registries, driver factories, or generic “sim backend” interfaces
- A parallel firmware control loop written only for simulation
- Extra orchestration objects between `SimulatedRobotConfig` and Isaac
- CAN / leg-MCU simulation in the pin-sim tree (separate hardware path)

## Further reading

- Doc index (tests, Isaac, HAL fullsim): [DOCUMENTATION.md](DOCUMENTATION.md)
- Pin table, virtual serial: [README.md](README.md)
