# Firmware simulation — documentation index

Start with [ARCHITECTURE.md](ARCHITECTURE.md). Use this table to pick a doc and a command.

| Goal | Doc | Command |
|------|-----|---------|
| Stack layout (host vs AVR) | [ARCHITECTURE.md](ARCHITECTURE.md) | — |
| Pin/joint map, Makefile targets | [README.md](README.md) | `make -C firmware host-actuator` |
| Host C++ tests | — | `make -C firmware test-native` |
| Python firmware tests | — | `make -C firmware test-firmware` |
| Three-board / virtual USB / I2C sim | [ROLE_BOOT.md](ROLE_BOOT.md), [VIRTUAL_SERIAL.md](VIRTUAL_SERIAL.md), [I2C_SIM_SCOPE.md](I2C_SIM_SCOPE.md) | see linked docs |
| GUI + current-sense on sim MCU | [CURRENT_SENSE_SIM_DEMO.md](../gui/CURRENT_SENSE_SIM_DEMO.md) | demo script in that file |
| Isaac firmware pytest (GPU image) | [images/isaacsim/README.md](../../images/isaacsim/README.md) | `make -C firmware test-firmware-isaac` (needs `host-actuator` + image; verdict via `summarize_firmware_isaac_results.py`) |
| HAL **fullsim** (18 DOF hex) | [hal/server/isaac/README.md](../../hal/server/isaac/README.md) | `./scripts/run_isaac_hal_server.sh --hexapod --teleop-ip 127.0.0.1` |
| Fleet + Isaac fullsim smoke | [fleet/FULLSIM-FLEET-TELEOP-SMOKE.md](../../fleet/FULLSIM-FLEET-TELEOP-SMOKE.md) | procedure in doc (`krabby agent` + Isaac HAL, not `krabby run`) |
| Isaac image build | [images/isaacsim/README.md](../../images/isaacsim/README.md) | `make build-isaacsim-image` |

Physical Mega wiring (when correlating pins): [firmware/SETUP.md](../SETUP.md) § 1. HAL joint names: `hal/server/robot_definition_krabby_hex.py`.
