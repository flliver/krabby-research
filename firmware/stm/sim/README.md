# Host board simulation (`sim/`)

Isaac / unit-test backends for `board_pins_t` and virtual CAN.

| File | Role |
|------|------|
| `board_pins_isaac.c` | `board_pins_isaac_create(node_id)` — joint state for one leg in sim |
| `can_sim.c` | `vcan0` RX/TX queues (`can_init`, `can_send`, `can_receive`) |

Built into `libkrabby_stm_host.so` via `make -C firmware host-stm`. Test hooks: `board_pins_isaac.h`, `can_sim.h`.