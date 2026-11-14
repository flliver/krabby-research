# MCU board port (`board/`)

STM32G474 leg controller: pin HAL, FDCAN2, and `main.c` entry.

| File | Role |
|------|------|
| `board_pins_stm32.c` | `board_pins_t` for DRV8262, pots, halls |
| `can_stm32.c` | `can_init` / `can_send` / `can_receive` |
| `main.c` | `app_init(board_pins_stm32())` + `app_tick` loop |

HAL bodies copy from `hardware/PCB-v0.3/Firmware/` when the MCU build is wired to CubeMX. Build with `make -C firmware stm-build`.