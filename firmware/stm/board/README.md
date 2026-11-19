# MCU board port (`board/`)

STM32G474 leg controller: pin HAL, FDCAN2, and `main.c` entry.

| File | Role |
|------|------|
| `stm32_bringup.c` | VTOR, `HAL_Init`, clocks, `stm32_cubemx_periph_init`, `MX_FDCAN2` |
| `stm32_cubemx_init.c` | CubeMX GPIO/ADC/TIM/I2C (from `firmware/stm/uno-v0.3/Src/main.c`) |
| `board_pins_stm32.h` | `board_pins_stm32_poll()` from `main` loop |
| `hw_handles.h` | `hfdcan2`, app flash base |
| `can_stm32.c` | FDCAN start, IRQ RX queue, `can_send` / `can_receive` |
| `board_pins_stm32.c` | `board_pins_t` for DRV8262, pots, halls |
| `main.c` | `stm32_hw_init()` → `app_init` → `app_tick` loop |

HAL bodies copy from `firmware/stm/uno-v0.3/` when the MCU build is wired to CubeMX.

**Build:** `make -C firmware stm-build` — toolchain install: [`../docs/MCU_BUILD.md`](../docs/MCU_BUILD.md)

- With **`arm-none-eabi-gcc`** on `PATH`: cross-builds **`krabby_leg_app.elf`** using the reference **OpenBLT** linker script (`STM32G474xx_FLASH.ld`), CubeMX startup, and HAL objects from `firmware/stm/uno-v0.3/` (override **`KRABBY_HW_FIRMWARE`** if that tree relocates; sync `firmware/stm/cmake/` + Makefile).
- Without the toolchain: host compile only (smoke link for CI); no `.elf`.