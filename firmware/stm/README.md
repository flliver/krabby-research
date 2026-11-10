# Leg controller firmware (STM32)

MCU application, OpenBLT bootloader, host Python, and CAN schema for six identical leg boards.

## Specifications

[`docs/README.md`](docs/README.md) · [`dbc/krabby_leg.dbc`](dbc/krabby_leg.dbc)

## Layout

| Path | Role |
|------|------|
| `core/` | `board.h`, `can_handler`, `telemetry_handler`, `version.h` |
| `board/` | MCU `board_hw.c` (pins / drivers) |
| `sim/` | Host `board_sim.c` for `libkrabby_stm_host` |
| `cli/` | Optional `krabby-firmware-stm` entry (primary UX: `krabby-firmware --stm`) |
| `codec.py`, `mcu_can.py`, `can_transport.py`, `cli_stm.py` | Host CAN stack (stubs until CLI/SDK wired) |
| `ids.py`, `board_ids.h`, `include/krabby/` | Identity + version |
| `dbc/` | `krabby_leg.dbc`, `generated/`, `make stm-dbc` |
| `app/` | STM32 application (CubeMX / main) |
| `bootloader/` | OpenBLT (`docs/FLASH.md`) |
| `scripts/` | Orin CAN setup, joint sweep |

## Build

```bash
make -C firmware stm-dbc          # regen DBC C (repo .venv + cantools)
make -C firmware stm-firmware     # CMake scaffold (stub target by default)
make -C firmware host-stm         # libkrabby_stm_host.so for sim/tests
```

MCU image: `KRABBY_STM_BUILD_FIRMWARE=ON` + ARM toolchain → `firmware/build/stm-leg/`.

## Host Python

```bash
pip install -e 'firmware[stm]'    # python-can, cantools
krabby-firmware --stm …           # normative CLI (HOST_INTEGRATION.md)
krabby-firmware-stm help          # alias; prefer krabby-firmware --stm
```
