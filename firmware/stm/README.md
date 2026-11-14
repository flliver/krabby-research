# Leg controller firmware (STM32)

MCU application, host Python, and **`krabby.dbc`** for six leg boards.

## Specifications

[`docs/README.md`](docs/README.md) · [`dbc/krabby.dbc`](dbc/krabby.dbc)

## Layout

| Path | Role |
|------|------|
| `core/` | `can_handler`, `telemetry_handler`, `calibration` (target) |
| `board/` | `board_pins_stm32`, `can_stm32`, `main` (target) |
| `sim/` | `board_pins_isaac`, `can_sim` (target) |
| `cli/` | `krabby-firmware-stm` |
| `codec.py`, `ids.py`, `can_transport.py` | Host CAN stack |
| `dbc/` | `krabby.dbc`, `generated/`, `make stm-dbc` |
| `bootloader/` | OpenBLT layout from `hardware/PCB-v0.3/Firmware/` (preserve linker regions) |

## Build

```bash
make -C firmware stm-dbc
make -C firmware stm-build       # MCU leg app (OpenBLT layout from hardware/PCB-v0.3/Firmware/)
make -C firmware host-stm        # libkrabby_stm_host for sim/tests
```

## Host

```bash
pip install -e "firmware[stm]"
pytest tests/unit/firmware/stm/
```