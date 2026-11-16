# OpenBLT (bootloader)

Bootloader sources and CubeMX/OpenBLT project files land here. The shipped memory map and CAN programming flow live in [`../docs/FLASH.md`](../docs/FLASH.md).

**Linker regions** for the leg MCU build must match the OpenBLT project under `firmware/stm/uno-v0.3/` until this tree carries its own copy. Do not move the application base (`0x08006000`) or NV page (`0x0801F800`) without updating both trees and `FLASH.md`.