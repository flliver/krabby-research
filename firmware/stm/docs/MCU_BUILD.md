# MCU cross-build (`stm-build`)

## Purpose

Produce **`krabby_leg_app.elf`** for STM32G474 leg boards using **`firmware/stm`** plus the reference CubeMX project under **`hardware/PCB-v0.3/Firmware/`**.

## What you need

| Item | Notes |
|------|--------|
| **CMake** | ≥ 3.22 (same as `firmware/Makefile`) |
| **ARM GNU toolchain** | `arm-none-eabi-gcc`, `arm-none-eabi-g++`, `arm-none-eabi-objcopy` on **`PATH`** |
| **Repo paths** | `hardware/PCB-v0.3/Firmware/` present (linker, startup, HAL sources) |
| **Generated DBC C** | `firmware/stm/dbc/generated/` — run `make -C firmware stm-dbc` if missing |

Ninja is **not** required for `make -C firmware stm-build` (default CMake generator is fine). Full builds inside **`hardware/PCB-v0.3/Firmware/`** may use Ninja; see that tree’s README.

Reference firmware was audited with **ST GNU 14.3.1+st.2**; other recent **arm-none-eabi** GCC 12–14 builds are usually fine.

## Install toolchain

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi
arm-none-eabi-gcc --version
```

Distro packages are often older; for parity with the reference audit, use the [ST STM32Cube CLT / GNU toolchain](https://www.st.com/en/development-tools/stm32cubeclt.html) and add its `bin` directory to `PATH`.

### macOS

```bash
brew install arm-none-eabi-gcc
arm-none-eabi-gcc --version
```

### Windows

Install [STM32CubeCLT](https://www.st.com/en/development-tools/stm32cubeclt.html) or [xPack GNU Arm Embedded GCC](https://xpack.github.io/dev-tools/arm-none-eabi-gcc/), then ensure `arm-none-eabi-gcc` is on `PATH` in the shell you use for `make`.

## Build steps

From repo root:

```bash
make -C firmware stm-dbc          # once, or when krabby.dbc changes
make -C firmware stm-build
```

- If **`arm-none-eabi-gcc`** is on `PATH`: configures with `firmware/stm/cmake/gcc-arm-none-eabi.cmake`, links **`firmware/build/stm-leg/krabby_leg_app.elf`** using the reference OpenBLT linker script.
- If the toolchain is **missing**: Make prints a warning and runs a **host** smoke link only (no `.elf` for flash).

Override CubeMX tree location:

```bash
make -C firmware stm-build KRABBY_HW_FIRMWARE=/path/to/Firmware
```

## Verify

```bash
file firmware/build/stm-leg/krabby_leg_app.elf    # expects ARM, EABI5
arm-none-eabi-size firmware/build/stm-leg/krabby_leg_app.elf
```

## Reference

| Topic | Path |
|--------|------|
| OpenBLT / flash layout | [FLASH.md](./FLASH.md), `firmware/stm/bootloader/README.md` |
| Reference motor/bootloader builds | `hardware/PCB-v0.3/Firmware/README.md` |
| Board port stubs | `firmware/stm/board/README.md` |