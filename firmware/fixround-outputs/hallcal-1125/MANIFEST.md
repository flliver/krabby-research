# hallcal-1125: build-only hexes of the calibration fix round

Built on the desktop build host (fractal-ubuntu) on 2026-11-25. They are build-only: nothing was flashed and no serial port was opened.

These replace the earlier hallcal-1125 builds, which are superseded: source `dc98f280` (Rev 3 sha256 `f89578c0…`, Rev 2 `0e9cdec7…`) and source `0a7d3177` (Rev 3 `d0746e84…`, Rev 2 `378a4890…`; same code, older comments). This build adds the host check's fixes: the knee stroke guard before the park (F1), the deeper re-home WARN, and items a-f.

- **Source:** branch `feat/hall-calibration-end-detection-fixround`, commit `bf500eea`. The hexes are committed in the commit after it.
- **Version strings,** checked with `strings` on each `.elf`:
  - `KRABBY_FW_VERSION` = `hallcal-1125`;
  - `KRABBY_FW_BRANCH` = `feat/hall-calibration-end-detection-fixround`;
  - `KRABBY_FW_COMMIT` = `bf500eea`.

  They come from a temporary `firmware/arduino/version.h`, restored with `git checkout` after each build. The committed file still reads "dev-local".

| File | Board | sha256 | Flash | RAM (static) | Free RAM (compile-time) |
|---|---|---|---|---|---|
| `hallcal-1125_rev3.hex` | PIN_REV 3 (quadrature halls) | `9e0cfdcc6c3d8fa5a3aaa47a979f194c44c9929e039e4b07b580a65cc103b1ab` | 83,018 B | 4,881 B | 3,311 B |
| `hallcal-1125_rev2.hex` | PIN_REV 2 (pots only) | `76e3ea14e5ee8438416a8b574155273457238515e939cd26c92bcb8dab8f5a51` | 80,228 B | 4,859 B | 3,333 B |

**Untagged comparison** (same commit, the committed "dev-local" `version.h`):
- Rev 3: 82,962 B flash, 4,825 B RAM, 3,367 B free.
- Rev 2: 80,172 B flash, 4,803 B RAM, 3,389 B free.

The tagged builds are 56 B larger in flash and 56 B in RAM, which is the size of the three version strings. Free RAM stays above the 3,000 B floor, which applies to the compile-time figure (owner decision F5).

## Build command

Run from the worktree root, once per revision, with `firmware/arduino/version.h` temporarily holding the three defines above:

```bash
make -C firmware -o arduino-core -o fetch-libs compile-firmware ARDUINO_CLI=/home/nickmagus/krabby/fwtools/bin/arduino-cli PIN_REV=3
make -C firmware -o arduino-core -o fetch-libs compile-firmware ARDUINO_CLI=/home/nickmagus/krabby/fwtools/bin/arduino-cli PIN_REV=2
```

The output is `firmware/build/arduino.ino.hex`, copied out after each build.

## Toolchain

- arduino-cli 1.1.1 (commit fa6eafcb) with `arduino:avr` 1.8.8.
- Libraries in `~/Arduino/libraries`:
  - SparkFun Qwiic 6Dof LSM6DSO 1.0.4;
  - SparkFun Qwiic OLED 1.0.13;
  - SparkFun INA2XX 1.0.0;
  - **SparkFun Toolkit 1.2.1**, not the Makefile's pinned 1.2.0. Installing INA2XX pulls in the newest Toolkit as a dependency, which is a known quirk.

## Note

Rev 3 is the hall robot's board. On Rev 2 every joint is a pot, so it keeps the old SWEEP order: its hip still runs to its uncurl end, which is not table-safe.
