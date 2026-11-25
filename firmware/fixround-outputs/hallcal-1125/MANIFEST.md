# hallcal-1125: build-only hexes of the calibration fix round

Built on the desktop build host (fractal-ubuntu) on 2026-11-25. They are build-only: nothing was flashed and no serial port was opened.

These replace every earlier hallcal-1125 build, all superseded:
- `dc98f280`: Rev 3 sha256 `f89578c0…`, Rev 2 `0e9cdec7…`;
- `0a7d3177` and `bf500eea`: Rev 3 `d0746e84…` / `9e0cfdcc…`, Rev 2 `378a4890…` / `76e3ea14…` (a 761 home-relative knee bound);
- `0f2d199d`: Rev 3 `c55b5585…`, Rev 2 `a4242632…` (quiet band 769-801).

This build adds:
- the host's F1 design with its amendment: the knee parks 521 from its proven uncurl switch, its travel is quiet at 765-801 (S - 521 inside 244-280), WARNs at 745-764 or 802-825, and fails beyond, before the park;
- the knee-frame check and the extend-switch label;
- items a-f.

- **Source:** branch `feat/hall-calibration-end-detection-fixround`, commit `6cb51640`. The hexes are committed in the commit after it.
- **Version strings,** checked with `strings` on each `.elf`:
  - `KRABBY_FW_VERSION` = `hallcal-1125`;
  - `KRABBY_FW_BRANCH` = `feat/hall-calibration-end-detection-fixround`;
  - `KRABBY_FW_COMMIT` = `6cb51640`.

  They come from a temporary `firmware/arduino/version.h`, restored with `git checkout` after each build. The committed file still reads "dev-local".

| File | Board | sha256 | Flash | RAM (static) | Free RAM (compile-time) |
|---|---|---|---|---|---|
| `hallcal-1125_rev3.hex` | PIN_REV 3 (quadrature halls) | `d1d33cea508645a2f1fb5c9be62b8228bb2dfdc3b2f34f7a835d639b362f7c21` | 85,024 B | 4,881 B | 3,311 B |
| `hallcal-1125_rev2.hex` | PIN_REV 2 (pots only) | `b08ebb7cb87d38dcc2f2da408e664c68193e9daaccd85bbdaaa12903b345d770` | 82,208 B | 4,859 B | 3,333 B |

**Untagged comparison** (same commit, the committed "dev-local" `version.h`):
- Rev 3: 84,968 B flash, 4,825 B RAM, 3,367 B free.
- Rev 2: 82,152 B flash, 4,803 B RAM, 3,389 B free.

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
