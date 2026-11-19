# Flash over CAN (OpenBLT)

Leg-board programming uses **OpenBLT** on CAN-FD (see `firmware/stm/uno-v0.3/` for the shipped bootloader layout). Flash traffic is **not** defined in **`krabby.dbc`**; host **`krabby-firmware --stm update`** is not implemented yet.

## Purpose

In-field programming of leg-board **application** firmware over CAN-FD using **OpenBLT** on the board. Fleet **`update`** from Krabby CLI is future work (`HOST_INTEGRATION.md`).

## Components

- **Bootloader:** **OpenBLT** on **STM32G474**, CAN-FD transport + **XCP** programming protocol
- **Artifacts:** **`leg_app.elf`**, **`leg_bootloader.elf`** on S3 (`krabby-firmware-public`)
- **Host:** `firmware/stm/flash.py`, invoked from CLI **`update`**
- **Preserved NV:** provisioning + joint cal (`LEG_BOARD_CONTROL.md`, `PROVISIONING.md`)

## Critical points

- **Scope:** normal **`update`** flashes **application** only. **`leg_bootloader.elf`** is for factory/recovery (separate CLI flag **`--bootloader`**, manufacturing use); not run on every fleet OTA.
- **Dual artifact CI:** each release branch build publishes **`leg_app.elf`**, **`leg_bootloader.elf`**, **`leg_app_sha256`**, **`leg_boot_elf_sha256`** via `firmware/manifest.py` and `publish-firmware.yml`.
- **Memory model (linker scripts in `firmware/stm/` must match the OpenBLT layout in `firmware/stm/uno-v0.3/`):**

| Region | Flash start | Size | Erased on app OTA |
|--------|-------------|------|-------------------|
| Bootloader (OpenBLT) | `0x08000000` | **24 KiB** | **Never** by `update` |
| Application image | `0x08006000` | **100 KiB** | Yes (full app slot) |
| Config / NV mirror | `0x0801F800` | **2 KiB** | **Never** by `update` |

Application link base **`0x08006000`**. Config page holds a copy of leg/cal NV (`LEG_BOARD_CONTROL.md`); MCU also mirrors hot fields in RAM/EEPROM emulation — both must survive app programming. Bootloader update reprograms **`0x08000000`–`0x08005FFF`** only via **`--bootloader`**.
- **Boot entry (application running):**
  1. Host starts an OpenBLT programming session for **one** leg on its bus (OpenBLT CAN transport).
  2. Application finishes current CAN RX, de-energizes bridges (safe state), acknowledges, resets into OpenBLT.
  3. Within **200 ms**, the node is in **bootloader** mode (visible to the host via OpenBLT / CLI diagnostics).
- **Boot entry (cold power):** OpenBLT runs when (a) application vector table fails CRC, (b) a latched “enter bootloader” flag is set in backup register (STM32 startup).
- **Boot exit (success):** OpenBLT validates programmed app CRC, clears session latch, jumps to **`0x08006000`**; application runs and **`BOARD_INFO_*`** reflects the new build when telemetry is up.
- **Boot exit (abort):** host aborts the session; OpenBLT stays running; operator may retry **`update`** without power cycle.

### Host `krabby-firmware --stm update`

Operator contract matches **`HOST_INTEGRATION.md`**; wire steps:

1. Resolve S3 build → cache **`~/.cache/krabby-firmware/<branch>/<commit>/leg_app.elf`**.
2. Verify SHA against manifest before any CAN traffic.
3. **Target set:** default **six** provisioned legs; **`--leg <code>`** restricts to one (`FL`, `FR`, `ML`, `MR`, `RL`, `RR`).
4. Per leg (parallel per bus, max three concurrent on `can0`, three on `can1`):
   - Start OpenBLT session → confirm **bootloader** mode.
   - Stream **`.elf`** loadable segments through **XCP** (OpenBLT CAN driver).
   - **Verify** (OpenBLT post-program check).
   - Commit session → confirm **application** mode and expected version/SHA (e.g. **`BOARD_INFO_*`**, CLI).
5. Exit **0** only when every targeted leg reports success; print per-leg progress lines.

**Bootloader recovery:** `krabby-firmware --stm update --bootloader --leg <code>` uses **`leg_bootloader.elf`** and the same session machinery; **one leg at a time** only; operator confirmation required in CLI (manufacturing).

### Power loss mid-update

| State after power returns | Expected behavior |
|---------------------------|-------------------|
| OpenBLT running, app slot incomplete | Node stays in **bootloader**; re-run **same** `krabby-firmware --stm update` (same branch/commit); OpenBLT resumes programming (erase/program from last completed sector per OpenBLT log). |
| OpenBLT running, transfer idle | Same — retry **`update`**. |
| Valid app + incomplete session flag | Application runs; retry **`update`** starts a new session. |
| NV / config page | Unchanged — leg ID and cal remain (`show` / telemetry confirm). |

**Power loss during update:** cut power mid-flash, restore power, run **`update`** again until complete — no manual JTAG.

### OpenBLT configuration (firmware)

- **Transport:** CAN-FD, same bit timing as application (`CAN_PROTOCOL.md`).
- **Connection:** logical **point-to-point** per leg index (host addresses one node even on daisy chain).
- **Clock / MCU:** **STM32G474**; OpenBLT port uses on-chip flash driver; vector table offset for app **`0x06000`**.
- **Watchdog:** held off during programming session; re-enabled in application.

### Regions that must survive application `update`

| Data | Location | Verified after OTA |
|------|----------|---------------------|
| `leg_code` | NV / config page | `krabby-firmware --stm show` |
| Joint `minStop` / `maxStop` | NV / config page | autocal status + telemetry limits |
| OpenBLT bootloader | `0x08000000` | unchanged unless `--bootloader` |

Host **`update`** must not send **`ProvisionSet`**; provisioning is independent (`PROVISIONING.md`).

## Reference

### S3 keys (per branch build)

| File | Upload path pattern |
|------|---------------------|
| `leg_app.elf` | `{branch}/builds/{build_key}/leg_app.elf` |
| `leg_bootloader.elf` | `{branch}/builds/{build_key}/leg_bootloader.elf` |

### Host `image_kind` (CLI)

| Value | ELF used |
|-------|----------|
| 0 | `leg_app.elf` |
| 1 | `leg_bootloader.elf` |

### Error codes (host / OpenBLT session)

| Code | Meaning |
|------|---------|
| 0 | OK |
| 1 | CRC mismatch |
| 2 | Erase failed |
| 3 | Program failed |
| 4 | Verify failed |
| 5 | Wrong leg index / busy |
| 6 | NV protect violation |

### Implementation ownership

| Piece | Location |
|-------|----------|
| OpenBLT + linker memory map | `firmware/stm/` STM32 tree |
| `flash.py` + CLI `update` | `firmware/stm/flash.py`, `cli_stm.py` |
| CI publish `.elf` | `firmware/manifest.py`, `publish-firmware.yml` |

Change this doc when the memory map or OpenBLT transport contract changes.