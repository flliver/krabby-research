# Flash over CAN (OpenBLT)

## Purpose

In-field programming of leg-board **application** firmware over CAN-FD using **OpenBLT**, with host UX aligned to **`krabby-firmware --stm update`** (`HOST_INTEGRATION.md`).

## Components

- **Bootloader:** **OpenBLT** on **STM32G474**, CAN-FD transport + **XCP** programming protocol
- **Artifacts:** **`leg_app.elf`**, **`leg_bootloader.elf`** on S3 (`krabby-firmware-public`)
- **Host:** `firmware/stm/flash.py`, invoked from CLI **`update`**
- **Preserved NV:** provisioning + joint cal (`LEG_BOARD_CONTROL.md`, `PROVISIONING.md`)
- **CAN IDs:** flash session + XCP tunnel in **`0x0C000300`–`0x0C0003FF`** (`CAN_PROTOCOL.md` pointer)

## Critical points

- **Scope:** normal **`update`** flashes **application** only. **`leg_bootloader.elf`** is for factory/recovery (separate CLI flag **`--bootloader`**, manufacturing use); not run on every fleet OTA.
- **Dual artifact CI:** each release branch build publishes **`leg_app.elf`**, **`leg_bootloader.elf`**, **`leg_app_sha256`**, **`leg_boot_elf_sha256`** via `firmware/manifest.py` and `publish-firmware.yml`.
- **Memory model (normative layout — linker scripts in `firmware/stm/` must match):**

| Region | Flash start | Size | Erased on app OTA |
|--------|-------------|------|-------------------|
| Bootloader (OpenBLT) | `0x08000000` | **24 KiB** | **Never** by `update` |
| Application image | `0x08006000` | **100 KiB** | Yes (full app slot) |
| Config / NV mirror | `0x0801F800` | **2 KiB** | **Never** by `update` |

Application link base **`0x08006000`**. Config page holds a copy of leg/cal NV (`LEG_BOARD_CONTROL.md`); MCU also mirrors hot fields in RAM/EEPROM emulation — both must survive app programming. Bootloader update reprograms **`0x08000000`–`0x08005FFF`** only via **`--bootloader`**.
- **Boot entry (application running):**
  1. Host sends **`FlashSessionOpen`** (below) to **one** leg index on its bus.
  2. Application finishes current CAN RX, de-energizes bridges (safe state), acknowledges, resets into OpenBLT.
  3. Within **200 ms**, node emits **`NodeStatus`** with **`mode = bootloader`**.
- **Boot entry (cold power):** OpenBLT starts when (a) application vector table fails CRC, (b) **`FlashSessionOpen`** latched flag set in backup register (STM32 startup).
- **Boot exit (success):** OpenBLT validates programmed app CRC, clears session latch, jumps to **`0x08006000`**, application emits **`NodeStatus`** **`mode = application`** with new identity string.
- **Boot exit (abort):** host **`FlashSessionClose`** with **abort** code leaves OpenBLT running; host may retry **`update`** without power cycle.

### Host `krabby-firmware --stm update`

Operator contract matches **`HOST_INTEGRATION.md`**; wire steps:

1. Resolve S3 build → cache **`~/.cache/krabby-firmware/<branch>/<commit>/leg_app.elf`**.
2. Verify SHA against manifest before any CAN traffic.
3. **Target set:** default **six** provisioned legs; **`--leg <code>`** restricts to one (`FL`, `FR`, `ML`, `MR`, `RL`, `RR`).
4. Per leg (parallel per bus, max three concurrent on `can0`, three on `can1`):
   - **`FlashSessionOpen`** → wait **`NodeStatus` bootloader**.
   - Stream **`.elf`** loadable segments through **XCP** via **`FlashXcpRequest` / `FlashXcpResponse`** (OpenBLT CAN driver).
   - **Verify** (OpenBLT post-program check).
   - **`FlashSessionClose`** **commit** → wait **`NodeStatus` application** with expected version/SHA prefix.
5. Exit **0** only when every targeted leg reports success; print per-leg progress lines.

**Bootloader recovery:** `krabby-firmware --stm update --bootloader --leg <code>` uses **`leg_bootloader.elf`** and the same session machinery; **one leg at a time** only; operator confirmation required in CLI (manufacturing).

### Power loss mid-update

| State after power returns | Expected behavior |
|---------------------------|-------------------|
| OpenBLT running, app slot incomplete | Node stays in **`bootloader`**; **`NodeStatus`** shows **`mode = bootloader`**. Re-run **same** `krabby-firmware --stm update` (same branch/commit); OpenBLT resumes programming (erase/program from last completed sector per OpenBLT log). |
| OpenBLT running, transfer idle | Same — retry **`update`**. |
| Valid app + incomplete session flag | Application runs; retry **`update`** performs normal **`FlashSessionOpen`** again. |
| NV / config page | Unchanged — leg ID and cal remain (`show` / telemetry confirm). |

**Power loss during update:** cut power mid-flash, restore power, run **`update`** again until complete — no manual JTAG.

### CAN flash message IDs (normative)

Extended IDs; **leg index** (0–5) in byte 0 of payload for node-specific frames. Implementers mirror in **`krabby_leg.dbc`** (`CAN_PROTOCOL.md` normative IDs).

| Message | CAN ID | Dir | Role |
|---------|--------|-----|------|
| `FlashSessionOpen` | `0x0C0007F1` | host → leg | Enter OpenBLT; payload: `leg_index`, `session_nonce[4]`, `image_kind` (0=app, 1=bootloader) |
| `FlashSessionClose` | `0x0C0007F2` | host → leg | `commit` (0) or `abort` (1) |
| `FlashXcpRequest` | `0x0C000300 + leg_index` | host → leg | OpenBLT XCP request bytes (≤62 B FD data) |
| `FlashXcpResponse` | `0x0C000380 + leg_index` | leg → host | OpenBLT XCP response bytes |
| `FlashProgress` | `0x0C0003C0 + leg_index` | leg → host | Optional percent / error code during programming |

**Range discipline:** only **`0x0C000300`–`0x0C0003FF`** for flash/XCP. Do not overlap **`LegCommand` / `LegTelemetry` / `NodeStatus`** bands.

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

### `image_kind` (`FlashSessionOpen`)

| Value | ELF used |
|-------|----------|
| 0 | `leg_app.elf` |
| 1 | `leg_bootloader.elf` |

### Error codes (`FlashProgress` / failed session)

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
| CAN IDs in DBC | `firmware/stm/dbc/krabby_leg.dbc` |
| `flash.py` + CLI `update` | `firmware/stm/flash.py`, `cli_stm.py` |
| CI publish `.elf` | `firmware/manifest.py`, `publish-firmware.yml` |

Change this doc only when the memory map or CAN flash IDs change in the build.
