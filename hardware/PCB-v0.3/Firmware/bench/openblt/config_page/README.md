# One-page persistent configuration reservation

Historical reservation-stage evidence. The current application now implements
SAVE/LOAD and configuration schema 2: see ../../application/README.md and
../../../docs/APPLICATION_PROTOCOL.md. The memory reservation below is unchanged.

Installed and physically tested on the single node-0 Krabby board, 2026-11-11.

| Region | Inclusive addresses | Size |
|---|---|---|
| OpenBLT, hardware write-protected | 0x08000000..0x08002FFF | 12 KiB |
| Application | 0x08003000..0x0801EFFF | 112 KiB |
| Application-managed configuration | 0x0801F000..0x0801FFFF | 4 KiB |

DBANK=0 remains unchanged: the erase unit is 4 KiB. Only one page is reserved,
for approximately 200 bytes of data. No second copy, transactional save guarantee,
or wear-leveling framework is provided. An interrupted application configuration
save can lose the configuration; this risk was explicitly accepted.

At this historical stage the change reserved/protected storage from firmware updates. It did not implement
configuration fields, serialization, save/load APIs, USB commands, or automatic
initialization. Future application code should use a version, length, integrity
check, and defaults when contents are invalid. Follow the existing HAL's flash
programming alignment and erase requirements. Never erase an application or boot page
while saving configuration. Flash operations in single-bank mode can stall execution;
future save code needs its own timing and interruption tests.

## Enforcement

- Root linker: FLASH length 112K; empty CONFIG region 4K. Configuration is not
  included in ELF/SREC/BIN payloads. App/Inc/app_boot.h exports address/size constants.
- Bootloader/flash_layout.c excludes the configuration page from erase/program sectors.
- Bootloader/hooks.c rejects programming into/crossing configuration; read-only XCP
  UPLOAD still permits reading all physical flash, including configuration.
- tools/openblt_canable.py limits applications to 0x0801F000 exclusive; physical
  MCU_FLASH_END remains 0x08020000. S-record and binary images entering configuration
  are rejected before application erasure.
- tools/test_openblt.py uses the reduced application limit, checks boundary rejection,
  and compares the entire configuration page throughout the physical recovery suite.

No IOC changes or CubeMX regeneration are required. Preserve the linker reservation
and app_boot.h constants during future development. The bootloader was reinstalled
using ST-Link with temporary removal and subsequent restoration of pages 0..2 write
protection. No mass erase was used. All 128 KiB were backed up first to
pre_install_flash.bin; pre_install_config.bin preserves the original last-page bytes.

## Evidence

- test.log/results.json: 41 passing physical checks, including full 114,688-byte
  application programming/readback, configuration/boundary erase/write rejection,
  interrupted-transfer/reset recovery, and motor-safe register checks.
- Every configuration-page byte matched its baseline after these tests. That baseline
  is deliberately non-erased (4,080 non-FF bytes left by the earlier maximal-image
  test); it is test data, not a valid application configuration.
- rebuilt_images.log: verification of newly linked demonstration images and final
  Application 2 restoration. Whole-image CAN readback and USB hello identify builds.
- Original 116 KiB evidence stays in the parent directory; old images are archived
  under ../legacy_116k. The root full_application_region.bin now uses 112 KiB.

The reservation survives ordinary CAN application updates. ST-Link mass erase,
restoring the old standalone firmware, or installing the historical 116 KiB
bootloader can destroy it. No physical power-cut test has been performed, and no
configuration-write durability claim is made.

Existing uploader commands and node identifiers are unchanged. Small historical
applications still fit, but future builds must use the current 112 KiB linker limit.
