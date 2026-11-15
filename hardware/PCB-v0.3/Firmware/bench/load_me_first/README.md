# LoadMeFirst ? initial USB provisioning package

Built 2026-11-12 for STM32G474RBT3 / Krabby v0.3. This is a separate build, not the motor application.

## Files

- LoadMeFirst.hex: combined addressed Intel HEX for initial USB DFU programming. Unchanged OpenBLT at 0x08000000 (12,000 bytes), padding through 0x08002FFF, indicator application at 0x08003000 (35,972 bytes). Combined span 48,260 bytes. Application OpenBLT signature is already installed, so DFU programming does not require CAN finalization.
- LoadMeFirst_app.srec: application only, for the existing CAN updater on an already provisioned board. It cannot install the bootloader on a blank board.
- LoadMeFirst_app.elf/.map: symbols and memory map.
- manifest.json: addresses, sizes and SHA-256 checksums.
- verification.txt: software packaging checks; no physical programming or LED acceptance in this build task.

No configuration-page bytes or option bytes are included. No self-installing bootloader copy operation: the combined HEX installs bootloader and application directly in their final regions.

## Behavior

One processor-controlled LED is on at a time, advancing every 100 ms in this order:

1. M1 left (PC2)
2. M1 right (PC3)
3. M2 left (PA3)
4. M2 right (PA4)
5. M3 left (PB6)
6. M3 right (PD2)
7. Status (PC8)
8. CAN Activity (PC9)

The sequence repeats every approximately 800 ms after OpenBLT's two-second startup window. Three red driver-fault indicators are not GPIO-controlled and are excluded.

Motor sleeps PB10/PB14/PB15 stay LOW; TIM1 MOE remains off. No motor control, ADC acquisition, DAC programming, or configuration SAVE/LOAD service. The normal motor-service source is excluded from this target. No application configuration is read or changed. USB hello identifies LoadMeFirst; status, node uid and node reboot work. Manual led on/off commands are rejected to preserve the chase. Existing CAN diagnostic bridge is retained.

Read-only CAN VERSION discovery responds with protocol 1, schema 0, capabilities 32 (UID support), deliberately preventing motor GUIs from treating this as a motor node. CAN application ARM/MOTOR/configuration operations return unsupported. Existing XCP CONNECT reset re-entry allows the CAN updater to replace this application, without USB being connected. Firmware CAN timing and switch addressing are unchanged (1M nominal / 5M data BRS).

## Initial USB installation on a new, unprotected board

1. Keep motor power disconnected; connect logic power and USB. ST's factory ROM can drive PB14 HIGH during interface probing, independently of our firmware.
2. Hold BOOT, press/release RESET, release BOOT. Select USB DFU in STM32CubeProgrammer.
3. Read option bytes. This OpenBLT build requires DBANK=0 (single-bank, 4 KiB erase pages); set it if needed, leaving unrelated options unchanged. Option reload may disconnect DFU: re-enter BOOT/reset and reconnect.
4. Program LoadMeFirst.hex with verification. Use its embedded addresses; do not load the application-only file at 0x08000000. No whole-chip erase is required for a genuinely blank device.
5. Reset with BOOT released. Observe the two-second bootloader window followed by the eight-light chase.
6. Probe the selected switch ID over CAN, then update to the normal motor application using openblt_canable.py.
7. Once installation and CAN replacement are verified, separately apply bootloader write protection if desired: WRP1A_STRT=0, WRP1A_END=2. Do not change RDP or BOOT_LOCK as part of this procedure.

Our current development board already has bootloader write protection. Do not attempt combined-image programming over it or disable protection just to test the lights. For it, reset out of factory DFU and use the application-only CAN route below. Configuration is retained.

From Firmware/tools:

```powershell
python .\openblt_canable.py ..\bench\load_me_first\LoadMeFirst_app.srec --node 7
python .\probe_nodes.py --port COM7
# Return to the tested motor application:
python .\openblt_canable.py ..\bench\motor_pages\application.srec --node 7
```

## Rebuild

With CMake/Ninja/ARM GCC on PATH, from Firmware root:

```powershell
cmake -S . -B build/LoadMeFirst -G Ninja "-DCMAKE_TOOLCHAIN_FILE=cmake/gcc-arm-none-eabi.cmake" -DCMAKE_BUILD_TYPE=MinSizeRel -DLOAD_ME_FIRST=ON
cmake --build build/LoadMeFirst
cmake --build build/Bootloader
python tools/package_load_me_first.py
```

Packaging consumes addressed application S-records. The build now also generates compact raw binaries by converting those S-records rather than the ELF directly, avoiding padding from an empty loadable TLS section toward RAM addresses. No linker or firmware memory-layout change was needed.

Normal builds default LOAD_ME_FIRST=OFF. Root main.c and USB edits stay inside CubeMX user sections; .ioc configuration unchanged and no regeneration needed. Normal application also rebuilt successfully at 85,512 bytes.

## Acceptance status

Build passed without warnings. Packaging verified independently: Intel HEX checksums/address bounds, unchanged bootloader byte equality, application S-record equivalence, valid OpenBLT vector checksum and exclusion of persistent configuration. Existing board factory USB DFU identification/flash reads were demonstrated in the preceding task. This combined image has NOT been programmed via DFU, the LED chase has NOT been observed, and CAN replacement from this particular indicator build remains to be physically verified.
