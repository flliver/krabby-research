# Krabby Uno v0.3 STM32 firmware

STM32G474RBT3 (128 KiB flash), CubeMX + GCC/CMake/Ninja, FDCAN2 PB12 RX / PB13 TX, USB CDC. This directory is a firmware subproject of the surrounding repository; do not stage unrelated parent-repository changes when committing it.

## Current state

- Physically tested CAN FD at 1 Mbps nominal / 5 Mbps data with BRS on one board.
- OpenBLT at 0x08000000..0x08002FFF; application at 0x08003000..0x0801EFFF (112 KiB); persistent configuration at 0x0801F000..0x0801FFFF (4 KiB).
- Current motor application schema 4: node page 48 bytes, motor pages 56 bytes, per-motor output reversal and native/normalized travel limits. MCU UID command supported.
- Latest physically tested motor image: bench/uid/application.srec. Earlier bench images are historical versions, not interchangeable current releases.
- LoadMeFirst USB provisioning package: bench/load_me_first/LoadMeFirst.hex. This combines OpenBLT with a motor-disabled eight-LED chase. Built/format-checked; complete blank-chip USB provisioning and CAN replacement from this indicator image still await physical acceptance.
- One-node bench verification does not establish six-node operation, physical power-cut recovery, or sustained-load/thermal qualification. CAN updates are not authenticated.

## Build from source

Requirements: CMake >=3.22, Ninja, ARM GNU toolchain with arm-none-eabi-gcc and arm-none-eabi-objcopy on PATH. Clean builds during the audit used ST GNU 14.3.1+st.2, CMake 4.4.0 and Ninja 1.13.2. No existing build cache is required.

From this directory:

```powershell
cmake --preset Debug
cmake --build --preset Debug
cmake --preset LoadMeFirst
cmake --build --preset LoadMeFirst
cmake -S Bootloader -B build/Bootloader -G Ninja "-DCMAKE_TOOLCHAIN_FILE=../cmake/gcc-arm-none-eabi.cmake" -DCMAKE_BUILD_TYPE=Release
cmake --build build/Bootloader
python tools/package_load_me_first.py
```

Build output lives in ignored build/. Application .bin files are generated via addressed S-records to avoid erroneous flash-to-RAM padding. No linker relocation or configuration reservation is changed by packaging.

The .ioc is the peripheral source of truth. Application additions remain in user sections/separate App sources. Run bench/application/regenerate.script from this directory if using CubeMX's script interface. Recheck the relocated linker/vector signature after regeneration; do not assume regeneration preserves custom linker/startup files automatically.

## Python tools

Use Python 3.10+ (64-bit on Windows), with tkinter available. On Windows tkinter is normally included; Linux distributions may package it separately. No dependencies are installed automatically.

```powershell
python -m pip install -r tools/requirements.txt
python tools/probe_nodes.py --port COM7
python tools/krabby.py --port COM7 --node 1 uid
python tools/krabby_position_gui.py --port COM7 --node 1
# Read-only diagnostic viewer:
python tools/krabby_debug_gui.py --port COM7 --node 1
```

Switch-selected IDs 0..7 are currently accepted; assign unique 1..6 for the intended six boards. Switches are read at startup. Use the ID reported by discovery, not an old bench example. Release COM7/COM3 from other terminals before connecting. CANable commands configure normaldotcom/canable2 for the tested 1M/5M timing.

Configuration edits belong on Configuration > Motor pages. Write staged page applies RAM settings; SAVE persists. Position tab configuration boxes are read-only. Position control is a PC proportional loop, not a firmware servo or velocity controller.

See tools/CONTROL_GUI.md, tools/POSITION_GUI.md and docs/APPLICATION_PROTOCOL.md.

### CAN firmware uploader dependency

The current uploader backend is Windows-only and additionally requires the official 64-bit LibOpenBLT shared library. The local DLL/BootCommander executables are deliberately ignored by Git; a fresh clone does not include them. Diagnostic/control tools do not require the DLL.

Obtain the complete upstream OpenBLT distribution/source (the target source revision is recorded in vendor/openblt/UPSTREAM.txt). Copy the trusted 64-bit Host/libopenblt.dll to vendor/openblt/Host/libopenblt.dll, or build it from the complete upstream host sources. The selected Host/Source files in this repository are inspection references, not a complete DLL build tree. BootCommander.exe is not used by our Python uploader.

Upstream build/runtime instructions: https://www.feaser.com/openblt/doku.php?id=manual:libopenblt . Match library and Python architecture. Linux needs a loader/backend change before this updater can use libopenblt.so; native upstream Linux support does not mean this wrapper has been validated there.

```powershell
python tools/openblt_canable.py bench/uid/application.srec --port COM7 --node 1
```

This changes the selected board's application. For a blank chip, follow bench/load_me_first/README.md instead; that image installs bootloader plus indicator through factory USB DFU. Motor power must be off during ROM DFU because ROM initializes pins used for motor-driver sleep.

## Offline validation (no hardware access)

```powershell
python tools/check_artifacts.py
python tools/krabby_debug_gui.py --smoke-test
python tools/test_control_gui.py
python tools/test_jog_gui.py
python tools/test_position_gui.py
python -m compileall -q tools
```

GUI tests require a graphical session/display; they use synthetic devices. They do not prove motor hardware operation. Offline runs print results without rewriting bench evidence. Only explicit hardware runs append their records under bench/application.

Physical suites are opt-in. test_application.py, test_application_stress.py and test_openblt.py require --hardware and assume the documented node-0 COM7/COM3 fixture; they overwrite configuration and/or firmware and are not generic non-destructive tests. test_travel_limits.py and test_position_motion.py require --hardware PORT and explicitly move motors. Do not include hardware suites in unattended CI.

For ST-Link bench tools, add STM32_Programmer_CLI to PATH or set STM32_PROGRAMMER_CLI to its executable. Windows Cube tool-bundle lookup under the current user's home is a fallback. No fixed developer username is required.

## Repository contents

Commit source, .ioc, linker/startup files, build configuration, documentation, schemas and intentional bench evidence. Build/cache/editor files are ignored. bench/ contains historical images, logs, configuration backups and deliberately corrupt flash fixtures: preserve them as evidence, never bulk-flash that directory. There are no firmware files above GitHub's 100 MB file limit after binary cleanup.

No commits/staging are performed by the audit. Before staging, inspect git status from the repository root and restrict the selection to this firmware subproject. See docs/PROJECT_AUDIT.md for findings and limitations, and THIRD_PARTY.md for license/provenance boundaries.
