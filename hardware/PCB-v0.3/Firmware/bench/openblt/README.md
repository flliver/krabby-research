# Krabby v0.3 OpenBLT implementation and physical tests

The integrated node application is now documented in [application acceptance](../application/README.md).
Bootloader/protection/layout below remain current; earlier application-only logs
are historical evidence, not the current application feature set.

2026-11-11. One connected board, node 0. ST-Link V2 serial
`38FF74063446393036440857`; CANable2 COM7 firmware
`16e7497-dirty github.com/normaldotcom/canable2.git`; application USB COM3.

## Installed state

- Feaser OpenBLT source revision `8fc277c74298a2fedeb99e85f906d846bf0018e3`
  (core identifies as 1.22.0), selected upstream sources in `vendor/openblt`.
- Bootloader actual flash footprint **12,000 bytes**, 12 KiB reserved. Linker enforces
  the limit. RAM allocation including heap/stack: 3,960 bytes.
- Application **112 KiB**, `0x08003000` through `0x0801EFFF`.
- Persistent configuration: one **4 KiB page**, `0x0801F000` through `0x0801FFFF`.
  Reserved for approximately 200 bytes of application-managed data; OpenBLT cannot
  erase/program it. See [configuration-page notes](config_page/README.md).
- Application vector signature: `0x080031D8`. Linker asserts the reserved word's
  symbol/address, so regenerated startup code cannot silently remove it.
- Current application: integrated node services plus diagnostic bridge, 84,280 flash
  bytes, 250 ms status LED toggle. Current known-good image: ../application/known_good.srec.
  Root app_125ms/app_250ms and led_apps images are historical diagnostic applications.
- DBANK was changed from 1 to **0** for contiguous 128 KiB flash; erase pages are 4 KiB.
- Bootloader hardware write protection: **WRP1A_STRT=0, WRP1A_END=2**.
  RDP remains Level 0; no other option-byte setting was intentionally changed.
- Motor sleeps PB10/PB14/PB15 remain LOW, TIM1 MOE remains off. The bootloader uses
  the upstream polling TIM1 timebase with PWM outputs disabled.
- Default reset entry window is two seconds. An invalid application remains in
  OpenBLT. No automatic application rollback, authentication, or watchdog was added.

CAN remains the physically proven 80 MHz PCLK1 configuration: FDCAN2 PB12/PB13 AF9,
nominal 1 Mbps `(5,13,2,1)`, data 5 Mbps `(1,13,2,1)` in HAL prescaler/seg1/seg2/SJW
units, both sample points 87.5%, TDC enabled offset 13/filter 0. Register values:
`NBTP=00040C01 DBTP=00800C10 TDCR=00000D00`. No CANable firmware changes were made.

## Code organization and upstream changes

`Bootloader/` contains a separate CMake target, board startup, configuration,
application-only flash layout, and input-validation hooks. `Bootloader/can.c` is the
upstream G4 driver with these local changes: HAL clock-query API compatible with our
HAL release, explicit verified timing instead of upstream timing search results,
and transmit pause disabled. Protocol, erase/program management, deferred vector
block finalization, and application handoff remain OpenBLT's implementation.

The root application linker is relocated, startup reserves the signature, `main.c`
sets VTOR before HAL initialization, and status reports blink interval/node/VTOR.
The existing CAN receiver recognizes a two-byte XCP CONNECT on its node's boot
identifier and resets, using the upstream demo's application re-entry behavior.
It does not initialize a second competing CAN receiver.

Switch numbering is unchanged: PC13 -> bit 2, PB4 -> bit 1, PB5 -> bit 0, active LOW.
Bootloader samples it at boot. Host->target ID is `0x667 + node`; response is
`0x7E1 + node`, standard identifiers, node range 0..7. Only actual switch setting
0 was tested; a real wrong-node request was ignored. Simultaneous multi-node behavior
and other physical switch settings have not been tested.

## Host interface

`tools/openblt_canable.py` uses the official upstream 64-bit `libopenblt.dll` through
Python ctypes. It retains the upstream XCP update engine. A local Python relay maps
its existing XCP-over-TCP packet format to CANable2's serial FD+BRS format; the
relay binds only to loopback on a dynamically assigned port. It reads TCP stream
lengths using the upstream loader's known command shapes, supports the required
programming subset, and rejects unsupported opcodes rather than guessing framing.
The frontend validates S-record checksums, vectors, addresses, and the complete
112 KiB application range before erasing; data entering the configuration page is rejected. It finalizes only complete transfers, then reads
the entire image back over CAN and compares every byte, accounting for the signature
OpenBLT generates. Only after successful readback does it request application start.

No SDK, custom Windows driver, adapter firmware change, or native host build is needed.
Requires existing 64-bit Windows Python and `pyserial`. The downloaded DLL is x64;
it reports LibOpenBLT version 1.06.00. Upstream host/target version numbers differ.

Close COM3 terminals and release any existing COM7 PowerShell SerialPort handle first.
From the Firmware directory:

```powershell
# Update from the running application, verify over CAN, then start it.
python tools/openblt_canable.py build/Debug/krabby_v03_bringup.srec --node 0

# Alternate known application; --stay retains OpenBLT after verification.
python tools/openblt_canable.py bench/openblt/app_125ms.srec --node 0 --stay

# Destructive physical suite: replaces application, resets target, preserves config,
# restores 250 ms app; logs/results belong in bench/openblt/config_page.
python tools/test_openblt.py --hardware

# CAN bridge regression after the application has started.
python tools/test_can_bridge.py --nominal 1000000 --data 5000000 --check-status --repeat 10
```

Programmer write/erase protection is enforced independently of the Python frontend:
OpenBLT's flash layout contains only application pages; hooks reject unsupported or
truncated commands, invalid MTAs, reads beyond flash, and writes outside application
flash. This is prototype robustness, not an authenticated security boundary.

## Physical evidence

Current configuration-page validation: **41 physical checks passed**, archived in
`config_page/test.log` and `config_page/results.json`; all 4 KiB survived full-region
updates and reset/interruption recovery unchanged. The following root-level logs describe the historical
116 KiB layout before the configuration page was reserved; their original images
and bootloader are retained under `legacy_116k/`. Do not install that bootloader
when configuration preservation is required.

`test.log` and `results.json`: **35 passing checks on the final protected bootloader**:

- A/B/A updates (250/125/250 ms), full-image CAN readback each time, USB application
  identity and relocated VTOR, re-entry from each running version.
- Bootloader erase/write rejection, invalid address/end-of-flash operations,
  malformed/unsupported commands, empty CAN frame survival, wrong-node filtering.
- Host abandonment after erase, early transfer, mid transfer, and immediately before
  finalization. Each was followed by an ST-Link reset, waiting beyond the entry window,
  confirmation of erased initial application vectors, and a complete CAN-only recovery.
- ST-Link reset during an active transfer caused an expected host programming failure;
  subsequent complete recovery and readback used CAN only, without reflashing OpenBLT.
- All **118,784 application bytes** programmed and read back, including the final flash
  page, then the application booted. The synthetic full-region test image is archived.
- Further alternating updates, unchanged bootloader hash, and final 250 ms app restored.
- Live GPIOB ODR and TIM1 BDTR read through ST-Link: all motor sleeps LOW, MOE disabled.

`can_regression.log`: **1,400 frame deliveries passed** on the relocated bridge at
1 Mbps / 5 Mbps, all checked application error counters/loss counters zero.

`boot_can_health.txt`: after another complete CAN upload and full-image readback,
bootloader ECR=0 and PEA/PED protocol-error flags clear. The debug block read also
included reserved FDCAN address 0x4C, which can set ARA; that is not a CAN protocol error.

These are actual serial/CAN/ST-Link bench operations, not simulated tests. USB status
confirms the compiled blink settings; optical LED frequency was not measured.
**Actual power loss, brownout/ECC corner cases during a flash pulse, six-node operation,
long cables, security qualification, and sustained bus/queue saturation are not proven.**

OpenBLT's default validity signature covers initial vectors, not a whole-image CRC or
cryptographic signature. Host full-image readback detects transfer/storage mismatches
at update time. A future on-device whole-image integrity requirement needs its own
checksum-hook design and tests. The single-image layout provides recovery, not rollback.

## Builds

Use the existing STM32 GCC/Ninja/CMake bundle paths. In this environment:

```powershell
$env:PATH = 'C:/Users/Me/AppData/Local/stm32cube/bundles/gnu-tools-for-stm32/14.3.1+st.2/bin;C:/Users/Me/AppData/Local/stm32cube/bundles/ninja/1.13.2+st.1/bin;' + $env:PATH
$cmake = 'C:/Users/Me/AppData/Local/stm32cube/bundles/cmake/4.4.0+st.1/bin/cmake.exe'
& $cmake -S Bootloader -B build/Bootloader -G Ninja '-DCMAKE_TOOLCHAIN_FILE=C:/krabby-research/hardware/PCB-v0.3/Firmware/cmake/gcc-arm-none-eabi.cmake' '-DCMAKE_BUILD_TYPE=Release'
& $cmake --build build/Bootloader
& $cmake -S . -B build/Debug '-DAPP_BLINK_MS=250'
& $cmake --build --preset Debug
```

`APP_BLINK_MS=125` builds the alternate application. Both targets produce ELF, binary,
and S-record files. The known linker emits unused newlib syscall-stub warnings in the
bootloader build; no bootloader file I/O is used. No configuration payload is linked into firmware: the linker reserves an empty
`CONFIG` region, with application address/size constants in `App/Inc/app_boot.h`.
No configuration serialization or save/load commands have been added.

Do not enlarge the bootloader region
without also relocating/rebuilding every application and updating host bounds.

No `.ioc` settings changed for this work. No peripheral regeneration is needed.
After any future CubeMX regeneration, preserve the relocated linker and signature
entry; the linker assertion catches a missing/moved checksum symbol.

## ST-Link recovery and original bridge restoration

Prefer CAN recovery first. Reset/power cycle and begin connecting within two seconds
if a valid but malfunctioning application is present. Invalid signatures keep OpenBLT
running indefinitely. A missing 5 V CAN transceiver rail still prevents CAN recovery.

To replace OpenBLT using ST-Link, its hardware write protection must first be disabled.
From the Firmware directory (only when deliberately recovering/reinstalling):

```powershell
$programmer = 'C:/Users/Me/AppData/Local/stm32cube/bundles/programmer/2.23.0/bin/STM32_Programmer_CLI.exe'
& $programmer -c port=SWD mode=UR -ob WRP1A_STRT=127 WRP1A_END=0
& $programmer -c port=SWD mode=UR -w bench/openblt/krabby_boot.elf -v -rst
# Reapply protection after verifying the installation.
& $programmer -c port=SWD mode=UR -ob WRP1A_STRT=0 WRP1A_END=2
```

`pre_bank1.bin` and `pre_bank2.bin` contain the original 64 KiB physical banks;
`options_before.txt` records original options; `pre_source.zip` records changed original
project files. Current options are in `options_installed.txt`.

To deliberately return to the old standalone bridge, disable protection first,
restore DBANK=1, then erase/program the pre-OpenBLT standalone ELF:

```powershell
& $programmer -c port=SWD mode=UR -ob WRP1A_STRT=127 WRP1A_END=0
& $programmer -c port=SWD mode=UR -ob DBANK=1
& $programmer -c port=SWD mode=UR -e all -w bench/can_1m_5m/krabby_v03_bringup.elf -v -rst
```

That deliberately removes OpenBLT, the current application, and persistent configuration. It does not restore
source files automatically. Source restoration requires the matching archived files;
do not build the old base-zero application while expecting OpenBLT to remain installed.

## License

OpenBLT and derivative integration files are GPL-3.0-or-later; see
`vendor/openblt/Doc/license.html` and upstream notices. ST-generated/HAL files retain
their original licenses. Commercial/proprietary distribution licensing remains a
separate unresolved release decision; this prototype is not a licensing clearance.
