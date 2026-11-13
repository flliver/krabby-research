# Krabby Uno v0.3 — Design Conversation Decision Record

**Captured:** 2026-11-11. Intended to retain important context when this discussion resumes. This is a summarized decision record, **not a verbatim transcript**. Keep the original conversation in ChatGPT history as well.

## Conversation mission

Define and document firmware **intent and requirements**, then give Codex a trustworthy specification and staged implementation tasks. We are *not* coding the entire firmware in this chat. Engineer supplies product decisions and bench validation; Codex should choose/document implementation details without silently changing agreed behavior. We explicitly allow short ad-hoc tests and do not require formal commits or saved milestones for each experiment.

## What has been physically achieved

- STM32G474RBT3 PCB-v0.3, ST-Link V2 recognized via SWD; processor ID and flash read.
- VS Code STM32Cube/CMake/GCC build, debug and upload working.
- SysTick had been directed to default interrupt handler because CubeMX did not generate `SysTick_Handler()`; CubeMX NVIC options corrected, 1 Hz HAL_Delay LED verified.
- PC8 status LED active HIGH. GPIO/LED helper functions introduced in `App/Inc/board.h`, `App/Src/board.c`; actual local tree is authoritative.
- USB CDC middleware generated, Windows virtual COM port enumerates, bidirectional serial confirmed.
- Physical node-ID switches read successfully over USB. User corrected silkscreen bit order: **NODE_ID0 PC13=4, NODE_ID1 PB4=2, NODE_ID2 PB5=1**, active-low. Earlier assistant-proposed mapping PC13=1/PB5=4 is superseded.
- Board currently tested without motors or other external loads; ~12 V bench supply limited to ~100 mA; driver fuses removed. This describes the current bench, not final use.

## Settled product decisions

1. Six identical **three-motor CAN FD I/O expanders** (18 total motors), physical address from 3 switches, desktop-class master performs all higher-level control and calibration. Node not a gait/position controller.
2. Master normally sends a **complete three-motor signed 8-bit output snapshot** per node at >=50 Hz. CAN payload/ID layout delegated to Codex for documented review.
3. Per-motor **control mode 0**: direct commanded power, no closed-loop control except local protection. Other modes reserved. Mode 0 is default and transparent.
4. Feedback sensor selectable per motor: analog position potentiometer or quadrature encoder; **reporting mode 0** native (filtered ADC), **mode 1** normalized/scaled, 2+ reserved. Sensor acquisition independent of reporting mode.
5. **Current-limit mode 0** uses DAC7574 independently programmed per motor from current-limit configuration; modes 1+ reserved. Current remains queryable in all modes. All relevant ADCs acquire/filter continuously.
6. One RAM dataset with a node page and 3 motor pages; numeric compact access on CAN and descriptions mainly on master. Config can be modified during operation without recompilation. FLASH save/load separate explicit operations. Optional future master-managed full-page loading at boot is a long-term goal; local nonvolatile support comes first.
7. **Flash checksum only verifies saved configuration integrity**: erased/missing/invalid/incompatible flash image => one `NODE_NO_CONFIG` condition. It is *not* mechanical correctness validation and *not* a prerequisite for ARM. A partially configured unit may be provisionally loaded and commissioned via master/USB, then saved.
8. Board-wide ARM permission: every boot DISARMED. ARM clears all three stored commands to zero, maintains safe outputs, and only a **fresh motor frame after ARM** may produce motion. DISARM/timeout zero commands and disarm. A new ARM is required after timeout.
9. Board-wide watchdog: `watchdog_mode` uint8 default 0; `watchdog_timeout` uint8 units 10 ms with zero disabling, maximum 2550 ms; refresh **only on valid addressed motor-command messages**. On timeout apply **per-motor** safe state (default brake; alternative Hi-Z), disarm, clear commands. Specific default nonzero timeout and timer-start semantics TBD.
10. One **node status byte** plus **three motor status bytes** as bitfields of independently observable conditions; not exclusive enums. Candidate flags include ready, armed, no config, watchdog timeout; and per-motor ready/active/travel limit/current limit/driver signal/inhibited. Exact bit assignments provisional.
11. **No node error history**, no broad fault-handling framework. Travel/current limits are often ordinary constraints, not automatically faults. Any actual latched exceptional conditions require concrete design justification later; don't invent a clearing protocol.
12. All sensor and status values queryable on CAN/USB; configurable periodic telemetry in addition. USB service/debug optional in final product. One shared application service backend for both transports.
13. CAN node reboot is **protected two-stage** authorization (`REBOOT_ARM`, then `REBOOT_NOW` with magic key and short window); unrelated motor commands must not accidentally cause reboot.
14. Firmware update over CAN FD uses **Feaser OpenBLT** and established XCP protocol as closely as practicable. Bootloader must be independently recoverable when application image invalid, read physical node IDs, preserve saved config, and keep outputs safe. Decide flash map early; implementation may come later.
15. Optional I2C expansion modules (IMU and status LCD) need future hooks, **not implementations now**. DAC shares board I2C bus.

## Corrected / superseded positions — do not resurrect

- Earlier suggestion: require checksum-verified complete active config before ARM. **Rejected.** Partial provisional RAM configuration is allowed for commissioning; only storage integrity is checked on FLASH save/load.
- Earlier suggestion: after watchdog timeout, next motor command automatically resumes. **Superseded.** Timeout disarms; new ARM clears commands; only a later new motor frame can cause movement.
- Earlier suggestion: watchdog refreshed by any valid addressed CAN message. **Superseded.** Mode 0 refreshes on valid addressed **motor commands** only.
- Earlier proposed fault enums, fault classes and mandatory latched fault manager. **Simplified.** Node/motor byte bitfields for observed conditions and no historical log.
- Earlier proposed node-ID bit packing PC13=1/PB5=4. **Wrong for silkscreen.** Use PC13=4/PB4=2/PB5=1.
- Earlier repeated use of terms 'intelligent motor controller' or node-driven position control. **Wrong product model.** Node is bus I/O expansion with transparent mode-0 motor output.
- Earlier proposal to finalize CAN protocol ourselves before Codex starts. **Rejected.** Codex must **create and document** a reviewed protocol as an explicit deliverable.
- Earlier proposal for separate bespoke firmware-update entry protocol. **Not agreed.** Prefer OpenBLT standard entry/update mechanisms; reboot handshake is a separate application command.

## Open engineering decisions (delegate proposals, review as necessary)

- Actual CAN FD identifier/payload/response protocol, versioning, broadcast semantics and telemetry efficiency.
- Configuration schema field offsets/types/units/defaults, normalized position convention, parameter update lifecycle, versioning.
- Watchdog default timeout and exact start behavior on ARM vs first motor command.
- Position limit direction rules, feedback unavailable/failure behaviors, current-limit conversion and actual detection of current limiting.
- Node/motor status bit assignments and coexistence semantics.
- OpenBLT STM32G474RBT3 128 KB flash partition, linker changes, CAN FD FDCAN2 integration, bootloader activation, recovery test and licensing.
- A clear policy for what hardware conditions are severe enough to inhibit outputs; don't blanket-classify hitting limits as faults.
- Timing/filter methods for ADC and encoder readings, based on measured physical signals.

## Project locations and source documentation

- Parent project: `C:\krabby-research\hardware\PCB-v0.3`
- Firmware: `C:\krabby-research\hardware\PCB-v0.3\Firmware`
- Prior uploaded source snapshots: `krabby_v03_bringup(3).ioc`, `krabby_v03_firmware_pin_definition(1).csv`, `pcb_netlist_extracted(1).csv`, `krabby_v03_firmware_readiness(1).md`, `krabby_v03_bringup(1).txt`, `krabby_v03_ioc_validation(1).md`, `stm32_U6_physical_pins(1).csv`, `krabby_v03_peripheral_assignments_reissued(1).json`.
- Those uploaded snapshots predate some user's successful GUI and local edits. **Always inspect present repository files.** In particular initial IOC snapshot was CubeIDE toolchain and USB CDC not yet generated; the local working project now uses CMake and USB CDC.

## Suggested way to resume

Open this record alongside `Krabby_v03_Firmware_Requirements.md` and `Krabby_v03_Codex_Implementation_Plan.md`. Confirm any new bench findings. Resume by reviewing Codex's first audit/flash-layout proposal or the next acceptance gate; avoid re-litigating decisions recorded as settled unless the engineer requests it.
