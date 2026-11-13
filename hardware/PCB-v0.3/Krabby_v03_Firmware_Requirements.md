# Krabby Uno v0.3 — Firmware Intent and Requirements

**Status:** Requirements baseline for engineering implementation; 2026-11-11.  
**Authority:** Decisions reached in the design conversation. **Not** an assertion that all software exists.  
**Companion files:** `Krabby_v03_Codex_Implementation_Plan.md`, `Krabby_v03_Codex_Starter_Prompt.md`, `Krabby_v03_Decision_Record.md`.

## 1. Product role and boundaries

A Krabby Uno v0.3 node is a **three-channel brushed-DC motor I/O expander** on a CAN FD network. Six identical nodes are intended to control 18 motors. Three physical active-low switches set each node's address. The master is desktop-PC-class hardware that makes all movement and calibration decisions.

The node **does** execute requested motor outputs, acquire and report sensors, apply configured output restrictions and minimal local hardware protection, expose configuration/status, and perform maintenance operations. The node **does not** independently plan motions, close position loops in default mode, infer robot joints/gait, or decide calibration suitability. USB CDC is a functional development/service port; deployed nodes normally operate over CAN FD without USB attached.

Hardware mapping, electronic details, and pin-safe policy live in existing PCB documentation and CubeMX `.ioc`. Do not rederive or copy all hardware details into the application specification. Read and reconcile the **current** workspace files before editing hardware mappings. The uploaded historical source set is listed in the decision record; it may not match subsequent local revisions.

## 2. Control and sensing, per motor

Each channel independently supports configurable modes, with mode 0 as the default and mode 1+ reserved except where specifically defined.

| Function | Mode 0 (default) | Mode 1 | Future |
|---|---|---|---|
| Motor control | Direct commanded signed output; no closed-loop motion control, except local protection | Not defined | Reserved |
| Feedback reporting | Native sensor values (ADC measurements filtered as configured; quadrature encoder counts) without position normalization | Apply locally configured calibration/scaling to a normalized position | 2+ reserved |
| Current limiting | Hardware DRV8262 limit from independently programmed DAC threshold | Not defined | 1+ reserved |

**Motor commands:** one signed 8-bit command per motor (working range and zero semantics must be precisely documented by protocol designer), normally one CAN FD frame containing commands for all three motors. Negative/positive denote opposite directions; magnitude denotes requested PWM. The upstream master ordinarily updates at **at least 50 Hz**. PWM carrier frequency is separate from the CAN update rate. Direction sign vs CW/CCW is board/application convention, not an inference about physical joint motion.

**Sensors:** all available sensors must be queryable; ordinary telemetry may be configured to publish without an explicit request. ADC acquisition should run continually and apply configurable filtering, regardless of feedback reporting mode. Retain access to native/unfiltered diagnostic measurements where practical. Quadrature and potentiometer position feedback must be independently selectable **without firmware recompilation**. The node does not independently target a position.

**Motor page** contains, at minimum: control mode, feedback source, feedback reporting mode, current limit mode/threshold, position/travel limits, scaling/calibration, ADC filter coefficients, and watchdog safe-output choice. It must be extensible without assuming all motor types have identical calibration data. Exact data layout, units, ranges and unsupported-mode behavior are engineering tasks that must be documented.

**Current:** all three measured motor currents remain available for interrogation and periodic telemetry in **every** current-limit mode. Hardware limiting is applied via DAC7574 outputs A/B/C (one per motor) and DRV8262 drivers. DAC setup is required before any enable that relies on the hardware current limit. A hardware current limit being reached is not automatically classified as an error.

**Travel limits:** configurable and locally restrictive when applicable, but reaching an end of travel is not intrinsically a fault. Exact enforcement semantics, direction convention, unavailable feedback behavior and measurements are unresolved engineering questions; do not invent dangerous behavior.

## 3. Board-level control and arming

**ARM/DISARM is board-wide** for all three motor outputs. It is independent of config-save checksums.

- Boot always begins **DISARMED**, with motor outputs safe.
- `ARM` grants permission for **future** commands; it is **not** a movement command.
- As part of ARM, clear every stored motor command to zero and preserve safe outputs. **Only a newly received motor-command frame after ARM** can request nonzero output.
- `DISARM` immediately applies each motor's configured safe-output policy and clears stored commands.
- A watchdog timeout also disarms, applies safe states, and clears commands; after communication resumes a new explicit `ARM`, then a new motor frame, are required. This **supersedes earlier automatic rearming** proposals.
- A partially populated provisional RAM configuration supplied by a master during controlled commissioning can be used for arming. No configuration completeness, mechanical calibration correctness, or flash checksum check is an ARM prerequisite. The firmware still must ensure needed low-level driver hardware setup (e.g. safe DAC programming) before outputs are energized.

**Safe output:** configured independently for each motor, default **brake**, alternative **Hi-Z / driver sleep**. In the present slow non-backdrivable actuator application these are functionally comparable mechanically, but differ electrically. In DRV8262 PH/EN mode PWM=0 with the driver awake is braking, **not Hi-Z**. Ensure driver power, fault behavior and electrical signals are correctly handled.

## 4. Command watchdog

Node-level configuration contains:

- `watchdog_mode`: unsigned 8-bit; **mode 0 default**; modes 1+ reserved.
- `watchdog_timeout`: unsigned 8-bit multiples of **10 ms**. `0` disables watchdog; values `1..255` represent 10..2550 ms. **Default nonzero timeout TBD.**

**Mode 0:** a valid properly addressed **motor command frame** (not an arbitrary query or configuration message) refreshes the single board-wide watchdog. On expiry, all three motors are disarmed and placed in their individually configured safe-output states; stored motor commands are cleared. A fresh ARM and subsequent fresh motor command are required to resume output. Exact timer-start behavior at ARM vs first command remains to be confirmed; design a conservative bounded implementation and flag the choice for review.

The master normally sends all three commands together in one frame; treat this as the expected protocol, so partial motor-command persistence is not a design objective.

## 5. Configuration and persistence

There is **one logical dataset with independently addressable node and three motor pages**. Optional expansion-device configuration must be possible later. Configuration should be accessible by numeric IDs/offsets over compact CAN payloads, with named metadata primarily maintained by master/service tools. Never expose unchecked arbitrary memory writes.

- **Active RAM configuration** drives behavior; individual fields/pages can be read and edited without recompilation or flash writes.
- **Factory defaults** provide defined RAM values but **do not assert that a working motor configuration has been calibrated**.
- **Saved flash image** is an optional persistent copy. Boot normally attempts to load it into active RAM.
- `SAVE` writes active configuration to flash with an integrity checksum/CRC (and sufficient identifying metadata). `LOAD` verifies stored-image integrity before loading RAM; explicit load/save/default operations should exist.
- Missing, erased, incompatible or checksum-failed stored images are **the same `NODE_NO_CONFIG` status path**. No elaborate distinction is required.
- `NODE_NO_CONFIG` refers only to **absence of a verified saved image**, not a prohibition on loading provisional RAM settings, ARM, or commissioning. Editing RAM does not imply saved configuration exists. Define precisely when the bit clears on successful save/verification or load.
- Checksum is **only** for read/write integrity of the internal flash image; **it does not certify active RAM configuration suitability** and is **not** an arming gate.
- The master may preload partial provisional settings, conduct controlled empirical calibration (possibly human-in-the-loop), then send final pages and issue SAVE. The master will eventually own the canonical machine-specific configuration and reload nodes at boot; the standalone RAM/flash workflow must work **first**, without a working master.
- Local validation is limited to data format and hardware-operable parameter bounds. The master owns suitability of current limits, calibration, range of travel, etc. A configuration update requiring hardware action must report whether that action succeeded.

Exact flash layout, checksum algorithm, versioning, interrupted-write policy and write endurance are implementation design tasks; protect the bootloader and config partition from accidental erasure.

## 6. Status model

Expose **four independent one-byte bitfields**: one board/node status and one status per motor. Bit mappings below are **provisional proposals**, not a locked wire format; Codex should propose a documented canonical map before protocol implementation.

Node conditions may include `READY`, `NO_CONFIG`, `WATCHDOG_TIMEOUT`, `ARMED`, and presently undefined hardware/configuration conditions. Motor conditions may include `READY`, `ACTIVE`, lower/upper travel limit, current limiting, driver fault indication, and inhibited output. Multiple concurrent conditions must be representable. Status should describe things the node can actually detect: e.g., don't claim motor current limiting from a DAC setting alone.

**No error history** on the node. Current conditions are reported; transient normal limits need not latch. Earlier discussion proposed latching actual faults, but also concluded that a generic fault engine and clearing protocol were premature. **Do not implement general-purpose latched fault history, auto-recovery policy or elaborate fault framework without a concrete condition and a reviewed behavior.** A runtime exception that necessitates disabling outputs can be handled locally and surfaced in status as needed.

## 7. Communications

### CAN FD — primary production interface

- Six identical nodes on a common CAN FD bus; physical address from three switches.
- Node addressing, query/response, broadcast rules, payload formats, frame sizes, identifiers, telemetry intervals, versioning and errors must be **designed and documented as a Codex work item** and reviewed before committing the wire format.
- Nominal master cycle >= 50 Hz; normally one frame carries all three signed 8-bit motor commands for a node.
- All available sensors and active configuration fields/pages are interrogable; configurable periodic status/telemetry eliminates unnecessary query traffic.
- Minimal command repertoire includes motor output, ARM, DISARM, get status, sensor query, config read/write/save/load, node reboot, and firmware update support. Interfaces must not confuse supervisory state with sensor data.
- No higher-level motor trajectory or pose decisions in node.
- `NODE_REBOOT` is protected against accidental triggering by **two individually addressed transactions in sequence**, e.g. `REBOOT_ARM` then `REBOOT_NOW` + magic constant, with short expiry and one-shot consumption; exact key/timeout/message IDs must be documented. Reboot clears commands, leaves outputs safe, resets, and returns disarmed.

### USB CDC — service/debug

USB CDC ACM is **already physically verified** (enumerates as a Windows COM port and bidirectional communication confirmed). Keep available for bench diagnostics/configuration and debugging; normal deployment does not depend on a USB cable. CAN and USB should invoke shared application services, not duplicate motor/configuration logic. Serial commands may be human-readable; CAN need not contain labels.

## 8. Bootloader and firmware updates

Use **Feaser OpenBLT**, targeting its established **CAN/CAN FD update architecture and XCP protocol**, with minimum board-specific adaptation. Do not conflate with ST's separate similarly named middleware. Examine the **actual OpenBLT source, licensing, STM32G474 support, flash erase geometry, FDCAN2 requirements and master-side host integration** before selecting memory and linker addresses.

Requirements:

- Master-driven firmware update over CAN; individually identify one of six nodes via physical node-ID switches, including when application firmware is missing.
- Resident bootloader must permit recovery after an interrupted/invalid application update without needing a functioning application.
- Keep motor outputs safe throughout bootloader operation.
- Firmware updates must not overwrite retained configuration.
- Preserve OpenBLT's standardized transfer and bootloader activation behavior unless a demonstrable requirement necessitates a board-specific change. Ordinary two-stage reboot protection **does not automatically imply changing OpenBLT's entry protocol**.
- Normal `NODE_REBOOT` remains separate from intentional firmware-update initiation.
- STM32G474RBT3 has 128 KB flash; a dual-image solution cannot simply be assumed. **Flash partition reservation must be designed early**, while actual OpenBLT implementation may be gated later.

## 9. Expansion and modularity

Board exposes I2C1. Some nodes may have future optional external modules including an **IMU** and a **status LCD**. Firmware should leave clean module hooks and configuration/extensibility paths without implementing those devices now or conflating them with the onboard DAC7574 sharing I2C.

Prefer lightweight modules: hardware-specific board support; sensing; motor output/protection; configuration and flash store; serial service; CAN transport/protocol; status; bootloader boundary. Keep CubeMX-generated initialization in generated areas and application code in separately owned files and user-code sections. Keep a responsive bare-metal architecture unless measured timing requirements justify more infrastructure. No unnecessary dynamic allocation, framework or RTOS.

## 10. Development baseline and test environment

- Project root: `C:\krabby-research\hardware\PCB-v0.3`; all firmware under `Firmware/`.
- MCU: STM32G474RBT3, LQFP64; STM32CubeMX, CMake/GCC via STM32Cube VS Code extension, ST-Link V2, and USB CDC working.
- Existing bring-up: SWD ID/memory read, build/debug/flash, LED blink and SysTick handler, USB CDC COM enumeration and bidirectional serial, three node switches, LED GPIO board functions.
- **Verified correction:** NODE_ID0 PC13 has value **4**, NODE_ID1 PB4 **2**, NODE_ID2 PB5 **1**. All switches active-low; values match board silkscreen. Older pin-definition documents imply different bit ordering and must not overwrite this correction.
- Initial bench conditions: approx. 12 V current-limited bench supply (~100 mA), no motors/other loads attached, motor-driver fuses removed. These are **current test conditions, not assumptions about final product hardware**.
- CubeMX-generated interrupt settings previously omitted SysTick; corrected via CubeMX NVIC code-generation and confirmed 1 Hz LED timing. Do not regress.

## 11. Explicitly delegated design work / unresolved decisions

1. Full **CAN FD wire protocol** with numeric IDs, lengths, formats, master response routing, broadcast rules, rate/bandwidth assumptions, retries and protocol versioning. Must be documented and reviewed before implementation.
2. Full **configuration schema**: node/motor pages, types, scaling units, defaults, mode reservations, field IDs and service semantics. Document before freezing it in code.
3. **OpenBLT integration feasibility and memory map**: exact OpenBLT distribution, MCU flash layout/linker offset, config reserve, recovery verification and licensing. Resolve allocation early; implementation later.
4. **Motor protection semantics**, including current DAC conversion, limit behavior, position-limit direction/feedback failures and actual DRV8262 status observability. Keep decisions grounded in circuits and bench tests.
5. Specific **watchdog default timeout** and precise timing from ARM / first motor frame.
6. Status **bit assignments, meaning and initialization**, and any real hardware conditions that merit special recovery behavior.
7. CAN FD vs previously generated classic-CAN CubeMX configuration: set up and verify CAN FD explicitly before claiming production readiness.
8. Valid handling of optional future I2C devices and telemetry bandwidth; no IMU/LCD driver yet.

**Decision discipline:** Do not silently turn proposals into firm requirements. Record tradeoffs, ask for approval when changing established behavior, and use small testable stages.
