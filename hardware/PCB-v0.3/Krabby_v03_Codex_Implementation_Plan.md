# Krabby Uno v0.3 — Codex Execution Plan and Acceptance Gates

**Purpose:** Incrementally implement the firmware defined in `Krabby_v03_Firmware_Requirements.md` without assuming a functioning master controller. This is an ordered engineering plan, **not** a demand for formal commits or release artifacts after every bench experiment.

## Working contract for every gate

1. Inspect current firmware and verified board documentation before edits; preserve working startup, USB CDC, debug, status LEDs and safety pin behavior.
2. Clearly distinguish **confirmed requirement**, **proposed engineering design**, and **unresolved decision**. Do not invent master behavior or change established control semantics.
3. Prefer CubeMX for peripheral configuration. Keep application modules separate from generated files and use durable CMake integration. Verify generated files after any CubeMX run; retain SysTick/NVIC fix.
4. Use the existing USB CDC terminal for diagnostics. Do not make functioning CAN/master a prerequisite for testing the node.
5. Keep hardware output safe and respect actual bench conditions; do not assume motors are connected. Before driver exercise, verify power domains, DAC limit and protections.
6. Build after changes. Flash/debug when the specific task authorizes it. When physical tests are needed, specify expected signals and ask for measurements rather than claiming a bench pass.
7. Report only: files changed, what was implemented, build/flash result, tests performed and their evidence, user bench checks, and unresolved issues. No automatic commits, branches or release process.
8. For each gate, supply a short reproducible acceptance test. Keep throwaway diagnostic code disposable; preserve confirmed discoveries in documentation.

## Early design gate — memory and bootloader feasibility (no bootloader implementation yet)

**Deliverables:** Investigate the selected Feaser OpenBLT STM32G474/CAN FD implementation; determine board-specific differences (FDCAN2, physical node IDs, flash geometry, config page); propose a feasible 128 KB partition/linker strategy, recovery path and any legal/license dependencies. Record proof and open questions. **Do not relocate the working application until reviewed.**

**Acceptance:** Written memory layout proposal checked against actual device flash parameters and specific OpenBLT code; normal firmware build unaffected. This must happen early enough to avoid locking application code to an incompatible flash address.

## Gate 0 — existing validated bring-up

**Status:** Already achieved by user testing: ST-Link SWD ID/memory access, GCC/CMake compile, flashing/debugger, PC8 GPIO 1 Hz, USB CDC enumeration and bidirectional communication, node-ID switches, LED module. Reconfirm current workspace state; don't needlessly recreate.

## Gate 1 — application modules and active configuration

**Implement:** Clean minimal C module layout; node config + three motor pages in RAM; factory defaults; field read/write service and consistently formatted USB inspection/edit commands. Keep future per-page numeric access possible. Mode 0, mode 1 fields as documented, modes 2+ reserved. Include fixed switch-order correction.

**Acceptance:** USB can enumerate/read/write individual values on all three motor pages, with no recompile and no unrequested flash write. Build/debug/LED/USB remain functional. Record config schema *draft* for review.

## Gate 1A — CAN FD protocol design (design-only review gate)

**Implement:** Design/document identifiers, physical node addressing, command/response frames, signed three-motor updates, node/motor status, parameter/page transactions, telemetry subscriptions, reboot handshake, and version negotiation. Include arbitration/bandwidth estimate at six nodes and >=50 Hz plus representative telemetry, explicit error/invalid-frame behavior, and OpenBLT coexistence plan.

**Acceptance:** Complete human-readable protocol specification and machine-readable constants/schema proposal, example frames and expected responses, plus questions that affect product semantics. **Do not implement or freeze wire protocol until reviewed.** Do not claim classic CAN configuration is CAN FD.

## Gate 2 — acquisition and feedback

**Implement:** Continuous/paced ADC acquisition for M1–M3 current sensing, potentiometers, board thermistor and VBUS; configurable filtering; encoder acquisition TIM2/TIM8/TIM5; per-motor feedback source and report modes 0 (native filtered ADC/counts) and 1 (calibration-normalized), ensuring raw diagnostic access.

**Acceptance:** USB reports stable readings; externally stimulated inputs and encoder counts change as expected. Document sample rates, filter math, encoder channel direction (M2 A/B reversed relative to timer CH1/CH2), calibration scaling. No motor output required.

## Gate 3 — DAC and basic motor output hardware

**Implement:** DAC7574 independent channel programming for three motor VREF limits; low-level DRV8262 PH/EN motor outputs; PWM duty, direction and configured safe brake/Hi-Z behavior. Coordinate output enable with DAC readiness. Do not add closed-loop motion control in mode 0.

**Acceptance:** With bench-safe fixture, measured DAC reference levels, GPIO PH direction, and PWM waveform match commands on all three channels; no unintended output on reset/recompile/reconnect. Distinguish electrical test from motor test.

## Gate 4 — arming/watchdog/status/protection

**Implement:** Board-wide DISARMED/ARMED permission. ARM zeroes command registers and preserves neutral output. Only a **subsequent** motor command applies output. DISARM and board-wide watchdog timeout disarm, clear commands, apply each configured safe state; new ARM required for recovery. Node/motor status bytes are independent condition bitfields. Minimal hard protections only; no historical event log or generic fault framework.

**Acceptance:** Demonstrate each transition with USB and GPIO/PWM scope. Replayed pre-ARM motor command never causes movement. Timeout works with all three outputs and individual safe settings; irrelevant requests do not refresh watchdog. Document watchdog default and ARM timer starting point for approval.

## Gate 5 — CAN FD electrical/transport bring-up

**Implement:** FDCAN2 CAN FD configuration and driver with hardware node ID, frame filters, error observability and basic diagnostics. Initially use loopback and external CAN FD interface; then apply reviewed application-layer protocol for three-motor command frame, status and query/replies.

**Acceptance:** Properly addressed node responds, other node addresses do not, 50 Hz command stream and diagnostics work, watchdog is refreshed **only** by valid motor frames, and USB remains functional. Master may be a PC script/test fixture.

## Gate 6 — persistent configuration

**Implement:** Flash image header and checksum/CRC; SAVE active RAM to nonvolatile, LOAD verified image to RAM, default reset; `NODE_NO_CONFIG` on missing/invalid/incompatible stored image. Protect OpenBLT and configuration regions. Checksum is **not** an ARM gate and does not validate mechanical calibrations.

**Acceptance:** Save/reboot/load round-trip; erased/corrupted saved data produces identical `NODE_NO_CONFIG` path; provisional RAM config can still be edited and used for explicit ARM/calibration. Flash operations do not overwrite program or bootloader. Test interrupted save/recovery approach.

## Gate 7 — operational telemetry and management

**Implement:** Configurable periodic telemetry, request-based interrogation of all sensors/status, page transfers, reboot two-stage authorization, protocol version/capabilities. Existing USB commands remain useful service functions. Future I2C expansion hooks only.

**Acceptance:** Bus-rate test with representative six-node traffic, node-specific reboot does not reset neighbors, incomplete/wrong-key reboot handshake never resets, CAN/USB configuration access consistent. No full robot master necessary.

## Gate 8 — OpenBLT CAN firmware update

**Implement:** Selected original OpenBLT CAN FD/XCP solution adapted as minimally as feasible for FDCAN2, 128 KB memory, board ID switches, flash partition and safe outputs. Preserve OpenBLT expected activation/update conventions unless reason to deviate is documented and approved.

**Acceptance:** One node updated over CAN using compatible host interface, new application boots and reports version, saved configuration preserved, interrupted/invalid update recoverable over CAN without valid application, other nodes unaffected.

## Optional later gates

- IMU and LCD on optional I2C expansion, including device detection, configuration and interrogation.
- Further motor control/feedback/current protection modes **only when specified**.
- More sophisticated master-managed startup/config distribution and empirical calibration procedures on master.

## Sequencing notes

The order is a practical dependency guide, not an absolute waterfall. For example, a CAN FD loopback test can occur before fully defining each telemetry field. Conversely, the OpenBLT **memory plan** must be resolved early even though bootloader implementation is late. If a hardware test reveals a defect, isolate and document it before continuing. Unresolved semantic choices must return for review rather than silently hardening into code.
