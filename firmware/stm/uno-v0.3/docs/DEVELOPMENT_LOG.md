# Application development log ? 2026-11-11

Scope: autonomous remaining-firmware assignment; one physical node 0, no motors or
loads. New assignment authorizes driver testing but requires proven prerequisites.
No option bytes, bootloader flash, CANable firmware, or hardware protection changed.

## Configuration and shared services

Added App/Inc/app_service.h and App/Src/app_service.c; wired CMake and CubeMX user
sections. One RAM dataset (4 logical pages, 192 bytes), factory defaults, parameter
and whole-page reads/writes, validation, per-page/all defaults, SAVE/LOAD, CRC and
schema metadata. Active RAM is authoritative; NO_CONFIG never gates arming.
The stored image is 208 bytes inside the existing final 4 KiB reservation. Schema 2
adds optional NTC coefficients. Saves are disarmed/Hi-Z only, payload before header.
One-page configuration-loss risk remains accepted; there is no redundant copy.

## Acquisition and hardware backend

Paced ADC1/2 scan groups and configurable integer IIR filters; TIM2/8/5 encoder
counter extension (M2 sign corrected), pot/encoder selection, native/normalized
position, NTC resistance and optional configured beta conversion, nominal USB VBUS
and current reporting. DAC7574 selected-channel writes and repeated-start digital
readback per TI. PCB archive identified R59/60/61 as 1.43k; R33 beta is unspecified.

Initial scan testing found occasional interrupt-induced overrun. Changed only ADC
LowPowerAutoWait in the IOC and regenerated using installed CubeMX; user sections,
CAN timing, linker/vector signature and USB stayed intact. Final tests report zero
ADC errors. No guessed Celsius calibration; beta=0 reports unavailable sentinel.

## Control and protocol

Direct signed duty, PH/EN output backend, 20 kHz PWM, wake/reversal delays, brake vs
Hi-Z, live nFAULT and travel restrictions. Boot has sleeps LOW and MOE off. ARM
initializes neutral PWM only after DAC readiness and clears commands. Fresh motor
transactions required after ARM, including generation fences for queued CAN and
partially received USB commands. Watchdog defaults to 250 ms starting at ARM,
SysTick-enforced independently of communication queues; timeout requires rearm.
Specific hardware failures force Hi-Z; no generic fault-history framework.

Protocol v1 documented in APPLICATION_PROTOCOL.md, with machine-readable schema and
Python reference library tools/krabby.py. Shared CAN/USB command backend, bounded
queues, periodic rotating telemetry, two-stage one-shot transport-scoped reboot.
Existing bridge/USB diagnostic commands remain. Bootloader entry forces immediate
sleep before reset. No autonomous position controller, IMU/LCD or RTOS added.

## Build, deployment and acceptance

Final Debug image: 84,280 flash bytes of 114,688 available; RAM allocation 23,456 bytes
of 131,072 including configured heap/stack. No compiler warnings from application
changes. All ordinary deployments used CAN OpenBLT and complete-image readback.
Known-good ELF/SREC/BIN/MAP are archived under bench/application/known_good.*.

37 physical application checks passed: configuration isolation/defaults/bounds,
CAN/USB consistency, actual ADC sampling, encoder-register injection (not physical
encoder motion), DAC digital readback, inhibited signed commands, ARM/watchdog
transitions and 50 Hz command stream, queued pre-ARM USB rejection, malformed and
wrong-node requests, rotating telemetry, reboot authorization, save/load/reset,
corrupt/erased/partial configuration, and interrupted application update recovery.
ST-Link supplied resets and page-scoped configuration fixtures, not bootloader
reprogramming. Interrupted firmware was recovered using CAN programming only.

Stress: 400 diagnostic frames submitted with no USB reader, watchdog still expired;
500 CAN status queries, 100 cross-interface page comparisons, 3 human USB commands,
and 50 command cycles at 50 Hz addressed to one real + five absent node addresses.
Diagnostic queue overflow was observed/reported while application queues and ADC
remained healthy. Absent-node traffic is not six-board physical validation.

CAN bridge regression: 12 matrices, 1,680 deliveries passed at 1M/5M with zero bus
errors/loss. Updated test greeting match for application identity and excluded
reserved service identifiers from diagnostic allocation at high repeat counts.
Earlier test-client defects (short malformed packets, non-repeatable encoder counter
injection and watchdog expiry during slow ST-Link inspection) were fixed. Live flash
fixture programming required under-reset connection, not HOTPLUG; no options changed.

## Remaining supervised validation

VM supply is not ADC-observable; earlier authoritative bench notes report removed
motor fuses. Supply/fixture prerequisites could not be independently established.
Accordingly no driver was intentionally awakened for energized tests. Physical
GPIO reads verified all sleeps LOW; PWM startup MOE was verified off. Electrical
PWM/PH waveforms, brake behavior at driver outputs, DAC VREF analog voltages, loaded
current regulation, mechanical travel and externally driven encoders remain unproven.
NTC beta must be supplied from the actual part before calibrated Celsius is useful.
Actual power cuts/brownout/ECC during flash operations and multi-board operation are
not qualified. CRC-corrupt records were tested; uncorrectable flash ECC is a separate
unverified failure mode that may require ST-Link recovery. Commercial OpenBLT license
clearance remains a release decision. This is commissioning firmware, not a claim of
completed loaded-motor or safety qualification.


## 2026-11-12 ? separate writable PC control console

Preserved tools/krabby_debug_gui.py as read-only. Added krabby_control_gui.py and
CONTROL_GUI.md: shared live diagnostics, staged parameter/page edits, defaults,
active-RAM export/import, SAVE/LOAD, explicit ARM/DISARM, signed snapshots, zero,
repeat, protected reboot, telemetry receiver, raw/legacy USB commands and a front
end for the existing OpenBLT uploader. No firmware, option or protection changes.

One I/O worker prioritizes motor deadlines and STOP, with generation-tagged GUI
motor jobs, no automatic rearm, repeat OFF initially, and best-effort DISARM on
close. Repeat interval capped at watchdog/4; ARM+repeat starts with fresh zeros.
Fixed coarse Windows event-wait timing by using short sleeps; requested 20 ms gaps
were observed around 20.7 ms maximum in short final bench runs. Short 50/10 ms and
disabled-watchdog profiles also exercised. Timing remains best-effort, not guaranteed.

Added test_control_gui.py. Synthetic protocol/GUI tests cover all action families,
staged rendering, missed deadline detection, stale queued commands, STOP and close.
Physical CAN/USB workers on the current board (now reporting physical node 7)
exercised zero-only repetition, parameter writes, flash SAVE/LOAD and restoration
of original RAM and saved datasets. Hidden-window real GUI handlers also exercised
connect, repeat/stop, staged write/restore and all periodic telemetry topics.

Physical tests required pre-existing Hi-Z policies in RAM and flash and left the
board disarmed. No energized nonzero drive was attempted. Default/reboot/raw/update
handlers were tested synthetically; the control GUI did not reflash firmware.
Evidence: bench/application/control_gui*_results.json; tools/CONTROL_GUI.md.


## 2026-11-12 ? separate hold-to-run Jog version

Added tools/krabby_jog_gui.py, test_jog_gui.py and JOG_GUI.md. Existing writable
and read-only GUI source files remain unchanged (fingerprints checked). All old
tabs are retained, plus per-motor direction buttons and editable jog magnitude.
Press stops prior repetition, arms, verifies the hold is still current and sends
fresh watchdog-paced commands. Release sends all-zero then DISARM; queued or
in-flight ARM cannot become delayed nonzero motion. Leaving the tab/focus, minimize,
Escape, STOP or closing cancel the hold. One jog button at a time; no auto-rearm.

Synthetic tests cover all six directions, repetition, quick release before/during
ARM, no restart, zero-before-disarm wire order, ARM rejection and closing while held.
Physical CAN and USB zero-only tests on node 7 exercised all six press/release paths.
Configuration stayed unchanged; final commands zero and board disarmed. No nonzero
energized motion, firmware update or flash/configuration write was performed.
Evidence: bench/application/jog_gui_test_results.json and jog_gui_sha256.json.


## 2026-11-12 ? PC-only position-control console

User confirmed Python-level loop only; no firmware changes authorized. Added
krabby_position_gui.py as a separate version, retaining prior tools. New tab has
Enable, normalized target sliders, per-motor 0..127 maximum drive boxes, live
normalized/current/drive readouts, P gain/deadband and optional output-sign reversal.
Normalization is local from native sensor data and existing calibration, independent
of firmware reporting mode; no automatic configuration writes. Enable initializes
targets to feedback and keeps each channel neutral until its slider moves.

Synthetic plant and UI tests passed. CAN and USB physical tests used zero speed
limits only, and hidden real-GUI tests checked enable/sliders/readouts/disable.
Board finished disarmed, all commands zero, RAM unchanged. No firmware/configuration
flash writes or energized motor tracking tests. Supervised direction/calibration
checks and stability/load tuning remain. See tools/POSITION_GUI.md and position_gui
bench results. Position loop stays on PC; firmware remains direct motor I/O.


## 2026-11-12 ? Firmware-owned polarity and selectable travel units

Configuration schema 3 uses reserved node fields 8/9 as per-motor output-reversal and normalized-limit masks. Page sizes, flash allocation and CAN timing unchanged. Schema-2 stored configuration is accepted with zero masks; the physical board retained all original values. Command signs now describe feedback direction; electrical reversal occurs after shared travel inhibition. Limit status and enforcement share native/unsaturated-normalized comparison. Configuration changes seed ADC filters from latest samples, fixing a discovered false zero-position ramp.

Python position control no longer reverses commands locally. Its board configuration editors write reversal, units and explicit lower/upper thresholds while disarmed; SAVE remains explicit. Generic configuration schema, JSON import/export and GUI schema checks updated. Old schema-2 JSON imports with zero masks. All GUIs use the shared firmware polarity.

Built and CAN-programmed 84,812 bytes with complete readback. Actual node 7: 42 energized polarity/travel checks, exact encoder equality checks, invalid-mask rejection, reset persistence, real GUI worker configuration transaction, and PC position-controller motion in both directions on all three motors passed. Synthetic Control/Jog/Position tests passed separately. Higher test P gain was required to overcome low-PWM stalling; defaults unchanged. Board left disarmed with original settings plus measured reversal mask 3 (M1/M2 reversed, M3 normal). Evidence and repeat commands: bench/travel_limits/README.md. No CubeMX regeneration required.


## 2026-11-12 ? Move polarity and travel units onto motor pages

Schema 4 moves reversal and normalized-limit selection to motor fields 12/13 (offsets 48/52), both booleans. Node masks removed; node wire page remains 48 bytes, motor wire pages expand to 56 bytes. Internal stored records are 56 bytes each (node has two zero padding fields), dataset 224 bytes plus 16-byte header, within the unchanged 4 KiB configuration reservation. Full motor SET is 62 bytes including request header and fits a single CAN FD frame. Motor flags, thresholds and units now update atomically together.

Firmware startup/LOAD migrates schema-2/3 flash without modifying flash until SAVE. Actual schema-3 saved settings migrated exactly, including M1/M2 reversal and all original limits/current/calibration settings. Python page reads/writes, schemas, generic configuration tables, per-motor editors, JSON import/export, tests and USB text commands updated. Schema-4 SAVE/reset persistence verified.

Build/CAN readback: 85,328 application bytes. Repeated 42 physical travel checks and real PC position motion on all three motors in both directions. CAN/USB complete-page reads, bounds/boolean rejection and actual GUI-worker motor-page transaction passed. Synthetic Control/Jog/Position regressions passed. Board left disarmed with original settings preserved. Evidence: bench/motor_pages/README.md. No .ioc or linker changes; no regeneration required.


## 2026-11-12 ? LoadMeFirst build/package

Separate LOAD_ME_FIRST CMake option selects App/Src/load_me_first.c instead of motor services. Eight GPIO LEDs chase at 100 ms/step, USB hello identifies image, motor sleeps remain low and MOE off. VERSION/schema 0 discovery and CAN bootloader re-entry retained; ARM/MOTOR/config operations unsupported. Minimal build 35,804 bytes; unchanged OpenBLT 12,000 bytes. Package tool generates signed combined Intel HEX and application-only S-record, excluding configuration and option bytes. Combined flash span 48,092 bytes. Normal application also rebuilt. Software image/checksum checks passed; no hardware programming in this task. USB DFU initial install must confirm DBANK=0 and keep motor power off; see bench/load_me_first/README.md for protection and physical acceptance caveats.


## 2026-11-12 ? Factory MCU unique ID

Added read-only application opcode 14 UID (three little-endian uint32 words) and USB node uid command. VERSION capability bit 5 advertises support; schema unchanged. Python Node.uid() formats word0/1/2 into 24 uppercase hex digits. Shared GUI identity line populates at connect; older firmware shows unavailable. LoadMeFirst also exposes UID and has regenerated provisioning artifacts.

Normal firmware 85,512 bytes programmed/read back over CAN to physical node 1 (switch setting changed since earlier node-7 tests). UID 00200021524E501020343738 matched over CAN and USB. All configuration pages unchanged, board disarmed, commands zero. Control/Jog/Position GUI synthetic tests passed, including the new connect-time query path. Results and updated motor image in bench/uid. LoadMeFirst builds at 35,972 bytes; its LED/application behavior still awaits physical acceptance.


## 2026-11-12 ? Compact binary generation and artifact cleanup

Fixed root CMake post-build binary conversion to use the addressed S-record generated from the ELF. Empty loadable TLS sections no longer inflate the raw image toward RAM addresses. No linker, firmware behavior, or address-layout changes. Regenerated seven oversized binaries from matching ELF/S-record data, verifying record checksums, address bounds and byte equality before replacing each file. Historical known_good.bin remains its original 84,280-byte firmware, not a new application version. Normal and LoadMeFirst rebuilds passed, producing 85,512-byte and 35,972-byte binaries. Reclaimed 2,818,034,604 bytes. Per-file evidence: bench/binary_cleanup/results.json. No hardware accessed.


## 2026-11-12 ? Pre-commit source/tooling audit

Audited Firmware only, no hardware access. Clean Debug/OpenBLT/LoadMeFirst builds match existing binaries. Fixed stale schema expectation in physical acceptance, added explicit hardware guards, portable CubeProgrammer lookup, synthetic-test evidence preservation, build/dependency documentation, LoadMeFirst preset, repository ignore/attribute hygiene and offline artifact verification. Native upstream Windows executables remain locally available but are ignored pending complete distribution provenance; missing USB middleware standalone license remains documented. Full findings: docs/PROJECT_AUDIT.md; machine-readable verification: bench/project_audit/results.json. No embedded firmware behavior, peripheral configuration or memory layout changes.
