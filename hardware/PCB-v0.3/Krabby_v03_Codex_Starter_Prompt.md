# Codex Handoff — Krabby Uno v0.3 Firmware

**Paste the following into Codex after placing the companion documents in the firmware workspace (or giving Codex access to their paths).**

---

You are assisting development of the Krabby Uno v0.3 STM32G474RBT3 three-brushed-motor CAN FD I/O node.

## Authoritative inputs

Read the following completely before proposing edits:

1. `Krabby_v03_Firmware_Requirements.md` — product intent and non-negotiable semantics.
2. `Krabby_v03_Codex_Implementation_Plan.md` — staged work and acceptance gates.
3. `Krabby_v03_Decision_Record.md` — settled decisions, revisions, deliberate omissions and unresolved questions.
4. The **current** repository firmware, `.ioc`, PCB/hardware-definition documents, CMake files and current working implementation. The documents above describe intended behavior; verify what is actually built before changing anything.

If these planning documents are outside the open `Firmware/` workspace, ask for the exact accessible paths instead of guessing.

## Engineering stance

The node is a transparent three-motor bus expansion module. The master makes movement decisions; the node applies signed power commands, acquires sensors and restricts hardware operation when necessary. Default motor control mode is 0 (no local motion loop). Configuration is in active RAM and may be saved/loaded to flash; commissioning may start with partial provisional RAM settings. Never turn checksum verification into a requirement to ARM. ARM always zeroes stored motor commands; only a later new motor command can cause movement. A watchdog timeout disarms and requires a new ARM.

Use the specific existing board mappings, including the verified switch correction: PC13 = bit value 4, PB4 = 2, PB5 = 1, active low. Preserve working USB CDC, ST-Link debug, SysTick, LED functions and motor-safe GPIO behavior.

Do not invent product semantics for ambiguous matters. You may make technical implementation proposals, but identify whether they are confirmed requirements or recommendations. Avoid creating an RTOS, elaborate fault subsystem, calibration intelligence or software motion controller unless a requirement clearly demands it.

## First work item — inspect, reconcile and plan; don't implement yet

1. Audit actual source tree, CubeMX-generated files, CMake integration, existing USB/LED/board modules and installed build/debug tools. Identify what presently works and what exists only as a plan.
2. Compare the current `.ioc` against the requirements. Identify any conflicts, particularly FDCAN2 classic CAN vs intended CAN FD, NVIC, ADC/encoder configuration, output safe states and regenerated USB middleware. Do not silently rewrite these settings.
3. Investigate **Feaser OpenBLT** upstream implementation appropriate for STM32G474 and CAN FD (XCP), including FDCAN2 board porting, application reset/entry behavior, licensing, 128 KB flash erase geometry and configuration flash preservation. Propose a realistic flash partition and linker strategy with evidence. This is an early design deliverable, **not authorization to install/relink a bootloader yet**.
4. Propose an application-owned C module structure with a configuration RAM service, board support, sensors, motor output/protection, status, USB service and future CAN transport. Specify which files are generated and how hand-maintained sources survive CubeMX regeneration.
5. Identify the smallest safe next coding increment (normally active RAM node/motor configuration read/write through USB) and its acceptance test. Flag decisions that must be reviewed before implementation.

## Deliver response

Provide:

- Verified current project state, citing exact source paths and configuration fields.
- Conflicts or risks discovered; **do not claim a build or hardware test you did not perform**.
- Proposed bootloader/flash memory layout, clearly marked pending approval.
- Proposed module structure and order of work.
- A specific Gate 1 implementation plan with files to touch, commands/tests to run, and measurable acceptance.
- At most a handful of questions that truly affect requirements. Do not stop for trivial naming or internal implementation choices.

**During this first task: read, analyze and propose. Do not change files, compile, flash, erase, modify MCU option bytes, or operate motor outputs.** After review, implementation will proceed one acceptance gate at a time with explicit permission for its scope.

## Subsequent gates

When instructed to implement a particular gate: work within that gate, make the smallest sound change, build and fix compilation errors, provide clear USB-based diagnostic methods, and report exact modified files, verified tests, and physical bench checks that require the engineer. Design and document the full CAN FD protocol as its own reviewable deliverable; it is intentionally **not pre-specified byte-for-byte**.
