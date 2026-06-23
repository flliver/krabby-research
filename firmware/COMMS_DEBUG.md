# Primary↔Follower Comms Failure — Root Cause & Repro

M17 Task 1, acceptance criteria **1c** (root-cause + captured logs) and **1d** (repro that
fails pre-fix, passes post-fix). Companion to [`SETUP.md`](SETUP.md) §2.1.

## Symptom

The USB-connected primary (FRONT) does not reliably drive the LEFT/RIGHT follower
boards. Observed as one or more of:

- host logs `… can't keep up …` / dropped or corrupt actuators in telemetry;
- a forwarded `get --board left|right` returns `no response from board`;
- followers visibly stop updating actuators.

## Which stage fails (staged analysis)

The forward path has four stages. Mapping each to what we found:

| Stage | Status | Notes |
|---|---|---|
| **SYNC role election** | N/A | Removed in Task 1 §3 (EEPROM roles). Not a factor post-refactor — see `eepromLoad()`/`applyRole()` in `arduino.ino`. |
| **Leader → follower forward** | OK | The leader does write the bare `SET`/`GET`/`J`/`B` line out `leftSerial`/`rightSerial`. Verified: bytes leave the leader. |
| **Follower receive / apply** | **FAILS** | Two independent causes — a firmware starvation bug and wiring (below). This is where it actually broke. |
| **Hardware / wiring** | **FAILS** | Wrong JST port and a half-open cable (below). |

## Prime suspect (RX buffer) — investigated, *not* the bench bug

The originally-documented prime suspect (SETUP.md §2.1) was the AVR `HardwareSerial`
RX buffer defaulting to **64 bytes**, too small for a leader holding a ~200-byte
forwarded telemetry line from each of Serial1/Serial2. The Makefile now bakes
`-DSERIAL_RX_BUFFER_SIZE=256` into every build (matching CI), guarded by
[`test_makefile_build_flags.py`](../tests/unit/firmware/test_makefile_build_flags.py).

**However, on the deployed core (`arduino:avr` 1.8.7) this define is a no-op:** that
core already sizes the Mega 2560's RX buffer at 256 bytes (it scales the ring buffer
by available RAM, and the 2560's 8 KB lands at 256). So the buffer was already 256 on
the bench, and bumping it changed nothing there.

**We keep the define anyway** as defensive hygiene — it guarantees 256 on any core
version or board variant whose default falls below it, and makes `make`, CI, and IDE
builds produce an identical binary. But it did **not** cause or fix the failure we
actually hit. The real causes follow.

## Root cause #1 — floating-RX flood starvation (firmware)

**Mechanism.** A follower's `loop()` drains its uplink (`mainSerial`) *before* it
reaches `processConfig()` (the USB config handler) and the actuator update. The drain
was an **unbounded** `while (mainSerial->available()) { … }`. A follower whose uplink
RX line is **disconnected or dangling** floats — the bare pin acts as an antenna, EMI
induces a continuous stream of phantom bytes on Serial1/Serial2, and the unbounded
drain **never exits**. Result: USB config *and* the actuator update are starved. The
board looks dead ("no response") even though USB is fine — it's stuck draining noise.

A bare header pin stays quiet enough; a ~30 cm dangling cable is the antenna that
triggers it.

**Captured evidence.** With a dangling cable on a follower's Serial2 and
`KRABBY_MCU_RAW_RX=1`, the host sees a relentless stream of garbage bytes on that
channel, and `get --port <follower> role` over USB times out: `0/N` reads succeed.

**Fix** (commit `042b319`, `arduino.ino`):

1. Pull the follower RX pins up so a disconnected line **idles high** instead of
   floating into noise (a driven leader TX still overrides):
   ```cpp
   #define SERIAL_LEFT_RX  19   // RX1
   #define SERIAL_RIGHT_RX 17   // RX2
   // in setup(), after the Serial begins:
   pinMode(SERIAL_LEFT_RX,  INPUT_PULLUP);
   pinMode(SERIAL_RIGHT_RX, INPUT_PULLUP);
   ```
2. **Bound** every drain loop so no single channel can starve the rest:
   ```cpp
   constexpr int RX_DRAIN_BUDGET = 64;   // max lines/bytes drained per pass
   // loop() main drain and processConfig() both:
   int rxBudget = RX_DRAIN_BUDGET;
   while (port.available() && rxBudget-- > 0) { … }
   ```

**Before/after.** RIGHT on USB with a dangling cable on Serial2: **0/N** `role` reads
→ **8/8** after the fix. Guarded by
[`test_floating_rx_guard.py`](../tests/unit/firmware/test_floating_rx_guard.py).

> Operational relevance: this isn't only a debug-port nicety. A follower whose uplink
> works loose in the chassis (or a leader tri-stating its TX at boot) would otherwise
> freeze the follower's *control loop*, not just its USB console.

## Root cause #2 — wiring

Two distinct physical faults, both presenting as "follower never receives":

- **Wrong JST port.** RIGHT's inter-board cable was plugged into **Serial 1**, but
  RIGHT listens on **Serial 2**. Rule: *same Serial number on both ends* —
  **LEFT = Serial 1, RIGHT = Serial 2** (matches `SERIAL_LEFT = Serial1`,
  `SERIAL_RIGHT = Serial2`). The shield silk and `MOTOR_HEADER_PINOUT.md` agree.
- **Half-open cable.** A hand-crimped JST cable with one open conductor: the
  FRONT-TX→follower-RX leg was open while follower-TX→FRONT-RX was good. Telemetry
  flowed *up* but forwarded commands never reached the follower, so `get` failed in
  the *down* direction only.

**Diagnostic technique.** `KRABBY_MCU_RAW_RX=1` dumps raw serial RX; count the relayed
telemetry prefixes the leader emits — `FRONT;`, `LEFT ;`, `RIGHT;`. A **dead link**
shows the leader relaying **only its own `FRONT;`** lines, never any `LEFT ;`/`RIGHT;`
— proving no follower data crosses the link at all (so it's a link/wiring problem, not
a GET-parse problem). A **half-open** link shows `LEFT ;` telemetry arriving (up
direction alive) while `get` still fails (down direction open) — which pinpoints the
open conductor's direction.

```text
# dead link (follower not crossing): only FRONT; lines relayed
FRONT; FLHY 0.50 512 0 1 1 0 0 0;FLHL …;…
FRONT; FLHY 0.50 511 0 1 1 0 0 0;FLHL …;…           # no LEFT ; / RIGHT; ever

# healthy link: follower telemetry relayed through the leader
FRONT; FLHY 0.50 512 0 …;
LEFT ; RLHY 0.50 498 0 …;                            # follower data crossing → up
RIGHT; RRHY 0.50 503 0 …;
```

## Repro (AC 1d)

### Manual repro — demonstrates the bug pre-fix, passes post-fix

The floating-RX starvation is the firmware bug with a clean hardware repro:

1. Flash a board, assign it `role=RIGHT` (`set --port <p> role=RIGHT`).
2. Connect it to the host by **USB only**, and plug a ~30 cm jumper into its
   **Serial 2 RX** with the **far end unconnected** (dangling antenna).
3. **Pre-fix firmware:** `get --port <p> role` returns `no response from board`
   repeatedly (0/N) — the loop is stuck draining induced noise.
4. **Post-fix firmware** (pull-ups + `RX_DRAIN_BUDGET`): the same command returns
   `role=RIGHT` reliably (8/8). The dangling line idles high and the drain is bounded,
   so USB config is serviced every pass.

### Automated guards

- [`test_makefile_build_flags.py`](../tests/unit/firmware/test_makefile_build_flags.py)
  — the `-DSERIAL_RX_BUFFER_SIZE=256` build define (defensive; see caveat above).
- [`test_floating_rx_guard.py`](../tests/unit/firmware/test_floating_rx_guard.py)
  — asserts the RX pull-ups and the bounded `RX_DRAIN_BUDGET` drains are present in
  `arduino.ino`; fails if the floating-RX fix is reverted.

## Summary

The forward path itself was sound. The failure lived at **follower receive**, from two
causes: a firmware **floating-RX starvation** bug (fixed with RX pull-ups + a bounded
drain) and **wiring** (wrong Serial port + a half-open hand-crimped cable). The
RX-buffer define — the original prime suspect — is correct hygiene but was a no-op on
the deployed core and not the bench bug.
