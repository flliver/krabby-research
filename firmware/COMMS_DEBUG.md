# Primary↔Follower Comms Failure — Root Cause & Repro

M17 Task 1, acceptance criteria **1c** (root-cause + captured logs) and **1d** (repro that
fails pre-fix, passes post-fix). Companion to [`SETUP.md`](SETUP.md) §2.1.

> The root-causing was done on the (now abandoned) `m17` branch; the firmware fixes,
> guard tests, and the `SET`/`GET` config commands the analysis relies on are all
> ported onto this branch.

## Symptom

The USB-connected primary (FRONT) does not reliably drive the LEFT/RIGHT follower
boards. Observed as one or more of:

- host logs `… can't keep up …` / dropped or corrupt actuators in telemetry;
- no follower telemetry (`LEFT ;` / `RIGHT;` lines) relayed by the leader, and the
  combined `V` reply carries `-` in the follower slots (`show` prints no follower
  versions);
- a forwarded `get --board left|right` returns `no response from board`;
- followers visibly stop updating actuators.

## Which stage fails (staged analysis)

The forward path has four stages. Mapping each to what we found:

| Stage | Status | Notes |
|---|---|---|
| **SYNC role election** | N/A | Removed: roles now load from EEPROM on boot and are set explicitly via `SET role …` (see `eepromLoad()`/`applyRole()` in `arduino.ino` and SETUP.md §3 "Board roles"). The old election also required all three boards to boot within the same ~3 s window, but it was **not** the receive failure. |
| **Leader → follower forward** | OK | The leader does write the bare `T`/`B`/`J`/`H` line out `leftSerial`/`rightSerial`. Verified: bytes leave the leader. |
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

**Mechanism.** A board's `loop()` drains its main channel (`mainSerial`) *before*
everything else in the pass — the follower forward, the actuator update (and, on m17,
the USB config handler). The drain was an **unbounded**
`while (mainSerial->available()) { … }`. A board whose RX line is **disconnected or
dangling** floats — the bare pin acts as an antenna, EMI induces a continuous stream
of phantom bytes, and the unbounded drain **never exits**. Result: everything after
the drain is starved. The board looks dead ("no response") even though the link the
operator is using is fine — it's stuck draining noise.

A bare header pin stays quiet enough; a ~30 cm dangling cable is the antenna that
triggers it.

**Captured evidence (m17 bench).** With a dangling cable on a follower's Serial2 and
`KRABBY_MCU_RAW_RX=1`, the host sees a relentless stream of garbage bytes on that
channel, and `get --port <follower> role` over USB times out: `0/N` reads succeed.

**Fix** (ported from m17, `arduino.ino`):

1. Pull the follower RX pins up so a disconnected line **idles high** instead of
   floating into noise (a driven leader TX still overrides):
   ```cpp
   #define SERIAL_LEFT_RX  19   // RX1
   #define SERIAL_RIGHT_RX 17   // RX2
   // after the Serial begins:
   pinMode(SERIAL_LEFT_RX,  INPUT_PULLUP);
   pinMode(SERIAL_RIGHT_RX, INPUT_PULLUP);
   ```
2. **Bound** every drain loop so no single channel can starve the rest:
   ```cpp
   constexpr int RX_DRAIN_BUDGET = 16;   // max lines drained per pass
   int rxBudget = RX_DRAIN_BUDGET;
   while (mainSerial->available() && rxBudget-- > 0) { … }
   ```

**Before/after (m17 bench).** RIGHT on USB with a dangling cable on Serial2: **0/N**
`role` reads → **8/8** after the fix. Guarded by
[`test_floating_rx_guard.py`](../tests/unit/firmware/test_floating_rx_guard.py).

> Operational relevance: this isn't only a debug-port nicety. A follower whose uplink
> works loose in the chassis (or a leader tri-stating its TX at boot) would otherwise
> freeze the follower's *control loop*, not just its USB console.

### Leader-side twin — motor-EMI garbage capturing `loop()` (bench 2026-07-03)

The same failure mode hit the **leader** on a follower-less bench: the Serial1/Serial2
RX lines idle on the weak pull-up, and a 120 W brushed motor's EMI bursts punch
through it as a continuous garbage-byte stream into `forwardFullLines()` — whose drain
was also unbounded. `loop()` was captured for seconds at a time; jog-stop commands
were processed later and later until a motor ran away (FLHY). Additional hardening,
also ported:

- `forwardFullLines()` drains are bounded (`FWD_DRAIN_BUDGET`);
- only clean **printable-ASCII** complete lines are forwarded upstream — EMI framing
  garbage is dropped instead of becoming blocking TX writes;
- unknown bytes on the main channel are discarded **one byte at a time** — no
  blocking `readStringUntil()` (50 ms + a heap `String` each) on line noise;
- **RX0 (pin 0) is pulled up** too: the USB serial chip tri-states it when EMI knocks
  it off the bus (observed re-enumerating mid-session), leaving RX0 floating next to
  the motor;
- blocking line reads are capped at 50 ms (`setTimeout(50)`) instead of the 1 s
  default.

## Root cause #2 — wiring

Two distinct physical faults, both presenting as "follower never receives":

- **Wrong JST port.** RIGHT's inter-board cable was plugged into **Serial 1**, but
  RIGHT listens on **Serial 2**. Rule: *same Serial number on both ends* —
  **LEFT = Serial 1, RIGHT = Serial 2** (matches `SERIAL_LEFT = Serial1`,
  `SERIAL_RIGHT = Serial2`). The shield silk and `MOTOR_HEADER_PINOUT.md` agree.
- **Half-open cable.** A hand-crimped JST cable with one open conductor: the
  FRONT-TX→follower-RX leg was open while follower-TX→FRONT-RX was good. Telemetry
  flowed *up* but forwarded commands never reached the follower, so the *down*
  direction alone failed.

**Diagnostic technique.** `KRABBY_MCU_RAW_RX=1` dumps raw serial RX; count the relayed
telemetry prefixes the leader emits — `FRONT;`, `LEFT ;`, `RIGHT;`. A **dead link**
shows the leader relaying **only its own `FRONT;`** lines, never any `LEFT ;`/`RIGHT;`
— proving no follower data crosses the link at all (so it's a link/wiring problem, not
a protocol problem). A **half-open** link shows `LEFT ;` telemetry arriving (up
direction alive) while forwarded commands still fail (down direction open) — which
pinpoints the open conductor's direction.

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

The floating-RX starvation has a clean hardware repro:

1. Flash a board, assign it `role=RIGHT` (`set --port <p> role=RIGHT`).
2. Connect it to the host by **USB only**, and plug a ~30 cm jumper into its
   **Serial 2 RX** with the **far end unconnected** (dangling antenna).
3. **Pre-fix firmware:** `get --port <p> role` returns `no response from board`
   repeatedly (0/N) — the loop is stuck draining induced noise.
4. **Post-fix firmware** (pull-ups + `RX_DRAIN_BUDGET`): the same command returns
   `role=RIGHT` reliably (8/8). The dangling line idles high and the drain is bounded,
   so the rest of the pass runs every time.

A tooling-free variant of the same repro: with the dangling jumper and a brushed
motor running nearby, pre-fix firmware's telemetry stream stalls and jog stops arrive
late or never; post-fix telemetry keeps streaming at 20 Hz and jogs stop promptly.

### Automated guards

- [`test_makefile_build_flags.py`](../tests/unit/firmware/test_makefile_build_flags.py)
  — the `-DSERIAL_RX_BUFFER_SIZE=256` build define (defensive; see caveat above).
- [`test_floating_rx_guard.py`](../tests/unit/firmware/test_floating_rx_guard.py)
  — asserts the RX pull-ups, the bounded drains, the printable-ASCII forward gate,
  and the single-byte unknown-byte discard are present in `arduino.ino`; fails if
  any part of the fix is reverted.

## Summary

The forward path itself was sound. The failure lived at **follower receive**, from two
causes: a firmware **floating-RX starvation** bug (fixed with RX pull-ups + bounded
drains, plus leader-side EMI hardening after the 2026-07-03 runaway) and **wiring**
(wrong Serial port + a half-open hand-crimped cable). The RX-buffer define — the
original prime suspect — is correct hygiene but was a no-op on the deployed core and
not the bench bug.
