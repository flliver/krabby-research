# Krab OLED Status Render — Specification (M16 Task 2)

Contract for the krab status screen, and the hardware limits that shape it.

- **Target panel:** SparkFun Qwiic OLED 1.3" (`Qwiic1in3OLED`), 128×64, 1-bit.
- **Library:** SparkFun Qwiic OLED, pinned **v1.0.9** (newer versions pull in
  C++ `<map>` and break the AVR build).
- **MCU:** ATmega2560 (Arduino Mega), AVR, Harvard architecture.

## 1. There is one render

`DisplayRenderer<Canvas>` (`firmware/arduino/src/display/`) draws every
pixel. It is a template over the eight calls it makes, so the same code serves
three canvases:

| Canvas | Where | What it does with a draw call |
|---|---|---|
| `Ssd1306Canvas` | firmware | forwards it to `Qwiic1in3OLED` |
| `TraceCanvas` | `oled_sim/native/` | writes it out as a line of text |
| `RecordingCanvas` | native tests | stores it for assertions |

The sim runs `oled_trace`, which renders with `TraceCanvas`, and replays the
resulting draw calls through `ssd1306.py` — a verified port of the library's own
primitives and fonts. **What the sim shows is what the panel shows because the
same code drew both.**

This used to be two implementations kept in step by hand, and it drifted: the
sim had extend pointing up after the firmware flipped it, different leg
geometry, and none of Task 3's no-signal states. Anything in this document that
reads like "keep X and Y in lockstep" was a symptom of that and is gone.

If the sim and the panel disagree now, the bug is in `ssd1306.py`'s emulation of
a primitive — fix it there, never in the render.

## 2. Simulated state (`krab.KrabState`)

The browser supplies hardware-facing state. `oled_trace` builds each controller
state by constructing controller freshness and actuator glyph state, derives
IMU tilt, then calls `buildDisplayFrame`. Simulated battery voltages pass through
`setBatteryVoltages`, which derives the bar fills and summed pack voltage.

| field | type | meaning / units | range |
|---|---|---|---|
| `role` | `str` | `FRONT`/`LEFT`/`RIGHT`/`UNKWN` | ≤5 glyphs |
| `legs` | `[(y,h,k)]×6` | 6 legs `[FL,FR,ML,MR,RL,RR]`, each (yaw, hip, knee) | `hold`/`extend`/`retract`/`disc`/`unverified` |
| `battery_volts` | `(float,float)` | battery A/B voltages | volts; negative means unavailable |
| `front`,`left`,`right` | `bool` | board present/detected (v0.2 groups by side) | — |
| `roll`,`pitch` | `int` | degrees from IMU | **clamped ±99** for display |
| `imu_valid` | `bool` | IMU measurement succeeded | — |

## 3. Primitive semantics that constrain the design

Library behaviours the design must respect, all replicated exactly in
`ssd1306.py`.

- **`pixel(x,y)`** clips at `x≥128 || y≥64` (and, in the sim, `x<0 || y<0`). On
  AVR the args are `uint8_t`, so a negative coordinate must never reach the API
  (§4.2).
- **`line()`** is the library's steep-swap Bresenham (`err = dx/2`). Horizontal,
  vertical, and exact-45° lines are unambiguous; **other slopes depend on this
  exact tie-breaking**, so the sim ports the algorithm verbatim rather than
  approximating.
- **`rectangle(x,y,w,h)` — the sharp edge.** The library draws top+bottom always,
  but the **vertical side walls only when height ≥ 4** (`y1-y0 ≥ 3`); `w≤1 || h≤1`
  degenerates to a line. **A closed outline ≤3 px tall is impossible via
  `rectangle()`** — it renders open-ended. Consequences:
  - Body (32×31) and the battery bars (18×7) are ≥4 px tall, so both get closed
    outlines. Fletcher found an earlier 3 px rail unreadable for this reason.
  - Eyes (3×3): closed hollow boxes drawn as **4 explicit `line()` walls**, not
    via `rectangle()`. This is the canonical pattern for any small closed box.
- **`rectangleFill(x,y,w,h)`** fills `w×h` pixels inclusive (`x..x+w-1`,
  `y..y+h-1`).
- **`text(x,y,s)`** uses `QW_FONT_5X7`: 6 px advance (5 wide + 1), origin
  top-left, clips at the right edge. Budget ≈ **21 glyphs / 128 px** per line.
  Despite the name the font is **5×8** and its blit can light an eighth row, so
  clear a text field at `SSD1306_TEXT_HEIGHT`, not the 7 px the glyphs look.

### Battery gauges (AC 2b, extended)

Two cells in the strip between the header rule (y=9) and the body (y=22). Each
is `"A"` + the bar + its voltage, right-aligned in 5 characters:

```
  A[############        ] 13.0V   B[##########          ] 12.8V
  ^cellX                                ^cellX + CELL_PITCH
```

The bar alone said neither which battery it was nor how full it was as a number,
which is what you need to read the two against each other. Geometry is in
`display_constants.h` (`SSD1306_BATTERY_*`), with `static_assert`s pinning that
the cells do not overlap each other, the header rule, or the body.

Placement is off the body rather than AC-2b's "two stacked bars on the rear" —
an intentional, Fletcher-approved deviation, since legible cells beat
rear-stacked ones on a 1-bit 128×64 panel.

## 4. Firmware limits (why the code looks the way it does)

1. **No `%f`.** AVR `snprintf` drops float support unless you link
   `-lprintf_flt` (flash cost). Pack voltage is formatted from **integer
   decivolts** as two `%d` fields (`dv/10`, `dv%10`).
2. **Signed coordinate math.** Leg/glyph offsets go negative before clipping.
   The renderer computes in signed `int` and each canvas narrows to `uint8_t` at
   the boundary, where a negative wraps to ~200 and streaks across the panel.
   All three canvases narrow **identically**, so unlike the old Python render,
   the sim now reproduces that wrap instead of hiding it behind unbounded ints.
3. **Explicit flush.** Nothing appears until `display()`. That call lives in
   `Ssd1306Adapter`, not the renderer, because it is the part that costs I2C
   time — and it is wrapped in the 400 kHz clock change.
4. **Draw mode = copy (default).** The render assumes set-pixel semantics. Do
   **not** switch to XOR mode — glyphs drawn over a filled body region would
   invert instead of set.
5. **Font in PROGMEM.** The library keeps font tables in flash; use
   `QW_FONT_5X7`. A different font = different glyph bytes = sim mismatch.
6. **Update timing.** Full frame ≈ **120 ms**; a single dirty 8 px band ≈
   **5.8 ms**, measured on the panel (recorded at `SSD1306_BAND_HEIGHT` in
   `display_constants.h`). This is why `render()` diffs against the previous
   model and redraws only changed elements — a redundant full redraw is a missed
   telemetry slot, not just wasted work.

## 5. Trace format

`oled_trace` reads `key=value` lines from stdin, one frame per blank-line-
separated block, and writes a `frame` marker followed by that frame's calls.
Coordinates are already narrowed to `uint8_t`.

```
frame          erase                    rect  <x> <y> <w> <h>
font <name>    pixel <x> <y>            fill  <x> <y> <w> <h> <color>
               line  <x0> <y0> <x1> <y1>  text  <x> <y> <string to end of line>
```

Successive frames go through one renderer, so frame N+1 carries only what
changed and `krab.render_sequence()` replays it onto frame N's pixels. An
element that fails to erase its old self shows up in the sim for the same reason
it would show up on hardware.

## 6. Verification

- `tests/native/firmware/arduino/src/display/test_display_renderer.cpp` —
  draw-call assertions over the real renderer: arrow orientation, each leg
  joint's clear span, per-element dirty redraw, gauge labels and voltages.
  Run: `make -C firmware test-native`.
- `tests/unit/oled_sim/` — `ssd1306.py`'s primitives against the library's
  semantics. Run: `python -m pytest tests/unit/oled_sim/ -q`. Skipped
  automatically when the SparkFun font headers cannot be found.
- Text fidelity confirmed on the physical panel ("KRABBY").
- Live preview: `firmware/oled_sim/serve.py` → http://127.0.0.1:8080. The page
  has one screen and live controls for boards, joints, IMU, and power. It reloads
  after source edits and rebuilds `oled_trace` when needed.

## 7. Battery voltage and fill

The simulator accepts battery A/B voltages. The native display model sums them
for the pack voltage and derives each bar's fill linearly from 12.0 V (empty) to
13.4 V (full), clamped to that range. This is a coarse resting-voltage gauge,
not a state-of-charge estimate.

Unavailable readings show `--.-V` with an empty bar. The pack voltage is unavailable
if either battery reading is unavailable. Firmware without power-monitor inputs
leaves all voltage readings unavailable.
