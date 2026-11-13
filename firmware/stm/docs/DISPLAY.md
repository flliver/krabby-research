# Qwiic Krabby health screen (leg board)

## Purpose

Optional **SparkFun Qwiic OLED** (SSD1306) on **one** leg PCB shows the same operator health UI as legacy Mega firmware, driven from **CAN-FD traffic the board receives**, not from a host display service.

## Components

- **Panel:** Qwiic **1.3″ SSD1306** — default I2C **`0x3D`** (jumper **`0x3C`**), same as [`firmware/arduino/src/display/`](../../arduino/src/display/) and [`firmware/oled_sim/`](../../oled_sim/)
- **Bus:** **I2C1** with **DAC7574** (`0x4C`) and optional **Qwiic IMU** (`ROBOT_SENSORS.md`) — Qwiic daisy chain allowed
- **CAN mirror:** `can_handler` decodes in-bound **`LegCommand`**, **`LegTelemetry`**, **`NodeStatus`**, and **`QwiicImu`** into a RAM **bus state table** (all leg indices heard on this segment)
- **UI:** port **`DisplayRenderer`** + **`buildDisplayFrame()`** semantics from Arduino; layout constants in `display_constants.h`

## Critical points

- **Single screen per robot:** at most **one** leg board has a Qwiic OLED installed. Every identical firmware image **probes** for the panel at boot; only the populated board runs the draw loop.
- **Auto-detect:** after DAC7574 init, probe SSD1306 address(es). On ACK: init panel, set **`display_active`**, run periodic refresh. On fail: **no** I2C traffic to the OLED and **no** CPU spent on layout.
- **State source is CAN only (for joints):** the hosting board **does not** drive the full-robot picture from its three local joints alone. In **`can_handler_on_frame()`** (and RX ISR deferral), **parse every** supported extended ID on the wire and update the mirror — same bus segment sees **host + up to three leg nodes** (and all of their telemetry/command traffic). Do **not** add a separate **`DisplayState`** CAN TX; the screen is a **local sink**.
- **Cross-bus limit:** a leg on **`can0`** does **not** hear **`can1`** frames. The UI shows **every node on this physical bus**; legs on the other bus appear **stale / absent** until a future dual-bus feed is defined. Prefer mounting the OLED on a board that matches operator visibility needs (e.g. mid leg on the side that should look “complete”).
- **Refresh rate:** target **~4 Hz** redraw (**250 ms**), matching Mega `OLED_REDRAW_INTERVAL_MILLISECONDS`. Batch I2C transfers after the control loop; a long draw must not block **1 kHz** joint control or FDCAN TX slots.
- **IMU header on panel:** if this board also has a **Qwiic IMU**, use live samples for tilt/roll header. Else if the mirror holds a recent **`QwiicImu`** from another node, optional use — otherwise show **`---`** / omit tilt (do not invent values).
- **Pack / battery bars:** leg PCBs have **no** pack INA228. Omit battery chrome unless a **`BATT`**-equivalent stream later appears on CAN (`ROBOT_SENSORS.md`). Do not show fabricated pack voltage.
- **Glyphs:** map each mirrored joint’s latest **`LegTelemetry`** + command knowledge into **`ActuatorGlyph`** (hold / extend / retract / disconnected) using the same thresholds as `selectActuatorGlyph()` / connection bits in `LEG_BOARD_CONTROL.md`.
- **Failure isolation:** OLED I2C errors are logged/dropped; they **must not** fault the motor path or DAC.

## Reference

### Bring-up checklist (board with screen)

1. Flash identical leg application; set leg ID switch.
2. Plug Qwiic OLED on **I2C1**; confirm **3V3** Qwiic chain with DAC.
3. Connect CAN to a segment with traffic; verify **`CAN_ACTIVITY_LED`** and panel header leg role / joint glyphs track **`krabby-firmware --stm telemetry`** on that bus.
4. Interactive layout preview (host): [`firmware/oled_sim/serve.py`](../../oled_sim/serve.py) with trace input — not required on robot.

### Port map (firmware work)

| Mega / Arduino | Leg STM32 |
|--------------|-----------|
| `buildDisplayFrame()` inputs from 6 local actuators + 3 controller freshness | Mirror built from **decoded CAN** per leg/joint |
| Leader-only OLED in `arduino.ino` | **Any** node with successful SSD1306 probe |
| `SparkFun_Qwiic_OLED` / `Ssd1306Adapter` | STM32 I2C bit-bang or HAL **I2C1** driver wrapping same adapter |

Normative IDs and rates: [`CAN_PROTOCOL.md`](./CAN_PROTOCOL.md). Joint field meanings: [`LEG_BOARD_CONTROL.md`](./LEG_BOARD_CONTROL.md).
