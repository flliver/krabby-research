# I2C simulation scope (shipping firmware)

Host I2C simulation must mirror **what the host-built firmware actually probes**.
This file is the inventory of bus, devices, telemetry, and host behavior.

**Authoritative code:** `firmware/arduino/arduino.ino`, `src/imu/`, `src/display/`,
`src/i2c/`. Machine-readable summary: [`i2c_shipping_scope.py`](i2c_shipping_scope.py).

**Operator wiring:** [`firmware/SETUP.md`](../SETUP.md) (I2C Sensor Cluster / LSM6DSO).

---

## Bus (leader FRONT / UNKWN only)

| Parameter | Shipping value | Source |
|-----------|----------------|--------|
| SDA / SCL | Mega **D20** / **D21** | Hardware I2C |
| Clock | **100 kHz** | `I2C_DEFAULT_BUS_CLOCK_HZ` in `imu_constants.h` |
| Timeout | 10 ms (`Wire.setWireTimeout`) | `I2C_BUS_TIMEOUT_MICROSECONDS` |
| Followers | **No I2C** | `setup()` runs IMU/OLED only when `ROLE_FRONT` or `ROLE_UNKNOWN` |

Host role election treats FRONT as IMU-capable via `hostBoardHasFrontImu()` (see
[`ROLE_BOOT.md`](ROLE_BOOT.md)).

---

## Devices (planned vs shipping)

| Device | Common design reference | **Shipping firmware** | Sim status |
|--------|-------------------------|------------------------|------------|
| IMU | Some docs describe **BMI270** `0x68` / `0x69` | SparkFun **LSM6DSO** **`0x6B` / `0x6A`** | **Simulate LSM6DSO** register map (`SparkFunLSM6DSO`) |
| OLED | Qwiic 1.3" SSD1306 **`0x3D`** | Same address; **`Ssd1306Adapter`** + `SparkFun_Qwiic_OLED` | **Simulate SSD1306** I2C stream; pixel preview in [`firmware/oled_sim/`](../oled_sim/) |
| Pack monitor | INA228 **`0x40`** | **Not on Mega firmware** | **Host sim** ACK + `;BATT` from `sim_battery_model` |
| Midpoint INA228 | **`0x41`** | **Not on Mega firmware** | **Host sim** ACK; midpoint V in `;BATT` |

There is **no** central `sensors_config.h` in the tree; constants live in `imu_constants.h`,
`display_constants.h`, and adapter headers.

---

## Telemetry segments (host / SDK)

| Segment | On wire (leader line) | Firmware | Python parser |
|---------|----------------------|----------|---------------|
| Joints | `FRONT;` + 6 nine-token segments | `actuator_manager.h` | `joint_telemetry.py` |
| IMU | `;IMU ax ay az gx gy gz temp valid` | `appendImuMeasurement()` in `telemetry.h` | `imu_telemetry.py`, `TelemetryFrame` |
| BATT | `;BATT ...` (documented pack schema) | **Absent** | **`battery_telemetry.py` + host append** |

IMU units on the wire: accel **m/s^2**, gyro **rad/s**, temp **deg C**, `valid` **0|1**.
When init fails, firmware still appends IMU with zeros and `valid=0` (does not stall the loop).

---

## OLED simulator

The health-screen preview lives under **`firmware/oled_sim/`** (`ssd1306.py`, `krab.py`,
`viewer.py`, `RENDER_SPEC.md`). Browser replay is optional; the in-loop host path is
below.

### Host OLED (no Arduino sensor libraries)

Host sim **on purpose** does **not** link **`SparkFun_Qwiic_OLED`**, **`SparkFunLSM6DSO`**
(Adafruit/SparkFun Arduino packages), or **`Ssd1306Adapter`** / **`Lsm6dsoAdapter`** into
`krabby_host`. Those libraries are not vendored in this repo and would pull a large
Arduino-only dependency graph into native CI.

Instead the host path reuses shipping **`buildDisplayFrame`** + **`DisplayRenderer`**
(`host_display.cpp`), flushes a GDDRAM image over **`Wire`** via **`HostOledCanvas`**, and
captures bytes in **`sim_ssd1306`**. On-device firmware still uses the adapters + libraries;
host sim matches **layout and live state**, not byte-identical driver I2C init sequences.

### Non-fatal sensor init (host)

Leader **`setup()`** always completes: **`leaderImu`**, **`leaderBattery`**, and **`hostDisplaySetup`**
probe **`Wire`** but never abort **`krabby_host_firmware_setup`**. Missing devices yield
**`;IMU … valid=0`**, omit **`;BATT`**, or skip OLED draws; actuator USB commands and
telemetry joints keep running.

---

## Host sim today (serial + Isaac paths)

| Component | State |
|-----------|--------|
| `Wire` / addressed slave model | **`sim_i2c` + host `Wire.h`** (probe + register read stub) |
| `krabby_host` firmware loop | Leader **`;IMU`** via `host_leader_imu` + `sim_imu_model` |
| `host_multi_board.cpp` | FRONT line appends **`;IMU`**; followers unchanged |
| `firmware/oled_sim` | Browser viewer; host **`DisplayRenderer` + `sim_ssd1306` Wire** path |

---

## Remaining

- **INA228 on Mega firmware** — host sim ACKs `0x40`/`0x41` and appends **`;BATT`**; the sketch has no pack driver yet.
- **Optional: vendored SparkFun/Adafruit libs on host** — only for adapter-level I2C parity; the host path uses register models + **`DisplayRenderer`** instead (see Host OLED above).

---

## Verification

```bash
pytest tests/unit/firmware/test_i2c_shipping_scope.py -v
```

Keeps Python inventory aligned with `imu_constants.h` / `display_constants.h`. Host `Wire`
addresses include sim INA228 at `0x40`/`0x41` even though Mega sketch does not probe them yet.
