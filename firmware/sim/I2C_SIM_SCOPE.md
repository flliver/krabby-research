# I2C simulation scope (shipping firmware)

Host I2C simulation must mirror **what the host-built firmware actually probes**.
This file is the inventory of bus, devices, telemetry, and host behavior.

**Authoritative code:** `firmware/arduino/arduino.ino`, `src/imu/`, `src/display/`,
`src/i2c/`, and `src/power_monitor/`. Machine-readable summary: [`i2c_shipping_scope.py`](i2c_shipping_scope.py).

**Operator wiring:** [`firmware/SETUP.md`](../SETUP.md) (I2C Sensor Cluster / LSM6DSO).

---

## Bus (leader FRONT / UNKWN only)

| Parameter | Shipping value | Source |
|-----------|----------------|--------|
| SDA / SCL | Mega **D20** / **D21** | Hardware I2C |
| Clock | **100 kHz** | `I2C_DEFAULT_BUS_CLOCK_HZ` in `imu_constants.h` |
| Timeout | 10 ms (`Wire.setWireTimeout`) | `I2C_BUS_TIMEOUT_MICROSECONDS` |
| Followers | **No I2C** | `setup()` runs IMU/OLED/INA228 only when `ROLE_FRONT` or `ROLE_UNKNOWN` |

Host role election treats FRONT as IMU-capable via `hostBoardHasFrontImu()` (see
[`ROLE_BOOT.md`](ROLE_BOOT.md)).

---

## Devices on the leader bus

| Device | I2C address | Firmware | Host sim `Wire` |
|--------|-------------|----------|-----------------|
| IMU (SparkFun LSM6DSO) | **`0x6B`** / **`0x6A`** (ADR jumper) | `lsm6dso_adapter.h`, `hasFrontImu()` | LSM6DSO register model; burst from **`0x20`** |
| OLED (SSD1306) | **`0x3D`** / **`0x3C`** (jumper) | `Ssd1306Adapter`, `SparkFun_Qwiic_OLED` | SSD1306 command/data stream; preview in [`firmware/oled_sim/`](../oled_sim/) |
| Pack INA228 | **`0x40`** | `Ina228Adapter` + real SparkFun INA2XX driver, external shunt | Shared register-level INA228 model |
| Midpoint INA228 | **`0x41`** | Independent `Ina228Adapter`, voltage only | Separate instance of the same register model |

Unlisted addresses (for example **`0x68`**) are not on the leader bus inventory and
**NACK** in host sim (`test_host_i2c.cpp`). Full list: [`i2c_shipping_scope.py`](i2c_shipping_scope.py).

There is **no** central `sensors_config.h` in the tree; constants live in `imu_constants.h`,
`display_constants.h`, and adapter headers.

---

## Telemetry segments (host / SDK)

| Segment | On wire (leader line) | Firmware | Python parser |
|---------|----------------------|----------|---------------|
| Joints | `FRONT;` + 6 nine-token segments | `actuator_manager.h` | `joint_telemetry.py` |
| IMU | `;IMU ax ay az gx gy gz temp valid` | `appendImuMeasurement()` in `telemetry.h` | `imu_telemetry.py`, `TelemetryFrame` |
| BATT | `;BATT pack_v pack_i pack_w charge batt_a batt_b div region pack_valid midpoint_valid` | Task 3 **`appendBatteryTelemetry`** serializer | `BatteryTelemetry` + `TelemetryFrame` |

IMU units on the wire: accel **m/s^2**, gyro **rad/s**, temp **deg C**, `valid` **0|1**.
When init fails, firmware still appends IMU with zeros and `valid=0` (does not stall the loop).

---

## INA228 runtime path

The runtime uses the same `Ina228Adapter`, pinned SparkFun INA2XX driver, and
register-level device model as the task 3 native tests:

```text
physical pack/midpoint inputs → simulated INA228 registers
    ↕ runtime Wire bus
real SparkFun driver → Ina228Adapter → PowerCalibration
    → latest measurements → BATT serializer and OLED frame
```

`devices/ina228.{h,cpp}` is shared by the runtime and native-test I2C buses.
It models identity, configuration, shunt calibration, register quantization,
signed current/charge, unsigned power, and accumulator resets. Native tests can
inject register NACKs through that same model. Runtime operation logs are off by
default so long-running simulation does not accumulate diagnostic history.

The leader reads both monitors once per 50 ms telemetry tick; the OLED uses the
cached measurements. A missing or failing monitor reports its own invalid flag,
while the other monitor continues. Reconnection follows the production driver's
retry policy. Calibration is loaded from the simulated FRONT EEPROM, and the
production driver resets the pack charge accumulator on startup.

Set physical inputs from the Python host bridge after firmware setup:

```python
host.set_battery_inputs(
    pack_volts=26.25, pack_amps=-3.125,
    battery_a_volts=12.75, charge_coulombs=-1000.0,
)
host.i2c_set_device_present(0x41, False)  # disconnect only the midpoint sensor
```

These are controlled electrical inputs, not a battery discharge/sag model.
The runtime does not yet execute task 4's power-safety state machine or expose
its calibration commands. Runtime BATT uses task 3's validity-aware format and
NORMAL region; hardware task 4 adds its lifecycle field. The Python parser still
accepts historical telemetry from older firmware.

---

## OLED simulator

The health-screen preview lives under **`firmware/oled_sim/`** (`ssd1306.py`, `krab.py`,
`viewer.py`, `RENDER_SPEC.md`). Browser replay is optional; the in-loop host path is
below.

### Host OLED (separate from the real INA228 driver)

Host sim **on purpose** does **not** link **`SparkFun_Qwiic_OLED`**, **`SparkFunLSM6DSO`**
or **`Ssd1306Adapter`** / **`Lsm6dsoAdapter`** into
`krabby_host`. This scope is unchanged for IMU/OLED. INA228 uses its real pinned SparkFun library,
compiled against host Arduino/Wire shims.

Instead the host path reuses shipping **`buildDisplayFrame`** + **`DisplayRenderer`**
(`host_display.cpp`), flushes a GDDRAM image over **`Wire`** via **`HostOledCanvas`**, and
captures bytes in **`sim_ssd1306`**. On-device firmware still uses the adapters + libraries;
host sim matches **layout and live state**, not byte-identical driver I2C init sequences.

### Non-fatal sensor init (host)

Leader **`setup()`** always completes: **`leaderImu`**, **`leaderBattery`**, and **`hostDisplaySetup`**
probe **`Wire`** but never abort **`krabby_host_firmware_setup`**. Missing devices yield
**`;IMU … valid=0`**, per-monitor invalid flags in **`;BATT`**, or skipped OLED draws; actuator USB commands and
telemetry joints keep running.

---

## Host sim today (serial + Isaac paths)

| Component | State |
|-----------|--------|
| `Wire` / addressed slave model | **`sim_i2c` + host `Wire.h`** (INA228 register reads/writes, IMU registers, OLED commands) |
| `krabby_host` firmware loop | Leader **`;IMU`** via `host_leader_imu` + `sim_imu_model` |
| `host_multi_board.cpp` | FRONT line appends **`;IMU`**; followers unchanged |
| `firmware/oled_sim` | Browser viewer; host **`DisplayRenderer` + `sim_ssd1306` Wire** path |

Three-board health screen (controller thirds + missing follower):  
`tests/unit/firmware/test_host_oled_health_screen_three_board.py`.

---

## Verification

```bash
pytest tests/unit/firmware/test_i2c_shipping_scope.py -v
```

Keeps Python inventory aligned with `imu_constants.h`, `display_constants.h`, and
`power_monitor_constants.h`. Host `Wire` uses the same `0x40`/`0x41` addresses as shipping firmware.
