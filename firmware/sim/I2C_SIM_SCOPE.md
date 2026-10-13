# I2C simulation scope (shipping firmware)

Host I2C simulation must mirror **what the host-built firmware actually probes**.
This file is the inventory and blocker list before implementing a host `Wire` shim.

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
`viewer.py`, `RENDER_SPEC.md`). It renders from the same layout constants as
`display_constants.h` but is **not yet** connected to a host `Wire` backend.

---

## Host sim today (serial + Isaac paths)

| Component | State |
|-----------|--------|
| `Wire` / addressed slave model | **`sim_i2c` + host `Wire.h`** (probe + register read stub) |
| `krabby_host` firmware loop | Leader **`;IMU`** via `host_leader_imu` + `sim_imu_model` |
| `host_multi_board.cpp` | FRONT line appends **`;IMU`**; followers unchanged |
| `firmware/oled_sim` | Standalone / browser viewer; **no** firmware-in-the-loop I2C |

---

## Blockers before full I2C sensor sim

1. ~~**Host `Wire` shim**~~ - done: `firmware/sim/host/sim_i2c.*`, `Wire.h`; tests in
   `test_host_i2c.cpp`.
2. **Link IMU + display into host build** - compile `Lsm6dsoAdapter` / `Ssd1306Adapter`
   (or targeted stubs) under `KRABBY_HOST_FIRMWARE=1`, or split telemetry append without
   full driver until drivers compile on host.
3. **Leader telemetry parity** - call `appendImuMeasurement()` on the host USB path when
   role is FRONT (match `arduino.ino` loop).
4. ~~**INA228 + `;BATT` on host**~~ - `sim_battery_model`, `host_battery_telemetry`,
   default ACK at `0x40`/`0x41`. Mega firmware driver still TBD.
5. **OLED E2E** - route sim `Wire` writes to `oled_sim` (or trace buffer) so
   `oledRenderer.render()` drives a visible krab screen in sim.

---

## Verification

```bash
pytest tests/unit/firmware/test_i2c_shipping_scope.py -v
```

After the `Wire` shim exists, extend tests to probe each **shipping** address through the
sim API; ignore `0x40`/`0x41` until firmware references them.
