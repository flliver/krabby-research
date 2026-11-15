# Qwiic pack monitor

## Purpose

Optional INA228 pack / midpoint sensing on **I2C1**.

## Status

**Not in `krabby.dbc`.** Pack telemetry for HAL `BatteryTelemetry` still uses the power-board / Mega path until a CAN message exists.

## Reference

Arduino `power_monitor` semantics (`firmware/arduino/`).