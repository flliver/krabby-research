# Krabby live debug viewer

From Firmware/tools:

```powershell
# CANable2 COM7, node 0 (leave COM3 available for a separate terminal).
python .\krabby_debug_gui.py

# Direct STM32 USB CDC COM3; close any terminal using COM3 first.
python .\krabby_debug_gui.py --usb --port COM3

# Another physical CAN node:
python .\krabby_debug_gui.py --port COM7 --node 2
```

Requires Python 3 with tkinter (normally included in Windows Python) and the existing
pyserial dependency. No firmware change or extra GUI package is required.

Choose interface/port/CAN node and polling interval, then Connect. Connection settings
are captured at Connect; disconnect before applying changed settings. The CAN node
selector applies only to CANable; USB directly reads its attached physical board.
Release the selected port from other programs. Default polling is 200 ms (5 Hz).

Live sensors shows all nine measurements for each motor, with ADC pin labels,
plus all nine board measurements. ADC counts are 0..4095; voltage/current conversions
are nominal. Reported position follows the selected feedback/reporting configuration.
Temperature shows unavailable until valid NTC coefficients are configured.

Configuration pages shows every field in the active RAM node page and all three
motor pages, including reserved fields. Names, units, defaults and bounds come from
../docs/config_schema.json. Values refresh every 2 seconds; saved flash can differ.

The viewer sends only VERSION, STATUS, SENSORS and GET. It cannot arm, command motors,
edit/save configuration, reboot, or refresh the motor watchdog. Do not expect it to
keep an already armed node alive. CANable connection configures its serial/CAN channel
at the established 1 Mbps / 5 Mbps; disconnect closes that adapter channel.

Communication runs on one worker thread; serial requests never block Tk event handling.
On timeout/reset/update, the viewer disconnects and marks retained values stale.
Reconnect after the port returns. A closed window waits for its worker to release
the serial port. It does not silently reconnect after errors.

Validation: Tk widget/render smoke test passed. The actual GUI worker rendered five
live snapshots and all four configuration pages on both physical COM7/CAN and
COM3/USB, then cleanly released both ports. Evidence:
../bench/application/gui_test_results.json. The test windows were hidden; interactive
window sizing/usability still benefits from user review. No target state was modified.
