# Motor command LED update ? 2026-11-12

Offline build only. No serial port, ST-Link, or physical hardware was accessed.
Load krabby_motor_leds.srec with the existing CAN/OpenBLT tool.

Each motor: negative applied drive lights LEFT; positive applied drive lights RIGHT.
Zero/inhibited drive, wake/reversal neutral delays, ARM neutral, DISARM, watchdog,
and fatal/reboot paths turn the motor indications off. These indicate processor
output commands, not measured mechanical movement. Brightness is not duty-scaled.
The status heartbeat is unchanged.

Only application source App/Src/app_service.c changed. LED GPIO mappings and output
initialization already existed; no IOC change/regeneration. No bootloader, protocol,
configuration schema, defaults, or flash partition changes. Existing saved configuration
is retained by ordinary OpenBLT application updates. Old known-good images preserved.

Build: 84,452 flash bytes / 112 KiB, RAM allocation 23,456 bytes / 128 KiB.
Offline S-record address/vector checks passed. Physical LED behavior is unverified.

From Firmware/tools:

```powershell
python .\openblt_canable.py ..\bench\motor_leds\krabby_motor_leds.srec --port COM7 --node 7
```

Or select this SREC in the GUI Firmware update tab, using the actual board address.
