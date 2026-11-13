# LED update applications

Built 2026-11-11. Not flashed or physically verified in this change.
All retain CAN FD/USB bridge, OpenBLT re-entry, and motor sleeps LOW.
Rebuilt with the 112 KiB application limit; 0x0801F000..0x0801FFFF is reserved
for persistent configuration and is excluded from application outputs.

| Image | Active-high LED | Toggle interval | Full blink period |
|---|---|---|---|
| app1_m1_left.srec | M1 left PC2 | 250 ms | 500 ms |
| app2_m2_left.srec | M2 left PA3 | 500 ms | 1 s |
| app3_m3_left.srec | M3 left PB6 | 750 ms | 1.5 s |

USB command `hello` reports APP-1/2/3, LED pin, and toggle interval.
The status LED remains off unless explicitly controlled by `led on`.
`led on/off` retains its existing status-LED control and suspends automatic blinking;
reset/reload the application to resume blinking.

From Firmware/tools, run one:

```powershell
python .\openblt_canable.py ..\bench\openblt\led_apps\app1_m1_left.srec --node 0
python .\openblt_canable.py ..\bench\openblt\led_apps\app2_m2_left.srec --node 0
python .\openblt_canable.py ..\bench\openblt\led_apps\app3_m3_left.srec --node 0
```

Rebuild with CMake APP_VARIANT=1/2/3 and APP_BLINK_MS=250/500/750 respectively,
in separate build directories LED1/LED2/LED3. Default APP_VARIANT=0 retains status LED.
