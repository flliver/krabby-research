MilePebble1
Proccessor is up and communicating, USB serial, Diagnostic LED controls, and CAN data is flowing

CAN is of particualr interest. Codex in VScode was able to write and execute python code to control the CANable USB adaptor and the serial port the Krabby board is on - fully automated the test loop with 1400 frames of testing and zero errors at 1MHz / 5MHz CANFD rates.

One unclear hardware issue. 
U1 is a LM5069MM-2 Positive High-Voltage Hot Swap and In-Rush Current Controller with Power Limiting
For reasons I've yet to uncover, it initially refused to turn the power on.
Worse, it allowed enough leakage to support the 3.3v rail (possibly including backfeed by STLink and / or USB)
which let the processor run, LEDs flash and USB send data, but did not support the 5V rail and lead me on a 
wild goose chase to get the CAN peripheral running.

Based on the resistor values in the UVLO sense circuit, it should turn on at about 16V. My bench supply maxes at about 27V and it still wouldn't turn on. Over the course of a couple hours of measuring, and reading datasheets, some step triggered it to on. and it now behaves as expected returning after multiple power cycle events.

No clear answer as to why.
Also, no clear answer to why this input protection was included in this project.



MilePebble2
CAN bootloader has been implemented and tested.
The Python tool updates a Krabby application over CAN using the CANable2 adapter. It verifies the complete image by CAN readback, then starts the application.


In this test and instructions, the board is plugged into serial (com3)- not neccesary for bootloader
The CANable interface has shown up in windows as COM7.
Instructions: 

Before uploading:

1. Power the Krabby board, including its CAN transceiver’s 5 V rail.
2. Connect CANH, CANL, and ground, with the tested termination arrangement.
3. Close the COM3 serial terminal and release COM7 from other programs.
4. Open PowerShell in `Firmware\tools`.

Run:

```powershell
python .\openblt_canable.py PATH_TO_APPLICATION.srec --port COM7 --node 0
```

The tool configures CANable2 for **1 Mbps nominal / 5 Mbps data** automatically. COM7 and node 0 are the defaults.

Load one of the demonstration applications:

```powershell
# Application 1: M1 left LED
python .\openblt_canable.py ..\bench\openblt\led_apps\app1_m1_left.srec --node 0

# Application 2: M2 left LED
python .\openblt_canable.py ..\bench\openblt\led_apps\app2_m2_left.srec --node 0

# Application 3: M3 left LED
python .\openblt_canable.py ..\bench\openblt\led_apps\app3_m3_left.srec --node 0
```

To upload the normal build:

```powershell
python .\openblt_canable.py ..\build\Debug\krabby_v03_bringup.srec --node 0
```

Success produces:

```text
PASS: ... application bytes read back over CAN, including OpenBLT signature
```

Afterwards, **reopen COM3** and send `hello` followed by Enter to identify the running application. Reopening is necessary because USB disconnects during bootloading.

Optional arguments:

| Argument | Purpose |
|---|---|
| `--port COM7` | Select the CANable serial port |
| `--node 0` | Select the board’s switch-defined node ID |
| `--stay` | Verify the upload but remain in OpenBLT |

If an update is interrupted, rerun the same upload command. An invalid application leaves OpenBLT available for recovery. If a valid application is unresponsive, reset the board and begin connecting within its two-second bootloader entry window.

Use application images built for **`0x08003000`**. This command replaces the application; it does not replace the protected bootloader.



###############################################
MilePebble3
This is the result of a fully integrated hardware+AI loop, a full set of instructions and a couple hours of AI time.
Specifically, this is the save before further testing.

