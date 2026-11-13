STM32CubeProgrammer is Windows software you'll need to download and install. This process only requires a USB connection to the board. You don't need an ST-Link hardware programmer or a CAN interface.

The firmware is intended to communicate over CAN, but nearly all of the commands are also available over USB, which makes it an easy place to start testing.

This procedure is an easy way to bring up a blank board for initial USB testing before a CAN bus is available. It installs the CAN bootloader and full application. Once the system is running with the full CAN bus connected, firmware updates use a different, easier process through the installed CAN bootloader. Those updates do not require physical access to the board or the BOOT/RESET button sequence described below.

To program the board:

1. Power the board normally.
2. Hold the BOOT button (the one closer to the edge), press and release RESET, then release BOOT.
3. In DFU mode, Windows Device Manager shows the board under **Universal Serial Bus devices > STM32 BootLoader**, not as a COM port. Open STM32CubeProgrammer and select USB as the connection type. It should find the board as USB1. Click Connect. This can be a little confusing: most documentation calls this DFU mode, but STM32CubeProgrammer calls it USB and enumerates devices as USB1, USB2, etc. The separate UART option lists Windows COM ports (COM1, COM2, etc.). You want USB, not UART.
4. Open **Option bytes** (box with "OB"), expand **User Configuration**, and find **DBANK**. Clear its checkbox so **DBANK=0** (single-bank mode), then click **Apply**. If it is already 0, leave it unchanged. Leave all other option bytes unchanged.
5. After Apply, reset the target and re-enter DFU using the BOOT/RESET sequence in step 2, then reconnect using USB. I've found this transition buggy: it has sometimes taken a couple of resets and attempts to re-enter DFU before normal programming would work. Refresh the USB device list as needed. Once reconnected, read the option bytes and confirm **DBANK=0** before continuing.
6. Open **Erasing & Programming** (box with downward pointing arrow) and select [FirstForUSBBootload.hex](FirstForUSBBootload.hex), in the same folder as these instructions:
   `hardware/PCB-v0.3/Firmware/Instructions/FirstForUSBBootload.hex`
7. Click **Start Programming**. Once programming completes successfully, release BOOT and press RESET.

After reset, allow approximately two seconds for the bootloader. The top-left LED should then blink at approximately 2 Hz. The **STM32 BootLoader** device is replaced by the application's USB serial device, which Windows Device Manager lists under **Ports (COM & LPT)** with an assigned COM port. Use that COM port for the GUI below.

Next, from hardware/PCB-v0.3/Firmware/tools, run:

python krabby_position_gui.py

In the GUI, select USB CDC, choose the COM port Windows assigned, and click Connect. The Node setting doesn't matter in USB mode.

You should see the sensor data populate and update. The attached screenshot shows what it looks like on my board.

You don't need motors connected for this. Some measurements will be zero or meaningless without the rest of the hardware, but the ADC readings, temperature, USB voltage, and other telemetry should be available.

I've tested this on my board, including the CAN bootloader and application updates. There's still plenty to do, but getting the firmware running and communicating on another board will be a significant milestone.
