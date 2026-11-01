"""Launch the Krabby firmware test GUI: python -m firmware.gui [--device PATH] [--debug]"""
import sys
import argparse
import logging

from firmware.gui.app import KrabbyTestGUI
from firmware.krabby_mcu import DEFAULT_BAUD, logger
from firmware.mcu_port import McuSerialDevice, McuSerialTarget, McuSerialUrl


def main():
    parser = argparse.ArgumentParser(
        description="Krabby MCU test GUI",
        epilog=(
            "Simulated MCU + Isaac current-sense demo: see "
            "firmware/gui/CURRENT_SENSE_SIM_DEMO.md"
        ),
    )
    dest = parser.add_mutually_exclusive_group()
    dest.add_argument(
        "--device",
        default=None,
        help="Serial device path (COM port, /dev/ttyACM*, PTY slave)",
    )
    dest.add_argument(
        "--serial-url",
        default=None,
        help="Pyserial URL (e.g. socket://127.0.0.1:54321 for Windows sim TCP)",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--debug", action="store_true", help="Log every command sent to the MCU")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    target: McuSerialTarget | None = None
    if args.device is not None:
        target = McuSerialDevice(path=args.device)
    elif args.serial_url is not None:
        target = McuSerialUrl(url=args.serial_url)

    app = KrabbyTestGUI(target=target, baud=args.baud)
    app.mainloop()


if __name__ == "__main__":
    main()
