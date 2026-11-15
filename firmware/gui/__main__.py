"""Launch the Krabby firmware test GUI: python -m firmware.gui [--device PATH] [--debug]"""
from __future__ import annotations

import argparse
import logging
import sys

from firmware.gui.app import KrabbyTestGUI
from firmware.krabby_mcu import DEFAULT_BAUD, logger
from firmware.mcu_port import McuSerialDevice, McuSerialTarget, McuSerialUrl


def build_parser() -> argparse.ArgumentParser:
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
    dest.add_argument(
        "--stm",
        action="store_true",
        help="Use STM leg boards over CAN (KrabbyMCUCANSDK) instead of Mega serial",
    )
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="With --stm: connect to virtual CAN (vcan0) instead of can0+can1",
    )
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log every command sent to the MCU",
    )
    return parser


def parse_gui_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.virtual and not args.stm:
        parser.error("--virtual requires --stm")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_gui_args(argv)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.stm:
        app = KrabbyTestGUI(stm=True, virtual=args.virtual)
        app.mainloop()
        return

    target: McuSerialTarget | None = None
    if args.device is not None:
        target = McuSerialDevice(path=args.device)
    elif args.serial_url is not None:
        target = McuSerialUrl(url=args.serial_url)

    app = KrabbyTestGUI(target=target, baud=args.baud)
    app.mainloop()


if __name__ == "__main__":
    main()