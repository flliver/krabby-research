"""Launch the Krabby firmware test GUI: python -m firmware.gui [--port COM5] [--debug]"""
import sys
import argparse
import logging

from firmware.gui.app import KrabbyTestGUI
from firmware.krabby_mcu import DEFAULT_BAUD, logger


def main():
    parser = argparse.ArgumentParser(
        description="Krabby MCU test GUI",
        epilog=(
            "Simulated MCU + Isaac current-sense demo: see "
            "firmware/gui/CURRENT_SENSE_SIM_DEMO.md"
        ),
    )
    parser.add_argument("--port", default=None, help="Serial port override")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--debug", action="store_true", help="Log every command sent to the MCU")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    app = KrabbyTestGUI(port=args.port, baud=args.baud)
    app.mainloop()


if __name__ == "__main__":
    main()
