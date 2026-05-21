"""Launch the Krabby firmware test GUI: python -m firmware.gui [--port COM5]"""
import sys
import argparse

from firmware.gui.app import KrabbyTestGUI


def main():
    parser = argparse.ArgumentParser(description="Krabby MCU test GUI")
    parser.add_argument("--port", default=None, help="Serial port override")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args()

    app = KrabbyTestGUI(port=args.port, baud=args.baud)
    app.mainloop()


if __name__ == "__main__":
    main()
