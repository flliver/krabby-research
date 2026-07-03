"""Launch the Krabby firmware test GUI.

Local board:         python -m firmware.gui [--port COM5]
Remote board (ssh):  python -m firmware.gui --remote krabby-orin [--serial /dev/ttyACM0]

--remote replaces the manual two-terminal bridge dance: it ssh-launches
firmware/tools/serial_tcp_bridge.py on the host, tunnels a local port to it, and
tears the whole thing down when the GUI closes — freeing the remote serial port
for flash-remote, with no bridge process to hunt down and kill.
"""
import argparse
import sys

from firmware.gui.remote import (
    DEFAULT_BRIDGE_PORT,
    DEFAULT_REMOTE_DIR,
    RemoteBridge,
    RemoteBridgeError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Krabby MCU test GUI")
    parser.add_argument("--port", default=None,
                        help="Serial port override (e.g. COM5 or socket://host:5331)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--remote", metavar="HOST", default=None,
                        help="ssh host the MCU is attached to; auto-starts the serial/TCP "
                             "bridge there and connects through a tunnel")
    parser.add_argument("--serial", default="/dev/ttyACM0",
                        help="serial device on the remote host (only with --remote)")
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT,
                        help="TCP port the remote bridge listens on (only with --remote)")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR,
                        help="firmware checkout on the remote host (only with --remote)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.remote and args.port:
        parser.error("--remote and --port are mutually exclusive (--remote picks the port itself)")

    # Imported here so parser-only paths (--help, argument errors, unit tests)
    # don't require tkinter/a display.
    from firmware.gui.app import KrabbyTestGUI

    bridge = None
    port = args.port
    if args.remote:
        bridge = RemoteBridge(args.remote, serial_dev=args.serial,
                              bridge_port=args.bridge_port, remote_dir=args.remote_dir)
        try:
            port = bridge.start()
        except RemoteBridgeError as e:
            sys.exit(f"error: {e}")

    try:
        app = KrabbyTestGUI(port=port, baud=args.baud)
        app.mainloop()
    finally:
        if bridge:
            bridge.stop()


if __name__ == "__main__":
    main()
