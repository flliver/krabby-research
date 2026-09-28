#!/usr/bin/env python3
"""Print every MCU serial device on this host (one per line), filtered by USB
VID:PID. Used by the Makefile's flash-remote-all target to discover all boards
on a remote without naming each port.

Self-contained: depends only on pyserial, so the Makefile can pipe it to a
remote host over ssh stdin and run it without the rest of the firmware package.
"""
from __future__ import annotations

import sys

# USB VID/PID pairs for boards we drive. Keep in sync with firmware/mcu_port.py
# (MEGA_USB_IDS). Duplicated here so this script stands alone on a remote host.
MEGA_USB_IDS = {
    ("2341", "0042"), ("2341", "0010"), ("2341", "0110"),  # Arduino Mega native USB
    ("1a86", "7523"), ("1a86", "5523"),                    # CH340 / CH341 (Krabby-Uno shield)
}


def main() -> int:
    try:
        from serial.tools import list_ports
    except ImportError:
        print("pyserial is required (pip install pyserial)", file=sys.stderr)
        return 1

    matched = []
    for p in list_ports.comports():
        vid = f"{p.vid:04x}" if p.vid else ""
        pid = f"{p.pid:04x}" if p.pid else ""
        if (vid, pid) in MEGA_USB_IDS:
            matched.append(p.device)

    for dev in sorted(matched):
        print(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
