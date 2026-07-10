"""
Serial port auto-detection for Krabby MCU (Arduino Mega over USB).
Used by krabby_mcu.KrabbyMCUSDK and firmware/scripts/detect_mcu_port.py / Makefile.
"""
from __future__ import annotations

import os

# USB VID/PID pairs for boards we drive — the canonical table; firmware/cli.py
# imports it and firmware/scripts/list_mcu_ports.py duplicates it (that script
# must stand alone on a remote flash host).
MEGA_USB_IDS = {
    ("2341", "0042"), ("2341", "0010"), ("2341", "0110"),  # Arduino Mega native USB
    ("1a86", "7523"), ("1a86", "5523"),                    # CH340 / CH341 (Krabby-Uno shield)
}


def default_port() -> str:
    """
    Best-effort auto-detection of the MCU serial port.
    Priority:
      1) KRABBY_MCU_PORT env var (explicit override)
      2) USB VID:PID match (exact; CH340s often enumerate with a bare
         "USB Serial" description that keyword matching misses)
      3) USB description/manufacturer containing common board identifiers
      4) OS-specific fallback
    """
    env_port = os.getenv("KRABBY_MCU_PORT")
    if env_port:
        return env_port

    try:
        from serial.tools import list_ports
    except ImportError:
        raise RuntimeError(
            "pyserial is required for port auto-detection (pip install pyserial)"
        ) from None

    preferred_keywords = (
        "arduino",
        "dfrobot",
        "dfduino",
        "ch340",
        "cp210",
        "usb-serial",
    )

    ports = list(list_ports.comports())

    for p in ports:
        vid = f"{p.vid:04x}" if p.vid else ""
        pid = f"{p.pid:04x}" if p.pid else ""
        if (vid, pid) in MEGA_USB_IDS:
            return p.device

    for p in ports:
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if any(k in desc or k in manuf for k in preferred_keywords):
            return p.device

    return "COM5" if os.name == "nt" else "/dev/ttyACM0"
