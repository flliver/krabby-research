"""
Serial port auto-detection for Krabby MCU (Arduino Mega over USB).
Used by krabby_mcu.KrabbyMCUSDK and firmware/scripts/detect_mcu_port.py / Makefile.
"""
from __future__ import annotations

import os


def default_port() -> str:
    """
    Best-effort auto-detection of the MCU serial port.
    Priority:
      1) KRABBY_MCU_PORT env var (explicit override)
      2) USB description/manufacturer containing common board identifiers
      3) OS-specific fallback
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

    for p in list_ports.comports():
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if any(k in desc or k in manuf for k in preferred_keywords):
            return p.device

    return "COM5" if os.name == "nt" else "/dev/ttyACM0"
