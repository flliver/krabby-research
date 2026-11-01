"""
Serial endpoint for Krabby MCU (Arduino Mega over USB or sim virtual serial).

Used by krabby_mcu.KrabbyMCUSDK, firmware/scripts/detect_mcu_port.py, and Makefile.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import serial

# USB VID/PID pairs for boards we drive. Shared with firmware.cli port scanning.
MEGA_USB_IDS = {
    ("2341", "0042"), ("2341", "0010"), ("2341", "0110"),  # Arduino Mega native USB
    ("1a86", "7523"), ("1a86", "5523"),                    # CH340 / CH341 (Krabby-Uno shield)
}


@dataclass(frozen=True, slots=True)
class McuSerialDevice:
    """Local serial device (USB ACM, PTY slave path, COM port)."""

    path: str

    def __str__(self) -> str:
        return self.path

    def open(
        self,
        *,
        baud: int,
        timeout: float = 0.5,
        write_timeout: float = 0.5,
    ) -> serial.Serial:
        ser = serial.Serial()
        ser.port = self.path
        ser.baudrate = baud
        ser.timeout = timeout
        ser.write_timeout = write_timeout
        ser.dtr = False
        ser.open()
        return ser


@dataclass(frozen=True, slots=True)
class McuSerialUrl:
    """Pyserial URL (e.g. ``socket://127.0.0.1:54321`` for Windows sim TCP bridge)."""

    url: str

    def __str__(self) -> str:
        return self.url

    def open(
        self,
        *,
        baud: int,
        timeout: float = 0.5,
        write_timeout: float = 0.5,
    ) -> serial.Serial:
        ser = serial.serial_for_url(
            self.url,
            baudrate=baud,
            timeout=timeout,
            write_timeout=write_timeout,
        )
        ser.dtr = False
        return ser


McuSerialTarget = Union[McuSerialDevice, McuSerialUrl]


def parse_krabby_mcu_port_env(value: str) -> McuSerialTarget:
    """Parse ``KRABBY_MCU_PORT`` (device path/COM or pyserial URL).

    This is the only place that distinguishes URL vs device by the ``://`` heuristic.
    """
    raw = value.strip()
    if not raw:
        raise ValueError("KRABBY_MCU_PORT is empty")
    if "://" in raw:
        return McuSerialUrl(url=raw)
    return McuSerialDevice(path=raw)


def _autodetect_device_path() -> str:
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
        "usb serial",
    )

    ports = list_ports.comports()

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


def default_target() -> McuSerialTarget:
    """
    Default MCU serial endpoint for ``KrabbyMCUSDK()``.

    Priority:
      1) ``KRABBY_MCU_PORT`` env (parsed by :func:`parse_krabby_mcu_port_env`)
      2) USB VID/PID or description heuristics → :class:`McuSerialDevice`
      3) OS fallback ``/dev/ttyACM0`` or ``COM5``
    """
    env_port = os.getenv("KRABBY_MCU_PORT")
    if env_port:
        return parse_krabby_mcu_port_env(env_port)
    return McuSerialDevice(path=_autodetect_device_path())
