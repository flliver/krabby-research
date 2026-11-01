from types import SimpleNamespace
from unittest.mock import patch

from firmware.mcu_port import (
    McuSerialDevice,
    McuSerialUrl,
    default_target,
    parse_krabby_mcu_port_env,
)


def _port(device, description=None, manufacturer=None, vid=None, pid=None):
    return SimpleNamespace(
        device=device, description=description, manufacturer=manufacturer,
        vid=vid, pid=pid,
    )


def _detect(ports, monkeypatch, env_port=None):
    monkeypatch.delenv("KRABBY_MCU_PORT", raising=False)
    if env_port:
        monkeypatch.setenv("KRABBY_MCU_PORT", env_port)
    with patch("serial.tools.list_ports.comports", return_value=ports):
        return default_target()


class TestParseKrabbyMcuPortEnv:
    def test_device_path(self):
        t = parse_krabby_mcu_port_env("/dev/ttyACM0")
        assert t == McuSerialDevice(path="/dev/ttyACM0")

    def test_socket_url(self):
        t = parse_krabby_mcu_port_env("socket://127.0.0.1:54321")
        assert t == McuSerialUrl(url="socket://127.0.0.1:54321")


class TestDefaultTarget:
    def test_env_device_override_wins(self, monkeypatch):
        ports = [_port("/dev/ttyUSB0", "Arduino Mega")]
        t = _detect(ports, monkeypatch, env_port="/dev/ttyXYZ")
        assert t == McuSerialDevice(path="/dev/ttyXYZ")

    def test_env_url_override(self, monkeypatch):
        ports = [_port("/dev/ttyUSB0", "Arduino Mega")]
        t = _detect(ports, monkeypatch, env_port="socket://127.0.0.1:9")
        assert t == McuSerialUrl(url="socket://127.0.0.1:9")

    def test_ch340_matched_by_usb_id_despite_generic_description(self, monkeypatch):
        ports = [_port("/dev/ttyUSB0", "USB Serial", None, vid=0x1A86, pid=0x7523)]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyUSB0")

    def test_mega_native_usb_matched_by_id(self, monkeypatch):
        ports = [
            _port("/dev/ttyS0", "ttyS0"),
            _port("/dev/ttyACM1", "weird name", None, vid=0x2341, pid=0x0042),
        ]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyACM1")

    def test_id_match_preferred_over_keyword_match(self, monkeypatch):
        ports = [
            _port("/dev/ttyUSB1", "CP2102 USB to UART", "Silicon Labs"),
            _port("/dev/ttyUSB0", "USB Serial", None, vid=0x1A86, pid=0x7523),
        ]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyUSB0")

    def test_keyword_match_in_description(self, monkeypatch):
        ports = [_port("/dev/ttyACM2", "Arduino Mega 2560")]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyACM2")

    def test_keyword_match_usb_serial_with_space(self, monkeypatch):
        ports = [_port("/dev/ttyUSB3", "USB Serial Converter")]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyUSB3")

    def test_fallback_when_nothing_matches(self, monkeypatch):
        ports = [_port("/dev/ttyS0", "ttyS0")]
        t = _detect(ports, monkeypatch)
        assert t == McuSerialDevice(path="/dev/ttyACM0")
