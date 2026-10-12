"""ctypes bridge to the host firmware actuator library (libkrabby_host)."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path


class HostLibraryNotBuiltError(FileNotFoundError):
    """Raised when libkrabby_host was not built (run make -C firmware test-native)."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _library_candidates() -> list[Path]:
    root = _repo_root()
    name = "krabby_host.dll" if sys.platform == "win32" else "libkrabby_host.so"
    return [root / "build" / "native-firmware-tests" / "krabby_sim_host" / name]


def find_host_library() -> Path:
    for path in _library_candidates():
        if path.is_file():
            return path
    raise HostLibraryNotBuiltError(
        "libkrabby_host not found; build with: make -C firmware test-native"
    )


class HostActuatorBridge:
    """Calls shipping handleJog / manualDrive path in libkrabby_host."""

    def __init__(self, lib_path: Path | None = None) -> None:
        path = lib_path or find_host_library()
        self._lib = ctypes.CDLL(str(path))
        self._lib.krabby_host_reset.argtypes = []
        self._lib.krabby_host_reset.restype = None
        self._lib.krabby_host_handle_jog.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.krabby_host_handle_jog.restype = None
        self._lib.krabby_host_set_target.argtypes = [ctypes.c_char_p, ctypes.c_float]
        self._lib.krabby_host_set_target.restype = None
        self._lib.krabby_host_set_actuator_stops.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.krabby_host_set_actuator_stops.restype = None
        self._lib.krabby_host_update_all.argtypes = []
        self._lib.krabby_host_update_all.restype = None
        self._lib.krabby_host_pin_pwm.argtypes = [ctypes.c_int]
        self._lib.krabby_host_pin_pwm.restype = ctypes.c_int
        self._lib.krabby_host_pin_digital.argtypes = [ctypes.c_int]
        self._lib.krabby_host_pin_digital.restype = ctypes.c_int
        self._lib.krabby_host_set_analog_read.argtypes = [ctypes.c_int, ctypes.c_int]
        self._lib.krabby_host_set_analog_read.restype = None
        self._lib.krabby_host_analog_read.argtypes = [ctypes.c_int]
        self._lib.krabby_host_analog_read.restype = ctypes.c_int
        self._lib.krabby_host_advance_millis.argtypes = [ctypes.c_ulong]
        self._lib.krabby_host_advance_millis.restype = None
        self._lib.krabby_host_millis.argtypes = []
        self._lib.krabby_host_millis.restype = ctypes.c_ulong

    def reset(self) -> None:
        self._lib.krabby_host_reset()

    def handle_jog(self, actuator_name: str, pwm: int) -> None:
        self._lib.krabby_host_handle_jog(actuator_name.encode("ascii"), int(pwm))

    def set_target(self, actuator_name: str, position: float) -> None:
        self._lib.krabby_host_set_target(actuator_name.encode("ascii"), float(position))

    def set_actuator_stops(self, actuator_name: str, min_stop: int, max_stop: int) -> None:
        self._lib.krabby_host_set_actuator_stops(
            actuator_name.encode("ascii"),
            int(min_stop),
            int(max_stop),
        )

    def update_all(self) -> None:
        """One firmware step: ActuatorManager::updateAll (sensors + per-actuator update)."""
        self._lib.krabby_host_update_all()

    def pin_pwm(self, pin: int) -> int:
        return int(self._lib.krabby_host_pin_pwm(int(pin)))

    def pin_digital(self, pin: int) -> int:
        return int(self._lib.krabby_host_pin_digital(int(pin)))

    def set_analog_read(self, pin: int, value: int) -> None:
        self._lib.krabby_host_set_analog_read(int(pin), int(value))

    def analog_read(self, pin: int) -> int:
        return int(self._lib.krabby_host_analog_read(int(pin)))

    def advance_millis(self, delta_ms: int) -> None:
        self._lib.krabby_host_advance_millis(int(delta_ms))

    def millis(self) -> int:
        return int(self._lib.krabby_host_millis())


def load_host_bridge() -> HostActuatorBridge:
    return HostActuatorBridge()
