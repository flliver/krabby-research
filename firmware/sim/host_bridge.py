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
        self._lib.krabby_host_set_hall_edges_per_full_travel.argtypes = [ctypes.c_uint32]
        self._lib.krabby_host_set_hall_edges_per_full_travel.restype = None
        self._lib.krabby_host_sync_hall_joint.argtypes = [
            ctypes.c_uint8,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        self._lib.krabby_host_sync_hall_joint.restype = None
        self._lib.krabby_host_hall_edge_count.argtypes = [ctypes.c_uint8]
        self._lib.krabby_host_hall_edge_count.restype = ctypes.c_int32
        self._lib.krabby_host_eeprom_read.argtypes = [ctypes.c_int]
        self._lib.krabby_host_eeprom_read.restype = ctypes.c_uint8
        self._lib.krabby_host_eeprom_update.argtypes = [ctypes.c_int, ctypes.c_uint8]
        self._lib.krabby_host_eeprom_update.restype = None
        self._lib.krabby_host_save_calibration.argtypes = []
        self._lib.krabby_host_save_calibration.restype = None
        self._lib.krabby_host_load_calibration.argtypes = []
        self._lib.krabby_host_load_calibration.restype = None
        self._lib.krabby_host_actuator_min_stop.argtypes = [ctypes.c_char_p]
        self._lib.krabby_host_actuator_min_stop.restype = ctypes.c_int
        self._lib.krabby_host_actuator_max_stop.argtypes = [ctypes.c_char_p]
        self._lib.krabby_host_actuator_max_stop.restype = ctypes.c_int
        self._lib.krabby_host_serial_usb_rx_write.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
        ]
        self._lib.krabby_host_serial_usb_rx_write.restype = None
        self._lib.krabby_host_serial_usb_rx_available.argtypes = []
        self._lib.krabby_host_serial_usb_rx_available.restype = ctypes.c_int
        self._lib.krabby_host_serial_usb_rx_read.argtypes = []
        self._lib.krabby_host_serial_usb_rx_read.restype = ctypes.c_int
        self._lib.krabby_host_serial_usb_tx_available.argtypes = []
        self._lib.krabby_host_serial_usb_tx_available.restype = ctypes.c_int
        self._lib.krabby_host_serial_usb_tx_read.argtypes = []
        self._lib.krabby_host_serial_usb_tx_read.restype = ctypes.c_int
        self._lib.krabby_host_serial_usb_println.argtypes = [ctypes.c_char_p]
        self._lib.krabby_host_serial_usb_println.restype = None
        self._lib.krabby_host_firmware_setup.argtypes = []
        self._lib.krabby_host_firmware_setup.restype = None
        self._lib.krabby_host_firmware_loop_once.argtypes = []
        self._lib.krabby_host_firmware_loop_once.restype = None
        self._lib.krabby_host_firmware_reset_loop_state.argtypes = []
        self._lib.krabby_host_firmware_reset_loop_state.restype = None
        self._lib.krabby_host_run_three_board_role_election.argtypes = [
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._lib.krabby_host_run_three_board_role_election.restype = None
        self._lib.krabby_host_leader_board_id.argtypes = []
        self._lib.krabby_host_leader_board_id.restype = ctypes.c_int
        self._lib.krabby_host_board_usb_tx_line_available.argtypes = [ctypes.c_int]
        self._lib.krabby_host_board_usb_tx_line_available.restype = ctypes.c_int
        self._lib.krabby_host_board_usb_pop_tx_line.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.krabby_host_board_usb_pop_tx_line.restype = ctypes.c_int
        self._lib.krabby_host_enable_three_board_firmware_loop.argtypes = [ctypes.c_int]
        self._lib.krabby_host_enable_three_board_firmware_loop.restype = None
        self._lib.krabby_host_enable_single_leader_eighteen_joint.argtypes = [
            ctypes.c_int,
        ]
        self._lib.krabby_host_enable_single_leader_eighteen_joint.restype = None
        self._lib.krabby_host_three_board_actuator_command_pwm.argtypes = [
            ctypes.c_char_p,
        ]
        self._lib.krabby_host_three_board_actuator_command_pwm.restype = ctypes.c_int
        self._lib.krabby_host_i2c_set_device_present.argtypes = [
            ctypes.c_uint8,
            ctypes.c_int,
        ]
        self._lib.krabby_host_i2c_set_device_present.restype = None
        self._lib.krabby_host_oled_framebuffer_crc32.argtypes = []
        self._lib.krabby_host_oled_framebuffer_crc32.restype = ctypes.c_uint32
        self._lib.krabby_host_oled_invalidate.argtypes = []
        self._lib.krabby_host_oled_invalidate.restype = None
        self._lib.krabby_host_oled_wire_data_bytes_received.argtypes = []
        self._lib.krabby_host_oled_wire_data_bytes_received.restype = ctypes.c_size_t

    def reset(self) -> None:
        self._lib.krabby_host_reset()

    def firmware_setup(self) -> None:
        self._lib.krabby_host_firmware_setup()

    def firmware_loop_once(self) -> None:
        self._lib.krabby_host_firmware_loop_once()

    def firmware_reset_loop_state(self) -> None:
        """Clear setup/telemetry state without wiping EEPROM (simulated power-cycle)."""
        self._lib.krabby_host_firmware_reset_loop_state()

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

    def set_hall_edges_per_full_travel(self, edges: int) -> None:
        self._lib.krabby_host_set_hall_edges_per_full_travel(int(edges))

    def sync_hall_joint(
        self,
        hall_slot: int,
        angle_rad: float,
        lower_rad: float,
        upper_rad: float,
    ) -> None:
        self._lib.krabby_host_sync_hall_joint(
            int(hall_slot),
            float(angle_rad),
            float(lower_rad),
            float(upper_rad),
        )

    def hall_edge_count(self, hall_slot: int) -> int:
        return int(self._lib.krabby_host_hall_edge_count(int(hall_slot)))

    def eeprom_read(self, addr: int) -> int:
        return int(self._lib.krabby_host_eeprom_read(int(addr)))

    def eeprom_update(self, addr: int, value: int) -> None:
        self._lib.krabby_host_eeprom_update(int(addr), int(value) & 0xFF)

    def save_calibration(self) -> None:
        self._lib.krabby_host_save_calibration()

    def load_calibration(self) -> None:
        self._lib.krabby_host_load_calibration()

    def actuator_min_stop(self, actuator_name: str) -> int:
        return int(
            self._lib.krabby_host_actuator_min_stop(actuator_name.encode("ascii"))
        )

    def actuator_max_stop(self, actuator_name: str) -> int:
        return int(
            self._lib.krabby_host_actuator_max_stop(actuator_name.encode("ascii"))
        )

    def serial_usb_rx_write(self, data: bytes) -> None:
        if not data:
            return
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        self._lib.krabby_host_serial_usb_rx_write(buf, len(data))

    def serial_usb_rx_available(self) -> int:
        return int(self._lib.krabby_host_serial_usb_rx_available())

    def serial_usb_rx_read(self) -> int:
        return int(self._lib.krabby_host_serial_usb_rx_read())

    def serial_usb_tx_available(self) -> int:
        return int(self._lib.krabby_host_serial_usb_tx_available())

    def serial_usb_tx_read(self) -> int:
        return int(self._lib.krabby_host_serial_usb_tx_read())

    def serial_usb_println(self, text: str) -> None:
        self._lib.krabby_host_serial_usb_println(text.encode("ascii"))

    def enable_three_board_firmware_loop(self, enabled: bool = True) -> None:
        self._lib.krabby_host_enable_three_board_firmware_loop(1 if enabled else 0)

    def enable_single_leader_eighteen_joint(self, enabled: bool = True) -> None:
        self._lib.krabby_host_enable_single_leader_eighteen_joint(1 if enabled else 0)

    def three_board_actuator_command_pwm(self, actuator_name: str) -> int:
        return int(
            self._lib.krabby_host_three_board_actuator_command_pwm(
                actuator_name.encode("ascii")
            )
        )

    def i2c_set_device_present(self, address: int, present: bool) -> None:
        self._lib.krabby_host_i2c_set_device_present(
            int(address) & 0xFF,
            1 if present else 0,
        )

    def oled_framebuffer_crc32(self) -> int:
        return int(self._lib.krabby_host_oled_framebuffer_crc32())

    def oled_invalidate(self) -> None:
        self._lib.krabby_host_oled_invalidate()

    def oled_wire_data_bytes_received(self) -> int:
        return int(self._lib.krabby_host_oled_wire_data_bytes_received())

    def run_three_board_role_election(self) -> list[int]:
        roles = (ctypes.c_int32 * 3)()
        self._lib.krabby_host_run_three_board_role_election(roles)
        return [int(roles[i]) for i in range(3)]

    def leader_board_id(self) -> int:
        return int(self._lib.krabby_host_leader_board_id())

    def board_usb_pop_tx_line(self, board_id: int) -> str | None:
        buf = ctypes.create_string_buffer(512)
        n = int(
            self._lib.krabby_host_board_usb_pop_tx_line(
                int(board_id),
                buf,
                len(buf),
            )
        )
        if n <= 0:
            return None
        return buf.value.decode("ascii", errors="replace")

    def drain_board_usb_lines(self, board_id: int) -> list[str]:
        lines: list[str] = []
        while True:
            line = self.board_usb_pop_tx_line(board_id)
            if line is None:
                break
            lines.append(line)
        return lines


def load_host_bridge() -> HostActuatorBridge:
    return HostActuatorBridge()
