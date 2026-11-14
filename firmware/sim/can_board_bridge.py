"""ctypes bridge to libkrabby_stm_host (Isaac leg boards + CAN sim queues)."""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path


class StmHostLibraryNotBuiltError(FileNotFoundError):
    """Raise when libkrabby_stm_host is missing (run make -C firmware host-stm)."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def stm_host_library_candidates() -> list[Path]:
    root = _repo_root()
    name = "krabby_stm_host.dll" if sys.platform == "win32" else "libkrabby_stm_host.so"
    return [
        root / "firmware" / "build" / "stm-host" / name,
    ]


def find_stm_host_library() -> Path:
    for path in stm_host_library_candidates():
        if path.is_file():
            return path
    raise StmHostLibraryNotBuiltError(
        "libkrabby_stm_host not found; build with: make -C firmware host-stm"
    )


class Leg(ctypes.Structure):
    _fields_ = [
        ("leg_index", ctypes.c_uint8),
        ("joint_count", ctypes.c_uint8),
        ("pins", ctypes.c_void_p),
    ]


def _ensure_built(lib_path: Path) -> None:
    if lib_path.is_file():
        return
    subprocess.run(
        ["make", "-C", str(_repo_root() / "firmware"), "host-stm"],
        check=True,
    )


class StmBoardBridge:
    """One krabby_stm_host process: six legs, shared sim clock, per-leg CAN queues."""

    def __init__(self, lib_path: Path | None = None) -> None:
        path = lib_path or find_stm_host_library()
        _ensure_built(path)
        self._lib = ctypes.CDLL(str(path))
        self._bind()

    def _bind(self) -> None:
        lib = self._lib
        lib.board_pins_isaac_create.argtypes = [ctypes.c_uint8]
        lib.board_pins_isaac_create.restype = ctypes.c_void_p
        lib.app_init.argtypes = [ctypes.POINTER(Leg), ctypes.c_void_p]
        lib.app_init.restype = None
        lib.app_tick.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint32]
        lib.app_tick.restype = None
        lib.krabby_sim_set_tick_ms.argtypes = [ctypes.c_int]
        lib.krabby_sim_set_tick_ms.restype = None
        lib.krabby_sim_inject_rx.argtypes = [
            ctypes.POINTER(Leg),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint8,
        ]
        lib.krabby_sim_inject_rx.restype = None
        lib.krabby_sim_last_tx.argtypes = [
            ctypes.POINTER(Leg),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.krabby_sim_last_tx.restype = ctypes.c_int
        lib.krabby_sim_set_pot.argtypes = [
            ctypes.POINTER(Leg),
            ctypes.c_int,
            ctypes.c_uint16,
        ]
        lib.krabby_sim_set_pot.restype = None
        lib.krabby_sim_set_hall.argtypes = [
            ctypes.POINTER(Leg),
            ctypes.c_int,
            ctypes.c_int32,
        ]
        lib.krabby_sim_set_hall.restype = None
        lib.krabby_sim_get_duty.argtypes = [ctypes.POINTER(Leg), ctypes.c_int]
        lib.krabby_sim_get_duty.restype = ctypes.c_float

    def create_leg(self, node_id: int) -> Leg:
        pins = self._lib.board_pins_isaac_create(node_id)
        if not pins:
            raise ValueError(f"invalid node_id: {node_id!r}")
        leg = Leg()
        self._lib.app_init(ctypes.byref(leg), pins)
        return leg

    def set_tick_ms(self, ms: int) -> None:
        self._lib.krabby_sim_set_tick_ms(int(ms))

    def app_tick(self, leg: Leg, now_ms: int) -> None:
        self._lib.app_tick(ctypes.byref(leg), ctypes.c_uint32(now_ms))

    def inject_rx(self, leg: Leg, can_id: int, data: bytes) -> None:
        buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
        self._lib.krabby_sim_inject_rx(
            ctypes.byref(leg),
            ctypes.c_uint32(can_id),
            buf,
            ctypes.c_uint8(len(data)),
        )

    def last_tx(self, leg: Leg) -> tuple[int, bytes] | None:
        can_id = ctypes.c_uint32()
        length = ctypes.c_uint8()
        buf = (ctypes.c_uint8 * 64)()
        rc = self._lib.krabby_sim_last_tx(
            ctypes.byref(leg),
            ctypes.byref(can_id),
            buf,
            ctypes.byref(length),
        )
        if rc != 0 or length.value == 0:
            return None
        return int(can_id.value), bytes(buf[: length.value])

    def set_pot(self, leg: Leg, motor_index: int, raw_adc: int) -> None:
        self._lib.krabby_sim_set_pot(
            ctypes.byref(leg),
            ctypes.c_int(motor_index),
            ctypes.c_uint16(raw_adc),
        )

    def set_hall(self, leg: Leg, motor_index: int, count: int) -> None:
        self._lib.krabby_sim_set_hall(
            ctypes.byref(leg),
            ctypes.c_int(motor_index),
            ctypes.c_int32(count),
        )

    def get_duty(self, leg: Leg, motor_index: int) -> float:
        return float(self._lib.krabby_sim_get_duty(ctypes.byref(leg), ctypes.c_int(motor_index)))


def load_stm_board_bridge(lib_path: Path | None = None) -> StmBoardBridge:
    return StmBoardBridge(lib_path=lib_path)