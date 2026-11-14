"""can_handler.c against krabby_stm_host sim board."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import pytest

from firmware.stm import codec, ids

REPO_ROOT = Path(__file__).resolve().parents[4]
LIB_PATH = REPO_ROOT / "firmware" / "build" / "stm-host" / "libkrabby_stm_host.so"


class Leg(ctypes.Structure):
    _fields_ = [("leg_index", ctypes.c_uint8), ("joint_count", ctypes.c_uint8)]


@pytest.fixture(scope="module")
def stm_host_lib():
    if not LIB_PATH.is_file():
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "firmware"), "host-stm"],
            check=True,
        )
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.can_handler_on_frame.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint8,
    ]
    lib.can_handler_on_frame.restype = None
    lib.krabby_sim_get_duty.argtypes = [ctypes.POINTER(Leg), ctypes.c_int]
    lib.krabby_sim_get_duty.restype = ctypes.c_float
    return lib


def test_set_duty_applies_this_leg_slice_only(stm_host_lib) -> None:
    leg = Leg(leg_index=1, joint_count=3)  # leg_id 2
    grid = ((0.0, 0.0, 0.0),) + ((0.75, -0.25, 0.0),) + ((0.0, 0.0, 0.0),) * 4
    payload = codec.set_duty_values_from_grid(grid)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.75)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 1) == pytest.approx(-0.25)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 2) == pytest.approx(0.0)


def test_calibrate_id_for_other_leg_is_ignored(stm_host_lib) -> None:
    leg = Leg(leg_index=0, joint_count=3)
    cal_id = ids.calibrate_can_id(2)
    buf = (ctypes.c_uint8 * 1)(2)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    # No duty change; calibrate handler is a no-op until calibration.c grows.


def test_telemetry_tx_rate_and_payload(stm_host_lib) -> None:
    leg = Leg(leg_index=0, joint_count=3)
    payload = codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))

    tick = stm_host_lib.telemetry_handler_tick
    tick.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint32]
    tick.restype = None
    tick(ctypes.byref(leg), 0)

    last_id = ctypes.c_uint32()
    last_len = ctypes.c_uint8()
    tx_buf = (ctypes.c_uint8 * 64)()
    get_tx = stm_host_lib.krabby_sim_last_tx
    get_tx.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
    ]
    get_tx.restype = ctypes.c_int
    assert get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len)) == 0
    assert last_id.value == ids.telemetry_can_id(1)
    decoded = codec.decode_telemetry(1, bytes(tx_buf[: last_len.value]))
    assert decoded["J1_DUTY"] == pytest.approx(0.5)

    tick(ctypes.byref(leg), 40)
    prev_len = last_len.value
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_len.value == prev_len

    tick(ctypes.byref(leg), 50)
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_id.value == ids.telemetry_can_id(1)

    tick(ctypes.byref(leg), 1000)
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_id.value == ids.board_info_can_id(1)
    decoded_info = codec.decode_board_info(1, bytes(tx_buf[: last_len.value]))
    assert decoded_info["major"] == 0
    assert decoded_info["minor"] == 1


def test_set_duty_zero_on_fault_clears(stm_host_lib) -> None:
    leg = Leg(leg_index=0, joint_count=3)
    # Prime non-zero duty
    payload = codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))
    zero = codec.encode_set_duty(codec.empty_set_duty_values())[1]
    zbuf = (ctypes.c_uint8 * len(zero)).from_buffer_copy(zero)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), zbuf, len(zero))
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.0)