"""Core handlers via krabby_stm_host (app, pins, CAN)."""

from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path

import pytest

from firmware.stm import codec, ids

from tests.unit.firmware.stm.dbc_test_utils import dbc_float64

REPO_ROOT = Path(__file__).resolve().parents[4]
LIB_PATH = REPO_ROOT / "firmware" / "build" / "stm-host" / "libkrabby_stm_host.so"


class Leg(ctypes.Structure):
    _fields_ = [
        ("leg_index", ctypes.c_uint8),
        ("joint_count", ctypes.c_uint8),
        ("pins", ctypes.c_void_p),
    ]


def _bind_leg_helpers(lib: ctypes.CDLL) -> None:
    lib.board_pins_isaac_create.argtypes = [ctypes.c_uint8]
    lib.board_pins_isaac_create.restype = ctypes.c_void_p
    lib.app_init.argtypes = [ctypes.POINTER(Leg), ctypes.c_void_p]
    lib.app_init.restype = None
    lib.app_tick.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint32]
    lib.app_tick.restype = None
    lib.can_handler_on_frame.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint8,
    ]
    lib.can_handler_on_frame.restype = None
    lib.telemetry_handler_tick.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint32]
    lib.telemetry_handler_tick.restype = None
    lib.krabby_sim_get_duty.argtypes = [ctypes.POINTER(Leg), ctypes.c_int]
    lib.krabby_sim_get_duty.restype = ctypes.c_float
    lib.krabby_sim_inject_rx.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint8,
    ]
    lib.krabby_sim_inject_rx.restype = None
    lib.krabby_sim_fail_next_tx.argtypes = [ctypes.POINTER(Leg)]
    lib.krabby_sim_fail_next_tx.restype = None
    lib.krabby_sim_set_hall.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.c_int,
        ctypes.c_int32,
    ]
    lib.krabby_sim_set_hall.restype = None
    lib.krabby_sim_set_pot.argtypes = [
        ctypes.POINTER(Leg),
        ctypes.c_int,
        ctypes.c_uint16,
    ]
    lib.krabby_sim_set_pot.restype = None
    lib.calibration_init.argtypes = [ctypes.POINTER(Leg)]
    lib.calibration_init.restype = None
    lib.calibration_joint_calibrating.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint8]
    lib.calibration_joint_calibrating.restype = ctypes.c_bool
    lib.calibration_joint_calibrated.argtypes = [ctypes.POINTER(Leg), ctypes.c_uint8]
    lib.calibration_joint_calibrated.restype = ctypes.c_bool


def _leg_for_node(lib: ctypes.CDLL, node_id: int) -> Leg:
    pins = lib.board_pins_isaac_create(node_id)
    leg = Leg()
    lib.app_init(ctypes.byref(leg), pins)
    return leg


@pytest.fixture(scope="module")
def stm_host_lib():
    if not LIB_PATH.is_file():
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "firmware"), "host-stm"],
            check=True,
        )
    lib = ctypes.CDLL(str(LIB_PATH))
    _bind_leg_helpers(lib)
    return lib


def test_app_tick_drains_rx_queue(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    payload = codec.set_duty_values_from_grid(((0.25, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.krabby_sim_inject_rx(
        ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data)
    )
    stm_host_lib.app_tick(ctypes.byref(leg), 0)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.25)


def test_set_duty_applies_this_leg_slice_only(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 2)
    grid = ((0.0, 0.0, 0.0),) + ((0.75, -0.25, 0.0),) + ((0.0, 0.0, 0.0),) * 4
    payload = codec.set_duty_values_from_grid(grid)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.75)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 1) == pytest.approx(-0.25)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 2) == pytest.approx(0.0)


def test_set_duty_only_other_leg_rows_change_this_board(stm_host_lib) -> None:
    """Updates on other legs' SET_DUTY slots do not change this node's three duties."""
    leg = _leg_for_node(stm_host_lib, 1)
    grid_a = ((0.35, 0.05, 0.0),) + ((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 4
    grid_b = ((0.35, 0.05, 0.0),) + ((0.9, -0.2, 0.1),) + ((0.0, 0.0, 0.0),) * 4
    _send_set_duty_grid(stm_host_lib, leg, grid_a)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.35)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 1) == pytest.approx(0.05)

    _send_set_duty_grid(stm_host_lib, leg, grid_b)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.35)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 1) == pytest.approx(0.05)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 2) == pytest.approx(0.0)


def test_set_duty_negative_half_retract_power_500(stm_host_lib) -> None:
    """Negative duty maps to direction false and |duty| 0.5 (power 500) in Isaac pin storage."""
    leg = _leg_for_node(stm_host_lib, 1)
    _send_set_duty_leg(stm_host_lib, leg, leg_id=1, yaw=-0.5, hip=0.0, knee=0.0)
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(-0.5)

    decoded = _decode_telemetry_at(stm_host_lib, leg, 0)
    assert decoded["J1_DUTY"] == pytest.approx(-0.5)


def test_telemetry_reflects_pot_set_from_python(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 0, 2507)
    decoded = _decode_telemetry_at(stm_host_lib, leg, 0)
    assert decoded["J1_POT"] == 2507


def _send_set_duty_grid(stm_host_lib, leg: Leg, grid: tuple[tuple[float, float, float], ...]) -> None:
    _frame_id, data = codec.encode_set_duty(codec.set_duty_values_from_grid(grid))
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))


def test_calibrate_rejects_joint_one(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    cal_id = ids.calibrate_can_id(1)
    buf = (ctypes.c_uint8 * 1)(1)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert not stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), 1)
    assert not stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 1)


def _send_set_duty_leg(
    stm_host_lib, leg: Leg, *, leg_id: int, yaw: float, hip: float, knee: float
) -> None:
    rows: list[tuple[float, float, float]] = []
    for lid in ids.LEG_IDS:
        rows.append((yaw, hip, knee) if lid == leg_id else (0.0, 0.0, 0.0))
    _frame_id, data = codec.encode_set_duty(codec.set_duty_values_from_grid(tuple(rows)))
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))


def _run_stall_cal_host(stm_host_lib, leg: Leg, joint_id: int, motor: int) -> None:
    """Simulate hall motion then 250 ms quiet per phase (grant stall rule)."""
    cal_id = ids.calibrate_can_id(1)
    buf = (ctypes.c_uint8 * 1)(joint_id)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), joint_id)

    # Extend phase: moving hall, then hold for KRABBY_CAL_STALL_QUIET_MS (250).
    for ms in range(0, 40):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, ms * 10)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)
    stall_hall = 500
    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, stall_hall)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), motor, 3000)
    for ms in range(40, 40 + 260):
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    # Retract phase: move hall again, then second stall.
    for ms in range(300, 340):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, stall_hall - (ms - 300) * 5)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)
    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, 100)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), motor, 400)
    for ms in range(340, 340 + 260):
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    assert stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), joint_id)
    assert not stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), joint_id)


def test_calibrated_reflects_persisted_pin_storage(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    assert not stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 2)
    _run_stall_cal_host(stm_host_lib, leg, 2, 1)
    assert stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 2)
    stm_host_lib.calibration_init(ctypes.byref(leg))
    assert stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 2)
    assert not stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 1)


def test_calibrate_joint_two_stall_finishes_on_host(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    _run_stall_cal_host(stm_host_lib, leg, 2, 1)


def test_calibrate_joint_three_stall_finishes_on_host(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    _run_stall_cal_host(stm_host_lib, leg, 3, 2)


def _run_stall_cal_flat_pot_host(
    stm_host_lib, leg: Leg, leg_id: int, joint_id: int, motor: int
) -> None:
    """Stall cal with identical pot at both ends so position uses hall interpolation."""
    cal_id = ids.calibrate_can_id(leg_id)
    buf = (ctypes.c_uint8 * 1)(joint_id)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), joint_id)

    flat_pot = 1500
    for ms in range(0, 40):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, ms * 10)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)
    stall_hall = 800
    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, stall_hall)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), motor, flat_pot)
    for ms in range(40, 40 + 260):
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    for ms in range(300, 340):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, stall_hall - (ms - 300) * 5)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)
    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, 200)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), motor, flat_pot)
    for ms in range(340, 340 + 260):
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    assert stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), joint_id)


def test_set_duty_during_cal_aborts_and_retains_prior_cal(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    _run_stall_cal_host(stm_host_lib, leg, 2, 1)

    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 100)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 1, 400)
    decoded = _decode_telemetry_at(stm_host_lib, leg, 800)
    pos_before = dbc_float64(decoded["J2_POSITION"])
    assert pos_before == pytest.approx(0.0, abs=0.01)

    cal_id = ids.calibrate_can_id(1)
    buf = (ctypes.c_uint8 * 1)(2)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), 2)
    for ms in range(900, 920):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 200 + (ms - 900) * 5)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    _send_set_duty_leg(stm_host_lib, leg, leg_id=1, yaw=0.0, hip=0.5, knee=0.0)
    stm_host_lib.app_tick(ctypes.byref(leg), 920)

    assert not stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), 2)
    assert stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), 2)

    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 100)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 1, 400)
    decoded_after = _decode_telemetry_at(stm_host_lib, leg, 950)
    assert dbc_float64(decoded_after["J2_POSITION"]) == pytest.approx(pos_before, abs=0.01)


def test_calibrate_never_stall_times_out_without_save(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 3)
    joint_id = 2
    motor = 1
    cal_id = ids.calibrate_can_id(3)
    buf = (ctypes.c_uint8 * 1)(joint_id)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), joint_id)

    for ms in range(0, 10100, 50):
        stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), motor, ms)
        stm_host_lib.app_tick(ctypes.byref(leg), ms)

    assert not stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), joint_id)
    assert not stm_host_lib.calibration_joint_calibrated(ctypes.byref(leg), joint_id)


def test_telemetry_position_uses_hall_when_pot_span_flat(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 4)
    _run_stall_cal_flat_pot_host(stm_host_lib, leg, 4, 2, 1)

    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 500)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 1, 1500)
    decoded = _decode_telemetry_at(stm_host_lib, leg, 650)
    assert dbc_float64(decoded["J2_POSITION"]) == pytest.approx(0.5, abs=0.02)


def test_telemetry_joint_position_at_calibration_stops(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)

    decoded = _decode_telemetry_at(stm_host_lib, leg, 0)
    assert math.isnan(dbc_float64(decoded["J2_POSITION"]))
    assert decoded["J2_CALIBRATED"] == 0

    _run_stall_cal_host(stm_host_lib, leg, 2, 1)

    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 100)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 1, 400)
    decoded = _decode_telemetry_at(stm_host_lib, leg, 650)
    assert dbc_float64(decoded["J2_POSITION"]) == pytest.approx(0.0)
    assert decoded["J2_CALIBRATED"] == 1
    assert math.isnan(dbc_float64(decoded["J1_POSITION"]))
    assert decoded["J1_CALIBRATED"] == 0

    stm_host_lib.krabby_sim_set_hall(ctypes.byref(leg), 1, 500)
    stm_host_lib.krabby_sim_set_pot(ctypes.byref(leg), 1, 3000)
    decoded = _decode_telemetry_at(stm_host_lib, leg, 700)
    assert dbc_float64(decoded["J2_POSITION"]) == pytest.approx(1.0)


def test_calibrate_id_for_other_leg_is_ignored(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    cal_id = ids.calibrate_can_id(2)
    buf = (ctypes.c_uint8 * 1)(2)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), cal_id, buf, 1)
    assert not stm_host_lib.calibration_joint_calibrating(ctypes.byref(leg), 2)


def _decode_telemetry_at(stm_host_lib, leg: Leg, now_ms: int) -> dict[str, object]:
    leg_id = int(leg.leg_index) + 1
    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), now_ms)
    last_id, payload = _last_tx(stm_host_lib, leg)
    assert last_id == ids.telemetry_can_id(leg_id)
    return codec.decode_telemetry(leg_id, payload)


def _last_tx(stm_host_lib, leg: Leg) -> tuple[int, bytes]:
    last_id = ctypes.c_uint32()
    last_len = ctypes.c_uint8()
    tx_buf = (ctypes.c_uint8 * 64)()
    get_tx = stm_host_lib.krabby_sim_last_tx
    assert (
        get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len)) == 0
    )
    return last_id.value, bytes(tx_buf[: last_len.value])


def test_telemetry_dropped_when_can_send_fails(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 0)
    prev_id, prev_payload = _last_tx(stm_host_lib, leg)

    stm_host_lib.krabby_sim_fail_next_tx(ctypes.byref(leg))
    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 50)

    last_id, last_payload = _last_tx(stm_host_lib, leg)
    assert last_id == prev_id
    assert last_payload == prev_payload

    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 100)
    last_id, _last_payload = _last_tx(stm_host_lib, leg)
    assert last_id == ids.telemetry_can_id(1)


def test_telemetry_tx_rate_and_payload(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    payload = codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))

    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 0)

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

    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 40)
    prev_len = last_len.value
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_len.value == prev_len

    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 50)
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_id.value == ids.telemetry_can_id(1)

    stm_host_lib.telemetry_handler_tick(ctypes.byref(leg), 1000)
    get_tx(ctypes.byref(leg), ctypes.byref(last_id), tx_buf, ctypes.byref(last_len))
    assert last_id.value == ids.board_info_can_id(1)
    decoded_info = codec.decode_board_info(1, bytes(tx_buf[: last_len.value]))
    assert decoded_info["major"] == 0
    assert decoded_info["minor"] == 1


def test_set_duty_zero_on_fault_clears(stm_host_lib) -> None:
    leg = _leg_for_node(stm_host_lib, 1)
    payload = codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _frame_id, data = codec.encode_set_duty(payload)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), buf, len(data))
    zero = codec.encode_set_duty(codec.empty_set_duty_values())[1]
    zbuf = (ctypes.c_uint8 * len(zero)).from_buffer_copy(zero)
    stm_host_lib.can_handler_on_frame(ctypes.byref(leg), ids.set_duty_can_id(), zbuf, len(zero))
    assert stm_host_lib.krabby_sim_get_duty(ctypes.byref(leg), 0) == pytest.approx(0.0)