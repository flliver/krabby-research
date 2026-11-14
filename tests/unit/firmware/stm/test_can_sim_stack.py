"""CAN Isaac stack: ctypes bridge and RX routing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from firmware.sim.can_board_bridge import Leg, StmBoardBridge, find_stm_host_library
from firmware.sim.can_robot_core import leg_motor_from_actuator_id, route_rx_to_legs
from firmware.stm import codec, ids

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def stm_bridge() -> StmBoardBridge:
    lib_path = find_stm_host_library()
    if not lib_path.is_file():
        subprocess.run(
            ["make", "-C", str(REPO_ROOT / "firmware"), "host-stm"],
            check=True,
        )
    return StmBoardBridge(lib_path=lib_path)


def test_leg_motor_from_actuator_id() -> None:
    assert leg_motor_from_actuator_id("FLHY") == (1, 0)
    assert leg_motor_from_actuator_id("RRKL") == (6, 2)


def test_route_rx_set_duty_broadcast() -> None:
    legs = tuple(Leg() for _ in range(6))
    injected: list[tuple[int, int]] = []
    payload = codec.encode_set_duty(codec.set_duty_values_from_grid(((0.5, 0, 0),) + ((0,) * 3,) * 5))[1]

    def inject(leg: Leg, can_id: int, data: bytes) -> None:
        injected.append((legs.index(leg), can_id))

    route_rx_to_legs(ids.set_duty_can_id(), payload, legs, inject)
    assert len(injected) == 6


def test_route_rx_calibrate_one_leg() -> None:
    legs = tuple(Leg() for _ in range(6))
    hits: list[int] = []

    def inject(leg: Leg, can_id: int, data: bytes) -> None:
        hits.append(legs.index(leg))

    route_rx_to_legs(ids.calibrate_can_id(3), b"\x02", legs, inject)
    assert hits == [2]


def test_bridge_set_duty_via_inject(stm_bridge: StmBoardBridge) -> None:
    leg = stm_bridge.create_leg(1)
    payload = codec.set_duty_values_from_grid(((0.4, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    _fid, data = codec.encode_set_duty(payload)
    stm_bridge.inject_rx(leg, ids.set_duty_can_id(), data)
    stm_bridge.set_tick_ms(0)
    stm_bridge.app_tick(leg, 0)
    assert stm_bridge.get_duty(leg, 0) == pytest.approx(0.4)


def test_bridge_last_tx_after_telemetry(stm_bridge: StmBoardBridge) -> None:
    leg = stm_bridge.create_leg(2)
    stm_bridge.set_pot(leg, 0, 1800)
    stm_bridge.set_tick_ms(0)
    stm_bridge.app_tick(leg, 0)
    stm_bridge.app_tick(leg, 50)
    tx = stm_bridge.last_tx(leg)
    assert tx is not None
    frame_id, _payload = tx
    assert frame_id == ids.telemetry_can_id(2)