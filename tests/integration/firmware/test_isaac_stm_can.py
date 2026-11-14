"""Isaac + virtual CAN: SetDuty isolation, sensor readback, calibrate flags (leg 1)."""

from __future__ import annotations

import math
import struct

import pytest

from firmware.sim.plant_jog_expectations import (
    min_commanded_jog_travel_rad,
    skip_plant_jog_coupling_check,
)
from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
from firmware.stm import codec, ids
from firmware.stm.can_transport import send_fd_frame

pytestmark = pytest.mark.isaacsim

DRIVE_STEPS = 90
SETTLE_STEPS = 25
OTHER_JOINT_RATIO = 4
CAL_MAX_STEPS = 600


def _duty_grid_for_leg(leg_id: int, y: float, h: float, k: float) -> tuple[tuple[float, float, float], ...]:
    rows: list[tuple[float, float, float]] = []
    for lid in ids.LEG_IDS:
        rows.append((y, h, k) if lid == leg_id else (0.0, 0.0, 0.0))
    return tuple(rows)


def send_set_duty(bus, leg_id: int, y: float, h: float, k: float) -> None:
    grid = _duty_grid_for_leg(leg_id, y, h, k)
    frame_id, payload = codec.encode_set_duty(codec.set_duty_values_from_grid(grid))
    send_fd_frame(bus, frame_id, payload)


def send_zero_duty(bus) -> None:
    frame_id, payload = codec.encode_set_duty(codec.set_duty_values_from_grid(((0.0, 0.0, 0.0),) * 6))
    send_fd_frame(bus, frame_id, payload)


def send_calibrate(bus, leg_id: int, joint_id: int) -> None:
    frame_id, payload = codec.encode_calibrate(leg_id, joint_id)
    send_fd_frame(bus, frame_id, payload)


def drain_leg_telemetry(bus, leg_id: int) -> dict | None:
    target = ids.telemetry_can_id(leg_id)
    latest = None
    while True:
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break
        if int(msg.arbitration_id) == target:
            latest = codec.decode_telemetry(leg_id, bytes(msg.data))
    return latest


def run_steps(core, bus, steps: int) -> None:
    for _ in range(steps):
        core.step()
    drain_leg_telemetry(bus, 1)


def _float_position(raw: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", raw & ((1 << 64) - 1)))[0]


def _isaac_index_for_flhy(indices: dict[str, int]) -> int:
    slot = ThreeBoardSimConfigs.rev3().slot_by_firmware_id("FLHY")
    return indices[slot.isaac_joint_name]


def _park_stm(core, robot, bus) -> None:
    import torch

    send_zero_duty(bus)
    with torch.inference_mode():
        pos = robot.data.default_joint_pos.clone()
        vel = robot.data.default_joint_vel.clone()
        robot.write_joint_state_to_sim(pos, vel)
        robot.set_joint_velocity_target(vel)
        robot.set_joint_position_target(pos)
    core.reset()
    for _ in range(SETTLE_STEPS):
        core.step()


def test_set_duty_leg1_yaw_only_and_sensor_readback(stm_can_rig) -> None:
    core, robot, bus, indices, _bridge = stm_can_rig
    flhy_idx = _isaac_index_for_flhy(indices)
    _park_stm(core, robot, bus)

    telem_before = drain_leg_telemetry(bus, 1)
    pos_before = robot.data.joint_pos[0, flhy_idx].item()
    pot_before = telem_before["J1_POT"] if telem_before else 0
    hall_before = telem_before["J1_HALL"] if telem_before else 0

    send_set_duty(bus, 1, 1.0, 0.0, 0.0)
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}
    run_steps(core, bus, DRIVE_STEPS)
    after = {name: robot.data.joint_pos[0, idx].item() - before[name] for name, idx in indices.items()}

    flhy_name = ThreeBoardSimConfigs.rev3().slot_by_firmware_id("FLHY").isaac_joint_name
    cmd_travel = after[flhy_name]
    min_travel = min_commanded_jog_travel_rad(robot)
    assert abs(cmd_travel) > min_travel, f"FLHY did not move: {after}"

    for name, delta in after.items():
        if name == flhy_name:
            continue
        if skip_plant_jog_coupling_check(robot, flhy_name, name):
            continue
        assert abs(delta) * OTHER_JOINT_RATIO < abs(cmd_travel), (
            f"duty on leg 1 yaw moved {name}: {after}"
        )

    assert robot.data.joint_pos[0, flhy_idx].item() > pos_before
    telem_fwd = drain_leg_telemetry(bus, 1)
    assert telem_fwd is not None
    assert telem_fwd["J1_POT"] > pot_before
    assert telem_fwd["J1_HALL"] > hall_before

    send_set_duty(bus, 1, -1.0, 0.0, 0.0)
    mid_pos = robot.data.joint_pos[0, flhy_idx].item()
    run_steps(core, bus, DRIVE_STEPS)
    assert robot.data.joint_pos[0, flhy_idx].item() < mid_pos

    telem_rev = drain_leg_telemetry(bus, 1)
    assert telem_rev is not None
    assert telem_rev["J1_HALL"] < telem_fwd["J1_HALL"]

    send_set_duty(bus, 1, 0.0, 0.0, 0.0)
    stop_pos = robot.data.joint_pos[0, flhy_idx].item()
    run_steps(core, bus, SETTLE_STEPS)
    assert abs(robot.data.joint_pos[0, flhy_idx].item() - stop_pos) < min_travel


def test_calibrate_leg1_hip_flags_and_partial_drive(stm_can_rig) -> None:
    core, robot, bus, indices, _bridge = stm_can_rig
    flhl_idx = indices[ThreeBoardSimConfigs.rev3().slot_by_firmware_id("FLHL").isaac_joint_name]
    _park_stm(core, robot, bus)

    send_calibrate(bus, 1, 2)
    calibrated = False
    for _ in range(CAL_MAX_STEPS):
        core.step()
        telem = drain_leg_telemetry(bus, 1)
        if telem is None:
            continue
        if telem["J2_CALIBRATING"] == 0 and telem["J2_CALIBRATED"] == 1:
            calibrated = True
            break
    assert calibrated, "joint 2 calibrate did not finish (calibrated bit)"

    pos_raw = telem["J2_POSITION"] if telem else 0
    pos = _float_position(pos_raw)
    assert pos == pytest.approx(0.0) or math.isnan(pos)

    send_set_duty(bus, 1, 0.0, 0.5, 0.0)
    hip_before = robot.data.joint_pos[0, flhl_idx].item()
    run_steps(core, bus, DRIVE_STEPS)
    hip_after = robot.data.joint_pos[0, flhl_idx].item()
    min_travel = min_commanded_jog_travel_rad(robot)
    assert abs(hip_after - hip_before) > min_travel

    telem_mid = drain_leg_telemetry(bus, 1)
    assert telem_mid is not None
    assert telem_mid["J2_CALIBRATED"] == 1