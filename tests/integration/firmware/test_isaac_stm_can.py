"""Isaac + virtual CAN: SetDuty isolation, sensor readback, calibrate flags (leg 1)."""

from __future__ import annotations

import math
import struct

import pytest

from firmware.sim.isaac_scene import physics_step_ms
from firmware.sim.plant_jog_expectations import (
    max_zero_pwm_joint_drift_rad,
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


def step_and_tally_frames(
    core,
    bus,
    steps: int,
    *,
    watch_ids: frozenset[int],
) -> tuple[dict[int, int], list[int], dict | None]:
    """Run Isaac steps and count selected CAN RX frames; sample leg 1 J1_POT from telemetry."""
    counts: dict[int, int] = {}
    pots: list[int] = []
    latest_telem: dict | None = None
    telem_id = ids.telemetry_can_id(1)
    for _ in range(steps):
        core.step()
        while True:
            msg = bus.recv(timeout=0.0)
            if msg is None:
                break
            aid = int(msg.arbitration_id)
            if aid not in watch_ids:
                continue
            counts[aid] = counts.get(aid, 0) + 1
            if aid == telem_id:
                decoded = codec.decode_telemetry(1, bytes(msg.data))
                pots.append(int(decoded["J1_POT"]))
                latest_telem = decoded
    return counts, pots, latest_telem


def _float_position(raw: int | float) -> float:
    if isinstance(raw, int):
        return struct.unpack("<d", struct.pack("<Q", raw & ((1 << 64) - 1)))[0]
    return float(raw)


def _wait_for_joint_calibrated(
    core,
    bus,
    leg_id: int,
    joint_id: int,
    *,
    max_steps: int = CAL_MAX_STEPS,
) -> dict:
    """Wait for stall cal to finish; require a calibrating phase (session legs may already be calibrated)."""
    telem = None
    saw_calibrating = False
    for _ in range(max_steps):
        core.step()
        telem = drain_leg_telemetry(bus, leg_id)
        if telem is None:
            continue
        if telem[f"J{joint_id}_CALIBRATING"] == 1:
            saw_calibrating = True
        if not saw_calibrating:
            continue
        if telem[f"J{joint_id}_CALIBRATING"] == 0 and telem[f"J{joint_id}_CALIBRATED"] == 1:
            return telem
    raise AssertionError(f"leg {leg_id} joint {joint_id} calibrate did not finish (calibrated bit)")


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
    dt_ms = max(1, physics_step_ms(core.env))
    telem_id = ids.telemetry_can_id(1)
    info_id = ids.board_info_can_id(1)
    watch = frozenset({telem_id, info_id})
    counts, pots, telem_fwd = step_and_tally_frames(core, bus, DRIVE_STEPS, watch_ids=watch)
    sim_ms = DRIVE_STEPS * dt_ms
    min_telem_frames = max(1, sim_ms // 50 - 2)
    assert counts.get(telem_id, 0) >= min_telem_frames
    assert len(pots) >= 2 and pots[-1] != pots[0]
    if sim_ms < 1050:
        extra_steps = (1050 - sim_ms + dt_ms - 1) // dt_ms
        extra_counts, _, telem_latest = step_and_tally_frames(
            core, bus, extra_steps, watch_ids=frozenset({info_id, telem_id})
        )
        assert counts.get(info_id, 0) + extra_counts.get(info_id, 0) >= 1
        if telem_latest is not None:
            telem_fwd = telem_latest
    else:
        assert counts.get(info_id, 0) >= 1
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
    max_drift = max_zero_pwm_joint_drift_rad(robot, "FLHY")
    assert abs(robot.data.joint_pos[0, flhy_idx].item() - stop_pos) < max_drift


def test_calibrate_leg1_hip_flags_and_partial_drive(stm_can_rig) -> None:
    core, robot, bus, indices, _bridge = stm_can_rig
    flhl_idx = indices[ThreeBoardSimConfigs.rev3().slot_by_firmware_id("FLHL").isaac_joint_name]
    _park_stm(core, robot, bus)

    send_calibrate(bus, 1, 2)
    telem = _wait_for_joint_calibrated(core, bus, 1, 2)

    pos = _float_position(telem["J2_POSITION"])
    assert pos == pytest.approx(0.0, abs=0.2)

    send_set_duty(bus, 1, 0.0, 0.5, 0.0)
    hip_before = robot.data.joint_pos[0, flhl_idx].item()
    for _ in range(DRIVE_STEPS):
        core.step()
    hip_after = robot.data.joint_pos[0, flhl_idx].item()
    min_travel = min_commanded_jog_travel_rad(robot)
    assert abs(hip_after - hip_before) > min_travel

    telem_mid = drain_leg_telemetry(bus, 1)
    assert telem_mid is not None
    assert telem_mid["J2_CALIBRATED"] == 1


def test_calibrate_leg1_hip_position_ramp_and_restart(stm_can_rig) -> None:
    core, robot, bus, indices, bridge = stm_can_rig
    flhl_idx = indices[ThreeBoardSimConfigs.rev3().slot_by_firmware_id("FLHL").isaac_joint_name]
    _park_stm(core, robot, bus)

    send_calibrate(bus, 1, 2)
    telem = _wait_for_joint_calibrated(core, bus, 1, 2)

    pos_retracted = _float_position(telem["J2_POSITION"])
    assert pos_retracted == pytest.approx(0.0, abs=0.2)

    send_set_duty(bus, 1, 0.0, 0.5, 0.0)
    hip_before = robot.data.joint_pos[0, flhl_idx].item()
    for _ in range(DRIVE_STEPS):
        core.step()
    assert abs(robot.data.joint_pos[0, flhl_idx].item() - hip_before) > min_commanded_jog_travel_rad(robot)

    telem_mid = drain_leg_telemetry(bus, 1)
    assert telem_mid is not None
    pos_mid = _float_position(telem_mid["J2_POSITION"])
    assert pos_mid > 0.05
    assert pos_mid < 1.0
    assert telem_mid["J2_CALIBRATED"] == 1

    send_zero_duty(bus)
    for _ in range(SETTLE_STEPS):
        core.step()

    leg1 = core.legs[0]
    pos_before_restart = pos_mid
    bridge.restart_leg_app(leg1)
    for _ in range(SETTLE_STEPS):
        core.step()

    telem_after = drain_leg_telemetry(bus, 1)
    assert telem_after is not None
    assert telem_after["J2_CALIBRATED"] == 1
    pos_after = _float_position(telem_after["J2_POSITION"])
    assert not math.isnan(pos_after)
    assert pos_after == pytest.approx(pos_before_restart, abs=0.25)