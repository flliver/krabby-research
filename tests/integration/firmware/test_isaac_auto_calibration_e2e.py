"""Isaac: ``C`` runs the robot calibration (SEEK, then SWEEP) on the FRONT board's sim joints.

The single FRONT board curls each leg (knee, hip, knee again), then sweeps each hip
and knee one joint at a time between its sim pot limits and saves them; the yaws are
never driven and keep their stops. The hall variant runs the table-safe order on a
quadrature board whose hips and knees are hall encoders (knee parked at -264, hip
swept by count to CAL_HIP_SWEEP_EDGES = 840 and back) and saves canonical strokes in
edges (a hall hip saves its derived CAL_HIP_STROKE_EDGES = 1069, not a measured one).
"""

from __future__ import annotations

import struct

import pytest

pytestmark = pytest.mark.isaacsim

# ActuatorManager::CalData v2: int16 minVals[6], int16 maxVals[6], uint16 magic at 24,
# uint16 sensor bits at 26 (joint j at bits 2j..2j+1: 0 Unknown, 1 Pot, 2 Hall),
# uint16 check at 28 (30 bytes on the Mega and the host build alike).
CAL_MAGIC = 0xCA2B
CAL_DATA_MAGIC_OFFSET = 6 * 2 + 6 * 2
CAL_DATA_SENSOR_BITS_OFFSET = CAL_DATA_MAGIC_OFFSET + 2
# CAL_HIP_STROKE_EDGES (calibration_constants.h): a hall hip's saved max (derived).
CAL_HIP_STROKE_EDGES = 1069
# Isaac counts a joint's whole span as hall_edges_per_full_travel edges. The JSON
# scale (3200) is above CAL_HALL_MAX_STROKE_EDGES (1500), so a knee would fail its
# stroke check; use a span inside 450..1500 and longer than the hip's 840 sweep.
ISAAC_HALL_EDGES_PER_FULL_TRAVEL = 1100
# SEEK curls 6 joints (knee, hip, knee per leg) and SWEEP moves 12 (pot legs; 14 for
# the table-safe hall legs); a hip nudge (with its return) or the repeat after a late
# pot classification adds moves.
MIN_MOVES = 6 + 12
# Filtered pot at a hard stop can lag the band endpoint by a few counts.
ADC_TOLERANCE = 28
# Each move lasts at least ramp + grace + one stall window (~3 s) and a full
# stroke at the default PWM can take tens of seconds; 16 moves on one board.
MAX_CAL_SIM_MS = 30 * 60 * 1000
ROBOT_END_EVENTS = ("CAL ROBOT DONE", "CAL ROBOT FAIL", "CAL ROBOT ABORTED", "CAL ROBOT ERROR")


def _read_usb_line(bridge) -> str:
    chars: list[str] = []
    while bridge.serial_usb_tx_available():
        byte = bridge.serial_usb_tx_read()
        if byte == ord("\n"):
            break
        if byte != ord("\r"):
            chars.append(chr(byte))
    return "".join(chars)


def _drain_usb_tx(bridge) -> list[str]:
    lines: list[str] = []
    while bridge.serial_usb_tx_available():
        line = _read_usb_line(bridge)
        if line:
            lines.append(line)
    return lines


def _is_yaw(actuator_id: str) -> bool:
    return actuator_id.endswith("HY")


def _park_joints_mid_travel(core, robot, config, indices) -> None:
    """Mid-span pose so each hip and knee travels to both sim soft limits."""
    import torch

    from firmware.sim.pot_position import normalize_joint_travel_limits

    with torch.inference_mode():
        limits = robot.data.soft_joint_pos_limits[0]
        pos = robot.data.joint_pos.clone()
        vel = torch.zeros_like(pos)
        for slot in config.slots:
            joint_index = indices[slot.isaac_joint_name]
            lo = limits[joint_index, 0].item()
            hi = limits[joint_index, 1].item()
            lo, hi = normalize_joint_travel_limits(lo, hi)
            pos[0, joint_index] = 0.5 * (lo + hi)
        robot.write_joint_state_to_sim(pos, vel)
        robot.set_joint_velocity_target(vel)
        robot.set_joint_position_target(pos)
    core.reset()
    core.step()


def _run_usb_calibration(core) -> list[str]:
    from firmware.sim.isaac_scene import physics_step_ms

    bridge = core.bridge
    bridge.serial_usb_rx_write(b"C\n")
    log: list[str] = []
    max_steps = MAX_CAL_SIM_MS // max(1, int(physics_step_ms(core.env)))
    for _ in range(max_steps):
        core.step(process_usb_serial=True)
        log.extend(_drain_usb_tx(bridge))
        if any(line.startswith(ROBOT_END_EVENTS) for line in log):
            return log
    tail = log[-8:] if log else []
    pytest.fail(f"calibration did not finish in {MAX_CAL_SIM_MS} ms of sim time; tail={tail!r}")


def test_calibration_sweeps_hips_and_knees_on_sim_pot_limits(front_board_rig):
    core, robot, config, indices = front_board_rig
    bridge = core.bridge

    _park_joints_mid_travel(core, robot, config, indices)
    bridge.firmware_setup()
    _drain_usb_tx(bridge)
    yaw_stops_before = {
        slot.firmware_actuator_id: (
            bridge.actuator_min_stop(slot.firmware_actuator_id),
            bridge.actuator_max_stop(slot.firmware_actuator_id),
        )
        for slot in config.slots
        if _is_yaw(slot.firmware_actuator_id)
    }

    log = _run_usb_calibration(core)
    cal = [line for line in log if line.startswith("CAL ")]
    assert "CAL ROBOT DONE" in cal, cal[-8:]
    assert "CAL ROBOT START pwm=128 passes=SEEK,SWEEP" in cal
    assert cal.index("CAL FRONT PASS_DONE SEEK") < cal.index("CAL FRONT PASS SWEEP pwm=128")
    assert "CAL FRONT PASS_DONE SWEEP saved" in cal
    assert not any(line.startswith("CAL FRONT FAIL") for line in cal)
    moves = [line.split()[3] for line in cal if line.startswith("CAL FRONT MOVE ")]
    assert len(moves) >= MIN_MOVES
    assert not any(_is_yaw(joint) for joint in moves), moves

    magic_bytes = bytes(
        bridge.eeprom_read(i)
        for i in range(CAL_DATA_MAGIC_OFFSET, CAL_DATA_MAGIC_OFFSET + 2)
    )
    assert struct.unpack_from("<H", magic_bytes)[0] == CAL_MAGIC

    saved: dict[str, tuple[int, int]] = {}
    for slot in config.slots:
        aid = slot.firmware_actuator_id
        mn = bridge.actuator_min_stop(aid)
        mx = bridge.actuator_max_stop(aid)
        saved[aid] = (mn, mx)
        if _is_yaw(aid):
            assert (mn, mx) == yaw_stops_before[aid], f"yaw {aid} stops changed"
            continue
        band = config.pot_band_for_slot(slot.slot_index)
        # min = end reached with -PWM (retracted), max = with +PWM (extended).
        assert mn < mx, f"{aid}: minStop={mn} maxStop={mx} band={band}"
        assert abs(mn - band.min_adc) <= ADC_TOLERANCE, f"{aid} minStop {mn} vs {band.min_adc}"
        assert abs(mx - band.max_adc) <= ADC_TOLERANCE, f"{aid} maxStop {mx} vs {band.max_adc}"
        assert f"CAL FRONT JOINT {aid} min={mn} max={mx}" in cal

    for slot in config.slots:
        bridge.set_actuator_stops(slot.firmware_actuator_id, 0, 1023)

    bridge.firmware_reset_loop_state()
    bridge.firmware_setup()
    _drain_usb_tx(bridge)

    for slot in config.slots:
        aid = slot.firmware_actuator_id
        assert (bridge.actuator_min_stop(aid), bridge.actuator_max_stop(aid)) == saved[aid]


def test_hall_robot_calibration_saves_canonical_strokes(front_board_rig):
    """Quadrature board, every hip and knee a hall encoder: strokes are saved in edges."""
    from firmware.sim.robot_core import RobotCore
    from firmware.sim.simulated_robot_config import SimulatedRobotConfig

    from dataclasses import replace

    pot_core, robot, _pot_config, indices = front_board_rig
    config = replace(
        SimulatedRobotConfig.rev3_hall(hall_edges_per_stroke=ISAAC_HALL_EDGES_PER_FULL_TRAVEL),
        hall_edges_per_full_travel=ISAAC_HALL_EDGES_PER_FULL_TRAVEL,
    )
    core = RobotCore.front_board(
        env=pot_core.env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=pot_core.bridge,
        config=config,
        max_joint_speed=2.0,
    )
    bridge = core.bridge

    _park_joints_mid_travel(core, robot, config, indices)
    bridge.firmware_setup()
    _drain_usb_tx(bridge)

    log = _run_usb_calibration(core)
    cal = [line for line in log if line.startswith("CAL ")]
    assert "CAL ROBOT DONE" in cal, cal[-8:]
    assert "CAL FRONT PASS_DONE SWEEP saved" in cal
    assert not any(line.startswith("CAL FRONT FAIL") for line in cal)
    moves = [line.split()[3] for line in cal if line.startswith("CAL FRONT MOVE ")]
    assert len(moves) >= MIN_MOVES
    assert not any(_is_yaw(joint) for joint in moves), moves
    assert any(line.startswith("SENS FRONT ") for line in log)

    record = bytes(bridge.eeprom_read(i) for i in range(CAL_DATA_SENSOR_BITS_OFFSET + 2))
    magic, sensor_bits = struct.unpack_from("<HH", record, CAL_DATA_MAGIC_OFFSET)
    assert magic == CAL_MAGIC
    assert sensor_bits == 0x0A28  # hips and knees Hall, yaws Unknown

    for slot in config.slots:
        aid = slot.firmware_actuator_id
        limits = (bridge.actuator_min_stop(aid), bridge.actuator_max_stop(aid))
        if _is_yaw(aid):
            assert limits == (0, 0), aid
            continue
        stroke = slot.hall_edges_per_stroke
        # Curl end at 0: a hall hip saves its derived stroke (0..+1069), knees
        # their full stroke (-S..0).
        if aid[2:] == "HL":
            expected, suffix = (0, CAL_HIP_STROKE_EDGES), " edges (extend derived from FL, not switch-verified)"
        else:
            expected, suffix = (-stroke, 0), " edges"
        assert limits == expected, aid
        assert f"CAL FRONT JOINT {aid} min={expected[0]} max={expected[1]}{suffix}" in cal
