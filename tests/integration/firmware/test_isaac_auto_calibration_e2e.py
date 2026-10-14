"""Isaac: ``C`` auto-cal runs ``updateCalibration`` stop-to-stop on sim joint limits."""

from __future__ import annotations

import struct

import pytest

pytestmark = pytest.mark.isaacsim

CAL_MAGIC = 0xDEADBEEF
# ActuatorManager::CalData: minVals[6], maxVals[6], magic (see actuator_manager.h).
CAL_DATA_MAGIC_OFFSET = 6 * 4 + 6 * 4
# Filtered pot at a hard stop can lag the band endpoint by a few counts.
ADC_TOLERANCE = 28
MAX_CAL_STEPS = 12_000


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


def _park_joints_mid_travel(core, robot, config, indices) -> None:
    """Mid-span pose so each joint can reach both sim soft limits during auto-cal."""
    import torch

    with torch.inference_mode():
        limits = robot.data.soft_joint_pos_limits[0]
        pos = robot.data.joint_pos.clone()
        vel = torch.zeros_like(pos)
        for slot in config.slots:
            joint_index = indices[slot.isaac_joint_name]
            lo = limits[joint_index, 0].item()
            hi = limits[joint_index, 1].item()
            pos[0, joint_index] = 0.5 * (lo + hi)
        robot.write_joint_state_to_sim(pos, vel)
        robot.set_joint_velocity_target(vel)
        robot.set_joint_position_target(pos)
    core.reset()
    core.step()


def _run_usb_auto_cal(core) -> list[str]:
    bridge = core.bridge
    bridge.serial_usb_rx_write(b"C\n")
    log: list[str] = []
    for _ in range(MAX_CAL_STEPS):
        core.step(process_usb_serial=True)
        log.extend(_drain_usb_tx(bridge))
        if any("CALIBRATION COMPLETE" in line for line in log):
            return log
    tail = log[-8:] if log else []
    pytest.fail(f"auto-cal did not finish in {MAX_CAL_STEPS} steps; tail={tail!r}")


def test_auto_calibration_stop_to_stop_on_sim_pot_limits(front_board_rig):
    core, robot, config, indices = front_board_rig
    bridge = core.bridge

    _park_joints_mid_travel(core, robot, config, indices)
    bridge.firmware_setup()
    _drain_usb_tx(bridge)

    log = _run_usb_auto_cal(core)
    assert any("Starting Auto-Calibration" in line for line in log)
    assert any("Limits saved to EEPROM" in line for line in log)

    magic_bytes = bytes(
        bridge.eeprom_read(i)
        for i in range(CAL_DATA_MAGIC_OFFSET, CAL_DATA_MAGIC_OFFSET + 4)
    )
    assert struct.unpack_from("<I", magic_bytes)[0] == CAL_MAGIC

    saved: dict[str, tuple[int, int]] = {}
    mins_seen: set[int] = set()
    for slot in config.slots:
        band = config.pot_band_for_slot(slot.slot_index)
        aid = slot.firmware_actuator_id
        mn = bridge.actuator_min_stop(aid)
        mx = bridge.actuator_max_stop(aid)
        saved[aid] = (mn, mx)
        mins_seen.add(mn)
        assert mn < mx, f"{aid}: minStop={mn} maxStop={mx} band={band}"
        assert abs(mn - band.min_adc) <= ADC_TOLERANCE, f"{aid} minStop {mn} vs {band.min_adc}"
        assert abs(mx - band.max_adc) <= ADC_TOLERANCE, f"{aid} maxStop {mx} vs {band.max_adc}"

    assert len(mins_seen) == len(config.slots)

    for slot in config.slots:
        bridge.set_actuator_stops(slot.firmware_actuator_id, 0, 1023)

    bridge.firmware_reset_loop_state()
    bridge.firmware_setup()
    _drain_usb_tx(bridge)

    for slot in config.slots:
        aid = slot.firmware_actuator_id
        expected = saved[aid]
        assert bridge.actuator_min_stop(aid) == expected[0]
        assert bridge.actuator_max_stop(aid) == expected[1]
