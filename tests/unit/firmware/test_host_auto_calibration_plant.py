"""Host: actuator calibration (C [SEEK|SWEEP] [pwm]) against a minimal plant (no Isaac).

The FRONT board runs alone, so the leader calibrates only its own six actuators:
a SEEK pass (curl every leg), then a SWEEP pass that measures each hip and knee,
one joint at a time, yaws untouched. The plant is
``firmware.sim.pot_autocal_plant.CalibrationTravelPlant``; every case runs on
three robots:

- ``rev2_pot``: pots, no quadrature halls (the sim default; these goldens predate halls);
- ``rev3_hall``: hall encoders on a quadrature board, as on the robot;
- ``mixed``: quadrature board, FL leg hall, FR leg pot.

Invariants are checked after every firmware loop (the native
``test_host_calibration_timeline`` checks them at every pin write).

A hall hip's SWEEP goes out ``CAL_HIP_SWEEP_EDGES`` (700, about 70%) and back, and
saves [0, 700]; a pot hip and every knee sweep end to end. The robot-like cases
add the measured stored-knee behaviour (``ROBOT_STORED_KNEE``: break-free delay,
spring-out after a nudge, a PWM 200 park deeper than PWM 80 reaches) and the
curled tibia meeting the body at 941 edges of hip travel.
"""

from __future__ import annotations

import dataclasses
import math
import re
import struct

import pytest

from firmware.interfaces.joint_telemetry import ActuatorConnection, JointTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.pot_autocal_plant import (
    HALL_B_REST_HIGH_ADC,
    HALL_B_REST_LOW_ADC,
    ROBOT_STORED_KNEE,
    CalibrationTravelPlant,
)
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

# ActuatorManager::CalData v2 (30 bytes on the Mega and the host build): int16
# minVals[6] at 0, int16 maxVals[6] at 12, uint16 magic at 24, uint16 sensor bits
# at 26 (joint j at bits 2j..2j+1), uint16 check at 28.
CAL_MAGIC = 0xCA2B
CAL_DATA_BYTES = 30
CAL_DATA_MAGIC_OFFSET = 6 * 2 + 6 * 2
CAL_DATA_SENSOR_BITS_OFFSET = CAL_DATA_MAGIC_OFFSET + 2
CAL_DATA_CHECK_OFFSET = CAL_DATA_SENSOR_BITS_OFFSET + 2
CAL_DATA_CHECK_XOR = 0x5A5A
EEPROM_RECORDS_BYTES = 34  # CalData, then the role record at 32
SENSOR_BITS = {"unknown": 0, "pot": 1, "hall": 2}
ADC_TOLERANCE = 4
FIRMWARE_DT_MS = 10
MAX_CAL_TICKS = 30_000
# Firmware actuator order on every board: 0/3 yaw, 1/4 hip, 2/5 knee.
FRONT_ORDER = ("FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL")
YAWS = ("FLHY", "FRHY")
LEGS = (("FLHL", "FLKL"), ("FRHL", "FRKL"))
HIPS_AND_KNEES = ("FLHL", "FLKL", "FRHL", "FRKL")
# Confirmed on the robot: +PWM extends; hips curl by retracting, knees by extending.
CURL_SIGN = {"FLHL": -1, "FRHL": -1, "FLKL": +1, "FRKL": +1}
RAMP_STEP = 5
USAGE_ERROR = "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150"
# CAL_HIP_NUDGE_EDGES and CAL_KNEE_NUDGE_EDGES (calibration_constants.h). The knee
# nudge is the shipped default (CAL_DEFAULT_OPTIONS), so every C command uses it.
HIP_NUDGE_EDGES = 24
KNEE_NUDGE_EDGES = 12
# CAL_HIP_SWEEP_EDGES: a hall hip's fixed SWEEP from its curl end, and its saved max.
CAL_HIP_SWEEP_EDGES = 700
# CAL_HALL_CURL_END_TOLERANCE_EDGES: the knee return may stall this far short of rest.
HALL_CURL_END_TOLERANCE_EDGES = 40
# FLHL's curled tibia met the bottom of the robot at 941 edges of hip travel.
BODY_CONTACT_HIP_EDGES = 941
# Encoder T deadband (ACTUATOR_ENC_ERR_DEADBAND_EDGES); the plant also ignores
# |PWM| below its drive threshold of 20, which the encoder P loop (Kp 1 per edge)
# reaches 20 edges from the target.
ENCODER_T_SETTLE_EDGES = 20
# ACTUATOR_SENS_PERIOD_MS (2000) in loops, with margin.
SENS_PERIOD_TICKS = 2100 // FIRMWARE_DT_MS

CONFIGS = {
    "rev2_pot": SimulatedRobotConfig.rev2_pot,
    "rev3_hall": SimulatedRobotConfig.rev3_hall,
    "mixed": SimulatedRobotConfig.mixed,
}
QUADRATURE_CONFIGS = ("rev3_hall", "mixed")


@pytest.fixture(scope="module")
def host():
    try:
        return load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))


@pytest.fixture(params=list(CONFIGS))
def config(request) -> SimulatedRobotConfig:
    return CONFIGS[request.param]()


@pytest.fixture(params=QUADRATURE_CONFIGS)
def quadrature_config(request) -> SimulatedRobotConfig:
    return CONFIGS[request.param]()


def _drain_usb(host) -> list[str]:
    lines: list[str] = []
    while host.serial_usb_tx_available():
        chars: list[str] = []
        while host.serial_usb_tx_available():
            byte = host.serial_usb_tx_read()
            if byte == ord("\n"):
                break
            if byte != ord("\r"):
                chars.append(chr(byte))
        line = "".join(chars)
        if line:
            lines.append(line)
    return lines


def _is_hall(config, aid: str) -> bool:
    return config.slot_by_firmware_id(aid).is_hall


def _classified_late(config, aid: str) -> bool:
    """A pot on a quadrature board: Unknown until its first move, which then repeats."""
    return config.quadrature and not _is_hall(config, aid)


def _expected_moves(config) -> list[str]:
    seek = []
    sweep = []
    for hip, knee in LEGS:
        for joint in (knee, hip):
            seek.append(f"{joint} curl")
            if _classified_late(config, joint):
                seek.append(f"{joint} curl")
        sweep += [
            f"{knee} curl",
            f"{hip} curl",
            f"{hip} uncurl",
            f"{hip} curl",
            f"{knee} uncurl",
            f"{knee} curl",
        ]
    return seek + sweep


def _hall_stroke(config, aid: str) -> int:
    return config.slot_by_firmware_id(aid).hall_edges_per_stroke


def _expected_limits(config, aid: str) -> tuple[int, int] | None:
    """Saved (min, max); None for a pot, whose limits follow its band."""
    if aid in YAWS:
        # Yaws are never driven; on a quadrature board they stay Unknown with 0/0.
        return (0, 0) if config.quadrature else None
    if not _is_hall(config, aid):
        return None
    if CURL_SIGN[aid] < 0:
        # A hall hip sweeps a fixed CAL_HIP_SWEEP_EDGES from its curl end.
        return (0, CAL_HIP_SWEEP_EDGES)
    # Canonical frame: the curl end is 0, knees retract to -S.
    return (-_hall_stroke(config, aid), 0)


def _joint_suffix(config, aid: str) -> str:
    """Unit suffix of a JOINT line: hall hips are a fixed sweep, other halls a stroke."""
    if not _is_hall(config, aid):
        return ""
    return " edges (fixed sweep)" if CURL_SIGN[aid] < 0 else " edges"


def _expected_sensor_bits(config) -> int:
    bits = 0
    for index, aid in enumerate(FRONT_ORDER):
        if aid in YAWS and config.quadrature:
            kind = "unknown"
        elif _is_hall(config, aid):
            kind = "hall"
        else:
            kind = "pot"
        bits |= SENSOR_BITS[kind] << (2 * index)
    return bits


class _Run:
    """Single FRONT board under calibration, with per-loop drive bookkeeping."""

    def __init__(
        self,
        host,
        config,
        travel: dict[str, float] | None = None,
        **plant_options,
    ) -> None:
        self.host = host
        self.config = config
        self.plant = CalibrationTravelPlant(config, travel=travel, **plant_options)
        # Furthest each hip and knee has been from its deepest curl, in hall edges.
        self.peak_out = {aid: 0.0 for aid in HIPS_AND_KNEES}
        self.log: list[str] = []
        self.violations: list[str] = []
        self.drive = {aid: 0 for aid in FRONT_ORDER}
        self.max_pwm: int | None = None
        self.stopped = False

    @property
    def travel(self) -> dict[str, float]:
        return self.plant.travel

    def _power_on(self) -> None:
        if self.config.quadrature:
            # Cleared by every reset; the joints' kinds are set from it at setup.
            self.host.board_set_hall_quadrature(self.config.board_id, True)

    def boot(self) -> None:
        self.host.reset()
        self._power_on()
        for slot in self.config.slots:
            band = self.config.pot_band_for_slot(slot.slot_index)
            self.host.set_actuator_stops(slot.firmware_actuator_id, band.min_adc, band.max_adc)
        self.plant.sync_inputs(self.host)
        self.host.firmware_setup()
        for _ in range(100):
            self.tick()
        self.log.clear()

    def power_cycle(self) -> None:
        """Reset the board (counts, homes, RAM) and keep its EEPROM."""
        eeprom = [self.host.eeprom_read(i) for i in range(EEPROM_RECORDS_BYTES)]
        # Statics survive the sim reset: wipe the limits so only the load restores them.
        for slot in self.config.slots:
            self.host.set_actuator_stops(slot.firmware_actuator_id, 0, 1023)
        self.host.reset()
        self._power_on()
        for addr, value in enumerate(eeprom):
            self.host.eeprom_update(addr, value)
        self.drive = {aid: 0 for aid in FRONT_ORDER}
        self.plant.sync_inputs(self.host)
        self.host.firmware_setup()

    def tick(self, *, check: bool = True) -> None:
        """One firmware loop; ``check`` applies the calibration drive invariants."""
        self.plant.sync_inputs(self.host)
        self.host.advance_millis(FIRMWARE_DT_MS)
        self.host.firmware_loop_once()
        if check:
            self._check_drives()
        self.plant.advance(self.host)
        for aid in HIPS_AND_KNEES:
            self.peak_out[aid] = max(self.peak_out[aid], self.out_edges(aid))
        self.log.extend(_drain_usb(self.host))

    def out_edges(self, aid: str) -> float:
        """Plant distance from the deepest curl (a hip curls to 0.0, a knee to 1.0)."""
        travel = self.travel[aid]
        out = travel if CURL_SIGN[aid] < 0 else 1.0 - travel
        return out * _hall_stroke(self.config, aid)

    def run_until(self, line: str, max_ticks: int = MAX_CAL_TICKS) -> None:
        for _ in range(max_ticks):
            if line in self.log:
                return
            self.tick()
        if line not in self.log:
            pytest.fail(f"missing {line!r}; log:\n" + "\n".join(self.log[-40:]))

    def _check_drives(self) -> None:
        driven = []
        for slot in self.config.slots:
            aid = slot.firmware_actuator_id
            cmd = self.plant.slot_pwm(self.host, slot)
            previous = self.drive[aid]
            self.drive[aid] = cmd
            if cmd == 0:
                continue
            driven.append(aid)
            if self.stopped:
                self.violations.append(f"{aid} driven after the stop ({cmd})")
            if aid in YAWS:
                self.violations.append(f"yaw {aid} driven ({cmd})")
            if self.max_pwm is not None and abs(cmd) > self.max_pwm:
                self.violations.append(f"{aid} at {cmd}, above {self.max_pwm}")
            if previous == 0 and abs(cmd) > RAMP_STEP:
                self.violations.append(f"{aid} jumped to {cmd} without ramping")
            if previous != 0 and (cmd > 0) != (previous > 0):
                self.violations.append(f"{aid} reversed {previous} -> {cmd}")
        if len(driven) > 1:
            self.violations.append(f"several actuators driven at once: {driven}")

    def cal_lines(self) -> list[str]:
        return [line for line in self.log if line.startswith("CAL ")]

    def eeprom_record(self) -> bytes:
        return bytes(self.host.eeprom_read(i) for i in range(CAL_DATA_BYTES))

    def eeprom_magic(self) -> int:
        return struct.unpack_from("<H", self.eeprom_record(), CAL_DATA_MAGIC_OFFSET)[0]

    def telemetry(self) -> dict[str, JointTelemetry]:
        """Joints of the last FRONT telemetry line in the log."""
        line = next(line for line in reversed(self.log) if line.startswith("FRONT;"))
        return {joint.name: joint for joint in TelemetryFrame.parse_line(line).joints}

    def tick_until_telemetry(self) -> dict[str, JointTelemetry]:
        """Clears the log and runs past one SENS period, so the log holds a SENS line."""
        self.log.clear()
        for _ in range(SENS_PERIOD_TICKS):
            self.tick()
        return self.telemetry()


def _assert_saved_calibration(host, run: _Run, config) -> None:
    record = run.eeprom_record()
    magic, sensor_bits, check = struct.unpack_from("<HHH", record, CAL_DATA_MAGIC_OFFSET)
    assert magic == CAL_MAGIC
    assert sensor_bits == _expected_sensor_bits(config)
    assert check == (sum(record[:CAL_DATA_CHECK_OFFSET]) & 0xFFFF) ^ CAL_DATA_CHECK_XOR
    for slot in config.slots:
        aid = slot.firmware_actuator_id
        limits = (host.actuator_min_stop(aid), host.actuator_max_stop(aid))
        expected = _expected_limits(config, aid)
        if expected is not None:
            assert limits == expected, aid
            continue
        band = config.pot_band_for_slot(slot.slot_index)
        # min = end reached with -PWM (retracted), max = with +PWM (extended);
        # yaws keep the limits they had.
        assert abs(limits[0] - band.min_adc) <= ADC_TOLERANCE, aid
        assert abs(limits[1] - band.max_adc) <= ADC_TOLERANCE, aid


def _full_calibration(host, config, *, wire: bytes = b"C\n", max_pwm: int = 128) -> _Run:
    run = _Run(host, config)
    run.boot()
    run.max_pwm = max_pwm
    host.serial_usb_rx_write(wire)
    run.run_until("CAL ROBOT DONE")
    return run


def test_calibration_curls_then_sweeps_one_joint_at_a_time_and_saves_limits(host, config):
    run = _full_calibration(host, config)

    assert run.violations == []
    lines = run.cal_lines()
    assert lines[0] == "CAL ROBOT START pwm=128 passes=SEEK,SWEEP"
    moves = [line.removeprefix("CAL FRONT MOVE ") for line in lines if " MOVE " in line]
    assert moves == _expected_moves(config)
    order = [
        "CAL FRONT PASS SEEK pwm=128",
        "CAL FRONT PASS_DONE SEEK",
        "CAL ROBOT SKIP LEFT no link",
        "CAL ROBOT SKIP RIGHT no link",
        "CAL FRONT PASS SWEEP pwm=128",
        "CAL FRONT PASS_DONE SWEEP saved",
        "CAL ROBOT DONE",
    ]
    positions = [lines.index(line) for line in order]
    assert positions == sorted(positions)
    assert not any(" WARN " in line or " FAIL " in line for line in lines)
    assert not host.firmware_calibration_active()
    assert all(value == 0 for value in run.drive.values())

    # CAL lines never read as telemetry.
    for line in lines:
        assert not TelemetryFrame.is_telemetry_line(line), line
        assert TelemetryFrame.parse_line(line).joints == ()

    # Kinds: proven on quadrature boards only; the hall joints are homed once, in SEEK.
    sensors = {
        tuple(line.split()[3:5]) for line in lines if line.startswith("CAL FRONT SENSOR ")
    }
    homes = [line.split()[3] for line in lines if line.startswith("CAL FRONT HOME ")]
    if config.quadrature:
        assert sensors == {(aid, config.slot_by_firmware_id(aid).sensor) for aid in HIPS_AND_KNEES}
        assert homes == [aid for aid in ("FLKL", "FLHL", "FRKL", "FRHL") if _is_hall(config, aid)]
    else:
        assert sensors == set()
        assert homes == []
    for aid in HIPS_AND_KNEES:
        mn, mx = host.actuator_min_stop(aid), host.actuator_max_stop(aid)
        assert f"CAL FRONT JOINT {aid} min={mn} max={mx}{_joint_suffix(config, aid)}" in lines
        if _is_hall(config, aid) and CURL_SIGN[aid] < 0:
            # Out CAL_HIP_SWEEP_EDGES plus at most the plant step that crosses it,
            # never on to the hip's end of travel (the plant's travel is continuous:
            # a counted edge is within half an edge of it).
            step_edges = run.plant.step * _hall_stroke(config, aid)
            peak = run.peak_out[aid]
            assert CAL_HIP_SWEEP_EDGES - 0.5 <= peak <= CAL_HIP_SWEEP_EDGES + step_edges, peak
            assert run.peak_out[aid] < _hall_stroke(config, aid)

    _assert_saved_calibration(host, run, config)

    # A power cycle reloads what was saved.
    run.power_cycle()
    run.log.extend(_drain_usb(host))
    assert "Calibration v2 loaded" in run.log
    _assert_saved_calibration(host, run, config)


def test_calibration_at_the_minimum_speed_completes(host, config):
    run = _full_calibration(host, config, wire=b"C 40\n", max_pwm=40)
    assert run.violations == []
    assert "CAL FRONT PASS_DONE SWEEP saved" in run.log
    assert not any(" FAIL " in line for line in run.cal_lines())
    assert run.eeprom_magic() == CAL_MAGIC
    assert struct.unpack_from("<H", run.eeprom_record(), CAL_DATA_SENSOR_BITS_OFFSET)[0] == (
        _expected_sensor_bits(config)
    )


def test_seek_only_curls_every_leg_without_saving(host, config):
    run = _Run(host, config)
    run.boot()
    run.max_pwm = 100
    host.serial_usb_rx_write(b"C SEEK 100\n")
    run.run_until("CAL ROBOT DONE")
    assert run.violations == []
    moves = [line for line in run.cal_lines() if " MOVE " in line]
    seek_moves = len(_expected_moves(config)) - 12
    assert moves == [f"CAL FRONT MOVE {move}" for move in _expected_moves(config)[:seek_moves]]
    for aid, sign in CURL_SIGN.items():
        assert run.travel[aid] == (0.0 if sign < 0 else 1.0), aid
    assert run.eeprom_magic() != CAL_MAGIC


@pytest.mark.parametrize(
    ("wire", "cause"),
    [(b"H\n", "H"), (b"JFLHY 0\n", "J"), (b"B FLHY 0 \n", "B"), (b"T\n", "T")],
)
def test_stop_commands_abort_calibration_and_nothing_moves_after(host, config, wire, cause):
    run = _Run(host, config)
    run.boot()
    run.max_pwm = 128
    host.serial_usb_rx_write(b"C\n")
    run.run_until("CAL FRONT MOVE FLHL curl")
    for _ in range(30):
        run.tick()
    assert run.drive["FLHL"] < 0

    host.serial_usb_rx_write(wire)
    run.stopped = True
    run.tick()
    assert all(value == 0 for value in run.drive.values())
    for _ in range(500):
        run.tick()

    assert run.violations == []
    assert f"CAL FRONT ABORTED {cause}" in run.log
    assert f"CAL ROBOT ABORTED {cause}" in run.log
    assert not host.firmware_calibration_active()
    assert run.eeprom_magic() != CAL_MAGIC


@pytest.mark.parametrize("wire", [b"C 39\n", b"C 151\n", b"C fast\n", b"C SEEK SWEEP\n"])
def test_bad_calibration_requests_are_rejected_and_nothing_moves(host, config, wire):
    run = _Run(host, config)
    run.boot()
    run.stopped = True  # nothing may move at all
    host.serial_usb_rx_write(wire)
    for _ in range(300):
        run.tick()
    assert run.violations == []
    assert run.cal_lines() == [USAGE_ERROR]
    assert not host.firmware_calibration_active()


def test_second_calibrate_while_running_is_busy(host, config):
    run = _Run(host, config)
    run.boot()
    run.max_pwm = 60
    host.serial_usb_rx_write(b"C 60\n")
    run.run_until("CAL FRONT MOVE FLHL curl")
    host.serial_usb_rx_write(b"C 150\n")
    run.run_until("CAL ROBOT BUSY", max_ticks=10)
    run.run_until("CAL ROBOT DONE")
    assert run.violations == []
    assert run.log.count("CAL ROBOT START pwm=60 passes=SEEK,SWEEP") == 1


def _sens_line(states: dict[str, str]) -> str:
    return "SENS FRONT " + " ".join(f"{aid}:{states[aid]}" for aid in FRONT_ORDER)


def test_boot_reports_every_joint_unknown_and_nothing_has_a_position(host, quadrature_config):
    config = quadrature_config
    run = _Run(host, config)
    run.host.reset()
    run._power_on()
    run.plant.sync_inputs(host)
    host.firmware_setup()
    run.log.extend(_drain_usb(host))
    assert "No calibration record: defaults" in run.log
    joints = run.tick_until_telemetry()

    assert _sens_line(dict.fromkeys(FRONT_ORDER, "?")) in run.log
    for aid in FRONT_ORDER:
        joint = joints[aid]
        assert math.isnan(joint.pos), aid
        assert joint.connection_state == ActuatorConnection.DISCONNECTED, aid
        assert joint.saf == 0, aid
        if _is_hall(config, aid):
            # Hall B keeps its pull-up and rests on a rail.
            assert joint.pot in (HALL_B_REST_LOW_ADC, HALL_B_REST_HIGH_ADC), aid


def test_hall_joints_need_seek_after_a_power_cycle_then_hold_a_target(host, quadrature_config):
    """Saved strokes reload unhomed (nan, T refused); SEEK homes them by their nudges."""
    config = quadrature_config
    run = _full_calibration(host, config)
    assert not any(" FAIL " in line for line in run.cal_lines())
    hall_joints = [aid for aid in HIPS_AND_KNEES if _is_hall(config, aid)]
    pot_joints = [aid for aid in HIPS_AND_KNEES if not _is_hall(config, aid)]

    run.power_cycle()
    joints = run.tick_until_telemetry()
    unhomed = {aid: "?" for aid in YAWS}
    unhomed.update({aid: "e" if aid in hall_joints else "P" for aid in HIPS_AND_KNEES})
    assert _sens_line(unhomed) in run.log
    for aid in hall_joints:
        assert math.isnan(joints[aid].pos), aid
        assert joints[aid].connection_state == ActuatorConnection.DISCONNECTED, aid
    for aid in pot_joints:
        assert math.isfinite(joints[aid].pos), aid

    # T to an unhomed encoder is refused and nothing moves. The refusal prints at most
    # once a second, and the sim reset keeps the manager's last refusal time, so the
    # host keeps sending as it would.
    run.log.clear()
    run.stopped = True
    for tick in range(120):
        if tick % 10 == 0:
            host.serial_usb_rx_write(b"T FLHL 0.500\n")
        run.tick()
    assert run.violations == []
    assert "ERR T refused FLHL (no position)" in run.log
    run.stopped = False

    # No jogs: the hall hips and knees stay parked on their curl ends and the default
    # run nudges each one out a little and straight back (the knee nudge is the
    # shipped default), which proves its sensor and homes it.
    nudge_peak = {aid: 0.0 for aid in hall_joints}
    run.log.clear()
    run.max_pwm = 128
    host.serial_usb_rx_write(b"C SEEK\n")
    for _ in range(MAX_CAL_TICKS):
        if "CAL ROBOT DONE" in run.log:
            break
        run.tick()
        for aid in hall_joints:
            # Distance from the curl end (a hip curls to 0.0, a knee to 1.0).
            out = run.travel[aid] if CURL_SIGN[aid] < 0 else 1.0 - run.travel[aid]
            nudge_peak[aid] = max(nudge_peak[aid], out)
    assert "CAL ROBOT DONE" in run.log, run.log[-20:]
    assert run.violations == []
    lines = run.cal_lines()
    assert not any(" FAIL " in line for line in lines)
    moves = [line.removeprefix("CAL FRONT MOVE ") for line in lines if " MOVE " in line]
    expected = []
    for hip, knee in LEGS:
        for joint in (knee, hip):
            if joint in hall_joints:
                expected += [f"{joint} curl", f"{joint} nudge", f"{joint} curl"]
            else:
                expected.append(f"{joint} curl")
    assert moves == expected
    for aid in hall_joints:
        assert any(line.startswith(f"CAL FRONT HOME {aid} count=") for line in lines), aid
        # Out at most CAL_HIP_NUDGE_EDGES (hip) or CAL_KNEE_NUDGE_EDGES (knee), plus
        # the one plant step that crosses it (the plant's travel is continuous: a
        # counted edge is within half an edge of it).
        bound = HIP_NUDGE_EDGES if CURL_SIGN[aid] < 0 else KNEE_NUDGE_EDGES
        nudge_edges = nudge_peak[aid] * _hall_stroke(config, aid)
        step_edges = run.plant.step * _hall_stroke(config, aid)
        assert bound - 0.5 <= nudge_edges <= bound + step_edges, (aid, nudge_edges)

    joints = run.tick_until_telemetry()
    ready = dict(unhomed)
    ready.update({aid: "E" for aid in hall_joints})
    assert _sens_line(ready) in run.log
    for aid in hall_joints:
        # Curl end: a hip reads 0.000, a knee 1.000.
        assert joints[aid].pos == (0.0 if CURL_SIGN[aid] < 0 else 1.0), aid

    # T now holds the hip mid-range, in the hall frame: the middle of its swept
    # [0, CAL_HIP_SWEEP_EDGES], not of the actuator's whole stroke.
    run.log.clear()
    host.serial_usb_rx_write(b"T FLHL 0.500\n")
    for _ in range(1500):
        run.tick(check=False)
    assert not any(line.startswith("ERR ") for line in run.log), run.log
    frame = round(run.travel["FLHL"] * _hall_stroke(config, "FLHL"))
    assert abs(frame - CAL_HIP_SWEEP_EDGES // 2) <= ENCODER_T_SETTLE_EDGES
    assert run.telemetry()["FLHL"].pos == pytest.approx(
        0.5, abs=ENCODER_T_SETTLE_EDGES / CAL_HIP_SWEEP_EDGES
    )


# --- Fixed hip sweep and the stored knee, against the robot's measurements -------

SWEEP_SHORT = re.compile(
    r"^CAL FRONT FAIL FLHL stopped (\d+) edges short of its 700-edge sweep: obstacle\?$"
)


def test_hall_hip_blocked_before_its_sweep_fails_in_place(host):
    """An obstacle at 600 edges: SweepShort names the shortfall; nothing moves after."""
    config = SimulatedRobotConfig.rev3_hall()
    stop = 600 / _hall_stroke(config, "FLHL")
    run = _Run(host, config, extend_stops={"FLHL": stop})
    run.boot()
    run.max_pwm = 128
    host.serial_usb_rx_write(b"C\n")
    run.run_until("CAL FRONT MOVE FLHL uncurl")
    for _ in range(MAX_CAL_TICKS):
        if any(line.startswith("CAL ROBOT FAIL ") for line in run.log):
            break
        run.tick()
    run.stopped = True
    for _ in range(500):
        run.tick()

    assert run.violations == []
    lines = run.cal_lines()
    fails = [line for line in lines if SWEEP_SHORT.match(line)]
    assert len(fails) == 1, lines[-10:]
    short = int(SWEEP_SHORT.match(fails[0]).group(1))
    assert 95 <= short <= 100, fails[0]
    reason = fails[0].removeprefix("CAL FRONT FAIL ")
    assert len(reason) <= 78
    assert f"CAL ROBOT FAIL FRONT: {reason}" in lines
    # Failed at the uncurl: the hip is not driven back and its knee does not move.
    moves = [line.removeprefix("CAL FRONT MOVE ") for line in lines if " MOVE " in line]
    assert moves[-1] == "FLHL uncurl"
    assert run.out_edges("FLHL") == pytest.approx(600, abs=1)
    assert "CAL FRONT PASS_DONE SWEEP saved" not in lines
    assert run.eeprom_magic() != CAL_MAGIC
    assert not host.firmware_calibration_active()


# The plant's stored knee keeps its measured PWM 80 curl end for every curl below
# PWM 150, so these cases run at the default PWM 128 as they did at 80 (the knee's
# curl end at 128 is not measured yet).
# A knee parked by a PWM 200 curl whose spring-out is this small ends its 12-edge
# nudge at (or deeper than) its PWM 80 curl end, 17 edges out, so its curl back
# moves fewer than CAL_KNEE_RETURN_MIN_EDGES (2): it fails "curled back only N
# edges after its nudge" once, rests at its PWM 80 end, and the next run passes.
SHORT_SPRING_MAX_EDGES = 6
NO_CURL_BACK = re.compile(r"^CAL FRONT FAIL (\w+) curled back only ([01]) edges after its nudge$")


def _run_to_robot_end(run: _Run) -> list[str]:
    run.log.clear()
    run.host.serial_usb_rx_write(b"C\n")
    for _ in range(4 * MAX_CAL_TICKS):
        if any(line.startswith(("CAL ROBOT FAIL ", "CAL ROBOT DONE")) for line in run.log):
            break
        run.tick()
    for _ in range(200):  # idle between runs, as an operator would
        run.tick()
    return run.cal_lines()


def _assert_short_spring_knees_fail_once_then_pass(host, run: _Run) -> None:
    knee_hip = {knee: hip for hip, knee in LEGS}
    failed: list[str] = []
    for _ in range(len(LEGS) + 1):
        lines = _run_to_robot_end(run)
        if "CAL ROBOT DONE" in lines:
            break
        fails = [line for line in lines if line.startswith("CAL FRONT FAIL ")]
        assert len(fails) == 1, lines[-10:]
        match = NO_CURL_BACK.match(fails[0])
        assert match, fails[0]
        knee = match.group(1)
        assert knee in knee_hip and knee not in failed, (knee, failed)
        failed.append(knee)
        # It failed before its hip was driven, and it rests near its PWM 80 curl end.
        assert run.peak_out[knee_hip[knee]] == 0.0, knee
        assert run.out_edges(knee) <= ROBOT_STORED_KNEE.weak_curl_end_edges + 1, knee
        assert "CAL FRONT PASS_DONE SEEK" not in lines
    else:
        pytest.fail(f"still failing after {failed}")
    # Each knee failed exactly once; then the full run passed and saved.
    assert sorted(failed) == sorted(knee_hip), failed
    assert run.violations == []
    assert not any(" FAIL " in line or " WARN " in line for line in lines), lines
    assert run.eeprom_magic() == CAL_MAGIC
    for hip, _knee in LEGS:
        assert (host.actuator_min_stop(hip), host.actuator_max_stop(hip)) == (0, CAL_HIP_SWEEP_EDGES)
        assert run.peak_out[hip] < BODY_CONTACT_HIP_EDGES - 200, hip


@pytest.mark.parametrize("spring", range(5, 10))
@pytest.mark.parametrize("park", ["pwm200", "pwm80"])
def test_robot_like_stored_legs_calibrate_with_the_knee_spring(host, park, spring):
    """Measured on FLHL/FLKL: a stored knee breaks free late and springs out 5-9
    edges after its nudge; one parked by a PWM 200 curl comes back only to the PWM 80
    end. The full run passes (after one "curled back only" retry per knee at the low
    end of the spring with a PWM 200 park), and no hip reaches the body (941 edges)."""
    config = SimulatedRobotConfig.rev3_hall()
    knee_model = dataclasses.replace(ROBOT_STORED_KNEE, spring_edges=spring)
    knee_rest_out = 0 if park == "pwm200" else knee_model.weak_curl_end_edges
    travel = {}
    for hip, knee in LEGS:
        travel[hip] = 0.0
        travel[knee] = 1.0 - knee_rest_out / _hall_stroke(config, knee)
    body = {hip: BODY_CONTACT_HIP_EDGES / _hall_stroke(config, hip) for hip, _ in LEGS}
    run = _Run(host, config, travel=travel, knee_model=knee_model, extend_stops=body)
    run.boot()
    run.max_pwm = 128
    if park == "pwm200" and spring <= SHORT_SPRING_MAX_EDGES:
        _assert_short_spring_knees_fail_once_then_pass(host, run)
        return
    host.serial_usb_rx_write(b"C\n")
    run.run_until("CAL FRONT PASS_DONE SEEK")
    seek_peak = dict(run.peak_out)
    seek_end = {aid: run.out_edges(aid) for aid in HIPS_AND_KNEES}
    run.run_until("CAL ROBOT DONE")

    assert run.violations == []
    lines = run.cal_lines()
    assert not any(" FAIL " in line or " WARN " in line for line in lines), lines
    moves = [line.removeprefix("CAL FRONT MOVE ") for line in lines if " MOVE " in line]
    seek = []
    for hip, knee in LEGS:
        for joint in (knee, hip):
            seek += [f"{joint} curl", f"{joint} nudge", f"{joint} curl"]
    assert moves == seek + _expected_moves(config)[-12:]
    for hip, knee in LEGS:
        step_edges = run.plant.step * _hall_stroke(config, knee)
        # The 12-edge nudge, then the accepted spring-out: about 18-21 edges out (the
        # plant's travel is continuous; a counted edge is within half an edge of it).
        nudge_out = seek_peak[knee] - knee_rest_out
        spring = knee_model.spring_edges
        low, high = KNEE_NUDGE_EDGES + spring - 0.5, KNEE_NUDGE_EDGES + spring + step_edges + 0.5
        assert low <= nudge_out <= high, (knee, nudge_out)
        # The return stalls at the PWM 80 curl end, within the tolerance of rest: a
        # knee parked by a PWM 200 curl comes back only about 4-5 edges.
        short_of_rest = seek_end[knee] - knee_rest_out
        assert short_of_rest <= HALL_CURL_END_TOLERANCE_EDGES, knee
        expected_short = knee_model.weak_curl_end_edges if park == "pwm200" else 0
        assert short_of_rest == pytest.approx(expected_short, abs=1), knee
        # The hip went out its fixed sweep, short of the body.
        assert run.peak_out[hip] < BODY_CONTACT_HIP_EDGES - 200, hip
        assert seek_end[hip] == pytest.approx(0.0, abs=1), hip
    assert run.eeprom_magic() == CAL_MAGIC
    for hip, knee in LEGS:
        assert (host.actuator_min_stop(hip), host.actuator_max_stop(hip)) == (
            0,
            CAL_HIP_SWEEP_EDGES,
        )
        assert f"CAL FRONT JOINT {hip} min=0 max={CAL_HIP_SWEEP_EDGES} edges (fixed sweep)" in lines
        knee_min, knee_max = host.actuator_min_stop(knee), host.actuator_max_stop(knee)
        stroke = _hall_stroke(config, knee)
        assert knee_max == 0
        assert stroke - knee_model.weak_curl_end_edges - 2 <= -knee_min <= stroke
        assert f"CAL FRONT JOINT {knee} min={knee_min} max=0 edges" in lines
