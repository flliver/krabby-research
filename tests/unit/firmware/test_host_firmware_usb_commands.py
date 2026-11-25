"""Host firmware loop USB dispatch using the same wire bytes as KrabbyMCUSDK."""

import pytest

from firmware.krabby_mcu import parse_ver_reply
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


@pytest.fixture(scope="module")
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return bridge


def _drain_usb_tx(host) -> None:
    while host.serial_usb_tx_available():
        host.serial_usb_tx_read()


def _read_usb_line(host) -> str:
    chars: list[str] = []
    while host.serial_usb_tx_available():
        byte = host.serial_usb_tx_read()
        if byte == ord("\n"):
            break
        if byte != ord("\r"):
            chars.append(chr(byte))
    return "".join(chars)


def _wire_joints(cmds_by_joint: dict[str, float]) -> bytes:
    parts = ["T"]
    for name, raw_val in cmds_by_joint.items():
        val = max(0.0, min(1.0, raw_val))
        parts.append(name)
        parts.append(f"{val:.3f}")
    return (" ".join(parts) + "\n").encode("utf-8")


def _wire_jog_batch(cmds_by_joint: dict[str, int]) -> bytes:
    parts = ["B"]
    for name, raw_pwm in cmds_by_joint.items():
        pwm = max(-255, min(255, int(raw_pwm)))
        parts.append(name)
        parts.append(str(pwm))
    return (" ".join(parts) + " \n").encode("utf-8")


def _wire_jog_single(joint_name: str, pwm: int) -> bytes:
    pwm = max(-255, min(255, int(pwm)))
    return f"J{joint_name} {pwm}\n".encode("utf-8")


def _loop(host, wire: bytes) -> None:
    host.reset()
    host.serial_usb_rx_write(wire)
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()


@pytest.mark.parametrize(
    ("wire_fn", "joint", "pwm", "en_high"),
    [
        (lambda: _wire_jog_single("FLHY", 100), "FLHY", 100, True),
        (lambda: _wire_jog_batch({"FLHY": -80}), "FLHY", -80, True),
    ],
)
def test_mcu_emitter_jog_wires(host, wire_fn, joint, pwm, en_high):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id(joint)
    _, _, pin_en = slot.drive_pins
    _loop(host, wire_fn())
    assert host.pin_digital(pin_en) == (1 if en_high else 0)


def test_mcu_emitter_joints_wire_drives_position_target(host):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    pwm_r, _pwm_l, pin_en = slot.drive_pins
    pot_pin = slot.pin_pot

    host.reset()
    host.set_actuator_stops("FLHY", 100, 900)
    host.set_analog_read(pot_pin, 100)
    host.serial_usb_rx_write(_wire_joints({"FLHY": 1.0}))
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()
    for _ in range(8):
        host.advance_millis(10)
        host.firmware_loop_once()

    assert host.pin_digital(pin_en) == 1
    assert host.pin_pwm(pwm_r) > 0


def test_mcu_emitter_calibrate_command(host):
    _loop(host, b"C\n")
    assert _read_usb_line(host) == "CAL ROBOT START pwm=128 passes=SEEK,SWEEP"
    assert _read_usb_line(host) == "CAL FRONT PASS SEEK pwm=128"
    assert _read_usb_line(host) == "CAL FRONT MOVE FLKL curl"
    assert host.firmware_calibration_active()

    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    assert not host.firmware_calibration_active()


@pytest.mark.parametrize(
    ("wire", "start"),
    [
        (b"C 120\n", "CAL ROBOT START pwm=120 passes=SEEK,SWEEP"),
        (b"C SEEK\n", "CAL ROBOT START pwm=128 passes=SEEK"),
        (b"C SWEEP 110\n", "CAL ROBOT START pwm=110 passes=SWEEP"),
        (b"C 150\n", "CAL ROBOT START pwm=150 passes=SEEK,SWEEP"),
        # PWM 110-150 only (owner decision F4): 40 and 80 stall mid-stroke.
        (b"C 109\n", "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150"),
        (b"C SWEEP 80\n", "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150"),
        (b"C 40\n", "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150"),
        (b"C 151\n", "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150"),
    ],
)
def test_mcu_emitter_calibrate_command_forms(host, wire, start):
    _loop(host, wire)
    assert _read_usb_line(host) == start
    assert host.firmware_calibration_active() == (" START " in start)


def test_mcu_emitter_hold_command(host):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    _, _, pin_en = slot.drive_pins

    _loop(host, _wire_jog_single("FLHY", 100))
    assert host.pin_digital(pin_en) == 1

    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    assert host.pin_digital(pin_en) == 0


def _usb_lines(host) -> list[str]:
    lines = []
    while host.serial_usb_tx_available():
        line = _read_usb_line(host)
        if line:
            lines.append(line)
    return lines


def _quadrature_setup(host, config: SimulatedRobotConfig) -> None:
    host.reset()
    host.board_set_hall_quadrature(config.board_id, config.quadrature)
    host.firmware_setup()
    _drain_usb_tx(host)


@pytest.mark.parametrize(
    "factory",
    [SimulatedRobotConfig.rev3_hall, SimulatedRobotConfig.mixed],
    ids=["rev3_hall", "mixed"],
)
def test_t_to_joints_without_a_position_is_refused_and_nothing_moves(host, factory):
    """Quadrature board after boot: every joint is Unknown, so T names them in one ERR line."""
    config = factory()
    _quadrature_setup(host, config)
    lines: list[str] = []
    # At most one refusal line a second; the host repeats T as it streams.
    for tick in range(120):
        if tick % 10 == 0:
            host.serial_usb_rx_write(_wire_joints({"FLHY": 0.5, "FLHL": 0.5, "FRKL": 0.25}))
        host.advance_millis(10)
        host.firmware_loop_once()
        lines += _usb_lines(host)
        for slot in config.slots:
            assert host.pin_digital(slot.pin_en) == 0, slot.firmware_actuator_id
    refused = [line for line in lines if line.startswith("ERR T refused")]
    assert refused, lines[-10:]
    assert set(refused) == {"ERR T refused FLHY FLHL FRKL (no position)"}
    assert len(refused) <= 2


def test_t_on_a_board_without_quadrature_is_not_refused(host):
    config = SimulatedRobotConfig.rev2_pot()
    host.reset()
    for slot in config.slots:
        host.set_actuator_stops(slot.firmware_actuator_id, 100, 900)
        host.set_analog_read(slot.pin_pot, 500)
    host.firmware_setup()
    _drain_usb_tx(host)
    lines: list[str] = []
    host.serial_usb_rx_write(_wire_joints({"FLHL": 0.9}))
    # Under the T stall guard's 1 s window: the static pot would trip it after that.
    for _ in range(30):
        host.advance_millis(10)
        host.firmware_loop_once()
        lines += _usb_lines(host)
    assert not any(line.startswith(("ERR ", "SENS ")) for line in lines)
    assert host.pin_digital(config.slot_by_firmware_id("FLHL").pin_en) == 1


@pytest.mark.parametrize(
    "factory",
    [SimulatedRobotConfig.rev2_pot, SimulatedRobotConfig.rev3_hall, SimulatedRobotConfig.mixed],
    ids=["rev2_pot", "rev3_hall", "mixed"],
)
def test_mcu_emitter_calibrate_command_on_every_robot(host, factory):
    config = factory()
    _quadrature_setup(host, config)
    host.serial_usb_rx_write(b"C\n")
    host.firmware_loop_once()
    cal = [line for line in _usb_lines(host) if line.startswith("CAL ")]
    assert cal[:3] == [
        "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
        "CAL FRONT PASS SEEK pwm=128",
        "CAL FRONT MOVE FLKL curl",
    ]
    assert host.firmware_calibration_active()
    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    assert not host.firmware_calibration_active()


def test_mcu_emitter_version_wire(host):
    _loop(host, b"V\n")
    line = _read_usb_line(host)
    parsed = parse_ver_reply(line)
    assert parsed is not None
    assert len(parsed) == 3
    assert parsed[0][0] != "-"
