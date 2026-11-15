"""Unit tests for firmware.gui.stm_adapter (mocked CAN SDK)."""

from __future__ import annotations

from unittest import mock

import pytest

from firmware.gui.stm_adapter import (
    KrabbyStmGuiAdapter,
    _GUI_JOINT_MAP,
    _duty_from_pwm,
)
from firmware.gui.__main__ import build_parser, parse_gui_args
from firmware.gui.app import create_mcu_sdk
from firmware.gui.stm_adapter import KrabbyStmGuiAdapter
from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.stm.mcu_can import KrabbyMCUCANSDK
from tests.unit.firmware.stm.stm_sdk_virtual_peer import (
    create_stm_sdk_virtual_peer,
    wait_with_peer,
)


def test_cli_help_lists_stm_and_virtual_flags() -> None:
    parser = build_parser()
    for action in parser._actions:
        opts = action.option_strings
        if "--stm" in opts:
            break
    else:
        raise AssertionError("--stm not in argparse")
    for action in parser._actions:
        opts = action.option_strings
        if "--virtual" in opts:
            break
    else:
        raise AssertionError("--virtual not in argparse")


def test_cli_virtual_requires_stm() -> None:
    with pytest.raises(SystemExit):
        parse_gui_args(["--virtual"])


def test_app_stm_mode_uses_adapter() -> None:
    mcu = create_mcu_sdk(stm=True, virtual=True)
    assert isinstance(mcu, KrabbyStmGuiAdapter)


def test_default_create_mcu_sdk_uses_mega_serial() -> None:
    mcu = create_mcu_sdk(stm=False)
    assert isinstance(mcu, KrabbyMCUSDK)


def test_gui_adapter_virtual_can_jog_and_joint_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = create_stm_sdk_virtual_peer()
    monkeypatch.setattr(
        "firmware.stm.can_transport.open_socketcan_fd",
        lambda _channel, **_kw: peer.host_bus,
    )
    adapter = KrabbyStmGuiAdapter(virtual=True)
    adapter.connect()
    try:
        pwm = 200
        expected_duty = pwm / 255.0
        adapter.send_command_jog("FLHY", pwm)
        wait_with_peer(
            lambda: abs(peer.leg_duty(1, 0) - expected_duty) < 0.05,
            peer,
        )

        def flhy_telemetry_ready() -> bool:
            jt = adapter.joints.get("FLHY")
            return jt is not None and "J1_POT" in (adapter._sdk.telemetry.get(1) or {})  # noqa: SLF001

        wait_with_peer(flhy_telemetry_ready, peer)
        jt = adapter.joints["FLHY"]
        assert jt is not None
        assert jt.pot >= 0
        assert jt.saf is not None
    finally:
        adapter.close()
        peer.close()


def test_calibrate_joint_two_calls_sdk() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.send_calibration = mock.Mock()  # type: ignore[method-assign]
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    adapter.send_calibration_for_joint("FLHL")
    sdk.send_calibration.assert_called_once_with(1, 2)


def test_calibrate_rejects_joint_one() -> None:
    adapter = KrabbyStmGuiAdapter(sdk=KrabbyMCUCANSDK(virtual=True), virtual=True)
    with pytest.raises(ValueError, match="joint 1"):
        adapter.send_calibration_for_joint("FLHY")


def test_adapter_maps_flhy_to_leg_one_joint_one() -> None:
    assert _GUI_JOINT_MAP["FLHY"] == (1, 1)


def test_adapter_duty_from_pwm() -> None:
    assert _duty_from_pwm(255) == pytest.approx(1.0)
    assert _duty_from_pwm(128) == pytest.approx(128 / 255.0)


def test_jog_sends_set_duty_for_named_joint() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk._buses = [mock.Mock()]  # noqa: SLF001
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    adapter.send_command_jog("FLHY", 128)
    with sdk._lock:  # noqa: SLF001
        assert sdk._duty[0][0] == pytest.approx(128 / 255.0)
        assert sdk._duty[0][1] == pytest.approx(0.0)


def test_jog_preserves_other_joint_duty_on_same_leg() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk._buses = [mock.Mock()]  # noqa: SLF001
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    sdk.set_duty(1, 0.5, 0.0, 0.0)
    adapter.send_command_jog("FLHL", 128)
    with sdk._lock:  # noqa: SLF001
        assert sdk._duty[0][0] == pytest.approx(0.5)
        assert sdk._duty[0][1] == pytest.approx(128 / 255.0)


def test_hold_zeros_all_duties() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk._buses = [mock.Mock()]  # noqa: SLF001
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    sdk.set_duty(1, 0.5, -0.25, 0.1)
    sdk.set_duty(2, 0.3, 0.0, 0.0)
    adapter.send_command_joints_hold()
    with sdk._lock:  # noqa: SLF001
        for row in sdk._duty:
            assert row == [0.0, 0.0, 0.0]


def test_read_version_formats_board_info_ver_line() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    sdk._connected = True  # noqa: SLF001
    with sdk._lock:  # noqa: SLF001
        sdk._board_info[1] = {"major": 1, "minor": 2, "patch": 3, "commit": 99}
        sdk._board_info[2] = {"major": 1, "minor": 2, "patch": 3, "commit": 100}
    line = adapter.read_version(timeout=0.05)
    assert line == "VER 1.2.3|1.2.3|-|-|-|- -|-|-|-|-|- 99|100|-|-|-|-"


def test_read_version_returns_none_when_not_connected() -> None:
    adapter = KrabbyStmGuiAdapter(sdk=KrabbyMCUCANSDK(virtual=True), virtual=True)
    assert adapter.read_version() is None


def test_read_version_returns_none_on_timeout() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    sdk._connected = True  # noqa: SLF001
    assert adapter.read_version(timeout=0.05) is None


def test_joints_map_builds_joint_telemetry_from_can_fields() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    adapter = KrabbyStmGuiAdapter(sdk=sdk, virtual=True)
    with sdk._lock:  # noqa: SLF001
        sdk._telemetry[1] = {
            "J1_POT": 100,
            "J1_HALL": 42,
            "J1_DUTY": 0.5,
            "J1_POSITION": 0.12,
        }
    jt = adapter.joints["FLHY"]
    assert jt is not None
    assert jt.pot == 100
    assert jt.saf == 42
    assert jt.current == 0
    assert jt.pos == pytest.approx(0.12)
    assert jt.pwm == (128, 0)
    assert adapter.last_feedback_ts is not None