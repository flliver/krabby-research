"""KrabbyMCUCANSDK duty cache, virtual CAN, and host leg sim."""

from __future__ import annotations

from unittest import mock

import pytest

from firmware.stm import ids
from firmware.stm.mcu_can import KrabbyMCUCANSDK

from tests.unit.firmware.stm.stm_sdk_virtual_peer import StmSdkVirtualPeer, wait_with_peer


def test_set_duty_updates_only_one_leg() -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk._buses = [mock.Mock()]  # noqa: SLF001 — unit test without SocketCAN
    sdk.set_duty(1, 0.5, 0.0, 0.0)
    sent = sdk._buses[0].send.call_args[0][0]  # noqa: SLF001
    assert sent.arbitration_id == ids.set_duty_can_id()
    assert sent.data[0:2] == (500).to_bytes(2, "little", signed=True)


def test_interfaces_for_virtual() -> None:
    from firmware.stm import can_transport

    assert can_transport.interfaces_for_mode(virtual=True) == ("vcan0",)
    assert can_transport.interfaces_for_mode(virtual=False) == ("can0", "can1")


def test_connect_opens_buses(monkeypatch) -> None:
    import can

    opened: list[str] = []

    class FakeBus:
        def __init__(self, channel: str) -> None:
            self.channel = channel
            opened.append(channel)

        def shutdown(self) -> None:
            pass

        def send(self, _msg) -> None:
            pass

    monkeypatch.setattr(
        "firmware.stm.can_transport.open_socketcan_fd",
        lambda channel, **_kw: FakeBus(channel),
    )
    monkeypatch.setattr(can, "Notifier", lambda _bus, _listeners: mock.Mock(stop=mock.Mock()))

    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    assert opened == ["vcan0"]
    sdk.close()


def test_virtual_bus_set_duty_reaches_host_leg(stm_sdk_virtual_peer: StmSdkVirtualPeer) -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.5, 0.0, 0.0)
        wait_with_peer(
            lambda: abs(stm_sdk_virtual_peer.leg_duty(1, 0) - 0.5) < 1e-5,
            stm_sdk_virtual_peer,
        )
        assert stm_sdk_virtual_peer.leg_duty(1, 0) == pytest.approx(0.5)
    finally:
        sdk.close()


def test_virtual_bus_second_leg_set_duty_keeps_first_leg(stm_sdk_virtual_peer: StmSdkVirtualPeer) -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.5, 0.1, 0.0)
        sdk.set_duty(2, -0.25, 0.0, 0.3)
        wait_with_peer(
            lambda: abs(stm_sdk_virtual_peer.last_set_duty.get("L2_J1", 0.0) + 0.25) < 1e-5,
            stm_sdk_virtual_peer,
        )
        decoded = stm_sdk_virtual_peer.last_set_duty
        assert decoded["L1_J1"] == pytest.approx(0.5)
        assert decoded["L1_J2"] == pytest.approx(0.1)
        assert decoded["L2_J1"] == pytest.approx(-0.25)
        assert decoded["L2_J3"] == pytest.approx(0.3)
        assert stm_sdk_virtual_peer.leg_duty(1, 0) == pytest.approx(0.5)
        assert stm_sdk_virtual_peer.leg_duty(2, 0) == pytest.approx(-0.25)
    finally:
        sdk.close()


def test_virtual_bus_telemetry_returns_joint_fields(stm_sdk_virtual_peer: StmSdkVirtualPeer) -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.42, 0.0, 0.0)

        def telemetry_has_duty() -> bool:
            fields = sdk.telemetry.get(1) or {}
            if "J1_DUTY" not in fields:
                return False
            return abs(float(fields["J1_DUTY"]) - 0.42) < 0.02

        wait_with_peer(telemetry_has_duty, stm_sdk_virtual_peer)
        joints = sdk.telemetry[1]
        assert joints["J1_DUTY"] == pytest.approx(0.42, abs=0.02)
        assert "J1_POT" in joints
        assert "J1_HALL" in joints
    finally:
        sdk.close()