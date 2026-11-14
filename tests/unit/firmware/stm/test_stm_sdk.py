"""KrabbyMCUCANSDK duty cache and TX payload."""

from __future__ import annotations

from unittest import mock

from firmware.stm import ids
from firmware.stm.mcu_can import KrabbyMCUCANSDK


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