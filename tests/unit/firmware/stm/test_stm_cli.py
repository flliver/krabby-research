"""krabby-firmware-stm CLI and codec helpers."""

from __future__ import annotations

from unittest import mock

import pytest

from firmware.stm import codec, ids
from firmware.stm.cli import _build_parser
from firmware.stm.mcu_can import KrabbyMCUCANSDK


def test_cli_duty_parses() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--virtual", "duty", "1", "0.5", "0.0", "0.0"])
    assert args.virtual is True
    assert args.command == "duty"
    assert args.leg_id == 1
    assert args.y == pytest.approx(0.5)


def test_encode_calibrate_frame_id() -> None:
    frame_id, data = codec.encode_calibrate(1, 2)
    assert frame_id == ids.calibrate_can_id(1)
    assert len(data) == 1
    assert data[0] == 2


def test_send_calibration_targets_leg_bus(monkeypatch) -> None:
    sent: list[tuple[str, int, bytes]] = []

    class FakeBus:
        def __init__(self, channel: str) -> None:
            self.channel = channel

        def shutdown(self) -> None:
            pass

        def send(self, msg) -> None:
            pass

    monkeypatch.setattr(
        "firmware.stm.can_transport.open_socketcan_fd",
        lambda channel, **_kw: FakeBus(channel),
    )
    monkeypatch.setattr(
        "firmware.stm.can_transport.send_fd_frame",
        lambda bus, fid, data: sent.append((bus.channel, fid, data)),
    )
    monkeypatch.setattr("can.Notifier", lambda _bus, _listeners: mock.Mock(stop=mock.Mock()))

    sdk = KrabbyMCUCANSDK(virtual=False, interfaces=("can0", "can1"))
    sdk.connect()
    sdk.send_calibration(2, 3)
    sdk.close()
    assert len(sent) == 1
    channel, frame_id, _payload = sent[0]
    assert channel == ids.leg_bus(2)
    assert frame_id == ids.calibrate_can_id(2)