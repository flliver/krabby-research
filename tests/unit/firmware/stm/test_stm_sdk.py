"""KrabbyMCUCANSDK duty cache, virtual CAN, and host leg sim."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from unittest import mock

import pytest

from firmware.sim.can_board_bridge import StmBoardBridge, Leg
from firmware.stm import can_transport, codec, ids
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


def _open_virtual_fd_bus(channel: str):
    import can

    kwargs = {
        "interface": "virtual",
        "channel": channel,
        "fd": True,
        "bitrate": can_transport.ARBITRATION_BITRATE,
        "data_bitrate": can_transport.DATA_BITRATE,
    }
    return can.interface.Bus(**kwargs)


class _VirtualLegSim:
    """One python-can virtual channel: SDK bus + in-process leg firmware (ctypes)."""

    def __init__(self, channel: str) -> None:
        self._channel = channel
        self._bridge = StmBoardBridge()
        self._legs: dict[int, Leg] = {
            1: self._bridge.create_leg(1),
            2: self._bridge.create_leg(2),
        }
        self._host_bus = _open_virtual_fd_bus(channel)
        self._sim_bus = _open_virtual_fd_bus(channel)
        self._stop = threading.Event()
        self._last_set_duty: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        tick = 0
        set_duty_id = ids.set_duty_can_id()
        db = codec.load_database()
        set_duty_msg = db.get_message_by_name("SET_DUTY")
        while not self._stop.is_set():
            msg = self._sim_bus.recv(timeout=0.05)
            if msg is None:
                continue
            if int(msg.arbitration_id) != set_duty_id:
                continue
            data = bytes(msg.data)
            with self._lock:
                self._last_set_duty = set_duty_msg.decode(data)
            for leg in self._legs.values():
                self._bridge.inject_rx(leg, set_duty_id, data)
            for leg in self._legs.values():
                self._bridge.set_tick_ms(tick)
                self._bridge.app_tick(leg, tick)
                tx = self._bridge.last_tx(leg)
                if tx is not None:
                    frame_id, payload = tx
                    can_transport.send_fd_frame(self._sim_bus, frame_id, payload)
            tick = min(tick + 50, 10_000)

    @property
    def host_bus(self):
        return self._host_bus

    def last_set_duty(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_set_duty)

    def leg_duty(self, leg_id: int, motor: int) -> float:
        return self._bridge.get_duty(self._legs[leg_id], motor)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._host_bus.shutdown()
        self._sim_bus.shutdown()


@pytest.fixture
def virtual_leg_sim(monkeypatch) -> _VirtualLegSim:
    channel = f"krabby-stm-sdk-{uuid.uuid4().hex}"
    sim = _VirtualLegSim(channel)
    monkeypatch.setattr(
        "firmware.stm.can_transport.open_socketcan_fd",
        lambda _channel, **_kw: sim.host_bus,
    )
    yield sim
    sim.close()


def _wait_for(predicate, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


def test_virtual_bus_set_duty_reaches_host_leg(virtual_leg_sim: _VirtualLegSim) -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.5, 0.0, 0.0)
        _wait_for(lambda: abs(virtual_leg_sim.leg_duty(1, 0) - 0.5) < 1e-5)
        assert virtual_leg_sim.leg_duty(1, 0) == pytest.approx(0.5)
    finally:
        sdk.close()


def test_virtual_bus_second_leg_set_duty_keeps_first_leg(virtual_leg_sim: _VirtualLegSim) -> None:
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.5, 0.1, 0.0)
        sdk.set_duty(2, -0.25, 0.0, 0.3)
        _wait_for(lambda: abs(virtual_leg_sim.last_set_duty().get("L2_J1", 0.0) + 0.25) < 1e-5)
        decoded = virtual_leg_sim.last_set_duty()
        assert decoded["L1_J1"] == pytest.approx(0.5)
        assert decoded["L1_J2"] == pytest.approx(0.1)
        assert decoded["L2_J1"] == pytest.approx(-0.25)
        assert decoded["L2_J3"] == pytest.approx(0.3)
        assert virtual_leg_sim.leg_duty(1, 0) == pytest.approx(0.5)
        assert virtual_leg_sim.leg_duty(2, 0) == pytest.approx(-0.25)
    finally:
        sdk.close()


def test_virtual_bus_telemetry_returns_joint_fields(virtual_leg_sim: _VirtualLegSim) -> None:
    """Decoded TELEMETRY_* fields (J1_DUTY, …) land in SDK telemetry cache."""
    sdk = KrabbyMCUCANSDK(virtual=True)
    sdk.connect()
    try:
        sdk.set_duty(1, 0.42, 0.0, 0.0)

        def _joint_duty_ok() -> bool:
            fields = sdk.telemetry.get(1) or {}
            if "J1_DUTY" not in fields:
                return False
            return abs(float(fields["J1_DUTY"]) - 0.42) < 0.02

        _wait_for(_joint_duty_ok)
        joints = sdk.telemetry[1]
        assert joints["J1_DUTY"] == pytest.approx(0.42, abs=0.02)
        assert "J1_POT" in joints
        assert "J1_HALL" in joints
    finally:
        sdk.close()
