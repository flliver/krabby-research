"""python-can virtual peer for KrabbyMCUCANSDK unit tests."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from firmware.sim.can_board_bridge import Leg, StmBoardBridge
from firmware.stm import can_transport, codec, ids


def open_virtual_fd_bus(channel: str):
    import can

    return can.interface.Bus(
        interface="virtual",
        channel=channel,
        fd=True,
        bitrate=can_transport.ARBITRATION_BITRATE,
        data_bitrate=can_transport.DATA_BITRATE,
    )


@dataclass
class StmSdkVirtualPeer:
    """SDK on host_bus; SET_DUTY is applied on peer_bus via StmBoardBridge."""

    host_bus: Any
    peer_bus: Any
    bridge: StmBoardBridge
    legs: dict[int, Leg]
    tick: int = 0
    last_set_duty: dict[str, Any] = field(default_factory=dict)
    _set_duty_msg: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._set_duty_msg is None:
            self._set_duty_msg = codec.load_database().get_message_by_name("SET_DUTY")

    def drain(self) -> None:
        msg = self.peer_bus.recv(timeout=0.0)
        if msg is None or int(msg.arbitration_id) != ids.set_duty_can_id():
            return
        data = bytes(msg.data)
        self.last_set_duty = self._set_duty_msg.decode(data)
        for leg in self.legs.values():
            self.bridge.inject_rx(leg, ids.set_duty_can_id(), data)
            self.bridge.set_tick_ms(self.tick)
            self.bridge.app_tick(leg, self.tick)
            tx = self.bridge.last_tx(leg)
            if tx is not None:
                can_transport.send_fd_frame(self.peer_bus, tx[0], tx[1])
        self.tick = min(self.tick + 50, 10_000)

    def leg_duty(self, leg_id: int, motor: int) -> float:
        return self.bridge.get_duty(self.legs[leg_id], motor)

    def close(self) -> None:
        self.host_bus.shutdown()
        self.peer_bus.shutdown()


def wait_with_peer(
    predicate: Callable[[], bool],
    peer: StmSdkVirtualPeer,
    *,
    timeout_s: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        peer.drain()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


def create_stm_sdk_virtual_peer(*, channel: str | None = None) -> StmSdkVirtualPeer:
    bus_channel = channel or f"krabby-stm-sdk-{uuid.uuid4().hex}"
    bridge = StmBoardBridge()
    return StmSdkVirtualPeer(
        host_bus=open_virtual_fd_bus(bus_channel),
        peer_bus=open_virtual_fd_bus(bus_channel),
        bridge=bridge,
        legs={1: bridge.create_leg(1), 2: bridge.create_leg(2)},
    )