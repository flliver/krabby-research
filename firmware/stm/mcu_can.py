"""KrabbyMCUCANSDK — CAN-FD leg bus."""

from __future__ import annotations

import threading
from typing import Any

from firmware.stm import can_transport, codec, ids

DutyGrid = list[list[float]]


def _clamp_duty(value: float) -> float:
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return float(value)


class KrabbyMCUCANSDK:
    """Host SDK: dual-bus SET_DUTY TX and background telemetry decode."""

    def __init__(
        self,
        *,
        virtual: bool = False,
        interfaces: tuple[str, ...] | None = None,
    ) -> None:
        self._virtual = virtual
        self._interfaces = interfaces
        self._buses: list[Any] = []
        self._buses_by_channel: dict[str, Any] = {}
        self._notifiers: list[Any] = []
        self._lock = threading.Lock()
        self._duty: DutyGrid = [[0.0, 0.0, 0.0] for _ in range(6)]
        self._telemetry: dict[int, dict[str, Any]] = {leg: {} for leg in ids.LEG_IDS}
        self._board_info: dict[int, dict[str, Any]] = {leg: {} for leg in ids.LEG_IDS}
        self._connected = False

    @property
    def telemetry(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {leg: dict(fields) for leg, fields in self._telemetry.items()}

    @property
    def board_info(self) -> dict[int, dict[str, Any]]:
        with self._lock:
            return {leg: dict(fields) for leg, fields in self._board_info.items()}

    def connect(self) -> None:
        if self._connected:
            return
        import can

        ifaces = self._interfaces
        if ifaces is None:
            ifaces = can_transport.interfaces_for_mode(virtual=self._virtual)
        codec.load_database()
        for channel in ifaces:
            bus = can_transport.open_socketcan_fd(channel)
            self._buses.append(bus)
            self._buses_by_channel[channel] = bus
            self._notifiers.append(can.Notifier(bus, [self._on_message]))
        self._connected = True

    def close(self) -> None:
        for notifier in self._notifiers:
            notifier.stop()
        for bus in self._buses:
            bus.shutdown()
        self._notifiers.clear()
        self._buses.clear()
        self._buses_by_channel.clear()
        self._connected = False

    def send_calibration(self, leg_id: int, joint_id: int) -> None:
        if leg_id not in ids.LEG_IDS:
            raise ValueError(f"leg_id out of range: {leg_id!r}")
        if joint_id not in (2, 3):
            raise ValueError(f"joint_id must be 2 or 3, got {joint_id!r}")
        frame_id, data = codec.encode_calibrate(leg_id, joint_id)
        self._send_on_leg_bus(leg_id, frame_id, data)

    def set_duty(self, leg_id: int, j1: float, j2: float, j3: float) -> None:
        if leg_id not in ids.LEG_IDS:
            raise ValueError(f"leg_id out of range: {leg_id!r}")
        with self._lock:
            row = leg_id - 1
            self._duty[row] = [_clamp_duty(j1), _clamp_duty(j2), _clamp_duty(j3)]
            self._transmit_set_duty_locked()

    def set_duties(self, duties: tuple[float, ...] | list[float]) -> None:
        payload = codec.set_duty_values_flat(tuple(duties))
        with self._lock:
            for leg_id in ids.LEG_IDS:
                row = leg_id - 1
                self._duty[row] = [
                    _clamp_duty(payload[f"L{leg_id}_J{j}"]) for j in ids.JOINT_IDS
                ]
            self._transmit_set_duty_locked()

    def _send_on_leg_bus(self, leg_id: int, frame_id: int, data: bytes) -> None:
        channel = ids.leg_bus(leg_id)
        bus = self._buses_by_channel.get(channel)
        if bus is None and self._buses:
            bus = self._buses[0]
        if bus is None:
            return
        can_transport.send_fd_frame(bus, frame_id, data)

    def _transmit_set_duty_locked(self) -> None:
        if not self._buses:
            return
        grid = tuple(tuple(row) for row in self._duty)
        payload = codec.set_duty_values_from_grid(grid)
        frame_id, data = codec.encode_set_duty(payload)
        for bus in self._buses:
            can_transport.send_fd_frame(bus, frame_id, data)

    def _on_message(self, msg: Any) -> None:
        can_id = int(msg.arbitration_id)
        payload = bytes(msg.data)
        leg_id = ids.telemetry_leg_id_from_frame(can_id)
        if leg_id is not None:
            fields = codec.decode_telemetry(leg_id, payload)
            with self._lock:
                self._telemetry[leg_id] = fields
            return
        leg_id = ids.board_info_leg_id_from_frame(can_id)
        if leg_id is not None:
            fields = codec.decode_board_info(leg_id, payload)
            with self._lock:
                self._board_info[leg_id] = fields
