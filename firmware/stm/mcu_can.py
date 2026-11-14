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
            self._notifiers.append(can.Notifier(bus, [self._on_message]))
        self._connected = True

    def close(self) -> None:
        for notifier in self._notifiers:
            notifier.stop()
        for bus in self._buses:
            bus.shutdown()
        self._notifiers.clear()
        self._buses.clear()
        self._connected = False

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

    def _transmit_set_duty_locked(self) -> None:
        if not self._buses:
            return
        grid = tuple(tuple(row) for row in self._duty)
        payload = codec.set_duty_values_from_grid(grid)
        frame_id, data = codec.encode_set_duty(payload)
        for bus in self._buses:
            can_transport.send_fd_frame(bus, frame_id, data)

    def _on_message(self, msg: Any) -> None:
        decoded = codec.decode_frame(msg.arbitration_id, bytes(msg.data))
        if decoded is None:
            return
        name, fields = decoded
        with self._lock:
            if name.startswith("TELEMETRY_"):
                leg_id = int(name.split("_", 1)[1])
                self._telemetry[leg_id] = fields
            elif name.startswith("BOARD_INFO_"):
                leg_id = int(name.split("_", 1)[1])
                self._board_info[leg_id] = fields