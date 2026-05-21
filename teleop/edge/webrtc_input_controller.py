"""WebRTC-fed gamepad state for the portal → HAL path (mirrors pygame ``InputController`` API).

``GamepadToKrabbyHALMapper`` only reads boolean/float attributes; keep field names aligned with
``controller.input.state.ControllerState``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RemoteGamepadState:
    """Normalized controller mirrors ``controller.input.state.ControllerState`` field names."""

    LT: bool = False
    LB: bool = False
    LS: bool = False
    RS: bool = False
    RT: bool = False
    RB: bool = False
    LX: float = 0.0
    LY: float = 0.0
    RX: float = 0.0
    RY: float = 0.0


class WebRTCInputController:
    """Thread-safe state holder + callback broadcaster for remote gamepad payloads."""

    def __init__(self) -> None:
        self._state = RemoteGamepadState()
        self._state_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._callbacks: list[Callable[[RemoteGamepadState], None]] = []
        self._running = False

    def start(self, *, update_rate_hz: float = 50.0, device_id: int | None = None) -> None:
        """Match ``InputController.start`` signature for drop-in wiring."""
        if update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be greater than 0, got {update_rate_hz}")
        _ = device_id
        self._running = True

    def stop(self) -> None:
        self._running = False

    def register_callback(self, callback: Callable[[RemoteGamepadState], None]) -> None:
        with self._callback_lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[RemoteGamepadState], None]) -> None:
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_state(self) -> RemoteGamepadState:
        with self._state_lock:
            s = self._state
            return RemoteGamepadState(
                LT=s.LT,
                LB=s.LB,
                LS=s.LS,
                RS=s.RS,
                RT=s.RT,
                RB=s.RB,
                LX=s.LX,
                LY=s.LY,
                RX=s.RX,
                RY=s.RY,
            )

    def update_from_payload(self, payload: dict[str, Any]) -> RemoteGamepadState:
        """Validate + apply a remote controller payload, then notify callbacks."""
        if not self._running:
            raise RuntimeError("WebRTCInputController must be started before update_from_payload")
        state = self._state_from_payload(payload)
        with self._state_lock:
            self._state = state
        self._notify_callbacks(state)
        return state

    @staticmethod
    def _state_from_payload(payload: dict[str, Any]) -> RemoteGamepadState:
        def _b(name: str) -> bool:
            return bool(payload.get(name, False))

        def _f(name: str) -> float:
            raw = payload.get(name, 0.0)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"controller payload field {name!r} must be numeric")
            if val < -1.0:
                return -1.0
            if val > 1.0:
                return 1.0
            return val

        return RemoteGamepadState(
            LT=_b("LT"),
            LB=_b("LB"),
            LS=_b("LS"),
            RS=_b("RS"),
            RT=_b("RT"),
            RB=_b("RB"),
            LX=_f("LX"),
            LY=_f("LY"),
            RX=_f("RX"),
            RY=_f("RY"),
        )

    def _notify_callbacks(self, state: RemoteGamepadState) -> None:
        with self._callback_lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            cb(state)
