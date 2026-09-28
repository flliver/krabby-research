"""awscrt return-shape helpers (Future vs (Future, packet_id))."""
from __future__ import annotations

from krabby_fleet_service._mqtt import _await_crt


class _FakeFuture:
    def __init__(self) -> None:
        self.waited = False

    def result(self, timeout: float | None = None) -> None:
        self.waited = True
        _ = timeout


def test_await_crt_accepts_future() -> None:
    fut = _FakeFuture()
    _await_crt(fut, timeout=1.0)
    assert fut.waited


def test_await_crt_accepts_tuple() -> None:
    fut = _FakeFuture()
    _await_crt((fut, 42), timeout=1.0)
    assert fut.waited
