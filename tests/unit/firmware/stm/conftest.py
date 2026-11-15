"""Fixtures for tests/unit/firmware/stm."""

from __future__ import annotations

import pytest

from tests.unit.firmware.stm.stm_sdk_virtual_peer import (
    StmSdkVirtualPeer,
    create_stm_sdk_virtual_peer,
)


@pytest.fixture
def stm_sdk_virtual_peer(monkeypatch) -> StmSdkVirtualPeer:
    peer = create_stm_sdk_virtual_peer()
    monkeypatch.setattr(
        "firmware.stm.can_transport.open_socketcan_fd",
        lambda _channel, **_kw: peer.host_bus,
    )
    yield peer
    peer.close()