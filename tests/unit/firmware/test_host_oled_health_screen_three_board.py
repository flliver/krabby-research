"""OLED health screen over sim Wire with three live controllers and missing follower."""

from __future__ import annotations

import pytest

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.three_board_boot import (
    ROLE_FRONT,
    ROLE_LEFT,
    ROLE_RIGHT,
    boot_three_board_emulators,
)

OLED_DRAW_MS = 250
TELEMETRY_MS = 50
CONTROLLER_STALE_MS = 500


@pytest.fixture(scope="module")
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return bridge


def _stage_front_sensors(host) -> None:
    for name in ("FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"):
        host.set_actuator_stops(name, 100, 900)


def _tick(host, *, steps: int, step_ms: int = 10) -> None:
    for _ in range(steps):
        host.advance_millis(step_ms)
        host.firmware_loop_once()


def _draw_oled(host) -> None:
    _tick(host, steps=(OLED_DRAW_MS // 10) + 2)


def _boot_three_board(host) -> None:
    boot_three_board_emulators(host)
    host.enable_three_board_firmware_loop(True)
    host.firmware_setup()
    _stage_front_sensors(host)


def test_oled_health_screen_shows_three_controller_thirds(host):
    _boot_three_board(host)
    _tick(host, steps=40)
    _draw_oled(host)

    assert host.oled_wire_data_bytes_received() > 0
    assert host.oled_controller_shown(ROLE_FRONT)
    assert host.oled_controller_shown(ROLE_LEFT)
    assert host.oled_controller_shown(ROLE_RIGHT)


def test_oled_health_screen_drops_missing_left_follower(host):
    _boot_three_board(host)
    _tick(host, steps=40)
    _draw_oled(host)
    assert host.oled_controller_shown(ROLE_LEFT)

    baseline_crc = host.oled_framebuffer_crc32()
    host.set_three_board_follower_telemetry(left=False, right=True)
    host.oled_invalidate()
    stale_steps = (CONTROLLER_STALE_MS + OLED_DRAW_MS) // 10 + 4
    _tick(host, steps=stale_steps)
    _draw_oled(host)

    assert not host.oled_controller_shown(ROLE_LEFT)
    assert host.oled_controller_shown(ROLE_FRONT)
    assert host.oled_controller_shown(ROLE_RIGHT)
    assert host.oled_framebuffer_crc32() != baseline_crc
