"""Hall edge scale for simulated ``saf`` telemetry."""

import re
from pathlib import Path

from firmware.sim.hall_encoder import (
    DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    edges_for_angle_delta,
    edges_per_radian,
    load_hall_encoder_scale,
    resolve_hall_edges_per_full_travel,
)
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HPP = _REPO_ROOT / "firmware/sim/host/hall_encoder_scale.hpp"


def test_full_travel_delta_matches_configured_edges():
    span = 1.25
    total = edges_for_angle_delta(span, span)
    assert total == DEFAULT_HALL_EDGES_PER_FULL_TRAVEL


def test_partial_travel_scales_linearly():
    span = 2.0
    half = edges_for_angle_delta(1.0, span)
    assert half == DEFAULT_HALL_EDGES_PER_FULL_TRAVEL // 2


def test_edges_per_radian_inverts_span():
    span = 0.5
    rate = edges_per_radian(span)
    assert rate == DEFAULT_HALL_EDGES_PER_FULL_TRAVEL / span


def test_simulated_robot_config_carries_hall_scale():
    cfg = SimulatedRobotConfig.front_board_rev3()
    assert cfg.hall_edges_per_full_travel == DEFAULT_HALL_EDGES_PER_FULL_TRAVEL
    custom = SimulatedRobotConfig.front_board_rev3(hall_edges_per_full_travel=1800)
    assert custom.hall_edges_per_full_travel == 1800


def test_hall_encoder_scale_json_matches_python_and_cpp():
    data = load_hall_encoder_scale()
    edges = int(data["edges_per_full_travel"])
    assert resolve_hall_edges_per_full_travel() == edges
    assert DEFAULT_HALL_EDGES_PER_FULL_TRAVEL == edges
    hpp = _HPP.read_text(encoding="utf-8")
    match = re.search(r"kHallEdgesPerFullTravel = (\d+)u", hpp)
    assert match is not None
    assert int(match.group(1)) == edges
