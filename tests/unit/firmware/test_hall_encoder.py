"""Hall edge scale for simulated ``saf`` telemetry."""

import re
from pathlib import Path

from firmware.sim.hall_encoder import (
    DEFAULT_HALL_EDGES_PER_FULL_TRAVEL,
    HALL_EXTEND_COUNT_SIGN,
    DEFAULT_STM_COUNTS_PER_FULL_TRAVEL,
    edges_for_angle_delta,
    edges_per_radian,
    load_hall_encoder_scale,
    resolve_hall_edges_per_full_travel,
    resolve_stm_counts_per_full_travel,
    signed_edges_for_angle_delta,
)
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HPP = _REPO_ROOT / "firmware/sim/host/hall_encoder_scale.hpp"
_ACTUATOR_CONSTANTS_H = _REPO_ROOT / "firmware/arduino/src/actuator/actuator_constants.h"


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


def test_stm_counts_per_full_travel_in_json():
    data = load_hall_encoder_scale()
    counts = int(data["stm_counts_per_full_travel"])
    assert counts == 6400
    assert resolve_stm_counts_per_full_travel() == counts
    assert DEFAULT_STM_COUNTS_PER_FULL_TRAVEL == counts


def test_extend_count_sign_matches_firmware_constant():
    header = _ACTUATOR_CONSTANTS_H.read_text(encoding="utf-8")
    match = re.search(r"ACTUATOR_HALL_EXTEND_COUNT_SIGN = (-?\d+);", header)
    assert match is not None
    assert int(match.group(1)) == HALL_EXTEND_COUNT_SIGN == -1


def test_extending_counts_down_and_retracting_counts_up():
    span = 2.0
    half = DEFAULT_HALL_EDGES_PER_FULL_TRAVEL // 2
    assert signed_edges_for_angle_delta(1.0, span) == -half
    assert signed_edges_for_angle_delta(-1.0, span) == half
    assert signed_edges_for_angle_delta(0.0, span) == 0
    assert signed_edges_for_angle_delta(1.0, 0.0) == 0
    assert signed_edges_for_angle_delta(0.5, 1.0, edges_per_full_travel=100) == -50
