"""Unit tests for the travel-frame helpers (no Isaac import).

Pins two invariants: (1) every helper is the identity -- returning the same
tensor object, or the bitwise-identical old expression -- when the travel-frame
angle/offset is exactly 0.0, so the forward task stays byte-identical; and
(2) at travel_yaw=+pi/2 the travel frame is the body frame rotated so that
travel speed/heading track body +y instead of body +x.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

MOD_DIR = Path(__file__).resolve().parents[2] / "parkour" / "parkour_isaaclab" / "envs" / "mdp"
if str(MOD_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_DIR))

import travel_frame  # noqa: E402
from travel_frame import (  # noqa: E402
    heading_with_offset,
    rotate_xy,
    track_lin_vel_exp,
    vel_travel_xy,
    wrap_to_pi,
    yaw_error,
)

HALF_PI = math.pi / 2.0


def test_vel_travel_xy_identity_at_zero_returns_same_object():
    g = torch.Generator().manual_seed(0)
    v = torch.randn(8, 2, generator=g)
    out = vel_travel_xy(v, 0.0)
    assert out is v


def test_vel_travel_xy_quarter_turn_maps_vx_vy_to_vy_minus_vx():
    g = torch.Generator().manual_seed(1)
    v = torch.randn(64, 2, generator=g)
    out = vel_travel_xy(v, HALF_PI)
    assert torch.allclose(out[:, 0], v[:, 1], atol=1e-6)
    assert torch.allclose(out[:, 1], -v[:, 0], atol=1e-6)


def test_vel_travel_xy_inverts_rotate_by_travel_yaw():
    g = torch.Generator().manual_seed(2)
    v = torch.randn(32, 2, generator=g)
    for a in (0.3, HALF_PI, -HALF_PI):
        roundtrip = rotate_xy(vel_travel_xy(v, a), a)
        assert torch.allclose(roundtrip, v, atol=1e-6)


def test_track_lin_vel_exp_zero_is_bitwise_isaaclab_formula():
    g = torch.Generator().manual_seed(3)
    c = torch.randn(16, 2, generator=g)
    v = torch.randn(16, 2, generator=g)
    std = 0.5
    out = track_lin_vel_exp(c, v, std, 0.0)
    expected = torch.exp(-torch.sum(torch.square(c - v), dim=1) / std**2)
    assert torch.equal(out, expected)


def test_track_lin_vel_exp_sideways_rewards_body_vy():
    cmd = torch.tensor([[0.3, 0.0]])
    vel = torch.tensor([[0.0, 0.3]])
    std = 0.5

    out_sideways = track_lin_vel_exp(cmd, vel, std, HALF_PI)
    assert torch.allclose(out_sideways, torch.tensor([1.0]), atol=1e-6)

    out_forward = track_lin_vel_exp(cmd, vel, std, 0.0)
    expected_forward = math.exp(-0.18 / std**2)
    assert torch.allclose(out_forward, torch.tensor([expected_forward]), atol=1e-6)


def test_yaw_error_zero_is_bitwise_old_expression():
    g = torch.Generator().manual_seed(4)
    t = torch.empty(100).uniform_(-4.0, 4.0, generator=g)
    y = torch.empty(100).uniform_(-4.0, 4.0, generator=g)
    out = yaw_error(t, y, 0.0)
    assert torch.equal(out, t - y)


def test_yaw_error_offset_wraps():
    # target=0, yaw=-pi/2, travel=pi/2 -> 0 - (-pi/2) - pi/2 = 0
    out1 = yaw_error(torch.tensor([0.0]), torch.tensor([-HALF_PI]), HALF_PI)
    assert torch.allclose(out1, torch.tensor([0.0]), atol=1e-6)

    # target=0, yaw=3pi/4, travel=pi/2 -> 0 - 3pi/4 - pi/2 = -5pi/4, wrapped = 3pi/4
    out2 = yaw_error(torch.tensor([0.0]), torch.tensor([3 * math.pi / 4]), HALF_PI)
    assert torch.allclose(out2, torch.tensor([3 * math.pi / 4]), atol=1e-6)

    g = torch.Generator().manual_seed(5)
    t = torch.empty(200).uniform_(-4.0, 4.0, generator=g)
    y = torch.empty(200).uniform_(-4.0, 4.0, generator=g)
    out = yaw_error(t, y, HALF_PI)
    assert (out.abs() <= math.pi + 1e-6).all()


def test_wrap_to_pi_keeps_plus_pi():
    # IsaacLab semantics (isaaclab/utils/math.py, wrap_to_pi): +pi maps to +pi,
    # -pi maps to -pi, odd positive multiples of pi map to +pi.
    assert wrap_to_pi(torch.tensor([math.pi])).item() == pytest.approx(math.pi)
    assert wrap_to_pi(torch.tensor([3 * math.pi])).item() == pytest.approx(math.pi)
    assert wrap_to_pi(torch.tensor([-math.pi])).item() == pytest.approx(-math.pi)


def test_heading_with_offset():
    g = torch.Generator().manual_seed(6)
    h = torch.randn(10, generator=g)
    assert heading_with_offset(h, 0.0) is h
    assert torch.equal(heading_with_offset(h, -HALF_PI), h + (-HALF_PI))
