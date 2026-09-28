"""Unit tests for the crab-hex Whitworth slotted-lever cam mapping (Phase E).

Pure torch -- no Isaac Sim needed. These pin down the closed-form derivation in
``crab_hex_cam_mapping.py``'s docstring: exact swing bounds, the analytic velocity against a
numeric derivative, the round-trip inverse, and the quick-return asymmetry that distinguishes
this mechanism from the ``sin(theta_shaft)`` placeholder it replaces.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour"
    / "parkour_tasks"
    / "parkour_tasks"
    / "crab_hex_forward_task"
    / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from crab_hex_cam_mapping import (  # noqa: E402
    THETA_HIP_MAX,
    _K,
    cam_shaft_to_hip,
    hip_to_cam_shaft_default,
)


def test_swing_bounds_exactly_theta_hip_max():
    theta_shaft = torch.linspace(0, 2 * math.pi, 200_000, dtype=torch.float64)
    theta_hip, _ = cam_shaft_to_hip(theta_shaft, torch.zeros_like(theta_shaft))
    assert theta_hip.abs().max().item() == pytest.approx(THETA_HIP_MAX, abs=1e-6)


def test_zero_shaft_angle_gives_zero_hip_angle():
    theta_hip, omega_hip = cam_shaft_to_hip(torch.tensor([0.0]), torch.tensor([1.0]))
    assert theta_hip.item() == pytest.approx(0.0, abs=1e-12)
    # mechanical advantage is not zero at the neutral point (unlike a sinusoid at its own zero,
    # this and sin() actually agree here since both have max slope at theta_shaft=0)
    assert omega_hip.item() > 0


def test_analytic_velocity_matches_numeric_derivative():
    theta_shaft = torch.linspace(0.01, 2 * math.pi - 0.01, 50_000, dtype=torch.float64)
    eps = 1e-6
    th_plus, _ = cam_shaft_to_hip(theta_shaft + eps, torch.zeros_like(theta_shaft))
    th_minus, _ = cam_shaft_to_hip(theta_shaft - eps, torch.zeros_like(theta_shaft))
    numeric = (th_plus - th_minus) / (2 * eps)

    omega_shaft = torch.ones_like(theta_shaft)
    _, analytic = cam_shaft_to_hip(theta_shaft, omega_shaft)

    # exclude the atan2 branch-cut neighborhood (theta_shaft near pi), where the central
    # difference of an angle-valued function wraps and is not a valid finite-difference estimate
    near_wrap = (theta_shaft - math.pi).abs() < 0.05
    assert torch.allclose(numeric[~near_wrap], analytic[~near_wrap], atol=1e-6)


def test_velocity_scales_linearly_with_shaft_speed():
    theta_shaft = torch.tensor([0.3, 1.0, 2.5])
    _, omega_1x = cam_shaft_to_hip(theta_shaft, torch.ones(3))
    _, omega_3x = cam_shaft_to_hip(theta_shaft, torch.full((3,), 3.0))
    assert torch.allclose(omega_3x, 3.0 * omega_1x)


def test_odd_symmetry_about_shaft_zero():
    """A continuous one-directional spin should trace the same swing shape whichever way it
    starts -- theta_hip(-theta_shaft) == -theta_hip(theta_shaft)."""
    theta_shaft = torch.linspace(0.01, math.pi - 0.01, 1000, dtype=torch.float64)
    pos, _ = cam_shaft_to_hip(theta_shaft, torch.zeros_like(theta_shaft))
    neg, _ = cam_shaft_to_hip(-theta_shaft, torch.zeros_like(theta_shaft))
    assert torch.allclose(pos, -neg, atol=1e-10)


def test_quick_return_asymmetry_distinguishes_from_sinusoid():
    """The defining physical signature of this mechanism vs. the sin() placeholder: the peak
    occurs well past 90 deg of shaft rotation (exact value follows from K; see the module
    docstring's peak-angle formula theta_s_peak = 180 - acos(K))."""
    theta_shaft = torch.linspace(0, math.pi, 100_000, dtype=torch.float64)
    theta_hip, _ = cam_shaft_to_hip(theta_shaft, torch.zeros_like(theta_shaft))
    peak_idx = torch.argmax(theta_hip)
    peak_shaft_deg = math.degrees(theta_shaft[peak_idx].item())
    expected_peak_deg = 180.0 - math.degrees(math.acos(_K))
    assert peak_shaft_deg == pytest.approx(expected_peak_deg, abs=0.5)
    assert peak_shaft_deg > 100.0  # well clear of a sinusoid's 90 deg, not just noise


def test_inverse_round_trips_exactly():
    # within the mechanism's own +/-THETA_HIP_MAX (~28.5 deg) range -- the leg defaults actually
    # used in crab_hex_scene_cfg.py, post rescale (see that file's Phase-E note)
    for theta_hip_default in (0.342492, -0.342492, 0.142705, -0.142705, 0.0):
        theta_shaft = hip_to_cam_shaft_default(theta_hip_default)
        # torch.tensor(...) defaults to float32 (~1e-7 precision); the double-precision math
        # itself round-trips to machine epsilon, as float64 inputs elsewhere in this file confirm.
        back, _ = cam_shaft_to_hip(torch.tensor([theta_shaft]), torch.tensor([0.0]))
        assert back.item() == pytest.approx(theta_hip_default, abs=1e-6)


def test_inverse_rejects_unreachable_default():
    """A default pose the mechanism cannot physically reach (e.g. the pre-Phase-E +/-0.6 rad
    front/rear splay, which exceeds the corrected +/-28.5 deg limit) must fail loudly, not
    silently return NaN -- this is exactly the bug that caught the stale defaults."""
    assert math.sin(0.6) / _K > 1.0  # confirms 0.6 rad is indeed out of range for this K
    with pytest.raises(ValueError, match="exceeds the mechanism's own swing limit"):
        hip_to_cam_shaft_default(0.6)


def test_theta_hip_max_is_derived_from_k_not_the_reverse():
    """THETA_HIP_MAX = asin(K) exactly. Since 2026-08-20, K itself comes from the
    hardware-measured yaw throw (+-25 deg, crab_hex_dimensions.YAW_K = sin(throw)) rather
    than the KrabV3-Legs.svg slot geometry (r/L = 0.4778 -> 28.54 deg) -- the physical
    crank/slot as built differs from that SVG. The mapping's causality is unchanged: K is
    the measured geometry, the swing limit is its consequence."""
    assert THETA_HIP_MAX == pytest.approx(math.asin(_K))
    assert _K == pytest.approx(math.sin(math.radians(25.0)))  # measured throw, exactly


def test_batched_and_broadcastable():
    """Matches how apply_actions calls this every substep across all envs/legs at once."""
    theta_shaft = torch.randn(256, 6, dtype=torch.float64)
    omega_shaft = torch.randn(256, 6, dtype=torch.float64)
    theta_hip, omega_hip = cam_shaft_to_hip(theta_shaft, omega_shaft)
    assert theta_hip.shape == theta_shaft.shape
    assert omega_hip.shape == theta_shaft.shape
    assert torch.isfinite(theta_hip).all()
    assert torch.isfinite(omega_hip).all()


def test_multi_turn_periodicity():
    """Velocity-driven shafts rotate continuously: the map must be exactly 2*pi-periodic so a
    multi-turn shaft angle (e.g. 400 rad after a 60 s episode at ~6 rad/s) yields the same hip
    angle as its wrapped equivalent, and stays bounded by THETA_HIP_MAX."""
    theta = torch.linspace(-math.pi, math.pi, 97, dtype=torch.float64)
    omega = torch.ones_like(theta)
    base_hip, base_vel = cam_shaft_to_hip(theta, omega)
    for k in (-3, -2, -1, 1, 2, 3):
        hip_k, vel_k = cam_shaft_to_hip(theta + 2 * math.pi * k, omega)
        assert torch.allclose(hip_k, base_hip, atol=1e-9)
        assert torch.allclose(vel_k, base_vel, atol=1e-9)
    # far multi-turn angles stay finite and inside the mechanism's swing limit
    far = torch.linspace(-400.0, 400.0, 4001, dtype=torch.float64)
    hip_far, _ = cam_shaft_to_hip(far, torch.zeros_like(far))
    assert torch.isfinite(hip_far).all()
    assert hip_far.abs().max().item() <= THETA_HIP_MAX + 1e-9
    # float32 note: at |theta| ~ 400 rad, float32 phase resolution is ~2e-5 rad -- fine for
    # control, but keep float64 here so the periodicity assertion itself is exact.
