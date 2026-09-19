"""Unit tests for the crab-hex linear-actuator linkage kinematics.

Pure torch -- no Isaac Sim needed. Pins the closed-form geometry in
``crab_hex_linkage.py``: exact inverses, analytic derivatives vs finite differences, the
feasibility envelope (rod lengths stay inside the physical stroke window across the
measured ROM -- this is the tripwire for a wrong anchor estimate OR an inconsistent
ROM/stroke combination), the hip->knee coupling sign, and the mechanism's real
joint-space-corner unreachability.
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

import crab_hex_dimensions as dims  # noqa: E402
import crab_hex_linkage as lk  # noqa: E402

RAD = math.radians
HIP_ROM = tuple(RAD(v) for v in dims.HIP_SIM_LIMITS_DEG)  # (-45, +60) deg
KNEE_ROM = tuple(RAD(v) for v in dims.KNEE_SIM_LIMITS_LEFT_DEG)  # (-50, +85) deg

# Anchor-estimate slack: the hip anchor is solved from the rod window, not yet measured.
LEN_TOL_M = 0.005


def _hip_grid(n: int = 64) -> torch.Tensor:
    return torch.linspace(HIP_ROM[0], HIP_ROM[1], n, dtype=torch.float64)


def _knee_grid(n: int = 64) -> torch.Tensor:
    return torch.linspace(KNEE_ROM[0], KNEE_ROM[1], n, dtype=torch.float64)


# ---------------------------------------------------------------- inverses / round trips
def test_hip_length_angle_round_trip():
    th = _hip_grid()
    back = lk.hip_angle_from_length(lk.hip_actuator_length(th))
    assert torch.allclose(back, th, atol=1e-10)


def test_knee_length_angle_round_trip():
    tk = _knee_grid()
    th = torch.full_like(tk, lk.hip_default_rad())
    back = lk.knee_angle_from_length(lk.knee_actuator_length(th, tk), th)
    assert torch.allclose(back, tk, atol=1e-10)


# ------------------------------------------------------------------ analytic derivatives
def test_hip_moment_arm_matches_finite_difference():
    eps = 1e-7
    for val in (-0.6, -0.2, 0.11, 0.5, 1.0):
        t = torch.tensor(val, dtype=torch.float64)
        fd = (lk.hip_actuator_length(t + eps) - lk.hip_actuator_length(t - eps)) / (2 * eps)
        assert float(lk.hip_moment_arm(t)) == pytest.approx(abs(float(fd)), abs=1e-6)


def test_knee_derivatives_match_finite_difference():
    eps = 1e-7
    th = torch.tensor(0.2, dtype=torch.float64)
    tk = torch.tensor(0.3, dtype=torch.float64)
    fd_k = (lk.knee_actuator_length(th, tk + eps) - lk.knee_actuator_length(th, tk - eps)) / (2 * eps)
    fd_h = (lk.knee_actuator_length(th + eps, tk) - lk.knee_actuator_length(th - eps, tk)) / (2 * eps)
    assert float(lk.knee_moment_arm(th, tk)) == pytest.approx(abs(float(fd_k)), abs=1e-6)
    assert float(lk.knee_hip_coupling(th, tk)) == pytest.approx(float(fd_h), abs=1e-6)


# ------------------------------------------------------- feasibility / stroke consistency
def test_hip_rod_length_spans_stroke_window_over_rom():
    """The measured ROM must correspond to rod lengths inside the physical stroke window.
    This is the ROM-vs-stroke consistency check from the plan: it fails if the anchor
    estimate, the attachment radius, the ROM, or the stroke contradict each other."""
    lam = lk.hip_actuator_length(_hip_grid())
    assert float(lam.min()) >= lk.HIP_LEN_MIN_M - LEN_TOL_M
    assert float(lam.max()) <= lk.HIP_LEN_MAX_M + LEN_TOL_M
    # and the ROM should USE most of the stroke (not a sliver of it):
    assert float(lam.max() - lam.min()) >= 0.85 * dims.HIP_ACTUATOR["stroke_m"]


def test_hip_length_monotonic_and_no_dead_center():
    lam = lk.hip_actuator_length(_hip_grid())
    assert bool((lam.diff() > 0).all()), "rod length must be monotonic over the ROM"
    ma = lk.hip_moment_arm(_hip_grid())
    assert float(ma.min()) > 0.05, "moment arm collapses inside the ROM (dead center)"


def test_knee_rod_length_inside_window_at_hip_default():
    th = torch.full((64,), lk.hip_default_rad(), dtype=torch.float64)
    lam = lk.knee_actuator_length(th, _knee_grid())
    assert float(lam.min()) >= lk.KNEE_LEN_MIN_M - LEN_TOL_M
    assert float(lam.max()) <= lk.KNEE_LEN_MAX_M + LEN_TOL_M
    assert bool((lam.diff() > 0).all())


def test_knee_folded_corner_is_unreachable():
    """Hip fully extended down + knee fully folded exceeds the rod's max length: the box
    joint limits over-approximate the reachable set. The action-term rod-length clamp is
    what enforces this physically."""
    corner = lk.knee_actuator_length(
        torch.tensor(HIP_ROM[1], dtype=torch.float64),
        torch.tensor(KNEE_ROM[1], dtype=torch.float64),
    )
    assert float(corner) > lk.KNEE_LEN_MAX_M


def test_knee_coupling_sign_and_magnitude():
    """Extending the hip (femur down) lengthens the knee rod => at constant rod length the
    knee unfolds. The coupling is a sizeable fraction of the knee's own lever."""
    th = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    tk = torch.tensor(lk.knee_default_left_rad(), dtype=torch.float64)
    coupling = float(lk.knee_hip_coupling(th, tk))
    assert coupling > 0
    assert coupling == pytest.approx(0.0692, abs=0.01)


# ------------------------------------------------------------------------- capability
def test_hip_capability_characterization():
    """Headline sim-to-real correction: ~240 N*m peak but only ~0.23 rad/s at the default
    pose (vs the old rotary model's 1500 N*m / 6 rad/s)."""
    th = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    assert float(lk.hip_torque_limit(th)) == pytest.approx(240.0, abs=15.0)
    assert float(lk.hip_joint_vel_limit(th)) == pytest.approx(0.235, abs=0.02)
    grid_tau = lk.hip_torque_limit(_hip_grid())
    assert float(grid_tau.min()) > 100.0


def test_knee_capability_characterization():
    th = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    tk = torch.tensor(lk.knee_default_left_rad(), dtype=torch.float64)
    assert float(lk.knee_torque_limit(th, tk)) == pytest.approx(36.0, abs=5.0)
    assert float(lk.knee_joint_vel_limit(th, tk)) == pytest.approx(0.46, abs=0.08)


# ---------------------------------------------------------------------------- defaults
def test_mid_stroke_defaults_inside_soft_limits():
    factor = 0.9  # crab_hex_scene_cfg soft_joint_pos_limit_factor
    for default, (lo, hi) in (
        (lk.hip_default_rad(), HIP_ROM),
        (lk.knee_default_left_rad(), KNEE_ROM),
    ):
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        assert mid - factor * half < default < mid + factor * half


def test_defaults_are_mid_stroke_exactly():
    hip_mid = (lk.HIP_LEN_MIN_M + lk.HIP_LEN_MAX_M) / 2.0
    th = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    assert float(lk.hip_actuator_length(th)) == pytest.approx(hip_mid, abs=1e-9)
    knee_mid = (lk.KNEE_LEN_MIN_M + lk.KNEE_LEN_MAX_M) / 2.0
    tk = torch.tensor(lk.knee_default_left_rad(), dtype=torch.float64)
    assert float(lk.knee_actuator_length(th, tk)) == pytest.approx(knee_mid, abs=1e-9)
    assert lk.knee_default_right_rad() == -lk.knee_default_left_rad()


def test_batched_and_broadcastable():
    """Matches how the action term will call this every substep across envs/legs.

    Samples stay inside the physical ROMs (fixed seed): the closed-form inverse is only
    defined on the mechanism's operating branch, which covers the ROM but not arbitrary
    angles — unseeded out-of-ROM samples made this flaky.
    """
    gen = torch.Generator().manual_seed(20260820)
    u_h = torch.rand(256, 6, generator=gen, dtype=torch.float64)
    u_k = torch.rand(256, 6, generator=gen, dtype=torch.float64)
    th = HIP_ROM[0] + (HIP_ROM[1] - HIP_ROM[0]) * u_h
    tk = KNEE_ROM[0] + (KNEE_ROM[1] - KNEE_ROM[0]) * u_k
    lam = lk.knee_actuator_length(th, tk)
    assert lam.shape == th.shape
    assert torch.isfinite(lam).all()
    back = lk.knee_angle_from_length(lam, th)
    assert torch.allclose(back, tk, atol=1e-8)
