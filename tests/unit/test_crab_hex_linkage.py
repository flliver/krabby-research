"""Unit tests for the crab-hex linear-actuator linkage kinematics.

Pure torch -- no Isaac Sim needed. Pins the closed-form geometry in
``crab_hex_linkage.py``: exact inverses, analytic derivatives vs finite differences, the
feasibility envelope (rod lengths stay inside the physical stroke window across the
measured ROM -- this is the tripwire for a wrong anchor estimate OR an inconsistent
ROM/stroke combination), and both knee mounts: the legacy hip-plate model's hip->knee
coupling sign and joint-space-corner unreachability, and the femur-mounted model the
hardware has (2026-11-24) -- hip-independent, agreeing with the hip plate at the default
pose, and reproducing the robot's belly contact that the hip-plate model cannot.
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
    """LEGACY hip-plate mount only: hip fully extended down + knee fully folded exceeds the
    rod's max length, so the action-term rod-length clamp makes the corner unreachable. The
    femur mount (the hardware) reaches it -- see test_femur_rod_window_covers_the_knee_rom."""
    corner = lk.knee_actuator_length(
        torch.tensor(HIP_ROM[1], dtype=torch.float64),
        torch.tensor(KNEE_ROM[1], dtype=torch.float64),
    )
    assert float(corner) > lk.KNEE_LEN_MAX_M


def test_knee_coupling_sign_and_magnitude():
    """LEGACY hip-plate mount: extending the hip (femur down) lengthens the knee rod => at
    constant rod length the knee unfolds. The coupling is a sizeable fraction of the knee's
    own lever (d(knee)/d(hip) ~ -0.9). Zero on the femur mount."""
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


# ------------------------------------------------------------- knee mount (2026-11-24)
# The owner saw the knee actuator ride on the upper leg: its rear end is on the FEMUR, not
# the hip plate. The femur mount is the hardware; the hip plate stays the default (legacy).
FEMUR = lk.KNEE_MOUNT_FEMUR
HIP_PLATE = lk.KNEE_MOUNT_HIP_PLATE


def test_default_mount_is_the_legacy_hip_plate():
    assert lk.KNEE_MOUNTS == (HIP_PLATE, FEMUR)
    assert lk.DEFAULT_KNEE_MOUNT == HIP_PLATE
    assert lk.check_knee_mount(None) == HIP_PLATE
    for mount in lk.KNEE_MOUNTS:
        assert lk.check_knee_mount(mount) == mount


@pytest.mark.parametrize("bad", ["Femur", "femur ", "hip", "", "plate", 0])
def test_unknown_mounts_are_refused(bad):
    """A misspelled mount must not train on the legacy plant without a word."""
    z = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="knee mount"):
        lk.check_knee_mount(bad)
    for fn in (lk.knee_actuator_length, lk.knee_moment_arm, lk.knee_hip_coupling):
        with pytest.raises(ValueError, match="knee mount"):
            fn(z, z, mount=bad)
    with pytest.raises(ValueError, match="knee mount"):
        lk.knee_angle_from_length(z + 0.5, z, mount=bad)


def test_femur_anchor_is_the_plate_anchor_at_the_hip_default():
    """The femur-frame anchor is the hip-plate estimate rotated into the femur frame at the
    hip default (pending measurement): near the femur's hip end, just past its top edge."""
    th0 = lk.hip_default_rad()
    s, n = lk.KNEE_FEMUR_ANCHOR_S_M, lk.KNEE_FEMUR_ANCHOR_N_M
    assert s * math.cos(th0) - n * math.sin(th0) == pytest.approx(lk.KNEE_ANCHOR_U_M, abs=1e-12)
    assert s * math.sin(th0) + n * math.cos(th0) == pytest.approx(lk.KNEE_ANCHOR_D_M, abs=1e-12)
    assert 0.0 < s < 0.15  # within ~6 in of the hip pivot, along the femur
    assert -0.1 < n < -dims.FEMUR_WIDTH_IN / 2.0 * dims.IN_TO_M + 0.01  # lever side, at the top edge


def test_femur_length_angle_round_trip():
    tk = _knee_grid()
    back = lk.knee_femur_angle_from_length(lk.knee_femur_actuator_length(tk))
    assert torch.allclose(back, tk, atol=1e-10)


def test_femur_moment_arm_matches_finite_difference():
    eps = 1e-7
    for val in (-0.8, -0.2, 0.31, 0.9, 1.4):
        t = torch.tensor(val, dtype=torch.float64)
        fd = (lk.knee_femur_actuator_length(t + eps) - lk.knee_femur_actuator_length(t - eps)) / (2 * eps)
        assert float(lk.knee_femur_moment_arm(t)) == pytest.approx(abs(float(fd)), abs=1e-6)


def test_femur_mount_ignores_the_hip():
    """A held knee rod holds the knee angle whatever the hip does; the coupling is zero."""
    tk = _knee_grid()
    lam = lk.knee_femur_actuator_length(tk)
    for val in (HIP_ROM[0], -0.3, 0.0, lk.hip_default_rad(), 0.4, HIP_ROM[1]):
        th = torch.full_like(tk, val)
        assert torch.equal(lk.knee_actuator_length(th, tk, mount=FEMUR), lam)
        assert torch.equal(lk.knee_moment_arm(th, tk, mount=FEMUR), lk.knee_femur_moment_arm(tk))
        assert torch.equal(lk.knee_hip_coupling(th, tk, mount=FEMUR), torch.zeros_like(tk))
        assert torch.equal(lk.knee_angle_from_length(lam, th, mount=FEMUR), lk.knee_femur_angle_from_length(lam))
    # the contrast: the hip-plate rod length moves with the hip at a fixed knee
    plate = [lk.knee_actuator_length(torch.full_like(tk, v), tk) for v in (HIP_ROM[0], HIP_ROM[1])]
    assert float((plate[1] - plate[0]).abs().min()) > 0.03


def test_mounts_agree_at_the_hip_default():
    """At the default hip both mounts give the same rod length, lever and inverse for every
    knee angle, so the policy's knee command -> rod mapping (always taken at the default hip)
    and the mid-stroke knee default do not depend on the mount."""
    tk = _knee_grid()
    th0 = torch.full_like(tk, lk.hip_default_rad())
    lam = lk.knee_actuator_length(th0, tk)
    assert torch.allclose(lk.knee_femur_actuator_length(tk), lam, atol=1e-12, rtol=0)
    assert torch.allclose(lk.knee_femur_moment_arm(tk), lk.knee_moment_arm(th0, tk), atol=1e-12, rtol=0)
    assert torch.allclose(lk.knee_femur_angle_from_length(lam), lk.knee_angle_from_length(lam, th0), atol=1e-12, rtol=0)
    knee_mid = torch.tensor((lk.KNEE_LEN_MIN_M + lk.KNEE_LEN_MAX_M) / 2.0, dtype=torch.float64)
    assert float(lk.knee_femur_angle_from_length(knee_mid)) == pytest.approx(lk.knee_default_left_rad(), abs=1e-12)
    th1 = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    tk1 = torch.tensor(lk.knee_default_left_rad(), dtype=torch.float64)
    for fn in (lk.knee_torque_limit, lk.knee_joint_vel_limit):
        assert float(fn(th1, tk1, mount=FEMUR)) == pytest.approx(float(fn(th1, tk1)), abs=1e-9)


def test_femur_rod_window_covers_the_knee_rom():
    """The whole knee ROM fits the stroke window at EVERY hip angle (rod extension = curl, as
    the YH8 curls by extending), so the hip-plate model's 'unreachable' corner (hip full down,
    knee fully folded) is reachable -- it is where the robot's tibia meets its belly."""
    lam = lk.knee_femur_actuator_length(_knee_grid())
    assert float(lam.min()) >= lk.KNEE_LEN_MIN_M - LEN_TOL_M
    assert float(lam.max()) <= lk.KNEE_LEN_MAX_M + LEN_TOL_M
    assert bool((lam.diff() > 0).all())
    # no dead centre inside the ROM (the lever shrinks toward the fold stop: ~0.014 m/rad at +85 deg)
    assert float(lk.knee_femur_moment_arm(_knee_grid()).min()) > 0.01, "knee lever collapses inside the ROM"
    corner = lk.knee_actuator_length(
        torch.tensor(HIP_ROM[1], dtype=torch.float64), torch.tensor(KNEE_ROM[1], dtype=torch.float64), mount=FEMUR
    )
    assert float(corner) <= lk.KNEE_LEN_MAX_M


def test_femur_batched_and_broadcastable():
    gen = torch.Generator().manual_seed(20261124)
    th = HIP_ROM[0] + (HIP_ROM[1] - HIP_ROM[0]) * torch.rand(256, 6, generator=gen, dtype=torch.float64)
    tk = KNEE_ROM[0] + (KNEE_ROM[1] - KNEE_ROM[0]) * torch.rand(256, 6, generator=gen, dtype=torch.float64)
    lam = lk.knee_actuator_length(th, tk, mount=FEMUR)
    assert lam.shape == th.shape and torch.isfinite(lam).all()
    assert torch.allclose(lk.knee_angle_from_length(lam, th, mount=FEMUR), tk, atol=1e-8)
    # one row of knee angles against a batch of hips: the hip-plate result's shape
    row = tk[0]
    for fn in (lk.knee_actuator_length, lk.knee_moment_arm, lk.knee_hip_coupling):
        assert fn(th, row, mount=FEMUR).shape == fn(th, row).shape == (256, 6)
    assert lk.knee_angle_from_length(lam[0], th, mount=FEMUR).shape == (256, 6)


# ------------------------------------------ the robot's belly contact (owner, 2026-11-24)
#: Sagittal plane, left-leg convention, origin = femur pivot, +u outboard, +d down. The
#: body's side face and bottom face bound the belly (the pivot hangs below and outboard).
WALL_U_M = -dims.FEMUR_PIVOT_OUTBOARD_IN * dims.IN_TO_M
BELLY_D_M = -dims.FEMUR_PIVOT_BELOW_BODY_IN * dims.IN_TO_M


def _tibia_belly_clearance(theta_h: float, theta_k: float, n: int = 200) -> float:
    """Signed clearance (m) of the tibia plank's belly-side edge from the belly: > 0 clear,
    <= 0 pressing into it. The plank is 5 in wide at the knee, tapering to ~2.5 in at the toe
    (the 0.75 taper volume factor); its centre line runs knee -> toe."""
    phi = theta_h + theta_k  # tibia absolute angle from vertical-down; toe direction (-sin, cos)
    s = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    half_w = (dims.TIBIA_WIDTH_IN / 2.0) * (1.0 - 0.5 * s) * dims.IN_TO_M
    pu = lk.FEMUR_LEN_M * math.cos(theta_h) - s * dims.TIBIA_KNEE_TO_TOE_M * math.sin(phi) - half_w * math.cos(phi)
    pd = lk.FEMUR_LEN_M * math.sin(theta_h) + s * dims.TIBIA_KNEE_TO_TOE_M * math.cos(phi) - half_w * math.sin(phi)
    du, dd = (pu - WALL_U_M).clamp_min(0.0), (pd - BELLY_D_M).clamp_min(0.0)
    inside = (pu <= WALL_U_M) & (pd <= BELLY_D_M)
    return float(torch.where(inside, pd - BELLY_D_M, torch.sqrt(du * du + dd * dd)).min())


def test_belly_contact_with_the_knee_curled_reproduces_the_robot():
    """Robot, 2026-11-24: knee rod held at its curl end, uncurling the hip presses the folded
    tibia into the belly at 941-1003 hip hall counts (176-188 mm of hip rod at the 5.33
    counts/mm measured that day). The femur mount reproduces the contact inside the hip ROM;
    the hip-plate mount, holding the same knee rod, keeps the tibia ~0.1 m clear over the whole
    hip ROM. Folded to the ROM table's 5 deg interior the femur model touches ~25 mm of hip
    travel early (~152 mm at the plank edge); a 10-12.5 deg real fold stop would match."""
    hips = [HIP_ROM[0] + (HIP_ROM[1] - HIP_ROM[0]) * i / 1050 for i in range(1051)]
    tk_curl = KNEE_ROM[1]  # interior 5 deg: the knee's fold stop
    onset = next((th for th in hips if _tibia_belly_clearance(th, tk_curl) <= 0.0), None)
    assert onset is not None, "femur mount: the curled tibia never reaches the belly"
    travel_mm = 1000.0 * (float(lk.hip_actuator_length(torch.tensor(onset, dtype=torch.float64))) - lk.HIP_LEN_MIN_M)
    assert 120.0 < travel_mm <= 188.0, travel_mm  # inside the ROM and no later than the robot's contact

    th0 = torch.tensor(lk.hip_default_rad(), dtype=torch.float64)
    lam_curl = lk.knee_actuator_length(th0, torch.tensor(tk_curl, dtype=torch.float64))  # same in both mounts
    hip_t = torch.tensor(hips, dtype=torch.float64)
    tk_plate = lk.knee_angle_from_length(lam_curl.expand_as(hip_t), hip_t).clamp(*KNEE_ROM)
    plate = min(_tibia_belly_clearance(th, float(tk)) for th, tk in zip(hips, tk_plate))
    assert plate > 0.08, plate  # ~0.10 m (plank edge), ~0.14 m (centre line)
