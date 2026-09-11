"""Unit tests for the support-polygon / fall-direction metrics (PLAN G, 2026-09-02).

Pure numpy. Known geometries: a rectangular six-foot stance around a centred CoM, a tripod
stance whose forward edge is a diagonal, a CoM projected outside the polygon, and
quaternion-encoded nose-down / roll attitudes.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gait_eval import metrics as M  # noqa: E402

DT = 0.02
# foot order FL, FR, ML, MR, RL, RR; robot walks +x; F at -x, R at +x (generator convention)
FEET = np.array([
    [-0.5, -0.8, 0.0], [-0.5, 0.8, 0.0],
    [0.0, -0.9, 0.0], [0.0, 0.9, 0.0],
    [0.5, -0.8, 0.0], [0.5, 0.8, 0.0],
])


def _quat_yaw_pitch(yaw=0.0, pitch=0.0, roll=0.0):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    # ZYX composition -> (w, x, y, z)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _trace(T, feet, forces, root_z=1.0, yaw=0.0, root_x=0.0):
    foot_pos = np.repeat(feet[None], T, axis=0).astype(float)
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        xy = foot_pos[..., :2].copy()
        foot_pos[..., 0] = c * xy[..., 0] - s * xy[..., 1]
        foot_pos[..., 1] = s * xy[..., 0] + c * xy[..., 1]
    foot_pos[..., 0] += root_x
    force = np.repeat(np.asarray(forces, float)[None], T, axis=0)
    root = np.repeat(np.array([[root_x, 0.0, root_z]]), T, axis=0)
    quat = np.repeat(_quat_yaw_pitch(yaw=yaw)[None], T, axis=0)
    return foot_pos, force, root, quat


def test_six_foot_stance_margins_are_geometric():
    fp, ff, root, quat = _trace(20, FEET, [400] * 6)
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    w = res["walking"]
    assert w["fwd_ray_margin_m"]["p50"] == pytest.approx(0.5, abs=1e-6)
    assert w["fore_aft_span_m"]["p50"] == pytest.approx(1.0, abs=1e-6)
    assert w["com_offset_x_m"]["p50"] == pytest.approx(0.0, abs=1e-6)
    assert w["tip_angle_fwd_deg"]["p50"] == pytest.approx(math.degrees(math.atan2(0.5, 1.0)), abs=1e-6)
    assert w["lead_contact_tip_deg"]["p50"] == pytest.approx(w["tip_angle_fwd_deg"]["p50"], abs=1e-6)
    assert w["frac_neg_margin"] == 0.0
    assert w["frac_underdetermined"] == 0.0


def test_heading_frame_is_yaw_invariant():
    fp, ff, root, quat = _trace(10, FEET, [400] * 6, yaw=1.1)
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    assert res["walking"]["fwd_ray_margin_m"]["p50"] == pytest.approx(0.5, abs=1e-6)


def test_tripod_stance_forward_edge_is_the_diagonal():
    # set B loaded: FR, ML, RR -> forward edge at the sagittal plane is the FR-ML diagonal,
    # halfway between x=-0.5 and x=0 at y=0 ... hull is triangle FR(-0.5,0.8) ML(0,-0.9) RR(0.5,0.8)
    forces = [0, 400, 400, 0, 0, 400]
    fp, ff, root, quat = _trace(10, FEET, forces)
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    w = res["walking"]
    # along y=0: ML->RR edge from (0,-0.9) to (0.5,0.8): x at y=0 is 0.5*0.9/1.7 = 0.2647
    assert w["fwd_ray_margin_m"]["p50"] == pytest.approx(0.5 * 0.9 / 1.7, abs=1e-6)
    assert w["lead_contact_tip_deg"]["p50"] > w["tip_angle_fwd_deg"]["p50"]  # historical def is looser


def test_com_ahead_of_polygon_gives_negative_margin():
    forces = [400, 400, 400, 400, 0, 0]  # only the trailing/mid rows loaded, CoM at x=0.2 ahead of them
    fp, ff, root, quat = _trace(10, FEET, forces)
    root[:, 0] = 0.2
    fp[..., 0] += 0.0  # feet stay put in world; root moved forward
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    w = res["walking"]
    assert w["fwd_ray_margin_m"]["p50"] == pytest.approx(-0.2, abs=1e-6)
    assert w["frac_neg_margin"] == 1.0
    assert w["min_edge_margin_m"]["p50"] < 0.0


def test_underdetermined_frames_are_counted_not_scored():
    forces = [400, 0, 0, 0, 0, 400]
    fp, ff, root, quat = _trace(10, FEET, forces)
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    assert res["walking"]["frac_underdetermined"] == 1.0
    assert res["walking"]["fwd_ray_margin_m"]["n"] == 0


def test_prefall_window_excludes_the_last_steps():
    T = 200
    fp, ff, root, quat = _trace(T, FEET, [400] * 6)
    fail = np.zeros(T, bool)
    fail[150] = True
    res = M.support_polygon_metrics(fp, ff, root, quat, dt=DT, crab_failure=fail, return_series=True)
    mask = res["series"]["prefall_mask"]
    assert mask.sum() == 45  # 1.0 s = 50 steps minus the last 0.1 s = 5 steps
    assert mask[100] and mask[144] and not mask[145] and not mask[150]


def test_fall_direction_classes():
    T = 30
    fail = np.zeros(T, bool)
    fail[20] = True
    q_fwd = np.repeat(_quat_yaw_pitch(pitch=0.49)[None], T, axis=0)
    r = M.fall_direction_metrics(q_fwd, fail, dt=DT)
    assert r["fall_class"] == "pitch_fwd" and r["t_fail_s"] == pytest.approx(0.4)
    assert r["pitch_at_fail"] == pytest.approx(0.49, abs=1e-6)
    q_back = np.repeat(_quat_yaw_pitch(pitch=-0.49)[None], T, axis=0)
    assert M.fall_direction_metrics(q_back, fail, dt=DT)["fall_class"] == "pitch_back"
    q_roll = np.repeat(_quat_yaw_pitch(roll=0.49)[None], T, axis=0)
    assert M.fall_direction_metrics(q_roll, fail, dt=DT)["fall_class"] == "roll"
    assert M.fall_direction_metrics(q_fwd, np.zeros(T, bool), dt=DT)["fall_class"] == "none"
    w = np.zeros((T, 3))
    w[15:21, 1] = 2.5
    assert M.fall_direction_metrics(q_fwd, fail, dt=DT, root_ang_vel_b=w)["max_pitch_rate"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# travel frame (sideways-task harness sharing, 2026-09-11): travel_yaw rotates the failure-step
# quaternion (and the ang-vel xy plane) before the roll/pitch decomposition. Convention pinned:
# at travel_yaw = +pi/2 (body +y is the travel axis) travel-frame roll' = body pitch and
# travel-frame pitch' = -body roll; ang-vel travel-frame (wx', wy') = (wy, -wx).
# ---------------------------------------------------------------------------
def test_fall_direction_travel_yaw_zero_is_unchanged():
    T = 30
    fail = np.zeros(T, bool)
    fail[20] = True
    q = np.repeat(_quat_yaw_pitch(pitch=0.49)[None], T, axis=0)
    w = np.zeros((T, 3))
    w[15:21, 1] = 2.5
    a = M.fall_direction_metrics(q, fail, dt=DT, root_ang_vel_b=w)
    b = M.fall_direction_metrics(q, fail, dt=DT, root_ang_vel_b=w, travel_yaw=0.0)
    assert a == b


def test_fall_direction_in_travel_frame():
    T = 30
    fail = np.zeros(T, bool)
    fail[20] = True

    # +roll fall of the +y-leading robot tips it toward travel-frame nose-back.
    q_roll_pos = np.repeat(_quat_yaw_pitch(roll=0.49)[None], T, axis=0)
    r = M.fall_direction_metrics(q_roll_pos, fail, dt=DT, travel_yaw=math.pi / 2)
    assert r["fall_class"] == "pitch_back"
    assert r["pitch_at_fail"] == pytest.approx(-0.49, abs=1e-6)

    # -roll -> travel-frame nose-fwd.
    q_roll_neg = np.repeat(_quat_yaw_pitch(roll=-0.49)[None], T, axis=0)
    r2 = M.fall_direction_metrics(q_roll_neg, fail, dt=DT, travel_yaw=math.pi / 2)
    assert r2["fall_class"] == "pitch_fwd"
    assert r2["pitch_at_fail"] == pytest.approx(0.49, abs=1e-6)

    # +pitch (body) -> travel-frame roll.
    q_pitch = np.repeat(_quat_yaw_pitch(pitch=0.49)[None], T, axis=0)
    r3 = M.fall_direction_metrics(q_pitch, fail, dt=DT, travel_yaw=math.pi / 2)
    assert r3["fall_class"] == "roll"
    assert r3["roll_at_fail"] == pytest.approx(0.49, abs=1e-6)

    # ang vel: root_ang_vel_b[..., 0] (wx, body roll rate) -> travel-frame pitch rate is -wx.
    w = np.zeros((T, 3))
    w[15:21, 0] = 2.5
    r4 = M.fall_direction_metrics(q_pitch, fail, dt=DT, root_ang_vel_b=w, travel_yaw=math.pi / 2)
    assert r4["max_pitch_rate"] == pytest.approx(2.5)


def _rotate_world_xy(arr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the xy components of ``arr[..., :2]`` by ``angle`` radians in the world frame."""
    c, s = math.cos(angle), math.sin(angle)
    out = arr.copy()
    x, y = arr[..., 0], arr[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out


def _assert_polygon_close(a: dict, b: dict, *, abs_tol: float = 1e-9) -> None:
    for window in ("walking", "prefall"):
        wa, wb = a[window], b[window]
        assert wa["n_frames"] == wb["n_frames"]
        for key in (
            "tip_angle_fwd_deg", "lead_contact_tip_deg", "fwd_ray_margin_m",
            "min_edge_margin_m", "fore_aft_span_m", "com_offset_x_m",
        ):
            for stat in ("p10", "p50", "p90", "n"):
                va, vb = wa[key][stat], wb[key][stat]
                if va is None or vb is None:
                    assert va == vb
                else:
                    assert va == pytest.approx(vb, abs=abs_tol)
        for key in ("frac_neg_margin", "frac_underdetermined"):
            va, vb = wa[key], wb[key]
            if va is None or vb is None:
                assert va == vb
            else:
                assert va == pytest.approx(vb, abs=abs_tol)


def test_support_polygon_travel_yaw_zero_is_unchanged():
    fp, ff, root, quat = _trace(20, FEET, [400] * 6)
    a = M.support_polygon_metrics(fp, ff, root, quat, dt=DT)
    b = M.support_polygon_metrics(fp, ff, root, quat, dt=DT, travel_yaw=0.0)
    assert a == b


def test_support_polygon_travel_yaw_rotates_heading_frame():
    """``metrics(feet, travel_yaw=pi/2)`` == ``metrics(feet rotated by -pi/2 in world, travel_yaw=0)``.

    The heading frame is built from ``yaw_from_quat + travel_yaw``; rotating the world-frame feet
    (and root) by ``-travel_yaw`` ahead of time and leaving ``travel_yaw`` at 0 must land on the
    exact same heading-frame geometry, since the rotation is linear.
    """
    fp, ff, root, quat = _trace(20, FEET, [400] * 6)
    a = M.support_polygon_metrics(fp, ff, root, quat, dt=DT, travel_yaw=math.pi / 2)

    fp_rot = _rotate_world_xy(fp, -math.pi / 2)
    root_rot = _rotate_world_xy(root, -math.pi / 2)
    b = M.support_polygon_metrics(fp_rot, ff, root_rot, quat, dt=DT, travel_yaw=0.0)

    _assert_polygon_close(a, b)


def test_support_polygon_spawn_yaw_invariance():
    """A trace at body yaw -pi/2 with travel_yaw=pi/2 equals a trace at yaw 0 with travel_yaw=pi/2,
    up to the rotation of the feet -- the heading frame is built from ``yaw + travel_yaw``, and
    ``-pi/2 + pi/2 == 0 + pi/2 - pi/2`` is the same total rotation either way.
    """
    fp0, ff0, root0, quat0 = _trace(20, FEET, [400] * 6, yaw=0.0)
    a = M.support_polygon_metrics(fp0, ff0, root0, quat0, dt=DT, travel_yaw=math.pi / 2)

    fp1, ff1, root1, quat1 = _trace(20, FEET, [400] * 6, yaw=-math.pi / 2)
    b = M.support_polygon_metrics(fp1, ff1, root1, quat1, dt=DT, travel_yaw=math.pi / 2)

    _assert_polygon_close(a, b)


def test_orientation_travel_yaw_swaps_roll_and_pitch():
    """At travel_yaw=+pi/2, travel-frame roll_rms/pitch_rms equal the body pitch_rms/roll_rms.

    Samples are single-axis (pure roll OR pure pitch, never both at once): the exact swap
    (roll' == pitch, pitch' == -roll) only holds bit-for-bit when the two are not combined in the
    same sample -- combining them couples the axes under quaternion composition. Verified
    numerically (2026-09-11) before pinning this test.
    """
    T = 20
    rng = np.random.default_rng(0)
    rolls = rng.uniform(-0.3, 0.3, T // 2)
    pitches = rng.uniform(-0.2, 0.2, T // 2)
    quat = np.concatenate(
        [
            np.stack([_quat_yaw_pitch(roll=r) for r in rolls]),
            np.stack([_quat_yaw_pitch(pitch=p) for p in pitches]),
        ]
    )

    body = M.orientation_metrics(quat)
    travel = M.orientation_metrics(quat, travel_yaw=math.pi / 2)
    assert travel["roll_rms"] == pytest.approx(body["pitch_rms"], abs=1e-9)
    assert travel["pitch_rms"] == pytest.approx(body["roll_rms"], abs=1e-9)
    assert travel["roll_max_abs"] == pytest.approx(body["pitch_max_abs"], abs=1e-9)
    assert travel["pitch_max_abs"] == pytest.approx(body["roll_max_abs"], abs=1e-9)

    zero = M.orientation_metrics(quat, travel_yaw=0.0)
    assert zero == body


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
