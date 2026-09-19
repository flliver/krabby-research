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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
