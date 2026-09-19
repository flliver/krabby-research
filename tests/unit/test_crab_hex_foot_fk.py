"""Unit tests for the pure-numpy leg FK used by the leg-mount morphology campaign (PLAN G).

Pins the frame chain against the numbers the generator/dimensions module authors, the
outward-splay sign convention (leading row R toes move +x on BOTH sides), and the rigid
mount-transform identity used by the kinematic screen.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

MDP_DIR = (
    Path(__file__).resolve().parents[2]
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

import crab_hex_foot_fk as fk  # noqa: E402


# The legacy golden geometry (2026-08-20 build): the numbers below were transcribed from that
# asset. FK defaults moved to the A15+B plant of record on 2026-09-09, so the base is explicit.
LEGACY = dict(splay_deg=fk.LEGACY_SPLAY_DEG, outer_axis_in=fk.LEGACY_OUTER_AXIS_IN)


def test_zero_pose_matches_generator_geometry():
    pts = fk.leg_points("FL", **LEGACY)
    # legacy generator: FL_Footpad translate (-0.2159, -1.2573, -1.0668); knee_y = -(0.6096+0.0635+0.5842)
    assert pts["toe"] == pytest.approx([-0.2159, -1.2573, -1.0668], abs=2e-4)
    assert pts["knee"] == pytest.approx([-0.2159, -1.2573, -0.2413], abs=2e-4)
    assert pts["pivot"] == pytest.approx([-0.2159, -0.6731, -0.2413], abs=2e-4)
    r = fk.leg_points("RR", **LEGACY)
    assert r["toe"] == pytest.approx([0.2159, 1.2573, -1.0668], abs=2e-4)


def test_default_pose_is_the_plant_of_record():
    """FK defaults = A15+B: outer mounts at +-(14-2.5) in, outer rows splayed 15 deg outward."""
    assert fk.DEFAULT_SPLAY_DEG == 15.0 and fk.DEFAULT_OUTER_AXIS_IN == 2.5
    pts = fk.leg_points("FL")
    assert pts["mount"] == pytest.approx([-0.2921, -0.6096, 0.0], abs=2e-4)
    # toe = mount + R_z(-15 deg) @ (0, -0.6477, -1.0668)
    assert pts["toe"] == pytest.approx([-0.4597, -1.2352, -1.0668], abs=2e-4)
    assert fk.leg_points("ML")["toe"] == pytest.approx(fk.leg_points("ML", **LEGACY)["toe"])  # mid legs unchanged


def test_toe_sweep_at_cam_throw():
    # knee/toe radius from the yaw axis 0.6477 m -> +-0.2737 m fore-aft at +-25 deg
    assert fk.toe_x_sweep_m() == pytest.approx(0.2737, abs=1e-3)
    p = fk.leg_points("ML", yaw=fk.YAW_THROW_RAD)
    m = fk.mount_point("ML")
    assert abs(p["toe"][0] - m[0]) == pytest.approx(0.2737, abs=1e-3)
    assert np.hypot(*(p["toe"][:2] - m[:2])) == pytest.approx(0.6477, abs=1e-3)


@pytest.mark.parametrize("name", ["RL", "RR"])
def test_outward_splay_moves_leading_row_toes_toward_travel(name):
    base = fk.leg_points(name, **LEGACY)["toe"]
    splayed = fk.leg_points(name, splay_deg=20.0, outer_axis_in=fk.LEGACY_OUTER_AXIS_IN)["toe"]
    assert splayed[0] - base[0] == pytest.approx(0.6477 * math.sin(math.radians(20.0)), abs=1e-3)
    assert splayed[0] > base[0]


@pytest.mark.parametrize("name", ["FL", "FR"])
def test_outward_splay_moves_trailing_row_toes_away_from_travel(name):
    base = fk.leg_points(name, **LEGACY)["toe"]
    splayed = fk.leg_points(name, splay_deg=20.0, outer_axis_in=fk.LEGACY_OUTER_AXIS_IN)["toe"]
    assert splayed[0] < base[0]
    assert base[0] - splayed[0] == pytest.approx(0.6477 * math.sin(math.radians(20.0)), abs=1e-3)


def test_mid_legs_never_splay():
    for name in ("ML", "MR"):
        assert fk.leg_points(name, splay_deg=20.0)["toe"] == pytest.approx(fk.leg_points(name)["toe"])


def test_splay_is_left_right_mirror_symmetric():
    l = fk.leg_points("RL", splay_deg=15.0)["toe"]
    r = fk.leg_points("RR", splay_deg=15.0)["toe"]
    assert l[0] == pytest.approx(r[0])
    assert l[1] == pytest.approx(-r[1])
    assert l[2] == pytest.approx(r[2])


def test_splay_preserves_toe_height_and_lateral_reach_loss():
    base = fk.leg_points("RR", **LEGACY)["toe"]
    splayed = fk.leg_points("RR", splay_deg=20.0, outer_axis_in=fk.LEGACY_OUTER_AXIS_IN)["toe"]
    assert splayed[2] == pytest.approx(base[2])  # rotation about a vertical axis
    lost = (base[1] - fk.WALL_Y_M) - (splayed[1] - fk.WALL_Y_M)
    assert lost == pytest.approx(0.6477 * (1.0 - math.cos(math.radians(20.0))), abs=1e-3)  # ~39 mm


def test_outer_axis_move_translates_outer_rows_only():
    for name, sign in (("FL", -1.0), ("RR", 1.0)):
        base = fk.leg_points(name, **LEGACY)["toe"]
        moved = fk.leg_points(name, splay_deg=0.0, outer_axis_in=2.5)["toe"]
        assert moved[0] - base[0] == pytest.approx(sign * 3.0 * fk.dims.IN_TO_M, abs=1e-9)
        assert moved[1:] == pytest.approx(base[1:])
    assert fk.leg_points("MR", outer_axis_in=2.5)["toe"] == pytest.approx(fk.leg_points("MR", **LEGACY)["toe"])


def test_recorded_foot_transform_matches_fk():
    """Rigid mount transform of a recorded foot == FK at the same joints under the variant."""
    for name in fk.LEG_NAMES:
        joints = dict(yaw=0.2, hip=0.15, knee=0.3)
        # legacy-golden recording -> A20+B
        base_toe = fk.leg_points(name, **joints, **LEGACY)["toe"]
        expected = fk.leg_points(name, **joints, splay_deg=20.0, outer_axis_in=2.5)["toe"]
        got = fk.transform_recorded_foot(name, base_toe, splay_deg=20.0, outer_axis_in=2.5,
                                         base_outer_axis_in=fk.LEGACY_OUTER_AXIS_IN, base_splay_deg=fk.LEGACY_SPLAY_DEG)
        assert got == pytest.approx(expected, abs=1e-9)
        # plant-of-record recording (defaults) -> A20+B: only the 5 deg of extra splay
        main_toe = fk.leg_points(name, **joints)["toe"]
        got2 = fk.transform_recorded_foot(name, main_toe, splay_deg=20.0, outer_axis_in=2.5)
        assert got2 == pytest.approx(expected, abs=1e-9)


def test_hip_down_lowers_knee_and_knee_fold_moves_toe_inboard():
    p0 = fk.leg_points("FL", **LEGACY)
    down = fk.leg_points("FL", hip=0.3, **LEGACY)
    assert down["knee"][2] < p0["knee"][2]
    folded = fk.leg_points("FL", knee=0.4, **LEGACY)
    assert abs(folded["toe"][1]) < abs(p0["toe"][1])  # inboard = toward the body (smaller |y|)


def test_splay_cap_and_sideways_reachability():
    with pytest.raises(ValueError):
        fk.mount_yaw("RL", 25.0)
    assert fk.perpendicular_reachable(20.0)
    assert not fk.perpendicular_reachable(30.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --- FK vs one Isaac static (PLAN G rung ii, golden plant) -----------------------------------
def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@pytest.mark.parametrize("fixture", sorted((Path(__file__).parent / "fixtures").glob("crab_hex_*_settle.json")), ids=lambda p: p.stem)
def test_fk_matches_isaac_settled_footpads_within_5mm(fixture):
    """Toe FK reproduces the recorded footpad positions of a settled plant (all six legs;
    right legs negate the raw knee angle per the module convention). Each fixture names its
    plant geometry (``splay_deg`` / ``outer_axis_in``; legacy golden when absent)."""
    fix = json.loads(fixture.read_text())
    geom = dict(splay_deg=float(fix.get("splay_deg", fk.LEGACY_SPLAY_DEG)),
                outer_axis_in=float(fix.get("outer_axis_in", fk.LEGACY_OUTER_AXIS_IN)))
    jn = fix["joint_names"]
    q = np.asarray(fix["joint_pos"])
    R = _quat_to_rot(fix["root_quat_w_wxyz"])
    root = np.asarray(fix["root_pos_w"])
    for leg, foot_w in fix["footpad_pos_w"].items():
        yaw = q[jn.index(f"{leg}_Body_Hip_RevoluteJoint")]
        hip = q[jn.index(f"{leg}_Hip_Femur_RevoluteJoint")]
        knee = q[jn.index(f"{leg}_Femur_Tibia_RevoluteJoint")]
        if leg not in fk.LEFT:
            knee = -knee
        toe = fk.leg_points(leg, yaw, hip, knee, **geom)["toe"]
        foot_b = R.T @ (np.asarray(foot_w) - root)
        assert np.linalg.norm(foot_b - toe) < 5e-3, (leg, foot_b, toe)
