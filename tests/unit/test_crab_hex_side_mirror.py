"""Unit tests for the sideways task's fore-aft mirror maps (mirror plane x = 0).

Same pins as ``test_crab_hex_mirror.py`` for the L/R mirror -- involutions, bijections, sign
conventions, scan flip, contact pattern -- plus the two checks that make the *derivation*
falsifiable rather than transcribed:

* ``test_kinematic_head_signs_independent`` rebuilds the proprio head channels from a reflected
  root state (quaternion ``(w, x, -y, -z)``, world pseudo-/true-vector rules) and compares against
  ``HEAD_SIGNS`` -- the roll/pitch part comes from the eval harness's own
  ``roll_pitch_from_quat_wxyz``, not from the mirror module;
* ``test_fk_reflection_fore_aft`` checks the leg kinematics themselves reflect
  (``leg_points(dst, -yaw, hip, knee) == diag(-1, 1, 1) @ leg_points(src, yaw, hip, knee)``),
  with the L/R reflection as the negative control.

Pure torch + numpy; both mdp dirs go on ``sys.path`` (the sideways module path-loads the forward
one itself).
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FWD_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/mdp"
SIDE_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_sideways_task/mdp"
METRICS_PY = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/gait_eval/metrics.py"
for _d in (FWD_MDP, SIDE_MDP):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import crab_hex_foot_fk as fk  # noqa: E402
import crab_hex_side_mirror as side  # noqa: E402

from crab_hex_side_mirror import (  # noqa: E402
    HEAD_SIGNS,
    JOINT_SIGN_RULES,
    LEG_SWAP,
    VEC3_SIGNS,
    CrabHexSideMirror,
)

_fwd = side._fwd  # the forward mirror module the sideways class is built on (path-loaded)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


metrics = _load(METRICS_PY, "gait_eval_metrics__side_test")

# Same representative layout the forward mirror test uses (the real order is resolved at runtime;
# the builder must work for ANY order, which these exercise).
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
JOINT_TYPES = [
    "_Body_CamShaft_RevoluteJoint",
    "_Body_Hip_RevoluteJoint",
    "_Hip_Femur_RevoluteJoint",
    "_Femur_Tibia_RevoluteJoint",
]
JOINT_NAMES = [leg + jt for jt in JOINT_TYPES for leg in LEGS]  # type-major, 24
ACTION_NAMES = [
    leg + jt
    for jt in ("_Body_CamShaft_RevoluteJoint", "_Hip_Femur_RevoluteJoint", "_Femur_Tibia_RevoluteJoint")
    for leg in LEGS
]  # 18
CONTACT_NAMES = [leg + "_Footpad" for leg in LEGS]
HISTORY = 10
NX, NY = 12, 11

FORE_AFT = {"F": "R", "R": "F", "M": "M"}
MX = np.diag([-1.0, 1.0, 1.0])   # reflection through the x = 0 plane
MY = np.diag([1.0, -1.0, 1.0])   # the L/R reflection (negative control)


def _build(cls):
    return cls(
        joint_names=JOINT_NAMES,
        action_joint_names=ACTION_NAMES,
        contact_body_names=CONTACT_NAMES,
        scan_nx=NX,
        scan_ny=NY,
        scan_ordering="xy",
        history_length=HISTORY,
        device="cpu",
    )


@pytest.fixture()
def mirror() -> CrabHexSideMirror:
    return _build(CrabHexSideMirror)


@pytest.fixture()
def lr_mirror():
    """The forward L/R mirror on the same layout, for the sign contrasts."""
    return _build(_fwd.CrabHexMirror)


# ------------------------------------------------------------------------------ shape / algebra
def test_dims(mirror):
    assert mirror.n_prop == 17 + 24 + 24 + 18 + 6 == 89
    assert mirror.obs_dim == 89 + NX * NY + 9 + (5 + 48) + HISTORY * 89 == 1173


def test_obs_involution(mirror):
    x = torch.randn(7, mirror.obs_dim)
    assert torch.equal(mirror.mirror_obs(mirror.mirror_obs(x)), x)


def test_action_involution(mirror):
    a = torch.randn(5, 18)
    assert torch.equal(mirror.mirror_actions(mirror.mirror_actions(a)), a)


def test_obs_perm_is_bijection(mirror):
    assert sorted(mirror.obs_perm.tolist()) == list(range(mirror.obs_dim))
    assert sorted(mirror.act_perm.tolist()) == list(range(18))


def test_mirror_is_not_identity(mirror):
    x = torch.randn(3, mirror.obs_dim)
    assert not torch.equal(mirror.mirror_obs(x), x)


# ------------------------------------------------------------------------------ leg / joint maps
def test_leg_permutation_is_fore_aft():
    perm = _fwd.build_name_permutation(JOINT_NAMES, leg_swap=LEG_SWAP)
    assert sorted(perm) == list(range(len(JOINT_NAMES)))
    for i, name in enumerate(JOINT_NAMES):
        partner = JOINT_NAMES[perm[i]]
        assert partner[1] == name[1], name          # side letter preserved (L stays L)
        assert partner[0] == FORE_AFT[name[0]], name
        assert partner[2:] == name[2:], name
        if name[:2] in ("ML", "MR"):
            assert perm[i] == i, name               # mid legs are fixed points


def test_joint_sign_table(mirror, lr_mirror):
    assert JOINT_SIGN_RULES == (
        ("_Body_CamShaft_", -1.0),
        ("_Body_Hip_", -1.0),
        ("_Hip_Femur_", 1.0),
        ("_Femur_Tibia_", 1.0),
    )
    assert LEG_SWAP == {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR", "ML": "ML", "MR": "MR"}

    def probe(m, src, dst):
        a = torch.zeros(1, 18)
        a[0, ACTION_NAMES.index(src)] = 1.0
        return m.mirror_actions(a)[0, ACTION_NAMES.index(dst)].item()

    # cam shaft: fore-aft swap, sign flip (yaw-type)
    assert probe(mirror, "FL_Body_CamShaft_RevoluteJoint", "RL_Body_CamShaft_RevoluteJoint") == -1.0
    # hip pitch on a mid leg: fixed point, no flip
    assert probe(mirror, "ML_Hip_Femur_RevoluteJoint", "ML_Hip_Femur_RevoluteJoint") == 1.0
    # knee: +1 here (pitch axis along the plane normal, no F/R USD flip) ...
    assert probe(mirror, "RL_Femur_Tibia_RevoluteJoint", "FL_Femur_Tibia_RevoluteJoint") == 1.0
    # ... where the L/R mirror's knee is -1 (180-deg Z USD flip on the right legs)
    assert probe(lr_mirror, "RL_Femur_Tibia_RevoluteJoint", "RR_Femur_Tibia_RevoluteJoint") == -1.0
    assert _fwd._JOINT_SIGN_RULES[3] == ("_Femur_Tibia_", -1.0)


# ------------------------------------------------------------------------------ proprio head
def test_head_signs_table(mirror):
    assert HEAD_SIGNS == [1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                          -1.0, 1.0, 1.0, 1.0]
    # one-hot probes on the live map (head dims are not permuted, only signed)
    for idx, expected, label in [
        (0, 1.0, "wx"), (1, -1.0, "wy"), (2, -1.0, "wz"),
        (3, 1.0, "roll"), (4, -1.0, "pitch"),
        (6, -1.0, "delta_yaw"), (7, -1.0, "delta_next_yaw"),
        (10, 1.0, "cmd"), (13, -1.0, "lin_vx"), (14, 1.0, "lin_vy"),
        (15, 1.0, "clock_sin"), (16, 1.0, "clock_cos"),
    ]:
        x = torch.zeros(1, mirror.obs_dim)
        x[0, idx] = 1.0
        assert mirror.mirror_obs(x)[0, idx] == expected, label


def test_head_signs_differ_from_lr_mirror_at():
    differ = {i for i in range(17) if _fwd._HEAD_SIGNS[i] != HEAD_SIGNS[i]}
    # wx, wy, roll, pitch, the (always-zero) lateral command slot, lin_vx, lin_vy, clock sin/cos
    assert differ == {0, 1, 3, 4, 9, 13, 14, 15, 16}


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (body -> world) of a unit ``(w, x, y, z)`` quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def test_kinematic_head_signs_independent():
    """Build the head channels from a root state and from its x -> -x reflection, independently
    of the mirror module, and check the ratios are ``HEAD_SIGNS``."""
    rng = np.random.default_rng(20260911)
    for _ in range(64):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        q_ref = np.array([w, x, -y, -z])          # reflection of the rotation through x = 0

        # the reflected orientation really is M R M (this is what makes q_ref the right state)
        R, R_ref = _quat_to_matrix(q), _quat_to_matrix(q_ref)
        assert np.allclose(R_ref, MX @ R @ MX, atol=1e-12)

        # attitude: roll/pitch through the harness's own decomposition
        roll, pitch = metrics.roll_pitch_from_quat_wxyz(q)
        roll_ref, pitch_ref = metrics.roll_pitch_from_quat_wxyz(q_ref)
        assert roll_ref == pytest.approx(HEAD_SIGNS[3] * roll, abs=1e-12)
        assert pitch_ref == pytest.approx(HEAD_SIGNS[4] * pitch, abs=1e-12)

        # angular velocity (pseudo-vector in world: w' = -M w) and linear velocity (true
        # vector: v' = M v), both expressed in the (reflected) body frame
        ang_w, lin_w = rng.normal(size=3), rng.normal(size=3)
        ang_b, lin_b = R.T @ ang_w, R.T @ lin_w
        ang_b_ref = R_ref.T @ (-MX @ ang_w)
        lin_b_ref = R_ref.T @ (MX @ lin_w)
        assert np.allclose(ang_b_ref, np.array(HEAD_SIGNS[0:3]) * ang_b, atol=1e-12)
        assert np.allclose(lin_b_ref[:2], np.array(HEAD_SIGNS[13:15]) * lin_b[:2], atol=1e-12)


def test_contact_pattern_A_is_a_fixed_point(mirror):
    # contact fill block sits at [17+48+18 : 89]; order FL,FR,ML,MR,RL,RR
    base = 17 + 48 + 18
    x = torch.zeros(1, mirror.obs_dim)
    for i, planted in enumerate([1, 0, 0, 1, 1, 0]):  # tripod set A planted (FL, MR, RL)
        x[0, base + i] = float(planted)
    m = mirror.mirror_obs(x)
    # the fore-aft reflection maps set A onto itself (FL<->RL, MR fixed)
    assert m[0, base:base + 6].tolist() == [1, 0, 0, 1, 1, 0]


# ------------------------------------------------------------------------------ scan / priv
@pytest.mark.parametrize("ordering", ["xy", "yx"])
def test_scan_permutation_is_fore_aft_flip(ordering):
    perm = _fwd.build_scan_permutation(NX, NY, ordering, "x")
    assert sorted(perm) == list(range(NX * NY))
    assert [perm[perm[i]] for i in range(NX * NY)] == list(range(NX * NY))
    if ordering == "xy":
        assert perm[3 * NX + 5] == 3 * NX + (NX - 1 - 5)      # iy kept, ix flipped
    else:
        assert perm[5 * NY + 3] == (NX - 1 - 5) * NY + 3


def test_priv_explicit_and_com_flip_x(mirror):
    assert VEC3_SIGNS == (-1.0, 1.0, 1.0)
    base = 89 + NX * NY
    perm = mirror.obs_perm.tolist()
    sign = mirror.obs_sign.tolist()
    # priv_explicit: three body-frame 3-vectors, no permutation, x flips
    assert perm[base:base + 9] == list(range(base, base + 9))
    assert sign[base:base + 9] == [-1.0, 1.0, 1.0] * 3
    # priv_latent head: mass(+1), com(x flips), friction(+1)
    assert sign[base + 9] == 1.0
    assert sign[base + 10:base + 13] == [-1.0, 1.0, 1.0]
    assert sign[base + 13] == 1.0
    assert perm[base + 9:base + 14] == list(range(base + 9, base + 14))


def test_history_slots_use_step_map(mirror):
    hist_base = 89 + NX * NY + 9 + 53
    assert hist_base + HISTORY * 89 == mirror.obs_dim      # layout sanity
    slot = 4
    # head dim 13 (lin_vx) stays in its slot and flips
    x = torch.zeros(1, mirror.obs_dim)
    x[0, hist_base + slot * 89 + 13] = 2.0
    assert mirror.mirror_obs(x)[0, hist_base + slot * 89 + 13] == -2.0
    # joint_pos block inside the same slot permutes fore-aft with the cam-shaft sign
    x = torch.zeros(1, mirror.obs_dim)
    rl_cam = JOINT_NAMES.index("RL_Body_CamShaft_RevoluteJoint")
    fl_cam = JOINT_NAMES.index("FL_Body_CamShaft_RevoluteJoint")
    x[0, hist_base + slot * 89 + 17 + rl_cam] = 1.0
    m = mirror.mirror_obs(x)
    assert m[0, hist_base + slot * 89 + 17 + fl_cam] == -1.0


def test_priv_latent_ratios_permute_without_flip(mirror):
    # stiffness-ratio block: after scan + priv_explicit(9) + mass/com/friction(5)
    base = 89 + NX * NY + 9 + 5
    x = torch.zeros(1, mirror.obs_dim)
    x[0, base + JOINT_NAMES.index("FL_Body_CamShaft_RevoluteJoint")] = 0.7
    m = mirror.mirror_obs(x)
    rl = base + JOINT_NAMES.index("RL_Body_CamShaft_RevoluteJoint")
    assert m[0, rl] == pytest.approx(0.7)      # magnitude ratio: permute, no sign flip


# ------------------------------------------------------------------------------ leg kinematics
@pytest.mark.parametrize("src,dst", [("FL", "RL"), ("FR", "RR"), ("ML", "ML"), ("MR", "MR")])
def test_fk_reflection_fore_aft(src, dst):
    """The mirror's leg map is a real reflection of the kinematics, not just a relabelling.

    The destination angles are built FROM ``JOINT_SIGN_RULES``, so this pins the sign table:
    yaw negates, hip and knee do not. Both members of a pair are on the same side of the body,
    so the 180-deg Z USD flip the right legs carry on their pitch joints is common to both and
    cancels -- which is exactly why the knee sign is +1 here and -1 for the L/R mirror.
    """
    sgn = dict(JOINT_SIGN_RULES)
    # the Whitworth cam map theta_hip = atan2(K sin t, 1 + K cos t) is odd, so the shaft channel
    # carries the same sign as the hip yaw it drives
    assert sgn["_Body_CamShaft_"] == sgn["_Body_Hip_"]
    rng = np.random.default_rng(11092026)
    for _ in range(32):
        a = float(rng.uniform(-0.4, 0.4))
        h = float(rng.uniform(-0.3, 0.3))
        k = float(rng.uniform(-0.4, 0.4))
        p_src = fk.leg_points(src, a, h, k)
        p_dst = fk.leg_points(
            dst, sgn["_Body_Hip_"] * a, sgn["_Hip_Femur_"] * h, sgn["_Femur_Tibia_"] * k
        )
        for key in ("mount", "pivot", "knee", "toe"):
            assert np.allclose(p_dst[key], MX @ p_src[key], atol=1e-9), (src, dst, key, a, h, k)


def test_fk_reflection_pins_each_joint_sign():
    """Every rule is load-bearing: flipping any one of the three breaks the identity by >= 0.1 m."""
    a, h, k = 0.31, 0.17, -0.23
    ref = MX @ fk.leg_points("FL", a, h, k)["toe"]
    assert np.allclose(fk.leg_points("RL", -a, h, k)["toe"], ref, atol=1e-9)
    for label, wrong in (("yaw +1", (a, h, k)), ("hip -1", (-a, -h, k)), ("knee -1", (-a, h, -k))):
        err = float(np.linalg.norm(fk.leg_points("RL", *wrong)["toe"] - ref))
        assert err > 0.1, (label, err)


def test_fk_reflection_is_not_the_lr_reflection():
    """FR is the *y*-reflection of FL; taking it for the x-reflection would be a whole body
    length out (the sign error the fore-aft mirror must not make)."""
    a, h, k = 0.31, 0.17, -0.23
    fl = fk.leg_points("FL", a, h, k)["toe"]
    fr = fk.leg_points("FR", -a, h, k)["toe"]
    assert np.allclose(MY @ fl, fr, atol=1e-9)
    assert not np.allclose(MX @ fl, fr, atol=1e-3)
    assert np.linalg.norm(MX @ fl - fr) > 0.5
    # and the fore-aft partner of FL is RL, a body length away from FL
    rl = fk.leg_points("RL", -a, h, k)["toe"]
    assert abs(rl[0] - fl[0]) > 0.3 and math.copysign(1.0, rl[1]) == math.copysign(1.0, fl[1])
