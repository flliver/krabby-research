"""The omni task's four-image symmetry maps (omni plan D5/A8): L/R, fore-aft, and R_z(pi).

Wrong signs here are invisible: training still runs, the mirror loss still falls, and the
policy quietly learns a symmetry the robot does not have. So the two 17-dim head tables are
checked THREE ways -- against the plan literals, against an independently rebuilt root state
(quaternion reflection + world pseudo-/true-vector rules, with roll/pitch coming from the
eval harness's own decomposition), and against the leg kinematics themselves -- and the
composed image is checked to be the elementwise product AND the real 180-degree rotation.

Pure torch + numpy. Every module is loaded BY FILE PATH under a private name: nothing is
registered under ``parkour_isaaclab.*`` / ``isaaclab.*``, so this file is independent of
whatever other unit modules have already stubbed (``test_crab_hex_omni_command.py`` installs
synthetic ``parkour_isaaclab`` packages; collection order must not matter).
"""

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI_MIRROR_PY = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_mirror.py"
FK_PY = TASKS / "crab_hex_forward_task/mdp/crab_hex_foot_fk.py"
METRICS_PY = TASKS / "crab_hex_forward_task/scripts/gait_eval/metrics.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


omni = _load(OMNI_MIRROR_PY, "crab_hex_omni_mirror__test")
fk = _load(FK_PY, "crab_hex_foot_fk__omni_test")
metrics = _load(METRICS_PY, "gait_eval_metrics__omni_test")

_fwd = omni._fwd    # the forward (L/R) mirror module, path-loaded by the module under test
_side = omni._side  # the sideways (fore-aft) mirror module

MY = np.diag([1.0, -1.0, 1.0])   # reflection through the y = 0 plane (L/R)
MX = np.diag([-1.0, 1.0, 1.0])   # reflection through the x = 0 plane (fore-aft)
RZ_PI = MX @ MY                  # = diag(-1, -1, 1)

# The real crab-hex layout: 24 joints, 18 actions, 6 footpads, the 9x12 scanner grid and the
# 10-step history -- so ``obs_dim`` below is the shipped 1149.
LEGS = ["FL", "FR", "ML", "MR", "RL", "RR"]
JOINT_TYPES = [
    "_Body_CamShaft_RevoluteJoint",
    "_Body_Hip_RevoluteJoint",
    "_Hip_Femur_RevoluteJoint",
    "_Femur_Tibia_RevoluteJoint",
]
JOINT_NAMES = [leg + jt for jt in JOINT_TYPES for leg in LEGS]
ACTION_NAMES = [
    leg + jt
    for jt in ("_Body_CamShaft_RevoluteJoint", "_Hip_Femur_RevoluteJoint",
               "_Femur_Tibia_RevoluteJoint")
    for leg in LEGS
]
CONTACT_NAMES = [leg + "_Footpad" for leg in LEGS]
SCAN_SIZE = (2.4, 3.4)
SCAN_RES = 0.3
NX, NY = 9, 12
HISTORY = 10
N_PROP = 17 + 24 + 24 + 18 + 6            # 89
OBS_DIM = N_PROP + NX * NY + 9 + 5 + 48 + HISTORY * N_PROP   # 1149


# --------------------------------------------------------------------------- fake env
class _Pattern:
    size = SCAN_SIZE
    resolution = SCAN_RES
    ordering = "xy"


class _Scanner:
    class cfg:
        pattern_cfg = _Pattern()


class _Contacts:
    body_names = CONTACT_NAMES + ["body"]      # _resolve_layout filters to *_Footpad


class _ActionTerm:
    _joint_names = ACTION_NAMES


class _Robot:
    joint_names = JOINT_NAMES


class _Scene:
    sensors = {"contact_forces": _Contacts(), "height_scanner": _Scanner()}

    def __getitem__(self, key):
        assert key == "robot"
        return _Robot()


class _ActionManager:
    def get_term(self, name):
        assert name == "joint_pos"
        return _ActionTerm()


class _ObsTerm:
    """The omni observation term as the env cfg declares it -- the params the mirror reads."""

    def __init__(self, encoding="raw", *, declared=True):
        self.params = {"history_length": 10, "facing_obs_clip": 0.6}
        if declared:
            self.params["facing_obs_encoding"] = encoding


class _EnvCfg:
    def __init__(self, encoding="raw", *, declared=True):
        self.observations = types.SimpleNamespace(
            policy=types.SimpleNamespace(
                extreme_parkour_observations=_ObsTerm(encoding, declared=declared)
            )
        )


class _Env:
    """The live env the mirror resolves its layout AND its facing encoding from."""

    scene = _Scene()
    action_manager = _ActionManager()
    device = "cpu"

    def __init__(self, encoding="raw", *, declared=True, with_cfg=True):
        if with_cfg:
            self.cfg = _EnvCfg(encoding, declared=declared)


@pytest.fixture()
def mirrors():
    """The shipped builder on the shipped layout, ``raw`` encoding -- ``(lr, fa)``."""
    return omni.build_mirrors(_Env(), OBS_DIM)


@pytest.fixture()
def sincos_mirrors():
    """The same builder on a head whose obs term declares ``facing_obs_encoding="sincos"``."""
    return omni.build_mirrors(_Env("sincos"), OBS_DIM)


# --------------------------------------------------------------------------- the tables
def test_head_tables_are_the_plan_literals():
    """Both encodings, pinned as literals -- the module constants stay the ``raw`` ones."""
    assert omni.HEAD_SIGNS_LR == [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
                                  1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0]
    assert omni.HEAD_SIGNS_FA == [1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                                  1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    assert len(omni.HEAD_SIGNS_LR) == len(omni.HEAD_SIGNS_FA) == 17 == omni.N_HEAD
    assert all(abs(s) == 1.0 for s in omni.HEAD_SIGNS_LR + omni.HEAD_SIGNS_FA)

    # ``raw`` -> the constants themselves, as copies (a caller must not be able to mutate the
    # module tables through the return value)
    lr, fa, rot = omni.head_signs("raw")
    assert lr == omni.HEAD_SIGNS_LR and fa == omni.HEAD_SIGNS_FA
    assert rot == omni.HEAD_SIGNS_ROT_Z_PI
    assert lr is not omni.HEAD_SIGNS_LR and fa is not omni.HEAD_SIGNS_FA
    assert omni.head_signs() == omni.head_signs("raw"), "raw is the default"

    # ``sincos`` -> slot 6 = sin(err) keeps -1 in BOTH tables (sin is odd, and both planes
    # negate the error); slot 7 = cos(err) becomes +1 in BOTH (cos is even). Nothing else moves.
    s_lr, s_fa, s_rot = omni.head_signs("sincos")
    assert s_lr[6] == s_fa[6] == -1.0, "sin(err) is odd: the raw -1 survives"
    assert s_lr[7] == s_fa[7] == +1.0, "cos(err) is even: the one entry that changes"
    assert {i for i in range(17) if s_lr[i] != omni.HEAD_SIGNS_LR[i]} == {7}
    assert {i for i in range(17) if s_fa[i] != omni.HEAD_SIGNS_FA[i]} == {7}
    # R_z(pi) maps err -> err, so both facing slots are +1 -- and the composed table is
    # ENCODING-INDEPENDENT, because the two planes' slot-7 entries flip together
    assert s_rot == [a * b for a, b in zip(s_lr, s_fa)]
    assert s_rot[6] == s_rot[7] == +1.0
    assert s_rot == omni.HEAD_SIGNS_ROT_Z_PI
    assert omni.FACING_OBS_ENCODINGS == ("raw", "sincos")
    assert (omni.FACING_SIN_SLOT, omni.FACING_COS_SLOT) == (6, 7)


@pytest.mark.parametrize("bad", ["", "sin_cos", "RAW ", "cos", None, 1])
def test_head_signs_refuses_an_unknown_encoding(bad):
    """No default: the wrong table is a silent training corruption, so it must raise."""
    if bad == "RAW ":
        assert omni.head_signs(bad) == omni.head_signs("raw"), "case/space tolerated"
        return
    with pytest.raises(ValueError, match="encoding"):
        omni.head_signs(bad)


def test_build_mirrors_reads_the_encoding_off_the_observation_term():
    """The mirror and the observation are two halves of one contract, resolved per env."""
    raw_lr, raw_fa = omni.build_mirrors(_Env("raw"), OBS_DIM)
    sc_lr, sc_fa = omni.build_mirrors(_Env("sincos"), OBS_DIM)
    for m, table in ((raw_lr, omni.HEAD_SIGNS_LR), (raw_fa, omni.HEAD_SIGNS_FA)):
        assert m.obs_sign[:17].tolist() == table
    s_lr, s_fa, _ = omni.head_signs("sincos")
    assert sc_lr.obs_sign[:17].tolist() == s_lr
    assert sc_fa.obs_sign[:17].tolist() == s_fa
    # exactly one head slot differs between the two builds, in each plane
    for a, b in ((raw_lr, sc_lr), (raw_fa, sc_fa)):
        differ = {i for i in range(17) if a.obs_sign[i] != b.obs_sign[i]}
        assert differ == {7}
    # a term that simply does not carry the key is the pre-A2 cfg: "raw", not a refusal
    legacy_lr, _ = omni.build_mirrors(_Env(declared=False), OBS_DIM)
    assert legacy_lr.obs_sign[:17].tolist() == omni.HEAD_SIGNS_LR


def test_build_mirrors_refuses_an_unresolvable_observation_cfg():
    """Never a silent ``raw``: an env whose obs cfg cannot be walked is a REFUSAL.

    The failure this prevents is invisible: a ``sincos`` head mirrored with the raw table
    learns that its cosine channel flips under a reflection it does not flip under, and every
    training curve (including the mirror loss, which falls happily) looks normal.
    """
    for env in (_Env(with_cfg=False),                       # no cfg at all
                types.SimpleNamespace(cfg=types.SimpleNamespace()),          # no observations
                types.SimpleNamespace(cfg=types.SimpleNamespace(             # no policy group
                    observations=types.SimpleNamespace())),
                types.SimpleNamespace(cfg=types.SimpleNamespace(             # no term
                    observations=types.SimpleNamespace(policy=types.SimpleNamespace())))):
        with pytest.raises(ValueError, match="facing"):
            omni._facing_obs_encoding(env)
    with pytest.raises(ValueError, match="facing"):
        omni.build_mirrors(_Env(with_cfg=False), OBS_DIM)
    # an unknown spelling in a resolvable cfg raises too (via head_signs), rather than
    # falling back to either table
    with pytest.raises(ValueError, match="encoding"):
        omni.build_mirrors(_Env("sin_cos"), OBS_DIM)
    # ... and a wrapped env is resolved through ``unwrapped``, like the layout is
    wrapped = types.SimpleNamespace(unwrapped=_Env("sincos"))
    assert omni._facing_obs_encoding(wrapped) == "sincos"


def test_difference_from_the_two_sibling_tables():
    """Which slots' SIGNS actually move, vs the plan's "{5, 10}" / "{5, 8, 10}".

    The plan (D5) names slots 5 and 10 for the L/R table and 5, 8, 10 for the fore-aft one.
    Those are the slots whose MEANING the omni observation changes (a dead ``0 * delta_yaw``
    becomes the yaw-rate command; ``cmd_vx`` becomes ``||cmd_xy||``; a dead ``0 * cmd_vx``
    becomes the real ``cmd_vx``). The SIGNS coincide at all of them but slot 8 of the
    fore-aft table -- a yaw pseudo-scalar wants the -1 the forward head's dead slot already
    carried, and a norm wants the +1 the forward ``cmd_vx`` carried under the L/R plane.
    Pinned as a fact so a later edit cannot drift either table by accident.
    """
    differ_lr = {i for i in range(17) if _fwd._HEAD_SIGNS[i] != omni.HEAD_SIGNS_LR[i]}
    differ_fa = {i for i in range(17) if _side.HEAD_SIGNS[i] != omni.HEAD_SIGNS_FA[i]}
    assert differ_lr == set()
    assert differ_fa == {8}
    # ... and slot 8 is exactly the one the sideways head left structurally zero.
    assert _side.HEAD_SIGNS[8] == 1.0 and omni.HEAD_SIGNS_FA[8] == -1.0


def test_the_sibling_modules_are_loaded_once():
    """The sideways module reuses the forward one this module already put in sys.modules."""
    assert _side._fwd is _fwd
    assert sys.modules["crab_hex_mirror__forward"] is _fwd


def test_entry_point_matches_the_dotted_string_the_runner_cfg_uses():
    dotted = (
        "parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:"
        "crab_hex_omni_symmetry_augmentation"
    )
    mod_path, _, func_name = dotted.partition(":")
    on_disk = str(OMNI_MIRROR_PY.relative_to(TASKS.parent))[: -len(".py")]
    assert mod_path.replace(".", "/") == on_disk
    assert callable(getattr(omni, func_name))


def test_plane_constants_are_borrowed_not_retyped():
    assert omni.LEG_SWAP_LR is _fwd._LEG_SWAP
    assert omni.JOINT_SIGN_RULES_LR is _fwd._JOINT_SIGN_RULES
    assert omni.LEG_SWAP_FA is _side.LEG_SWAP
    assert omni.JOINT_SIGN_RULES_FA is _side.JOINT_SIGN_RULES
    assert omni.VEC3_SIGNS_FA == _side.VEC3_SIGNS == (-1.0, 1.0, 1.0)
    assert omni.VEC3_SIGNS_LR == (1.0, -1.0, 1.0)
    assert (omni.SCAN_AXIS_LR, omni.SCAN_AXIS_FA) == ("y", "x")
    # the knee is the sign the two planes disagree on
    assert dict(omni.JOINT_SIGN_RULES_LR)["_Femur_Tibia_"] == -1.0
    assert dict(omni.JOINT_SIGN_RULES_FA)["_Femur_Tibia_"] == 1.0


def test_layout_is_the_shipped_observation(mirrors):
    lr, fa = mirrors
    assert lr.n_prop == fa.n_prop == 89
    assert lr.obs_dim == fa.obs_dim == 1149
    assert lr.act_perm.shape == fa.act_perm.shape == (18,)


# --------------------------------------------------------------------------- algebra
def test_each_image_is_an_involution(mirrors):
    lr, fa = mirrors
    x = torch.randn(5, OBS_DIM)
    a = torch.randn(5, 18)
    for m in (lr, fa):
        assert torch.equal(m.mirror_obs(m.mirror_obs(x)), x)
        assert torch.equal(m.mirror_actions(m.mirror_actions(a)), a)
        assert not torch.equal(m.mirror_obs(x), x)


def test_the_two_planes_commute(mirrors):
    lr, fa = mirrors
    x = torch.randn(4, OBS_DIM)
    a = torch.randn(4, 18)
    assert torch.equal(fa.mirror_obs(lr.mirror_obs(x)), lr.mirror_obs(fa.mirror_obs(x)))
    assert torch.equal(fa.mirror_actions(lr.mirror_actions(a)),
                       lr.mirror_actions(fa.mirror_actions(a)))


def test_composition_is_the_elementwise_product_map(mirrors):
    """M_x M_y is a single perm+sign map: the composed permutation with the product signs."""
    lr, fa = mirrors
    comp_perm = lr.obs_perm[fa.obs_perm]            # apply fa's gather to lr's gather
    comp_sign = lr.obs_sign[fa.obs_perm] * fa.obs_sign
    x = torch.randn(3, OBS_DIM)
    assert torch.equal(fa.mirror_obs(lr.mirror_obs(x)), x[..., comp_perm] * comp_sign)
    # the head block is not permuted, so its composed signs are the plain product
    assert comp_sign[:17].tolist() == omni.HEAD_SIGNS_ROT_Z_PI
    assert omni.HEAD_SIGNS_ROT_Z_PI == [
        -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
        -1.0, -1.0,
    ]


def test_composition_is_the_180_degree_leg_relabelling(mirrors):
    """FL<->RR, FR<->RL, ML<->MR -- and the knee is the one joint sign that survives."""
    lr, fa = mirrors

    def probe(src, dst):
        a = torch.zeros(1, 18)
        a[0, ACTION_NAMES.index(src)] = 1.0
        return fa.mirror_actions(lr.mirror_actions(a))[0, ACTION_NAMES.index(dst)].item()

    for src_leg, dst_leg in (("FL", "RR"), ("FR", "RL"), ("ML", "MR"), ("MR", "ML")):
        # cam shaft: (-1) * (-1) = +1 under the composition (a rotation does not flip yaw)
        assert probe(src_leg + "_Body_CamShaft_RevoluteJoint",
                     dst_leg + "_Body_CamShaft_RevoluteJoint") == 1.0
        assert probe(src_leg + "_Hip_Femur_RevoluteJoint",
                     dst_leg + "_Hip_Femur_RevoluteJoint") == 1.0
        # knee: (-1) * (+1) = -1 -- the composed pair sits on opposite sides of the body,
        # so the 180-degree Z USD flip the right legs carry applies exactly once
        assert probe(src_leg + "_Femur_Tibia_RevoluteJoint",
                     dst_leg + "_Femur_Tibia_RevoluteJoint") == -1.0


def test_contact_pattern_tripod_sets_under_each_image(mirrors):
    """Set A = {FL, MR, RL}: swapped by M_y, fixed by M_x, swapped by the composition."""
    lr, fa = mirrors
    base = 17 + 48 + 18                       # contact fill block, order FL,FR,ML,MR,RL,RR
    set_a = [1, 0, 0, 1, 1, 0]
    set_b = [0, 1, 1, 0, 0, 1]
    x = torch.zeros(1, OBS_DIM)
    x[0, base:base + 6] = torch.tensor(set_a, dtype=torch.float32)
    assert lr.mirror_obs(x)[0, base:base + 6].tolist() == set_b
    assert fa.mirror_obs(x)[0, base:base + 6].tolist() == set_a
    assert fa.mirror_obs(lr.mirror_obs(x))[0, base:base + 6].tolist() == set_b
    # the clock sign follows: -1 where the sets swap (schedule advanced by pi), +1 where not
    assert (omni.HEAD_SIGNS_LR[15], omni.HEAD_SIGNS_LR[16]) == (-1.0, -1.0)
    assert (omni.HEAD_SIGNS_FA[15], omni.HEAD_SIGNS_FA[16]) == (1.0, 1.0)
    assert omni.HEAD_SIGNS_ROT_Z_PI[15] == omni.HEAD_SIGNS_ROT_Z_PI[16] == -1.0


# --------------------------------------------------------------------------- entry point
def test_four_image_ordering_is_identical_for_obs_and_actions(mirrors):
    lr, fa = mirrors
    env = _Env()
    omni._CACHE.pop(id(env), None)
    b = 7
    obs = torch.randn(b, OBS_DIM)
    act = torch.randn(b, 18)
    obs_out, act_out = omni.crab_hex_omni_symmetry_augmentation(
        obs=obs, actions=act, env=env, obs_type="policy"
    )
    assert obs_out.shape == (4 * b, OBS_DIM)
    assert act_out.shape == (4 * b, 18)
    assert int(obs_out.shape[0] / b) == int(act_out.shape[0] / b) == 4   # the fork's num_aug
    want_obs = [obs, lr.mirror_obs(obs), fa.mirror_obs(obs), fa.mirror_obs(lr.mirror_obs(obs))]
    want_act = [act, lr.mirror_actions(act), fa.mirror_actions(act),
                fa.mirror_actions(lr.mirror_actions(act))]
    for k, (wo, wa) in enumerate(zip(want_obs, want_act)):
        assert torch.equal(obs_out[k * b:(k + 1) * b], wo), omni.IMAGE_NAMES[k]
        assert torch.equal(act_out[k * b:(k + 1) * b], wa), omni.IMAGE_NAMES[k]
    assert omni.IMAGE_NAMES == ("identity", "M_y", "M_x", "M_x M_y")


def test_actions_only_call_reuses_the_cached_mirrors():
    """The fork's second call passes obs=None; it must produce the same 4 images."""
    env = _Env()
    omni._CACHE.pop(id(env), None)
    with pytest.raises(RuntimeError):
        omni.crab_hex_omni_symmetry_augmentation(obs=None, actions=torch.randn(2, 18), env=env)
    obs = torch.randn(3, OBS_DIM)
    omni.crab_hex_omni_symmetry_augmentation(obs=obs, actions=None, env=env)
    act = torch.randn(3, 18)
    none_obs, act_out = omni.crab_hex_omni_symmetry_augmentation(obs=None, actions=act, env=env)
    assert none_obs is None
    assert act_out.shape == (12, 18)
    lr, fa = omni._CACHE[id(env)]
    assert torch.equal(act_out[:3], act)
    assert torch.equal(act_out[3:6], lr.mirror_actions(act))
    assert torch.equal(act_out[6:9], fa.mirror_actions(act))
    assert torch.equal(act_out[9:], fa.mirror_actions(lr.mirror_actions(act)))


def test_wrong_obs_width_is_refused():
    env = _Env()
    omni._CACHE.pop(id(env), None)
    omni.crab_hex_omni_symmetry_augmentation(obs=torch.randn(2, OBS_DIM), env=env)
    with pytest.raises(ValueError):
        omni.crab_hex_omni_symmetry_augmentation(obs=torch.randn(2, OBS_DIM - 1), env=env)


# --------------------------------------------------------------------------- kinematics
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _heading(R: np.ndarray) -> float:
    """``heading_w``: the yaw of the labelled body +x axis, as isaaclab computes it."""
    fwd = R @ np.array([1.0, 0.0, 0.0])
    return math.atan2(fwd[1], fwd[0])


@pytest.mark.parametrize("encoding", ["raw", "sincos"])
@pytest.mark.parametrize(
    "label,M,q_reflect,plane",
    [
        ("M_y", MY, lambda q: np.array([q[0], -q[1], q[2], -q[3]]), 0),
        ("M_x", MX, lambda q: np.array([q[0], q[1], -q[2], -q[3]]), 1),
    ],
)
def test_kinematic_head_signs_independent(label, M, q_reflect, plane, encoding):
    """Rebuild every head channel from a root state and from its reflection.

    Nothing here reads the mirror module's tables except the final comparison: the
    quaternion reflection rule (axis ``v' = -M v`` for an improper ``M``), the world
    pseudo-/true-vector rules and the harness's own roll/pitch decomposition do the work.
    Slot 5 is the yaw-rate command (a pseudo-scalar: the P-control output on the facing
    error), slots 6-7 the facing error (or its sine and cosine under ``sincos``), 8-9 the
    body-frame command (a true vector) and slot 10 its norm.

    Run for BOTH facing encodings: the signs come from ``head_signs(encoding)``, and the
    quantity actually written into slots 6/7 changes with it -- which is the whole point, and
    the only place a ``sincos`` head's cosine channel can be caught flipping when it must not.
    """
    signs = omni.head_signs(encoding)[plane]
    rng = np.random.default_rng(20260916)
    for _ in range(64):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q_ref = q_reflect(q)
        R, R_ref = _quat_to_matrix(q), _quat_to_matrix(q_ref)
        assert np.allclose(R_ref, M @ R @ M, atol=1e-12)

        # roll / pitch (slots 3, 4) through the eval harness's decomposition
        roll, pitch = metrics.roll_pitch_from_quat_wxyz(q)
        roll_ref, pitch_ref = metrics.roll_pitch_from_quat_wxyz(q_ref)
        assert roll_ref == pytest.approx(signs[3] * roll, abs=1e-12)
        assert pitch_ref == pytest.approx(signs[4] * pitch, abs=1e-12)

        # angular velocity (slots 0-2, pseudo-vector) and linear velocity (13-14, true)
        ang_w, lin_w = rng.normal(size=3), rng.normal(size=3)
        ang_b, lin_b = R.T @ ang_w, R.T @ lin_w
        ang_b_ref = R_ref.T @ (-M @ ang_w)
        lin_b_ref = R_ref.T @ (M @ lin_w)
        assert np.allclose(ang_b_ref, np.array(signs[0:3]) * ang_b, atol=1e-12)
        assert np.allclose(lin_b_ref[:2], np.array(signs[13:15]) * lin_b[:2], atol=1e-12)

        # facing (slots 5-7). BOTH reflections send a robot-front heading to its negative
        # (for M_x because the front/rear relabelling turns the reflected front around), so
        # the facing target -- also a front heading -- and the yaw negate together and the
        # error negates with them.
        heading, heading_ref = _heading(R), _heading(R_ref)
        assert heading_ref == pytest.approx(-heading, abs=1e-9)
        psi = float(rng.uniform(-1.0, 1.0)) + heading      # |err| <= 1 rad: no wrap corner
        err = psi - heading
        err_ref = (-psi) - heading_ref
        assert err_ref == pytest.approx(-err, abs=1e-12), "both planes negate the error"
        if encoding == "raw":
            # slots 6, 7 both carry the angle itself: a yaw pseudo-scalar
            assert err_ref == pytest.approx(signs[6] * err, abs=1e-12)
            assert err_ref == pytest.approx(signs[7] * err, abs=1e-12)
        else:
            # slot 6 = sin(err) is ODD -> -1; slot 7 = cos(err) is EVEN -> +1
            assert math.sin(err_ref) == pytest.approx(signs[6] * math.sin(err), abs=1e-12)
            assert math.cos(err_ref) == pytest.approx(signs[7] * math.cos(err), abs=1e-12)
            assert (signs[6], signs[7]) == (-1.0, 1.0)
            # and the raw table would be WRONG here: cos does not flip
            assert math.cos(err_ref) != pytest.approx(
                omni.HEAD_SIGNS_LR[7] * math.cos(err), abs=1e-6
            ) or abs(math.cos(err)) < 1e-6
        # slot 5 = the P-control on that error: clip(k * err, +-cap) -- odd, so same sign
        wz = float(np.clip(1.5 * err, -0.3, 0.3))
        wz_ref = float(np.clip(1.5 * err_ref, -0.3, 0.3))
        assert wz_ref == pytest.approx(signs[5] * wz, abs=1e-12)

        # command (slots 8-10): a true vector in the body frame, plus its norm
        cmd_w = np.append(rng.normal(size=2), 0.0)
        cmd_b = R.T @ cmd_w
        cmd_b_ref = R_ref.T @ (M @ cmd_w)
        assert np.allclose(cmd_b_ref[:2], np.array(signs[8:10]) * cmd_b[:2], atol=1e-12)
        assert np.linalg.norm(cmd_b_ref[:2]) == pytest.approx(
            signs[10] * np.linalg.norm(cmd_b[:2]), abs=1e-12
        )
        assert signs[10] == 1.0      # a norm cannot flip


LEG_SWAP_ROT_Z_PI = {"FL": "RR", "FR": "RL", "ML": "MR", "MR": "ML", "RL": "FR", "RR": "FL"}

# ``crab_hex_foot_fk.leg_points`` takes the pitch angles in the LEFT-leg sim convention, i.e.
# with the 180-degree Z USD flip that the right legs carry already undone. In that convention
# every image below negates the hip YAW only (both reflections) or nothing at all (the
# rotation); the knee -1 that distinguishes the L/R sign table from the fore-aft one is a
# convention conversion, not geometry, and is pinned separately by
# ``test_the_knee_sign_is_exactly_the_side_flip``.
FK_ANGLE_SIGNS = {"M_y": (-1.0, 1.0, 1.0), "M_x": (-1.0, 1.0, 1.0), "M_x M_y": (1.0, 1.0, 1.0)}


@pytest.mark.parametrize(
    "label,M,swap",
    [
        ("M_y", MY, omni.LEG_SWAP_LR),
        ("M_x", MX, omni.LEG_SWAP_FA),
        ("M_x M_y", RZ_PI, LEG_SWAP_ROT_Z_PI),
    ],
)
def test_fk_reflection_for_both_planes_and_their_product(label, M, swap):
    """``leg_points(dst, signed angles) == M @ leg_points(src, angles)`` for every leg.

    This is the check that makes the leg maps real reflections of the KINEMATICS rather than
    relabellings that merely happen to be involutions -- including the composed image, whose
    ``det = +1`` makes it the proper rotation R_z(pi).
    """
    sa, sh, sk = FK_ANGLE_SIGNS[label]
    rng = np.random.default_rng(16092026)
    for src in LEGS:
        dst = swap[src]
        for _ in range(16):
            a = float(rng.uniform(-0.4, 0.4))
            h = float(rng.uniform(-0.3, 0.3))
            k = float(rng.uniform(-0.4, 0.4))
            p_src = fk.leg_points(src, a, h, k)
            p_dst = fk.leg_points(dst, sa * a, sh * h, sk * k)
            for key in ("mount", "pivot", "knee", "toe"):
                assert np.allclose(p_dst[key], M @ p_src[key], atol=1e-9), (label, src, key)
    assert np.linalg.det(M) == pytest.approx(-1.0 if label != "M_x M_y" else 1.0)


def test_fk_reflection_pins_each_angle_sign():
    """Every sign is load-bearing: flipping one moves the toe by >= 5 cm."""
    a, h, k = 0.31, 0.17, -0.23
    for label, M, dst, angles in (
        ("M_y", MY, "FR", (-a, h, k)),
        ("M_x", MX, "RL", (-a, h, k)),
        ("M_x M_y", RZ_PI, "RR", (a, h, k)),
    ):
        ref = M @ fk.leg_points("FL", a, h, k)["toe"]
        assert np.allclose(fk.leg_points(dst, *angles)["toe"], ref, atol=1e-9), label
        for i in range(3):
            wrong = list(angles)
            wrong[i] = -wrong[i]
            err = float(np.linalg.norm(fk.leg_points(dst, *wrong)["toe"] - ref))
            assert err > 0.05, (label, i, err)


def test_the_knee_sign_is_exactly_the_side_flip():
    """The knee entry of each SIM sign table is -1 iff the leg map changes side.

    ``_Femur_Tibia_`` is -1 for the L/R plane and +1 for the fore-aft one; the composed map
    is their product, -1. The rule behind all three is one thing: the right legs' 180-degree
    Z USD flip makes the sim knee angle odd across the body, so the sign is -1 exactly when
    the image sends a leg to the other side.
    """
    for label, swap, knee in (
        ("M_y", omni.LEG_SWAP_LR, dict(omni.JOINT_SIGN_RULES_LR)["_Femur_Tibia_"]),
        ("M_x", omni.LEG_SWAP_FA, dict(omni.JOINT_SIGN_RULES_FA)["_Femur_Tibia_"]),
    ):
        changes_side = {src[1] != dst[1] for src, dst in swap.items()}
        assert len(changes_side) == 1, (label, swap)     # all legs agree
        assert knee == (-1.0 if changes_side.pop() else 1.0), label
    composed = (dict(omni.JOINT_SIGN_RULES_LR)["_Femur_Tibia_"]
                * dict(omni.JOINT_SIGN_RULES_FA)["_Femur_Tibia_"])
    assert composed == -1.0
    assert all(src[1] != dst[1] for src, dst in LEG_SWAP_ROT_Z_PI.items())


# --------------------------------------------------------------------------- scan grid
def _grid_xy():
    x = np.arange(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2 + 1e-9, SCAN_RES)
    y = np.arange(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2 + 1e-9, SCAN_RES)
    assert (len(x), len(y)) == (NX, NY)
    # ordering "xy": flat index iy * nx + ix (isaaclab's meshgrid(x, y, indexing="xy"))
    return np.array([[x[ix], y[iy]] for iy in range(NY) for ix in range(NX)])


def test_scan_flip_maps_the_grid_onto_itself_for_both_axes():
    """Both flips are symmetries of the ray grid -- given a scanner centred at (0, 0).

    Exact for x (the 2.4 m span is a whole number of 0.3 m steps). The y span (3.4 m) is
    NOT, so ``arange`` stops at +1.6 and the L/R flip is the true reflection composed with a
    uniform 0.1 m shift -- a property INHERITED from the forward task's scanner (same size,
    same resolution, same flip), pinned here so it is a known quantity and not a surprise.
    """
    xy = _grid_xy()
    perm_x = _fwd.build_scan_permutation(NX, NY, "xy", "x")
    perm_y = _fwd.build_scan_permutation(NX, NY, "xy", "y")

    flipped_x = xy[perm_x]
    assert np.allclose(flipped_x[:, 0], -xy[:, 0], atol=1e-12)      # exact reflection
    assert np.allclose(flipped_x[:, 1], xy[:, 1], atol=1e-12)
    assert set(map(tuple, np.round(flipped_x, 9))) == set(map(tuple, np.round(xy, 9)))

    flipped_y = xy[perm_y]
    assert np.allclose(flipped_y[:, 0], xy[:, 0], atol=1e-12)
    residual = flipped_y[:, 1] + xy[:, 1]
    assert np.allclose(residual, residual[0], atol=1e-12)           # a uniform shift ...
    assert residual[0] == pytest.approx(-0.1, abs=1e-9)             # ... of 0.1 m


def test_required_scanner_offset_lies_on_both_planes():
    """(0, 0, 20) is on x = 0 AND y = 0; the forward scanner's 0.6 m lead is not."""
    assert omni.REQUIRED_SCANNER_OFFSET_XY == (0.0, 0.0)
    scene_cfg = (TASKS / "crab_hex_forward_task/config/crab_hex/crab_hex_scene_cfg.py").read_text()
    assert "pos=(0.6, 0.0, 20.0)" in scene_cfg      # what the omni scene must NOT inherit
    fwd_x = 0.6
    assert fwd_x != 0.0                              # -> the fore-aft plane misses the grid
    for axis_offset in omni.REQUIRED_SCANNER_OFFSET_XY:
        assert axis_offset == 0.0
