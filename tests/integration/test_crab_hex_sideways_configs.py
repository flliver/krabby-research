"""The sideways-vs-forward config delta: the acceptance criterion for "nothing else changed".

``crab_hex_sideways_task`` subclasses the forward cfg classes and re-declares every cfg-valued
class attribute with the forward class-body literal (knob timing: ``import_packages`` imports the
forward package first, so forward cfg objects are built BEFORE the sideways activation exports the
``KRABBY_*`` knobs). That makes the sideways package look redundant and makes a silent drift --
an attribute that was not re-declared, a literal that was mistyped, a forward refactor the sibling
did not follow -- invisible in review. This module is the machine check: build every sideways task
per phase inside Isaac Sim, diff its ``class_to_dict`` dump against the forward twin under the
same phase, and assert the set of differing config paths is EXACTLY the travel-axis list below.

The forward dumps come from the imported ``dumps`` fixture of
``tests/integration/test_crab_hex_phase_configs.py`` (that file stays unedited and is itself a
guard); the sideways dumps are produced here by the same ``dump()`` helper.

Opt in (≈ 5 extra Kit boots)::

    RUN_CRAB_HEX_CFG_IDENTITY=1 KRABBY_CFG_DUMP_DIR=<dir> pytest \\
        tests/integration/test_crab_hex_phase_configs.py \\
        tests/integration/test_crab_hex_sideways_configs.py -v

Set ``KRABBY_CFG_DUMP_DIR`` so the forward dumps (``cfg_<fixture>.json``) are reused instead of
re-booting Kit for them; the sideways dumps land beside them as ``cfg_side_<phase>.json``.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import pytest

from tests.integration.test_crab_hex_phase_configs import (  # noqa: F401  -- ``dumps`` is a fixture
    AGENT_VOLATILE,
    FLAT,
    FLAT_PLAY,
    PLANT,
    STUDENT,
    STUDENT_ONLY,
    STUDENT_PLAY,
    STUDENT_SHARED,
    TEACHER,
    TEACHER_PLAY,
    _require,
    diff_paths,
    dump,
    dumps,
    fmt_diffs,
    normalize,
    runtime_resolve,
    strip,
    subtree,
)

pytestmark = pytest.mark.isaacsim

SIDE_WALK = "Isaac-Crab-Hex-Side-Walk-v0"
SIDE_WALK_PLAY = "Isaac-Crab-Hex-Side-Walk-Play-v0"
SIDE_TEACHER = "Isaac-Crab-Hex-Side-Teacher-v0"
SIDE_TEACHER_PLAY = "Isaac-Crab-Hex-Side-Teacher-Play-v0"
SIDE_STUDENT = "Isaac-Crab-Hex-Side-Student-v0"
SIDE_STUDENT_PLAY = "Isaac-Crab-Hex-Side-Student-Play-v0"

# phase -> (env for the dump process, tasks to build). The sideways schedule has no
# ``legacy_golden_*`` and no "no plant" fixture: those are forward reproduction cases.
SIDE_FIXTURES: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "1a": ({"KRABBY_SIDE_PHASE": "1a", "KRABBY_PLANT": PLANT}, (SIDE_WALK, SIDE_WALK_PLAY)),
    "2a": ({"KRABBY_SIDE_PHASE": "2a", "KRABBY_PLANT": PLANT}, (SIDE_TEACHER, SIDE_WALK)),
    "2b": ({"KRABBY_SIDE_PHASE": "2b", "KRABBY_PLANT": PLANT}, (SIDE_TEACHER, SIDE_WALK)),
    "2c": ({"KRABBY_SIDE_PHASE": "2c", "KRABBY_PLANT": PLANT},
           (SIDE_TEACHER, SIDE_WALK, SIDE_TEACHER_PLAY, SIDE_WALK_PLAY)),
    "3a": ({"KRABBY_SIDE_PHASE": "3a", "KRABBY_PLANT": PLANT}, (SIDE_STUDENT, SIDE_STUDENT_PLAY)),
}

# (pair id, phase, forward task, sideways task). The teacher head of a phase-2 window trains in
# the Flat-Walk task in the lineage of record, so both task ids are paired for every phase-2 fixture.
RL_PAIRS = [
    ("1a-walk", "1a", FLAT, SIDE_WALK),
    ("2a-teacher", "2a", TEACHER, SIDE_TEACHER),
    ("2a-walk", "2a", FLAT, SIDE_WALK),
    ("2b-teacher", "2b", TEACHER, SIDE_TEACHER),
    ("2b-walk", "2b", FLAT, SIDE_WALK),
    ("2c-teacher", "2c", TEACHER, SIDE_TEACHER),
    ("2c-walk", "2c", FLAT, SIDE_WALK),
]
STUDENT_PAIR = ("3a-student", "3a", STUDENT, SIDE_STUDENT)
# The PLAY twins: same delta expected (the PLAY heads are applied on top of the train cfgs on both
# sides); no extra Kit boots -- every task below is already in its phase's fixture.
PLAY_PAIRS = [
    ("1a-walk-play", "1a", FLAT_PLAY, SIDE_WALK_PLAY),
    ("2c-teacher-play", "2c", TEACHER_PLAY, SIDE_TEACHER_PLAY),
    ("2c-walk-play", "2c", FLAT_PLAY, SIDE_WALK_PLAY),
    ("3a-student-play", "3a", STUDENT_PLAY, SIDE_STUDENT_PLAY),
]
ALL_PAIRS = RL_PAIRS + [STUDENT_PAIR] + PLAY_PAIRS
STUDENT_TASKS = (STUDENT, STUDENT_PLAY)

# The forward depth-camera rot (``parkour_tasks/default_cfg.py::CAMERA_CFG.offset.rot``, built in
# float32) and the sideways product q_z(+90 deg) * forward. Pinned HERE, not at import of the
# sideways scene cfg (which only warns): optical axis body (0.94, 0, -0.342) -> (0, 0.94, -0.342).
FWD_CAMERA_ROT = (-0.4055798, 0.5792280, -0.5792280, 0.4055798)
SIDE_CAMERA_ROT = (-0.5735764, 0.8191520, 0.0, 0.0)
TOL_CAMERA_ROT_PIN = 1e-6

MISSING = "<missing>"
SQRT_HALF = math.sqrt(0.5)
FWD_MIRROR = "parkour_tasks.crab_hex_forward_task.mdp.crab_hex_mirror:crab_hex_symmetry_augmentation"
SIDE_MIRROR = ("parkour_tasks.crab_hex_sideways_task.mdp.crab_hex_side_mirror:"
               "crab_hex_side_symmetry_augmentation")


class _Present:
    """Leaf sentinel: "this key exists on the forward side and its value is not checked"."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<present>"


PRESENT = _Present()

# ---------------------------------------------------------------------------------------------
# The expected delta (plan Appendix A). Every entry is a deliberate consequence of turning the
# travel axis a quarter turn onto body +y; nothing else about the MDP may differ.
EXPECTED_RL: dict[tuple, tuple] = {
    # the travel frame itself: a field the forward cfg classes do NOT have, so every seam in
    # parkour_isaaclab keeps its identity path (``getattr(env.cfg, "travel_yaw", 0.0)``)
    ("travel_yaw",): (MISSING, math.pi / 2),
    # the command term steers to a WORLD heading; the offset holds the -pi/2 spawn attitude
    ("commands", "base_velocity", "heading_offset"): (MISSING, -math.pi / 2),
    # RSI off: the bank of record is a forward gait (identity yaw), so the presets drop
    # KRABBY_RSI_FRAC / KRABBY_RSI_BANK and the presence-based event term is never created
    ("events", "rsi_reference_reset"): (PRESENT, MISSING),
    # same term NAME, weight and params -- only the function reads the velocity in the travel frame
    ("rewards", "track_lin_vel_xy_exp", "func"): (
        "isaaclab.envs.mdp.rewards:track_lin_vel_xy_exp",
        "parkour_isaaclab.envs.mdp.rewards:track_lin_vel_travel_exp",
    ),
    # the 0.6 m scan look-ahead moves from body +x to body +y (size/resolution/ordering KEPT)
    ("scene", "height_scanner", "offset", "pos", 0): (0.6, 0.0),
    ("scene", "height_scanner", "offset", "pos", 1): (0.0, 0.6),
    # spawn yaw -pi/2 so body +y points down the course (world +x)
    ("scene", "robot", "init_state", "rot", 0): (1.0, SQRT_HALF),
    ("scene", "robot", "init_state", "rot", 3): (0.0, -SQRT_HALF),
}
#: The reward function swap belongs to the RL runners only: the student MDP has no tracking term.
REWARD_FUNC_PATH = ("rewards", "track_lin_vel_xy_exp", "func")
#: Student-only additions: the depth camera placeholder on the +y body face.
CAMERA_POS = (("scene", "depth_camera", "offset", "pos", 0), ("scene", "depth_camera", "offset", "pos", 1))
#: Absolute tolerances for the leaf-value checks (the camera quaternion is a float32 product).
TOL_DEFAULT = 1e-9
TOL_CAMERA_ROT = 1e-6

#: Quarter turn about body +z as (w, x, y, z) -- the sideways camera rotation is this times the
#: forward one (``crab_hex_side_scene_cfg.DEPTH_CAMERA_ROT``).
QZ90_WXYZ = (SQRT_HALF, 0.0, 0.0, SQRT_HALF)


# ------------------------------------------------------------------ helpers
def _quat_mul(q1, q2):
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _leaf_ok(actual, expected, tol: float) -> bool:
    if expected is PRESENT:
        return actual != MISSING
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return abs(float(actual) - expected) <= tol
    return actual == expected


def _expected_for(kind: str, fwd_env: dict) -> dict[tuple, tuple]:
    """The expected (forward value, sideways value) per differing path for one pair."""
    exp = dict(EXPECTED_RL)
    if kind != "student":
        return exp
    del exp[REWARD_FUNC_PATH]
    exp[CAMERA_POS[0]] = (0.37, 0.0)
    exp[CAMERA_POS[1]] = (0.0, 0.625)
    fwd_rot = [float(v) for v in subtree(fwd_env, ("scene", "depth_camera", "offset", "rot"))]
    side_rot = _quat_mul(QZ90_WXYZ, fwd_rot)
    for i in range(4):
        exp[("scene", "depth_camera", "offset", "rot", i)] = (fwd_rot[i], side_rot[i])
    return exp


def _tol_for(path: tuple) -> float:
    return TOL_CAMERA_ROT if path[:4] == ("scene", "depth_camera", "offset", "rot") else TOL_DEFAULT


def _env(d: dict, task: str) -> dict:
    return normalize(d[task]["env"])


def _agent(d: dict, task: str) -> dict:
    return normalize(d[task]["agent"])


def _joint_default(joint_pos: dict, name: str) -> float:
    """``init_state.joint_pos`` keys are joint-name REGEXES (e.g. ``.*_Hip_Femur_RevoluteJoint``)."""
    hits = [v for pat, v in joint_pos.items() if re.fullmatch(pat, name)]
    assert len(hits) == 1, f"{name}: expected exactly one matching default, got {len(hits)}"
    return float(hits[0])


# ------------------------------------------------------------------ dump production
@pytest.fixture(scope="module")
def side_dumps(tmp_path_factory) -> dict[str, dict]:
    """One headless Kit boot per sideways phase (reused from ``KRABBY_CFG_DUMP_DIR`` when present)."""
    _require()
    reuse = os.environ.get("KRABBY_CFG_DUMP_DIR")
    d = Path(reuse) if reuse else tmp_path_factory.mktemp("cfgdump_side")
    out: dict[str, dict] = {}
    for name, (env_vars, tasks) in SIDE_FIXTURES.items():
        f = d / f"cfg_side_{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else dump(env_vars, tasks, f)
    return out


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_env_delta_is_exactly_the_travel_axis(dumps, side_dumps, pair):
    """The sideways env cfg differs from its forward twin on the travel-axis paths and no others."""
    pair_id, phase, fwd_task, side_task = pair
    kind = "student" if fwd_task in STUDENT_TASKS else "rl"
    fwd, side = _env(dumps[phase], fwd_task), _env(side_dumps[phase], side_task)

    diffs = diff_paths(fwd, side, limit=10_000)
    assert len(diffs) < 10_000, f"{pair_id}: diff limit hit -- the two cfgs are unrelated"
    expected = _expected_for(kind, fwd)
    got = {p for p, _, _ in diffs}
    assert got == set(expected), (
        f"{pair_id}: env delta is not the travel axis\n"
        f"  unexpected: {sorted(got - set(expected))}\n"
        f"  missing:    {sorted(set(expected) - got)}\n"
        f"{fmt_diffs(diffs)}"
    )

    for path, a, b in diffs:
        exp_a, exp_b = expected[path]
        tol = _tol_for(path)
        assert _leaf_ok(a, exp_a, tol), f"{pair_id}: {'/'.join(map(str, path))} forward value {a!r} != {exp_a!r}"
        assert _leaf_ok(b, exp_b, tol), f"{pair_id}: {'/'.join(map(str, path))} sideways value {b!r} != {exp_b!r}"

    if kind == "student":
        # the camera literals themselves (the sideways scene cfg only warns on drift at import)
        f_rot = [float(v) for v in subtree(fwd, ("scene", "depth_camera", "offset", "rot"))]
        s_rot = [float(v) for v in subtree(side, ("scene", "depth_camera", "offset", "rot"))]
        assert all(abs(a - b) <= TOL_CAMERA_ROT_PIN for a, b in zip(f_rot, FWD_CAMERA_ROT)), \
            f"{pair_id}: forward depth-camera rot drifted: {f_rot} != {FWD_CAMERA_ROT}"
        assert all(abs(a - b) <= TOL_CAMERA_ROT_PIN for a, b in zip(s_rot, SIDE_CAMERA_ROT)), \
            f"{pair_id}: sideways depth-camera rot is not q_z(+90) * forward: {s_rot} != {SIDE_CAMERA_ROT}"

    # negative pillars: the forward cfg must carry NEITHER new field (that is what keeps the
    # forward dumps, mdp_pins.yaml and the recorded lineage windows byte-identical)
    assert "travel_yaw" not in fwd, "the forward env cfg grew a travel_yaw field"
    assert "heading_offset" not in fwd["commands"]["base_velocity"], \
        "the forward command cfg grew a heading_offset field"

    # the scan grid is NOT transposed: same 108 rays in the same layout, so the scan mirror
    # permutation, the 1149-dim observation and every warm start stay valid
    f_pat = subtree(fwd, ("scene", "height_scanner", "pattern_cfg"))
    s_pat = subtree(side, ("scene", "height_scanner", "pattern_cfg"))
    for key in ("size", "resolution", "ordering", "direction"):
        assert f_pat[key] == s_pat[key], f"{pair_id}: scan pattern {key} changed: {f_pat[key]} -> {s_pat[key]}"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_agent_delta_is_exactly_names(dumps, side_dumps, pair):
    """The runner delta is the experiment name (+ the mirror the symmetry loss points at)."""
    pair_id, phase, fwd_task, side_task = pair
    is_student = fwd_task in STUDENT_TASKS
    fwd, side = _agent(dumps[phase], fwd_task), _agent(side_dumps[phase], side_task)

    diffs = diff_paths(fwd, side, limit=10_000)
    expected = {("experiment_name",)}
    if not is_student:  # distillation has no symmetry loss
        expected.add(("algorithm", "symmetry_cfg", "data_augmentation_func"))
    got = {p for p, _, _ in diffs}
    assert got == expected, (
        f"{pair_id}: runner delta is not the names\n"
        f"  unexpected: {sorted(got - expected)}\n  missing: {sorted(expected - got)}\n{fmt_diffs(diffs)}"
    )

    assert side["experiment_name"] == "crab_hex_side_" + fwd["experiment_name"][len("crab_hex_"):], \
        f"{pair_id}: {fwd['experiment_name']!r} -> {side['experiment_name']!r}"
    if not is_student:
        assert subtree(fwd, ("algorithm", "symmetry_cfg", "data_augmentation_func")) == FWD_MIRROR
        assert subtree(side, ("algorithm", "symmetry_cfg", "data_augmentation_func")) == SIDE_MIRROR


@pytest.mark.parametrize("phase", sorted(SIDE_FIXTURES))
def test_environ_delta(dumps, side_dumps, phase):
    """The expanded knob environment differs only by the preset variable and the dropped RSI keys."""
    fwd, side = dumps[phase]["_environ"], side_dumps[phase]["_environ"]
    assert set(fwd) - set(side) == {"KRABBY_PHASE", "KRABBY_RSI_FRAC", "KRABBY_RSI_BANK"}, \
        f"{phase}: forward-only keys {sorted(set(fwd) - set(side))}"
    side_only = {"KRABBY_SIDE_PHASE"}
    if phase == "3a":
        # the forward student cfg gates on the forward is_student_phase(), which reads
        # KRABBY_PHASE / KRABBY_STUDENT_MDP only -- the sideways 3a preset exports the flag
        side_only.add("KRABBY_STUDENT_MDP")
    assert set(side) - set(fwd) == side_only, f"{phase}: sideways-only keys {sorted(set(side) - set(fwd))}"
    for k in sorted(set(fwd) & set(side)):
        assert fwd[k] == side[k], f"{phase}: {k} = {fwd[k]!r} (forward) != {side[k]!r} (sideways)"
    assert side["KRABBY_SIDE_PHASE"] == phase
    assert "KRABBY_PHASE" not in side, "the sideways activation must never write KRABBY_PHASE"


@pytest.mark.parametrize("phase", ["2a", "2b", "2c"])
def test_side_phase2_teacher_mode_matches_side_walk(side_dumps, phase):
    """Twin of the forward guard: the phase-2 teacher task and the Side-Walk task build the same MDP."""
    t, f = side_dumps[phase][SIDE_TEACHER], side_dumps[phase][SIDE_WALK]
    env_diffs = diff_paths(normalize(t["env"]), normalize(f["env"]), limit=10_000)
    assert not env_diffs, f"{phase}: Side-Teacher-v0 env cfg != Side-Walk-v0 env cfg\n{fmt_diffs(env_diffs)}"
    agent_diffs = diff_paths(strip(t["agent"], AGENT_VOLATILE), strip(f["agent"], AGENT_VOLATILE), limit=10_000)
    assert not agent_diffs, f"{phase}: side teacher runner != side flat-walk runner\n{fmt_diffs(agent_diffs)}"
    assert t["agent"]["experiment_name"] == "crab_hex_side_teacher"
    assert side_dumps[phase]["_environ"]["KRABBY_HEX_TEACHER_MODE"] == phase


def test_side_student_mirrors_2c_mdp(side_dumps):
    """Twin of the forward guard: phase 3a carries the sideways 2c MDP into distillation."""
    s = runtime_resolve(strip(side_dumps["3a"][SIDE_STUDENT]["env"], STUDENT_ONLY))
    t = runtime_resolve(strip(side_dumps["2c"][SIDE_TEACHER]["env"], STUDENT_ONLY))
    diffs = []
    for p in STUDENT_SHARED:
        try:
            diffs += diff_paths(subtree(s, p), subtree(t, p), p, limit=10_000)
        except KeyError:
            diffs.append((p, "<missing in one cfg>", ""))
    assert not diffs, f"3a side student MDP != 2c side teacher MDP on the shared managers\n{fmt_diffs(diffs)}"
    assert side_dumps["3a"][SIDE_STUDENT]["agent"]["algorithm"]["class_name"] == "DistillationWithExtractor"
    # the vec-env wrapper clips RAW policy actions before the action term scales them; an unset
    # student clip on the teacher's full action space drove the joints at ~5x the trained authority
    assert side_dumps["3a"][SIDE_STUDENT]["agent"]["clip_actions"] == \
        side_dumps["2c"][SIDE_TEACHER]["agent"]["clip_actions"], \
        "side student runner clip_actions differs from the 2c side teacher runner"
    assert "KRABBY_PHASEOUT" not in side_dumps["3a"]["_environ"]


def test_side_play_variants_keep_the_train_mdp(dumps, side_dumps):
    """The PLAY twins are visualization only: same MDP, same terrain difficulty as training."""
    a = normalize(side_dumps["2c"][SIDE_TEACHER_PLAY]["env"])
    b = normalize(side_dumps["2c"][SIDE_WALK_PLAY]["env"])
    diffs = diff_paths(a, b, limit=10_000)
    assert not diffs, f"2c: Side-Teacher-Play != Side-Walk-Play\n{fmt_diffs(diffs)}"
    train = normalize(side_dumps["2c"][SIDE_TEACHER]["env"])
    assert subtree(a, ("scene", "terrain", "terrain_generator", "difficulty_range")) == \
        subtree(train, ("scene", "terrain", "terrain_generator", "difficulty_range"))
    # the PLAY head (episode_length_s applied AFTER the knobs) is the forward one, copied
    for side_play, fwd_play, phase in ((SIDE_TEACHER_PLAY, TEACHER_PLAY, "2c"),
                                       (SIDE_WALK_PLAY, FLAT_PLAY, "2c"),
                                       (SIDE_WALK_PLAY, FLAT_PLAY, "1a")):
        assert _env(side_dumps[phase], side_play)["episode_length_s"] == \
            _env(dumps[phase], fwd_play)["episode_length_s"]
    assert _env(side_dumps["3a"], SIDE_STUDENT_PLAY)["episode_length_s"] == \
        _env(dumps["3a"], STUDENT_PLAY)["episode_length_s"]


def test_fore_aft_defaults_are_reflection_consistent(side_dumps):
    """The spawn pose must be a fixed point of the fore-aft mirror the symmetry loss uses.

    Reflecting through the plane x = 0 swaps F<->R on each side letter; the yaw-type joints
    (``Body_CamShaft``, ``Body_Hip``) flip sign and the pitch-type ones (``Hip_Femur``,
    ``Femur_Tibia``, whose axes lie along x for legs on the +-y faces) keep it. All the yaw-type
    defaults are 0 today, so the constraint is satisfied trivially -- this test is the tripwire for
    a future non-zero default that would make the mirrored spawn a different pose.
    """
    joint_pos = subtree(_env(side_dumps["1a"], SIDE_WALK), ("scene", "robot", "init_state", "joint_pos"))
    for letter in ("L", "R"):
        for joint, sign in (("Hip_Femur", 1.0), ("Femur_Tibia", 1.0),
                            ("Body_Hip", -1.0), ("Body_CamShaft", -1.0)):
            front = _joint_default(joint_pos, f"F{letter}_{joint}_RevoluteJoint")
            rear = _joint_default(joint_pos, f"R{letter}_{joint}_RevoluteJoint")
            assert math.isclose(rear, sign * front, abs_tol=1e-12), \
                f"{joint} F{letter}={front} vs R{letter}={rear} (expected factor {sign})"
    for letter in ("L", "R"):  # the middle legs are their own mirror image
        for joint, sign in (("Body_Hip", -1.0), ("Body_CamShaft", -1.0)):
            v = _joint_default(joint_pos, f"M{letter}_{joint}_RevoluteJoint")
            assert math.isclose(v, sign * v, abs_tol=1e-12), f"M{letter}_{joint} default {v} is not mirror-fixed"


def test_scan_offset_lies_in_the_mirror_plane(dumps, side_dumps):
    """Each task's scan look-ahead sits on its own mirror plane, so flipping the grid is a symmetry.

    Forward mirrors L<->R through y = 0 and biases the scan along +x; sideways mirrors F<->R
    through x = 0 and biases it along +y. An off-plane offset would make the scan permutation
    (``build_scan_permutation``) wrong for the observation it augments.
    """
    fwd = subtree(_env(dumps["1a"], FLAT), ("scene", "height_scanner", "offset", "pos"))
    side = subtree(_env(side_dumps["1a"], SIDE_WALK), ("scene", "height_scanner", "offset", "pos"))
    assert fwd[1] == 0.0, f"forward scan offset is off the y = 0 mirror plane: {fwd}"
    assert side[0] == 0.0, f"sideways scan offset is off the x = 0 mirror plane: {side}"
    assert side[1] == fwd[0], "the sideways look-ahead must be the forward one, turned onto +y"
    assert side[2] == fwd[2], "the scanner ray origin height changed"


def test_side_flat_walk_cfg_builds(dumps, side_dumps):
    """The sideways flat-walk task carries the same 35 reward terms and the same terminations."""
    fwd, side = _env(dumps["1a"], FLAT), _env(side_dumps["1a"], SIDE_WALK)
    assert set(side["rewards"]) == set(fwd["rewards"]), \
        f"reward term names changed: {sorted(set(side['rewards']) ^ set(fwd['rewards']))}"
    assert len(side["rewards"]) == 35, f"expected 35 flat-walk reward terms, got {len(side['rewards'])}"
    assert set(side["terminations"]) == set(fwd["terminations"]), \
        f"termination names changed: {sorted(set(side['terminations']) ^ set(fwd['terminations']))}"
    # the swapped tracking term keeps its weight and params (KRABBY_TRACK_SIGMA2 assigns
    # ``std`` by attribute name, KRABBY_PHASEOUT and the driver's exposure parser key on the name)
    f_t, s_t = fwd["rewards"]["track_lin_vel_xy_exp"], side["rewards"]["track_lin_vel_xy_exp"]
    assert f_t["weight"] == s_t["weight"] and f_t["params"] == s_t["params"]
