"""The omni-vs-forward config delta: the acceptance criterion for "nothing else changed".

``crab_hex_omni_task`` subclasses the forward cfg classes and re-declares every cfg-valued
class attribute with the forward class-body literal (knob timing: ``import_packages`` imports
the forward package first, so forward cfg objects are built BEFORE the omni activation exports
the ``KRABBY_*`` knobs). That makes the omni package look redundant and makes a silent drift --
an attribute that was not re-declared, a literal that was mistyped, a forward refactor the
sibling did not follow -- invisible in review. This module is the machine check: build every
omni task per phase inside Isaac Sim, diff its ``class_to_dict`` dump against the forward twin
under the same window, and assert the set of differing config paths is EXACTLY the plan's
Appendix A.

Five paths are whole SUBTREES rather than leaf lists -- the command term, the parkour term, the
ring generator, the spawn event and the RSI event are each replaced wholesale. They are
stripped from the generic diff and asserted against explicit blocks below, so "the omni command
term grew a field nobody reviewed" still fails.

Pairing: the three phase-1 stages (1a, 1b, 1c) all derive from the forward WINDOW 0 env, so all
three pair against the forward ``1a`` fixture; 2a/2b/2c pair against their own forward window.

The forward dumps come from the imported ``dumps`` fixture of
``tests/integration/test_crab_hex_phase_configs.py``, and the sideways ones from
``test_crab_hex_sideways_configs.py`` (both files stay unedited and are themselves guards); the
omni dumps are produced here by the same ``dump()`` helper.

Opt in (≈ 6 extra Kit boots)::

    RUN_CRAB_HEX_CFG_IDENTITY=1 KRABBY_CFG_DUMP_DIR=<dir> pytest \\
        tests/integration/test_crab_hex_phase_configs.py \\
        tests/integration/test_crab_hex_sideways_configs.py \\
        tests/integration/test_crab_hex_omni_configs.py -v

Set ``KRABBY_CFG_DUMP_DIR`` so the forward (``cfg_<fixture>.json``) and sideways
(``cfg_side_<phase>.json``) dumps are reused instead of re-booting Kit for them; the omni dumps
land beside them as ``cfg_omni_<phase>.json``.
"""

from __future__ import annotations

import json
import math
import os
import types
from pathlib import Path

import pytest

from tests.integration.test_crab_hex_phase_configs import (  # noqa: F401  -- ``dumps`` is a fixture
    AGENT_VOLATILE,
    FLAT,
    FLAT_PLAY,
    PLANT,
    TEACHER,
    TEACHER_PLAY,
    _require,
    diff_paths,
    dump,
    dumps,
    fmt_diffs,
    normalize,
    strip,
    subtree,
)
from tests.integration.test_crab_hex_sideways_configs import (  # noqa: F401  -- fixture + ids
    SIDE_TEACHER,
    SIDE_WALK,
    side_dumps,
)

pytestmark = pytest.mark.isaacsim

OMNI_WALK = "Isaac-Crab-Hex-Omni-Walk-v0"
OMNI_WALK_PLAY = "Isaac-Crab-Hex-Omni-Walk-Play-v0"
OMNI_TEACHER = "Isaac-Crab-Hex-Omni-Teacher-v0"
OMNI_TEACHER_PLAY = "Isaac-Crab-Hex-Omni-Teacher-Play-v0"

# phase -> (env for the dump process, tasks to build). No student fixture and no
# ``legacy_golden_*``: this plan is teacher-only and has no reproduction schedule.
OMNI_FIXTURES: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {
    "1a": ({"KRABBY_OMNI_PHASE": "1a", "KRABBY_PLANT": PLANT}, (OMNI_WALK, OMNI_WALK_PLAY)),
    "1b": ({"KRABBY_OMNI_PHASE": "1b", "KRABBY_PLANT": PLANT}, (OMNI_WALK,)),
    "1c": ({"KRABBY_OMNI_PHASE": "1c", "KRABBY_PLANT": PLANT}, (OMNI_WALK,)),
    "2a": ({"KRABBY_OMNI_PHASE": "2a", "KRABBY_PLANT": PLANT}, (OMNI_TEACHER, OMNI_WALK)),
    "2b": ({"KRABBY_OMNI_PHASE": "2b", "KRABBY_PLANT": PLANT}, (OMNI_TEACHER, OMNI_WALK)),
    "2c": ({"KRABBY_OMNI_PHASE": "2c", "KRABBY_PLANT": PLANT},
           (OMNI_TEACHER, OMNI_WALK, OMNI_TEACHER_PLAY, OMNI_WALK_PLAY)),
}

# (pair id, omni phase, forward fixture, forward task, omni task). The three phase-1 stages all
# derive from forward window 0, so all three pair against the forward ``1a`` fixture. The
# teacher head of a phase-2 window trains in the Walk task in the lineage of record, so both
# task ids are paired for every phase-2 fixture.
RL_PAIRS = [
    ("1a-walk", "1a", "1a", FLAT, OMNI_WALK),
    ("1b-walk", "1b", "1a", FLAT, OMNI_WALK),
    ("1c-walk", "1c", "1a", FLAT, OMNI_WALK),
    ("2a-teacher", "2a", "2a", TEACHER, OMNI_TEACHER),
    ("2a-walk", "2a", "2a", FLAT, OMNI_WALK),
    ("2b-teacher", "2b", "2b", TEACHER, OMNI_TEACHER),
    ("2b-walk", "2b", "2b", FLAT, OMNI_WALK),
    ("2c-teacher", "2c", "2c", TEACHER, OMNI_TEACHER),
    ("2c-walk", "2c", "2c", FLAT, OMNI_WALK),
]
# The PLAY twins: same delta expected (the PLAY heads are applied on top of the train cfgs on
# both sides); no extra Kit boots -- every task below is already in its phase's fixture.
PLAY_PAIRS = [
    ("1a-walk-play", "1a", "1a", FLAT_PLAY, OMNI_WALK_PLAY),
    ("2c-teacher-play", "2c", "2c", TEACHER_PLAY, OMNI_TEACHER_PLAY),
    ("2c-walk-play", "2c", "2c", FLAT_PLAY, OMNI_WALK_PLAY),
]
ALL_PAIRS = RL_PAIRS + PLAY_PAIRS

MISSING = "<missing>"
TOL = 1e-9

FWD_MIRROR = "parkour_tasks.crab_hex_forward_task.mdp.crab_hex_mirror:crab_hex_symmetry_augmentation"
OMNI_MIRROR = ("parkour_tasks.crab_hex_omni_task.mdp.crab_hex_omni_mirror:"
               "crab_hex_omni_symmetry_augmentation")

_FWD_MDP = "parkour_tasks.crab_hex_forward_task.mdp"
_OMNI_MDP = "parkour_tasks.crab_hex_omni_task.mdp"
_LIB_REWARDS = "parkour_isaaclab.envs.mdp.rewards"
_OMNI_REWARDS = f"{_OMNI_MDP}.crab_hex_omni_rewards"


class _Present:
    """Leaf sentinel: "this key exists on the forward side and its value is not checked"."""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<present>"


PRESENT = _Present()

# ---------------------------------------------------------------------------------------------
# Whole subtrees that are REPLACED, not edited. Stripped from the generic diff and asserted
# against the explicit blocks further down.
SUBTREES = (
    ("commands", "base_velocity"),
    ("parkours", "base_parkour"),
    ("events", "reset_root_state"),
    ("events", "rsi_reference_reset"),
    ("scene", "terrain", "terrain_generator"),
)

# ---------------------------------------------------------------------------------------------
# The expected LEAF delta (plan Appendix A, as amended). Every entry is a deliberate consequence
# of commanding a world travel direction and a world facing; nothing else about the MDP may
# differ.
EXPECTED_RL: dict[tuple, tuple] = {
    # the two omni fields the forward cfg classes do NOT have, so every shared seam keeps its
    # identity path (``getattr(env.cfg, "omni_command", False)`` / ``..., 0.0``)
    ("omni_command",): (MISSING, True),
    ("omni_clock_turn_radius",): (MISSING, 1.2),
    # the observation: same width, same params, five slots re-filled
    ("observations", "policy", "extreme_parkour_observations", "func"): (
        f"{_FWD_MDP}.observations:CrabHexParkourObservations",
        f"{_OMNI_MDP}.crab_hex_omni_observations:CrabHexOmniParkourObservations",
    ),
    # Revision 3 (2026-09-17, user decision): the facing-error observation (steering channels
    # 6/7) is clipped to the 1c band's edge in every omni preset -- the forward term has no such
    # param, so this is an addition, not a value change (KRABBY_OMNI_FACING_OBS_CLIP=0.6).
    ("observations", "policy", "extreme_parkour_observations", "params", "facing_obs_clip"): (
        MISSING, 0.6,
    ),
    # 2026-09-18 (plan Part A2): the facing-error ENCODING, declared on the omni obs term at
    # its class default so KRABBY_OMNI_FACING_OBS_ENCODING has a key to overwrite (the applier
    # refuses a term without it). The forward term has no such param, so this is an addition,
    # and "raw" is bitwise the observation every omni run to date trained on -- no preset sets
    # the knob, so every dump must read "raw".
    ("observations", "policy", "extreme_parkour_observations", "params",
     "facing_obs_encoding"): (MISSING, "raw"),
    # four reward functions swapped UNDER THE FORWARD NAMES (weights, knobs and telemetry
    # labels unchanged), plus the params each twin needs
    ("rewards", "penalty_lin_vel_y", "func"): (
        f"{_LIB_REWARDS}:penalty_lin_vel_y_l2",
        f"{_OMNI_REWARDS}:penalty_cross_command_vel_l2",
    ),
    ("rewards", "reward_tracking_goal_vel", "func"): (
        f"{_LIB_REWARDS}:reward_tracking_goal_vel_on_parkour",
        f"{_OMNI_REWARDS}:reward_tracking_command_vel_on_obstacle",
    ),
    ("rewards", "reward_tracking_goal_vel", "params", "command_name"): (MISSING, "base_velocity"),
    ("rewards", "reward_tracking_yaw", "func"): (
        f"{_LIB_REWARDS}:reward_tracking_yaw_on_parkour",
        f"{_OMNI_REWARDS}:reward_facing_exp",
    ),
    ("rewards", "reward_tracking_yaw", "params", "command_name"): (MISSING, "base_velocity"),
    ("rewards", "reward_obstacle_clearance", "func"): (
        f"{_LIB_REWARDS}:reward_obstacle_clearance",
        f"{_OMNI_REWARDS}:reward_obstacle_clearance_omni",
    ),
    # THE ONE ADDED TERM (the 36th name, 2026-09-17): the forward side has no such key, so
    # ``diff_paths`` reports the whole block as one path. Its contents are asserted by
    # ``test_yaw_progress_term_block`` below.
    ("rewards", "reward_yaw_progress"): (MISSING, PRESENT),
    # THE SECOND ADDED TERM (the 37th name, 2026-09-18, plan Part A2): the facing term paid
    # only on the turn slots, shipped at weight 0.0 -- a NAME that logs 0 on every line until
    # KRABBY_OMNI_FACING_TURN_W arms it. Same whole-block path as above (no forward key); its
    # contents, weight 0.0 included, are asserted by ``test_facing_turn_term_block`` below.
    ("rewards", "reward_facing_turn"): (MISSING, PRESENT),
    # NOTE(Revision 4, 2026-09-18): ``track_ang_vel_z_exp.params.std`` is NOT in this list any
    # more. The omni rewards CLASS BODY still declares 0.15 (plan D4: gradient at the plant's
    # 0.15-0.3 rad/s turn rate), but every omni preset now sets KRABBY_OMNI_ANG_VEL_STD=0.5 --
    # the forward value, sqrt(0.25) -- so the two dumps AGREE on this path and it must not
    # appear in the delta. Why the reversal: at std 0.15 the 88 % of steps that command no turn
    # tax ordinary gait yaw wobble 0.36/s (at 0.10 rad/s) to 0.63/s (at 0.15), more than the
    # whole translation income, which is one of the four terms that made standing still the
    # optimum in 1c. The turning signal is carried by reward_yaw_progress instead (weight 1.0),
    # which did not exist when the plan chose 0.15. ``test_ang_vel_std_is_the_forward_kernel``
    # below pins the AGREEMENT, so a knob that stops being applied fails loudly rather than
    # silently reinstating the 0.15 kernel.
    #
    # Revision 4's other reward-shape change: the gait clock's schedule term switches from the
    # additive halves to swing_quality x stance_quality, so a policy that never lifts a foot
    # cannot collect the stance half for free. The MECHANISM is the forward task's own
    # (clock_schedule_income(..., combine=), 2026-08-22); the omni preset reaches it through
    # KRABBY_OMNI_CLOCK_COMBINE, which is applied after the forward block and therefore wins.
    ("rewards", "reward_clock_schedule", "params", "combine"): ("sum", "product"),
    # NOTE(Revision 4, 2026-09-18): the two re-pointed economy knobs (KRABBY_TRACK_SIGMA2,
    # KRABBY_TRACK_L1_W) now differ in VALUE between the forward window-0/2a/2b/2c fixtures
    # (0.1 / -1.0) and every omni preset (0.04 / -2.0, via OMNI_COMMON) -- verified against the
    # real cfg dump 2026-09-18 (RUN_CRAB_HEX_CFG_IDENTITY), which failed this test's env-delta
    # assertion until these two leaves were added. std values are sqrt(sigma2).
    ("rewards", "track_lin_vel_xy_exp", "params", "std"): (math.sqrt(0.1), math.sqrt(0.04)),
    ("rewards", "penalty_tracking_error_l1", "weight"): (-1.0, -2.0),
    # no goal cutoff: the ring's sector anchors are never "reached"
    ("terminations", "total_terminates", "func"): (
        "parkour_isaaclab.envs.mdp.terminations:terminate_episode",
        f"{_OMNI_REWARDS}:terminate_episode_no_goals",
    ),
    # the 0.6 m scan look-ahead is centred: travel is uniform over +-pi, and the origin is the
    # only point on BOTH mirror planes
    ("scene", "height_scanner", "offset", "pos", 0): (0.6, 0.0),
}

#: The facing reward is armed from 1c (``KRABBY_YAW_W=0.2``). The forward window 0 has no
#: ``KRABBY_YAW_W``, so the weight differs for the 1c pair only -- 1a/1b keep the forward 0.0,
#: and every phase-2 window has 0.2 on both sides (``ELEMENTS_5K``).
YAW_WEIGHT_PATH = ("rewards", "reward_tracking_yaw", "weight")

#: ``KRABBY_YAW_PROGRESS_W`` -- the added term's dose, the same in every omni window (the
#: preset arms it everywhere; 1a/1b command no turn, so it is behaviourally inert there).
EXPECTED_YAW_PROGRESS_W = 1.0

#: Revision 4 (2026-09-18): the kernel width both tasks now use -- forward's own sqrt(0.25).
EXPECTED_ANG_VEL_STD = 0.5
ANG_VEL_STD_PATH = ("rewards", "track_ang_vel_z_exp", "params", "std")
#: Revision 4: the gait-clock schedule term's combine mode per side.
CLOCK_COMBINE_PATH = ("rewards", "reward_clock_schedule", "params", "combine")

# ---------------------------------------------------------------------------------------------
# The replaced-subtree blocks.
#: ``commands.base_velocity``: the class, the seven new sampler fields and the tighter dead
#: band. ``ranges.heading`` additionally differs where the forward window runs the turning
#: school. ``cmd_pool`` (2026-09-18, plan Part A1) is the canonical pool STRING, dumped
#: verbatim and ``null`` in every preset -- a default-off addition, like the encoding param.
CMD_FIELDS_ADDED = ("slot_probs", "speed_ellipse", "facing_delta", "travel_dir_range",
                    "ang_vel_max", "dir_band", "cmd_pool")
FWD_CMD_CLASS = ("parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command:"
                 "UniformParkourCommand")
OMNI_CMD_CLASS = f"{_OMNI_MDP}.crab_hex_omni_command:CrabHexOmniCommand"

#: phase -> the per-window sampler values the presets ask for.
EXPECTED_CMD: dict[str, dict] = {
    "1a": {"slot_probs": [0.2, 0.8, 0.0, 0.0], "dir_band": [0.0, 0.0], "facing_delta": [0.2, 0.6]},
    "1b": {"slot_probs": [0.2, 0.8, 0.0, 0.0], "dir_band": [1.5708, 1.5708],
           "facing_delta": [0.2, 0.6]},
    "1c": {"slot_probs": [0.15, 0.55, 0.10, 0.20], "dir_band": None, "facing_delta": [0.2, 0.6]},
    "2a": {"slot_probs": [0.15, 0.45, 0.15, 0.25], "dir_band": None, "facing_delta": [0.2, 1.2]},
    "2b": {"slot_probs": [0.15, 0.45, 0.15, 0.25], "dir_band": None, "facing_delta": [0.2, 2.4]},
    "2c": {"slot_probs": [0.15, 0.45, 0.15, 0.25], "dir_band": None, "facing_delta": [0.2, 2.4]},
}
EXPECTED_SPEED_ELLIPSE = [0.35, 0.25, 0.25]
EXPECTED_ANG_VEL_MAX = 0.3
EXPECTED_ANG_VEL_CLIP = 0.2
FWD_ANG_VEL_CLIP = 0.4
#: The forward windows that run the turning school (``KRABBY_HEADING=-1.2:1.2``); the omni
#: sampler ignores ``ranges.heading`` entirely and the knob is dropped, so it stays (0, 0).
FWD_HEADING_WINDOWS = {"2b", "2c"}

#: ``parkours.base_parkour``: the class plus the ring fields, and the promotion fractions
#: (whose UNITS changed -- a dimensionless tracking ratio, not a fraction of commanded distance).
OMNI_PARKOUR_CLASS = f"{_OMNI_MDP}.omni_parkour_event:OmniParkourEvent"
FWD_PARKOUR_CLASS = "parkour_isaaclab.envs.mdp.parkours.parkour_event:ParkourEvent"
EXPECTED_PARKOUR_FIELDS = {
    "clear_radius_m": 2.0,
    "ring_outer_radius_m": 4.5,
    "n_sectors": 8,
    "min_cmd_dist_m": 1.0,
    "block_height_tol_m": 0.03,
    "foot_body_names": [".*_Footpad"],
    "command_name": "base_velocity",
}
#: phase -> the forward window's promotion fractions (window 0 never sets the knob; the
#: 40-s windows scale 0.45/0.25 by 20/40). The omni windows are 0.45 / 0.25 flat.
EXPECTED_PROMOTE_FWD = {"1a": (0.8, 0.4), "2a": (0.225, 0.125), "2b": (0.225, 0.125),
                        "2c": (0.225, 0.125)}
EXPECTED_PROMOTE_OMNI = (0.45, 0.25)

#: ``events.reset_root_state``: the ring spawn.
OMNI_RESET_FUNC = f"{_OMNI_MDP}.omni_events:reset_root_state_ring"
FWD_RESET_FUNC = "parkour_isaaclab.envs.mdp.events:reset_root_state"
#: ``events.rsi_reference_reset``: the re-yawed, provenance-tagged bank.
OMNI_RSI_FUNC = f"{_OMNI_MDP}.crab_hex_omni_rsi:reset_from_reference_states_omni"
FWD_RSI_FUNC = f"{_FWD_MDP}.crab_hex_rsi:reset_from_reference_states"
EXPECTED_RSI_SOURCE = {"1a": 0, "1b": 1, "1c": "all", "2a": "all", "2b": "all", "2c": "all"}
#: ``clips.lin_vel_clip`` 0.2 + ``WALK_MARGIN_M_S`` 0.01 -- derived in the cfg, pinned here.
EXPECTED_WALK_FLOOR = 0.21
#: ``KRABBY_OMNI_YAW_RANGE`` (the preset's literal, NOT ``math.pi``) reaches BOTH reset events:
#: the ring spawn and the RSI re-yaw. They must agree -- see ``omni_knobs._apply_spawn_yaw``.
EXPECTED_YAW_RANGE = [-3.14159, 3.14159]

#: ``scene.terrain.terrain_generator``: the ring generator (plan D8).
RING_SUB_KEYS = ("parkour_flat", "omni_ring_sparse", "omni_ring_dense")
EXPECTED_GENERATOR_SCALARS = {
    "size": [10.0, 10.0], "border_width": 20.0, "num_rows": 10, "num_cols": 20,
    "horizontal_scale": 0.08, "vertical_scale": 0.005, "slope_threshold": 1.5, "num_goals": 8,
}
EXPECTED_SUB_FUNCS = {
    "parkour_flat": "parkour_isaaclab.terrains.omni.omni_terrains:omni_flat_terrain",
    "omni_ring_sparse": "parkour_isaaclab.terrains.omni.omni_terrains:omni_ring_terrain",
    "omni_ring_dense": "parkour_isaaclab.terrains.omni.omni_terrains:omni_ring_terrain",
}
EXPECTED_RING_BLOCKS = {"omni_ring_sparse": 8, "omni_ring_dense": 16}
EXPECTED_RING_HEIGHTS = {
    "hurdle_height_range": "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
    "step_height": "0.02 + 0.06 * difficulty",
}
#: phase -> the terrain mix the window's knobs produce (light 80/10/10 at 1a-1c, the
#: curriculum 50/25/25 band from 2a). The share is ``(1 - flat_frac) / 2`` and is compared
#: with a tolerance because it is a float division.
EXPECTED_TERRAIN: dict[str, dict] = {
    "1a": {"curriculum": False, "difficulty_range": [0.05, 0.2], "flat": 0.8, "ring": 0.1},
    "1b": {"curriculum": False, "difficulty_range": [0.05, 0.2], "flat": 0.8, "ring": 0.1},
    "1c": {"curriculum": False, "difficulty_range": [0.05, 0.2], "flat": 0.8, "ring": 0.1},
    "2a": {"curriculum": True, "difficulty_range": [0.2, 0.7], "flat": 0.5, "ring": 0.25},
    "2b": {"curriculum": True, "difficulty_range": [0.2, 0.7], "flat": 0.5, "ring": 0.25},
    "2c": {"curriculum": True, "difficulty_range": [0.2, 0.7], "flat": 0.5, "ring": 0.25},
}

# ---------------------------------------------------------------------------------------------
# ``_environ``: the expanded knob environment (plan D11).
OMNI_COMMON_KEYS = {"KRABBY_OMNI_SPEED_ELLIPSE", "KRABBY_OMNI_ANG_VEL_MAX",
                    "KRABBY_OMNI_CLOCK_TURN_R", "KRABBY_OMNI_YAW_RANGE",
                    # the added term's weight: armed in every window, gated to zero in 1a/1b
                    "KRABBY_YAW_PROGRESS_W",
                    # Revision 3 (2026-09-17): the facing-error observation clip, fixed at 0.6
                    # in every window regardless of the facing band's own per-phase ramp.
                    "KRABBY_OMNI_FACING_OBS_CLIP",
                    # Revision 4 (2026-09-18): the reward economy, uniform along the chain.
                    "KRABBY_OMNI_ANG_VEL_STD", "KRABBY_OMNI_CLOCK_COMBINE"}
#: Revision 4: the two tracking knobs EVERY forward window also sets (FORMATION: 0.1 / -1.0), so
#: they are differing VALUES in the ``_environ`` delta rather than omni-only additions.
_ECONOMY_REPOINTED = {"KRABBY_TRACK_SIGMA2", "KRABBY_TRACK_L1_W"}
_STAGE1_ONLY = OMNI_COMMON_KEYS | {"KRABBY_FLAT_TERRAIN_GEOM", "KRABBY_TERRAIN_PROMOTE",
                                   "KRABBY_OMNI_SLOTS", "KRABBY_OMNI_RSI_SOURCE"}
_TEACHER_ONLY = OMNI_COMMON_KEYS | {"KRABBY_OMNI_SLOTS", "KRABBY_OMNI_RSI_SOURCE",
                                    "KRABBY_OMNI_FACING"}
_REPOINTED = {"KRABBY_RSI_BANK", "KRABBY_FLAT_TERRAIN_GEOM",
              "KRABBY_TERRAIN_PROMOTE"} | _ECONOMY_REPOINTED
_STAGE1_DIFFER = {"KRABBY_RSI_BANK"} | _ECONOMY_REPOINTED
#: omni phase -> (forward fixture, forward-only keys, omni-only keys, keys differing in value)
EXPECTED_ENVIRON: dict[str, tuple[str, set, set, set]] = {
    "1a": ("1a", set(), _STAGE1_ONLY | {"KRABBY_OMNI_DIR_BAND"}, _STAGE1_DIFFER),
    "1b": ("1a", set(), _STAGE1_ONLY | {"KRABBY_OMNI_DIR_BAND"}, _STAGE1_DIFFER),
    "1c": ("1a", set(), _STAGE1_ONLY | {"KRABBY_OMNI_FACING", "KRABBY_YAW_W"},
           _STAGE1_DIFFER),
    "2a": ("2a", set(), _TEACHER_ONLY, _REPOINTED),
    "2b": ("2b", {"KRABBY_HEADING"}, _TEACHER_ONLY, _REPOINTED),
    "2c": ("2c", {"KRABBY_HEADING"}, _TEACHER_ONLY, _REPOINTED),
}


# ------------------------------------------------------------------ helpers
def _leaf_ok(actual, expected, tol: float = TOL) -> bool:
    if expected is PRESENT:
        return actual != MISSING
    if isinstance(expected, float) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        return abs(float(actual) - expected) <= tol
    return actual == expected


def _env(d: dict, task: str) -> dict:
    return normalize(d[task]["env"])


def _agent(d: dict, task: str) -> dict:
    return normalize(d[task]["agent"])


def _expected_leaves(omni_phase: str) -> dict[tuple, tuple]:
    exp = dict(EXPECTED_RL)
    if omni_phase == "1c":
        exp[YAW_WEIGHT_PATH] = (0.0, 0.2)
    return exp


def _close(a, b, tol: float = 1e-9) -> bool:
    return isinstance(a, (int, float)) and abs(float(a) - float(b)) <= tol


def _close_list(got, want, tol: float = 1e-9) -> bool:
    return (isinstance(got, list) and len(got) == len(want)
            and all(_close(a, b, tol) for a, b in zip(got, want)))


def _indexed(path: tuple, n: int) -> set:
    """``diff_paths`` recurses into two lists of EQUAL length, so a differing tuple-valued field
    reports one path per index. (A list against ``None`` -- e.g. ``dir_band`` in 1c -- is not two
    lists, so it stays a single whole-field path.)"""
    return {path + (i,) for i in range(n)}


# ------------------------------------------------------------------ dump production
@pytest.fixture(scope="module")
def omni_dumps(tmp_path_factory) -> dict[str, dict]:
    """One headless Kit boot per omni phase (reused from ``KRABBY_CFG_DUMP_DIR`` when present)."""
    _require()
    reuse = os.environ.get("KRABBY_CFG_DUMP_DIR")
    d = Path(reuse) if reuse else tmp_path_factory.mktemp("cfgdump_omni")
    out: dict[str, dict] = {}
    for name, (env_vars, tasks) in OMNI_FIXTURES.items():
        f = d / f"cfg_omni_{name}.json"
        out[name] = json.loads(f.read_text()) if f.exists() else dump(env_vars, tasks, f)
    return out


# ------------------------------------------------------------------ the generic leaf delta
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_env_delta_is_exactly_the_omni_list(dumps, omni_dumps, pair):
    """Outside the five replaced subtrees, the omni env differs from its forward twin on the
    Appendix-A paths and no others."""
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    fwd = strip(_env(dumps[fwd_phase], fwd_task), SUBTREES)
    omni = strip(_env(omni_dumps[omni_phase], omni_task), SUBTREES)

    diffs = diff_paths(fwd, omni, limit=10_000)
    assert len(diffs) < 10_000, f"{pair_id}: diff limit hit -- the two cfgs are unrelated"
    expected = _expected_leaves(omni_phase)
    got = {p for p, _, _ in diffs}
    assert got == set(expected), (
        f"{pair_id}: env delta is not the omni list\n"
        f"  unexpected: {sorted(got - set(expected))}\n"
        f"  missing:    {sorted(set(expected) - got)}\n"
        f"{fmt_diffs(diffs)}"
    )
    for path, a, b in diffs:
        exp_a, exp_b = expected[path]
        assert _leaf_ok(a, exp_a), f"{pair_id}: {'/'.join(map(str, path))} forward value {a!r} != {exp_a!r}"
        assert _leaf_ok(b, exp_b), f"{pair_id}: {'/'.join(map(str, path))} omni value {b!r} != {exp_b!r}"


# ------------------------------------------------------------------ the replaced subtrees
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_command_term_block(dumps, omni_dumps, pair):
    """``commands.base_velocity``: the omni class, six new fields, the tighter dead band, and
    nothing else -- every inherited literal (stiffness, resampling, clips.lin_vel_clip,
    ranges.lin_vel_x, stand_frac, the visualizers) is still the forward one."""
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f = subtree(_env(dumps[fwd_phase], fwd_task), ("commands", "base_velocity"))
    o = subtree(_env(omni_dumps[omni_phase], omni_task), ("commands", "base_velocity"))

    expected = {("class_type",), ("clips", "ang_vel_clip")}
    expected |= {(k,) for k in CMD_FIELDS_ADDED}
    if fwd_phase in FWD_HEADING_WINDOWS:
        expected |= _indexed(("ranges", "heading"), 2)
    diffs = diff_paths(f, o, limit=10_000)
    got = {p for p, _, _ in diffs}
    assert got == expected, (
        f"{pair_id}: command-term delta\n  unexpected: {sorted(got - expected)}\n"
        f"  missing: {sorted(expected - got)}\n{fmt_diffs(diffs)}"
    )
    assert f["class_type"] == FWD_CMD_CLASS and o["class_type"] == OMNI_CMD_CLASS
    assert f["clips"]["ang_vel_clip"] == FWD_ANG_VEL_CLIP
    assert o["clips"]["ang_vel_clip"] == EXPECTED_ANG_VEL_CLIP
    assert o["clips"]["lin_vel_clip"] == f["clips"]["lin_vel_clip"] == 0.2
    assert o["cmd_pool"] is None, (
        f"{pair_id}: no preset ships a command pool -- the default-off sampler path is what "
        "every recorded omni window trained on"
    )
    assert "cmd_pool" not in f, f"{pair_id}: the forward command cfg grew cmd_pool"
    assert _close_list(o["speed_ellipse"], EXPECTED_SPEED_ELLIPSE)
    assert _close(o["ang_vel_max"], EXPECTED_ANG_VEL_MAX)
    assert _close_list(o["travel_dir_range"], [-math.pi, math.pi])
    want = EXPECTED_CMD[omni_phase]
    assert o["dir_band"] == want["dir_band"], f"{pair_id}: dir_band"
    for key in ("slot_probs", "facing_delta"):
        assert len(o[key]) == len(want[key])
        assert all(_close(a, b, 1e-9) for a, b in zip(o[key], want[key])), f"{pair_id}: {key}"
    # the inherited-but-ignored fields are kept so KRABBY_LIN_VEL_X / KRABBY_STAND_FRAC still
    # write the same dump paths on both sides
    assert o["ranges"]["lin_vel_x"] == f["ranges"]["lin_vel_x"]
    assert o["stand_frac"] == f["stand_frac"]
    assert o["small_commands_to_zero"] == f["small_commands_to_zero"]
    assert o["heading_control_stiffness"] == f["heading_control_stiffness"] == 1.5
    assert o["resampling_time_range"] == f["resampling_time_range"]
    if fwd_phase in FWD_HEADING_WINDOWS:
        assert f["ranges"]["heading"] == [-1.2, 1.2]
    assert o["ranges"]["heading"] == [0.0, 0.0], "the omni sampler must never see a heading band"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_parkour_term_block(dumps, omni_dumps, pair):
    """``parkours.base_parkour``: the omni class, the ring fields and the promotion fractions."""
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f = subtree(_env(dumps[fwd_phase], fwd_task), ("parkours", "base_parkour"))
    o = subtree(_env(omni_dumps[omni_phase], omni_task), ("parkours", "base_parkour"))

    expected = {("class_type",), ("move_up_frac",), ("move_down_frac",)}
    expected |= {(k,) for k in EXPECTED_PARKOUR_FIELDS}
    diffs = diff_paths(f, o, limit=10_000)
    got = {p for p, _, _ in diffs}
    assert got == expected, (
        f"{pair_id}: parkour-term delta\n  unexpected: {sorted(got - expected)}\n"
        f"  missing: {sorted(expected - got)}\n{fmt_diffs(diffs)}"
    )
    assert f["class_type"] == FWD_PARKOUR_CLASS and o["class_type"] == OMNI_PARKOUR_CLASS
    for key, want in EXPECTED_PARKOUR_FIELDS.items():
        if isinstance(want, float):
            assert _close(o[key], want, 1e-12), f"{pair_id}: {key} = {o[key]!r}"
        else:
            assert o[key] == want, f"{pair_id}: {key} = {o[key]!r}"
    up_f, down_f = EXPECTED_PROMOTE_FWD[fwd_phase]
    assert _close(f["move_up_frac"], up_f) and _close(f["move_down_frac"], down_f)
    assert _close(o["move_up_frac"], EXPECTED_PROMOTE_OMNI[0])
    assert _close(o["move_down_frac"], EXPECTED_PROMOTE_OMNI[1])
    # the goal / debug machinery is inherited verbatim
    assert o["next_goal_threshold"] == f["next_goal_threshold"]
    assert o["num_future_goal_obs"] == f["num_future_goal_obs"]
    assert o["arrow_num"] == f["arrow_num"]
    assert o["freeze_terrain_levels"] == f["freeze_terrain_levels"]


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_reset_events_block(dumps, omni_dumps, pair):
    """The two reset events: the ring spawn and the re-yawed, provenance-tagged RSI."""
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f_env, o_env = _env(dumps[fwd_phase], fwd_task), _env(omni_dumps[omni_phase], omni_task)

    f = subtree(f_env, ("events", "reset_root_state"))
    o = subtree(o_env, ("events", "reset_root_state"))
    diffs = {p for p, _, _ in diff_paths(f, o, limit=10_000)}
    assert diffs == {("func",), ("params", "offset"), ("params", "asset_cfg"),
                     ("params", "yaw_range")}, f"{pair_id}: reset_root_state delta {sorted(diffs)}"
    assert f["func"] == FWD_RESET_FUNC and o["func"] == OMNI_RESET_FUNC
    assert o["mode"] == f["mode"] == "reset"
    # the param NAME is load-bearing: ParkourEvent.__init__ hard-reads params['offset']
    assert o["params"]["offset"] == 0.0, "the ring spawns at the tile centre"
    assert _close_list(o["params"]["yaw_range"], EXPECTED_YAW_RANGE)
    assert o["params"]["asset_cfg"]["name"] == "robot"

    f = subtree(f_env, ("events", "rsi_reference_reset"))
    o = subtree(o_env, ("events", "rsi_reference_reset"))
    diffs = {p for p, _, _ in diff_paths(f, o, limit=10_000)}
    assert diffs == {("func",), ("params", "bank_path"), ("params", "fix_spawn"),
                     ("params", "yaw_range"), ("params", "walk_floor"),
                     ("params", "source")}, f"{pair_id}: rsi delta {sorted(diffs)}"
    assert f["func"] == FWD_RSI_FUNC and o["func"] == OMNI_RSI_FUNC
    assert o["params"]["fraction"] == f["params"]["fraction"] == 0.2
    assert o["params"]["bank_path"].endswith("rsi_bank_omni.npz")
    assert o["params"]["source"] == EXPECTED_RSI_SOURCE[omni_phase]
    assert _close(o["params"]["walk_floor"], EXPECTED_WALK_FLOOR)
    assert _close_list(o["params"]["yaw_range"], EXPECTED_YAW_RANGE)
    # the corridor-era spawn knob is gone, not carried with a falsy value
    assert "fix_spawn" not in o["params"]
    assert o["mode"] == f["mode"] == "reset"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_terrain_generator_block(dumps, omni_dumps, pair):
    """``scene.terrain.terrain_generator``: the ring generator at this window's mix.

    Asserted against an explicit block rather than diffed: the forward corridor generator and
    the ring generator share no sub-terrain family, so a path-by-path diff of the two carries
    no information.
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f = subtree(_env(dumps[fwd_phase], fwd_task), ("scene", "terrain", "terrain_generator"))
    o = subtree(_env(omni_dumps[omni_phase], omni_task), ("scene", "terrain", "terrain_generator"))

    assert set(o["sub_terrains"]) == set(RING_SUB_KEYS), f"{pair_id}: {sorted(o['sub_terrains'])}"
    assert set(f["sub_terrains"]) & set(RING_SUB_KEYS) == {"parkour_flat"}, (
        "the corridor generator must share only the load-bearing 'parkour_flat' key"
    )
    for key, want in EXPECTED_GENERATOR_SCALARS.items():
        got = o[key]
        if isinstance(want, list):
            assert [float(v) for v in got] == want, f"{pair_id}: generator {key} = {got!r}"
        else:
            assert _close(got, want, 1e-12), f"{pair_id}: generator {key} = {got!r}"
    want = EXPECTED_TERRAIN[omni_phase]
    assert o["curriculum"] is want["curriculum"], f"{pair_id}: curriculum"
    assert [float(v) for v in o["difficulty_range"]] == want["difficulty_range"]
    for key in RING_SUB_KEYS:
        sub = o["sub_terrains"][key]
        share = want["flat"] if key == "parkour_flat" else want["ring"]
        assert _close(sub["proportion"], share), f"{pair_id}: {key} proportion {sub['proportion']}"
        assert sub["function"] == EXPECTED_SUB_FUNCS[key], f"{pair_id}: {key} function"
        assert [float(v) for v in sub["noise_range"]] == [0.02, 0.02], f"{pair_id}: {key} noise"
        assert _close(sub["clear_radius"], 2.0) and _close(sub["ring_outer_radius"], 4.5)
        if key in EXPECTED_RING_BLOCKS:
            assert sub["n_blocks"] == EXPECTED_RING_BLOCKS[key], f"{pair_id}: {key} n_blocks"
            assert _close(sub["bar_fraction"], 0.5)
            for h_key, h_want in EXPECTED_RING_HEIGHTS.items():
                assert sub[h_key] == h_want, f"{pair_id}: {key} {h_key}"
    # the generator-level geometry the forward corridor and the ring do NOT share
    assert [float(v) for v in f["size"]] != [float(v) for v in o["size"]]


# ------------------------------------------------------------------ the runner delta
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_agent_delta_is_exactly_names(dumps, omni_dumps, pair):
    """The runner delta is the experiment name (+ the mirror the symmetry loss points at)."""
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    fwd, omni = _agent(dumps[fwd_phase], fwd_task), _agent(omni_dumps[omni_phase], omni_task)

    diffs = diff_paths(fwd, omni, limit=10_000)
    expected = {("experiment_name",), ("algorithm", "symmetry_cfg", "data_augmentation_func")}
    got = {p for p, _, _ in diffs}
    assert got == expected, (
        f"{pair_id}: runner delta is not the names\n"
        f"  unexpected: {sorted(got - expected)}\n  missing: {sorted(expected - got)}\n{fmt_diffs(diffs)}"
    )
    assert omni["experiment_name"] == "crab_hex_omni_" + fwd["experiment_name"][len("crab_hex_"):]
    assert subtree(fwd, ("algorithm", "symmetry_cfg", "data_augmentation_func")) == FWD_MIRROR
    assert subtree(omni, ("algorithm", "symmetry_cfg", "data_augmentation_func")) == OMNI_MIRROR


# ------------------------------------------------------------------ the knob environment
@pytest.mark.parametrize("phase", sorted(OMNI_FIXTURES))
def test_environ_delta(dumps, omni_dumps, phase):
    """The expanded knob environment differs only by the preset variable, the dropped turning
    school and the omni block."""
    fwd_phase, exp_fwd_only, exp_omni_only, exp_differ = EXPECTED_ENVIRON[phase]
    fwd, omni = dumps[fwd_phase]["_environ"], omni_dumps[phase]["_environ"]

    assert set(fwd) - set(omni) == exp_fwd_only | {"KRABBY_PHASE"}, \
        f"{phase}: forward-only keys {sorted(set(fwd) - set(omni))}"
    assert set(omni) - set(fwd) == exp_omni_only | {"KRABBY_OMNI_PHASE"}, \
        f"{phase}: omni-only keys {sorted(set(omni) - set(fwd))}"
    differ = {k for k in set(fwd) & set(omni) if fwd[k] != omni[k]}
    assert differ == exp_differ, f"{phase}: differing values {sorted(differ)}"
    assert omni["KRABBY_OMNI_PHASE"] == phase
    assert "KRABBY_PHASE" not in omni, "the omni activation must never write KRABBY_PHASE"
    assert "KRABBY_SIDE_PHASE" not in omni
    assert "KRABBY_HEADING" not in omni, "_check_omni refuses a set KRABBY_HEADING"
    assert omni["KRABBY_FLAT_TERRAIN_GEOM"] == "ring"
    assert omni["KRABBY_TERRAIN_PROMOTE"] == "0.45:0.25"


# ------------------------------------------------------------------ within-omni deltas
@pytest.mark.parametrize("pair", [("1a", "1b"), ("1a", "1c"), ("1b", "1c")],
                         ids=["1a-1b", "1a-1c", "1b-1c"])
def test_phase1_stages_differ_only_in_the_sampler(omni_dumps, pair):
    """The three phase-1 stages are ONE env: only the command sampler's direction band and slot
    mix, the RSI frame source and (for 1c) the facing reward weight may move."""
    a, b = pair
    ea, eb = _env(omni_dumps[a], OMNI_WALK), _env(omni_dumps[b], OMNI_WALK)
    diffs = diff_paths(ea, eb, limit=10_000)
    got = {p for p, _, _ in diffs}
    expected = {("events", "rsi_reference_reset", "params", "source")}
    if b == "1c":
        # 1c drops the band entirely: a list against None is one whole-field difference, and the
        # slot mix widens from 2 active slots to 4
        expected |= {("commands", "base_velocity", "dir_band"), YAW_WEIGHT_PATH}
        expected |= _indexed(("commands", "base_velocity", "slot_probs"), 4)
    else:
        # 1a -> 1b turns the band a quarter turn: two bands of equal length, compared per index
        expected |= _indexed(("commands", "base_velocity", "dir_band"), 2)
    assert got == expected, (
        f"{a} vs {b}: unexpected {sorted(got - expected)}, missing {sorted(expected - got)}\n"
        f"{fmt_diffs(diffs)}"
    )


@pytest.mark.parametrize("phase", ["2a", "2b", "2c"])
def test_omni_phase2_teacher_mode_matches_omni_walk(omni_dumps, phase):
    """Twin of the forward guard: the phase-2 teacher task and the Omni-Walk task build the same MDP."""
    t, w = omni_dumps[phase][OMNI_TEACHER], omni_dumps[phase][OMNI_WALK]
    env_diffs = diff_paths(normalize(t["env"]), normalize(w["env"]), limit=10_000)
    assert not env_diffs, f"{phase}: Omni-Teacher-v0 env cfg != Omni-Walk-v0 env cfg\n{fmt_diffs(env_diffs)}"
    agent_diffs = diff_paths(strip(t["agent"], AGENT_VOLATILE), strip(w["agent"], AGENT_VOLATILE),
                             limit=10_000)
    assert not agent_diffs, f"{phase}: omni teacher runner != omni walk runner\n{fmt_diffs(agent_diffs)}"
    assert t["agent"]["experiment_name"] == "crab_hex_omni_teacher"
    assert omni_dumps[phase]["_environ"]["KRABBY_HEX_TEACHER_MODE"] == phase


def test_omni_play_variants_keep_the_train_mdp(dumps, omni_dumps):
    """The PLAY twins are visualization only: same MDP, same terrain difficulty as training."""
    a = normalize(omni_dumps["2c"][OMNI_TEACHER_PLAY]["env"])
    b = normalize(omni_dumps["2c"][OMNI_WALK_PLAY]["env"])
    diffs = diff_paths(a, b, limit=10_000)
    assert not diffs, f"2c: Omni-Teacher-Play != Omni-Walk-Play\n{fmt_diffs(diffs)}"
    train = normalize(omni_dumps["2c"][OMNI_TEACHER]["env"])
    assert subtree(a, ("scene", "terrain", "terrain_generator", "difficulty_range")) == \
        subtree(train, ("scene", "terrain", "terrain_generator", "difficulty_range"))
    # the PLAY head (episode_length_s applied AFTER the knobs) is the forward one, copied
    for omni_play, fwd_play, phase in ((OMNI_TEACHER_PLAY, TEACHER_PLAY, "2c"),
                                       (OMNI_WALK_PLAY, FLAT_PLAY, "2c"),
                                       (OMNI_WALK_PLAY, FLAT_PLAY, "1a")):
        assert _env(omni_dumps[phase], omni_play)["episode_length_s"] == \
            _env(dumps[phase], fwd_play)["episode_length_s"]
    assert subtree(a, ("parkours", "base_parkour", "debug_vis")) is True
    assert subtree(a, ("commands", "base_velocity", "debug_vis")) is True


# ------------------------------------------------------------------ negative pillars
OMNI_CFG_ATTRS = ("omni_command", "omni_clock_turn_radius")
OMNI_CMD_ATTRS = CMD_FIELDS_ADDED
OMNI_PARKOUR_ATTRS = tuple(EXPECTED_PARKOUR_FIELDS)


@pytest.mark.parametrize("sibling", ["forward", "sideways"])
def test_siblings_carry_no_omni_attribute(dumps, side_dumps, sibling):
    """Neither lineage of record may grow an omni field -- that is what keeps the forward AND
    sideways dumps, their recorded windows and their bundled heads byte-identical."""
    if sibling == "forward":
        cases = [(dumps["1a"], FLAT), (dumps["2c"], TEACHER)]
    else:
        cases = [(side_dumps["1a"], SIDE_WALK), (side_dumps["2c"], SIDE_TEACHER)]
    for d, task in cases:
        e = _env(d, task)
        for attr in OMNI_CFG_ATTRS:
            assert attr not in e, f"{sibling} {task}: env cfg grew {attr}"
        for attr in OMNI_CMD_ATTRS:
            assert attr not in e["commands"]["base_velocity"], \
                f"{sibling} {task}: command cfg grew {attr}"
        for attr in OMNI_PARKOUR_ATTRS:
            assert attr not in e["parkours"]["base_parkour"], \
                f"{sibling} {task}: parkour term grew {attr}"
        assert "omni" not in e["parkours"]["base_parkour"]["class_type"].lower()
        # the added omni reward is omni-only; KRABBY_YAW_PROGRESS_W is inert in a
        # forward/sideways process (nothing there reads it)
        assert "reward_yaw_progress" not in e["rewards"], \
            f"{sibling} {task}: rewards grew reward_yaw_progress"
        # 2026-09-18 (plan Part A2): the second added term is omni-only too, and
        # KRABBY_OMNI_FACING_TURN_W is inert in a forward/sideways process (the omni knob
        # module is never imported there)
        assert "reward_facing_turn" not in e["rewards"], \
            f"{sibling} {task}: rewards grew reward_facing_turn"
        assert "facing_obs_encoding" not in e["observations"]["policy"]["extreme_parkour_observations"]["params"], \
            f"{sibling} {task}: obs term grew facing_obs_encoding"
        # Revision 3 (2026-09-17): the facing-error observation clip is omni-only -- neither
        # sibling's obs term takes a ``facing_obs_clip`` param, and KRABBY_OMNI_FACING_OBS_CLIP
        # is inert in a forward/sideways process (the omni knob module is never imported there)
        assert "facing_obs_clip" not in e["observations"]["policy"]["extreme_parkour_observations"]["params"], \
            f"{sibling} {task}: obs term grew facing_obs_clip"


@pytest.mark.parametrize("phase", sorted(OMNI_FIXTURES))
def test_omni_carries_no_travel_frame_field(omni_dumps, phase):
    """The omni cfg has NO ``travel_yaw`` / ``heading_offset``: with a 2-D body command the
    stock travel-frame seams are already correct, and the identity path is the right one."""
    task = OMNI_WALK if phase.startswith("1") else OMNI_TEACHER
    e = _env(omni_dumps[phase], task)
    assert "travel_yaw" not in e, f"{phase}: the omni env cfg grew a travel_yaw field"
    assert "heading_offset" not in e["commands"]["base_velocity"], \
        f"{phase}: the omni command cfg grew a heading_offset field"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_yaw_progress_term_block(omni_dumps, pair):
    """``rewards.reward_yaw_progress``: the added term, armed in EVERY window at 1.0.

    The weight comes from ``KRABBY_YAW_PROGRESS_W`` (``omni_knobs.apply_omni_knobs``), not from
    the forward ``_overrides`` table, so this also pins that the knob actually reaches the term
    -- the one thing that could leave the whole revision inert. ``clip: None`` must survive as
    a real ``None``: it is what makes the reward read the command's own ``ang_vel_max``, so
    retuning ``KRABBY_OMNI_ANG_VEL_MAX`` cannot desynchronise cap and reward.
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    o = subtree(_env(omni_dumps[omni_phase], omni_task), ("rewards", "reward_yaw_progress"))
    assert o["func"] == f"{_OMNI_REWARDS}:reward_yaw_progress", f"{pair_id}: {o['func']}"
    assert _close(o["weight"], EXPECTED_YAW_PROGRESS_W), f"{pair_id}: weight {o['weight']!r}"
    assert set(o["params"]) == {"command_name", "asset_cfg", "clip"}, sorted(o["params"])
    assert o["params"]["command_name"] == "base_velocity"
    assert o["params"]["clip"] is None, f"{pair_id}: clip must stay None (read the command cap)"
    assert o["params"]["asset_cfg"]["name"] == "robot"
    # the claim that the term has nothing to pay for in 1a/1b is a property of the SAMPLER, not
    # of the weight: those stages draw no turning slot, so no commanded turn exists there (a
    # yaw drift past the dead band can still open the gate, and is then paid as a heading hold).
    if omni_phase in ("1a", "1b"):
        slots = EXPECTED_CMD[omni_phase]["slot_probs"]
        assert slots[2] == slots[3] == 0.0, f"{pair_id}: 1a/1b must have no turning slots"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_facing_turn_term_block(omni_dumps, pair):
    """``rewards.reward_facing_turn``: the 37th name, present in every window at weight 0.0.

    The weight is the whole bitwise-identity claim of Part A2: at 0.0 this term contributes
    nothing to any return and its only trace is one extra ``Episode_Reward/reward_facing_turn``
    label printing 0 on every line -- which is exactly what Verification 3(a) allows a legacy
    smoke to gain. If a preset ever arms it, that comparison stops being valid and this test
    says so. The dose lives in ``KRABBY_OMNI_FACING_TURN_W``, which no preset exports (pinned
    preset-side by ``tests/unit/test_crab_hex_omni_phases.py``).
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    o = subtree(_env(omni_dumps[omni_phase], omni_task), ("rewards", "reward_facing_turn"))
    assert o["func"] == f"{_OMNI_REWARDS}:reward_facing_exp_turn_slot", f"{pair_id}: {o['func']}"
    assert o["weight"] == 0.0, f"{pair_id}: the shipped dose must be OFF, got {o['weight']!r}"
    assert set(o["params"]) == {"asset_cfg", "parkour_name", "command_name"}, sorted(o["params"])
    assert o["params"]["command_name"] == "base_velocity"
    assert o["params"]["parkour_name"] == "base_parkour"
    assert o["params"]["asset_cfg"]["name"] == "robot"
    # it is the ungated facing term plus a gate: same params, different func, own weight
    yaw = subtree(_env(omni_dumps[omni_phase], omni_task), ("rewards", "reward_tracking_yaw"))
    assert o["params"] == yaw["params"], f"{pair_id}: the two facing terms must agree on params"
    assert yaw["func"] == f"{_OMNI_REWARDS}:reward_facing_exp"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_facing_obs_encoding_is_raw_in_every_window(omni_dumps, pair):
    """The observation the policy trains on is the shipped ``raw`` one, in every window.

    The param exists only so ``KRABBY_OMNI_FACING_OBS_ENCODING`` has a declared key to write
    (the applier refuses a term without it); no preset sets the knob, so every dump must read
    ``"raw"`` -- the positive twin of the ``(MISSING, "raw")`` entry in ``EXPECTED_RL``. The
    facing-error CLIP is unchanged beside it: under ``raw`` it is what clamps slots 6/7.
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    params = subtree(
        _env(omni_dumps[omni_phase], omni_task),
        ("observations", "policy", "extreme_parkour_observations", "params"),
    )
    assert params["facing_obs_encoding"] == "raw", f"{pair_id}: {params['facing_obs_encoding']!r}"
    assert _close(params["facing_obs_clip"], 0.6), f"{pair_id}: {params['facing_obs_clip']!r}"


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_ang_vel_std_is_the_forward_kernel(dumps, omni_dumps, pair):
    """Revision 4 (2026-09-18): ``track_ang_vel_z_exp.params.std`` must AGREE with the forward
    task at sqrt(0.25) = 0.5, in every window.

    The omni rewards class body still declares 0.15 (plan D4), so the agreement exists ONLY
    because ``KRABBY_OMNI_ANG_VEL_STD=0.5`` in ``OMNI_COMMON`` reaches the term through
    ``omni_knobs._apply_ang_vel_std_knob``. That makes this the positive twin of the path's
    ABSENCE from ``EXPECTED_RL``: if the knob ever stops being applied, the delta test reports an
    unexpected path AND this test reports 0.15, instead of the run quietly re-acquiring the
    kernel that taxed gait wobble on 88 % of steps.
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f = _env(dumps[fwd_phase], fwd_task)
    o = _env(omni_dumps[omni_phase], omni_task)
    f_std = f["rewards"]["track_ang_vel_z_exp"]["params"]["std"]
    o_std = o["rewards"]["track_ang_vel_z_exp"]["params"]["std"]
    assert _close(o_std, EXPECTED_ANG_VEL_STD), f"{pair_id}: omni std {o_std!r}"
    assert _close(f_std, EXPECTED_ANG_VEL_STD), f"{pair_id}: forward std {f_std!r}"
    # the term itself is otherwise untouched: same func, same weight, same command
    assert o["rewards"]["track_ang_vel_z_exp"]["func"] == f["rewards"]["track_ang_vel_z_exp"]["func"]
    assert _close(o["rewards"]["track_ang_vel_z_exp"]["weight"],
                  f["rewards"]["track_ang_vel_z_exp"]["weight"])


@pytest.mark.parametrize("pair", ALL_PAIRS, ids=[p[0] for p in ALL_PAIRS])
def test_clock_schedule_combine_is_product_on_the_omni_side_only(dumps, omni_dumps, pair):
    """Revision 4: the omni clock schedule multiplies swing quality by stance quality.

    The forward side must still read ``"sum"`` -- the omni knob is applied by
    ``apply_omni_knobs`` in an omni process only, and no forward preset moves. The swing APEX
    term must carry NO ``combine`` key on either side: ``clock_swing_apex_income`` takes no such
    argument, and a key written there would raise inside the reward manager on the first step.
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    f = _env(dumps[fwd_phase], fwd_task)
    o = _env(omni_dumps[omni_phase], omni_task)
    assert f["rewards"]["reward_clock_schedule"]["params"]["combine"] == "sum", pair_id
    assert o["rewards"]["reward_clock_schedule"]["params"]["combine"] == "product", pair_id
    assert "combine" not in o["rewards"]["reward_clock_swing_apex"]["params"], pair_id
    assert "combine" not in f["rewards"]["reward_clock_swing_apex"]["params"], pair_id


@pytest.mark.parametrize("pair", RL_PAIRS, ids=[p[0] for p in RL_PAIRS])
def test_reward_names_and_scan_pattern_unchanged(dumps, omni_dumps, pair):
    """The forward 35 reward names are a SUBSET of the omni 37 and the extra two are
    ``reward_yaw_progress`` and ``reward_facing_turn``; same termination names; the scan grid
    NOT re-shaped.

    The subset direction is the load-bearing half: ``Episode_Reward/<name>``,
    ``KRABBY_PHASEOUT``'s term list and the driver's log scraping all key on the forward
    names, so a RENAME or a DROP breaks them silently. Pure additions are safe -- and they
    have to be these two and only these two, or the omni stack has grown shaping nobody
    reviewed. (2026-09-18: ``reward_facing_turn`` is the second, at weight 0.0 -- the ONLY new
    label a legacy-path smoke may print, and it must print 0 on every line.)
    """
    pair_id, omni_phase, fwd_phase, fwd_task, omni_task = pair
    fwd, omni = _env(dumps[fwd_phase], fwd_task), _env(omni_dumps[omni_phase], omni_task)
    assert set(fwd["rewards"]) <= set(omni["rewards"]), \
        f"{pair_id}: forward reward names missing from omni: {sorted(set(fwd['rewards']) - set(omni['rewards']))}"
    assert set(omni["rewards"]) - set(fwd["rewards"]) == {
        "reward_yaw_progress", "reward_facing_turn",
    }, f"{pair_id}: unexpected omni-only reward terms: {sorted(set(omni['rewards']) - set(fwd['rewards']))}"
    assert len(fwd["rewards"]) == 35, f"{pair_id}: expected 35 forward reward terms, got {len(fwd['rewards'])}"
    assert len(omni["rewards"]) == 37, f"{pair_id}: expected 37 omni reward terms, got {len(omni['rewards'])}"
    assert set(omni["terminations"]) == set(fwd["terminations"]), \
        f"{pair_id}: termination names changed"
    f_pat = subtree(fwd, ("scene", "height_scanner", "pattern_cfg"))
    o_pat = subtree(omni, ("scene", "height_scanner", "pattern_cfg"))
    for key in ("size", "resolution", "ordering", "direction"):
        assert f_pat[key] == o_pat[key], f"{pair_id}: scan pattern {key} changed"


def test_scan_offset_lies_on_both_mirror_planes(dumps, omni_dumps):
    """The omni symmetry augmentation reflects through BOTH body planes, so the scan look-ahead
    must sit on their intersection -- the body origin -- or the scan permutation is wrong for
    the observation it augments."""
    fwd = subtree(_env(dumps["1a"], FLAT), ("scene", "height_scanner", "offset", "pos"))
    omni = subtree(_env(omni_dumps["1a"], OMNI_WALK), ("scene", "height_scanner", "offset", "pos"))
    assert fwd[1] == 0.0, f"forward scan offset is off the y = 0 mirror plane: {fwd}"
    assert (omni[0], omni[1]) == (0.0, 0.0), f"omni scan offset is off a mirror plane: {omni}"
    assert omni[2] == fwd[2], "the scanner ray origin height changed"


# ------------------------------------------------ the STAGED LEVER (no dump; pure Python)
def test_staged_turn_gate_lever_is_off_in_every_dump(dumps, omni_dumps):
    """The shipped state: ``track_ang_vel_z_exp.func`` is the STOCK isaaclab term on BOTH sides.

    ``KRABBY_OMNI_ANG_TURN_ONLY`` (staged 2026-09-18, user decision) is set by no preset, so the
    omni dumps must carry the same func as the forward ones -- already asserted for every pair by
    ``test_ang_vel_std_is_the_forward_kernel``; this states the negative directly, and names the
    turn-gated twin so a grep for it finds the "must NOT be here" line too.
    """
    for pair_id, omni_phase, fwd_phase, fwd_task, omni_task in ALL_PAIRS:
        f = _env(dumps[fwd_phase], fwd_task)["rewards"]["track_ang_vel_z_exp"]["func"]
        o = _env(omni_dumps[omni_phase], omni_task)["rewards"]["track_ang_vel_z_exp"]["func"]
        assert o == f, f"{pair_id}: omni func {o!r} != forward {f!r}"
        gated = f"{_OMNI_REWARDS}:track_ang_vel_z_exp_turn_gated"
        assert o != gated, f"{pair_id}: the staged lever is armed and must not be"
        assert "isaaclab" in o and o.endswith(":track_ang_vel_z_exp"), f"{pair_id}: {o!r}"


def test_staged_turn_gate_lever_rewrites_exactly_one_func_path():
    """With the knob SET, the applier writes ``rewards.track_ang_vel_z_exp.func`` and nothing else.

    A pure unit test in this file on purpose: the delta above proves the OFF state that ships,
    and this proves what arming it would change -- without a seventh Kit boot, and without the
    run that arms it being the first thing to discover that the knob rewrites the wrong path.
    The two real functions are injected as stand-ins (the module's own ``gated_func``/
    ``stock_func`` hooks); ``tests/unit/test_crab_hex_omni_phases.py`` pins that those hooks
    resolve to the shipped pair, and ``tests/unit/test_crab_hex_omni_rewards.py`` pins that the
    twin is bitwise the stock term on turning rows and 0 elsewhere.
    """
    import importlib.util
    import sys

    knobs_py = (Path(__file__).resolve().parents[2] / "parkour/parkour_tasks/parkour_tasks"
                / "crab_hex_omni_task/config/crab_hex/omni_knobs.py")
    spec = importlib.util.spec_from_file_location("omni_knobs__delta_test", knobs_py)
    knobs = importlib.util.module_from_spec(spec)
    sys.modules["omni_knobs__delta_test"] = knobs
    spec.loader.exec_module(knobs)

    def _stock(env, std, command_name="base_velocity", asset_cfg=None):
        pass

    def _gated(env, std, command_name="base_velocity", asset_cfg=None):
        pass

    class _Term:
        def __init__(self, func, **params):
            self.func = func
            self.weight = 1.0
            self.params = dict(params)

        def snapshot(self):
            return (self.func, self.weight, dict(self.params))

    class _Cfg:
        def __init__(self):
            self.rewards = types.SimpleNamespace(
                track_ang_vel_z_exp=_Term(_stock, command_name="base_velocity", std=0.5),
                track_lin_vel_xy_exp=_Term(_stock, command_name="base_velocity", std=0.2),
                reward_yaw_progress=_Term(_stock, command_name="base_velocity", clip=None),
                reward_clock_schedule=_Term(_stock, command_name="base_velocity", combine="product"),
            )

    cfg = _Cfg()
    before = {n: t.snapshot() for n, t in vars(cfg.rewards).items()}
    knobs._apply_ang_turn_only_knob(
        cfg, {"KRABBY_OMNI_ANG_TURN_ONLY": "1"}, gated_func=_gated, stock_func=_stock
    )
    assert cfg.rewards.track_ang_vel_z_exp.func is _gated
    # the ONE path: same weight, same params, and every other reward term byte-identical
    assert cfg.rewards.track_ang_vel_z_exp.weight == before["track_ang_vel_z_exp"][1]
    assert cfg.rewards.track_ang_vel_z_exp.params == before["track_ang_vel_z_exp"][2]
    for name, term in vars(cfg.rewards).items():
        if name == "track_ang_vel_z_exp":
            continue
        assert term.snapshot() == before[name], f"{name} changed"

    # and the OFF spellings leave even that path alone
    for environ in ({}, {"KRABBY_OMNI_ANG_TURN_ONLY": "0"}, {"KRABBY_OMNI_ANG_TURN_ONLY": ""}):
        cfg2 = _Cfg()
        knobs._apply_ang_turn_only_knob(
            cfg2, environ, gated_func=_gated, stock_func=_stock
        )
        assert cfg2.rewards.track_ang_vel_z_exp.func is _stock, environ


def test_omni_spawn_rotation_is_identity(omni_dumps):
    """The ring reset draws a uniform world yaw at every reset; a baked-in spawn attitude would
    only bias the first frame (and is what the sideways scene does -- an easy copy-paste)."""
    rot = subtree(_env(omni_dumps["1a"], OMNI_WALK), ("scene", "robot", "init_state", "rot"))
    assert [float(v) for v in rot] == [1.0, 0.0, 0.0, 0.0], f"omni spawn rot is {rot}"
