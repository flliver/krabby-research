"""The travel-frame seams sit at exactly the sites the sideways-task plan names -- no more.

The "travel frame" (``parkour_isaaclab/envs/mdp/travel_frame.py``) lets a sibling task walk
along a rotated body axis while the forward-walk task keeps its byte-identical default path.
That only holds if the body-velocity substitution happens at the *audited* sites: the seven
body-vs-command comparisons and the yaw references enumerated in the plan. A stray extra
``_vel_travel_xy`` (or a missing one) is a silent behaviour change for the sideways task and a
silent review gap for the forward one, so this test reads the sources with ``ast`` and pins the
exact set. It imports nothing -- no torch, no Isaac.

The omni-task plan (D3) adds two more seams of the same shape, pinned the same way: the
command-direction gate ``_cmd_gate`` (five sites) and the gait-clock cadence
``_clock_cmd_speed`` / ``clock_command_speed`` (two reward sites plus the action term).
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour" / "parkour_isaaclab"
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
REWARDS_PY = LIB / "envs" / "mdp" / "rewards.py"
OBSERVATIONS_PY = PKG / "mdp" / "observations.py"
ACTIONS_PY = PKG / "mdp" / "parkour_actions.py"

# The only reward terms whose body velocity is read in the travel frame. Everything else --
# ``reward_tracking_goal_vel*`` (world-frame projection on the goal direction), ``reward_lin_vel_z``,
# ``penalty_base_pitch_forward_*``, ``penalty_backward_along_command``, ``reward_forward_speed_on_flat``,
# ``penalty_low_forward_speed_when_commanded``, ``RewardRecoverFromStall`` and every ``cmd[:, 0]``
# gate -- stays body-frame by decision (see the sideways-task plan, "What does NOT change").
EXPECTED_TRAVEL_VEL_SITES = {
    "penalty_lin_vel_y_l2",
    "reward_forward_progress_along_command",
    "penalty_tracking_error_l1",
    "reward_obstacle_clearance",
    "track_lin_vel_travel_exp",
    "RewardStrideLength",
}

# The command-direction seams (omni-task plan D3). ``_cmd_gate`` is the single place the
# "is a travel command active?" question is asked for a task whose command is not a scalar
# along body +x; every other ``cmd[:, 0]`` comparison in the file is a FORWARD-SPECIFIC term
# left inert for omni by decision (``penalty_backward_along_command``,
# ``penalty_base_pitch_forward_*``, ``reward_forward_speed_on_flat``,
# ``penalty_low_forward_speed_when_commanded``, ``RewardRecoverFromStall``, the bridge/2b2
# terms, ...). An extra site here is a silent behaviour change for the omni task; a missing
# one is a term that never fires under a lateral or backward command.
EXPECTED_CMD_GATE_SITES = {
    "reward_foot_clearance",
    "penalty_swing_min_clearance",
    "penalty_excess_feet_in_contact_forward",
    "reward_stance_support_feet_when_forward",
    "PenaltyFootIdleWhenForward",
}

# The gait clock's cadence, read through ``clock_command_speed``: the two clock-income
# rewards here, and ``process_actions`` in the action term (asserted separately below). All
# three must agree, or the reward prices a schedule the clock is not advancing.
EXPECTED_CLOCK_SPEED_SITES = {
    "RewardClockContactSchedule",
    "RewardClockSwingApex",
    # Revision 2 (2026-09-17): the command gate itself reads the clock speed in its omni branch,
    # so a pure turn command switches the gait-shaping terms on (the forward branch is untouched).
    "_cmd_gate",
}


def _top_level_defs(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(), str(path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _references(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _call_sites(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(), str(path))
    return sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    )


def test_travel_velocity_substitution_sites_are_exactly_the_audited_ones():
    defs = _top_level_defs(REWARDS_PY)
    got = {name for name, node in defs.items() if _references(node, "_vel_travel_xy")}
    assert got == EXPECTED_TRAVEL_VEL_SITES


def test_tracking_yaw_is_the_only_yaw_seam_in_rewards():
    defs = _top_level_defs(REWARDS_PY)
    assert "reward_tracking_yaw" in defs
    assert _references(defs["reward_tracking_yaw"], "_travel_yaw")


def test_observations_has_exactly_three_yaw_error_sites():
    # delta_yaw + delta_next_yaw in CrabHexParkourObservations, and the student gate in
    # CrabHexObservationDeltaYawOk. Dims 13/14 and _get_priv_explicit stay body-frame.
    assert _call_sites(OBSERVATIONS_PY, "yaw_error") == 3


def test_command_gate_sites_are_exactly_the_audited_ones():
    defs = _top_level_defs(REWARDS_PY)
    got = {name for name, node in defs.items() if _references(node, "_cmd_gate")}
    assert got == EXPECTED_CMD_GATE_SITES


def test_clock_command_speed_sites_are_exactly_the_audited_ones():
    defs = _top_level_defs(REWARDS_PY)
    got = {name for name, node in defs.items() if _references(node, "_clock_cmd_speed")}
    assert got == EXPECTED_CLOCK_SPEED_SITES


def test_the_gait_clock_advance_reads_the_shared_speed_seam():
    """The action term's ``process_actions`` must not keep its own copy of the expression."""
    tree = ast.parse(ACTIONS_PY.read_text(), str(ACTIONS_PY))
    methods = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "process_actions"
    ]
    assert len(methods) == 1, "expected exactly one process_actions in the action term"
    assert _references(methods[0], "clock_command_speed")
