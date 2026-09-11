"""The travel-frame seams sit at exactly the sites the sideways-task plan names -- no more.

The "travel frame" (``parkour_isaaclab/envs/mdp/travel_frame.py``) lets a sibling task walk
along a rotated body axis while the forward-walk task keeps its byte-identical default path.
That only holds if the body-velocity substitution happens at the *audited* sites: the seven
body-vs-command comparisons and the yaw references enumerated in the plan. A stray extra
``_vel_travel_xy`` (or a missing one) is a silent behaviour change for the sideways task and a
silent review gap for the forward one, so this test reads the sources with ``ast`` and pins the
exact set. It imports nothing -- no torch, no Isaac.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIB = REPO / "parkour" / "parkour_isaaclab"
PKG = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
REWARDS_PY = LIB / "envs" / "mdp" / "rewards.py"
OBSERVATIONS_PY = PKG / "mdp" / "observations.py"

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
