"""The omni reward twins, the ADDED yaw-progress term, the goal-free termination and their cfg
(plan D4/A6).

Four of these are a same-NAME, different-FUNCTION swap: nothing downstream (telemetry
labels, ``KRABBY_*_W`` knobs, the phase-out ramps, the driver's log scraping) can tell that
the arithmetic changed, so an error here surfaces only as a policy that quietly optimises
the wrong thing for ten hours. The fifth, ``reward_yaw_progress``, is the one ADDED term (the
36th name, 2026-09-17) and is worse: a sign error in it does not merely fail to help, it pays
the policy to spin AWAY from its facing target, and every other number in the training log
(``facing_err_abs``, ``error_vel_yaw``, the tripod score) looks the same either way -- which
is exactly how the 500-iteration yaw smoke burned 12 minutes turning the wrong way without
one metric saying so. Its sign chain is therefore pinned against the REAL
``heading_rate_command``, not against a restatement of the reward's own expression.

The anchors are therefore:

* the cross-command penalty must be BITWISE the forward ``penalty_lin_vel_y_l2`` whenever
  the command is the forward task's (a pure +-x command) -- the omni task is a
  generalisation, not a retuning;
* the command-velocity term must be the forward ``reward_tracking_goal_vel`` evaluated with
  the commanded direction and ``||cmd_xy||``;
* the facing reward must wrap (an error of ``2*pi - 0.1`` is an error of ``-0.1``);
* the yaw-progress reward must pay a body yaw rate that reduces the facing error the COMMAND
  MATH produces, clip at the command's own ``ang_vel_max``, and be exactly zero wherever no
  turn is commanded;
* the termination must be the forward one minus the goal cutoff;
* the cfg must carry the forward task's 35 term names plus ``reward_yaw_progress`` and
  ``reward_facing_turn`` (37) -- the second at weight 0.0, which is what makes its addition
  bitwise invisible to every run that does not set ``KRABBY_OMNI_FACING_TURN_W``;
* the two A2 CANDIDATES (2026-09-18, DEFAULT OFF) must each reduce to something already
  pinned: ``reward_facing_exp_turn_slot`` is ``reward_facing_exp`` times a slot mask (and
  raises without a ``slot_id``), and ``track_lin_vel_xy_exp_facing_coupled`` is the SHIPPED
  Isaac Lab ``track_lin_vel_xy_exp`` -- CALLED, not re-derived -- times a Gaussian of the
  wrapped facing error, bitwise the stock term at zero error;
* the STAGED LEVER ``track_ang_vel_z_exp_turn_gated`` (2026-09-18, DEFAULT OFF) must be
  BITWISE the shipped Isaac Lab ``track_ang_vel_z_exp`` on rows that command a turn and
  exactly 0.0 on rows that do not -- and must reach that by CALLING the stock function, not by
  restating its kernel. The stock half of that comparison is lifted out of Isaac Lab's own
  ``rewards.py`` (located by ``find_spec``, never imported), so "bitwise" means against what
  actually ships. The cfg's default-OFF half is pinned by
  ``test_track_ang_vel_z_exp_keeps_its_function_and_gets_std_0_15`` (the class body still
  declares the stock func) and, preset-side, by ``test_crab_hex_omni_phases.py``.

The shipped sources are lifted out of the modules with ``ast`` and exec'd against stubs --
no Isaac import, no ``sys.modules`` registration that could collide with another unit file.
"""

import ast
import importlib.util
import math
import sys
import textwrap
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
LIB_REWARDS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/rewards.py"
def _isaaclab_rewards_py():
    """Isaac Lab's OWN ``rewards.py`` on disk, or ``None`` -- located, never imported.

    The staged turn-gated lever is asserted BITWISE against the real shipped
    ``track_ang_vel_z_exp`` lifted out of this file, so "bitwise" means against what actually
    ships rather than against a restatement of the kernel. Two things rule out the obvious ways
    of finding it, and both were observed rather than guessed:

    * importing ``isaaclab`` drags in ``pxr`` (Kit), which this suite must not need;
    * ``importlib.util.find_spec`` consults ``sys.modules`` FIRST, and
      ``test_parkour_reward_manager_epsilon.py`` installs a stub ``sys.modules["isaaclab"]``
      whose ``__spec__`` is ``None`` -- so ``find_spec`` raises ``ValueError`` when that file is
      collected first, i.e. the anchor's availability would depend on collection order.
      ``PathFinder`` is no use either: Isaac Lab is an EDITABLE install reached through a
      ``sys.meta_path`` finder, not a plain ``sys.path`` directory.

    Walking ``sys.meta_path`` is what ``find_spec`` does minus the ``sys.modules`` lookup, so it
    resolves the same origin whether or not a stub is registered.
    """
    for finder in sys.meta_path:
        try:
            spec = finder.find_spec("isaaclab", None)
        except Exception:  # noqa: BLE001 -- a hostile finder must not break collection
            continue
        if spec is not None and spec.origin:
            path = Path(spec.origin).parent / "envs" / "mdp" / "rewards.py"
            return path if path.exists() else None
    return None


ISAACLAB_REWARDS_PY = _isaaclab_rewards_py()
LIB_TERMINATIONS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/terminations.py"
OMNI_REWARDS_PY = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_rewards.py"
OMNI_CMD_MATH_PY = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_command_math.py"
FWD_MDP_CFG_PY = TASKS / "crab_hex_forward_task/config/crab_hex/agents/parkour_mdp_cfg.py"
TRAVEL_FRAME_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/travel_frame.py"

N = 8


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


travel_frame = _load(TRAVEL_FRAME_PY, "travel_frame__omni_rew_test")
wrap_to_pi = travel_frame.wrap_to_pi

# The SHIPPED command math (pure torch, no Isaac): the yaw-progress sign chain is pinned
# against the real ``heading_rate_command``, so a change to the P-control's sign, its
# saturation or its dead band has to break this file too.
omni_cmd_math = _load(OMNI_CMD_MATH_PY, "omni_command_math__omni_rew_test")
heading_rate_command = omni_cmd_math.heading_rate_command


def _nodes(path: Path):
    src = path.read_text()
    return src, ast.parse(src, str(path))


def _segment(src, node) -> str:
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


def _exec_module_level(path: Path, names, ns: dict) -> dict:
    """Exec the named module-level functions / assignments from ``path`` into ``ns``."""
    src, tree = _nodes(path)
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            chunks.append(_segment(src, node))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            chunks.append(_segment(src, node))
    missing = [n for n in names if all(n not in c.split("\n")[0] for c in chunks)]
    assert not missing, f"{path.name}: not found {missing}"
    source = "from __future__ import annotations\n" + "\n\n".join(chunks)
    exec(compile(source, str(path), "exec"), ns)
    return ns


def _euler_xyz_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class _SceneEntityCfg:
    def __init__(self, name="robot", **kw):
        self.name = name
        self.kw = kw

    def __eq__(self, other):
        return (
            isinstance(other, _SceneEntityCfg) and self.name == other.name and self.kw == other.kw
        )

    def __repr__(self):
        return f"SceneEntityCfg({self.name!r}, {self.kw})"


class _MdpRewardsStub:
    """The two shared helpers the omni twins call, answered from the fake env."""

    @staticmethod
    def _parkour_flat_mask(env, parkour_name):
        assert parkour_name == "base_parkour"
        return env.flat_mask

    @staticmethod
    def reward_feet_stumble(env, sensor_cfg):
        return env.stumble


#: The REAL ``isaaclab.envs.mdp.rewards.track_ang_vel_z_exp``, lifted by source (never
#: imported). ``None`` when Isaac Lab is not installed -- the two bitwise tests skip, the
#: structural ones do not.
_STOCK_NS: dict = {
    "torch": torch,
    "SceneEntityCfg": _SceneEntityCfg,
    "RigidObject": object,
    "ManagerBasedRLEnv": object,
}
if ISAACLAB_REWARDS_PY is not None:
    _exec_module_level(
        ISAACLAB_REWARDS_PY, ("track_ang_vel_z_exp", "track_lin_vel_xy_exp"), _STOCK_NS
    )
track_ang_vel_z_exp = _STOCK_NS.get("track_ang_vel_z_exp")
track_lin_vel_xy_exp = _STOCK_NS.get("track_lin_vel_xy_exp")
requires_isaaclab_source = pytest.mark.skipif(
    track_ang_vel_z_exp is None,
    reason="isaaclab's rewards.py not locatable -- the bitwise anchor needs the shipped source",
)

_OMNI_NS = {
    "torch": torch,
    "wrap_to_pi": wrap_to_pi,
    "euler_xyz_from_quat": _euler_xyz_from_quat,
    "SceneEntityCfg": _SceneEntityCfg,
    "Articulation": object,
    "ParkourManagerBasedRLEnv": object,
    "mdp_rewards": _MdpRewardsStub,
    # the omni twins CALL these names (they do not re-derive the kernels)
    "track_ang_vel_z_exp": track_ang_vel_z_exp,
    "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
    # the REAL slot constant the turn-slot gate imports -- not a retyped 2
    "SLOT_TURN": omni_cmd_math.SLOT_TURN,
}
_exec_module_level(
    OMNI_REWARDS_PY,
    (
        "_DIR_EPS",
        "_TRACK_V_MIN",
        "OMNI_ANG_VEL_STD",
        "track_ang_vel_z_exp_turn_gated",
        "_command_dir_and_speed",
        "_facing_error",
        "penalty_cross_command_vel_l2",
        "reward_tracking_command_vel",
        "reward_tracking_command_vel_on_obstacle",
        "reward_facing_exp",
        "reward_obstacle_clearance_omni",
        "_yaw_progress_clip",
        "reward_yaw_progress",
        "_slot_is_turning",
        "reward_facing_exp_turn_slot",
        "track_lin_vel_xy_exp_facing_coupled",
        "terminate_episode_no_goals",
    ),
    _OMNI_NS,
)

_FWD_NS = {
    "torch": torch,
    "wrap_to_pi": wrap_to_pi,
    "euler_xyz_from_quat": _euler_xyz_from_quat,
    "SceneEntityCfg": _SceneEntityCfg,
    "Articulation": object,
    "ParkourManagerBasedRLEnv": object,
    "ParkourEvent": object,
    # travel_yaw is absent on both the forward and the omni cfg, so the travel frame is the
    # identity: this is the shipped ``_vel_travel_xy`` at ``travel_yaw == 0.0``.
    "_vel_travel_xy": lambda asset, env: asset.data.root_lin_vel_b[:, :2],
}
_exec_module_level(
    LIB_REWARDS_PY, ("penalty_lin_vel_y_l2", "reward_tracking_goal_vel"), _FWD_NS
)
_exec_module_level(LIB_TERMINATIONS_PY, ("terminate_episode",), _FWD_NS)

penalty_cross_command_vel_l2 = _OMNI_NS["penalty_cross_command_vel_l2"]
reward_tracking_command_vel = _OMNI_NS["reward_tracking_command_vel"]
reward_tracking_command_vel_on_obstacle = _OMNI_NS["reward_tracking_command_vel_on_obstacle"]
reward_facing_exp = _OMNI_NS["reward_facing_exp"]
reward_obstacle_clearance_omni = _OMNI_NS["reward_obstacle_clearance_omni"]
reward_yaw_progress = _OMNI_NS["reward_yaw_progress"]
reward_facing_exp_turn_slot = _OMNI_NS["reward_facing_exp_turn_slot"]
track_lin_vel_xy_exp_facing_coupled = _OMNI_NS["track_lin_vel_xy_exp_facing_coupled"]
track_ang_vel_z_exp_turn_gated = _OMNI_NS["track_ang_vel_z_exp_turn_gated"]
terminate_episode_no_goals = _OMNI_NS["terminate_episode_no_goals"]
penalty_lin_vel_y_l2 = _FWD_NS["penalty_lin_vel_y_l2"]
reward_tracking_goal_vel = _FWD_NS["reward_tracking_goal_vel"]
terminate_episode = _FWD_NS["terminate_episode"]

ROBOT = _SceneEntityCfg("robot")
FOOTPADS = _SceneEntityCfg("contact_forces", body_names=".*_Footpad")


# --------------------------------------------------------------------------- the fake env
class _Data:
    pass


#: The command cfg's yaw-rate cap / dead band / P gain, as the omni presets write them
#: (``KRABBY_OMNI_ANG_VEL_MAX`` 0.3, ``clips.ang_vel_clip`` 0.2, stiffness 1.5).
ANG_VEL_MAX = 0.3
ANG_VEL_CLIP = 0.2
STIFFNESS = 1.5
#: |facing error| below which the P-control's dead band zeroes the yaw-rate command.
DEAD_BAND_RAD = ANG_VEL_CLIP / STIFFNESS


class _Asset:
    def __init__(self, n):
        self.data = _Data()
        self.data.root_ang_vel_b = torch.zeros(n, 3)
        self.data.root_lin_vel_b = torch.zeros(n, 3)
        self.data.root_vel_w = torch.zeros(n, 6)
        self.data.root_state_w = torch.zeros(n, 13)
        self.data.root_state_w[:, 3] = 1.0
        self.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
        self.data.heading_w = torch.zeros(n)
        self.data.projected_gravity_b = torch.zeros(n, 3)
        self.data.projected_gravity_b[:, 2] = -1.0


class _CmdCfg:
    """The one cfg field ``reward_yaw_progress`` reads off the command term."""

    def __init__(self, ang_vel_max=ANG_VEL_MAX):
        self.ang_vel_max = ang_vel_max


class _CmdTerm:
    def __init__(self, n):
        self.heading_target = torch.zeros(n)
        #: the 2-bit command regime the omni sampler drew (``moving + 2 * turning``);
        #: ``SLOT_STAND`` on every row until a test says otherwise
        self.slot_id = torch.zeros(n, dtype=torch.long)
        self.cfg = _CmdCfg()


class _CommandManager:
    def __init__(self, n):
        self.cmd = torch.zeros(n, 4)
        self.term = _CmdTerm(n)

    def get_command(self, name):
        assert name == "base_velocity"
        return self.cmd

    def get_term(self, name):
        assert name == "base_velocity"
        return self.term


class _ParkourEvent:
    def __init__(self, n):
        self.target_pos_rel = torch.zeros(n, 2)
        self.target_pos_rel[:, 0] = 1.0
        self.cur_goal_idx = torch.zeros(n, dtype=torch.long)


class _ParkourManager:
    def __init__(self, n):
        self.event = _ParkourEvent(n)

    def get_term(self, name):
        return self.event


class _TerrainGenerator:
    num_goals = 8


class _Terrain:
    class cfg:
        terrain_generator = _TerrainGenerator()


class _Scene:
    def __init__(self, asset):
        self._asset = asset
        self.terrain = _Terrain()

    def __getitem__(self, key):
        assert key == "robot"
        return self._asset


class _Env:
    def __init__(self, n=N):
        self.num_envs = n
        self.device = "cpu"
        self.asset = _Asset(n)
        self.scene = _Scene(self.asset)
        self.command_manager = _CommandManager(n)
        self.parkour_manager = _ParkourManager(n)
        self.flat_mask = torch.zeros(n)          # 0 = obstacle tile, 1 = parkour_flat
        self.stumble = torch.zeros(n)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.max_episode_length = 1000


def _rot(v_xy, yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v_xy[:, 0] - s * v_xy[:, 1], s * v_xy[:, 0] + c * v_xy[:, 1]), dim=-1)


# ------------------------------------------------------- penalty_cross_command_vel_l2
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_cross_command_penalty_is_the_forward_penalty_for_an_x_command(sign):
    """The generalisation must reduce to the special case, bitwise."""
    g = torch.Generator().manual_seed(2026)
    env = _Env()
    # speeds straddling the 0.12 gate, including a couple below it
    vx = sign * torch.tensor([0.0, 0.05, 0.12, 0.1200001, 0.2, 0.35, 0.5, 1.0])
    env.command_manager.cmd[:, 0] = vx
    env.command_manager.cmd[:, 1] = 0.0
    env.asset.data.root_lin_vel_b[:, :2] = (torch.rand(N, 2, generator=g) - 0.5) * 2.0

    omni = penalty_cross_command_vel_l2(env, "base_velocity", ROBOT)
    fwd = penalty_lin_vel_y_l2(env, "base_velocity", ROBOT)
    assert torch.equal(omni, fwd)
    # and it is really the y velocity squared where the gate is open
    gate = vx.abs() > 0.12
    assert torch.equal(omni[gate], torch.square(env.asset.data.root_lin_vel_b[gate, 1]))
    assert torch.equal(omni[~gate], torch.zeros(int((~gate).sum())))


def test_cross_command_penalty_is_zero_without_a_command():
    env = _Env()
    env.asset.data.root_lin_vel_b[:, :2] = torch.tensor([[0.4, 0.3]] * N)
    env.command_manager.cmd[:, 0:2] = 0.0
    assert torch.equal(penalty_cross_command_vel_l2(env, "base_velocity", ROBOT), torch.zeros(N))
    # a pure turn-in-place command (slot 2 only) is still "not moving"
    env.command_manager.cmd[:, 2] = 0.3
    assert torch.equal(penalty_cross_command_vel_l2(env, "base_velocity", ROBOT), torch.zeros(N))
    # ... and a command just under the gate pays nothing either
    env.command_manager.cmd[:, 0] = 0.11
    assert torch.equal(penalty_cross_command_vel_l2(env, "base_velocity", ROBOT), torch.zeros(N))


def test_cross_command_penalty_prices_the_across_component_in_every_direction():
    """Rotate command and velocity together: the penalty is invariant."""
    g = torch.Generator().manual_seed(11)
    env = _Env()
    base_cmd = torch.tensor([[0.3, 0.0]] * N)
    base_vel = (torch.rand(N, 2, generator=g) - 0.5) * 1.2
    phi = torch.linspace(-math.pi, math.pi, N)
    env.command_manager.cmd[:, 0:2] = base_cmd
    env.asset.data.root_lin_vel_b[:, :2] = base_vel
    ref = penalty_cross_command_vel_l2(env, "base_velocity", ROBOT)
    env.command_manager.cmd[:, 0:2] = _rot(base_cmd, phi)
    env.asset.data.root_lin_vel_b[:, :2] = _rot(base_vel, phi)
    rotated = penalty_cross_command_vel_l2(env, "base_velocity", ROBOT)
    assert torch.allclose(rotated, ref, atol=1e-6)
    # explicit: a +y command prices the x velocity
    env.command_manager.cmd[:, 0:2] = torch.tensor([[0.0, 0.25]] * N)
    env.asset.data.root_lin_vel_b[:, :2] = torch.tensor([[0.2, 0.25]] * N)
    got = penalty_cross_command_vel_l2(env, "base_velocity", ROBOT)
    assert torch.allclose(got, torch.full((N,), 0.04), atol=1e-7)


# ------------------------------------------------------------------- reward_facing_exp
def test_facing_reward_wraps():
    env = _Env()
    env.asset.data.heading_w = torch.zeros(N)
    env.command_manager.term.heading_target = torch.tensor(
        [0.0, 0.1, -0.1, 2 * math.pi - 0.1, -2 * math.pi + 0.1, math.pi, -math.pi, 4 * math.pi]
    )
    got = reward_facing_exp(env, ROBOT, "base_parkour")
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(math.exp(-0.1), abs=1e-6)
    assert got[2] == pytest.approx(got[1].item(), abs=1e-7)
    assert got[3] == pytest.approx(got[1].item(), abs=1e-6)     # 2pi - 0.1  ==  -0.1
    assert got[4] == pytest.approx(got[1].item(), abs=1e-6)
    assert got[5] == pytest.approx(math.exp(-math.pi), abs=1e-6)
    assert got[6] == pytest.approx(math.exp(-math.pi), abs=1e-6)
    assert got[7] == pytest.approx(1.0, abs=1e-5)               # 4pi == 0


def test_facing_reward_has_no_command_gate_and_no_tile_mask():
    env = _Env()
    env.command_manager.cmd[:] = 0.0                            # standing
    env.flat_mask = torch.ones(N)                               # every env on a flat tile
    env.asset.data.heading_w = torch.full((N,), 0.5)
    env.command_manager.term.heading_target = torch.full((N,), 0.5)
    assert torch.allclose(reward_facing_exp(env, ROBOT, "base_parkour"), torch.ones(N))


def test_facing_reward_is_monotone_in_the_error():
    env = _Env(5)
    env.asset.data.heading_w = torch.zeros(5)
    env.command_manager.term.heading_target = torch.tensor([0.0, 0.3, 0.8, 1.5, 3.0])
    got = reward_facing_exp(env, ROBOT, "base_parkour")
    assert torch.all(got[1:] < got[:-1])
    assert float(got.max()) <= 1.0 and float(got.min()) > 0.0


# ----------------------------------------------------------------- reward_yaw_progress
def _yaw_env(errors: torch.Tensor, wz: torch.Tensor, *, yaw=0.0) -> "_Env":
    """An env whose yaw-rate command is produced by the REAL P-control from ``errors``.

    ``errors`` are signed facing errors ``wrap(heading_target - heading_w)``; the command is
    ``heading_rate_command`` at the omni presets' stiffness / cap / dead band, so the whole
    chain under test (error sign -> command sign -> rewarded rotation sign) is the shipped
    one and not a restatement.
    """
    n = errors.numel()
    env = _Env(n)
    env.asset.data.heading_w = torch.full((n,), float(yaw))
    env.command_manager.term.heading_target = errors + float(yaw)
    env.command_manager.cmd[:, 2] = heading_rate_command(
        env.command_manager.term.heading_target,
        env.asset.data.heading_w,
        STIFFNESS,
        ANG_VEL_MAX,
        ANG_VEL_CLIP,
    )
    env.asset.data.root_ang_vel_b[:, 2] = wz
    return env


def test_yaw_progress_sign_chain_follows_the_command_math():
    """positive facing error -> positive command -> positive body wz is PROGRESS (and mirrored).

    The one test that can catch a sign inversion. ``errors`` straddle the dead band on both
    sides; ``wz`` is the same magnitude everywhere, so only the SIGNS decide the answer.
    """
    errors = torch.tensor([1.0, 0.4, 0.2, -0.2, -0.4, -1.0])
    wz = torch.full((6,), 0.1)
    env = _yaw_env(errors, wz)
    cmd_wz = env.command_manager.cmd[:, 2]
    # the command math's own contract, restated so a change there is visible here
    assert torch.equal(torch.sign(cmd_wz), torch.sign(errors)), cmd_wz
    assert bool((cmd_wz.abs() <= ANG_VEL_MAX + 1e-9).all())

    got = reward_yaw_progress(env, "base_velocity", ROBOT)
    # turning LEFT (err > 0) with a left (positive) body rate is paid; the same rate against
    # a right turn is charged the same amount
    assert torch.allclose(got, torch.sign(errors) * 0.1, atol=1e-7), got
    assert bool((got[:3] > 0).all()) and bool((got[3:] < 0).all())

    # mirror the achieved rotation: every sign flips, nothing else
    env_neg = _yaw_env(errors, -wz)
    assert torch.allclose(reward_yaw_progress(env_neg, "base_velocity", ROBOT), -got, atol=1e-7)

    # and the reward is measured in rad/s: it IS the yaw rate when the turn is commanded left
    env_left = _yaw_env(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.05, 0.1, 0.25]))
    assert torch.allclose(
        reward_yaw_progress(env_left, "base_velocity", ROBOT),
        torch.tensor([0.05, 0.1, 0.25]),
        atol=1e-7,
    )


def test_yaw_progress_is_the_signed_drift_not_the_rectified_wobble():
    """WHY the term is signed rather than ``clamp(min=0)`` like the linear progress twin.

    The failure it exists to fix is a NET rotation of the wrong sign hidden inside gait
    wobble an order of magnitude larger (smoke: median -0.025 rad/s against a saturated +0.30
    command, mean |wz| 0.28). A half-cycle of symmetric wobble must therefore net ~0 -- a
    non-negative version would pay its positive halves and ignore its negative ones, earning
    income for wobble alone and leaving no gradient toward net rotation.
    """
    wobble = torch.tensor([0.28, -0.28, 0.20, -0.20, 0.05, -0.05])
    env = _yaw_env(torch.full((6,), 1.0), wobble)
    got = reward_yaw_progress(env, "base_velocity", ROBOT)
    assert float(got.sum()) == pytest.approx(0.0, abs=1e-7)
    assert float(torch.clamp(got, min=0.0).sum()) == pytest.approx(0.53, abs=1e-6)
    # a small net drift the wrong way is negative income even under large wobble (kept inside
    # the +-0.3 clip so the mean is the drift itself and not a clipping artefact)
    drifting = torch.tensor([0.20, -0.20, 0.15, -0.15, 0.05, -0.05]) - 0.025
    env2 = _yaw_env(torch.full((6,), 1.0), drifting)
    assert float(reward_yaw_progress(env2, "base_velocity", ROBOT).mean()) == pytest.approx(
        -0.025, abs=1e-6
    )


def test_yaw_progress_clips_symmetrically_at_the_command_cap():
    """Spinning faster than commanded earns nothing extra (and a spin-out costs a bounded max)."""
    wz = torch.tensor([0.29, 0.3, 0.3000001, 1.0, 20.0, -20.0])
    env = _yaw_env(torch.full((6,), 1.0), wz)
    got = reward_yaw_progress(env, "base_velocity", ROBOT)
    assert torch.allclose(
        got, torch.tensor([0.29, 0.3, 0.3, 0.3, 0.3, -0.3]), atol=1e-7
    ), got
    assert float(got.abs().max()) <= ANG_VEL_MAX + 1e-6      # float32 round-trip of 0.3
    # the cap comes from the COMMAND TERM's cfg, so retuning KRABBY_OMNI_ANG_VEL_MAX moves it
    env.command_manager.term.cfg.ang_vel_max = 0.2
    assert float(reward_yaw_progress(env, "base_velocity", ROBOT).max()) == pytest.approx(0.2)
    # an explicit ``clip`` overrides it (the cfg's params pass ``clip: None`` = "use the cap")
    assert float(
        reward_yaw_progress(env, "base_velocity", ROBOT, clip=0.5).max()
    ) == pytest.approx(0.5)


def test_yaw_progress_is_zero_when_no_turn_is_commanded():
    """The gate IS the P-control's dead band: no second threshold, no push against it."""
    # errors inside the dead band (|err| < 0.1333 rad) produce cmd_wz == 0 exactly
    assert DEAD_BAND_RAD == pytest.approx(0.13333, abs=1e-5)
    errors = torch.tensor([0.0, 0.05, -0.05, 0.13, -0.13, 0.1333])
    env = _yaw_env(errors, torch.full((6,), 0.28))     # full-amplitude wobble, still zero
    assert torch.equal(env.command_manager.cmd[:, 2], torch.zeros(6))
    got = reward_yaw_progress(env, "base_velocity", ROBOT)
    assert torch.equal(got, torch.zeros(6)), got
    assert not bool(torch.signbit(got).any())         # exactly +0.0, never -0.0
    # a pure TRANSLATION command pays nothing: only slot 2 counts (stages 1a/1b draw no
    # turning slot at all, so this is every commanded row they have)
    env2 = _Env(4)
    env2.command_manager.cmd[:, 0] = 0.35
    env2.command_manager.cmd[:, 1] = -0.25
    env2.asset.data.root_ang_vel_b[:, 2] = torch.tensor([0.3, -0.3, 0.1, 0.0])
    assert torch.equal(reward_yaw_progress(env2, "base_velocity", ROBOT), torch.zeros(4))


def test_yaw_progress_refuses_a_command_term_without_the_cap():
    """A leaked forward/sideways command term must raise, not fall back to a silent default."""
    env = _Env(2)
    env.command_manager.cmd[:, 2] = 0.3
    env.asset.data.root_ang_vel_b[:, 2] = 0.1
    del env.command_manager.term.cfg.ang_vel_max
    with pytest.raises(RuntimeError, match="ang_vel_max"):
        reward_yaw_progress(env, "base_velocity", ROBOT)
    # ... but an explicit clip is self-sufficient (the eval/replay path)
    assert float(reward_yaw_progress(env, "base_velocity", ROBOT, clip=0.3).max()) > 0.0
    env.command_manager.term.cfg.ang_vel_max = 0.3
    with pytest.raises(ValueError, match="clip must be positive"):
        reward_yaw_progress(env, "base_velocity", ROBOT, clip=0.0)


def test_yaw_progress_mirrors_the_linear_progress_terms_shape():
    """Structural parity with ``reward_forward_progress_along_command``: same four steps.

    Read the shipped source rather than trusting the docstring: the rotational twin must
    (1) read the command, (2) project the achieved rate on the commanded direction, (3) clip,
    (4) multiply by a command gate -- and must NOT reintroduce the linear term's ``min=0.0``
    rectification, which is the one factor it deliberately drops.
    """
    src, tree = _nodes(OMNI_REWARDS_PY)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "reward_yaw_progress"][0]
    body = "\n".join(
        ast.unparse(n) for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    assert "root_ang_vel_b[:, 2]" in body, body
    assert "torch.sign(cmd_wz)" in body
    assert "torch.clamp(progress, min=-cap, max=cap)" in body
    assert "cmd_wz != 0.0" in body
    assert "min=0.0" not in body, "the rotational twin must stay SIGNED"
    # the linear twin it mirrors is still the shape this one was derived from
    fwd_src, fwd_tree = _nodes(LIB_REWARDS_PY)
    fwd = [n for n in fwd_tree.body if isinstance(n, ast.FunctionDef)
           and n.name == "reward_forward_progress_along_command"][0]
    fwd_body = "\n".join(ast.unparse(n) for n in fwd.body)
    assert "torch.clamp(progress, min=0.0, max=max_speed_scale)" in fwd_body
    assert "active.float()" in fwd_body


# ------------------------- the A2 CANDIDATES (default OFF): reward_facing_exp_turn_slot
SLOT_STAND, SLOT_MOVE = omni_cmd_math.SLOT_STAND, omni_cmd_math.SLOT_MOVE
SLOT_TURN, SLOT_MOVE_TURN = omni_cmd_math.SLOT_TURN, omni_cmd_math.SLOT_MOVE_TURN
#: one row per slot, twice -- the gate must select exactly the two turning ones
ALL_SLOTS = torch.tensor([SLOT_STAND, SLOT_MOVE, SLOT_TURN, SLOT_MOVE_TURN,
                          SLOT_MOVE_TURN, SLOT_TURN, SLOT_MOVE, SLOT_STAND])
TURN_SLOT_ROWS = torch.tensor([False, False, True, True, True, True, False, False])


def _slot_env(errors: torch.Tensor, slots: torch.Tensor) -> "_Env":
    """An env with a per-row facing error and a per-row command slot."""
    n = errors.numel()
    env = _Env(n)
    env.asset.data.heading_w = torch.zeros(n)
    env.command_manager.term.heading_target = errors.clone()
    env.command_manager.term.slot_id = slots.clone().long()
    return env


def test_facing_turn_slot_is_the_facing_term_on_the_turn_slots_and_zero_elsewhere():
    """The whole contract: the SAME number as ``reward_facing_exp``, masked by the slot.

    The two turning slots are ``SLOT_TURN`` (2) and ``SLOT_MOVE_TURN`` (3) -- the rows whose
    drawn command asks for a turn; stand (0) and move (1) pay exactly nothing.
    """
    errors = torch.tensor([0.0, 0.3, 0.0, 0.3, -1.2, 2.0, -0.4, 3.0])
    env = _slot_env(errors, ALL_SLOTS)
    ungated = reward_facing_exp(env, ROBOT, "base_parkour")
    got = reward_facing_exp_turn_slot(env, ROBOT, "base_parkour")
    assert torch.equal(got[TURN_SLOT_ROWS], ungated[TURN_SLOT_ROWS])
    assert torch.equal(got[~TURN_SLOT_ROWS], torch.zeros(int((~TURN_SLOT_ROWS).sum())))
    assert not bool(torch.signbit(got).any()), "0.0, never -0.0"
    # the ungated term is genuinely non-trivial on the masked rows (not a vacuous equality)
    assert float(ungated.max()) > 0.9 and float(ungated.min()) < 0.2
    # and the mask really is the slot: move the SAME errors onto non-turning slots -> all zero
    env_flat = _slot_env(errors, torch.full_like(ALL_SLOTS, SLOT_MOVE))
    assert torch.equal(
        reward_facing_exp_turn_slot(env_flat, ROBOT, "base_parkour"), torch.zeros(8)
    )


def test_facing_turn_slot_gate_is_the_slot_not_the_yaw_dead_band():
    """Deliberately NOT ``cmd_wz != 0``: a settled turn must keep paying.

    ``reward_yaw_progress``'s gate closes as soon as the facing error is inside the P-control
    dead band -- i.e. the moment the turn is achieved. A facing-POSITION term gated that way
    would stop paying exactly when it is satisfied, which is an incentive to overshoot. This
    one is gated on the drawn slot, so an on-target turn row still collects ~1.0.
    """
    settled = torch.zeros(4)                       # error 0 -> the command's dead band
    env = _slot_env(settled, torch.tensor([SLOT_TURN, SLOT_MOVE_TURN, SLOT_MOVE, SLOT_STAND]))
    env.command_manager.cmd[:, 2] = heading_rate_command(
        env.command_manager.term.heading_target,
        env.asset.data.heading_w,
        STIFFNESS,
        ANG_VEL_MAX,
        ANG_VEL_CLIP,
    )
    assert torch.equal(env.command_manager.cmd[:, 2], torch.zeros(4)), "the dead band is shut"
    assert torch.equal(reward_yaw_progress(env, "base_velocity", ROBOT), torch.zeros(4))
    got = reward_facing_exp_turn_slot(env, ROBOT, "base_parkour")
    assert torch.allclose(got[:2], torch.ones(2), atol=1e-7), "a settled turn is still paid"
    assert torch.equal(got[2:], torch.zeros(2))


def test_facing_turn_slot_refuses_a_command_term_without_a_slot():
    """A leaked forward/sideways command term must raise, not pay on no row (or every row)."""
    env = _slot_env(torch.zeros(3), torch.full((3,), SLOT_TURN))
    del env.command_manager.term.slot_id
    with pytest.raises(RuntimeError, match="slot_id"):
        reward_facing_exp_turn_slot(env, ROBOT, "base_parkour")
    with pytest.raises(RuntimeError, match="slot_id"):
        _OMNI_NS["_slot_is_turning"](env, "base_velocity")


def test_facing_turn_slot_imports_the_slot_constant_rather_than_retyping_it():
    """Source-level: the turn bit comes from the command math module, not from a literal 2."""
    src, tree = _nodes(OMNI_REWARDS_PY)
    imports = [
        n for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module and n.module.endswith(
            "crab_hex_omni_command_math"
        )
    ]
    assert any(a.name == "SLOT_TURN" for imp in imports for a in imp.names), \
        "SLOT_TURN must be imported from crab_hex_omni_command_math"
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "_slot_is_turning"][0]
    body = "\n".join(
        ast.unparse(n) for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    assert "SLOT_TURN" in body and ">= 2" not in body
    # the gate really is "the turn bit", i.e. slots 2 and 3 and no others
    assert (omni_cmd_math.SLOT_TURN, omni_cmd_math.SLOT_MOVE_TURN) == (2, 3)
    assert omni_cmd_math.SLOT_NAMES == ("stand", "move", "turn", "move_turn")


# --------------------- the A2 CANDIDATES (default OFF): the facing-coupled tracking twin
def _coupled_env(errors: torch.Tensor, cmd_xy, vel_b):
    n = errors.numel()
    env = _Env(n)
    env.asset.data.heading_w = torch.zeros(n)
    env.command_manager.term.heading_target = errors.clone()
    env.command_manager.cmd[:, 0:2] = torch.tensor(cmd_xy).expand(n, 2).clone()
    env.asset.data.root_lin_vel_b[:, :2] = torch.tensor(vel_b).expand(n, 2).clone()
    return env


@requires_isaaclab_source
def test_facing_coupled_tracking_is_the_stock_kernel_times_the_facing_gaussian():
    """Value for value: ``stock x exp(-(err/sigma)^2)``, and BITWISE stock at zero error."""
    errors = torch.tensor([0.0, 0.2, -0.2, 0.5, -1.0, 2.0])
    env = _coupled_env(errors, (0.3, 0.0), (0.25, 0.05))
    sigma, std = 0.5, math.sqrt(0.04)
    stock = track_lin_vel_xy_exp(env, std=std, command_name="base_velocity", asset_cfg=ROBOT)
    got = track_lin_vel_xy_exp_facing_coupled(
        env, std=std, command_name="base_velocity", asset_cfg=ROBOT, facing_sigma=sigma
    )
    err = wrap_to_pi(env.command_manager.term.heading_target - env.asset.data.heading_w)
    assert torch.allclose(got, stock * torch.exp(-((err / sigma) ** 2)), atol=1e-7)
    # a zero facing error costs NOTHING: the factor is exactly 1.0 there
    assert torch.equal(got[0], stock[0])
    assert float(stock[0]) > 0.0
    # ... and every other row is strictly cheaper, monotonically in |err|
    assert bool((got[1:] < stock[1:]).all())
    assert got[1] == pytest.approx(float(got[2]), abs=1e-7), "even in the error"
    assert float(got[3]) > float(got[4]) > float(got[5])
    # the documented landmarks: exp(-1) at one sigma, exp(-4) at two
    at_sigma = _coupled_env(torch.tensor([sigma, 2 * sigma]), (0.3, 0.0), (0.3, 0.0))
    s2 = track_lin_vel_xy_exp(at_sigma, std=std, command_name="base_velocity", asset_cfg=ROBOT)
    c2 = track_lin_vel_xy_exp_facing_coupled(
        at_sigma, std=std, command_name="base_velocity", asset_cfg=ROBOT, facing_sigma=sigma
    )
    assert float(c2[0] / s2[0]) == pytest.approx(math.exp(-1.0), abs=1e-6)
    assert float(c2[1] / s2[1]) == pytest.approx(math.exp(-4.0), abs=1e-6)


@requires_isaaclab_source
def test_facing_coupled_tracking_passes_std_through_and_wraps_the_error():
    """``std`` is the window's (``KRABBY_TRACK_SIGMA2``), and the facing error WRAPS."""
    env = _coupled_env(torch.zeros(4), (0.3, 0.0), (0.1, 0.0))
    kw = dict(command_name="base_velocity", asset_cfg=ROBOT, facing_sigma=1.0)
    tight = track_lin_vel_xy_exp_facing_coupled(env, std=0.2, **kw)
    wide = track_lin_vel_xy_exp_facing_coupled(env, std=0.5, **kw)
    assert bool((tight < wide).all()), "a tighter translation kernel pays less off-target"
    assert torch.allclose(
        wide, track_lin_vel_xy_exp(env, std=0.5, command_name="base_velocity", asset_cfg=ROBOT),
        atol=1e-7,
    )
    # an error of 2 pi - 0.1 is an error of -0.1 (and pays the same as +0.1)
    wrapped = _coupled_env(
        torch.tensor([0.1, -0.1, 2 * math.pi - 0.1, -2 * math.pi + 0.1, 4 * math.pi]),
        (0.3, 0.0), (0.3, 0.0),
    )
    got = track_lin_vel_xy_exp_facing_coupled(wrapped, std=0.2, **kw)
    assert torch.allclose(got[:4], got[0].expand(4), atol=1e-5)
    stock = track_lin_vel_xy_exp(wrapped, std=0.2, command_name="base_velocity", asset_cfg=ROBOT)
    assert float(got[4]) == pytest.approx(float(stock[4]), abs=1e-5), "4 pi is no error at all"


def test_facing_coupled_tracking_refuses_a_degenerate_sigma():
    env = _coupled_env(torch.zeros(2), (0.3, 0.0), (0.3, 0.0))
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError, match="facing_sigma"):
            track_lin_vel_xy_exp_facing_coupled(
                env, std=0.2, command_name="base_velocity", asset_cfg=ROBOT, facing_sigma=bad
            )


def test_facing_coupled_tracking_calls_the_stock_term_and_keeps_its_signature():
    """Source-level: the stock kernel is CALLED, and ``facing_sigma`` is a REQUIRED keyword.

    The knob swaps the func and leaves the stock ``params`` in place, so the signature must be
    the stock one plus exactly one keyword-only argument with NO default -- a default would
    let a cfg that lost the param train under a silently chosen coupling width instead of
    failing in the reward manager's static check at env build.
    """
    src, tree = _nodes(OMNI_REWARDS_PY)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "track_lin_vel_xy_exp_facing_coupled"][0]
    body = "\n".join(
        ast.unparse(n) for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    assert "track_lin_vel_xy_exp(env, std=std" in body, body
    assert "torch.exp" in body and "lin_vel_error" not in body, "the kernel must not be re-derived"
    assert [a.arg for a in fn.args.args] == ["env", "std", "command_name", "asset_cfg"]
    assert [a.arg for a in fn.args.kwonlyargs] == ["facing_sigma"]
    assert fn.args.kw_defaults == [None], "facing_sigma must stay REQUIRED"
    # the module imports the stock symbol from isaaclab, like the ang-vel twin does
    imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
               and n.module == "isaaclab.envs.mdp.rewards"]
    assert any(a.name == "track_lin_vel_xy_exp" for imp in imports for a in imp.names)
    # ... and the isaaclab argument order is the one we mirrored
    if ISAACLAB_REWARDS_PY is not None:
        _, stock_tree = _nodes(ISAACLAB_REWARDS_PY)
        stock_fn = [n for n in stock_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "track_lin_vel_xy_exp"][0]
        assert [a.arg for a in stock_fn.args.args] == ["env", "std", "command_name", "asset_cfg"]


def test_facing_coupled_tracking_satisfies_the_managers_static_argument_check():
    """Isaac's ``ManagerBase._resolve_common_term_cfg`` check, replayed on the swapped term.

    The knob leaves the stock params and adds ``facing_sigma``; the manager compares
    ``set(args[1:])`` against ``set(term_params + args_with_defaults)`` after moving the
    defaulted arguments to the back. A keyword-only required argument has to survive that, or
    the armed run dies at env build instead of training.
    """
    import inspect

    params = {"command_name": "base_velocity", "std": math.sqrt(0.04), "facing_sigma": 0.5}
    args = inspect.signature(track_lin_vel_xy_exp_facing_coupled).parameters
    with_defaults = [a for a in args if args[a].default is not inspect.Parameter.empty]
    without = [a for a in args if args[a].default is inspect.Parameter.empty]
    ordered = without + with_defaults
    assert ordered[0] == "env"
    assert set(ordered[1:]) == set(list(params) + with_defaults)
    # and the call the manager would make actually works
    env = _coupled_env(torch.zeros(2), (0.3, 0.0), (0.3, 0.0))
    if track_lin_vel_xy_exp is not None:
        out = track_lin_vel_xy_exp_facing_coupled(env, asset_cfg=ROBOT, **params)
        assert out.shape == (2,)


# ------------------------------- the STAGED LEVER: track_ang_vel_z_exp_turn_gated (OFF)
def _mixed_turn_env(n=N):
    """Half the rows command a turn, half command none; every row has a non-zero yaw rate.

    The achieved rates deliberately include the exactly-zero-error row (row 0: cmd == wz) and
    rows on both sides of their command, so "bitwise on the turning rows" is exercised at the
    kernel's peak AND on its flanks rather than only where the exponential is flat.
    """
    env = _Env(n)
    env.command_manager.cmd[:, 2] = torch.tensor([0.30, 0.0, -0.30, 0.0, 0.12, 0.0, -0.08, 0.0])
    env.asset.data.root_ang_vel_b[:, 2] = torch.tensor(
        [0.30, 0.13, -0.11, -0.25, 0.02, 0.41, 0.09, 0.0]
    )
    return env


TURNING_ROWS = torch.tensor([True, False, True, False, True, False, True, False])


@requires_isaaclab_source
@pytest.mark.parametrize("std", [0.15, 0.5, math.sqrt(0.25)])
def test_turn_gated_ang_vel_is_bitwise_the_stock_term_on_turning_rows(std):
    """On ``cmd_wz != 0`` the twin must be the SHIPPED ``track_ang_vel_z_exp``, bit for bit.

    Not "close to": the lever's entire contract is that it multiplies the stock value by a
    0/1 mask, so a row where the command asks for a turn must be indistinguishable from the
    ungated term. Anything else means the kernel was re-derived here and can drift from
    upstream -- exactly the failure the shipped function's ``track_ang_vel_z_exp(...)`` call
    exists to make impossible.
    """
    env = _mixed_turn_env()
    stock = track_ang_vel_z_exp(env, std=std, command_name="base_velocity", asset_cfg=ROBOT)
    gated = track_ang_vel_z_exp_turn_gated(env, std=std, command_name="base_velocity",
                                           asset_cfg=ROBOT)
    assert torch.equal(gated[TURNING_ROWS], stock[TURNING_ROWS])
    # the stock term is genuinely non-trivial on those rows (a kernel that paid 0 everywhere
    # would satisfy the equality above vacuously)
    assert float(stock[TURNING_ROWS].max()) > 0.9
    assert float(stock[TURNING_ROWS].min()) < 0.9


@requires_isaaclab_source
def test_turn_gated_ang_vel_is_exactly_zero_where_no_turn_is_commanded():
    """``cmd_wz == 0`` -> 0.0 exactly, however small the yaw error -- the whole lever.

    The stock term pays a MOTIONLESS robot ~0.98/s on those rows (zero command, zero rate ->
    ``exp(0)``), which on the ~88 % of 1c steps that command no turn is ~45 % of a stander's
    positive income. If this ever stops being exactly zero the lever is inert and the offline
    margin it buys (+16.6 % -> +37.5 % on the fwd hold) is not what the run gets.
    """
    env = _mixed_turn_env()
    gated = track_ang_vel_z_exp_turn_gated(env, std=0.5, command_name="base_velocity",
                                           asset_cfg=ROBOT)
    non_turning = ~TURNING_ROWS
    assert torch.equal(gated[non_turning], torch.zeros(int(non_turning.sum())))
    assert not torch.signbit(gated[non_turning]).any(), "0.0, never -0.0"

    # the stander: no command, no motion -> the stock term pays ~1.0 and the gated one 0.0
    still = _Env()
    stock = track_ang_vel_z_exp(still, std=0.5, command_name="base_velocity", asset_cfg=ROBOT)
    assert torch.allclose(stock, torch.ones(N))
    assert torch.equal(
        track_ang_vel_z_exp_turn_gated(still, std=0.5, command_name="base_velocity",
                                       asset_cfg=ROBOT),
        torch.zeros(N),
    )


@requires_isaaclab_source
def test_turn_gated_ang_vel_gate_is_the_commands_own_dead_band():
    """The gate is ``cmd_wz != 0``, i.e. the P-control's dead band -- no second threshold.

    Pinned against the REAL ``heading_rate_command``, the same anchor ``reward_yaw_progress``
    uses: a facing error inside the dead band produces ``cmd_wz == 0`` and must be unpaid; one
    just outside it produces a non-zero command and must be paid the stock value. The two
    gated terms therefore switch on together, which is what makes "a turn is commanded" a
    single concept in this task.
    """
    env = _Env()
    inside, outside = DEAD_BAND_RAD * 0.5, DEAD_BAND_RAD * 1.5
    herr = torch.tensor([inside, -inside, outside, -outside, 0.0, inside, outside, -outside])
    # heading_target = the error, yaw = 0 -> wrap(target - yaw) IS the error
    env.command_manager.cmd[:, 2] = heading_rate_command(
        herr, torch.zeros(N), STIFFNESS, ANG_VEL_MAX, ANG_VEL_CLIP
    )
    env.asset.data.root_ang_vel_b[:, 2] = 0.05
    gated = track_ang_vel_z_exp_turn_gated(env, std=0.5, command_name="base_velocity",
                                           asset_cfg=ROBOT)
    stock = track_ang_vel_z_exp(env, std=0.5, command_name="base_velocity", asset_cfg=ROBOT)
    dead = env.command_manager.cmd[:, 2] == 0.0
    assert dead.tolist() == [True, True, False, False, True, True, False, False]
    assert torch.equal(gated[dead], torch.zeros(int(dead.sum())))
    assert torch.equal(gated[~dead], stock[~dead])
    # the same rows are gated on the ADDED rotational term
    yaw_prog = reward_yaw_progress(env, "base_velocity", ROBOT)
    assert torch.equal(yaw_prog[dead], torch.zeros(int(dead.sum())))
    assert (yaw_prog[~dead] != 0.0).all()


@requires_isaaclab_source
def test_turn_gated_ang_vel_passes_std_through_rather_than_owning_a_kernel():
    """``std`` reaches the stock function untouched: the lever changes the GATE, not the width.

    ``KRABBY_OMNI_ANG_VEL_STD`` writes ``params["std"]`` and the knob applier runs BEFORE the
    func swap, so the armed run must price the kernel that knob chose. A twin that captured a
    default would silently re-acquire plan D4's 0.15 -- the width Revision 4 reverted.
    """
    env = _mixed_turn_env()
    wide = track_ang_vel_z_exp_turn_gated(env, std=0.5, command_name="base_velocity",
                                          asset_cfg=ROBOT)
    tight = track_ang_vel_z_exp_turn_gated(env, std=0.15, command_name="base_velocity",
                                           asset_cfg=ROBOT)
    off = TURNING_ROWS & (env.command_manager.cmd[:, 2] != env.asset.data.root_ang_vel_b[:, 2])
    assert off.any()
    assert (tight[off] < wide[off]).all(), "a tighter kernel must pay less off-target"


def test_turn_gated_ang_vel_calls_the_stock_term_and_does_not_re_derive_it():
    """Source-level: the twin must CALL ``track_ang_vel_z_exp`` and own only the mask.

    The bitwise tests above prove today's equality; this one prevents tomorrow's copy-paste.
    A re-derived ``torch.exp(-... / std ** 2)`` would pass them the day it is written and
    diverge the day upstream changes the kernel.
    """
    src, tree = _nodes(OMNI_REWARDS_PY)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
          and n.name == "track_ang_vel_z_exp_turn_gated"][0]
    body = "\n".join(
        ast.unparse(n) for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    )
    assert "track_ang_vel_z_exp(env, std=std" in body, body
    assert "torch.exp" not in body, "the kernel must not be re-derived here"
    assert "[:, 2] != 0.0" in body
    # the same signature as the stock term, so the func swap needs no params change
    assert [a.arg for a in fn.args.args] == ["env", "std", "command_name", "asset_cfg"]
    assert not fn.args.kwonlyargs
    assert len(fn.args.defaults) == 2, "std must stay REQUIRED (no silent kernel)"
    # ... which is exactly isaaclab's own argument order
    if ISAACLAB_REWARDS_PY is not None:
        _, stock_tree = _nodes(ISAACLAB_REWARDS_PY)
        stock_fn = [n for n in stock_tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "track_ang_vel_z_exp"][0]
        assert [a.arg for a in fn.args.args] == [a.arg for a in stock_fn.args.args]


# ----------------------------------------------------- reward_tracking_command_vel(_on_*)
def test_command_vel_term_divides_by_the_command_norm():
    """Achieving exactly the commanded velocity pays 1.0, in every direction."""
    env = _Env()
    speeds = torch.tensor([0.21, 0.25, 0.30, 0.35, 0.21, 0.25, 0.30, 0.35])
    phi = torch.linspace(-math.pi, math.pi, N)
    yaw = torch.linspace(-2.0, 2.0, N)
    env.asset.data.heading_w = yaw
    env.command_manager.cmd[:, 0:2] = torch.stack(
        (speeds * torch.cos(phi), speeds * torch.sin(phi)), dim=-1
    )
    # world velocity = the commanded body velocity rotated by the yaw
    env.asset.data.root_vel_w[:, :2] = _rot(env.command_manager.cmd[:, 0:2], yaw)
    got = reward_tracking_command_vel(env, ROBOT, "base_velocity")
    assert torch.allclose(got, torch.ones(N), atol=1e-6)
    # half the commanded speed pays half, whatever the commanded speed is
    env.asset.data.root_vel_w[:, :2] *= 0.5
    assert torch.allclose(reward_tracking_command_vel(env, ROBOT, "base_velocity"),
                          torch.full((N,), 0.5), atol=1e-6)
    # over-achieving is capped at the command (``minimum(proj, speed)``)
    env.asset.data.root_vel_w[:, :2] *= 6.0
    assert torch.allclose(reward_tracking_command_vel(env, ROBOT, "base_velocity"),
                          torch.ones(N), atol=1e-6)


def test_command_vel_term_matches_the_forward_term_on_a_forward_command():
    """Same formula: put the forward term's goal direction on the commanded direction."""
    g = torch.Generator().manual_seed(5)
    env = _Env()
    env.asset.data.heading_w = torch.zeros(N)                    # body frame == world frame
    env.command_manager.cmd[:, 0] = torch.tensor([0.02, 0.05, 0.1, 0.21, 0.25, 0.3, 0.35, 0.5])
    env.asset.data.root_vel_w[:, :2] = (torch.rand(N, 2, generator=g) - 0.3)
    env.parkour_manager.event.target_pos_rel = torch.tensor([[1.0, 0.0]] * N) * 3.7
    omni = reward_tracking_command_vel(env, ROBOT, "base_velocity")
    fwd = reward_tracking_goal_vel(env, "base_parkour", ROBOT)
    assert torch.allclose(omni, fwd, atol=1e-5)


def test_command_vel_term_is_zero_at_zero_command_and_guarded_below_v_min():
    env = _Env()
    env.asset.data.root_vel_w[:, :2] = torch.tensor([[0.3, 0.2]] * N)
    env.command_manager.cmd[:, 0:2] = 0.0
    assert torch.equal(reward_tracking_command_vel(env, ROBOT, "base_velocity"), torch.zeros(N))
    # a tiny command divides by v_min, not by itself: bounded, never a blow-up
    env.command_manager.cmd[:, 0] = 0.001
    got = reward_tracking_command_vel(env, ROBOT, "base_velocity")
    assert torch.allclose(got, torch.full((N,), 0.001 / _OMNI_NS["_TRACK_V_MIN"]), atol=1e-6)
    assert torch.isfinite(got).all()
    assert _OMNI_NS["_TRACK_V_MIN"] == 0.05


def test_command_vel_term_is_masked_to_obstacle_tiles():
    env = _Env()
    env.asset.data.heading_w = torch.zeros(N)
    env.command_manager.cmd[:, 0] = 0.25
    env.asset.data.root_vel_w[:, 0] = 0.25
    env.flat_mask = torch.tensor([1.0, 0.0] * (N // 2))
    got = reward_tracking_command_vel_on_obstacle(env, "base_parkour", ROBOT, "base_velocity")
    assert torch.allclose(got, torch.tensor([0.0, 1.0] * (N // 2)), atol=1e-6)


# ------------------------------------------------------ reward_obstacle_clearance_omni
def _clearance_env(cmd_xy, vel_b):
    env = _Env(1)
    env.command_manager.cmd[:, 0:2] = torch.tensor([cmd_xy])
    env.asset.data.root_lin_vel_b[:, :2] = torch.tensor([vel_b])
    env.asset.data.root_vel_w[:, :2] = torch.tensor([vel_b])     # heading_w = 0
    return env


@pytest.mark.parametrize(
    "label,cmd_xy,vel_b,expected",
    [
        ("forward", (0.3, 0.0), (0.3, 0.0), 1.0),
        ("lateral", (0.0, 0.25), (0.0, 0.25), 1.0),     # the forward gate would kill this
        ("backward", (-0.25, 0.0), (-0.25, 0.0), 1.0),  # ... and this
        ("diagonal", (0.2, 0.2), (0.2, 0.2), 1.0),
        ("standing", (0.0, 0.0), (0.0, 0.0), 0.0),
        ("wrong way", (0.3, 0.0), (-0.3, 0.0), 0.0),
        ("too slow", (0.3, 0.0), (0.05, 0.0), 0.0),
    ],
)
def test_obstacle_clearance_reads_speed_along_the_command(label, cmd_xy, vel_b, expected):
    env = _clearance_env(cmd_xy, vel_b)
    got = reward_obstacle_clearance_omni(
        env, "base_parkour", FOOTPADS, ROBOT, "base_velocity",
        min_goal_progress=0.05, min_forward_speed=0.10, min_forward_speed_cmd=0.12,
    )
    assert float(got[0]) == pytest.approx(expected), label


def test_obstacle_clearance_keeps_the_other_three_gates():
    env = _clearance_env((0.3, 0.0), (0.3, 0.0))
    kw = dict(min_goal_progress=0.05, min_forward_speed=0.10, min_forward_speed_cmd=0.12)
    assert float(reward_obstacle_clearance_omni(env, "base_parkour", FOOTPADS, ROBOT,
                                                "base_velocity", **kw)[0]) == 1.0
    env.flat_mask = torch.ones(1)                 # flat tile -> no obstacle bonus
    assert float(reward_obstacle_clearance_omni(env, "base_parkour", FOOTPADS, ROBOT,
                                                "base_velocity", **kw)[0]) == 0.0
    env.flat_mask = torch.zeros(1)
    env.stumble = torch.ones(1)                   # stumbling -> no bonus
    assert float(reward_obstacle_clearance_omni(env, "base_parkour", FOOTPADS, ROBOT,
                                                "base_velocity", **kw)[0]) == 0.0
    env.stumble = torch.zeros(1)
    env.asset.data.projected_gravity_b[:, 0] = 0.5    # tilted -> not landed
    assert float(reward_obstacle_clearance_omni(env, "base_parkour", FOOTPADS, ROBOT,
                                                "base_velocity", **kw)[0]) == 0.0


# --------------------------------------------------------------------------- termination
def test_termination_is_the_forward_one_without_the_goal_cutoff():
    env = _Env(4)
    # roll, pitch, height, timeout: identical to the forward function
    env.asset.data.root_state_w[1, 3:7] = torch.tensor([math.cos(0.9), math.sin(0.9), 0.0, 0.0])
    env.asset.data.root_state_w[2, 2] = -0.5
    env.episode_length_buf = torch.tensor([0, 0, 0, 1000])
    assert torch.equal(
        terminate_episode_no_goals(env, ROBOT), terminate_episode(env, ROBOT)
    )
    assert terminate_episode_no_goals(env, ROBOT).tolist() == [False, True, True, True]
    # the goal cutoff fires in the forward function and not in the omni one
    env.parkour_manager.event.cur_goal_idx = torch.full((4,), 8)
    assert terminate_episode(env, ROBOT).tolist() == [True, True, True, True]
    assert terminate_episode_no_goals(env, ROBOT).tolist() == [False, True, True, True]


def test_termination_never_touches_the_parkour_manager():
    """A ring episode must not end because bookkeeping moved ``cur_goal_idx``."""
    src = OMNI_REWARDS_PY.read_text()
    tree = ast.parse(src)
    fn = [n for n in tree.body
          if isinstance(n, ast.FunctionDef) and n.name == "terminate_episode_no_goals"][0]
    statements = [
        n for n in fn.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))  # skip the docstring
    ]
    body = "\n".join(ast.unparse(n) for n in statements)
    for token in ("cur_goal_idx", "parkour_manager", "num_goals"):
        assert token not in body, token


# --------------------------------------------------------------------------- the cfg
def _class_fields(path: Path, cls_name: str) -> dict:
    src, tree = _nodes(path)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name][0]
    out = {}
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node.value
    return out


def _bases(path: Path, cls_name: str) -> list[str]:
    _, tree = _nodes(path)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name][0]
    return [b.id for b in cls.bases if isinstance(b, ast.Name)]


def _kwargs(call: ast.Call) -> dict:
    return {kw.arg: kw.value for kw in call.keywords}


FWD_FIELDS = _class_fields(FWD_MDP_CFG_PY, "CrabHexFlatWalkRewardsCfg")
OMNI_FIELDS = _class_fields(OMNI_REWARDS_PY, "CrabHexOmniFlatWalkRewardsCfg")


#: The re-declared (same-name, func-swapped or std-retuned) fields.
SWAPPED_FIELDS = {
    "track_ang_vel_z_exp",
    "penalty_lin_vel_y",
    "reward_tracking_goal_vel",
    "reward_tracking_yaw",
    "reward_obstacle_clearance",
}
#: The TWO added fields: the omni cfg is the forward 35 plus these (37), nothing else.
#: ``reward_facing_turn`` joined on 2026-09-18 (plan Part A2) at weight 0.0 -- a name that logs
#: zero on every line until ``KRABBY_OMNI_FACING_TURN_W`` arms it.
ADDED_FIELDS = {"reward_yaw_progress", "reward_facing_turn"}


def test_the_cfg_is_the_forward_35_plus_reward_yaw_progress():
    """The forward 35 are a SUBSET of the omni 37, and the extra two are the added terms.

    Every other name still has to match exactly: the telemetry labels, the ``KRABBY_PHASEOUT``
    term list and the driver's log scraping all key on the forward names, so an omni-only
    RENAME would break them silently -- only a pure ADDITION is safe, and only these two.
    """
    assert len(FWD_FIELDS) == 35
    assert _bases(OMNI_REWARDS_PY, "CrabHexOmniFlatWalkRewardsCfg") == ["CrabHexFlatWalkRewardsCfg"]
    assert set(OMNI_FIELDS) == SWAPPED_FIELDS | ADDED_FIELDS
    assert SWAPPED_FIELDS <= set(FWD_FIELDS)                # every swap is a parent field
    assert set(OMNI_FIELDS) - set(FWD_FIELDS) == ADDED_FIELDS
    # the full term set the built cfg exposes: 35 inherited names + the two added ones
    assert len(set(FWD_FIELDS) | set(OMNI_FIELDS)) == 37


def test_the_added_yaw_progress_term():
    """weight 1.0, the omni function, and ``clip: None`` = read the command's own cap."""
    call = _kwargs(OMNI_FIELDS["reward_yaw_progress"])
    assert isinstance(call["func"], ast.Name) and call["func"].id == "reward_yaw_progress"
    assert ast.literal_eval(call["weight"]) == 1.0
    params = {
        ast.literal_eval(k): v for k, v in zip(call["params"].keys, call["params"].values)
    }
    assert set(params) == {"command_name", "asset_cfg", "clip"}
    assert ast.literal_eval(params["command_name"]) == "base_velocity"
    assert ast.literal_eval(params["clip"]) is None, "the cap must come from the command cfg"
    assert ast.unparse(params["asset_cfg"]) == "SceneEntityCfg('robot')"
    # per-step income ceiling: weight x cap, below the linear progress term's 0.60 x 1.05
    fwd_prog = _kwargs(FWD_FIELDS["reward_forward_progress_along_command"])
    fwd_params = {
        ast.literal_eval(k): v
        for k, v in zip(fwd_prog["params"].keys, fwd_prog["params"].values)
    }
    fwd_ceiling = ast.literal_eval(fwd_prog["weight"]) * ast.literal_eval(
        fwd_params["max_speed_scale"]
    )
    assert 1.0 * ANG_VEL_MAX < fwd_ceiling


def test_the_added_facing_turn_term():
    """The 37th name: weight **0.0** (the shipped dose), the turn-slot function, and the same
    three params as ``reward_tracking_yaw``.

    The weight is the whole bitwise-identity claim for this step: at 0.0 the term appears in
    the log as ``Episode_Reward/reward_facing_turn`` printing 0 on every line and changes no
    other number, so the legacy path is untouched until ``KRABBY_OMNI_FACING_TURN_W`` is set.
    """
    call = _kwargs(OMNI_FIELDS["reward_facing_turn"])
    assert isinstance(call["func"], ast.Name)
    assert call["func"].id == "reward_facing_exp_turn_slot"
    assert ast.literal_eval(call["weight"]) == 0.0, "the shipped dose is OFF"
    params = {
        ast.literal_eval(k): v for k, v in zip(call["params"].keys, call["params"].values)
    }
    assert set(params) == {"asset_cfg", "parkour_name", "command_name"}
    assert ast.literal_eval(params["command_name"]) == "base_velocity"
    assert ast.literal_eval(params["parkour_name"]) == "base_parkour"
    assert ast.unparse(params["asset_cfg"]) == "SceneEntityCfg('robot')"
    # it is the FACING term with a gate: same params as the ungated one, term for term
    yaw_params = {
        ast.literal_eval(k): ast.unparse(v)
        for k, v in zip(
            _kwargs(OMNI_FIELDS["reward_tracking_yaw"])["params"].keys,
            _kwargs(OMNI_FIELDS["reward_tracking_yaw"])["params"].values,
        )
    }
    assert {k: ast.unparse(v) for k, v in params.items()} == yaw_params
    # ... and it is DECLARED AFTER reward_yaw_progress, so it is the 37th field, not the 36th
    order = [
        t.id
        for node in ast.parse(OMNI_REWARDS_PY.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "CrabHexOmniFlatWalkRewardsCfg"
        for stmt in node.body
        if isinstance(stmt, ast.Assign)
        for t in stmt.targets
        if isinstance(t, ast.Name)
    ]
    assert order[-2:] == ["reward_yaw_progress", "reward_facing_turn"]


@pytest.mark.parametrize(
    "term,func",
    [
        ("penalty_lin_vel_y", "penalty_cross_command_vel_l2"),
        ("reward_tracking_goal_vel", "reward_tracking_command_vel_on_obstacle"),
        ("reward_tracking_yaw", "reward_facing_exp"),
        ("reward_obstacle_clearance", "reward_obstacle_clearance_omni"),
    ],
)
def test_the_four_swapped_functions(term, func):
    node = _kwargs(OMNI_FIELDS[term])["func"]
    assert isinstance(node, ast.Name) and node.id == func
    assert func in _OMNI_NS, f"{func} is not a module-level function of the omni module"


def test_every_params_key_is_a_real_argument_of_the_swapped_function():
    """Isaac's term resolver checks this at env build; catch a typo here instead."""
    import inspect

    for term, func in (
        ("penalty_lin_vel_y", "penalty_cross_command_vel_l2"),
        ("reward_tracking_goal_vel", "reward_tracking_command_vel_on_obstacle"),
        ("reward_tracking_yaw", "reward_facing_exp"),
        ("reward_obstacle_clearance", "reward_obstacle_clearance_omni"),
        ("reward_yaw_progress", "reward_yaw_progress"),
        ("reward_facing_turn", "reward_facing_exp_turn_slot"),
    ):
        sig = inspect.signature(_OMNI_NS[func])
        args = set(sig.parameters) - {"env"}
        keys = {ast.literal_eval(k) for k in _kwargs(OMNI_FIELDS[term])["params"].keys}
        assert keys <= args, (term, func, keys - args)
        # and every argument without a default is supplied
        required = {
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and name != "env"
        }
        assert required <= keys, (term, func, required - keys)


def test_weights_are_the_forward_literals():
    """The knob-timing pillar: every RE-DECLARED term keeps the forward class-body weight.

    (The two added terms have no forward literal to match; their weights are pinned by
    ``test_the_added_yaw_progress_term`` / ``test_the_added_facing_turn_term`` and dosed by
    ``KRABBY_YAW_PROGRESS_W`` / ``KRABBY_OMNI_FACING_TURN_W``.)
    """
    for term in set(OMNI_FIELDS) - ADDED_FIELDS:
        omni_w = ast.literal_eval(_kwargs(OMNI_FIELDS[term])["weight"])
        fwd_w = ast.literal_eval(_kwargs(FWD_FIELDS[term])["weight"])
        assert omni_w == fwd_w, term


def test_track_ang_vel_z_exp_keeps_its_function_and_gets_std_0_15():
    omni_call = _kwargs(OMNI_FIELDS["track_ang_vel_z_exp"])
    fwd_call = _kwargs(FWD_FIELDS["track_ang_vel_z_exp"])
    # same function object: both name the symbol imported from isaaclab's mdp rewards
    assert isinstance(omni_call["func"], ast.Name) and omni_call["func"].id == "track_ang_vel_z_exp"
    assert isinstance(fwd_call["func"], ast.Name) and fwd_call["func"].id == "track_ang_vel_z_exp"
    for path in (OMNI_REWARDS_PY, FWD_MDP_CFG_PY):
        _, tree = _nodes(path)
        imports = [
            n for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.module == "isaaclab.envs.mdp.rewards"
        ]
        assert any(a.name == "track_ang_vel_z_exp" for imp in imports for a in imp.names), path

    omni_params = {
        ast.literal_eval(k): v for k, v in zip(omni_call["params"].keys, omni_call["params"].values)
    }
    fwd_params = {
        ast.literal_eval(k): v for k, v in zip(fwd_call["params"].keys, fwd_call["params"].values)
    }
    assert set(omni_params) == set(fwd_params) == {"command_name", "std"}
    assert ast.literal_eval(omni_params["command_name"]) == "base_velocity"
    assert _OMNI_NS["OMNI_ANG_VEL_STD"] == 0.15
    assert isinstance(omni_params["std"], ast.Name) and omni_params["std"].id == "OMNI_ANG_VEL_STD"
    # the forward value is sqrt(0.25) = 0.5 -- flat across this plant's whole turning band
    assert ast.unparse(fwd_params["std"]) == "math.sqrt(0.25)"
    assert math.sqrt(0.25) == 0.5


def test_the_reward_params_keep_the_forward_keys():
    """Params may GAIN keys (the twins need a command name); none may go missing."""
    for term in ("penalty_lin_vel_y", "reward_tracking_goal_vel", "reward_tracking_yaw",
                 "reward_obstacle_clearance"):
        omni_keys = {ast.literal_eval(k) for k in _kwargs(OMNI_FIELDS[term])["params"].keys}
        fwd_keys = {ast.literal_eval(k) for k in _kwargs(FWD_FIELDS[term])["params"].keys}
        assert fwd_keys <= omni_keys, (term, fwd_keys - omni_keys)
        assert omni_keys - fwd_keys <= {"command_name"}, (term, omni_keys - fwd_keys)
    # the non-command literals are the forward ones, value for value
    omni_clear = {
        ast.literal_eval(k): ast.unparse(v)
        for k, v in zip(
            _kwargs(OMNI_FIELDS["reward_obstacle_clearance"])["params"].keys,
            _kwargs(OMNI_FIELDS["reward_obstacle_clearance"])["params"].values,
        )
    }
    fwd_clear = {
        ast.literal_eval(k): ast.unparse(v)
        for k, v in zip(
            _kwargs(FWD_FIELDS["reward_obstacle_clearance"])["params"].keys,
            _kwargs(FWD_FIELDS["reward_obstacle_clearance"])["params"].values,
        )
    }
    for key, value in fwd_clear.items():
        assert omni_clear[key] == value, key


def test_the_terminations_cfg_swaps_only_total_terminates():
    fields = _class_fields(OMNI_REWARDS_PY, "CrabHexOmniFlatWalkTerminationsCfg")
    assert set(fields) == {"total_terminates"}
    assert _bases(OMNI_REWARDS_PY, "CrabHexOmniFlatWalkTerminationsCfg") == [
        "CrabHexFlatWalkTerminationsCfg"
    ]
    call = _kwargs(fields["total_terminates"])
    assert isinstance(call["func"], ast.Name) and call["func"].id == "terminate_episode_no_goals"
    assert ast.literal_eval(call["time_out"]) is True
    fwd_fields = _class_fields(FWD_MDP_CFG_PY, "CrabHexFlatWalkTerminationsCfg")
    fwd = _kwargs(fwd_fields["total_terminates"])
    assert ast.unparse(call["params"]) == ast.unparse(fwd["params"])
    assert set(_class_fields(FWD_MDP_CFG_PY, "CrabHexFlatWalkTerminationsCfg")) == {
        "total_terminates", "crab_failure",
    }
