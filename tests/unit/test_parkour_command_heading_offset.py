"""``UniformParkourCommand``'s ``heading_offset`` seam: identity for the forward task, and the
one thing that makes a rotated travel axis trainable at all.

The velocity command lives in ``vel_command_b[:, 0]`` and is read as "speed along the travel
axis". For a task whose travel axis is not body +x, the chassis must therefore hold a constant
yaw offset from the waypoint heading -- otherwise ``_update_command`` sees a permanent ~90 deg
heading error, saturates the yaw-rate command at the clip, and the robot spins instead of
walking (test ``e`` below pins exactly that failure mode).

``ParkourCommandCfg`` deliberately has NO ``heading_offset`` field; the sideways command cfg is
the only place it exists, and it is read with ``getattr(self.cfg, "heading_offset", 0.0)``. So
for the forward task every assertion here is "bit-for-bit what it did before".

No Isaac: the module is loaded from its file through a synthetic package (so its relative
imports ``..travel_frame`` and ``.command_sampling`` resolve against the real sources) with
``sys.modules`` stubs for the isaaclab/omni names it imports -- the pattern of
``test_crab_hex_rsi_spawn_fix.py``.
"""

import importlib
import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
MDP_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp"
CMD_DIR = MDP_DIR / "parkour_commands"

# Synthetic package names: distinct from ``parkour_isaaclab.*`` so importing the real package
# (which would drag in Isaac) is never attempted, while relative imports still resolve.
PKG = "_upc_under_test"
SUBPKG = PKG + ".parkour_commands"


def _synthetic_package(name: str, path: Path) -> types.ModuleType:
    """A package whose submodules are searched for in ``path`` -- no ``__init__.py`` executed."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
    return mod


_pkg = _synthetic_package(PKG, MDP_DIR)
_subpkg = _synthetic_package(SUBPKG, CMD_DIR)
_pkg.parkour_commands = _subpkg

# The real travel-frame helpers (torch only) -- also the source of the REAL ``wrap_to_pi`` the
# stubbed ``isaaclab.utils.math`` hands to ``_update_command``.
travel_frame = importlib.import_module(PKG + ".travel_frame")


class _CommandTerm:
    """Minimal stand-in for ``isaaclab.managers.CommandTerm``: only what the term's
    ``__init__`` consumes (cfg, env handle, sizes, the metrics dict)."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.metrics: dict[str, torch.Tensor] = {}


def _stub(name: str, **attrs) -> types.ModuleType:
    """Get-or-create ``sys.modules[name]`` and fill in only the attributes it lacks.

    Other pure suites share these ``sys.modules`` entries (and may have created them first),
    so this never overwrites what is already there.
    """
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for key, value in attrs.items():
        if not hasattr(mod, key):
            setattr(mod, key, value)
    return mod


_omni = _stub("omni")
_omni.log = _stub("omni.log", warn=lambda *a, **k: None, info=lambda *a, **k: None)
_isaaclab = _stub("isaaclab")
_isaaclab.utils = _stub("isaaclab.utils")
_isaaclab.utils.math = _stub(
    "isaaclab.utils.math",
    wrap_to_pi=travel_frame.wrap_to_pi,  # the real thing, copied from isaaclab/utils/math.py
    quat_from_euler_xyz=lambda *a, **k: None,  # debug-vis only
    quat_mul=lambda *a, **k: None,
)
_isaaclab.assets = _stub("isaaclab.assets", Articulation=type("Articulation", (), {}))
_isaaclab.managers = _stub(
    "isaaclab.managers",
    CommandTerm=_CommandTerm,
    # ``RewardManager`` is not used here; it keeps this shared stub a superset of what the
    # other pure suites expect from ``isaaclab.managers`` regardless of collection order.
    RewardManager=object,
)
_isaaclab.markers = _stub(
    "isaaclab.markers", VisualizationMarkers=type("VisualizationMarkers", (), {})
)

upc = importlib.import_module(SUBPKG + ".uniform_parkour_command")
command_sampling = importlib.import_module(SUBPKG + ".command_sampling")

N = 16
SEED = 1234
LIN_VEL_X = (0.0, 0.35)
STAND_FRAC = 0.2
LIN_VEL_CLIP = 0.2
ANG_VEL_CLIP = 0.4
RESAMPLE_S = 6.0
STEP_DT = 0.02
OFFSET = -math.pi / 2  # the sideways task's heading offset (travel axis = body +y)


class _Robot:
    def __init__(self, n: int):
        self.data = types.SimpleNamespace(
            root_lin_vel_b=torch.zeros(n, 3),
            root_ang_vel_b=torch.zeros(n, 3),
            heading_w=torch.zeros(n),
        )


class _Scene:
    def __init__(self, robot):
        self._items = {"robot": robot}

    def __getitem__(self, key):
        return self._items[key]


def _cfg(*, stand_frac=STAND_FRAC, heading_offset=None, heading=(0.0, 0.0)):
    cfg = types.SimpleNamespace(
        asset_name="robot",
        heading_control_stiffness=1.0,
        small_commands_to_zero=True,
        stand_frac=stand_frac,
        resampling_time_range=(RESAMPLE_S, RESAMPLE_S),
        ranges=types.SimpleNamespace(lin_vel_x=LIN_VEL_X, heading=heading),
        clips=types.SimpleNamespace(lin_vel_clip=LIN_VEL_CLIP, ang_vel_clip=ANG_VEL_CLIP),
    )
    if heading_offset is not None:  # the forward cfg simply does not have the attribute
        cfg.heading_offset = heading_offset
    return cfg


def _term(cfg):
    robot = _Robot(N)
    env = types.SimpleNamespace(
        num_envs=N,
        device="cpu",
        scene=_Scene(robot),
        step_dt=STEP_DT,
        max_episode_length=1000,
    )
    return upc.UniformParkourCommand(cfg, env), robot


MAX_COMMAND_STEP = RESAMPLE_S / STEP_DT


def _resampled_vx(cfg):
    term, _ = _term(cfg)
    ids = torch.arange(N)
    torch.manual_seed(SEED)
    term._resample_command(ids)
    return term, term.vel_command_b[ids, 0].clone()


# ------------------------------------------------------------------ (a) forward path unchanged
def test_resample_without_offset_is_the_untouched_draw():
    term, vx = _resampled_vx(_cfg())
    # Same seed, same draw ORDER: u_vel, then u_stand, then the heading uniform_ (which does
    # not feed vx). If the seam had reordered or added a draw, vx would differ.
    torch.manual_seed(SEED)
    u_vel = torch.rand(N)
    u_stand = torch.rand(N)
    expected, _stand = command_sampling.sample_slot(
        u_vel, u_stand, STAND_FRAC, LIN_VEL_X[0], LIN_VEL_X[1], LIN_VEL_CLIP
    )
    assert torch.equal(vx, expected)
    assert torch.equal(term.heading_target, torch.zeros(N))


# ------------------------------------------------------------------ (b) the offset is applied
def test_resample_with_offset_shifts_only_the_heading():
    _term_a, vx_a = _resampled_vx(_cfg())
    term_b, vx_b = _resampled_vx(_cfg(heading_offset=OFFSET))
    assert torch.equal(vx_b, vx_a)
    assert torch.allclose(term_b.heading_target, torch.full((N,), OFFSET), rtol=0.0, atol=1e-7)


def test_unarmed_slot_path_also_carries_the_offset():
    # The second heading write (``stand_frac`` unset -> the historical clip rule).
    term_a, vx_a = _resampled_vx(_cfg(stand_frac=None))
    term_b, vx_b = _resampled_vx(_cfg(stand_frac=None, heading_offset=OFFSET))
    assert torch.equal(vx_b, vx_a)
    assert torch.equal(term_a.heading_target, torch.zeros(N))
    assert torch.allclose(term_b.heading_target, torch.full((N,), OFFSET), rtol=0.0, atol=1e-7)


# ------------------------------------------------------------------ (c)/(d) the metric
def test_error_vel_xy_without_offset_is_bitwise_the_old_formula():
    cfg = _cfg()
    term, robot = _term(cfg)
    term.vel_command_b[:, 0] = 0.3
    robot.data.root_lin_vel_b[:, 0] = 0.21
    robot.data.root_lin_vel_b[:, 1] = -0.04
    term._update_metrics()
    expected = (
        torch.norm(term.vel_command_b[:, :2] - robot.data.root_lin_vel_b[:, :2], dim=-1)
        / MAX_COMMAND_STEP
    )
    assert torch.equal(term.metrics["error_vel_xy"], expected)


def test_error_vel_xy_with_offset_reads_the_travel_axis():
    cfg = _cfg(heading_offset=OFFSET)
    term, robot = _term(cfg)
    term.vel_command_b[:, 0] = 0.3  # 0.3 m/s along the travel axis ...
    robot.data.root_lin_vel_b[:, 1] = 0.3  # ... which is body +y: perfect tracking
    term._update_metrics()
    assert torch.allclose(
        term.metrics["error_vel_xy"], torch.zeros(N), rtol=0.0, atol=1e-6
    )


# ------------------------------------------------------------------ (e) the S1 failure mode
@pytest.mark.parametrize("heading_offset,expected", [(OFFSET, 0.0), (None, 1.0)])
def test_yaw_rate_command_saturates_without_the_offset(heading_offset, expected):
    cfg = _cfg(heading_offset=heading_offset)
    term, robot = _term(cfg)
    robot.data.heading_w[:] = -math.pi / 2  # the chassis yaw a body-+y travel axis needs
    torch.manual_seed(SEED)
    term._resample_command(torch.arange(N))
    term._update_command()
    # With the offset the heading target IS -pi/2 -> zero error -> zero yaw rate. Without it
    # the target is 0, the error is a permanent +pi/2, and the yaw-rate command pins at the
    # +-1 clip every step: the robot spins instead of walking sideways.
    assert torch.allclose(
        term.vel_command_b[:, 2], torch.full((N,), expected), rtol=0.0, atol=1e-7
    )
