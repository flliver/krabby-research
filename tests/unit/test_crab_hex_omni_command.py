"""The omni command term and its pure math (omni plan A4/D1), on CPU, without Isaac.

Wrong signs in this file are silent in training -- a lateral command that quietly points at
the robot's right, a facing P-control that steers away from its target, a speed cap that is
re-applied by rotating instead of clamping. So the assertions here are written against the
*conventions* (world vs body, ``+y`` = left, "hold the world direction, clamp the speed")
rather than against transcribed numbers, and the endpoint/continuity properties of the
ellipse are checked against an independently-derived closed form.

How the module is loaded (no Isaac Sim, no ``parkour_tasks`` package import):

* ``sys.modules`` stubs for the isaaclab/omni names ``parkour_command_cfg.py`` and
  ``uniform_parkour_command.py`` import -- the pattern of
  ``test_parkour_command_heading_offset.py:41-111``, extended with a small ``configclass``
  stand-in so the real cfg CLASS BODIES execute (this test therefore pins the shipped
  defaults, not a copy of them);
* synthetic packages whose ``__path__`` points at the real source dirs, so the modules'
  relative imports resolve -- ``parkour_isaaclab.envs.mdp.parkour_commands`` under its real
  dotted name (the omni module imports it absolutely) and the omni ``mdp`` dir under a
  private name (``test_crab_hex_side_mirror.py``'s path-loading idea).
"""

import copy
import dataclasses
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
LIB_MDP = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp"
CMD_DIR = LIB_MDP / "parkour_commands"
OMNI_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "mdp"


# --------------------------------------------------------------------------- loader plumbing
def _synthetic_package(name: str, path: Path) -> types.ModuleType:
    """A package whose submodules are searched for in ``path`` -- no ``__init__.py`` run."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
    return mod


def _stub(name: str, **attrs) -> types.ModuleType:
    """Get-or-create ``sys.modules[name]``, filling in only the attributes it lacks."""
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    for key, value in attrs.items():
        if not hasattr(mod, key):
            setattr(mod, key, value)
    return mod


# The real travel-frame helpers (torch only): the source of the REAL ``wrap_to_pi`` handed to
# the stubbed ``isaaclab.utils.math``, and the reference the omni module's private copy is
# pinned against below.
_lib_pkg = _synthetic_package("parkour_isaaclab", REPO / "parkour" / "parkour_isaaclab")
_envs_pkg = _synthetic_package("parkour_isaaclab.envs", REPO / "parkour" / "parkour_isaaclab" / "envs")
_mdp_pkg = _synthetic_package("parkour_isaaclab.envs.mdp", LIB_MDP)
_cmd_pkg = _synthetic_package("parkour_isaaclab.envs.mdp.parkour_commands", CMD_DIR)
_lib_pkg.envs = _envs_pkg
_envs_pkg.mdp = _mdp_pkg
_mdp_pkg.parkour_commands = _cmd_pkg

travel_frame = importlib.import_module("parkour_isaaclab.envs.mdp.travel_frame")


class _CommandTerm:
    """Stand-in for ``isaaclab.managers.CommandTerm``: only what the term's ctor consumes."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.metrics: dict[str, torch.Tensor] = {}


def _configclass(cls):
    """Minimal stand-in for ``isaaclab.utils.configclass``.

    Enough for the cfg class bodies under test: keyword-only construction, class defaults
    deep-copied per instance (so a nested ``Clips`` default is not shared), inherited
    fields included, ``__post_init__`` honoured. Nested cfg CLASSES (``Ranges``, ``Clips``)
    and methods are not fields.
    """
    names: list[str] = []
    for base in reversed(cls.__mro__):
        for key in base.__dict__.get("__annotations__", {}):
            if key not in names:
                names.append(key)
        for key, val in vars(base).items():
            if key.startswith("_") or key in names:
                continue
            if isinstance(val, (types.FunctionType, classmethod, staticmethod, property, type)):
                continue
            names.append(key)

    def __init__(self, **kwargs):
        for key in names:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))
            elif hasattr(type(self), key):
                setattr(self, key, copy.deepcopy(getattr(type(self), key)))
        if kwargs:
            raise TypeError(f"{cls.__name__} got unexpected fields {sorted(kwargs)}")
        post_init = getattr(self, "__post_init__", None)
        if post_init is not None:
            post_init()

    cls.__init__ = __init__
    cls.__cfg_fields__ = tuple(names)
    return cls


def _marker_proto(scale=(1.0, 1.0, 1.0), color=(0.0, 1.0, 0.0)):
    """One marker prototype: the two attributes the debug-vis code reads or recolours."""
    return types.SimpleNamespace(
        scale=scale, visual_material=types.SimpleNamespace(diffuse_color=color)
    )


class _MarkerCfg:
    """Stand-in for the arrow/frame ``VisualizationMarkersCfg`` singletons."""

    def __init__(self, prim_path=None, markers=None):
        self.prim_path = prim_path
        self.markers = markers or {"arrow": _marker_proto()}

    def replace(self, **kwargs):
        # SHALLOW on ``markers``, like the real ``dataclasses.replace`` -- the omni module's
        # deep-copy comment is only true if this stub reproduces the aliasing it avoids.
        new = _MarkerCfg(prim_path=kwargs.get("prim_path", self.prim_path), markers=self.markers)
        return new


def _quat_from_euler_xyz(roll, pitch, yaw):
    """The real ``isaaclab.utils.math.quat_from_euler_xyz`` (wxyz), transcribed.

    The debug-vis helpers are only meaningful against the real quaternion convention, so
    the stub is the arithmetic itself rather than a sentinel.
    """
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dim=-1,
    )


_omni_mod = _stub("omni")
_omni_mod.log = _stub("omni.log", warn=lambda *a, **k: None, info=lambda *a, **k: None)
_isaaclab = _stub("isaaclab")
_isaaclab.utils = _stub("isaaclab.utils", configclass=_configclass)
_isaaclab.utils.math = _stub(
    "isaaclab.utils.math",
    wrap_to_pi=travel_frame.wrap_to_pi,  # the real thing
    quat_from_euler_xyz=_quat_from_euler_xyz,  # the real thing (omni debug vis)
    quat_mul=lambda *a, **k: None,  # base-class debug vis only; omni never calls it
)
_isaaclab.assets = _stub("isaaclab.assets", Articulation=type("Articulation", (), {}))
_isaaclab.managers = _stub(
    "isaaclab.managers",
    CommandTerm=_CommandTerm,
    CommandTermCfg=_configclass(
        type(
            "CommandTermCfg",
            (),
            {
                "__annotations__": {
                    "class_type": type,
                    "resampling_time_range": tuple,
                    "debug_vis": bool,
                },
                "class_type": dataclasses.MISSING,
                "resampling_time_range": dataclasses.MISSING,
                "debug_vis": False,
            },
        )
    ),
    RewardManager=object,
)
_isaaclab.markers = _stub(
    "isaaclab.markers",
    VisualizationMarkers=type("VisualizationMarkers", (), {}),
    VisualizationMarkersCfg=_MarkerCfg,
)
_isaaclab.markers.config = _stub(
    "isaaclab.markers.config",
    BLUE_ARROW_X_MARKER_CFG=_MarkerCfg(markers={"arrow": _marker_proto(color=(0.0, 0.0, 1.0))}),
    GREEN_ARROW_X_MARKER_CFG=_MarkerCfg(markers={"arrow": _marker_proto(color=(0.0, 1.0, 0.0))}),
    FRAME_MARKER_CFG=_MarkerCfg(),
)

upc = importlib.import_module("parkour_isaaclab.envs.mdp.parkour_commands.uniform_parkour_command")
pcfg = importlib.import_module("parkour_isaaclab.envs.mdp.parkour_commands.parkour_command_cfg")
command_sampling = importlib.import_module("parkour_isaaclab.envs.mdp.parkour_commands.command_sampling")
# The synthetic package never ran its ``__init__.py``; give it the two names the omni module
# imports from it.
_cmd_pkg.UniformParkourCommand = upc.UniformParkourCommand
_cmd_pkg.ParkourCommandCfg = pcfg.ParkourCommandCfg

_omni_pkg = _synthetic_package("_omni_mdp_under_test", OMNI_MDP)
omni_math = importlib.import_module("_omni_mdp_under_test.crab_hex_omni_command_math")
omni = importlib.import_module("_omni_mdp_under_test.crab_hex_omni_command")

ParkourCommandCfg = pcfg.ParkourCommandCfg
CrabHexOmniCommand = omni.CrabHexOmniCommand
CrabHexOmniCommandCfg = omni.CrabHexOmniCommandCfg
S_STAND, S_MOVE, S_TURN, S_MOVE_TURN = (
    omni_math.SLOT_STAND,
    omni_math.SLOT_MOVE,
    omni_math.SLOT_TURN,
    omni_math.SLOT_MOVE_TURN,
)

# ------------------------------------------------------------------------------- fixtures
N = 4096
SEED = 20260916
STEP_DT = 0.02
RESAMPLE_S = 10.0
EPISODE_STEPS = 2000
A, B, C = 0.35, 0.25, 0.25
STIFFNESS = 1.5  # the per-phase value the knob block writes (the Go2 literal is 0.8)


class _Robot:
    def __init__(self, n: int, yaw: torch.Tensor | None = None):
        self.is_initialized = True
        self.data = types.SimpleNamespace(
            root_lin_vel_b=torch.zeros(n, 3),
            root_ang_vel_b=torch.zeros(n, 3),
            root_lin_vel_w=torch.zeros(n, 3),
            root_pos_w=torch.zeros(n, 3),
            heading_w=torch.zeros(n) if yaw is None else yaw.clone(),
        )


class _Scene:
    def __init__(self, robot):
        self._items = {"robot": robot}

    def __getitem__(self, key):
        return self._items[key]


def _cfg(**overrides) -> CrabHexOmniCommandCfg:
    """The shipped cfg class with the per-phase literals the env cfg writes."""
    cfg = CrabHexOmniCommandCfg(
        asset_name="robot",
        resampling_time_range=(RESAMPLE_S, RESAMPLE_S),
        heading_control_stiffness=STIFFNESS,
        ranges=ParkourCommandCfg.Ranges(lin_vel_x=(0.0, 0.35), heading=(0.0, 0.0)),
    )
    for key, value in overrides.items():
        if "." in key:
            head, tail = key.split(".", 1)
            setattr(getattr(cfg, head), tail, value)
        else:
            setattr(cfg, key, value)
    return cfg


def _term(cfg, *, n: int = N, yaw: torch.Tensor | None = None):
    robot = _Robot(n, yaw)
    env = types.SimpleNamespace(
        num_envs=n,
        device="cpu",
        scene=_Scene(robot),
        step_dt=STEP_DT,
        max_episode_length=EPISODE_STEPS,
    )
    return CrabHexOmniCommand(cfg, env), robot


def _spread_yaw(n: int) -> torch.Tensor:
    """A deterministic, non-trivial yaw spread over (-pi, pi] -- never all zeros, so a
    missing ``- yaw`` anywhere in the world->body conversion cannot pass unnoticed."""
    return omni_math.wrap_to_pi(torch.linspace(-3.0, 3.1, n))


def _resample(term, *, seed: int = SEED):
    ids = torch.arange(term.num_envs)
    torch.manual_seed(seed)
    term._resample_command(ids)
    return ids


# ========================================================================= the pure math
def test_wrap_to_pi_is_the_shared_implementation():
    """The module's private copy must not drift from ``travel_frame.wrap_to_pi``."""
    angles = torch.linspace(-4 * math.pi, 4 * math.pi, 100_001, dtype=torch.float64)
    assert torch.equal(omni_math.wrap_to_pi(angles), travel_frame.wrap_to_pi(angles))
    # the documented odd-multiple-of-pi convention
    assert float(omni_math.wrap_to_pi(torch.tensor([math.pi], dtype=torch.float64))) == pytest.approx(math.pi)
    assert float(omni_math.wrap_to_pi(torch.tensor([-math.pi], dtype=torch.float64))) == pytest.approx(-math.pi)


def test_slot_encoding_is_the_two_bit_mask():
    slots = torch.arange(4)
    assert torch.equal(omni_math.slot_is_moving(slots), torch.tensor([False, True, False, True]))
    assert torch.equal(omni_math.slot_is_turning(slots), torch.tensor([False, False, True, True]))
    assert (S_STAND, S_MOVE, S_TURN, S_MOVE_TURN) == (0, 1, 2, 3)
    assert omni_math.SLOT_NAMES == ("stand", "move", "turn", "move_turn")


def _cap_reference(phi: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """Independent closed form: the scale ``t`` at which ``t*(cos, sin)`` hits the ellipse.

    Solve ``(t cos)^2/A^2 + (t sin)^2/b^2 = 1`` -> ``t = 1/sqrt(cos^2/A^2 + sin^2/b^2)``.
    Derived from the ellipse equation rather than from the shipped expression.
    """
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    semi_x = torch.where(cos_phi >= 0.0, torch.full_like(phi, a), torch.full_like(phi, c))
    return 1.0 / torch.sqrt((cos_phi / semi_x) ** 2 + (sin_phi / b) ** 2)


def test_ellipse_cap_endpoints():
    phi = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi], dtype=torch.float64)
    cap = omni_math.ellipse_cap(phi, A, B, C)
    expected = torch.tensor([A, B, B, C, C], dtype=torch.float64)
    assert torch.allclose(cap, expected, rtol=0.0, atol=1e-12)


def test_ellipse_cap_endpoints_with_distinct_semi_axes():
    """a != b != c, so a transposed axis cannot hide behind the default b == c."""
    a, b, c = 0.40, 0.30, 0.24
    phi = torch.tensor([0.0, math.pi / 2, -math.pi / 2, math.pi], dtype=torch.float64)
    cap = omni_math.ellipse_cap(phi, a, b, c)
    assert torch.allclose(cap, torch.tensor([a, b, b, c], dtype=torch.float64), rtol=0.0, atol=1e-12)


def test_ellipse_cap_matches_the_ellipse_equation_everywhere():
    phi = torch.linspace(-math.pi, math.pi, 20_001, dtype=torch.float64)
    a, b, c = 0.40, 0.30, 0.24
    assert torch.allclose(omni_math.ellipse_cap(phi, a, b, c), _cap_reference(phi, a, b, c), rtol=0.0, atol=1e-12)


def test_ellipse_cap_is_continuous_and_bounded_by_its_semi_axes():
    a, b, c = 0.40, 0.30, 0.24
    phi = torch.linspace(-math.pi, math.pi, 200_001, dtype=torch.float64)
    cap = omni_math.ellipse_cap(phi, a, b, c)
    # bounded by the semi-axes (the property the walk-floor assertion relies on)
    assert float(cap.min()) >= min(a, b, c) - 1e-12
    assert float(cap.max()) <= max(a, b, c) + 1e-12
    # continuous, INCLUDING across the +-pi/2 seam where the semi-axis switches a <-> c
    step = (phi[1] - phi[0]).item()
    jump = (cap[1:] - cap[:-1]).abs().max().item()
    assert jump < 10.0 * step * max(a, b, c), f"largest step {jump} at grid spacing {step}"
    # and the seam value itself is b from BOTH sides
    for eps in (1e-9, -1e-9):
        assert float(omni_math.ellipse_cap(torch.tensor([math.pi / 2 + eps], dtype=torch.float64), a, b, c)) == (
            pytest.approx(b, abs=1e-9)
        )


def test_ellipse_cap_rejects_non_positive_axes():
    phi = torch.zeros(3)
    for bad in ((0.0, 0.25, 0.25), (0.35, -0.1, 0.25), (0.35, 0.25, float("nan"))):
        with pytest.raises(ValueError):
            omni_math.ellipse_cap(phi, *bad)


def test_body_command_from_world_uses_phi_equals_theta_minus_yaw():
    """+pi/2 in the body frame is the robot's LEFT (body +y), and a world direction is
    read as ``phi = theta - yaw`` -- not ``yaw - theta``."""
    yaw = torch.tensor([math.pi / 2], dtype=torch.float64)  # robot faces world +y
    theta = torch.tensor([0.0], dtype=torch.float64)  # travel towards world +x
    speed = torch.tensor([0.20], dtype=torch.float64)
    cmd = omni_math.body_command_from_world(theta, speed, yaw, A, B, C)
    # world +x is to the RIGHT of a robot facing world +y -> body -y
    assert cmd[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert cmd[0, 1] == pytest.approx(-0.20, abs=1e-12)


def test_body_command_zero_speed_is_exactly_zero():
    yaw = _spread_yaw(64).double()
    theta = torch.linspace(-3.0, 3.0, 64, dtype=torch.float64)
    cmd = omni_math.body_command_from_world(theta, torch.zeros(64, dtype=torch.float64), yaw, A, B, C)
    assert torch.equal(cmd, torch.zeros(64, 2, dtype=torch.float64))


def test_recap_on_rotation_is_clamp_only_and_holds_the_world_direction():
    """The core re-cap contract (plan D1): a forward command at the cap ``a``, after the
    body yaws by +pi/2, is a lateral command of magnitude ``b`` pointing at the SAME world
    direction -- clamped, not rotated -- and it does not come back up when the body
    returns."""
    theta = torch.zeros(1, dtype=torch.float64)  # world +x
    speed = torch.full((1,), A, dtype=torch.float64)  # exactly the fore-aft cap

    cmd0 = omni_math.body_command_from_world(theta, speed, torch.zeros(1, dtype=torch.float64), A, B, C)
    assert cmd0[0].tolist() == pytest.approx([A, 0.0], abs=1e-12)

    yaw1 = torch.full((1,), math.pi / 2, dtype=torch.float64)
    cmd1 = omni_math.body_command_from_world(theta, speed, yaw1, A, B, C)
    assert float(torch.linalg.norm(cmd1[0])) <= B + 1e-12
    assert float(torch.linalg.norm(cmd1[0])) == pytest.approx(B, abs=1e-12)
    # direction held in WORLD: yaw + atan2(vy_b, vx_b) == theta
    world_dir = omni_math.wrap_to_pi(yaw1 + torch.atan2(cmd1[:, 1], cmd1[:, 0]))
    assert float(world_dir) == pytest.approx(0.0, abs=1e-9)
    # a partial turn caps between b and a, always on the world ray
    yaw_half = torch.full((1,), math.pi / 4, dtype=torch.float64)
    cmd_half = omni_math.body_command_from_world(theta, speed, yaw_half, A, B, C)
    n_half = float(torch.linalg.norm(cmd_half[0]))
    assert B - 1e-12 <= n_half <= A + 1e-12
    assert float(omni_math.wrap_to_pi(yaw_half + torch.atan2(cmd_half[:, 1], cmd_half[:, 0]))) == pytest.approx(
        0.0, abs=1e-9
    )


def test_heading_rate_command_sign_cap_and_dead_band():
    yaw = torch.zeros(5, dtype=torch.float64)
    # errors: well inside the dead band, just inside, just outside, saturating, negative
    ang_vel_max, ang_vel_clip = 0.3, 0.2
    err = torch.tensor([0.0, 0.13, 0.14, 2.0, -2.0], dtype=torch.float64)
    wz = omni_math.heading_rate_command(err, yaw, STIFFNESS, ang_vel_max, ang_vel_clip)
    assert float(wz[0]) == 0.0
    assert float(wz[1]) == 0.0, "0.13 * 1.5 = 0.195 <= 0.2 -> dead band"
    assert float(wz[2]) == pytest.approx(0.21, abs=1e-12), "0.14 * 1.5 = 0.21 > 0.2 -> passes"
    assert float(wz[3]) == pytest.approx(ang_vel_max, abs=1e-12)
    assert float(wz[4]) == pytest.approx(-ang_vel_max, abs=1e-12), "a +wz turns CCW; the sign must follow the error"
    assert float(wz.abs().max()) <= ang_vel_max


def test_heading_rate_command_wraps_the_shortest_way():
    """Target just past +pi with the robot just under it: turn the SHORT way (positive)."""
    yaw = torch.tensor([3.0], dtype=torch.float64)
    target = torch.tensor([-3.0], dtype=torch.float64)  # = 3.283 rad the short (positive) way
    wz = omni_math.heading_rate_command(target, yaw, STIFFNESS, 0.3, 0.2)
    assert float(wz) == pytest.approx(0.3, abs=1e-12), "wrap(-3 - 3) = +2.283 -> saturated positive"


def test_heading_rate_command_matches_the_stock_expression_at_the_stock_cap():
    """Same semantics as ``uniform_parkour_command._update_command``, cap aside."""
    yaw = _spread_yaw(257).double()
    target = torch.linspace(-4.0, 4.0, 257, dtype=torch.float64)
    stiffness, clip = 0.8, 0.4
    got = omni_math.heading_rate_command(target, yaw, stiffness, 1.0, clip)
    stock = torch.clip(travel_frame.wrap_to_pi(target - yaw) * stiffness, min=-1, max=1)
    stock = stock * (stock.abs() > clip)
    assert torch.equal(got, stock)


def test_heading_rate_command_rejects_bad_limits():
    with pytest.raises(ValueError):
        omni_math.heading_rate_command(torch.zeros(1), torch.zeros(1), 1.0, 0.0, 0.2)
    with pytest.raises(ValueError):
        omni_math.heading_rate_command(torch.zeros(1), torch.zeros(1), 1.0, 0.3, -0.1)


def test_sample_slots_frequencies_and_draw_count():
    probs = (0.15, 0.55, 0.10, 0.20)
    gen = torch.Generator().manual_seed(7)
    slots = omni_math.sample_slots(gen, probs, 20_000)
    assert slots.dtype == torch.long
    freq = torch.bincount(slots, minlength=4).double() / 20_000
    assert torch.allclose(freq, torch.tensor(probs, dtype=torch.float64), atol=0.02)
    # exactly one uniform per env, so a different mix consumes the same stream
    gen_a = torch.Generator().manual_seed(11)
    gen_b = torch.Generator().manual_seed(11)
    omni_math.sample_slots(gen_a, (0.25, 0.25, 0.25, 0.25), 32)
    omni_math.sample_slots(gen_b, (1.0, 0.0, 0.0, 0.0), 32)
    assert torch.equal(torch.rand(4, generator=gen_a), torch.rand(4, generator=gen_b))


def test_sample_slots_honours_zero_probability_slots_and_normalisation():
    gen = torch.Generator().manual_seed(3)
    only_turn = omni_math.sample_slots(gen, (0.0, 0.0, 1.0, 0.0), 500)
    assert torch.equal(only_turn, torch.full((500,), S_TURN))
    # un-normalised weights are normalised, not rejected
    gen = torch.Generator().manual_seed(3)
    scaled = omni_math.sample_slots(gen, (0.0, 0.0, 7.0, 0.0), 500)
    assert torch.equal(scaled, only_turn)
    for bad in ((), (-0.1, 0.5, 0.3, 0.3), (0.0, 0.0, 0.0, 0.0), (float("nan"), 1.0, 0.0, 0.0)):
        with pytest.raises(ValueError):
            omni_math.sample_slots(None, bad, 4)


def test_sample_travel_dir_band_is_body_relative_and_uniform_is_world():
    yaw = _spread_yaw(4096).double()
    gen = torch.Generator().manual_seed(5)
    # dir_band None -> world-uniform, independent of yaw
    theta = omni_math.sample_travel_dir(yaw, None, (-math.pi, math.pi), gen)
    assert float(theta.min()) < -3.0 and float(theta.max()) > 3.0
    corr = torch.corrcoef(torch.stack([theta, yaw]))[0, 1]
    assert abs(float(corr)) < 0.05, "a world-uniform direction must not track the body yaw"
    # dir_band (0, 0) -> exactly the body +x axis, bitwise
    theta0 = omni_math.sample_travel_dir(yaw, (0.0, 0.0), (-math.pi, math.pi), gen)
    assert torch.equal(theta0, yaw)
    # dir_band (lo, hi) -> yaw + U(lo, hi)
    lo, hi = 0.25, 0.75
    theta_band = omni_math.sample_travel_dir(yaw, (lo, hi), (-math.pi, math.pi), gen)
    delta = omni_math.wrap_to_pi(theta_band - yaw)
    assert float(delta.min()) >= lo - 1e-9 and float(delta.max()) <= hi + 1e-9
    with pytest.raises(ValueError):
        omni_math.sample_travel_dir(yaw, (1.0, 0.0), (-math.pi, math.pi), gen)
    with pytest.raises(ValueError):
        omni_math.sample_travel_dir(yaw, None, (1.0, 0.0), gen)


def test_sample_facing_band_is_relative_and_two_sided():
    yaw = _spread_yaw(8192).double()
    gen = torch.Generator().manual_seed(9)
    lo, hi = 0.2, 0.6
    psi = omni_math.sample_facing(yaw, lo, hi, gen)
    delta = omni_math.wrap_to_pi(psi - yaw)
    assert float(delta.abs().min()) >= lo - 1e-9
    assert float(delta.abs().max()) <= hi + 1e-9
    frac_left = float((delta > 0).double().mean())
    assert 0.45 < frac_left < 0.55, "the sign must be random, not biased to one side"
    with pytest.raises(ValueError):
        omni_math.sample_facing(yaw, -0.1, 0.6, gen)
    with pytest.raises(ValueError):
        omni_math.sample_facing(yaw, 0.6, 0.2, gen)


# ===================================================================== the command term
def test_walk_floor_is_derived_from_the_shared_constants():
    cfg = _cfg()
    assert command_sampling.WALK_MARGIN_M_S == 0.01
    assert omni.omni_walk_floor(cfg) == pytest.approx(cfg.clips.lin_vel_clip + command_sampling.WALK_MARGIN_M_S)
    assert omni.omni_walk_floor(cfg) == pytest.approx(0.21)  # with the stock clip
    cfg.clips.lin_vel_clip = 0.15
    assert omni.omni_walk_floor(cfg) == pytest.approx(0.16)


def test_shipped_cfg_defaults_are_the_plan_values():
    cfg = CrabHexOmniCommandCfg(asset_name="robot", resampling_time_range=(6.0, 6.0))
    assert cfg.class_type is CrabHexOmniCommand
    assert cfg.slot_probs == (0.15, 0.55, 0.10, 0.20)
    assert cfg.speed_ellipse == (0.35, 0.25, 0.25)
    assert cfg.facing_delta == (0.2, 0.6)
    assert cfg.travel_dir_range == (-math.pi, math.pi)
    assert cfg.ang_vel_max == 0.3
    assert cfg.dir_band is None
    # DEFAULT-OFF: the pool is opt-in, so every shipped preset takes the legacy sampler
    assert cfg.cmd_pool is None
    assert cfg.clips.ang_vel_clip == 0.2
    assert cfg.clips.lin_vel_clip == 0.2
    # the commands-cfg literal carries the same, on the omni class
    bv = omni.CrabHexOmniCommandsCfg.base_velocity
    assert isinstance(bv, CrabHexOmniCommandCfg)
    assert bv.asset_name == "robot"
    assert bv.resampling_time_range == (6.0, 6.0)
    assert bv.heading_control_stiffness == 0.8
    assert bv.ranges.lin_vel_x == (0.3, 0.8)
    assert bv.ranges.heading == (-1.6, 1.6)
    assert bv.clips.lin_vel_clip == 0.2
    assert bv.clips.ang_vel_clip == 0.2  # the one deliberate departure from the Go2 literals


def test_init_rejects_an_ellipse_that_does_not_clear_the_walk_floor():
    floor = 0.2 + command_sampling.WALK_MARGIN_M_S  # 0.21
    # exactly at floor + headroom -> rejected (the assertion is strict)
    with pytest.raises(ValueError, match="walk floor"):
        _term(_cfg(speed_ellipse=(0.35, floor + omni.MIN_CAP_HEADROOM_M_S, 0.25)), n=8)
    with pytest.raises(ValueError, match="walk floor"):
        _term(_cfg(speed_ellipse=(0.35, 0.25, 0.15)), n=8)
    # a hair above is accepted
    term, _ = _term(_cfg(speed_ellipse=(0.35, floor + omni.MIN_CAP_HEADROOM_M_S + 1e-6, 0.25)), n=8)
    assert term is not None


def test_init_rejects_other_unhonourable_cfgs():
    with pytest.raises(ValueError):
        _term(_cfg(slot_probs=(0.0, 0.0, 0.0, 0.0)), n=8)
    with pytest.raises(ValueError):
        _term(_cfg(slot_probs=(0.5, 0.5, 0.5)), n=8)
    with pytest.raises(ValueError):
        _term(_cfg(facing_delta=(0.6, 0.2)), n=8)
    with pytest.raises(ValueError):
        _term(_cfg(dir_band=(1.0, 0.0)), n=8)
    with pytest.raises(ValueError, match="ang_vel_max"):
        # cap at or below the dead band -> every yaw-rate command would be zeroed
        _term(_cfg(ang_vel_max=0.2), n=8)
    # a cmd_pool string the grammar does not accept must fail HERE, at construction -- and
    # for the GRAMMAR's reason: this cfg would also fail the slot-projection comparison, so
    # the match is on the bad type name rather than on "some ValueError".
    with pytest.raises(ValueError, match="sideways"):
        _term(_cfg(cmd_pool="fwd:0.5,sideways:0.5", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8)


def test_state_and_metric_names_are_the_ones_other_steps_read():
    term, _ = _term(_cfg(), n=8)
    for name in ("travel_dir_w", "speed_cmd", "heading_target", "slot_id", "rsi_cmd_staged"):
        assert hasattr(term, name), name
    assert term.rsi_cmd_staged.shape == (8, 3)
    assert bool(torch.isnan(term.rsi_cmd_staged).all())
    assert set(term.metrics) == {
        "error_vel_xy",
        "error_vel_yaw",
        "stand_frac_actual",
        "move_frac_actual",
        "turn_frac_actual",
        "move_turn_frac_actual",
        "facing_err_abs",
        "recap_frac",
        "yaw_progress",
    }
    assert callable(term.schedule_hook)


def test_slot_frequencies_match_slot_probs():
    probs = (0.15, 0.55, 0.10, 0.20)
    term, _ = _term(_cfg(slot_probs=probs), n=20_000, yaw=_spread_yaw(20_000))
    _resample(term)
    freq = torch.bincount(term.slot_id, minlength=4).double() / 20_000
    assert torch.allclose(freq, torch.tensor(probs, dtype=torch.float64), atol=0.02)
    # and the regime each slot promises
    moving = omni_math.slot_is_moving(term.slot_id)
    turning = omni_math.slot_is_turning(term.slot_id)
    assert bool((term.speed_cmd[~moving] == 0.0).all())
    assert bool((term.speed_cmd[moving] > 0.0).all())
    yaw = _spread_yaw(20_000)
    face_err = omni_math.wrap_to_pi(term.heading_target - yaw).abs()
    assert torch.equal(face_err[~turning], torch.zeros_like(face_err[~turning]))
    assert bool((face_err[turning] >= 0.2 - 1e-6).all())


def test_moving_slots_draw_inside_the_direction_scaled_band():
    term, _ = _term(_cfg(slot_probs=(0.0, 1.0, 0.0, 0.0)), n=20_000, yaw=_spread_yaw(20_000))
    _resample(term)
    yaw = _spread_yaw(20_000)
    phi0 = omni_math.wrap_to_pi(term.travel_dir_w - yaw)
    cap = omni_math.ellipse_cap(phi0, A, B, C)
    floor = omni.omni_walk_floor(term.cfg)
    assert bool((term.speed_cmd >= floor - 1e-7).all()), "a moving slot never sub-floors (the clock would stop)"
    assert bool((term.speed_cmd <= cap + 1e-7).all()), "and never exceeds the ellipse at its own direction"
    # the band really is direction-scaled: forward-ish slots reach above the lateral cap
    fwd = phi0.abs() < 0.2
    assert float(term.speed_cmd[fwd].max()) > B + 0.02


def test_dir_band_zero_commands_body_plus_x_exactly():
    term, _ = _term(
        _cfg(slot_probs=(0.0, 1.0, 0.0, 0.0), dir_band=(0.0, 0.0)), n=512, yaw=_spread_yaw(512)
    )
    _resample(term)
    term._update_command()
    cmd = term.vel_command_b
    assert torch.equal(cmd[:, 1], torch.zeros(512)), "body +x only: vy_b is exactly zero"
    assert bool((cmd[:, 0] > 0.0).all())
    assert bool((cmd[:, 0] <= A + 1e-6).all())


def test_dir_band_half_pi_commands_body_plus_y():
    """Stage 1b: body +y, i.e. the sideways task's left-leading convention."""
    term, _ = _term(
        _cfg(slot_probs=(0.0, 1.0, 0.0, 0.0), dir_band=(math.pi / 2, math.pi / 2)),
        n=512,
        yaw=_spread_yaw(512),
    )
    _resample(term)
    term._update_command()
    cmd = term.vel_command_b
    assert torch.allclose(cmd[:, 0], torch.zeros(512), atol=1e-6)
    assert bool((cmd[:, 1] > 0.0).all()), "+y is the robot's LEFT"
    assert bool((cmd[:, 1] <= B + 1e-6).all()), "and the lateral cap applies"


def test_turning_slots_sample_a_relative_facing_band_both_ways():
    lo, hi = 0.2, 1.2
    yaw = _spread_yaw(8192)
    term, _ = _term(_cfg(slot_probs=(0.0, 0.0, 1.0, 0.0), facing_delta=(lo, hi)), n=8192, yaw=yaw)
    _resample(term)
    delta = omni_math.wrap_to_pi(term.heading_target - yaw)
    assert float(delta.abs().min()) >= lo - 1e-6
    assert float(delta.abs().max()) <= hi + 1e-6
    assert 0.45 < float((delta > 0).double().mean()) < 0.55
    assert bool((term.speed_cmd == 0.0).all()), "a turn-only slot does not travel"


def test_non_turning_slots_hold_the_current_facing_exactly():
    yaw = _spread_yaw(512)
    term, _ = _term(_cfg(slot_probs=(0.5, 0.5, 0.0, 0.0)), n=512, yaw=yaw)
    _resample(term)
    assert torch.equal(term.heading_target, yaw)
    term._update_command()
    assert torch.equal(term.vel_command_b[:, 2], torch.zeros(512)), "zero facing error -> zero yaw-rate command"


def test_update_command_recaps_on_rotation_and_flags_it():
    """End-to-end twin of the pure re-cap test, through the term's own buffers."""
    term, robot = _term(_cfg(slot_probs=(0.0, 1.0, 0.0, 0.0)), n=1, yaw=torch.zeros(1))
    term.travel_dir_w[:] = 0.0  # world +x
    term.speed_cmd[:] = A  # the fore-aft cap
    term.heading_target[:] = 0.0
    term._update_command()
    assert term.vel_command_b[0, :2].tolist() == pytest.approx([A, 0.0], abs=1e-6)
    assert not bool(term.recap_mask[0])

    robot.data.heading_w[:] = math.pi / 2  # the body turns; the world command is held
    term._update_command()
    assert float(torch.linalg.norm(term.vel_command_b[0, :2])) == pytest.approx(B, abs=1e-6)
    assert term.vel_command_b[0, 1] < 0.0, "world +x is on the robot's right once it faces world +y"
    assert bool(term.recap_mask[0]), "recap_frac must see the clamp"
    assert float(term.phi_b[0]) == pytest.approx(-math.pi / 2, abs=1e-6)

    robot.data.heading_w[:] = 0.0  # back again: the clamp is not sticky
    term._update_command()
    assert term.vel_command_b[0, :2].tolist() == pytest.approx([A, 0.0], abs=1e-6)
    assert not bool(term.recap_mask[0])
    assert float(term.speed_cmd[0]) == pytest.approx(A), "speed_cmd itself is never mutated by the re-cap"


def test_yaw_rate_command_respects_ang_vel_max_and_the_dead_band():
    yaw = torch.zeros(4)
    term, _ = _term(_cfg(ang_vel_max=0.3), n=4, yaw=yaw)
    term.speed_cmd[:] = 0.0
    term.travel_dir_w[:] = 0.0
    term.heading_target[:] = torch.tensor([0.0, 0.12, 2.0, -2.0])
    term._update_command()
    wz = term.vel_command_b[:, 2]
    assert float(wz[0]) == 0.0
    assert float(wz[1]) == 0.0, "0.12 * 1.5 = 0.18 <= ang_vel_clip 0.2"
    assert float(wz[2]) == pytest.approx(0.3, abs=1e-6)
    assert float(wz[3]) == pytest.approx(-0.3, abs=1e-6)
    # the cap is a cfg field, not a constant
    term2, _ = _term(_cfg(ang_vel_max=0.5), n=4, yaw=yaw)
    term2.heading_target[:] = torch.tensor([0.0, 0.12, 2.0, -2.0])
    term2._update_command()
    assert float(term2.vel_command_b[2, 2]) == pytest.approx(0.5, abs=1e-6)


# ------------------------------------------------------------------------ RSI staging
def test_staged_rows_are_consumed_exactly_once_and_re_nanned():
    yaw = _spread_yaw(16)
    term, _ = _term(_cfg(slot_probs=(1.0, 0.0, 0.0, 0.0)), n=16, yaw=yaw)  # stand-only, so a
    # staged row is unmistakable: only staging can produce a moving slot here.
    staged_rows = torch.tensor([1, 4, 9])
    theta_s = torch.tensor([0.3, -1.1, 2.7])
    v_s = torch.tensor([0.22, 0.24, 0.23])
    psi_s = yaw[staged_rows].clone()
    term.rsi_cmd_staged[staged_rows] = torch.stack([theta_s, v_s, psi_s], dim=1)

    _resample(term)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[staged_rows] = True
    assert torch.equal(term.travel_dir_w[mask], theta_s)
    assert torch.equal(term.speed_cmd[mask], v_s)
    assert torch.equal(term.heading_target[mask], psi_s)
    assert torch.equal(term.slot_id[mask], torch.full((3,), S_MOVE)), "a staged row is a MOVE slot"
    assert bool((term.slot_id[~mask] == S_STAND).all())
    assert bool((term.speed_cmd[~mask] == 0.0).all())
    # consumed: the buffer is NaN again everywhere
    assert bool(torch.isnan(term.rsi_cmd_staged).all())

    # ... and a second resample does NOT replay it
    _resample(term, seed=SEED + 1)
    assert bool((term.speed_cmd == 0.0).all()), "the staged command must not survive its slot"
    assert bool((term.slot_id == S_STAND).all())


def test_partially_written_staged_rows_are_ignored():
    """Only fully-finite rows count; a half-written row falls back to the sampler."""
    term, _ = _term(_cfg(slot_probs=(1.0, 0.0, 0.0, 0.0)), n=8, yaw=torch.zeros(8))
    term.rsi_cmd_staged[3, 0] = 0.5  # theta only; v and psi still NaN
    _resample(term)
    assert float(term.speed_cmd[3]) == 0.0
    assert int(term.slot_id[3]) == S_STAND
    assert bool(torch.isnan(term.rsi_cmd_staged).all())


# ------------------------------------------------------ independence from inherited fields
@pytest.mark.parametrize(
    "overrides",
    [
        {"ranges.lin_vel_x": (2.0, 3.0)},
        {"ranges.heading": (-1.6, 1.6)},
        {"stand_frac": 0.9},
        {"small_commands_to_zero": False},
        {"ranges.lin_vel_x": (0.9, 0.95), "stand_frac": 0.0, "small_commands_to_zero": False},
    ],
)
def test_sampler_ignores_the_inherited_forward_fields(overrides):
    """``ranges.lin_vel_x`` / ``ranges.heading`` / ``stand_frac`` /
    ``small_commands_to_zero`` are kept only so the shared knob block keeps writing the
    same dump paths; nothing in the omni sampler may read them."""
    yaw = _spread_yaw(2048)
    base, _ = _term(_cfg(), n=2048, yaw=yaw)
    other, _ = _term(_cfg(**overrides), n=2048, yaw=yaw)
    _resample(base)
    _resample(other)
    for name in ("travel_dir_w", "speed_cmd", "heading_target", "slot_id"):
        assert torch.equal(getattr(base, name), getattr(other, name)), name
    base._update_command()
    other._update_command()
    assert torch.equal(base.vel_command_b, other.vel_command_b)


# ------------------------------------------------------------------------- metrics
def test_metrics_split_the_four_slots_and_track_facing():
    n = 4000
    yaw = _spread_yaw(n)
    term, robot = _term(_cfg(slot_probs=(0.25, 0.25, 0.25, 0.25)), n=n, yaw=yaw)
    _resample(term)
    term._update_command()
    term._update_metrics()
    per_step = 1.0 / EPISODE_STEPS
    fracs = torch.tensor(
        [
            float(term.metrics[k].sum()) / per_step
            for k in ("stand_frac_actual", "move_frac_actual", "turn_frac_actual", "move_turn_frac_actual")
        ]
    )
    assert float(fracs.sum()) == pytest.approx(n), "every env is in exactly one slot"
    assert torch.allclose(fracs / n, torch.full((4,), 0.25), atol=0.02)
    # facing error is the wrapped absolute error against the CURRENT yaw
    expected = omni_math.wrap_to_pi(term.heading_target - robot.data.heading_w).abs() * per_step
    assert torch.allclose(term.metrics["facing_err_abs"], expected, rtol=0.0, atol=1e-7)


def test_a_pure_lateral_command_is_not_counted_as_standing():
    """The base class's ``stand_frac_actual`` rule (``vel_command_b[:, 0] == 0``) would call
    a full-speed sideways command a stop; the omni metric is slot-based."""
    term, _ = _term(_cfg(), n=4, yaw=torch.zeros(4))
    term.travel_dir_w[:] = math.pi / 2  # body +y at yaw 0
    term.speed_cmd[:] = B
    term.slot_id[:] = S_MOVE
    term._update_command()
    term._update_metrics()
    assert torch.allclose(term.vel_command_b[:, 0], torch.zeros(4), atol=1e-7)
    assert bool((term.vel_command_b[:, 1] > 0.2).all())
    assert bool((term.metrics["stand_frac_actual"] == 0.0).all())
    assert bool((term.metrics["move_frac_actual"] > 0.0).all())


def test_error_vel_xy_is_the_two_dimensional_command_error():
    term, robot = _term(_cfg(), n=4, yaw=torch.zeros(4))
    term.travel_dir_w[:] = math.pi / 2  # world +y == body +y at yaw 0
    term.speed_cmd[:] = 0.22
    term.heading_target[:] = 0.0
    term._update_command()
    robot.data.root_lin_vel_b[:, 1] = 0.22  # tracking it perfectly, sideways
    term._update_metrics()
    assert torch.allclose(term.metrics["error_vel_xy"], torch.zeros(4), atol=1e-7)
    # ... and a forward velocity against a lateral command is an error, not a success
    term2, robot2 = _term(_cfg(), n=4, yaw=torch.zeros(4))
    term2.travel_dir_w[:] = math.pi / 2
    term2.speed_cmd[:] = 0.22
    term2._update_command()
    robot2.data.root_lin_vel_b[:, 0] = 0.22
    term2._update_metrics()
    assert float(term2.metrics["error_vel_xy"].min()) > 0.0


def test_yaw_progress_metric_is_the_signed_mean_over_the_turning_rows():
    """``Metrics/base_velocity/yaw_progress``: mean ``wz_b * sign(cmd_wz)`` where a turn is
    commanded, 0 when none -- the signed number the failed 2026-09-17 yaw smoke lacked.

    Rows 0/1 are commanded LEFT, rows 2/3 RIGHT, rows 4/5 not at all (facing already inside
    the dead band). Progress must be positive for a body rate in the commanded direction
    whichever way that is, and the standing rows must not dilute the average.
    """
    n = 6
    term, robot = _term(_cfg(), n=n, yaw=torch.zeros(n))
    term.heading_target[:] = torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.05])
    term._update_command()
    cmd_wz = term.vel_command_b[:, 2]
    assert torch.equal(torch.sign(cmd_wz), torch.tensor([1.0, 1.0, -1.0, -1.0, 0.0, 0.0]))

    # +0.2 rad/s: progress on the left-turn rows, the same magnitude AGAINST the right ones
    robot.data.root_ang_vel_b[:, 2] = 0.2
    assert float(term._yaw_progress_step()) == pytest.approx(0.0, abs=1e-7)
    # every commanded row turning the way it was told
    robot.data.root_ang_vel_b[:, 2] = torch.tensor([0.2, 0.1, -0.2, -0.1, 5.0, -5.0])
    assert float(term._yaw_progress_step()) == pytest.approx(0.15, abs=1e-6)
    # the drift the smoke measured: saturated command, small net rotation the WRONG way
    robot.data.root_ang_vel_b[:, 2] = torch.tensor([-0.025, -0.025, 0.025, 0.025, 0.0, 0.0])
    assert float(term._yaw_progress_step()) == pytest.approx(-0.025, abs=1e-7)

    # accumulated like its siblings: an episode mean, broadcast to every row
    term._update_metrics()
    expected = -0.025 / EPISODE_STEPS
    assert torch.allclose(term.metrics["yaw_progress"], torch.full((n,), expected), atol=1e-9)

    # no turn commanded anywhere -> exactly 0, never a division by zero
    calm, calm_robot = _term(_cfg(), n=4, yaw=torch.zeros(4))
    calm.heading_target[:] = 0.0
    calm.speed_cmd[:] = 0.25
    calm._update_command()
    calm_robot.data.root_ang_vel_b[:, 2] = torch.tensor([0.3, -0.3, 0.1, -0.1])
    assert float(calm._yaw_progress_step()) == 0.0
    calm._update_metrics()
    assert torch.equal(calm.metrics["yaw_progress"], torch.zeros(4))


# -------------------------------------------------------------------- the eval harness
@pytest.mark.parametrize("as_mapping", [True, False])
def test_schedule_hook_pins_a_world_hold_row(as_mapping):
    yaw = torch.full((6,), math.pi / 2)
    term, _ = _term(_cfg(), n=6, yaw=yaw)
    row = {"speed": 0.22, "dir": 0.0, "face": math.pi / 2}
    term.schedule_hook(row if as_mapping else types.SimpleNamespace(**row))
    assert bool((term.travel_dir_w == 0.0).all())
    assert bool((term.speed_cmd == 0.22).all())
    assert bool((term.heading_target == math.pi / 2).all())
    # world +x, facing world +y -> body -y, and no turning (facing error 0)
    assert torch.allclose(term.vel_command_b[:, 0], torch.zeros(6), atol=1e-6)
    assert torch.allclose(term.vel_command_b[:, 1], torch.full((6,), -0.22), atol=1e-6)
    assert torch.equal(term.vel_command_b[:, 2], torch.zeros(6))
    assert bool((term.slot_id == S_MOVE).all())


def test_schedule_hook_turn_hold_reports_a_turning_slot_and_does_not_eat_staging():
    term, _ = _term(_cfg(), n=4, yaw=torch.zeros(4))
    term.rsi_cmd_staged[1] = torch.tensor([0.1, 0.22, 0.0])
    term.schedule_hook({"speed": 0.0, "dir": 0.0, "face": math.pi / 2})
    assert float(term.vel_command_b[0, 2]) == pytest.approx(0.3, abs=1e-6)
    assert bool((term.slot_id == S_TURN).all())
    assert bool((term.vel_command_b[:, :2] == 0.0).all())
    assert bool(torch.isfinite(term.rsi_cmd_staged[1]).all()), "eval must not consume RSI staging"
    # a move+turn hold reports slot 3
    term.schedule_hook({"speed": 0.22, "dir": 0.0, "face": math.pi / 2})
    assert bool((term.slot_id == S_MOVE_TURN).all())
    # once the facing error settles inside the dead band the hold reports move only
    term.heading_target[:] = 0.0
    term.schedule_hook({"speed": 0.22, "dir": 0.0, "face": 0.0})
    assert bool((term.slot_id == S_MOVE).all())


def test_schedule_hook_yaw_rate_is_discontinuous_across_one_control_step():
    """One hold row, one control step of yaw drift -> the yaw-rate command can JUMP.

    This is why the eval harness must read the command at the instant the observation was
    built instead of comparing an observation against a command it re-derived at a different
    robot state: the world->body projection is smooth in yaw, but slot 2 is not.

    * ``wrap_to_pi`` flips a SATURATED command between ``+-ang_vel_max`` when the facing
      error passes ``+-pi`` -- a jump of ``2 * ang_vel_max`` (0.6 with the shipped cap).
      Measured in the first omni harness smoke: hold ``turn_l90`` (face ``+pi/2``) with a
      fallen robot whose yaw crossed ``-pi/2``.
    * the dead band flips it between 0 and ``+-ang_vel_clip / 1`` at
      ``|facing error| = ang_vel_clip / stiffness`` -- a jump of ~``ang_vel_clip``.
    """
    cfg = _cfg()
    ang_vel_max = float(cfg.ang_vel_max)
    ang_vel_clip = float(cfg.clips.ang_vel_clip)
    face = math.pi / 2
    row = {"speed": 0.0, "dir": 0.0, "face": face}

    # (a) the +-pi wrap: yaw one control step either side of face - pi.
    eps = 0.0018  # the step measured in the smoke (~0.09 rad/s of body yaw rate)
    wz = []
    for yaw in (-math.pi / 2 - eps, -math.pi / 2 + eps):
        term, _ = _term(cfg, n=2, yaw=torch.full((2,), yaw))
        term.schedule_hook(row)
        wz.append(float(term.vel_command_b[0, 2]))
    assert wz[0] == pytest.approx(-ang_vel_max, abs=1e-6)
    assert wz[1] == pytest.approx(+ang_vel_max, abs=1e-6)
    assert abs(wz[1] - wz[0]) == pytest.approx(2.0 * ang_vel_max, abs=1e-6)

    # (b) the dead-band edge: |err| = ang_vel_clip / stiffness, crossed in one step.
    edge = ang_vel_clip / STIFFNESS
    wz = []
    for err in (edge - 1e-4, edge + 1e-4):
        term, _ = _term(cfg, n=2, yaw=torch.full((2,), face - err))
        term.schedule_hook(row)
        wz.append(float(term.vel_command_b[0, 2]))
    assert wz[0] == 0.0
    assert wz[1] == pytest.approx(ang_vel_clip, abs=1e-3) and wz[1] > ang_vel_clip
    # ...and the xy channels stay smooth over the same drift, so only slot 2 needs the care.
    moving = {"speed": 0.22, "dir": 0.0, "face": face}
    cmds = []
    for yaw in (0.0, eps):
        term, _ = _term(cfg, n=2, yaw=torch.full((2,), yaw))
        term.schedule_hook(moving)
        cmds.append(term.vel_command_b[0, :2].clone())
    assert float((cmds[1] - cmds[0]).abs().max()) < 1e-3


def test_schedule_hook_restores_the_scheduled_row_after_a_reset_resample():
    """The invariant the harness's ``schedule_state_max_deviation`` asserts.

    ``CommandTerm.reset()`` calls ``_resample_command`` on every env that fell, which draws a
    RANDOM direction / speed / facing. The env then runs ``command_manager.compute()`` (the
    harness's override -> ``schedule_hook``) before the observation manager, so the scheduled
    row must be back in force -- for EVERY env, not only the ones that did not reset -- and
    the RSI staging buffer must be untouched by the eval path.
    """
    yaw = _spread_yaw(64)
    term, _ = _term(_cfg(), n=64, yaw=yaw)
    row = {"speed": 0.22, "dir": math.pi / 4, "face": -math.pi / 3}
    term.schedule_hook(row)

    # a subset "falls": reset() -> _resample_command on those rows only.
    fell = torch.arange(0, 64, 3)
    torch.manual_seed(SEED)
    term._resample_command(fell)
    assert float((term.travel_dir_w[fell] - row["dir"]).abs().max()) > 1e-3, "resample was inert"

    term.schedule_hook(row)
    assert float((term.travel_dir_w - row["dir"]).abs().max()) == 0.0
    assert float((term.speed_cmd - row["speed"]).abs().max()) == 0.0
    assert float((term.heading_target - row["face"]).abs().max()) == 0.0
    assert bool(torch.isnan(term.rsi_cmd_staged).all()), "eval must not stage RSI rows"


def test_schedule_hook_rejects_nonsense_rows():
    term, _ = _term(_cfg(), n=2, yaw=torch.zeros(2))
    with pytest.raises(ValueError):
        term.schedule_hook({"speed": -0.1, "dir": 0.0, "face": 0.0})
    with pytest.raises(ValueError):
        term.schedule_hook({"speed": 0.2, "dir": float("nan"), "face": 0.0})
    with pytest.raises(KeyError):
        term.schedule_hook({"speed": 0.2, "dir": 0.0})


# --------------------------------------------------------------- the forward term is untouched
def test_forward_command_class_and_cfg_have_none_of_the_omni_surface():
    omni_only = (
        "slot_probs",
        "speed_ellipse",
        "facing_delta",
        "travel_dir_range",
        "ang_vel_max",
        "dir_band",
        "cmd_pool",
    )
    for name in omni_only:
        assert not hasattr(pcfg.ParkourCommandCfg, name), f"ParkourCommandCfg gained {name}"
        assert hasattr(CrabHexOmniCommandCfg, name), name
    for name in ("rsi_cmd_staged", "schedule_hook", "travel_dir_w", "speed_cmd", "slot_id",
                 "type_id", "recap_mask"):
        assert not hasattr(upc.UniformParkourCommand, name), f"UniformParkourCommand gained {name}"
    # the omni term subclasses it but overrides the three implementation hooks
    assert issubclass(CrabHexOmniCommand, upc.UniformParkourCommand)
    for hook in ("_resample_command", "_update_command", "_update_metrics"):
        assert getattr(CrabHexOmniCommand, hook) is not getattr(upc.UniformParkourCommand, hook), hook
    # and the forward cfg still has the inherited fields the omni cfg keeps but ignores
    for name in ("stand_frac", "small_commands_to_zero"):
        assert hasattr(pcfg.ParkourCommandCfg, name)
        assert hasattr(CrabHexOmniCommandCfg, name)


# --------------------------------------------------------------------------- debug vis
# The markers themselves need Kit, so what is pinned here is (a) the pure arrow helpers
# against exact quaternion/scale values, and (b) that ``_debug_vis_callback`` feeds each
# visualiser the pose the colour table promises. A silently wrong arrow (a body-frame
# facing, a travel arrow pointing at the nose) is the exact failure the viewer exists to
# rule out, and it is invisible in a screenshot unless you already know the answer.

SQRT_HALF = math.sqrt(0.5)


class _FakeVisualizer:
    """Records the last ``visualize`` call; ``cfg`` mimics ``VisualizationMarkers.cfg``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.visible = None
        self.last = None

    def set_visibility(self, flag):
        self.visible = flag

    def visualize(self, translations=None, orientations=None, scales=None, marker_indices=None):
        self.last = (translations, orientations, scales)


def _attach_fake_visualizers(made):
    """Wire the three omni visualisers without Kit, exactly as ``_set_debug_vis_impl`` does."""
    term = made[0] if isinstance(made, tuple) else made
    term.travel_cmd_visualizer = _FakeVisualizer(omni.OMNI_TRAVEL_CMD_MARKER_CFG)
    term.facing_cmd_visualizer = _FakeVisualizer(omni.OMNI_FACING_CMD_MARKER_CFG)
    term.achieved_vel_visualizer = _FakeVisualizer(omni.OMNI_ACHIEVED_VEL_MARKER_CFG)
    return term


@pytest.mark.parametrize(
    "yaw, expected",
    [
        (0.0, (1.0, 0.0, 0.0, 0.0)),
        (math.pi / 2, (SQRT_HALF, 0.0, 0.0, SQRT_HALF)),
        (math.pi, (0.0, 0.0, 0.0, 1.0)),
        (-math.pi / 2, (SQRT_HALF, 0.0, 0.0, -SQRT_HALF)),
    ],
)
def test_arrow_quat_from_world_yaw_exact_values(yaw, expected):
    """wxyz, rotation about +z only -- the arrow's +x lands on the world yaw."""
    quat = omni.arrow_quat_from_world_yaw(torch.tensor([yaw]))
    assert quat.shape == (1, 4)
    assert torch.allclose(quat[0], torch.tensor(expected), atol=1e-6)


def test_arrow_quat_rotates_the_arrow_axis_onto_the_commanded_direction():
    """The property the exact values stand for: R(q) @ +x == (cos yaw, sin yaw, 0)."""
    yaw = torch.tensor([0.0, math.pi / 2, math.pi, -0.7, 2.9])
    quat = omni.arrow_quat_from_world_yaw(yaw)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # first column of the rotation matrix = image of the body +x axis
    axis = torch.stack(
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1
    )
    expected = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)
    assert torch.allclose(axis, expected, atol=1e-6)


def test_arrow_quat_is_yaw_only_and_batched():
    """No roll/pitch leaks in (the x/y components stay exactly zero) for every row."""
    yaw = torch.linspace(-math.pi, math.pi, 17)
    quat = omni.arrow_quat_from_world_yaw(yaw)
    assert quat.shape == (17, 4)
    assert torch.all(quat[:, 1] == 0.0) and torch.all(quat[:, 2] == 0.0)
    assert torch.allclose(torch.linalg.norm(quat, dim=1), torch.ones(17), atol=1e-6)


def test_arrow_scale_from_length_stretches_only_x():
    proto = (0.5, 0.35, 0.35)
    factor = torch.tensor([0.0, 1.0, 2.0])
    scale = omni.arrow_scale_from_length(proto, factor)
    assert scale.shape == (3, 3)
    assert torch.allclose(scale[:, 0], torch.tensor([0.0, 0.5, 1.0]))
    # the shaft keeps its thickness on every row
    assert torch.allclose(scale[:, 1], torch.full((3,), 0.35))
    assert torch.allclose(scale[:, 2], torch.full((3,), 0.35))


def test_arrow_scale_from_speed_is_proportional_and_zero_at_a_stop():
    """x scale 1.0 on the prototype, so the factor IS the marker x-scale; 0 m/s -> 0."""
    speed = torch.tensor([0.0, 0.2, 0.35])
    proto = omni.OMNI_TRAVEL_CMD_MARKER_CFG.markers["arrow"].scale
    assert proto[0] == 1.0
    scale = omni.arrow_scale_from_length(proto, speed * omni.TRAVEL_ARROW_GAIN)
    assert torch.allclose(scale[:, 0], speed * omni.TRAVEL_ARROW_GAIN, atol=1e-6)
    assert scale[0, 0] == 0.0
    assert scale[2, 0] > scale[1, 0] > 0.0


def test_speed_arrows_are_longer_than_they_are_thick_at_the_slowest_walk():
    """The first render of this viewer was three blobs: a 0.5 shaft under a 0.6 length.

    The walk floor (``lin_vel_clip + WALK_MARGIN_M_S`` = 0.21 m/s) is the slowest a moving
    slot can ever be, so it is where the aspect ratio is worst.
    """
    slowest = omni.omni_walk_floor(_cfg())
    length = slowest * omni.TRAVEL_ARROW_GAIN
    assert length / omni.ARROW_SHAFT_SCALE > 4.0, "the travel arrow is a blob again"
    assert omni.FACING_ARROW_LEN_FACTOR / omni.FACING_SHAFT_SCALE > 4.0
    assert omni.FACING_SHAFT_SCALE < omni.ARROW_SHAFT_SCALE, "facing reads thinner"


def test_facing_arrow_is_longer_than_any_travel_arrow_can_be():
    """"the long orange one is the facing" must survive the fastest command in the ellipse."""
    cap = max(CrabHexOmniCommandCfg.speed_ellipse)
    assert omni.FACING_ARROW_LEN_FACTOR > cap * omni.TRAVEL_ARROW_GAIN


def test_marker_pos_above_base_raises_z_and_does_not_alias():
    root = torch.tensor([[1.0, 2.0, 0.3], [-4.0, 0.5, 1.25]])
    pos = omni.marker_pos_above_base(root, omni.ARROW_Z_FACING_M)
    assert torch.allclose(pos[:, :2], root[:, :2])
    assert torch.allclose(pos[:, 2], root[:, 2] + omni.ARROW_Z_FACING_M)
    pos[:, 2] = -99.0
    assert torch.allclose(root[:, 2], torch.tensor([0.3, 1.25])), "clone(), not a view"


def test_the_three_arrow_heights_are_distinct_and_ordered():
    """Facing above travel above achieved -- they overlap in xy every single frame."""
    assert omni.ARROW_Z_ACHIEVED_M < omni.ARROW_Z_TRAVEL_M < omni.ARROW_Z_FACING_M


def test_marker_cfgs_are_private_copies_with_the_documented_colours():
    """The deep copy: recolouring must not leak into the shared stock arrow prototype."""
    cfgs = (
        omni.OMNI_TRAVEL_CMD_MARKER_CFG,
        omni.OMNI_FACING_CMD_MARKER_CFG,
        omni.OMNI_ACHIEVED_VEL_MARKER_CFG,
    )
    colours = [cfg.markers["arrow"].visual_material.diffuse_color for cfg in cfgs]
    assert colours == [omni.TRAVEL_ARROW_COLOR, omni.FACING_ARROW_COLOR, omni.ACHIEVED_ARROW_COLOR]
    assert len({id(cfg.markers["arrow"]) for cfg in cfgs}) == 3, "three separate prototypes"
    stock = sys.modules["isaaclab.markers.config"].GREEN_ARROW_X_MARKER_CFG
    assert stock.markers["arrow"].visual_material.diffuse_color == (0.0, 1.0, 0.0)
    assert len({cfg.prim_path for cfg in cfgs}) == 3, "three separate prim paths"


def test_debug_vis_callback_draws_world_travel_facing_and_achieved():
    """Travel along theta, facing along psi, achieved along the WORLD root velocity.

    The robot is yawed 90 deg with a command that is 90 deg to its RIGHT in the world, so a
    body/world mix-up in any of the three arrows moves it by a right angle rather than by a
    rounding error.
    """
    yaw = torch.tensor([math.pi / 2, math.pi / 2])
    term = _attach_fake_visualizers(_term(_cfg(), n=2, yaw=yaw))
    theta = torch.tensor([0.0, 0.0])  # world +x = the robot's right
    psi = torch.tensor([-math.pi / 2, math.pi])
    term.travel_dir_w[:] = theta
    term.speed_cmd[:] = torch.tensor([0.3, 0.0])  # second row stands
    term.heading_target[:] = psi
    term._write_command(slice(None))
    term.robot.data.root_pos_w[:] = torch.tensor([[1.0, 2.0, 0.4], [3.0, -1.0, 0.4]])
    term.robot.data.root_lin_vel_w[:, :2] = torch.tensor([[0.0, 0.1], [0.0, 0.0]])

    term._debug_vis_callback(None)

    travel_pos, travel_quat, travel_scale = term.travel_cmd_visualizer.last
    facing_pos, facing_quat, facing_scale = term.facing_cmd_visualizer.last
    ach_pos, ach_quat, ach_scale = term.achieved_vel_visualizer.last

    # -- every marker is over its own robot at its own height
    for pos, dz in ((travel_pos, omni.ARROW_Z_TRAVEL_M),
                    (facing_pos, omni.ARROW_Z_FACING_M),
                    (ach_pos, omni.ARROW_Z_ACHIEVED_M)):
        assert pos.shape == (2, 3)
        assert torch.allclose(pos[:, :2], term.robot.data.root_pos_w[:, :2])
        assert torch.allclose(pos[:, 2], term.robot.data.root_pos_w[:, 2] + dz)

    # -- travel: WORLD theta (not the body-frame command, which points at the robot's -y)
    assert torch.allclose(travel_quat, omni.arrow_quat_from_world_yaw(theta), atol=1e-6)
    # length is the EFFECTIVE commanded speed; the standing row is a zero-length arrow
    assert torch.allclose(
        travel_scale[:, 0],
        torch.linalg.norm(term.vel_command_b[:, :2], dim=1) * omni.TRAVEL_ARROW_GAIN,
        atol=1e-6,
    )
    assert travel_scale[1, 0] == 0.0

    # -- facing: the ABSOLUTE world yaw, fixed length on both rows
    assert torch.allclose(facing_quat, omni.arrow_quat_from_world_yaw(psi), atol=1e-6)
    assert torch.allclose(facing_scale[:, 0], torch.full((2,), omni.FACING_ARROW_LEN_FACTOR))

    # -- achieved: world +y at 0.1 m/s on row 0, nothing on row 1
    assert torch.allclose(ach_quat[0], omni.arrow_quat_from_world_yaw(torch.tensor([math.pi / 2]))[0], atol=1e-6)
    assert torch.allclose(ach_scale[:, 0], torch.tensor([0.1 * omni.TRAVEL_ARROW_GAIN, 0.0]), atol=1e-6)


def test_debug_vis_callback_is_a_noop_before_the_articulation_initialises():
    term = _attach_fake_visualizers(_term(_cfg(), n=2))
    term.robot.is_initialized = False
    term._debug_vis_callback(None)
    assert term.travel_cmd_visualizer.last is None


def test_omni_term_replaces_both_base_debug_vis_hooks_and_leaves_the_cfg_alone():
    """The green goal arrow is replaced, not kept -- and no cfg field was added for it."""
    for hook in ("_set_debug_vis_impl", "_debug_vis_callback"):
        assert getattr(CrabHexOmniCommand, hook) is not getattr(upc.UniformParkourCommand, hook), hook
    # the marker cfgs live at module scope precisely so the config dump does not move
    assert set(CrabHexOmniCommandCfg.__cfg_fields__) == set(pcfg.ParkourCommandCfg.__cfg_fields__) | {
        "slot_probs", "speed_ellipse", "facing_delta", "travel_dir_range", "ang_vel_max", "dir_band",
        # 2026-09-18: the seventh and (per the plan) last sampler field -- the command pool's
        # canonical string. It widens the omni-vs-forward config-dump delta by exactly one
        # entry, ``cmd_pool: null``, on every shipped window.
        "cmd_pool",
    }


# ============================================================ the per-type command pool
#
# The pool is an ADDITIVE lever: with ``cmd_pool`` unset the term must be the shipped sampler
# down to the last float and the last RNG draw (every recorded seed-3 run depends on it), and
# with a pool set it must be able to reproduce a legacy distribution exactly. The two claims
# are pinned in opposite directions below -- "the default takes the legacy path" and "these
# three pools are bitwise twins of these three legacy configs".

SLOTS_TEACHER = (0.15, 0.45, 0.15, 0.25)
POOL_1C = "stand:0.15,omni:0.55,turn:0.10,omni_turn:0.20"


def _pool_term(pool: str, slots, **overrides):
    """A term on ``pool`` with the matching slot projection, over the standard yaw spread."""
    n = overrides.pop("n", 4096)
    yaw = overrides.pop("yaw", None)
    if yaw is None:
        yaw = _spread_yaw(n)
    return _term(_cfg(cmd_pool=pool, slot_probs=slots, **overrides), n=n, yaw=yaw)


def _state(term):
    """The five tensors a twin has to match, cloned."""
    return {
        name: getattr(term, name).clone()
        for name in ("slot_id", "travel_dir_w", "speed_cmd", "heading_target", "vel_command_b")
    }


# ------------------------------------------------------------------- the pool type table
def test_pool_types_project_onto_the_two_bit_slot():
    """Every type's slot is exactly ``moving + 2 * turning`` for what its name promises.

    The projection is the whole reason a pool can be added without touching a single metric,
    reward, eval hold or piece of telemetry: they all keep reading ``slot_id``.
    """
    travelling = {"fwd", "side_l", "side_r", "back", "omni"}
    turning = {"turn", "fwd_turn", "side_l_turn", "side_r_turn", "back_turn", "omni_turn"}
    assert set(omni_math.POOL_TYPES) == {"stand"} | travelling | turning
    for name, spec in omni_math.POOL_TYPES.items():
        moves = name in travelling or (name.endswith("_turn") and name != "turn")
        turns = name in turning
        assert spec.slot == int(moves) + 2 * int(turns), name
        assert bool(omni_math.slot_is_moving(torch.tensor(spec.slot))) == moves, name
        assert bool(omni_math.slot_is_turning(torch.tensor(spec.slot))) == turns, name
    # the four slots are all reachable, and every travel type has a turning twin
    assert {s.slot for s in omni_math.POOL_TYPES.values()} == {0, 1, 2, 3}
    for name in travelling:
        assert f"{name}_turn" in omni_math.POOL_TYPES, name


def test_pool_type_bands_are_the_exact_body_axes():
    """``fwd``/``side_l``/``side_r``/``back`` are the zero-width literals the legacy
    ``dir_band`` uses, and only the two ``omni`` types draw world-uniform."""
    axes = {"fwd": 0.0, "side_l": math.pi / 2, "side_r": -math.pi / 2, "back": math.pi}
    for name, angle in axes.items():
        for full in (name, f"{name}_turn"):
            spec = omni_math.POOL_TYPES[full]
            assert spec.world is False, full
            assert spec.band == (angle, angle), full
    for name in ("omni", "omni_turn"):
        assert omni_math.POOL_TYPES[name].world is True
        assert omni_math.POOL_TYPES[name].band is None
    # the non-travelling types carry a zero-width body band (their speed is exactly 0, so the
    # held direction is unobservable; pinning it to the nose beats a random world ray)
    for name in ("stand", "turn"):
        assert omni_math.POOL_TYPES[name] == omni_math.PoolType(
            omni_math.POOL_TYPES[name].slot, False, (0.0, 0.0)
        )


def test_pool_floor_share_is_the_planned_constant():
    """The campaign's recipe validator imports this rather than re-spelling 0.025."""
    assert omni_math.POOL_FLOOR_SHARE == 0.025


# --------------------------------------------------------------------- grammar and render
def test_parse_cmd_pool_grammar():
    pool = omni_math.parse_cmd_pool("stand:0.2,fwd:0.8")
    assert pool.names == ("stand", "fwd")
    assert pool.weights == (0.2, 0.8)
    assert pool.ang_scale == (1.0, 1.0)
    assert pool.speed_scale == (1.0, 1.0)
    # whitespace is tolerated, and the optional scales are positional
    scaled = omni_math.parse_cmd_pool(" fwd : 0.6 : 0.3 , turn : 0.4 : 1.0 : 0.5 ")
    assert scaled.names == ("fwd", "turn")
    assert scaled.ang_scale == (0.3, 1.0)
    assert scaled.speed_scale == (1.0, 0.5)
    # a zero weight is LEGAL (a type kept in the list at no mass; see rescale_to)
    zeroed = omni_math.parse_cmd_pool("stand:0.0,fwd:1.0")
    assert zeroed.weights == (0.0, 1.0)
    # the order is the string's own, not POOL_TYPES'
    assert omni_math.parse_cmd_pool("back:0.5,fwd:0.5").names == ("back", "fwd")


@pytest.mark.parametrize(
    "bad",
    [
        "",                       # empty
        "   ",                    # blank
        "sideways:1.0",           # unknown type name
        "fwd:0.5,fwd:0.5",        # duplicate
        "stand:0.0,fwd:0.0",      # all-zero weights
        "fwd:-0.1,stand:1.0",     # negative weight
        "fwd:1.0:0",              # zero ang_scale
        "fwd:1.0:-1",             # negative ang_scale
        "fwd:1.0:1.0:0",          # zero speed_scale
        "fwd:1.0:1.0:1.0:1.0",    # too many fields
        "fwd",                    # no weight
        "fwd:",                   # empty weight field
        "fwd:1.0,",               # stray comma
        "fwd:nan",                # non-finite
        "fwd:abc",                # non-numeric
    ],
)
def test_parse_cmd_pool_rejections(bad):
    with pytest.raises(ValueError):
        omni_math.parse_cmd_pool(bad)


def test_render_cmd_pool_shortest_faithful_form():
    assert omni_math.render_cmd_pool(omni_math.parse_cmd_pool("stand:0.2,fwd:0.8")) == "stand:0.2,fwd:0.8"
    # only ang_scale differs -> three fields; speed_scale differs -> all four
    assert omni_math.render_cmd_pool(omni_math.parse_cmd_pool("fwd:1.0:0.3")) == "fwd:1.0:0.3"
    assert omni_math.render_cmd_pool(omni_math.parse_cmd_pool("fwd:1.0:1.0:0.5")) == "fwd:1.0:1.0:0.5"


def test_render_parse_round_trip_is_exact():
    """``repr`` precision, never fixed decimals: the driver renders a composed pool to ONE
    string per stage and records it as the truth, so a lossy render would leave the recorded
    pool disagreeing with the ``SLOTS`` projection the same recipe computed."""
    pool = omni_math.parse_cmd_pool(POOL_1C).rescale_to(SLOTS_TEACHER)
    text = omni_math.render_cmd_pool(pool)
    back = omni_math.parse_cmd_pool(text)
    assert back.names == pool.names
    assert back.weights == pool.weights, "repr() round-trips a float exactly"
    assert back.ang_scale == pool.ang_scale and back.speed_scale == pool.speed_scale
    assert omni_math.render_cmd_pool(back) == text
    # The two tolerances the plan names, asserted where they can actually BITE: a third of a
    # share (and c6's 0.035 / 0.025) is exactly what a ``%.3f`` render would round off. On the
    # rescaled pool above they would be vacuous -- ``back.weights == pool.weights`` already
    # holds there, which makes any downstream comparison exact by construction.
    thirds = omni_math.parse_cmd_pool(
        "fwd:0.3333333333333333,side_l:0.6666666666666666,"
        "side_l_turn:0.035,side_r_turn:0.025,turn:0.1"
    )
    thirds_back = omni_math.parse_cmd_pool(omni_math.render_cmd_pool(thirds))
    assert thirds_back.weights == thirds.weights
    for got, want in zip(thirds_back.projected_slot_probs(), thirds.projected_slot_probs()):
        assert abs(got - want) < 1e-6
    assert abs(sum(thirds_back.weights) - sum(thirds.weights)) < 1e-9


# ------------------------------------------------------------------ projection and rescale
def test_projected_slot_probs_sums_the_weights_per_slot():
    pool = omni_math.parse_cmd_pool(
        "stand:0.15,fwd:0.25,side_l:0.15,side_r:0.15,turn:0.10,fwd_turn:0.20"
    )
    assert pool.projected_slot_probs() == pytest.approx((0.15, 0.55, 0.10, 0.20), abs=1e-12)
    # un-normalised weights normalise, exactly like slot_probs
    doubled = omni_math.parse_cmd_pool("stand:0.3,fwd:1.1,turn:0.2,fwd_turn:0.4")
    assert doubled.projected_slot_probs() == pytest.approx((0.15, 0.55, 0.10, 0.20), abs=1e-12)
    assert pool.slot_of("side_r") == S_MOVE and pool.slot_of("omni_turn") == S_MOVE_TURN
    with pytest.raises(ValueError):
        pool.slot_of("sideways")


def test_rescale_to_round_trips_the_teacher_slots():
    """The teacher windows are described by a slot mix; this is how a curriculum's final pool
    is carried into them with its shape (and its scales) intact."""
    pool = omni_math.parse_cmd_pool(POOL_1C)
    rescaled = pool.rescale_to(SLOTS_TEACHER)
    assert rescaled.names == pool.names
    for got, want in zip(rescaled.projected_slot_probs(), SLOTS_TEACHER):
        assert abs(got - want) < 1e-6
    assert abs(sum(rescaled.weights) - 1.0) < 1e-9
    # WITHIN-slot proportions survive: the move_turn slot's two types keep their ratio
    mixed = omni_math.parse_cmd_pool("fwd_turn:0.06,side_l_turn:0.02,fwd:1.0")
    out = mixed.rescale_to((0.0, 0.5, 0.0, 0.5))
    assert out.weights[0] / out.weights[1] == pytest.approx(3.0)
    # and so do the per-type scales
    scaled = omni_math.parse_cmd_pool("fwd:0.5:0.3,turn:0.5:1.0:0.7")
    assert scaled.rescale_to((0.0, 0.8, 0.2, 0.0)).ang_scale == (0.3, 1.0)
    assert scaled.rescale_to((0.0, 0.8, 0.2, 0.0)).speed_scale == (1.0, 0.7)


def test_rescale_to_keeps_zeroed_types_and_refuses_an_unreachable_slot():
    pool = omni_math.parse_cmd_pool(POOL_1C)
    # a slot given zero mass keeps its types at weight 0.0 -- never dropped, so the render
    # round trip and the pool_<name>_* metric names are stable along a chain of stages
    zeroed = pool.rescale_to((0.2, 0.8, 0.0, 0.0))
    assert zeroed.names == pool.names
    assert zeroed.weights[pool.names.index("turn")] == 0.0
    assert zeroed.weights[pool.names.index("omni_turn")] == 0.0
    assert omni_math.parse_cmd_pool(omni_math.render_cmd_pool(zeroed)).names == pool.names
    # ... and a zeroed type can be brought back by a later rescale (uniformly within its slot)
    assert zeroed.rescale_to(SLOTS_TEACHER).projected_slot_probs() == pytest.approx(
        SLOTS_TEACHER, abs=1e-9
    )
    # a slot with positive mass and NO type is a loud failure, not silently dropped mass
    no_turn = omni_math.parse_cmd_pool("stand:0.2,fwd:0.8")
    with pytest.raises(ValueError, match="turn"):
        no_turn.rescale_to(SLOTS_TEACHER)
    for bad in ((0.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (-0.1, 1.0, 0.0, 0.0)):
        with pytest.raises(ValueError):
            pool.rescale_to(bad)


# ------------------------------------------------------------------------ the gather tables
def test_pool_tables_trailing_entry_is_the_unit_sentinel():
    """``type_id = -1`` must land on UNIT scales, which negative indexing gives for free.

    A K-entry table indexed with -1 would hand every eval hold and every RSI-staged row the
    LAST type's scales -- silent, and exactly wrong in the direction that makes an eval look
    like a policy failure.
    """
    pool = omni_math.parse_cmd_pool("fwd:0.5:0.3:0.4,turn:0.5:0.2:0.6")
    tables = omni_math.pool_tables(pool, device="cpu", dtype=torch.float32)
    for table in (tables.slot, tables.lo, tables.hi, tables.world,
                  tables.ang_scale, tables.speed_scale):
        assert table.shape == (3,), "K + 1 entries"
    assert float(tables.ang_scale[-1]) == 1.0
    assert float(tables.speed_scale[-1]) == 1.0
    assert bool(tables.world[-1]) is False
    assert float(tables.lo[-1]) == 0.0 and float(tables.hi[-1]) == 0.0
    # the sentinel's SLOT is what a staged row is forced to; unused today (the pool branch
    # only ever gathers slots for rows sample_slots drew), but the docstring advertises it
    assert int(tables.slot[-1]) == S_MOVE
    # the K real entries are the pool's own, in order
    assert tables.ang_scale[:2].tolist() == pytest.approx([0.3, 0.2])
    assert tables.speed_scale[:2].tolist() == pytest.approx([0.4, 0.6])
    assert tables.slot[:2].tolist() == [S_MOVE, S_TURN]
    # the index a -1 row really takes
    sentinel = torch.tensor([-1, 0, 1])
    assert tables.ang_scale[sentinel].tolist() == pytest.approx([1.0, 0.3, 0.2])


def test_heading_rate_command_rows_is_the_scalar_version_at_unit_scale():
    yaw = _spread_yaw(257).double()
    target = torch.linspace(-4.0, 4.0, 257, dtype=torch.float64)
    scalar = omni_math.heading_rate_command(target, yaw, STIFFNESS, 0.3, 0.2)
    rows = omni_math.heading_rate_command_rows(
        target, yaw, STIFFNESS, torch.full((257,), 0.3, dtype=torch.float64),
        torch.full((257,), 0.2, dtype=torch.float64),
    )
    assert torch.equal(scalar, rows)
    # halving BOTH limits keeps their ratio (0.1/0.15 == 0.2/0.3) and halves the effective
    # facing-error threshold: a scaled-down type turns at a smaller rate for smaller errors,
    # which is what "a gentler turn demand" has to mean. Scaling the cap ALONE would leave the
    # unchanged dead band swallowing twice the share of it and make the type's turns vanish --
    # the silent failure ``ang_scale`` scales both to avoid.
    half = omni_math.heading_rate_command_rows(
        target, yaw, STIFFNESS, torch.full((257,), 0.15, dtype=torch.float64),
        torch.full((257,), 0.1, dtype=torch.float64),
    )
    assert float(half.abs().max()) == pytest.approx(0.15, abs=1e-12)
    raw = omni_math.wrap_to_pi(target - yaw) * STIFFNESS
    assert torch.equal(scalar != 0.0, raw.abs() > 0.2), "the band is a threshold on the P output"
    assert torch.equal(half != 0.0, raw.abs() > 0.1)
    assert bool(((scalar != 0.0) <= (half != 0.0)).all()), "a smaller band can only let more through"
    # the band is a STRICT ``>``: a row sitting EXACTLY on the clip is zeroed, not passed.
    # No grid point of the linspace above lands on it, so the edge row is built by reading the
    # raw P output back out and handing it in as the clip -- an exact float equality.
    edge_target = torch.tensor([0.5], dtype=torch.float64)
    edge_raw = omni_math.wrap_to_pi(edge_target) * STIFFNESS          # yaw = 0
    edge = omni_math.heading_rate_command_rows(
        edge_target, torch.zeros(1, dtype=torch.float64), STIFFNESS,
        torch.full((1,), 10.0, dtype=torch.float64),                  # cap: no saturation
        edge_raw.abs(),
    )
    assert float(edge_raw[0]) != 0.0
    assert float(edge[0]) == 0.0, "|wz| == clip is INSIDE the dead band (strict >)"


def test_row_ang_limits_are_validated_once_not_per_step():
    """The guard lives in ``check_row_ang_limits``, NOT inside the per-step expression.

    Two whole-tensor reductions plus two ``bool()`` calls per physics step would be two
    host-device synchronisations per step on CUDA -- a pooled stage slower than the legacy
    stage it is compared against, bought with a check that cannot fire (the tables are the
    cfg values times ``ang_scale``, and both are already validated). So: the checker raises,
    and the hot function does NOT.
    """
    yaw = _spread_yaw(257).double()
    target = torch.linspace(-4.0, 4.0, 257, dtype=torch.float64)
    cap = torch.full((257,), 0.3, dtype=torch.float64)
    with pytest.raises(ValueError, match="ang_vel_max"):
        omni_math.check_row_ang_limits(torch.zeros(257), torch.zeros(257))
    with pytest.raises(ValueError, match="ang_vel_clip"):
        omni_math.check_row_ang_limits(cap, -torch.ones(257))
    omni_math.check_row_ang_limits(cap, torch.zeros(257))  # a zero dead band is legal
    # the hot path is guard-free: a zero cap saturates to zero rather than raising
    assert bool((omni_math.heading_rate_command_rows(
        target, yaw, STIFFNESS, torch.zeros(257, dtype=torch.float64),
        torch.zeros(257, dtype=torch.float64),
    ) == 0.0).all())


def test_construction_validates_the_pool_turn_limit_tables_once(monkeypatch):
    """A pooled term runs ``check_row_ang_limits`` on its built tables at construction.

    Belt and braces -- ``parse_cmd_pool`` already refuses ``ang_scale <= 0`` and
    ``_validate_omni_cfg`` the two cfg values -- but it is the ONLY place the per-row limits
    are ever checked now, so it has to actually run. Re-deriving the checker's verdict from
    outside would pass with the call DELETED, so this spies on the symbol the command module
    imported and pins the call itself: once, with the built tables times the two cfg values.
    """
    calls = []
    real = omni_math.check_row_ang_limits

    def spy(ang_vel_max, ang_vel_clip):
        calls.append((ang_vel_max.clone(), ang_vel_clip.clone()))
        return real(ang_vel_max, ang_vel_clip)

    monkeypatch.setattr(omni, "check_row_ang_limits", spy)
    term, _ = _pool_term("stand:0.2,fwd:0.4:0.5,fwd_turn:0.4:0.25", (0.2, 0.4, 0.0, 0.4), n=8)
    assert len(calls) == 1, "once at construction, never per step"
    tables = term._pool_tables
    assert torch.equal(calls[0][0], 0.3 * tables.ang_scale), "the CAP table, not the cfg scalar"
    assert torch.equal(calls[0][1], 0.2 * tables.ang_scale), "the CLIP table, scaled the same"
    # the checker itself still refuses what it is there to refuse
    with pytest.raises(ValueError, match="ang_vel_max"):
        real(0.0 * tables.ang_scale, 0.2 * tables.ang_scale)
    # ... and a pool-less term never calls it at all (the legacy path takes the scalar form)
    calls.clear()
    _term(_cfg(), n=8)
    assert calls == []


# ------------------------------------------------------------------ the legacy bitwise twins
def _independent_legacy_sampler(cfg, yaw, seed):
    """The legacy resample, re-derived from five EXPLICIT uniforms in the recorded order.

    Nothing here calls the module under test: the draw order (slot, direction, speed, facing
    magnitude, facing sign), each draw's shape, and each expression are written out again.
    That is the point -- every other twin in this file is RELATIVE (pool branch vs legacy
    branch), so a mutation applied to BOTH branches (swapping two draws, say) passes them all
    while destroying the RNG stream every recorded seed-3 run and every Verification-3 smoke
    reference depends on. This is the ABSOLUTE pin of THE ONE HARD INVARIANT.
    """
    n = yaw.numel()
    a, b, c = (float(x) for x in cfg.speed_ellipse)
    floor = float(cfg.clips.lin_vel_clip) + command_sampling.WALK_MARGIN_M_S
    torch.manual_seed(seed)

    # (1) the slot: one uniform against the float32 cumulative of the normalised weights
    p = torch.as_tensor(cfg.slot_probs, dtype=torch.float64)
    cum = torch.cumsum(p / p.sum(), dim=0)
    cum[-1] = 1.0
    u_slot = torch.rand(n)
    slot = torch.searchsorted(
        cum.to(torch.float32).contiguous(), u_slot.contiguous(), right=True
    ).clamp_(max=3).to(torch.long)

    # (2) the world travel direction: world-uniform, or body-relative at the resample yaw
    u_dir = torch.rand(n)
    if cfg.dir_band is None:
        lo, hi = (float(x) for x in cfg.travel_dir_range)
        theta = lo + u_dir * (hi - lo)
    else:
        lo, hi = (float(x) for x in cfg.dir_band)
        theta = yaw + (lo + u_dir * (hi - lo))

    # (3) the speed, on the ellipse evaluated at the STARTING body direction
    u_speed = torch.rand(n)
    phi0 = travel_frame.wrap_to_pi(theta - yaw)  # the shared implementation, not the copy
    cos0, sin0 = torch.cos(phi0), torch.sin(phi0)
    axis = torch.where(cos0 >= 0.0, torch.full_like(phi0, a), torch.full_like(phi0, c))
    cap = axis * b / torch.sqrt(torch.square(b * cos0) + torch.square(axis * sin0))
    moving = (slot & 1) != 0
    speed = torch.where(moving, floor + u_speed * (cap - floor), torch.zeros_like(u_speed))

    # (4, 5) the facing target: magnitude then sign, held on non-turning slots
    dlo, dhi = (float(x) for x in cfg.facing_delta)
    u_mag = torch.rand(n)
    u_sign = torch.rand(n)
    sign = torch.where(u_sign < 0.5, -torch.ones_like(yaw), torch.ones_like(yaw))
    psi = yaw + sign * (dlo + u_mag * (dhi - dlo))
    psi = torch.where((slot & 2) != 0, psi, yaw)
    return {"slot_id": slot, "travel_dir_w": theta, "speed_cmd": speed, "heading_target": psi}


@pytest.mark.parametrize(
    "legacy",
    [
        {"slot_probs": (0.2, 0.8, 0.0, 0.0), "dir_band": (0.0, 0.0)},                  # 1a
        {"slot_probs": (0.0, 1.0, 0.0, 0.0), "dir_band": (math.pi / 2, math.pi / 2)},  # 1b
        {"slot_probs": (0.15, 0.55, 0.10, 0.20), "facing_delta": (0.2, 0.6)},          # 1c
        {"slot_probs": (0.15, 0.45, 0.15, 0.25), "facing_delta": (0.2, 1.2)},          # 2a
    ],
    ids=["1a", "1b", "1c", "2a"],
)
def test_the_legacy_sampler_is_pinned_absolutely(legacy):
    """The legacy path is what it was at HEAD, checked against an independent re-derivation.

    Guards the RNG STREAM and every expression in it: five uniforms, in the order
    slot -> direction -> speed -> facing magnitude -> facing sign, each of shape ``(n,)``.
    Any reordering, any extra or missing draw, and any change to a sampled expression fails
    here even if the pool branch is changed identically alongside it.
    """
    yaw = _spread_yaw(2048)
    cfg = _cfg(**legacy)
    term, _ = _term(cfg, n=2048, yaw=yaw)
    assert cfg.cmd_pool is None, "this is the legacy path"
    _resample(term)
    want = _independent_legacy_sampler(cfg, yaw, SEED)
    for name, expected in want.items():
        assert torch.equal(getattr(term, name), expected), name
    # ... and the stream is exactly five draws long, so the NEXT numbers agree too
    torch.manual_seed(SEED)
    for _ in range(5):
        torch.rand(2048)
    after = torch.rand(8)
    _resample(term)
    assert torch.equal(after, torch.rand(8))


def test_default_cfg_takes_the_legacy_path():
    """No pool configured -> no pool state, no pool metrics, ``type_id`` all -1 forever."""
    term, _ = _term(_cfg(), n=64, yaw=_spread_yaw(64))
    assert term.cfg.cmd_pool is None
    assert term._pool is None and term._pool_tables is None
    assert torch.equal(term.type_id, torch.full((64,), -1, dtype=torch.long))
    _resample(term)
    assert torch.equal(term.type_id, torch.full((64,), -1, dtype=torch.long)), "still the sentinel"
    assert not any(k.startswith("pool_") for k in term.metrics)


@pytest.mark.parametrize(
    "pool, slots, legacy",
    [
        # 1a: body +x only, 20 % standing
        ("stand:0.2,fwd:0.8", (0.2, 0.8, 0.0, 0.0), {"dir_band": (0.0, 0.0)}),
        # 1b: body +y only, every row moving (math.pi/2 on BOTH sides, so the float is the same)
        ("side_l:1.0", (0.0, 1.0, 0.0, 0.0), {"dir_band": (math.pi / 2, math.pi / 2)}),
    ],
    ids=["stand+fwd == dir_band(0,0)", "side_l == dir_band(pi/2,pi/2)"],
)
def test_pool_is_a_bitwise_twin_of_the_legacy_sampler(pool, slots, legacy):
    """A pool that describes a legacy distribution reproduces it EXACTLY -- same five uniforms
    in the same order, same expressions, same floats.

    This is the property that makes the pool safe to add: a curriculum stage whose pool
    happens to be a shipped window is that window, not an approximation of it.
    """
    yaw = _spread_yaw(4096)
    legacy_term, _ = _term(_cfg(slot_probs=slots, **legacy), n=4096, yaw=yaw)
    pool_term, _ = _term(_cfg(cmd_pool=pool, slot_probs=slots), n=4096, yaw=yaw)
    _resample(legacy_term)
    _resample(pool_term)
    legacy_term._update_command()
    pool_term._update_command()
    a, b = _state(legacy_term), _state(pool_term)
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_pool_1c_twin_matches_the_legacy_world_uniform_window():
    """The 1c twin, where only the two WORLD types can match a ``dir_band=None`` window.

    Everything the command publishes is bitwise: slots, speeds, facings and the whole
    ``vel_command_b``. ``travel_dir_w`` matches on every MOVING row; on a standing or
    turn-in-place row the pool deliberately holds the body axis instead of a random world
    bearing, which is unobservable in the command (speed is exactly 0) and is the one
    documented behaviour difference of the pool path.
    """
    yaw = _spread_yaw(8192)
    slots = (0.15, 0.55, 0.10, 0.20)
    legacy_term, _ = _term(_cfg(slot_probs=slots), n=8192, yaw=yaw)   # dir_band None = 1c
    pool_term, _ = _term(_cfg(cmd_pool=POOL_1C, slot_probs=slots), n=8192, yaw=yaw)
    _resample(legacy_term)
    _resample(pool_term)
    legacy_term._update_command()
    pool_term._update_command()
    a, b = _state(legacy_term), _state(pool_term)
    for name in ("slot_id", "speed_cmd", "heading_target", "vel_command_b"):
        assert torch.equal(a[name], b[name]), name
    moving = omni_math.slot_is_moving(a["slot_id"])
    assert bool(moving.any())
    assert torch.equal(a["travel_dir_w"][moving], b["travel_dir_w"][moving])
    # the non-moving rows are exactly the documented difference, and they command nothing
    still = ~moving
    assert torch.equal(b["travel_dir_w"][still], yaw[still]), "the pool holds the body axis"
    assert bool((b["speed_cmd"][still] == 0.0).all())
    assert bool((b["vel_command_b"][still, :2] == 0.0).all())


@pytest.mark.parametrize(
    "legacy_band, pool, same",
    [
        (None, "stand:0.3,omni:0.4,turn:0.3", False),
        ((math.pi / 2, math.pi / 2), "stand:0.3,side_l:0.4,turn:0.3", False),
        ((0.0, 0.0), "stand:0.3,fwd:0.4,turn:0.3", True),
    ],
    ids=["dir_band None (1c)", "dir_band (pi/2,pi/2) (1b)", "dir_band (0,0) (1a)"],
)
def test_pool_standing_rows_hold_the_body_axis_not_the_windows_band(legacy_band, pool, same):
    """The ONE documented behaviour difference of the pool path, pinned in BOTH directions.

    A legacy window gives every row -- standing, turning, moving -- the same global direction
    band; a pool gives each row its TYPE's band, and ``stand`` / ``turn`` carry the zero-width
    ``(0, 0)``. So a pool twin of a window with a ``stand`` share diverges on the standing
    rows' ``travel_dir_w`` whenever the window's band is not ``(0, 0)`` -- a random world ray
    for ``dir_band=None``, ``yaw + pi/2`` for the 1b band. Not fixable while a type has one
    band (world ``stand`` types would merely move the mismatch onto the body-band windows);
    the ruling is recorded in the math module. What must hold either way: the MOVING rows are
    bitwise, and nothing the command publishes moves (speed is exactly 0 on those rows).
    """
    yaw = _spread_yaw(4096)
    slots = (0.3, 0.4, 0.3, 0.0)  # a real stand AND turn share on both sides
    legacy_term, _ = _term(_cfg(slot_probs=slots, dir_band=legacy_band), n=4096, yaw=yaw)
    pool_term, _ = _term(_cfg(cmd_pool=pool, slot_probs=slots), n=4096, yaw=yaw)
    _resample(legacy_term)
    _resample(pool_term)
    legacy_term._update_command()
    pool_term._update_command()
    a, b = _state(legacy_term), _state(pool_term)
    for name in ("slot_id", "speed_cmd", "heading_target", "vel_command_b"):
        assert torch.equal(a[name], b[name]), name
    moving = omni_math.slot_is_moving(a["slot_id"])
    still = ~moving
    assert bool(moving.any()) and bool(still.any())
    assert torch.equal(a["travel_dir_w"][moving], b["travel_dir_w"][moving]), "moving rows"
    # the pool's non-moving rows always hold the nose ...
    assert torch.equal(b["travel_dir_w"][still], yaw[still])
    # ... which IS the legacy value only for the (0, 0) window; the other two diverge, and
    # the divergence is unobservable (speed exactly 0 -> a zero body command)
    if same:
        assert torch.equal(a["travel_dir_w"][still], b["travel_dir_w"][still])
    else:
        assert not torch.equal(a["travel_dir_w"][still], b["travel_dir_w"][still])
    assert bool((b["speed_cmd"][still] == 0.0).all())
    assert bool((b["vel_command_b"][still, :2] == 0.0).all())


def _rand_call_shapes(monkeypatch, fn):
    """Every ``torch.rand`` shape ``fn`` consumes, in order (the module calls it by path)."""
    shapes = []
    real = torch.rand

    def spy(*args, **kwargs):
        first = args[0] if args else kwargs.get("size")
        shapes.append(tuple(first) if isinstance(first, (tuple, list, torch.Size)) else (first,))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", spy)
    try:
        fn()
    finally:
        monkeypatch.undo()
    return shapes


def test_a_pool_draws_the_same_five_uniforms_as_the_legacy_path(monkeypatch):
    """The RNG stream must not depend on the type count, or no pooled run is comparable with
    a legacy one and no seed-3 record reproduces."""
    for pool, slots in (("stand:0.2,fwd:0.8", (0.2, 0.8, 0.0, 0.0)),
                        (POOL_1C, (0.15, 0.55, 0.10, 0.20)),
                        ("stand:0.1,fwd:0.2,side_l:0.1,side_r:0.1,back:0.1,omni:0.2,"
                         "turn:0.1,fwd_turn:0.1", (0.1, 0.7, 0.1, 0.1))):
        term, _ = _term(_cfg(cmd_pool=pool, slot_probs=slots), n=32, yaw=_spread_yaw(32))
        legacy, _ = _term(_cfg(), n=32, yaw=_spread_yaw(32))
        # FIVE draws of (32,) on each path, in the same order -- counted, not inferred from
        # the stream position (a pool that drew four and a longer one would also "agree" if
        # the total happened to match).
        pool_shapes = _rand_call_shapes(
            monkeypatch, lambda: term._resample_command(torch.arange(32))
        )
        legacy_shapes = _rand_call_shapes(
            monkeypatch, lambda: legacy._resample_command(torch.arange(32))
        )
        assert pool_shapes == [(32,)] * 5, pool
        assert legacy_shapes == pool_shapes, pool
        # both consumed exactly the same number of draws, so the NEXT numbers agree
        torch.manual_seed(1234)
        term._resample_command(torch.arange(32))
        after_pool = torch.rand(8)
        torch.manual_seed(1234)
        legacy._resample_command(torch.arange(32))
        assert torch.equal(after_pool, torch.rand(8)), pool


# ------------------------------------------------------------------- what a pool samples
def test_pool_type_frequencies_and_slot_projection():
    # side_l and side_r carry DIFFERENT weights on purpose: equal ones would make the test
    # blind to the two being swapped, which is the mistake the pool table invites.
    pool = "stand:0.15,fwd:0.25,side_l:0.18,side_r:0.12,turn:0.10,fwd_turn:0.20"
    slots = (0.15, 0.55, 0.10, 0.20)
    term, _ = _pool_term(pool, slots, n=20_000)
    _resample(term)
    parsed = omni_math.parse_cmd_pool(pool)
    freq = torch.bincount(term.type_id, minlength=len(parsed.names)).double() / 20_000
    assert torch.allclose(freq, torch.tensor(parsed.weights, dtype=torch.float64), atol=0.02)
    # ... and the slot every consumer downstream reads is the projection of that
    slot_freq = torch.bincount(term.slot_id, minlength=4).double() / 20_000
    assert torch.allclose(slot_freq, torch.tensor(slots, dtype=torch.float64), atol=0.02)


def test_pool_body_axis_types_command_the_exact_body_axes():
    """``side_l`` is the robot's LEFT, ``side_r`` its RIGHT and ``back`` straight behind --
    the sign errors that are invisible in a log and obvious in a video."""
    yaw = _spread_yaw(3000)
    term, _ = _pool_term(
        "fwd:0.25,side_l:0.25,side_r:0.25,back:0.25", (0.0, 1.0, 0.0, 0.0), n=3000, yaw=yaw
    )
    _resample(term)
    term._update_command()
    cmd = term.vel_command_b
    index = {name: i for i, name in enumerate(term._pool.names)}
    fwd = term.type_id == index["fwd"]
    left = term.type_id == index["side_l"]
    right = term.type_id == index["side_r"]
    back = term.type_id == index["back"]
    for mask in (fwd, left, right, back):
        assert bool(mask.any())
    assert torch.equal(cmd[fwd, 1], torch.zeros(int(fwd.sum()))), "fwd: vy_b exactly zero"
    assert bool((cmd[fwd, 0] > 0.0).all())
    assert torch.allclose(cmd[left, 0], torch.zeros(int(left.sum())), atol=1e-6)
    assert bool((cmd[left, 1] > 0.0).all()), "+y is the robot's LEFT"
    assert torch.allclose(cmd[right, 0], torch.zeros(int(right.sum())), atol=1e-6)
    assert bool((cmd[right, 1] < 0.0).all()), "side_r is the MIRROR of side_l, not a repeat"
    assert torch.allclose(cmd[back, 1], torch.zeros(int(back.sum())), atol=1e-6)
    assert bool((cmd[back, 0] < 0.0).all())
    # the ellipse is read at each row's own direction: backward is capped at c, lateral at b
    assert float(cmd[back, 0].abs().max()) <= C + 1e-6
    assert float(cmd[left, 1].max()) <= B + 1e-6
    assert float(cmd[fwd, 0].max()) > B + 0.02
    # "EXACT body axes" taken literally: the held world direction is ``yaw + <the literal>``
    # BITWISE on every one of the four, not merely a command with the right signs (an
    # ``atol`` on ``vel_command_b`` would pass a band that is a hair off the axis).
    theta = term.travel_dir_w
    for mask, offset in ((fwd, 0.0), (left, math.pi / 2), (right, -math.pi / 2), (back, math.pi)):
        want = yaw[mask] if offset == 0.0 else yaw[mask] + offset
        assert torch.equal(theta[mask], want), offset


def test_pool_omni_type_is_world_uniform():
    yaw = _spread_yaw(8192)
    term, _ = _pool_term("omni:1.0", (0.0, 1.0, 0.0, 0.0), n=8192, yaw=yaw)
    _resample(term)
    theta = term.travel_dir_w
    assert float(theta.min()) < -3.0 and float(theta.max()) > 3.0
    corr = torch.corrcoef(torch.stack([theta.double(), yaw.double()]))[0, 1]
    assert abs(float(corr)) < 0.05, "a world type must not track the body yaw"
    # ... while a body type in the same pool does track it, exactly
    mixed, _ = _pool_term("omni:0.5,fwd:0.5", (0.0, 1.0, 0.0, 0.0), n=8192, yaw=yaw)
    _resample(mixed)
    is_fwd = mixed.type_id == mixed._pool.names.index("fwd")
    assert torch.equal(mixed.travel_dir_w[is_fwd], yaw[is_fwd])


def test_pool_turning_types_turn_and_travelling_types_do_not():
    term, _ = _pool_term(
        "stand:0.25,fwd:0.25,turn:0.25,fwd_turn:0.25", (0.25, 0.25, 0.25, 0.25), n=8000
    )
    _resample(term)
    yaw = _spread_yaw(8000)
    face_err = omni_math.wrap_to_pi(term.heading_target - yaw).abs()
    turning = omni_math.slot_is_turning(term.slot_id)
    moving = omni_math.slot_is_moving(term.slot_id)
    assert torch.equal(face_err[~turning], torch.zeros_like(face_err[~turning]))
    assert bool((face_err[turning] >= 0.2 - 1e-6).all())
    assert bool((term.speed_cmd[~moving] == 0.0).all())
    assert bool((term.speed_cmd[moving] > 0.0).all())


# ------------------------------------------------------------------------ the per-type scales
def test_speed_scale_applies_to_its_own_rows_only():
    """``speed_scale`` narrows the draw-time band of ONE type, leaving its twin alone.

    A roomier ellipse than the shipped one, because the shipped ``c`` (0.25 m/s) has only
    0.04 m/s of headroom over the walk floor -- a halved backward cap would sit UNDER it, and
    the construction check below (``test_a_speed_scale_that_sinks_a_type_under_the_walk_floor
    _is_refused``) refuses exactly that.
    """
    yaw = torch.zeros(8000)
    a, b, c = 0.60, 0.50, 0.50
    term, _ = _pool_term(
        "fwd:0.5,back:0.5:1.0:0.5", (0.0, 1.0, 0.0, 0.0), n=8000, yaw=yaw,
        speed_ellipse=(a, b, c),
    )
    _resample(term)
    floor = omni.omni_walk_floor(term.cfg)
    is_fwd = term.type_id == term._pool.names.index("fwd")
    is_back = term.type_id == term._pool.names.index("back")
    assert float(term.speed_cmd[is_fwd].max()) > a - 0.01, "fwd keeps the whole fore-aft band"
    # back's cap is c * 0.5, and the floor is unchanged -- so the band shrinks from ABOVE
    assert float(term.speed_cmd[is_back].max()) <= c * 0.5 + 1e-6
    assert float(term.speed_cmd[is_back].max()) > c * 0.5 - 0.01
    assert float(term.speed_cmd[is_back].min()) >= floor - 1e-7
    # ... and the per-step re-cap stays the GLOBAL ellipse: the scale picks a smaller speed,
    # it does not narrow the clamp the body's own rotation runs into
    term._update_command()
    assert float(term.vel_command_b[is_back, 0].abs().max()) == pytest.approx(
        float(term.speed_cmd[is_back].max()), abs=1e-6
    )


def test_a_speed_scale_that_sinks_a_type_under_the_walk_floor_is_refused():
    """A scaled cap below the walk floor would INVERT the speed band, silently.

    ``speed = floor + u * (cap - floor)`` with ``cap < floor`` draws speeds BELOW the floor --
    i.e. under ``CLOCK_CMD_STOP_M_S``, where the gait clock freezes. The type would still
    appear in the dump, still be drawn at its weight, and train a creeper. So it is refused at
    construction, with the type and both numbers named.
    """
    with pytest.raises(ValueError, match="walk floor"):
        _term(
            _cfg(cmd_pool="fwd:0.5,back:0.5:1.0:0.5", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8
        )
    # the check is per TYPE and reads each type's OWN cap: ``fwd`` only ever draws at phi = 0,
    # where the cap is the fore-aft semi-axis, so a scale the lateral axis could not take is
    # still legal for it
    term, _ = _term(_cfg(cmd_pool="fwd:1.0:1.0:0.7", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8)
    assert term is not None                       # 0.35 * 0.7 = 0.245 > 0.21 + 0.01
    with pytest.raises(ValueError, match="walk floor"):
        _term(_cfg(cmd_pool="fwd:1.0:1.0:0.6", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8)
    # a WORLD type is judged against the SMALLEST semi-axis: it can draw any direction, so the
    # same 0.7 that ``fwd`` may take sinks it (0.25 * 0.7 = 0.175)
    with pytest.raises(ValueError, match="walk floor"):
        _term(_cfg(cmd_pool="omni:1.0:1.0:0.7", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8)
    # a non-moving type never draws a speed at all, so its scale cannot sink anything
    term, _ = _term(
        _cfg(cmd_pool="stand:0.5:1.0:0.01,fwd:0.5", slot_probs=(0.5, 0.5, 0.0, 0.0)), n=8
    )
    assert term is not None


def test_pool_speed_scale_floor_is_the_bound_the_refusal_uses():
    """The refusal's bound is EXPORTED, so it is spelled once.

    The construction refusal above is loud but late: a ``pool_edit {speed_scale: ...}`` that
    cannot be honoured at the shipped ellipse dies when the config is built, after a GPU
    window has been booked. ``pool_speed_scale_floor`` is the same arithmetic as a plain
    function the campaign driver's recipe validator can call at COMPOSE time -- so this pins
    the two against each other: a scale just above the published bound builds, a scale just
    below it raises, for the roomiest type and for the tightest.
    """
    min_speed = omni.omni_walk_floor(_cfg()) + omni.MIN_CAP_HEADROOM_M_S
    # each type is judged at the smallest cap IT can draw: its own axis when the band is one
    # exact direction, the smallest semi-axis when it can draw anywhere
    assert omni_math.pool_type_min_cap("fwd", A, B, C) == pytest.approx(A, abs=1e-9)
    assert omni_math.pool_type_min_cap("side_l", A, B, C) == pytest.approx(B, abs=1e-9)
    assert omni_math.pool_type_min_cap("side_r", A, B, C) == pytest.approx(B, abs=1e-9)
    assert omni_math.pool_type_min_cap("back", A, B, C) == pytest.approx(C, abs=1e-9)
    assert omni_math.pool_type_min_cap("omni", A, B, C) == pytest.approx(min(A, B, C))
    assert omni_math.pool_type_min_cap("omni_turn", A, B, C) == pytest.approx(min(A, B, C))
    for name in ("fwd", "omni"):
        bound = omni_math.pool_speed_scale_floor(name, (A, B, C), min_speed)
        assert bound == pytest.approx(
            min_speed / omni_math.pool_type_min_cap(name, A, B, C), rel=1e-12
        )
        term, _ = _term(
            _cfg(cmd_pool=f"{name}:1.0:1.0:{bound * 1.02!r}", slot_probs=(0.0, 1.0, 0.0, 0.0)),
            n=8,
        )
        assert term is not None, f"{name}: just above the published bound must BUILD"
        with pytest.raises(ValueError, match="walk floor"):
            _term(
                _cfg(cmd_pool=f"{name}:1.0:1.0:{bound * 0.98!r}",
                     slot_probs=(0.0, 1.0, 0.0, 0.0)),
                n=8,
            )
    # ... and the refusal NAMES the bound, so the message alone says what would have worked
    with pytest.raises(ValueError, match="speed_scale > 0.88"):
        _term(_cfg(cmd_pool="omni:1.0:1.0:0.5", slot_probs=(0.0, 1.0, 0.0, 0.0)), n=8)


def test_ang_scale_applies_to_its_own_rows_and_minus_one_uses_the_global_cap():
    """The -1 sentinel test: a pool whose LAST type is scaled must not leak that scale onto
    eval / RSI-staged rows, which is exactly what a K-entry table indexed with -1 would do."""
    n = 6
    yaw = torch.zeros(n)
    # LAST type carries the reduced turn demand, so a mis-sized table is caught
    term, _ = _pool_term("turn:0.5,fwd_turn:0.5:0.3", (0.0, 0.0, 0.5, 0.5), n=n, yaw=yaw)
    assert term._pool.names[-1] == "fwd_turn"
    term.speed_cmd[:] = 0.0
    term.travel_dir_w[:] = 0.0
    term.heading_target[:] = 2.0                      # a saturating facing error on every row
    term.type_id[:] = torch.tensor([0, 0, 1, 1, -1, -1])
    term._update_command()
    wz = term.vel_command_b[:, 2]
    cap = float(term.cfg.ang_vel_max)
    assert float(wz[0]) == pytest.approx(cap, abs=1e-6), "unscaled type: the global cap"
    assert float(wz[2]) == pytest.approx(cap * 0.3, abs=1e-6), "scaled type: 0.3 x the cap"
    assert float(wz[4]) == pytest.approx(cap, abs=1e-6), "type_id -1 -> the GLOBAL cap"
    # the dead band scales with it, so the scaled type still turns at all
    small = 0.14                                       # 0.14 * 1.5 = 0.21, just over 0.2
    term.heading_target[:] = small
    term._update_command()
    assert float(term.vel_command_b[2, 2]) == pytest.approx(
        min(small * STIFFNESS, cap * 0.3), abs=1e-6
    ), "a 0.3-scaled dead band is 0.06, so this error still commands a turn"

    # ... and the OTHER half of the sentinel's failure space: the pool above leaves its FIRST
    # type unscaled, so a gather written ``ang_scale[type_id.clamp_min(0)]`` -- the plan's own
    # illustrative formula with its ``torch.where`` dropped, i.e. the expected slip -- would
    # fold -1 onto index 0 and still read 1.0. With EVERY type scaled, the unit entry is
    # reachable only by landing on the table's trailing sentinel.
    both, _ = _pool_term("turn:0.5:0.6,fwd_turn:0.5:0.3", (0.0, 0.0, 0.5, 0.5), n=n, yaw=yaw)
    both.speed_cmd[:] = 0.0
    both.travel_dir_w[:] = 0.0
    both.heading_target[:] = 2.0
    both.type_id[:] = torch.tensor([0, 0, 1, 1, -1, -1])
    both._update_command()
    wz2 = both.vel_command_b[:, 2]
    assert float(wz2[0]) == pytest.approx(cap * 0.6, abs=1e-6), "first type: 0.6 x the cap"
    assert float(wz2[2]) == pytest.approx(cap * 0.3, abs=1e-6), "last type: 0.3 x the cap"
    assert float(wz2[4]) == pytest.approx(cap, abs=1e-6), "-1 -> the trailing UNIT entry"
    assert abs(float(wz2[4]) - cap * 0.6) > 1e-3, "-1 must not fold onto index 0"
    assert abs(float(wz2[4]) - cap * 0.3) > 1e-3, "-1 must not fold onto the last type"
    # and directly, without going through the P-control: the gather an all-(-1) population
    # performs on EVERY table is the unit / neutral entry
    all_minus_one = torch.full((n,), -1, dtype=torch.long)
    tables = both._pool_tables
    assert torch.equal(tables.ang_scale[all_minus_one], torch.ones(n, dtype=tables.ang_scale.dtype))
    assert torch.equal(
        tables.speed_scale[all_minus_one], torch.ones(n, dtype=tables.speed_scale.dtype)
    )


def test_schedule_hook_clears_the_type_id():
    """An eval hold did not come from the pool: it must run at the GLOBAL turn cap, or the
    manifest's own ``hold_s >= settle_s + |dface| / ang_vel_max`` sizing rule is wrong."""
    # EVERY type scaled, so "the global cap" is a value no type could have produced -- a
    # sentinel that folded onto index 0 would read 0.6 x the cap here and pass a pool whose
    # first type happened to be unscaled.
    term, _ = _pool_term("turn:0.5:0.6,fwd_turn:0.5:0.3", (0.0, 0.0, 0.5, 0.5), n=4,
                         yaw=torch.zeros(4))
    _resample(term)
    assert bool((term.type_id >= 0).all()), "a resample really did draw from the pool"
    term.schedule_hook({"speed": 0.0, "dir": 0.0, "face": math.pi / 2})
    assert torch.equal(term.type_id, torch.full((4,), -1, dtype=torch.long))
    assert torch.allclose(
        term.vel_command_b[:, 2], torch.full((4,), float(term.cfg.ang_vel_max)), atol=1e-6
    ), "a hold runs at the GLOBAL cap, never at some type's reduced turn demand"


# ------------------------------------------------------------------------- the pool metrics
def test_pool_metrics_exist_only_with_a_pool():
    """The nine-name metric set of a shipped (pool-less) run must not move."""
    plain, _ = _term(_cfg(), n=8)
    assert set(plain.metrics) == {
        "error_vel_xy", "error_vel_yaw", "stand_frac_actual", "move_frac_actual",
        "turn_frac_actual", "move_turn_frac_actual", "facing_err_abs", "recap_frac",
        "yaw_progress",
    }
    pooled, _ = _pool_term("stand:0.2,fwd:0.8", (0.2, 0.8, 0.0, 0.0), n=8)
    assert set(pooled.metrics) - set(plain.metrics) == {
        "pool_stand_frac", "pool_fwd_frac", "pool_stand_draws", "pool_fwd_draws",
    }


def test_pool_frac_metrics_are_step_weighted_and_exclude_staged_rows():
    n = 8
    term, _ = _pool_term("stand:0.5,fwd:0.5", (0.5, 0.5, 0.0, 0.0), n=n, yaw=torch.zeros(n))
    term.type_id[:] = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1])
    term._update_command()
    term._update_metrics()
    per_step = 1.0 / EPISODE_STEPS
    assert torch.equal(
        term.metrics["pool_stand_frac"],
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * per_step,
    )
    assert torch.equal(
        term.metrics["pool_fwd_frac"],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]) * per_step,
    )
    # the staged rows are in NO type -- while the four slot fracs still sum to 1 per row,
    # because a staged row IS in a slot, it just did not come from the pool
    slot_sum = sum(
        term.metrics[k] for k in
        ("stand_frac_actual", "move_frac_actual", "turn_frac_actual", "move_turn_frac_actual")
    )
    assert torch.allclose(slot_sum, torch.full((n,), per_step), atol=1e-9)

    # STEP-WEIGHTED, the first half of this test's name: a second step ACCUMULATES on top of
    # the first (``+=``, like the four ``*_frac_actual`` siblings). One call cannot tell that
    # apart from an overwrite -- and under an overwrite the metric would report 1/episode_steps
    # for whichever type the env sits in on the LAST step of the episode instead of that
    # type's share of it, which is exactly the number the campaign reads per stage.
    term.type_id[0] = 1                    # env 0 moves from ``stand`` into ``fwd``
    term._update_command()
    term._update_metrics()
    assert torch.equal(
        term.metrics["pool_stand_frac"],
        torch.tensor([1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * per_step,
    ), "one step of stand on env 0, two on envs 1-2"
    assert torch.equal(
        term.metrics["pool_fwd_frac"],
        torch.tensor([1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0]) * per_step,
    ), "env 0's second step landed on fwd, and the first step's stand is still there"
    assert torch.equal(
        term.metrics["pool_stand_frac"][6:], torch.zeros(2)
    ), "the staged rows stay in NO type across steps"


def test_pool_draw_metrics_track_the_weights_and_skip_staged_rows():
    n = 4000
    weights = (0.3, 0.7)
    term, _ = _pool_term("stand:0.3,fwd:0.7", (0.3, 0.7, 0.0, 0.0), n=n, yaw=torch.zeros(n))
    for seed in range(4):
        _resample(term, seed=SEED + seed)
    term._update_metrics()
    got = [float(term.metrics[f"pool_{name}_draws"].mean()) for name in ("stand", "fwd")]
    assert got == pytest.approx(list(weights), abs=0.02)
    assert sum(got) == pytest.approx(1.0, abs=1e-6), "every draw picked exactly one type"

    # LIFETIME, not per-call: a second ``_update_metrics()`` with no intervening resample must
    # reproduce the EXACT same tensor -- an accumulating ``+=`` would double it, while the real
    # overwrite (``[:] =``) recomputes the same ratio from the same, unchanged lifetime counters.
    first_call = {name: term.metrics[f"pool_{name}_draws"].clone() for name in ("stand", "fwd")}
    term._update_metrics()
    for name in ("stand", "fwd"):
        assert torch.equal(term.metrics[f"pool_{name}_draws"], first_call[name]), (
            f"pool_{name}_draws changed on a second _update_metrics() with no resample in between"
        )

    # NOT RESET across resamples either: pin the lifetime ratio by hand against the observed
    # type_id history, on a population small enough to compute exactly rather than statistically.
    # Row 0 is never touched again (one lifetime draw survives untouched); rows 1-2 get a
    # genuine second draw; row 3's second "draw" is RSI-staged and must extend neither the
    # numerator nor the denominator of its lifetime ratio.
    small_n = 4
    small, _ = _pool_term("stand:0.5,fwd:0.5", (0.5, 0.5, 0.0, 0.0), n=small_n, yaw=torch.zeros(small_n))
    _resample(small, seed=SEED)
    first_types = small.type_id.clone()
    assert bool((first_types >= 0).all()), "nothing staged on the first draw"
    small._update_metrics()

    second_ids = torch.tensor([1, 2, 3])
    small.rsi_cmd_staged[3] = torch.tensor([0.0, 0.22, 0.0])  # row 3's second draw is staged
    torch.manual_seed(SEED + 100)
    small._resample_command(second_ids)
    second_types = small.type_id.clone()  # row 0 untouched; 1, 2 redrawn; 3 forced to -1
    assert int(second_types[3]) == -1
    small._update_metrics()

    names = ("stand", "fwd")
    expected = {name: torch.zeros(small_n) for name in names}
    for row in range(small_n):
        draws = [int(first_types[row])]
        if row in (1, 2):  # genuinely redrawn from the pool a second time
            draws.append(int(second_types[row]))
        # row 3's second "draw" was staged (-1): it does not extend the history at all
        total = len(draws)
        for index, name in enumerate(names):
            expected[name][row] = draws.count(index) / total

    for name in names:
        assert torch.equal(small.metrics[f"pool_{name}_draws"], expected[name]), (
            f"pool_{name}_draws is not the exact lifetime draw ratio ({name}) across two resamples"
        )

    # an RSI-staged row is not a pool draw: it counts in neither the numerator nor the total
    staged, _ = _pool_term("stand:0.5,fwd:0.5", (0.5, 0.5, 0.0, 0.0), n=8, yaw=torch.zeros(8))
    staged.rsi_cmd_staged[:] = torch.tensor([0.0, 0.22, 0.0]).expand(8, 3).clone()
    _resample(staged)
    assert torch.equal(staged.type_id, torch.full((8,), -1, dtype=torch.long))
    assert float(staged._pool_draw_n.sum()) == 0.0
    staged._update_metrics()
    for name in ("stand", "fwd"):
        assert torch.equal(staged.metrics[f"pool_{name}_draws"], torch.zeros(8))


# -------------------------------------------------------------- the two refusals at build
def test_pool_and_dir_band_are_refused_together():
    """Two spellings of one lever: whichever won would depend on knob application order."""
    with pytest.raises(ValueError, match="dir_band"):
        _term(
            _cfg(cmd_pool="stand:0.2,fwd:0.8", slot_probs=(0.2, 0.8, 0.0, 0.0),
                 dir_band=(0.0, 0.0)),
            n=8,
        )


def test_pool_that_disagrees_with_slot_probs_is_refused():
    """The pool is what is DRAWN and ``slot_probs`` is what every metric, reward and eval
    reads: a mismatch trains one distribution and reports another, in silence."""
    with pytest.raises(ValueError, match="slot_probs"):
        _term(_cfg(cmd_pool="stand:0.2,fwd:0.8", slot_probs=(0.15, 0.55, 0.10, 0.20)), n=8)
    # un-normalised slot_probs describing the SAME regime are fine (both sides normalise)
    term, _ = _term(_cfg(cmd_pool="stand:0.2,fwd:0.8", slot_probs=(2.0, 8.0, 0.0, 0.0)), n=8)
    assert term is not None
    # ... and so is a tolerable rounding difference, while a visible one is not
    tol = omni.POOL_SLOT_MATCH_TOL
    _term(_cfg(cmd_pool="stand:0.2,fwd:0.8", slot_probs=(0.2 + tol / 4, 0.8, 0.0, 0.0)), n=8)
    with pytest.raises(ValueError, match="slot_probs"):
        _term(_cfg(cmd_pool="stand:0.2,fwd:0.8", slot_probs=(0.21, 0.79, 0.0, 0.0)), n=8)


def test_a_pooled_term_still_consumes_rsi_staging_exactly_once():
    """Staging is orthogonal to the pool: a staged row is still forced to MOVE, still
    re-NaNned, and (new) still carries the -1 sentinel so it runs at the global cap."""
    yaw = _spread_yaw(16)
    term, _ = _pool_term("stand:1.0,fwd:0.0", (1.0, 0.0, 0.0, 0.0), n=16, yaw=yaw)
    rows = torch.tensor([1, 4, 9])
    theta_s, v_s = torch.tensor([0.3, -1.1, 2.7]), torch.tensor([0.22, 0.24, 0.23])
    term.rsi_cmd_staged[rows] = torch.stack([theta_s, v_s, yaw[rows].clone()], dim=1)
    _resample(term)
    mask = torch.zeros(16, dtype=torch.bool)
    mask[rows] = True
    assert torch.equal(term.travel_dir_w[mask], theta_s)
    assert torch.equal(term.slot_id[mask], torch.full((3,), S_MOVE))
    assert torch.equal(term.type_id[mask], torch.full((3,), -1))
    assert bool((term.type_id[~mask] == 0).all()), "the un-staged rows drew 'stand'"
    assert bool(torch.isnan(term.rsi_cmd_staged).all())
