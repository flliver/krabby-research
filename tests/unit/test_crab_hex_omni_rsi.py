"""Omni RSI reset: re-yawed reference frames and the staged command (plan D6 / spec A7).

Mock env, no Isaac Sim -- the pattern of ``test_crab_hex_rsi_spawn_fix.py``, with one
deliberate difference: the quaternion math is NOT stubbed. ``crab_hex_omni_rsi`` reaches its
``quat_mul`` / ``quat_apply`` / ``yaw_from_quat`` through ``omni_events`` (transcriptions of
the ``isaaclab.utils.math`` functions), so the rotations asserted here are the rotations that
run in training; a test that stubbed ``quat_apply`` as the identity -- as the forward RSI
test legitimately does, because the forward event only needs "some rotation" -- would assert
nothing about the two sign-bearing steps this event adds:

* ``q_new = q_z(psi - yaw(q_bank)) * q_bank`` must set the yaw to EXACTLY ``psi`` while
  leaving roll and pitch (the frame's gait attitude) untouched. Getting the multiplication
  order backwards tilts the robot instead of turning it, which looks like a bad bank.
* the staged command must be the frame's OWN momentum: direction from the world velocity,
  speed from the body velocity clamped into ``[walk_floor, ellipse_cap(phi_b)]``, facing
  ``psi``. A wrong sign here starts every seeded episode commanded away from the direction
  it is moving -- the policy sees a demonstration of failure.

Re-NaN-ing of ``rsi_cmd_staged`` is the COMMAND term's job (it consumes a staged row exactly
once) and is pinned in ``test_crab_hex_omni_command.py``; this file only checks what the
event writes.
"""
import importlib
import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PARKOURS_DIR = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "parkours"
OMNI_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "mdp"
PKG = "_omni_mdp_under_test"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Register the real file at ``path`` under ``dotted``, creating parent packages.

    The parent packages get the REAL source directories as their ``__path__`` rather than an
    empty one: other suites in the same pytest process path-load different modules out of
    the same dotted tree (``test_crab_hex_omni_command.py`` imports
    ``parkour_isaaclab.envs.mdp.travel_frame`` this way), and a stub package with no search
    path would make whichever suite ran second fail to import. Their real ``__init__.py``
    files are still never executed -- that is the point, they reach Isaac.
    """
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


# the real (pure-torch) exposure ledger module the omni exposure helpers import
_load_real_module(
    "parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py"
)

# isaaclab.managers.SceneEntityCfg: only used as ``omni_events``' default argument
_isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
_managers = sys.modules.get("isaaclab.managers")
if _managers is None:
    _managers = types.ModuleType("isaaclab.managers")
    sys.modules["isaaclab.managers"] = _managers
if not hasattr(_managers, "SceneEntityCfg"):
    class SceneEntityCfg:  # noqa: D401 - stand-in for the cfg dataclass
        def __init__(self, name="robot", **kw):
            self.name = name
    _managers.SceneEntityCfg = SceneEntityCfg
_isaaclab.managers = _managers

# the omni mdp dir as a package, so the modules' relative imports resolve
if PKG not in sys.modules:
    spec = importlib.machinery.ModuleSpec(PKG, loader=None, is_package=True)
    spec.submodule_search_locations = [str(OMNI_MDP)]
    sys.modules[PKG] = importlib.util.module_from_spec(spec)

rsi = importlib.import_module(f"{PKG}.crab_hex_omni_rsi")
oe = importlib.import_module(f"{PKG}.omni_events")
cmath_mod = importlib.import_module(f"{PKG}.crab_hex_omni_command_math")

N_JOINTS = 18
ELLIPSE = (0.35, 0.25, 0.25)
WALK_FLOOR = 0.21


# --------------------------------------------------------------------------- quaternions


def _quat_from_euler(roll, pitch, yaw):
    """``isaaclab.utils.math.quat_from_euler_xyz``, transcribed (test-side reference)."""
    roll, pitch, yaw = (torch.as_tensor(v, dtype=torch.float32) for v in (roll, pitch, yaw))
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack([
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ], dim=-1)


def _gravity_in_body(q):
    """World ``-z`` expressed in the body frame -- invariant under a WORLD yaw rotation.

    Comparing this before and after the re-yaw is a coordinate-free statement of "roll and
    pitch were preserved" that does not depend on an Euler convention.
    """
    conj = q.clone()
    conj[:, 1:] *= -1.0
    down = torch.tensor([0.0, 0.0, -1.0]).expand(q.shape[0], 3)
    return oe.quat_apply(conj, down)


# ------------------------------------------------------------------------------ mock env


class _Robot:
    def __init__(self, n):
        default = torch.zeros(n, 13)
        default[:, 2] = 0.30
        default[:, 3] = 1.0
        self.data = types.SimpleNamespace(default_root_state=default)
        self.writes = {}

    def write_root_pose_to_sim(self, pose, env_ids):
        self.writes["pose"] = (env_ids.clone(), pose.clone())

    def write_root_velocity_to_sim(self, vel, env_ids):
        self.writes["vel"] = (env_ids.clone(), vel.clone())

    def write_joint_state_to_sim(self, jp, jv, env_ids):
        self.writes["joints"] = (env_ids.clone(), jp.clone(), jv.clone())


class _CmdTerm:
    def __init__(self, n):
        self.cfg = types.SimpleNamespace(speed_ellipse=ELLIPSE)
        self.rsi_cmd_staged = torch.full((n, 3), float("nan"))


class _ParkourTerm:
    def __init__(self):
        self.calls = []

    def note_spawn_xy(self, env_ids, xy, kind):
        self.calls.append((env_ids.clone(), xy.clone(), kind))


def _env(n=6):
    robot = _Robot(n)
    origins = torch.zeros(n, 3)
    origins[:, 0] = torch.arange(n).float() * 10.0 + 25.0
    origins[:, 1] = -13.0
    action_term = types.SimpleNamespace(rsi_clock_staged=torch.full((n,), float("nan")))
    cmd_term = _CmdTerm(n)
    parkour_term = _ParkourTerm()
    env = types.SimpleNamespace(
        device="cpu",
        scene=_Scene(robot, origins),
        action_manager=types.SimpleNamespace(get_term=lambda name: action_term),
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd_term),
        parkour_manager=types.SimpleNamespace(
            active_terms=["base_parkour"], get_term=lambda name: parkour_term
        ),
    )
    return env, robot, origins, action_term, cmd_term, parkour_term


class _Scene:
    def __init__(self, robot, origins):
        self._items = {"robot": robot}
        self.env_origins = origins

    def __getitem__(self, key):
        return self._items[key]


# ---------------------------------------------------------------------------------- bank


def _make_bank(tmp_path, *, with_source=True, name="bank.npz"):
    """Two provenance groups with distinguishable joint_pos and body velocities.

    source 0 ("forward"): body velocity along +x, joint_pos == 0.0
    source 1 ("sideways"): body velocity along +y, joint_pos == 1.0
    """
    rolls = torch.tensor([0.08, -0.05, 0.02, 0.10])
    pitches = torch.tensor([-0.03, 0.06, 0.09, -0.07])
    yaws = torch.tensor([0.4, -2.0, 3.0, 1.1])
    quats = _quat_from_euler(rolls, pitches, yaws)
    vel = torch.tensor([[0.24, 0.01, 0.0], [0.30, -0.02, 0.0],
                        [0.005, 0.22, 0.0], [-0.01, 0.40, 0.0]])
    jp = torch.tensor([0.0, 0.0, 1.0, 1.0]).view(4, 1).repeat(1, N_JOINTS)
    arrays = dict(
        joint_pos=jp.numpy().astype(np.float32),
        joint_vel=(jp * 0.1).numpy().astype(np.float32),
        root_quat_w=quats.numpy().astype(np.float32),
        root_z=np.array([0.31, 0.32, 0.33, 0.34], np.float32),
        root_lin_vel_b=vel.numpy().astype(np.float32),
        clock_phase=np.array([0.1, 0.2, 0.3, 0.4], np.float32),
    )
    if with_source:
        arrays["source"] = np.array([0, 0, 1, 1], np.int8)
        arrays["meta"] = np.array('{"banks": ["fwd", "side"]}')
    path = tmp_path / name
    np.savez(path, **arrays)
    rsi._BANK_CACHE.clear()
    return str(path), quats, vel


@pytest.fixture()
def bank(tmp_path):
    return _make_bank(tmp_path)


# --------------------------------------------------------------------------------- tests


def test_yaw_is_exactly_psi_and_roll_pitch_are_preserved(bank):
    path, quats, _ = bank
    env, robot, _, _, _, _ = _env()
    psi = 1.2345
    rsi.reset_from_reference_states_omni(
        env, torch.arange(6), path, fraction=1.0, yaw_range=(psi, psi), source="all"
    )
    _, pose = robot.writes["pose"]
    q_new = pose[:, 3:7]
    assert torch.allclose(oe.yaw_from_quat(q_new), torch.full((6,), psi), atol=1e-6)
    # which bank row each env drew is random; the invariant is per-row, so match on the
    # body-frame gravity of the four possible source rows.
    g_new = _gravity_in_body(q_new)
    g_bank = _gravity_in_body(quats)
    for i in range(6):
        assert torch.isclose(g_bank, g_new[i], atol=1e-6).all(dim=1).any(), (
            "the re-yawed attitude must keep one of the bank rows' roll/pitch exactly"
        )


def test_yaw_range_is_sampled_not_pinned(bank):
    path, _, _ = bank
    env, robot, _, _, _, _ = _env(n=64)
    torch.manual_seed(0)
    rsi.reset_from_reference_states_omni(
        env, torch.arange(64), path, fraction=1.0, yaw_range=(-math.pi, math.pi)
    )
    _, pose = robot.writes["pose"]
    yaws = oe.yaw_from_quat(pose[:, 3:7])
    assert float(yaws.max() - yaws.min()) > 3.0, "spawn yaw must span the band"


def test_world_velocity_is_the_body_velocity_rotated_by_the_new_attitude(bank):
    path, _, _ = bank
    env, robot, _, _, _, _ = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0,
                                         yaw_range=(0.7, 0.7))
    _, pose = robot.writes["pose"]
    _, vel = robot.writes["vel"]
    q_new = pose[:, 3:7]
    v_w = vel[:, 0:3]
    # the body velocity is recoverable by rotating back; its magnitude is preserved
    conj = q_new.clone()
    conj[:, 1:] *= -1.0
    v_b = oe.quat_apply(conj, v_w)
    assert torch.allclose(torch.norm(v_w, dim=1), torch.norm(v_b, dim=1), atol=1e-6)
    # ... and the recovered body velocity is one of the bank rows
    bank_vel = rsi._load_bank(path, "cpu")["root_lin_vel_b"]
    for i in range(6):
        assert torch.isclose(bank_vel, v_b[i], atol=1e-5).all(dim=1).any()
    assert torch.allclose(vel[:, 3:6], torch.zeros(6, 3)), "angular velocity must be zeroed"


def test_forward_frames_are_commanded_forward_in_the_world(bank):
    """source 0 frames move along body +x; at spawn yaw psi the world direction is psi."""
    path, _, _ = bank
    env, robot, _, _, cmd_term, _ = _env()
    psi = -0.9
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0,
                                         yaw_range=(psi, psi), source=0)
    staged = cmd_term.rsi_cmd_staged
    assert torch.isfinite(staged).all()
    theta, speed, face = staged[:, 0], staged[:, 1], staged[:, 2]
    assert torch.allclose(face, torch.full((6,), psi))
    # body velocity is (0.24, 0.01) or (0.30, -0.02): |phi_b| < 0.07 rad of body +x
    assert torch.all(torch.abs(cmath_mod.wrap_to_pi(theta - psi)) < 0.12)


def test_sideways_frames_are_commanded_to_the_robots_left(bank):
    """source 1 frames move along body +y; the world direction is psi + pi/2."""
    path, _, _ = bank
    env, robot, _, _, cmd_term, _ = _env()
    psi = 2.0
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0,
                                         yaw_range=(psi, psi), source=1)
    theta = cmd_term.rsi_cmd_staged[:, 0]
    assert torch.all(torch.abs(cmath_mod.wrap_to_pi(theta - psi - math.pi / 2)) < 0.12)


def test_staged_speed_is_the_frame_speed_clamped_into_floor_and_cap(bank):
    path, _, _ = bank
    env, robot, _, _, cmd_term, _ = _env(n=200)
    torch.manual_seed(1)
    rsi.reset_from_reference_states_omni(env, torch.arange(200), path, fraction=1.0,
                                         walk_floor=WALK_FLOOR)
    speed = cmd_term.rsi_cmd_staged[:, 1]
    assert float(speed.min()) >= WALK_FLOOR - 1e-6
    assert float(speed.max()) <= max(ELLIPSE) + 1e-6
    # the two source-1 rows travel sideways at 0.220 and 0.400 m/s against a lateral cap of
    # 0.25, so the fast one must be CLAMPED DOWN to (essentially) the cap, not rotated.
    lateral = speed[torch.abs(speed - 0.25) < 0.01]
    assert lateral.numel() > 0, "the 0.40 m/s lateral frame must appear at the lateral cap"


def test_staged_speed_is_lifted_to_the_walk_floor(tmp_path):
    """A frame slower than the floor would command a stop and freeze the gait clock."""
    path, _, _ = _make_bank(tmp_path, name="slow.npz")
    raw = dict(np.load(path))
    raw["root_lin_vel_b"] = np.zeros_like(raw["root_lin_vel_b"])
    raw["root_lin_vel_b"][:, 0] = 0.02
    slow = tmp_path / "slow2.npz"
    np.savez(slow, **raw)
    rsi._BANK_CACHE.clear()
    env, _, _, _, cmd_term, _ = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), str(slow), fraction=1.0,
                                         walk_floor=WALK_FLOOR)
    assert torch.allclose(cmd_term.rsi_cmd_staged[:, 1], torch.full((6,), WALK_FLOOR), atol=1e-6)


def test_staged_speed_matches_the_ellipse_cap_exactly(bank):
    """Recompute the contract from the bank row the env actually drew."""
    path, _, _ = bank
    env, robot, _, _, cmd_term, _ = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0,
                                         walk_floor=WALK_FLOOR)
    b = rsi._load_bank(path, "cpu")
    v_b_all = b["root_lin_vel_b"]
    phi = torch.atan2(v_b_all[:, 1], v_b_all[:, 0])
    cap = cmath_mod.ellipse_cap(phi, *ELLIPSE)
    want = torch.minimum(torch.clamp(torch.norm(v_b_all[:, :2], dim=1), min=WALK_FLOOR), cap)
    for s in cmd_term.rsi_cmd_staged[:, 1]:
        assert torch.isclose(want, s, atol=1e-6).any()


def test_source_filter_selects_only_matching_rows(bank):
    path, _, _ = bank
    for source, joint_value in ((0, 0.0), ("0", 0.0), (1, 1.0), ("1", 1.0)):
        env, robot, _, _, _, _ = _env(n=32)
        rsi.reset_from_reference_states_omni(env, torch.arange(32), path, fraction=1.0,
                                             source=source)
        _, jp, _ = robot.writes["joints"]
        assert torch.allclose(jp, torch.full_like(jp, joint_value)), f"source={source!r}"


def test_source_all_draws_from_both_groups(bank):
    path, _, _ = bank
    env, robot, _, _, _, _ = _env(n=256)
    torch.manual_seed(2)
    rsi.reset_from_reference_states_omni(env, torch.arange(256), path, fraction=1.0, source="all")
    _, jp, _ = robot.writes["joints"]
    values = set(jp[:, 0].tolist())
    assert values == {0.0, 1.0}


def test_unknown_source_tag_raises(bank):
    path, _, _ = bank
    env, _, _, _, _, _ = _env()
    with pytest.raises(ValueError):
        rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0, source=7)
    with pytest.raises(ValueError):
        rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0, source="fwd")


def test_bank_without_source_is_all_forward_with_a_warning(tmp_path, capsys):
    path, _, _ = _make_bank(tmp_path, with_source=False, name="nosource.npz")
    env, robot, _, _, _, _ = _env(n=32)
    rsi.reset_from_reference_states_omni(env, torch.arange(32), path, fraction=1.0, source=0)
    out = capsys.readouterr().out
    assert "no 'source' array" in out
    b = rsi._load_bank(path, "cpu")
    assert torch.equal(b["source"], torch.zeros(4, dtype=torch.long))
    # ... so the existing forward bank works for stage 1a, and asking for sideways fails loud
    with pytest.raises(ValueError):
        rsi.reset_from_reference_states_omni(env, torch.arange(32), path, fraction=1.0, source=1)


def test_fraction_zero_is_a_no_op(bank):
    path, _, _ = bank
    env, robot, _, action_term, cmd_term, parkour_term = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=0.0)
    assert robot.writes == {}
    assert torch.isnan(cmd_term.rsi_cmd_staged).all()
    assert torch.isnan(action_term.rsi_clock_staged).all()
    assert parkour_term.calls == []


def test_placement_is_the_tile_centre_at_the_bank_height(bank):
    path, _, _ = bank
    env, robot, origins, _, _, _ = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0)
    _, pose = robot.writes["pose"]
    assert torch.allclose(pose[:, 0], origins[:, 0])
    assert torch.allclose(pose[:, 1], origins[:, 1])
    bank_z = rsi._load_bank(path, "cpu")["root_z"]
    for z in pose[:, 2]:
        assert torch.isclose(bank_z, z, atol=1e-6).any()


def test_clock_phase_is_staged_on_the_action_term(bank):
    path, _, _ = bank
    env, _, _, action_term, _, _ = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0)
    phases = action_term.rsi_clock_staged
    assert torch.isfinite(phases).all()
    bank_phase = rsi._load_bank(path, "cpu")["clock_phase"]
    for p in phases:
        assert torch.isclose(bank_phase, p, atol=1e-6).any()


def test_spawn_is_reported_to_the_parkour_term_as_rsi(bank):
    path, _, _ = bank
    env, robot, origins, _, _, parkour_term = _env()
    rsi.reset_from_reference_states_omni(env, torch.arange(6), path, fraction=1.0)
    assert len(parkour_term.calls) == 1
    ids, xy, kind = parkour_term.calls[0]
    assert kind == rsi.SPAWN_RSI == 2
    assert xy.shape == (6, 2)
    assert torch.allclose(xy, origins[:, :2])


def test_only_the_picked_subset_is_written(bank):
    path, _, _ = bank
    env, robot, _, action_term, cmd_term, parkour_term = _env(n=400)
    torch.manual_seed(3)
    rsi.reset_from_reference_states_omni(env, torch.arange(400), path, fraction=0.2)
    ids, _ = robot.writes["pose"]
    n = int(ids.numel())
    assert 0 < n < 400
    staged_rows = torch.isfinite(cmd_term.rsi_cmd_staged).all(dim=1)
    assert int(staged_rows.sum()) == n
    assert torch.equal(torch.nonzero(staged_rows).reshape(-1), ids)
    assert int(torch.isfinite(action_term.rsi_clock_staged).sum()) == n
