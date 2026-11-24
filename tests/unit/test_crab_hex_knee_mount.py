"""Knee-mount switch (2026-11-24) in the crab action term and the current-sense proxy, on CPU.

The owner saw the knee actuator ride on the upper leg: its rear end is anchored on the FEMUR, so on the robot a held
knee rod holds the knee angle while the hip pitches. ``crab_hex_linkage`` carries both mounts and the action term reads
``knee_mount`` off its cfg (unset = the legacy ``"hip_plate"`` every recorded run trained on). Pinned here:

* the hip-plate path of every knee function is bitwise the pre-change module's (commit 1b0b49c9), float32 and float64;
* ``__init__`` reads, validates and exposes the mount (unset -> hip_plate, a misspelling refused);
* ``_apply_pitch_linkage`` (exec'd from ``parkour_actions.py``, which imports Isaac): unset is bitwise ``"hip_plate"``;
  on the femur a held knee rod holds the knee target while the hip slews and the knee velocity feedforward is zero,
  where the hip plate swings the knee by about -0.9 x the hip; at a non-default hip the femur knee channel names the
  knee angle itself, the hip plate's names the angle the knee would have at the default hip;
* the current-sense proxy (``_get_contact_fill``, exec'd likewise) takes the knee lever of the action term's mount.

The hot path's AST identity with the pre-change commit (the mount keywords stripped) is pinned in
``test_rod_speed_dr.py``; the linkage geometry of both mounts in ``test_crab_hex_linkage.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests" / "unit"
FWD_MDP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task" / "mdp"
ACTIONS_PY = FWD_MDP / "parkour_actions.py"
OBSERVATIONS_PY = FWD_MDP / "observations.py"
LINKAGE_PY = FWD_MDP / "crab_hex_linkage.py"

#: The last commit whose ``crab_hex_linkage.py`` predates the knee-mount switch (the legacy identity reference).
PRE_CHANGE_COMMIT = "1b0b49c9"


def _load(path: Path, name: str):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


mkt = _load(TESTS / "test_crab_hex_omni_motion_knobs.py", "test_crab_hex_omni_motion_knobs__knee_mount")
linkage = mkt.linkage
dims = linkage.dims
NS = types.SimpleNamespace
ABSENT = object()

HIP_ROM = tuple(math.radians(v) for v in dims.HIP_SIM_LIMITS_DEG)
KNEE_ROM = tuple(math.radians(v) for v in dims.KNEE_SIM_LIMITS_LEFT_DEG)
# The stub action term's joint ids (``mkt._action_term``): hips 0..5, knees 6..11; action columns 6..11 / 12..17.
HIP_IDS = tuple(range(6))
KNEE_IDS = tuple(range(6, 12))


# ============================================================ A. the hip-plate path is the pre-change linkage
def _legacy_linkage():
    rel = LINKAGE_PY.relative_to(REPO).as_posix()
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_CHANGE_COMMIT}:{rel}"],
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"pre-change commit {PRE_CHANGE_COMMIT} not readable here")
    assert "knee_mount" not in src and "KNEE_MOUNT" not in src, "the reference must predate the switch"
    mod = types.ModuleType("crab_hex_linkage__pre_knee_mount")
    exec(compile(src, f"{PRE_CHANGE_COMMIT}:{rel}", "exec"), mod.__dict__)  # falls back to the flat dims import
    return mod


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_hip_plate_mount_is_bitwise_the_pre_change_linkage(dtype):
    old = _legacy_linkage()
    gen = torch.Generator().manual_seed(11)
    th = (HIP_ROM[0] + (HIP_ROM[1] - HIP_ROM[0]) * torch.rand(64, 6, generator=gen, dtype=torch.float64)).to(dtype)
    tk = (KNEE_ROM[0] + (KNEE_ROM[1] - KNEE_ROM[0]) * torch.rand(64, 6, generator=gen, dtype=torch.float64)).to(dtype)
    lam = old.knee_actuator_length(th, tk)
    for kw in ({}, {"mount": linkage.KNEE_MOUNT_HIP_PLATE}):
        assert torch.equal(linkage.knee_actuator_length(th, tk, **kw), lam)
        assert torch.equal(linkage.knee_moment_arm(th, tk, **kw), old.knee_moment_arm(th, tk))
        assert torch.equal(linkage.knee_hip_coupling(th, tk, **kw), old.knee_hip_coupling(th, tk))
        assert torch.equal(linkage.knee_angle_from_length(lam, th, **kw), old.knee_angle_from_length(lam, th))
        assert torch.equal(linkage.knee_torque_limit(th, tk, **kw), old.knee_torque_limit(th, tk))
        assert torch.equal(linkage.knee_joint_vel_limit(th, tk, **kw), old.knee_joint_vel_limit(th, tk))
    for name in ("hip_default_rad", "knee_default_left_rad", "knee_default_right_rad"):
        assert getattr(linkage, name)() == getattr(old, name)(), name


# ============================================================ B. __init__ reads, validates and exposes the mount
def _init_mount_stmt() -> ast.stmt:
    tree = ast.parse(ACTIONS_PY.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CrabHexDelayedJointPositionAction")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    stmts = [s for s in init.body if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "self.knee_mount"]
    assert len(stmts) == 1, "__init__ must set self.knee_mount exactly once"
    return stmts[0]


@pytest.mark.parametrize("cfg_attrs, want", [
    ({}, "hip_plate"),                          # the shipped cfgs carry no attribute: the legacy plant
    ({"knee_mount": None}, "hip_plate"),
    ({"knee_mount": "hip_plate"}, "hip_plate"),
    ({"knee_mount": "femur"}, "femur"),
])
def test_init_reads_the_mount_off_the_cfg(cfg_attrs, want):
    term = NS()
    code = compile(ast.Module(body=[_init_mount_stmt()], type_ignores=[]), str(ACTIONS_PY), "exec")
    exec(code, {"linkage": linkage, "self": term, "cfg": NS(**cfg_attrs)})
    assert term.knee_mount == want


@pytest.mark.parametrize("bad", ["Femur", "femur_mount", "hip", ""])
def test_init_refuses_a_misspelled_mount(bad):
    code = compile(ast.Module(body=[_init_mount_stmt()], type_ignores=[]), str(ACTIONS_PY), "exec")
    with pytest.raises(ValueError, match="knee mount"):
        exec(code, {"linkage": linkage, "self": NS(), "cfg": NS(knee_mount=bad)})


# ============================================================ C. the hot path, lifted from parkour_actions.py
def _term(n=4, mount=ABSENT):
    """``mkt._action_term`` (the real ``_apply_pitch_linkage`` on a stub), recording velocity targets too."""
    t, _ = mkt._action_term(n)
    vel = {}
    t._asset.vel_targets = vel
    t._asset.set_joint_velocity_target = lambda target, joint_ids: vel.__setitem__(tuple(joint_ids), target.clone())
    if mount is not ABSENT:
        t.knee_mount = mount
    return t


def test_unset_mount_is_bitwise_the_hip_plate_mount_over_many_substeps():
    n = 4
    plain, plate = _term(n), _term(n, linkage.KNEE_MOUNT_HIP_PLATE)
    gen = torch.Generator().manual_seed(3)
    for step in range(60):
        actions = torch.randn(n, 18, generator=gen) * 0.6
        for t in (plain, plate):
            t._processed_actions = actions.clone()
            t._apply_pitch_linkage()
        for attr in ("_hip_rod_len", "_knee_rod_len", "hip_rod_speed", "knee_rod_speed"):
            assert torch.equal(getattr(plain, attr), getattr(plate, attr)), (step, attr)
        for got, ref in ((plain._asset.pos_targets, plate._asset.pos_targets),
                         (plain._asset.vel_targets, plate._asset.vel_targets)):
            assert got.keys() == ref.keys()
            for key in got:
                assert torch.equal(got[key], ref[key]), (step, key)


def test_on_the_femur_a_held_knee_rod_holds_the_knee_while_the_hip_moves():
    """Knee channel at the synced pose (rod held), hip stepped: the femur knee target stays put with a zero velocity
    feedforward; the hip plate's swings by about -0.9 x the hip (the coupling the robot does not have)."""
    n = 2
    femur, plate = _term(n, linkage.KNEE_MOUNT_FEMUR), _term(n, linkage.KNEE_MOUNT_HIP_PLATE)
    actions = torch.zeros(n, 18)
    actions[:, 6:12] = 0.3                                    # hip step (rad); knee channel 0 = the synced knee
    for t in (femur, plate):
        t._processed_actions = actions.clone()
        for _ in range(200):                                  # 1 s of 5 ms substeps: the hip rod slews the whole time
            t._apply_pitch_linkage()
    hip = femur._asset.pos_targets[HIP_IDS]
    assert torch.equal(hip, plate._asset.pos_targets[HIP_IDS])  # the hip itself does not depend on the knee mount
    assert float(hip.min()) > 0.15, "premise: the hip really moved"
    assert torch.equal(femur.knee_rod_speed, torch.zeros_like(femur.knee_rod_speed))  # the knee rod never moved
    assert float(femur._asset.pos_targets[KNEE_IDS].abs().max()) < 1e-5
    assert float(femur._asset.vel_targets[KNEE_IDS].abs().max()) == 0.0
    knee_plate = plate._asset.pos_targets[KNEE_IDS]
    ratio = knee_plate / hip
    assert bool(((ratio > -1.1) & (ratio < -0.6)).all()), ratio
    assert float(plate._asset.vel_targets[KNEE_IDS].abs().min()) > 0.0  # its feedforward carries the coupling


def test_at_a_non_default_hip_the_femur_knee_channel_is_the_knee_angle():
    """Live pose hip 0.5 / knee 0.6, command the same. On the femur the reset sync and the command agree, so nothing
    slews and the targets are the live pose. On the hip plate the knee channel is read at the DEFAULT hip (0 here),
    so the same 0.6 asks for another rod length and the knee slews well away from 0.6."""
    n = 2
    femur, plate = _term(n, linkage.KNEE_MOUNT_FEMUR), _term(n, linkage.KNEE_MOUNT_HIP_PLATE)
    for t in (femur, plate):
        t._asset.data.joint_pos[:, list(HIP_IDS)] = 0.5
        t._asset.data.joint_pos[:, list(KNEE_IDS)] = 0.6
        a = torch.zeros(n, 18)
        a[:, 6:12], a[:, 12:18] = 0.5, 0.6
        t._processed_actions = a
        t._apply_pitch_linkage()
    assert torch.equal(femur.knee_rod_speed, torch.zeros_like(femur.knee_rod_speed))
    assert torch.allclose(femur._asset.pos_targets[KNEE_IDS], torch.full((n, 6), 0.6), atol=1e-5)
    assert torch.allclose(femur._asset.pos_targets[HIP_IDS], torch.full((n, 6), 0.5), atol=1e-5)
    for _ in range(400):
        plate._apply_pitch_linkage()
    assert float((plate._asset.pos_targets[KNEE_IDS] - 0.6).abs().min()) > 0.2


# ============================================================ D. the current-sense proxy follows the mount
def _module_constants(path: Path, names: tuple[str, ...]) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                out[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(out) == set(names), names
    return out


_CONSTS = ("_CURRENT_SENSE_NO_LOAD", "_CURRENT_SENSE_GATE_FRACTION", "_CURRENT_SENSE_MAX")


def _contact_fill():
    ns = {"torch": torch, "linkage": linkage, **_module_constants(OBSERVATIONS_PY, _CONSTS)}
    src = mkt._source_of(OBSERVATIONS_PY, cls="CrabHexParkourObservations", name="_get_contact_fill")
    exec(compile("from __future__ import annotations\n" + src, str(OBSERVATIONS_PY), "exec"), ns)
    return ns["_get_contact_fill"], ns


def test_the_current_proxy_takes_the_knee_lever_of_the_action_terms_mount():
    fill, ns = _contact_fill()
    n = 3
    jp = torch.zeros(n, 12)
    jp[:, :6], jp[:, 6:] = 0.6, 0.4                           # hip far from its default: the levers differ there
    tau = torch.zeros(n, 12)
    tau[:, 6:] = 20.0                                         # knee-loaded, hip idle
    def run(mount):
        attrs = {} if mount is ABSENT else {"knee_mount": mount}
        term = NS(hip_rod_speed=torch.zeros(n, 6), knee_rod_speed=torch.full((n, 6), linkage.KNEE_SPEED_M_S), **attrs)
        obs = NS(contact_sensor=object(), _num_contact=6, num_envs=n, device="cpu",
                 _env=NS(action_manager=NS(get_term=lambda name: term)),
                 asset=NS(data=NS(applied_torque=tau, joint_pos=jp)),
                 _pitch_joint_ids=list(HIP_IDS), _knee_joint_ids=list(range(6, 12)))
        return fill(obs)
    unset, plate, femur = run(ABSENT), run(linkage.KNEE_MOUNT_HIP_PLATE), run(linkage.KNEE_MOUNT_FEMUR)
    assert torch.equal(unset, plate)
    ma_k = linkage.knee_femur_moment_arm(jp[:, 6:].abs()).clamp_min(1e-4)
    force_k = tau[:, 6:].abs() / ma_k / linkage.KNEE_FORCE_N
    want = (ns["_CURRENT_SENSE_NO_LOAD"] + force_k).clamp(0.0, ns["_CURRENT_SENSE_MAX"]) - 0.5
    assert torch.allclose(femur, want, atol=1e-6)
    assert float((femur - plate).abs().min()) > 1e-3
