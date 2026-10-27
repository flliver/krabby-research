"""RSI cam staging and the spawn-kind tracking split (PPO-construction plan steps 2 and 4), on CPU.

Two env-side pieces of the PPO-construction plan, both DEFAULT OFF or additive:

* ``KRABBY_RSI_CAM_STAGE`` (step 4): an RSI-seeded env starts with the bank frame's cam shaft
  velocity as the applied cam target and in the cam columns of its action history. The event
  (``crab_hex_omni_rsi.reset_from_reference_states_omni``) only STAGES the values; the action
  term's reset -- which runs after the event -- applies them. Getting that order wrong is quiet:
  the action manager's reset would zero what the event wrote, and the run would train exactly as
  if staging were off. So the reset is driven here from the REAL method sources (lifted with
  ``ast``, the ``test_crab_hex_omni_motion_knobs.py`` pattern) and the event is the real module.
* ``track_ratio_mean_rsi`` / ``_norsi`` (step 2): the tracking ratio split by spawn kind over ALL
  terrain, absent (never 0.0) while its group is empty.

Also pinned: knob off is bitwise the pre-change reset and writes nothing; ``_repoint_rsi`` carries
the flags only when armed; the parser / whitelist / dump-check registration with the ladder.
"""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.util
import math
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
OMNI = TASKS / "crab_hex_omni_task"
OMNI_MDP = OMNI / "mdp"
PARKOURS_DIR = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
ACTIONS_PY = TASKS / "crab_hex_forward_task/mdp/parkour_actions.py"
JOINT_ACTIONS_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/parkour_actions/joint_actions.py"
MOTION_KNOBS_PY = OMNI / "config/crab_hex/motion_knobs.py"
OMNI_KNOBS_PY = OMNI / "config/crab_hex/omni_knobs.py"
PARKOUR_EVENT_PY = OMNI_MDP / "omni_parkour_event.py"
RUN_CANDIDATES_PY = OMNI / "experiments/tools/run_candidates.py"
#: The commit this worktree branched from: ``CrabHexDelayedJointPositionAction.reset`` as it was
#: before the staging existed (the knob-off identity reference).
PRE_CHANGE_COMMIT = "fd255444"

PI = math.pi
N_ACT = 18
#: Deliberately NOT contiguous and not the leading columns: the staging must write the term's
#: own cam columns, not "the first six".
CAM_COLS = [1, 4, 7, 10, 13, 16]
ROD_COLS = [c for c in range(N_ACT) if c not in CAM_COLS]
HIST = 4


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _method_src(src: str, cls: str, name: str) -> str:
    tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    node = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    return textwrap.dedent(" " * node.col_offset + ast.get_source_segment(src, node))


def _term_class(actions_src: str, methods: tuple[str, ...]):
    """A class whose BASE ``reset`` is ``DelayedJointPositionAction.reset`` and whose methods are
    the crab-hex term's, both lifted from their real sources (``super()`` then resolves as in the
    real class hierarchy)."""
    base = _method_src(JOINT_ACTIONS_PY.read_text(), "DelayedJointPositionAction", "reset")
    body = "\n".join(textwrap.indent(_method_src(actions_src, "CrabHexDelayedJointPositionAction", m), "    ")
                     for m in methods)
    code = ("from __future__ import annotations\n"
            "class _Base:\n" + textwrap.indent(base, "    ") + "\n"
            "class _Term(_Base):\n" + body + "\n")
    ns = {"torch": torch, "math": math, "Sequence": object}
    exec(compile(code, str(ACTIONS_PY), "exec"), ns)
    return ns["_Term"]


STAGE_METHODS = ("reset", "_cam_raw_bounds", "_apply_rsi_cam_stage", "_report_rsi_cam_stage")
Term = _term_class(ACTIONS_PY.read_text(), STAGE_METHODS)


def _make_term(cls=Term, n=5, clip=None, garbage=True):
    """The attributes ``reset`` touches, shaped like the real term's (``__init__`` needs Isaac).

    ``garbage`` fills the per-env state with non-zero values, so "reset to zero" and "left alone"
    are distinguishable.
    """
    t = cls.__new__(cls)
    t.num_envs, t.device = n, "cpu"
    t.cfg = types.SimpleNamespace(clip=clip)
    t._cam_action_cols = list(CAM_COLS)
    t._cam_shaft_joint_ids = [3, 7, 11, 15, 19, 23]
    scale = torch.full((n, N_ACT), 0.25)
    scale[:, CAM_COLS] = PI
    t._scale = scale
    if clip is not None:
        t._clip = torch.tensor(clip, dtype=torch.float32).expand(n, N_ACT, 2).clone()
    fill = (lambda *s: torch.randn(*s)) if garbage else (lambda *s: torch.zeros(*s))
    t._raw_actions = fill(n, N_ACT)
    t._action_history_buf = fill(n, HIST, N_ACT)
    t._cam_vel_applied = fill(n, len(CAM_COLS))
    t._rod_sync_needed = torch.zeros(n, dtype=torch.bool)
    t.clock_phase = torch.zeros(n)
    t.rsi_clock_staged = torch.full((n,), float("nan"))
    t.rsi_cam_vel_staged = torch.full((n, len(CAM_COLS)), float("nan"))
    t.rsi_cam_stage_pending = False
    t.rsi_cam_debug_jv = torch.full((n, len(CAM_COLS)), float("nan"))
    t.rsi_cam_debug_pending = False
    return t


def _clone_state(t) -> dict:
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in vars(t).items()}


JV_ENV1 = torch.tensor([4.0, -5.0, 1.2, -0.3, 0.0, PI])          # two beyond the +-pi target band
JV_ENV3 = torch.tensor([-1.0, 2.5, -3.5, 0.7, 3.0, -0.01])


# ============================================================ the action term's reset (lifted)
def test_staged_envs_get_the_clamped_target_and_every_history_slot():
    torch.manual_seed(0)
    t = _make_term()
    before = _clone_state(t)
    t.rsi_cam_vel_staged[1] = JV_ENV1
    t.rsi_cam_vel_staged[3] = JV_ENV3
    t.rsi_cam_stage_pending = True
    t.reset(torch.tensor([0, 1, 2, 3]))

    for env, jv in ((1, JV_ENV1), (3, JV_ENV3)):
        assert torch.equal(t._cam_vel_applied[env], jv.clamp(-PI, PI)), env          # rad/s, exact
        want = (jv / PI).clamp(-1.0, 1.0)
        for slot in range(HIST):                                                      # EVERY slot
            assert torch.equal(t._action_history_buf[env, slot, CAM_COLS], want), (env, slot)
        assert torch.equal(t._action_history_buf[env][:, ROD_COLS], torch.zeros(HIST, len(ROD_COLS)))
    # the 4.0 / -5.0 rad/s entries really were clamped (the band is the full-action speed)
    assert t._cam_vel_applied[1, 0] == pytest.approx(PI) and t._cam_vel_applied[1, 1] == pytest.approx(-PI)
    assert t._action_history_buf[1, 0, CAM_COLS[0]] == 1.0
    # unstaged resetting envs: exactly the plain reset (zeros)
    for env in (0, 2):
        assert torch.equal(t._cam_vel_applied[env], torch.zeros(len(CAM_COLS)))
        assert torch.equal(t._action_history_buf[env], torch.zeros(HIST, N_ACT))
    # _raw_actions is zeroed for every resetting env (the first process_actions overwrites it)
    assert torch.equal(t._raw_actions[:4], torch.zeros(4, N_ACT))
    # a non-resetting env is untouched
    assert torch.equal(t._cam_vel_applied[4], before["_cam_vel_applied"][4])
    assert torch.equal(t._action_history_buf[4], before["_action_history_buf"][4])
    # consumed once: the staging buffer is cleared and the flag dropped
    assert torch.isnan(t.rsi_cam_vel_staged).all()
    assert t.rsi_cam_stage_pending is False


def test_a_partially_finite_row_is_not_staged():
    """A row is staged only when all six cam velocities are finite (NaN = not staged)."""
    t = _make_term()
    row = JV_ENV1.clone()
    row[2] = float("nan")
    t.rsi_cam_vel_staged[1] = row
    t.rsi_cam_stage_pending = True
    t.reset(torch.tensor([1]))
    assert torch.equal(t._cam_vel_applied[1], torch.zeros(len(CAM_COLS)))
    assert torch.equal(t._action_history_buf[1], torch.zeros(HIST, N_ACT))


def test_the_terms_own_clip_narrows_the_staged_raw_action():
    """``KRABBY_CAM_CLIP_LO``-style asymmetric cam clip: the history holds CLIPPED raw actions, so
    the staged value respects it (and the applied target is the speed that raw action commands)."""
    clip = [[0.1, 1.0]] * N_ACT
    t = _make_term(clip=clip)
    t.rsi_cam_vel_staged[0] = JV_ENV3
    t.rsi_cam_stage_pending = True
    t.reset(torch.tensor([0]))
    lo = torch.tensor(0.1)
    assert torch.equal(t._action_history_buf[0, -1, CAM_COLS], torch.minimum(torch.maximum(JV_ENV3 / PI, lo), torch.tensor(1.0)))
    assert torch.equal(t._cam_vel_applied[0], torch.minimum(torch.maximum(JV_ENV3, lo * PI), torch.tensor(PI)))
    # a symmetric +-1 clip is the spec exactly
    t2 = _make_term(clip=[[-1.0, 1.0]] * N_ACT)
    t2.rsi_cam_vel_staged[0] = JV_ENV1
    t2.rsi_cam_stage_pending = True
    t2.reset(torch.tensor([0]))
    assert torch.equal(t2._cam_vel_applied[0], JV_ENV1.clamp(-PI, PI))
    assert torch.equal(t2._action_history_buf[0, 0, CAM_COLS], (JV_ENV1 / PI).clamp(-1, 1))


def test_reset_with_all_envs_none():
    t = _make_term()
    t.rsi_cam_vel_staged[2] = JV_ENV3
    t.rsi_cam_stage_pending = True
    t.reset(None)
    assert torch.equal(t._cam_vel_applied[2], JV_ENV3.clamp(-PI, PI))
    assert torch.equal(t._cam_vel_applied[[0, 1, 3, 4]], torch.zeros(4, len(CAM_COLS)))


def test_knob_off_never_reads_or_writes_the_staging_buffers():
    """Flag down (every run without the knob): even a finite row left in the buffer is NOT applied
    and NOT cleared -- reset() does not touch the staging state at all."""
    t = _make_term()
    stale = torch.randn(5, len(CAM_COLS))
    t.rsi_cam_vel_staged = stale.clone()
    t.rsi_cam_debug_jv = stale.clone()
    t.reset(torch.tensor([0, 1, 2]))
    assert torch.equal(t.rsi_cam_vel_staged, stale) and torch.equal(t.rsi_cam_debug_jv, stale)
    assert torch.equal(t._cam_vel_applied[:3], torch.zeros(3, len(CAM_COLS)))
    assert torch.equal(t._action_history_buf[:3], torch.zeros(3, HIST, N_ACT))


def _pre_change_source() -> str:
    rel = ACTIONS_PY.relative_to(REPO).as_posix()
    try:
        out = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_CHANGE_COMMIT}:{rel}"],
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"pre-change commit {PRE_CHANGE_COMMIT} not readable here")
    assert "rsi_cam" not in out.stdout, "the reference must predate the staging"
    return out.stdout


def test_knob_off_reset_is_bitwise_the_pre_change_reset():
    old_cls = _term_class(_pre_change_source(), ("reset",))
    for ids in (torch.tensor([0, 2, 4]), None, [1, 3]):
        torch.manual_seed(7)
        new = _make_term()
        torch.manual_seed(7)
        old = _make_term(cls=old_cls)
        torch.manual_seed(11)
        new.reset(ids)
        torch.manual_seed(11)
        old.reset(ids)
        a, b = _clone_state(new), _clone_state(old)
        for key in b:
            if isinstance(b[key], torch.Tensor):
                assert torch.equal(a[key], b[key]) or (torch.isnan(a[key]).all() and torch.isnan(b[key]).all()), key
    torch.manual_seed(5)
    t_new = _make_term()
    torch.manual_seed(5)
    t_old = _make_term(cls=old_cls)
    torch.manual_seed(9)
    t_new.reset(torch.tensor([0, 1]))
    after_new = torch.rand(4)
    torch.manual_seed(9)
    t_old.reset(torch.tensor([0, 1]))
    after_old = torch.rand(4)
    assert torch.equal(after_new, after_old), "the knob-off reset consumed a different number of draws"


def test_init_allocates_the_buffers_unstaged_and_the_flags_down():
    """``__init__`` needs Isaac, so its staging lines are pinned from the source."""
    src = _method_src(ACTIONS_PY.read_text(), "CrabHexDelayedJointPositionAction", "__init__")
    for line in ('self.rsi_cam_vel_staged = torch.full((self.num_envs, n_cam), float("nan"), device=self.device)',
                 "self.rsi_cam_stage_pending = False",
                 'self.rsi_cam_debug_jv = torch.full((self.num_envs, n_cam), float("nan"), device=self.device)',
                 "self.rsi_cam_debug_pending = False",
                 "n_cam = len(self._cam_action_cols)"):
        assert line in src, line
    # the staging happens AFTER the plain zeroing of the applied target in reset()
    reset = _method_src(ACTIONS_PY.read_text(), "CrabHexDelayedJointPositionAction", "reset")
    assert reset.index("self._cam_vel_applied[env_ids] = 0.0") < reset.index("self._apply_rsi_cam_stage(env_ids)")


# ------------------------------------------------------------------------------ debug line
def _kv(line: str) -> dict:
    return dict(kv.split("=", 1) for kv in line.split()[1:])


def test_debug_line_reports_the_staged_step0_state(capsys):
    t = _make_term()
    for env, jv in ((1, JV_ENV1), (3, JV_ENV3)):
        t.rsi_cam_vel_staged[env] = jv
        t.rsi_cam_debug_jv[env] = jv
    t.rsi_cam_stage_pending = t.rsi_cam_debug_pending = True
    t.reset(torch.tensor([0, 1, 2, 3]))
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("RSI_CAM_STAGE_DEBUG n_rsi=2 stage=on ")
    kv = _kv(out[0])
    jv = torch.stack((JV_ENV1, JV_ENV3))
    assert float(kv["target_err"]) == 0.0 and float(kv["hist_err"]) == 0.0           # per env, exact
    assert float(kv["target_absmax"]) == pytest.approx(PI, abs=1e-5)                  # clamped at +-pi
    assert float(kv["hist_absmax"]) == pytest.approx(1.0)
    assert float(kv["bank_absmax"]) == pytest.approx(float(jv.abs().max()), abs=1e-5)
    assert torch.isnan(t.rsi_cam_debug_jv).all() and t.rsi_cam_debug_pending is False


def test_debug_line_with_staging_off_reads_zeros(capsys):
    """The rung-0b staging-off rows: the same read-out, zeros where staging would have written."""
    t = _make_term()
    t.rsi_cam_debug_jv[1] = JV_ENV1
    t.rsi_cam_debug_pending = True
    t.reset(torch.tensor([1, 2]))
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("RSI_CAM_STAGE_DEBUG n_rsi=1 stage=off ")
    kv = _kv(out[0])
    assert float(kv["target_absmax"]) == 0.0 and float(kv["hist_absmax"]) == 0.0
    assert float(kv["bank_absmax"]) == pytest.approx(5.0)
    assert float(kv["target_err"]) == pytest.approx(PI, abs=1e-5)                       # |0 - clamp(-5)|
    assert float(kv["hist_err"]) == pytest.approx(1.0)
    assert torch.equal(t._cam_vel_applied[1], torch.zeros(len(CAM_COLS)))


def test_debug_line_catches_a_per_env_mismatch_that_means_would_hide(capsys):
    """Two staged envs whose errors cancel in a mean: the per-env max still reports them."""
    t = _make_term()
    for env, jv in ((1, JV_ENV1), (3, JV_ENV3)):
        t.rsi_cam_vel_staged[env] = jv
        t.rsi_cam_debug_jv[env] = jv
    t.rsi_cam_debug_jv[1] = JV_ENV3             # the read-out compares env 1 with the wrong frame
    t.rsi_cam_debug_jv[3] = JV_ENV1
    t.rsi_cam_stage_pending = t.rsi_cam_debug_pending = True
    t.reset(torch.tensor([1, 3]))
    kv = _kv(capsys.readouterr().out.strip())
    assert float(kv["target_err"]) > 1.0 and float(kv["hist_err"]) > 0.3


def _rung0b_read():
    path = OMNI / "experiments/2026-10-01_ppo_construction/scripts/rung0b_read.py"
    return _load(path, "_rung0b_read_under_test")


def test_rung0b_reader_verifies_the_envs_real_debug_output(capsys, tmp_path):
    """The env's own line, fed to the campaign reader (F4: the two used to disagree, so every row
    read UNVERIFIED). The [KNOB] consumed line names the knob and must never be parsed."""
    r0b = _rung0b_read()
    knob = "[KNOB] KRABBY_RSI_CAM_STAGE=1 consumed\n[KNOB] KRABBY_RSI_CAM_STAGE_DEBUG=1 consumed\n"
    # staging ON
    t = _make_term()
    for env, jv in ((1, JV_ENV1), (3, JV_ENV3)):
        t.rsi_cam_vel_staged[env] = jv
        t.rsi_cam_debug_jv[env] = jv
    t.rsi_cam_stage_pending = t.rsi_cam_debug_pending = True
    t.reset(torch.tensor([0, 1, 2, 3]))
    on_log = tmp_path / "on.log"
    on_log.write_text(knob + "some kit noise\n" + capsys.readouterr().out)
    v = r0b.verify_log(on_log, "rsi_on")
    assert v["status"] == "OK", v
    assert all(l.startswith("RSI_CAM_STAGE_DEBUG") for l in v["lines"])
    # the same staging-on log read as a staging-OFF row fails (absmax != 0)
    assert r0b.verify_log(on_log, "rsi_off")["status"] == "FAIL"
    # staging OFF
    t = _make_term()
    t.rsi_cam_debug_jv[1] = JV_ENV1
    t.rsi_cam_debug_pending = True
    t.reset(torch.tensor([1, 2]))
    off_log = tmp_path / "off.log"
    off_log.write_text(knob + capsys.readouterr().out)
    assert r0b.verify_log(off_log, "rsi_off")["status"] == "OK"
    assert r0b.verify_log(off_log, "rsi_on")["status"] == "FAIL"
    # a log holding only the [KNOB] lines (the debug never fired) is UNVERIFIED, never OK
    knob_only = tmp_path / "knob.log"
    knob_only.write_text(knob)
    assert r0b.verify_log(knob_only, "rsi_on")["status"] == "UNVERIFIED"
    assert r0b.verify_log(knob_only, "rsi_off")["status"] == "UNVERIFIED"
    # FRAC-0 rows: no RSI env, no line -> OK (nothing to stage); a line there is a FAIL
    assert r0b.verify_log(knob_only, "frac0_on")["status"] == "OK"
    assert r0b.verify_log(off_log, "frac0_off")["status"] == "FAIL"


def test_no_debug_line_without_the_debug_flag(capsys):
    t = _make_term()
    t.rsi_cam_vel_staged[1] = JV_ENV1
    t.rsi_cam_stage_pending = True
    t.reset(torch.tensor([1]))
    assert capsys.readouterr().out == ""


# ============================================================== the omni RSI event (real module)
def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """``test_crab_hex_omni_rsi.py``'s loader: the real file under its dotted name, parents created
    with their real source directories as ``__path__`` (their ``__init__`` files reach Isaac)."""
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


_load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
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

PKG = "_omni_mdp_cam_stage_under_test"
if PKG not in sys.modules:
    _spec = importlib.machinery.ModuleSpec(PKG, loader=None, is_package=True)
    _spec.submodule_search_locations = [str(OMNI_MDP)]
    sys.modules[PKG] = importlib.util.module_from_spec(_spec)
rsi = importlib.import_module(f"{PKG}.crab_hex_omni_rsi")
ox = importlib.import_module(f"{PKG}.omni_exposure")

N_JOINTS = 24
N_FRAMES = 4


class _Robot:
    def __init__(self, n):
        default = torch.zeros(n, 13)
        default[:, 2] = 0.30
        default[:, 3] = 1.0
        self.data = types.SimpleNamespace(default_root_state=default)

    def write_root_pose_to_sim(self, pose, env_ids):
        pass

    def write_root_velocity_to_sim(self, vel, env_ids):
        pass

    def write_joint_state_to_sim(self, jp, jv, env_ids):
        self.jv = (env_ids.clone(), jv.clone())


class _Scene:
    def __init__(self, robot, n):
        self._items = {"robot": robot}
        self.env_origins = torch.zeros(n, 3)

    def __getitem__(self, key):
        return self._items[key]


def _env(action_term, n):
    cmd_term = types.SimpleNamespace(cfg=types.SimpleNamespace(speed_ellipse=(0.35, 0.25, 0.25)),
                                     rsi_cmd_staged=torch.full((n, 3), float("nan")))
    return types.SimpleNamespace(
        device="cpu", scene=_Scene(_Robot(n), n),
        action_manager=types.SimpleNamespace(get_term=lambda name: action_term),
        command_manager=types.SimpleNamespace(get_term=lambda name: cmd_term),
    )


@pytest.fixture()
def bank(tmp_path):
    """Frame f's joint j velocity is ``(f + 1) * 0.5 - 0.02 * j`` rad/s (distinct per frame and
    joint; frame 3's early joints exceed pi), and its clock phase ``0.1 * (f + 1)`` identifies it."""
    f = torch.arange(N_FRAMES, dtype=torch.float32).view(-1, 1)
    j = torch.arange(N_JOINTS, dtype=torch.float32).view(1, -1)
    jv = (f + 1) * 0.5 - 0.02 * j
    jv[3, :] *= 2.0
    arrays = dict(
        joint_pos=np.zeros((N_FRAMES, N_JOINTS), np.float32),
        joint_vel=jv.numpy().astype(np.float32),
        root_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (N_FRAMES, 1)),
        root_z=np.full(N_FRAMES, 0.3, np.float32),
        root_lin_vel_b=np.tile(np.array([0.3, 0.0, 0.0], np.float32), (N_FRAMES, 1)),
        clock_phase=(0.1 * (np.arange(N_FRAMES) + 1)).astype(np.float32),
        source=np.zeros(N_FRAMES, np.int8),
    )
    path = tmp_path / "bank.npz"
    np.savez(path, **arrays)
    rsi._BANK_CACHE.clear()
    return str(path), jv


def _frame_of(phase: torch.Tensor) -> torch.Tensor:
    return torch.round(phase / 0.1).long() - 1


def test_event_stages_the_frames_cam_velocities_in_shaft_order(bank):
    path, jv = bank
    n = 6
    t = _make_term(n=n)
    rsi.reset_from_reference_states_omni(_env(t, n), torch.arange(n), path, fraction=1.0, stage_cam=True)
    frames = _frame_of(t.rsi_clock_staged)
    assert torch.equal(t.rsi_cam_vel_staged, jv[frames][:, t._cam_shaft_joint_ids])
    assert t.rsi_cam_stage_pending is True
    assert torch.isnan(t.rsi_cam_debug_jv).all() and t.rsi_cam_debug_pending is False


def test_event_without_the_param_touches_no_cam_state(bank):
    """Knob off: the param is absent, and the event reads no cam attribute at all (a term without
    the staging buffers -- the shipped event's view of it -- still works)."""
    path, _ = bank
    term = types.SimpleNamespace(rsi_clock_staged=torch.full((4,), float("nan")))
    rsi.reset_from_reference_states_omni(_env(term, 4), torch.arange(4), path, fraction=1.0)
    assert set(vars(term)) == {"rsi_clock_staged"}
    t = _make_term(n=4)
    before = _clone_state(t)
    rsi.reset_from_reference_states_omni(_env(t, 4), torch.arange(4), path, fraction=1.0, stage_cam=False)
    assert torch.isnan(t.rsi_cam_vel_staged).all() and t.rsi_cam_stage_pending is False
    assert torch.equal(t._cam_vel_applied, before["_cam_vel_applied"])


def test_event_fraction_zero_stages_nothing(bank):
    path, _ = bank
    t = _make_term(n=4)
    rsi.reset_from_reference_states_omni(_env(t, 4), torch.arange(4), path, fraction=0.0,
                                         stage_cam=True, stage_cam_debug=True)
    assert torch.isnan(t.rsi_cam_vel_staged).all() and not t.rsi_cam_stage_pending
    assert torch.isnan(t.rsi_cam_debug_jv).all() and not t.rsi_cam_debug_pending


def test_event_debug_alone_feeds_the_read_out_only(bank):
    path, jv = bank
    t = _make_term(n=3)
    rsi.reset_from_reference_states_omni(_env(t, 3), torch.arange(3), path, fraction=1.0, stage_cam_debug=True)
    assert torch.isnan(t.rsi_cam_vel_staged).all() and not t.rsi_cam_stage_pending
    frames = _frame_of(t.rsi_clock_staged)
    assert torch.equal(t.rsi_cam_debug_jv, jv[frames][:, t._cam_shaft_joint_ids])
    assert t.rsi_cam_debug_pending


def test_event_then_reset_end_to_end_only_rsi_rows_are_seeded(bank):
    """The env's reset order -- event first, then the action term -- with a Bernoulli subset:
    seeded envs carry their frame's cam state, every other resetting env is the plain reset."""
    path, jv = bank
    n = 200
    torch.manual_seed(1)
    t = _make_term(n=n)
    ids = torch.arange(n)
    rsi.reset_from_reference_states_omni(_env(t, n), ids, path, fraction=0.3, stage_cam=True)
    picked = torch.isfinite(t.rsi_clock_staged)
    assert 0 < int(picked.sum()) < n
    frames = _frame_of(t.rsi_clock_staged[picked])
    t.reset(ids)
    want = jv[frames][:, t._cam_shaft_joint_ids]
    assert torch.equal(t._cam_vel_applied[picked], want.clamp(-PI, PI))
    assert bool((want.abs() > PI).any()), "the bank fixture must exercise the clamp"
    assert torch.equal(t._action_history_buf[picked][:, :, CAM_COLS],
                       (want / PI).clamp(-1, 1).unsqueeze(1).expand(-1, HIST, -1))
    assert torch.equal(t._cam_vel_applied[~picked], torch.zeros(int((~picked).sum()), len(CAM_COLS)))
    assert torch.equal(t._action_history_buf[~picked], torch.zeros(int((~picked).sum()), HIST, N_ACT))
    assert torch.isnan(t.rsi_cam_vel_staged).all()


# ======================================================================= the knob layer
mk = _load(MOTION_KNOBS_PY, "motion_knobs__cam_stage_test")


def test_parsers_and_registry():
    assert mk.RSI_CAM_STAGE_VAR == "KRABBY_RSI_CAM_STAGE"
    assert mk.RSI_CAM_STAGE_DEBUG_VAR == "KRABBY_RSI_CAM_STAGE_DEBUG"
    assert mk.RSI_CAM_STAGE_PARAMS == {"KRABBY_RSI_CAM_STAGE": "stage_cam",
                                       "KRABBY_RSI_CAM_STAGE_DEBUG": "stage_cam_debug"}
    for var in mk.RSI_CAM_STAGE_PARAMS:
        parse = mk.MOTION_KNOB_PARSERS[var]
        assert parse("1") is True and parse("0") is False and parse("") is False and parse(None) is False
        for bad in ("2", "yes", "true", "on"):
            with pytest.raises(ValueError):
                parse(bad)
        assert var not in mk.AGENT_KNOBS                      # env side: params/env.yaml
        assert var not in mk.PPO_CONSTRUCTION_KNOBS
        assert var not in mk.KNOB_OFF_VALUES                  # never carried into an eval
    assert mk.rsi_cam_stage_params({}) == {}
    assert mk.rsi_cam_stage_params({"KRABBY_RSI_CAM_STAGE": "0", "KRABBY_RSI_CAM_STAGE_DEBUG": " "}) == {}
    assert mk.rsi_cam_stage_params({"KRABBY_RSI_CAM_STAGE": "1"}) == {"stage_cam": True}
    assert mk.rsi_cam_stage_params({"KRABBY_RSI_CAM_STAGE": "1", "KRABBY_RSI_CAM_STAGE_DEBUG": "1"}) == {
        "stage_cam": True, "stage_cam_debug": True}
    for name in ("RSI_CAM_STAGE_VAR", "RSI_CAM_STAGE_DEBUG_VAR", "RSI_CAM_STAGE_PARAMS", "rsi_cam_stage_params"):
        assert name in mk.__all__


@pytest.fixture(scope="module")
def knobs():
    return _load(OMNI_KNOBS_PY, "omni_knobs__cam_stage_test")


_STUB_RSI_FUNC = object()


@pytest.fixture()
def stub_omni_mdp(monkeypatch):
    """The two lazy imports of ``_repoint_rsi`` (both reach Isaac through their packages)."""
    root = "parkour_tasks.crab_hex_omni_task.mdp"
    rsi_mod = types.ModuleType(f"{root}.crab_hex_omni_rsi")
    rsi_mod.reset_from_reference_states_omni = _STUB_RSI_FUNC
    cmd_mod = types.ModuleType(f"{root}.crab_hex_omni_command")
    cmd_mod.omni_walk_floor = lambda cmd_cfg: 0.21
    mdp = types.ModuleType(root)
    mdp.crab_hex_omni_rsi, mdp.crab_hex_omni_command = rsi_mod, cmd_mod
    pkgs = [types.ModuleType("parkour_tasks"), types.ModuleType("parkour_tasks.crab_hex_omni_task"), mdp]
    for pkg in pkgs:
        pkg.__path__ = []
    for mod in (*pkgs, rsi_mod, cmd_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)


def _rsi_cfg(with_term=True):
    events = types.SimpleNamespace()
    if with_term:
        events.rsi_reference_reset = types.SimpleNamespace(
            func="forward", params={"bank_path": "/b/bank.npz", "fraction": 0.2, "fix_spawn": False})
    return types.SimpleNamespace(events=events, commands=types.SimpleNamespace(base_velocity=object()))


STOCK_PARAMS = {"bank_path", "fraction", "yaw_range", "walk_floor", "source"}


def test_repoint_rsi_writes_no_flag_while_off(knobs, stub_omni_mdp, capsys):
    for env in ({}, {"KRABBY_RSI_CAM_STAGE": ""}, {"KRABBY_RSI_CAM_STAGE": "0", "KRABBY_RSI_CAM_STAGE_DEBUG": "0"}):
        cfg = _rsi_cfg()
        knobs._repoint_rsi(cfg, env, None)
        term = cfg.events.rsi_reference_reset
        assert term.func is _STUB_RSI_FUNC
        assert set(term.params) == STOCK_PARAMS, env              # the param dict is the old one
    assert "[KNOB]" not in capsys.readouterr().out


def test_repoint_rsi_carries_the_armed_flags_with_a_consumed_line(knobs, stub_omni_mdp, capsys):
    cfg = _rsi_cfg()
    knobs._repoint_rsi(cfg, {"KRABBY_RSI_CAM_STAGE": "1"}, None)
    params = cfg.events.rsi_reference_reset.params
    assert params["stage_cam"] is True and "stage_cam_debug" not in params
    assert set(params) == STOCK_PARAMS | {"stage_cam"}
    assert "[KNOB] KRABBY_RSI_CAM_STAGE=1 consumed" in capsys.readouterr().out
    cfg = _rsi_cfg()
    knobs._repoint_rsi(cfg, {"KRABBY_RSI_CAM_STAGE": " 1 ", "KRABBY_RSI_CAM_STAGE_DEBUG": "1"}, None)
    params = cfg.events.rsi_reference_reset.params
    assert params["stage_cam"] is True and params["stage_cam_debug"] is True
    out = capsys.readouterr().out
    assert "[KNOB] KRABBY_RSI_CAM_STAGE=1 consumed" in out
    assert "[KNOB] KRABBY_RSI_CAM_STAGE_DEBUG=1 consumed" in out


def test_repoint_rsi_with_rsi_off_reports_instead_of_raising(knobs, stub_omni_mdp, capsys):
    """The rung-0b FRAC-0 staging-on rows build exactly this: no event, so nothing to stage on."""
    cfg = _rsi_cfg(with_term=False)
    knobs._repoint_rsi(cfg, {"KRABBY_RSI_CAM_STAGE": "1"}, None)
    assert not hasattr(cfg.events, "rsi_reference_reset")
    out = capsys.readouterr().out
    assert "KRABBY_RSI_CAM_STAGE=1 NOT consumed: RSI is off" in out and " consumed\n" not in out


def test_repoint_rsi_refuses_a_bad_spelling_even_with_rsi_off(knobs, stub_omni_mdp):
    for cfg in (_rsi_cfg(), _rsi_cfg(with_term=False)):
        with pytest.raises(ValueError, match="KRABBY_RSI_CAM_STAGE"):
            knobs._repoint_rsi(cfg, {"KRABBY_RSI_CAM_STAGE": "yes"}, None)


def test_omni_knobs_reexports_the_vars(knobs):
    names = {getattr(knobs, a) for a in dir(knobs) if a.endswith("_VAR")}
    assert {"KRABBY_RSI_CAM_STAGE", "KRABBY_RSI_CAM_STAGE_DEBUG"} <= names
    assert "RSI_CAM_STAGE_VAR" in knobs.__all__ and "RSI_CAM_STAGE_DEBUG_VAR" in knobs.__all__


def test_the_event_signature_takes_the_params_with_off_defaults():
    import inspect
    sig = inspect.signature(rsi.reset_from_reference_states_omni)
    assert sig.parameters["stage_cam"].default is False
    assert sig.parameters["stage_cam_debug"].default is False


# ========================================================================= the ladder audit
@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES_PY, "run_candidates__cam_stage_test")


ENV_YAML_RSI = textwrap.dedent("""\
    events:
      rsi_reference_reset:
        func: x:reset_from_reference_states_omni
        mode: reset
        params:
          bank_path: /b/bank.npz
          fraction: 0.2
          source: all
          {extra}
      reset_root_state:
        params:
          stage_cam: true
    scene:
      num_envs: 256
    """)


def _dump(extra=""):
    return ENV_YAML_RSI.replace("{extra}", extra)


def test_ladder_registration(rc):
    whitelist, parsers = rc.knob_whitelist(), rc.knob_parsers()
    for var in ("KRABBY_RSI_CAM_STAGE", "KRABBY_RSI_CAM_STAGE_DEBUG"):
        assert var in whitelist and var in parsers and var in rc.OVERLAY_DUMP_CHECKS, var
        assert var not in rc.AGENT_DUMP_KEYS
        reasons = []
        rc._check_knob(var, "1", "t", whitelist, parsers, reasons, set())
        assert not reasons, reasons
        rc._check_knob(var, "yes", "t", whitelist, parsers, reasons, set())
        assert reasons


def test_dump_check_on_and_off(rc):
    stage = rc.OVERLAY_DUMP_CHECKS["KRABBY_RSI_CAM_STAGE"]
    debug = rc.OVERLAY_DUMP_CHECKS["KRABBY_RSI_CAM_STAGE_DEBUG"]
    armed = _dump("stage_cam: true")
    plain = _dump("walk_floor: 0.21")
    no_event = "events:\n  reset_root_state:\n    params:\n      stage_cam: true\nscene:\n  num_envs: 256\n"
    assert stage(armed, "1") and not stage(plain, "1") and not stage(no_event, "1")
    assert stage(plain, "0") and stage(plain, "") and not stage(armed, "0")
    assert stage(no_event, "0")                      # off with RSI off: nothing to prove absent
    # a sibling term's same-named key is never read (scoped to the RSI event block)
    assert not stage(_dump("stage_cam_debug: true"), "1")
    assert debug(_dump("stage_cam_debug: true"), "1") and not debug(armed, "1") and debug(armed, "0")


def test_overlay_audit_end_to_end(rc, tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(_dump("stage_cam: true"))
    ckpt = tmp_path / "model_19.pt"
    ckpt.write_text("")
    checks = rc.overlay_checks({"ckpt": str(ckpt)}, {"KRABBY_RSI_CAM_STAGE": "1", "KRABBY_RSI_FRAC": "0.2"})
    assert checks and all(ok for _n, ok in checks), checks
    assert "overlay KRABBY_RSI_CAM_STAGE=1 in params/env.yaml" in dict(checks)
    bad = rc.overlay_checks({"ckpt": str(ckpt)}, {"KRABBY_RSI_CAM_STAGE": "0"})
    assert not all(ok for _n, ok in bad)


def test_composition_refuses_staging_on_an_rsi_off_stage(rc):
    r = []
    rc._check_rsi_cam_stage_composition({"KRABBY_RSI_CAM_STAGE": "1"}, "w", r)
    assert r and "KRABBY_RSI_FRAC" in r[0]
    r = []
    rc._check_rsi_cam_stage_composition({"KRABBY_RSI_CAM_STAGE_DEBUG": "1", "KRABBY_RSI_FRAC": "0"}, "w", r)
    assert r
    for env in ({"KRABBY_RSI_CAM_STAGE": "1", "KRABBY_RSI_FRAC": "0.2"}, {"KRABBY_RSI_CAM_STAGE": "0"},
                {}, {"KRABBY_RSI_CAM_STAGE": "bad"}):
        r = []
        rc._check_rsi_cam_stage_composition(env, "w", r)
        assert not r, (env, r)


# ======================================================== the spawn-kind tracking split
N_SECT = 4
RSI_KIND = 2


def _push(ledger, kind, is_obst, ratio, valid, move=None, move_valid=None):
    n = len(kind)
    per_ep = ox.episode_exposure_omni(torch.zeros(n), torch.zeros(n, N_SECT, dtype=torch.bool),
                                      torch.zeros(n), torch.zeros(n), torch.full((n,), 10.0), 2.0)
    ledger.push_episodes(
        per_ep, torch.tensor(kind), torch.tensor(is_obst), torch.zeros(n), torch.full((n,), 10.0),
        track_ratio=torch.tensor(ratio), track_ratio_valid=torch.tensor(valid),
        heading_err_rms=torch.zeros(n), block_steps=torch.zeros(n), r_max=torch.zeros(n),
        track_ratio_move=None if move is None else torch.tensor(move),
        track_ratio_move_valid=None if move_valid is None else torch.tensor(move_valid),
    )


def test_split_groups_every_rsi_episode_on_all_terrain():
    """RSI on a FLAT tile counts (the forward ``*_rsi`` convention would drop it); the validity
    rule is the parent key's; spread spawns are non-RSI."""
    assert ox.SPAWN_RSI == RSI_KIND
    ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    _push(ledger, kind=[2, 2, 2, 0, 1, 0], is_obst=[False, True, False, True, False, False],
          ratio=[0.9, 0.5, 100.0, 0.2, 0.4, 0.6], valid=[True, True, False, True, True, True],
          move=[0.8, 0.6, 0.1, 0.3, 50.0, 0.5], move_valid=[True, True, True, True, False, True])
    m = ledger.means()
    assert m["track_ratio_mean_rsi"] == pytest.approx(0.7)          # 0.9 (flat) + 0.5 (obst); 100 invalid
    assert m["track_ratio_mean_norsi"] == pytest.approx(0.4)        # 0.2, 0.4 (spread), 0.6
    assert m["track_ratio_move_mean_rsi"] == pytest.approx(0.5)     # 0.8, 0.6, 0.1
    assert m["track_ratio_move_mean_norsi"] == pytest.approx(0.4)   # 0.3, 0.5; 50 invalid
    assert m["track_ratio_mean"] == pytest.approx((0.9 + 0.5 + 0.2 + 0.4 + 0.6) / 5)   # parent unchanged


def test_an_empty_group_is_absent_not_zero():
    ledger = ox.OmniExposureLedger(N_SECT, capacity=64)
    assert not set(ox.OMNI_SPAWN_SPLIT_KEYS) & set(ledger.means())
    _push(ledger, kind=[0, 0], is_obst=[False, True], ratio=[0.3, 0.5], valid=[True, True])
    m = ledger.means()
    assert m["track_ratio_mean_norsi"] == pytest.approx(0.4)
    for key in ("track_ratio_mean_rsi", "track_ratio_move_mean_rsi", "track_ratio_move_mean_norsi"):
        assert key not in m, key                  # no RSI episode yet / no move ratio pushed
    # an RSI episode that is not VALID does not open the group either
    _push(ledger, kind=[2], is_obst=[False], ratio=[0.9], valid=[False])
    assert "track_ratio_mean_rsi" not in ledger.means()
    _push(ledger, kind=[2], is_obst=[False], ratio=[0.9], valid=[True])
    assert ledger.means()["track_ratio_mean_rsi"] == pytest.approx(0.9)


def test_split_keys_stay_out_of_the_emitted_key_list():
    """``emit_keys`` / ``OMNI_EXTRA_KEYS`` are emitted from iteration 0 (0.0 while empty) and pinned
    by the driver; the split keys must not join them."""
    assert not set(ox.OMNI_SPAWN_SPLIT_KEYS) & set(ox.OmniExposureLedger.emit_keys(N_SECT))
    assert ox.OMNI_EXTRA_KEYS == ("track_ratio_mean", "track_ratio_move_mean", "heading_err_rms_mean",
                                  "block_steps_mean", "r_max_mean")


def _finalize_fn():
    src = PARKOUR_EVENT_PY.read_text()
    code = "from __future__ import annotations\n" + _method_src(src, "OmniParkourEvent", "_finalize_episodes")
    ns = {"torch": torch, "episode_exposure_omni": ox.episode_exposure_omni,
          "tracking_ratio": ox.tracking_ratio, "OMNI_SPAWN_SPLIT_KEYS": ox.OMNI_SPAWN_SPLIT_KEYS}
    exec(compile(code, str(PARKOUR_EVENT_PY), "exec"), ns)
    return ns["_finalize_episodes"]


def _parkour_term(n):
    t = types.SimpleNamespace(num_envs=n, device="cpu", n_sectors=N_SECT, clear_radius=2.0, min_cmd_dist=0.5)
    t.exposure = ox.OmniExposureLedger(N_SECT, capacity=64)
    t.metrics = {k: torch.zeros(n) for k in ox.OmniExposureLedger.emit_keys(N_SECT)}
    t.ep_steps = torch.full((n,), 50.0)
    for name in ("ep_r_max", "ep_block_steps", "ep_field_steps", "ep_head_err_sq"):
        setattr(t, name, torch.zeros(n))
    t.ep_sector_seen = torch.zeros(n, N_SECT, dtype=torch.bool)
    t.ep_track_dist = torch.tensor([0.8, 0.4, 0.6, 0.2])
    t.ep_cmd_dist = torch.ones(n)
    t.ep_track_dist_move = torch.tensor([0.8, 0.4, 0.6, 0.2])
    t.ep_cmd_dist_move = torch.ones(n)
    t.is_obst_tile = torch.zeros(n, dtype=torch.bool)
    t.spawn_kind = torch.zeros(n, dtype=torch.long)
    t._failure_flags = lambda ids: torch.zeros(len(ids))
    return t


def test_the_parkour_term_creates_the_metric_only_when_its_group_fills():
    finalize = _finalize_fn()
    t = _parkour_term(4)
    finalize(t, torch.tensor([0, 1]))                      # two platform-spawned episodes
    assert "track_ratio_mean_norsi" in t.metrics and "track_ratio_mean_rsi" not in t.metrics
    assert torch.allclose(t.metrics["track_ratio_mean_norsi"], torch.full((4,), 0.6))   # seeded with the value
    t.spawn_kind[2] = RSI_KIND
    finalize(t, torch.tensor([2]))
    assert torch.allclose(t.metrics["track_ratio_mean_rsi"], torch.full((4,), 0.6))
    assert torch.allclose(t.metrics["track_ratio_move_mean_rsi"], torch.full((4,), 0.6))
    # every metric the base _update_metrics will broadcast exists (it indexes metrics[key])
    assert set(t._exposure_means) <= set(t.metrics)
