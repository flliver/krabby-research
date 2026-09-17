"""The omni observation term's five re-filled slots (omni plan D2/A5), on CPU, without Isaac.

``CrabHexOmniParkourObservations`` changes five columns of a 1149-wide vector and nothing
else, so every failure mode here is silent: a command the policy never sees, a facing error
that is blanked out of the history one step after it was observed, a yaw-rate command that
stays at the forward task's structural zero. The test therefore runs the REAL subclass on
top of the REAL forward ``__call__``:

* the forward ``CrabHexParkourObservations.__call__`` is lifted out of ``observations.py``
  with ``ast`` and exec'd as the ``__call__`` of a stub base class that supplies only the
  tensors it reads (the pattern of ``test_crab_hex_observation_hooks.py``);
* the omni subclass is exec'd from its own source AS A CLASS (not method by method), so its
  ``super().__call__`` resolves normally and the shipped slot-5 write really runs.

Nothing is registered under ``isaaclab.*`` / ``parkour_isaaclab.*``: this file must not
depend on which other unit modules have already been collected.
"""

import ast
import importlib.util
import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
FWD_OBS_PY = TASKS / "crab_hex_forward_task/mdp/observations.py"
OMNI_OBS_PY = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_observations.py"
TRAVEL_FRAME_PY = REPO / "parkour/parkour_isaaclab/envs/mdp/travel_frame.py"

N = 6
NJ, NA, NC = 24, 18, 6
NSCAN = 108
HISTORY = 10
N_PROP = 17 + 2 * NJ + NA + NC                    # 89
OBS_DIM = N_PROP + NSCAN + 9 + 5 + 2 * NJ + HISTORY * N_PROP   # 1149


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


travel_frame = _load(TRAVEL_FRAME_PY, "travel_frame__omni_obs_test")
wrap_to_pi = travel_frame.wrap_to_pi
yaw_error = travel_frame.yaw_error


def _euler_xyz_from_quat(q):
    """Stand-in for ``isaaclab.utils.math.euler_xyz_from_quat`` (roll, pitch, yaw)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _source_of(path: Path, *, cls: str | None, name: str) -> str:
    """Dedented source of a module-level name, or of one method of ``cls``."""
    src = path.read_text()
    tree = ast.parse(src, str(path))
    body = tree.body
    if cls is not None:
        body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    node = [
        n
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
    ][0]
    segment = ast.get_source_segment(src, node)
    return textwrap.dedent(" " * node.col_offset + segment)


# --------------------------------------------------------------------- the stub base class
class _Data:
    pass


class _Asset:
    def __init__(self):
        self.data = _Data()


class _CmdTerm:
    def __init__(self, heading_target):
        self.heading_target = heading_target


class _CommandManager:
    def __init__(self, cmd, cmd_term):
        self._cmd = cmd
        self._term = cmd_term

    def get_command(self, name):
        assert name == "base_velocity"
        return self._cmd

    def get_term(self, name):
        assert name == "base_velocity"
        return self._term


class _JointAction:
    def __init__(self, clock_phase, action_history_buf):
        self.clock_phase = clock_phase
        self.action_history_buf = action_history_buf


class _ActionManager:
    def __init__(self, term):
        self._term = term

    def get_term(self, name):
        assert name == "joint_pos"
        return self._term


class _Cfg:
    pass


class _ParkourEvent:
    def __init__(self, n):
        self.env_per_terrain_name = np.array([["parkour_flat"]] * n)
        self.target_yaw = torch.zeros(n)
        self.next_target_yaw = torch.zeros(n)


class _Env:
    def __init__(self, n, cmd, heading_target):
        self.cfg = _Cfg()
        self.num_envs = n
        self.common_step_counter = 0
        self.episode_length_buf = torch.full((n,), 50, dtype=torch.long)
        self.command_manager = _CommandManager(cmd, _CmdTerm(heading_target))
        self.action_manager = _ActionManager(
            _JointAction(torch.zeros(n), torch.zeros(n, 2, NA))
        )


_FWD_CALL_NS = {
    "torch": torch,
    "os": os,
    "wrap_to_pi": wrap_to_pi,
    "yaw_error": yaw_error,
    "euler_xyz_from_quat": _euler_xyz_from_quat,
    "warn_if_nonfinite": lambda *a, **k: None,
}
exec(
    compile(
        "from __future__ import annotations\n"
        + _source_of(FWD_OBS_PY, cls="CrabHexParkourObservations", name="__call__"),
        str(FWD_OBS_PY),
        "exec",
    ),
    _FWD_CALL_NS,
)
_FWD_HOOK_NS = dict(_FWD_CALL_NS)
for _hook in ("_steering_channels", "_command_channels"):
    exec(
        compile(
            "from __future__ import annotations\n"
            + _source_of(FWD_OBS_PY, cls="CrabHexParkourObservations", name=_hook),
            str(FWD_OBS_PY),
            "exec",
        ),
        _FWD_HOOK_NS,
    )


class _StubForwardObservations:
    """Only what the shipped forward ``__call__`` touches -- plus that ``__call__`` itself."""

    ZERO_STEERING_HISTORY = True
    __call__ = _FWD_CALL_NS["__call__"]
    _steering_channels = _FWD_HOOK_NS["_steering_channels"]
    _command_channels = _FWD_HOOK_NS["_command_channels"]

    def __init__(self, cfg, env):
        g = torch.Generator().manual_seed(20260916)
        n = env.num_envs
        self.num_envs = n
        self.device = "cpu"
        self.history_length = HISTORY
        self._env = env
        self.parkour_event = _ParkourEvent(n)
        self.asset = _Asset()
        self.asset.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n)
        self.asset.data.heading_w = (torch.rand(n, generator=g) - 0.5) * 6.0
        self.asset.data.root_ang_vel_b = torch.randn(n, 3, generator=g)
        self.asset.data.root_lin_vel_b = torch.randn(n, 3, generator=g)
        self.asset.data.joint_pos = torch.randn(n, NJ, generator=g)
        self.asset.data.default_joint_pos = torch.zeros(n, NJ)
        self.asset.data.joint_vel = torch.randn(n, NJ, generator=g)
        self._cam_shaft_obs_ids = list(range(6))
        self.delta_yaw = torch.zeros(n)
        self.delta_next_yaw = torch.zeros(n)
        self.measured_heights = torch.zeros(n, NSCAN)
        self._obs_history_buffer = torch.zeros(n, HISTORY, N_PROP)

    def _get_heights(self):
        return torch.zeros(self.num_envs, NSCAN)

    def _get_contact_fill(self):
        return torch.zeros(self.num_envs, NC) - 0.5

    def _get_priv_explicit(self):
        return torch.zeros(self.num_envs, 9)

    def _get_priv_latent(self):
        return torch.zeros(self.num_envs, 5 + 2 * NJ)


# ------------------------------------------------------------------ the shipped subclass
_OMNI_NS = {
    "torch": torch,
    "os": os,
    "wrap_to_pi": wrap_to_pi,
    "SceneEntityCfg": object,
    "ParkourManagerBasedRLEnv": object,
    "CrabHexParkourObservations": _StubForwardObservations,
}
exec(
    compile(
        "from __future__ import annotations\n"
        + _source_of(OMNI_OBS_PY, cls=None, name="_check_omni")
        + "\n\n"
        + _source_of(OMNI_OBS_PY, cls=None, name="CrabHexOmniParkourObservations"),
        str(OMNI_OBS_PY),
        "exec",
    ),
    _OMNI_NS,
)
OmniObs = _OMNI_NS["CrabHexOmniParkourObservations"]
check_omni = _OMNI_NS["_check_omni"]


def _make(term_cls=OmniObs, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    cmd = torch.zeros(N, 4)
    cmd[:, 0:2] = (torch.rand(N, 2, generator=g) - 0.5) * 0.6
    cmd[:, 2] = (torch.rand(N, generator=g) - 0.5) * 0.6
    heading_target = (torch.rand(N, generator=g) - 0.5) * 6.0
    env = _Env(N, cmd, heading_target)
    term = term_cls(None, env)
    return term, env, cmd


def _facing_error(term, env):
    return wrap_to_pi(
        env.command_manager.get_term("base_velocity").heading_target - term.asset.data.heading_w
    )


# --------------------------------------------------------------------------- class attrs
def test_class_attributes():
    assert OmniObs.ZERO_STEERING_HISTORY is False
    assert _StubForwardObservations.ZERO_STEERING_HISTORY is True      # the forward default
    assert OmniObs.OBSERVED_COMMAND_SLOTS == {8: 0, 9: 1, 5: 2}
    assert OmniObs.YAW_RATE_OBS_SLOT == 5
    assert OmniObs.CMD_NORM_OBS_SLOT == 10


@pytest.mark.parametrize("value", ["0:0", "-1.2:1.2", ""])
def test_check_omni_refuses_a_set_krabby_heading(monkeypatch, value):
    monkeypatch.setenv("KRABBY_HEADING", value)
    with pytest.raises(RuntimeError, match="KRABBY_HEADING"):
        check_omni()
    with pytest.raises(RuntimeError, match="KRABBY_HEADING"):
        _make()                                    # the term's __init__ calls it
    monkeypatch.delenv("KRABBY_HEADING")
    check_omni()
    check_omni(object())                           # the cfg overload the env cfg uses
    _make()


# --------------------------------------------------------------------------- slot contents
def test_width_is_unchanged(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, _ = _make()
    obs = term(env, None, None, "base_parkour", HISTORY)
    assert obs.shape == (N, OBS_DIM) == (N, 1149)
    assert term._obs_history_buffer.shape == (N, HISTORY, N_PROP) == (N, HISTORY, 89)


def test_slot_contents(monkeypatch):
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _make()
    obs = term(env, None, None, "base_parkour", HISTORY)
    err = _facing_error(term, env)

    assert torch.equal(obs[:, 5], cmd[:, 2])                       # yaw-rate command
    assert torch.equal(obs[:, 6], err)                             # facing error ...
    assert torch.equal(obs[:, 7], err)                             # ... in both channels
    assert torch.equal(obs[:, 8:10], cmd[:, 0:2])                  # body-frame xy command
    assert torch.equal(obs[:, 10], torch.linalg.norm(cmd[:, 0:2], dim=-1))
    # and the slots around them are untouched by the subclass
    assert torch.equal(obs[:, 0:3], term.asset.data.root_ang_vel_b * 0.25)
    assert torch.equal(obs[:, 13:15], term.asset.data.root_lin_vel_b[:, :2] * 2.0)


def test_observed_command_slots_matches_the_layout(monkeypatch):
    """The mapping the eval harness trusts to verify an injected command reached the policy."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _make()
    obs = term(env, None, None, "base_parkour", HISTORY)
    for slot, cmd_idx in OmniObs.OBSERVED_COMMAND_SLOTS.items():
        assert torch.equal(obs[:, slot], cmd[:, cmd_idx]), (slot, cmd_idx)
    assert torch.equal(
        obs[:, OmniObs.CMD_NORM_OBS_SLOT], torch.linalg.norm(cmd[:, 0:2], dim=-1)
    )
    # a norm is not a channel: it must not claim a command index
    assert OmniObs.CMD_NORM_OBS_SLOT not in OmniObs.OBSERVED_COMMAND_SLOTS


def test_the_steering_channels_ignore_the_flat_mask(monkeypatch):
    """Forward: zero on flat tiles. Omni: the facing error everywhere, flat or not."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, _ = _make()
    err = _facing_error(term, env)
    for on_flat in (torch.ones(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool)):
        a, b = term._steering_channels(env, on_flat)
        assert torch.equal(a, err) and torch.equal(b, err)
    # the forward hook on the same inputs zeroes the flat rows
    fwd_a, _ = _StubForwardObservations._steering_channels(
        term, env, torch.ones(N, dtype=torch.bool)
    )
    assert torch.equal(fwd_a, torch.zeros(N))


def test_command_channels_are_the_two_axes_and_their_norm():
    cmd = torch.tensor([[0.3, 0.0, 0.1, 0.0], [0.0, -0.25, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0],
                        [-0.2, 0.2, 0.0, 0.0]])
    got = OmniObs._command_channels(None, cmd)
    assert got.shape == (4, 3)
    assert torch.equal(got[:, 0:2], cmd[:, 0:2])
    assert got[0, 2] == pytest.approx(0.3)
    assert got[1, 2] == pytest.approx(0.25)
    assert got[2, 2] == 0.0                                   # a turn-in-place has no speed
    assert got[3, 2] == pytest.approx(math.hypot(0.2, 0.2))


# --------------------------------------------------------------------------- the history
def test_history_keeps_the_facing_error_and_the_yaw_rate(monkeypatch):
    """Two steps mid-episode: the row just pushed carries slots 5, 6, 7 -- unlike forward."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _make()
    err = _facing_error(term, env)
    for step in range(2):
        env.common_step_counter = step
        term(env, None, None, "base_parkour", HISTORY)
    last = term._obs_history_buffer[:, -1]
    assert torch.equal(last[:, 5], cmd[:, 2])
    assert torch.equal(last[:, 6], err)
    assert torch.equal(last[:, 7], err)
    assert torch.equal(last[:, 8:10], cmd[:, 0:2])

    # the forward term on the same inputs blanks 6-7 in the history (and has no yaw rate)
    fwd, fenv, _ = _make(term_cls=_StubForwardObservations)
    fwd(fenv, None, None, "base_parkour", HISTORY)
    fwd_last = fwd._obs_history_buffer[:, -1]
    assert torch.equal(fwd_last[:, 6:8], torch.zeros(N, 2))
    assert torch.equal(fwd_last[:, 5], torch.zeros(N))


def test_a_fresh_episode_replicates_the_yaw_rate_across_the_whole_history(monkeypatch):
    """``episode_length_buf <= 1`` makes the base stack H copies of the row: patch them all."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _make()
    env.episode_length_buf = torch.tensor([1, 1, 50, 50, 0, 50])
    term(env, None, None, "base_parkour", HISTORY)
    fresh = env.episode_length_buf <= 1
    for row in range(HISTORY):
        got = term._obs_history_buffer[fresh, row, 5]
        assert torch.equal(got, cmd[fresh, 2]), row
    # a non-fresh env keeps its older rows at zero and only the newest row is written
    stale = ~fresh
    assert torch.equal(term._obs_history_buffer[stale, -1, 5], cmd[stale, 2])
    assert torch.equal(
        term._obs_history_buffer[stale, :-1, 5], torch.zeros(int(stale.sum()), HISTORY - 1)
    )


def test_the_returned_history_block_is_the_patched_buffer(monkeypatch):
    """Step 2's observation carries step 1's history row, yaw-rate slot included."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    term, env, cmd = _make()
    term(env, None, None, "base_parkour", HISTORY)
    obs = term(env, None, None, "base_parkour", HISTORY)
    hist_base = N_PROP + NSCAN + 9 + 5 + 2 * NJ
    newest = hist_base + (HISTORY - 1) * N_PROP
    assert torch.equal(obs[:, newest + 5], cmd[:, 2])
    assert torch.equal(obs[:, newest + 6], _facing_error(term, env))


def test_nothing_outside_the_five_slots_moves(monkeypatch):
    """The subclass is a five-column change: everything else is the forward term's output."""
    monkeypatch.delenv("KRABBY_HEADING", raising=False)
    omni_term, omni_env, _ = _make()
    fwd_term, fwd_env, _ = _make(term_cls=_StubForwardObservations)
    omni_obs = omni_term(omni_env, None, None, "base_parkour", HISTORY)
    fwd_obs = fwd_term(fwd_env, None, None, "base_parkour", HISTORY)
    changed = [c for c in range(N_PROP) if not torch.equal(omni_obs[:, c], fwd_obs[:, c])]
    assert set(changed) <= {5, 6, 7, 8, 9, 10}
    assert torch.equal(omni_obs[:, N_PROP:], fwd_obs[:, N_PROP:])   # scan + priv + history


# ------------------------------------------------------------ facing-error clip (2026-09-17)
class _ClipCfg:
    def __init__(self, clip):
        self.params = {"facing_obs_clip": clip}


def test_facing_error_is_raw_by_default(monkeypatch):
    term, env, _ = _make()
    assert term.facing_obs_clip is None
    a, b = term._steering_channels(env, torch.zeros(N, dtype=torch.bool))
    assert torch.equal(a, _facing_error(term, env)) and torch.equal(a, b)


def test_facing_error_clip_saturates_both_channels_and_keeps_the_sign(monkeypatch):
    """Out-of-band targets read as the band's edge; in-band values are untouched."""
    g = torch.Generator().manual_seed(3)
    cmd = torch.zeros(N, 4)
    heading_target = (torch.rand(N, generator=g) - 0.5) * 6.0   # up to +-3 rad of error
    env = _Env(N, cmd, heading_target)
    term = OmniObs(_ClipCfg(0.6), env)
    assert term.facing_obs_clip == 0.6
    raw = _facing_error(term, env)
    a, b = term._steering_channels(env, torch.zeros(N, dtype=torch.bool))
    assert torch.equal(a, b)
    assert torch.equal(a, torch.clamp(raw, -0.6, 0.6))
    assert torch.all(a.abs() <= 0.6)
    inband = raw.abs() <= 0.6
    assert inband.any() and torch.equal(a[inband], raw[inband])
    assert torch.equal(torch.sign(a), torch.sign(raw))


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_facing_error_clip_rejects_non_positive(bad):
    env = _Env(N, torch.zeros(N, 4), torch.zeros(N))
    with pytest.raises(ValueError, match="facing_obs_clip"):
        OmniObs(_ClipCfg(bad), env)
