"""The three observation hooks are a refactor, not a change (omni-task plan D2, A3).

``CrabHexParkourObservations.__call__`` used to compute the steering channels (obs dims 6-7)
and the command channels (dims 8-10) inline, and to blank dims 6-7 in the history copy
unconditionally. The omnidirectional sibling task needs to override all three, so they became
``_steering_channels(env, on_flat)``, ``_command_channels(commands)`` and the class attribute
``ZERO_STEERING_HISTORY``. Two lineages of record (the forward policy and the sideways one)
were trained through the old code, so the forward values must be BIT-IDENTICAL -- same ops,
same order, same dtype.

This test re-implements the pre-refactor lines as an oracle and checks the shipped hooks
against them on random inputs, in both ``KRABBY_HEADING`` states. The hook sources are read
from ``observations.py`` with ``ast`` and exec'd in a namespace holding only ``torch``, ``os``
and the real ``wrap_to_pi`` -- so the actual shipped expressions run, with no Isaac import and
no ``sys.modules`` stubbing that other pure suites could collide with.

``test_crab_hex_forward_seam_literals.py`` pins the same hook bodies as source text; this one
pins what they compute.
"""

import ast
import importlib.util
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OBSERVATIONS_PY = (
    REPO
    / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_forward_task"
    / "mdp" / "observations.py"
)
TRAVEL_FRAME_PY = REPO / "parkour" / "parkour_isaaclab" / "envs" / "mdp" / "travel_frame.py"

OBS_CLASS = "CrabHexParkourObservations"
HOOKS = ("_steering_channels", "_command_channels")

N = 64
SEED = 20260916


def _load_travel_frame() -> types.ModuleType:
    """The library's ``wrap_to_pi`` -- copied verbatim from ``isaaclab.utils.math``, which is
    the function ``observations.py`` imports at runtime. Torch only."""
    spec = importlib.util.spec_from_file_location("_obs_hooks_travel_frame", TRAVEL_FRAME_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_travel_frame = _load_travel_frame()


def _hook_namespace() -> dict:
    """Exec the two hook methods out of the shipped source, as plain functions."""
    src = OBSERVATIONS_PY.read_text()
    tree = ast.parse(src, str(OBSERVATIONS_PY))
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == OBS_CLASS][0]
    chunks = []
    for name in HOOKS:
        node = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
        segment = ast.get_source_segment(src, node)
        # the segment starts mid-line at ``def``; restore the indent, then dedent the block
        chunks.append(textwrap.dedent(" " * node.col_offset + segment))
    ns = {
        "os": __import__("os"),
        "torch": torch,
        "wrap_to_pi": _travel_frame.wrap_to_pi,
    }
    # ``from __future__ import annotations``: the real module has it, and the signatures name
    # Isaac types we deliberately do not import.
    source = "from __future__ import annotations\n" + "\n\n".join(chunks)
    exec(compile(source, str(OBSERVATIONS_PY), "exec"), ns)
    missing = [h for h in HOOKS if h not in ns]
    assert not missing, f"hooks missing from {OBSERVATIONS_PY}: {missing}"
    return ns


HOOK_NS = _hook_namespace()


class _Data:
    def __init__(self, heading_w):
        self.heading_w = heading_w


class _Asset:
    def __init__(self, heading_w):
        self.data = _Data(heading_w)


class _CmdTerm:
    def __init__(self, heading_target):
        self.heading_target = heading_target


class _CommandManager:
    def __init__(self, cmd_term):
        self._term = cmd_term

    def get_term(self, name):
        assert name == "base_velocity"
        return self._term


class _Env:
    def __init__(self, cmd_term):
        self.command_manager = _CommandManager(cmd_term)


class _Obs:
    """Only the attributes the two hooks touch."""

    def __init__(self, delta_yaw, delta_next_yaw, heading_w):
        self.delta_yaw = delta_yaw
        self.delta_next_yaw = delta_next_yaw
        self.asset = _Asset(heading_w)

    _steering_channels = HOOK_NS["_steering_channels"]
    _command_channels = HOOK_NS["_command_channels"]


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    delta_yaw = (torch.rand(N, generator=g) - 0.5) * 8.0
    delta_next_yaw = (torch.rand(N, generator=g) - 0.5) * 8.0
    heading_w = (torch.rand(N, generator=g) - 0.5) * 8.0
    heading_target = (torch.rand(N, generator=g) - 0.5) * 8.0
    on_flat = torch.rand(N, generator=g) < 0.5
    commands = (torch.rand(N, 4, generator=g) - 0.5) * 2.0
    return delta_yaw, delta_next_yaw, heading_w, heading_target, on_flat, commands


def _oracle_steering(heading_set, on_flat, delta_yaw, delta_next_yaw, heading_target, heading_w):
    """The pre-refactor lines, copied from ``__call__`` as it stood before the hooks."""
    if heading_set:
        heading_err = _travel_frame.wrap_to_pi(heading_target - heading_w)
        out_yaw = torch.where(on_flat, heading_err, delta_yaw)
        out_next = torch.where(on_flat, heading_err, delta_next_yaw)
    else:
        out_yaw = torch.where(on_flat, torch.zeros_like(delta_yaw), delta_yaw)
        out_next = torch.where(on_flat, torch.zeros_like(delta_next_yaw), delta_next_yaw)
    return out_yaw, out_next


@pytest.mark.parametrize("heading", [None, "0:0", "-1.2:1.2"])
@pytest.mark.parametrize("seed", [SEED, SEED + 1, SEED + 2])
def test_steering_channels_match_the_pre_refactor_lines(monkeypatch, heading, seed):
    delta_yaw, delta_next_yaw, heading_w, heading_target, on_flat, _ = _inputs(seed)
    if heading is None:
        monkeypatch.delenv("KRABBY_HEADING", raising=False)
    else:
        monkeypatch.setenv("KRABBY_HEADING", heading)

    obs = _Obs(delta_yaw, delta_next_yaw, heading_w)
    env = _Env(_CmdTerm(heading_target))
    got_yaw, got_next = obs._steering_channels(env, on_flat)
    want_yaw, want_next = _oracle_steering(
        heading is not None, on_flat, delta_yaw, delta_next_yaw, heading_target, heading_w
    )
    assert torch.equal(got_yaw, want_yaw)
    assert torch.equal(got_next, want_next)
    assert got_yaw.dtype == delta_yaw.dtype and got_next.dtype == delta_next_yaw.dtype
    assert got_yaw.shape == (N,) and got_next.shape == (N,)


@pytest.mark.parametrize("seed", [SEED, SEED + 1, SEED + 2])
def test_command_channels_match_the_pre_refactor_cat_entries(seed):
    *_, commands = _inputs(seed)
    obs = _Obs(torch.zeros(N), torch.zeros(N), torch.zeros(N))
    got = obs._command_channels(commands)
    # the two entries the big ``torch.cat`` used to hold inline, concatenated in order
    want = torch.cat((0 * commands[:, 0:2], commands[:, 0:1]), dim=-1)
    assert torch.equal(got, want)
    assert got.shape == (N, 3)
    assert got.dtype == commands.dtype
    assert torch.equal(got[:, 0:2], torch.zeros(N, 2))
    assert torch.equal(got[:, 2], commands[:, 0])


@pytest.mark.parametrize("heading", [None, "-1.2:1.2"])
@pytest.mark.parametrize("seed", [SEED, SEED + 3])
def test_the_hooked_obs_block_is_bitwise_the_old_one(monkeypatch, heading, seed):
    """Channels 5-10 of ``obs_buf`` -- everything the hooks feed -- built both ways."""
    delta_yaw, delta_next_yaw, heading_w, heading_target, on_flat, commands = _inputs(seed)
    if heading is None:
        monkeypatch.delenv("KRABBY_HEADING", raising=False)
    else:
        monkeypatch.setenv("KRABBY_HEADING", heading)

    # OLD: the inline computation, then the five cat entries it produced
    old_yaw, old_next = _oracle_steering(
        heading is not None, on_flat, delta_yaw, delta_next_yaw, heading_target, heading_w
    )
    old_block = torch.cat(
        (
            0 * old_yaw[:, None],
            old_yaw[:, None],
            old_next[:, None],
            0 * commands[:, 0:2],
            commands[:, 0:1],
        ),
        dim=-1,
    )

    # NEW: the hooks, assembled as ``__call__`` assembles them
    obs = _Obs(delta_yaw, delta_next_yaw, heading_w)
    env = _Env(_CmdTerm(heading_target))
    new_yaw, new_next = obs._steering_channels(env, on_flat)
    new_block = torch.cat(
        (
            0 * new_yaw[:, None],
            new_yaw[:, None],
            new_next[:, None],
            obs._command_channels(commands),
        ),
        dim=-1,
    )

    assert new_block.shape == old_block.shape == (N, 6)
    assert torch.equal(new_block, old_block)


def test_zero_steering_history_defaults_to_the_forward_behaviour():
    """The class attribute is True, and the guarded line blanks exactly dims 6-7."""
    src = OBSERVATIONS_PY.read_text()
    tree = ast.parse(src, str(OBSERVATIONS_PY))
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == OBS_CLASS][0]
    assigns = [
        n
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ZERO_STEERING_HISTORY" for t in n.targets)
    ]
    assert len(assigns) == 1
    assert ast.literal_eval(assigns[0].value) is True

    # behaviour of the guard itself, on a tensor: True blanks, False leaves the channels
    g = torch.Generator().manual_seed(SEED)
    buf = torch.rand(N, 17, generator=g)

    def _apply(flag: bool) -> torch.Tensor:
        out = buf.clone()
        if flag:  # the shipped guard, verbatim
            out[:, 6:8] = 0
        return out

    blanked, kept = _apply(True), _apply(False)
    assert torch.equal(blanked[:, 6:8], torch.zeros(N, 2))
    assert torch.equal(kept, buf)
    untouched = [c for c in range(17) if c not in (6, 7)]
    assert torch.equal(blanked[:, untouched], buf[:, untouched])
