# SPDX-License-Identifier: BSD-3-Clause
"""The body-frame yaw reward set's pure math: a trailing yaw window and its two incomes.

Robot-frame velocity commands campaign (``experiments/2026-10-15_body_cmd``), PLAN D4 as amended
by the CHARTER on 2026-10-15 (the step-1a user decision "2-s window + tent 2.0"). DEFAULT OFF: only
``KRABBY_OMNI_COMMAND_FRAME=body`` (``omni_knobs._apply_body_reward_set``) swaps the two class
terms that call this module (``crab_hex_omni_rewards.RewardYawWindowExp`` /
``RewardYawProgressTent``) into ``reward_tracking_yaw`` / ``reward_yaw_progress``; a world-frame
run never imports it.

WHY A WINDOW. A body-frame command has no facing TARGET to price a heading error against: the
command is a yaw RATE ``wz``. A per-step rate term is wobble-blind (the gait's own yaw wobble is
0.1-0.3 rad/s against a 0.05 rad/s slow-turn command), so both terms compare INTEGRALS over the same
trailing window instead -- the achieved yaw ``sum(root_ang_vel_b_z * dt)`` against the commanded
yaw ``sum(wz_cmd * dt)``, over the last ``N = round(window_s / step_dt)`` steps (100 at 2.0 s and
dt 0.02). On a ``wz = 0`` row that is a 2-s heading hold; on a turning row it is rate tracking with
the wobble averaged out. The step-1a pricing check (CHARTER §3, ``step1a_turn_pricing_v2.md``)
priced exactly these two incomes on S1's measured traces.

THE TWO INCOMES (pure functions of the two window sums -- what the CPU tests pin):

* :func:`yaw_window_income` -- ``exp(-|S_ach - S_cmd| / sigma)``, ``sigma`` = 0.10 rad
  (= w_min 0.05 rad/s x the 2-s window): a perfect tracker earns 1.0; a still stander on a
  ``c = 0.05`` rad/s command earns ``exp(-1)``.
* :func:`yaw_tent_income` -- with ``cbar = S_cmd / T`` and ``p = sign(cbar) * S_ach / T``
  (``T`` = the window length in seconds), ``min(p, 2|cbar| - p)`` clamped below at ``-w_max``, and
  EXACTLY 0.0 where ``cbar == 0``: a tent that peaks at the commanded rate (``p = |cbar|``), pays
  nothing for twice the commanded rate, charges over-spin beyond it and wrong-way rotation, and
  cannot be farmed by spinning (the rotational form of the wheelie exploit the linear progress
  term's ``max_speed_scale`` closed). ``p`` is the 2-s MEAN rate, not a per-step one: a per-step
  rate makes the tent concave under gait wobble (PLAN D4, audit pass 3).

THE WINDOW (:class:`YawWindow`). Two preallocated per-env ring buffers on the env device; one
global write column (a Python int -- every env pushes every step, so there is no per-env pointer
and no host-device sync); the sums are FULL reductions each step, never a running add/subtract,
so a buffer of exact zeros sums to exactly 0.0 and the tent's "exactly 0.0 on straight rows" holds
bitwise. Zero-filled on the env's episode reset (``reset(env_ids)``), so early in an episode the
sums run over the steps since the reset (implementation pin of the CHARTER's "the window integral
is taken over the steps since the reset when fewer than 100 have elapsed"); ``T`` stays the full
window length. NOT reset on a command change: the commanded-history integral already carries the
change, because both sums run over the same trailing steps (a perfect tracker of a command that
flipped inside the window still earns 1.0).

Pure torch, no Isaac import: unit-tested on CPU in ``tests/unit/test_crab_hex_omni_body_cmd.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

#: The window length, seconds (CHARTER §3, 2026-10-15: "2-s window").
YAW_WINDOW_S = 2.0
#: The window term's kernel width, radians (CHARTER §3: "sigma = 0.10 rad", = w_min x the window).
YAW_WINDOW_SIGMA = 0.10


def window_steps(window_s: float, step_dt: float) -> int:
    """``round(window_s / step_dt)`` env steps, at least 1 -- or a named ``ValueError``."""
    window_s, step_dt = float(window_s), float(step_dt)
    if not (math.isfinite(window_s) and window_s > 0.0):
        raise ValueError(f"yaw window: window_s must be finite and positive, got {window_s}")
    if not (math.isfinite(step_dt) and step_dt > 0.0):
        raise ValueError(f"yaw window: step_dt must be finite and positive, got {step_dt}")
    n = int(round(window_s / step_dt))
    if n < 1:
        raise ValueError(f"yaw window: window_s {window_s} s is shorter than one step ({step_dt} s)")
    return n


class YawWindow:
    """Per-env trailing sums of achieved and commanded yaw over the last ``n_steps`` steps.

    ``push(ach_dt, cmd_dt)`` writes this step's two ``(N,)`` increments (radians: a yaw rate times
    ``step_dt``) into the current column and returns ``(S_ach, S_cmd)``, each ``(N,)``, the sums
    over the whole buffer -- i.e. over the last ``n_steps`` pushes, or over every push since the
    row's last :meth:`reset` when fewer have happened (the rest of the row is zeros).
    """

    def __init__(self, num_envs: int, n_steps: int, device=None, dtype: torch.dtype = torch.float32):
        if int(n_steps) < 1:
            raise ValueError(f"YawWindow: n_steps must be >= 1, got {n_steps}")
        self.n_steps = int(n_steps)
        self.ach = torch.zeros(int(num_envs), self.n_steps, device=device, dtype=dtype)
        self.cmd = torch.zeros(int(num_envs), self.n_steps, device=device, dtype=dtype)
        self._col = 0

    def push(self, ach_dt: torch.Tensor, cmd_dt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        col = self._col
        self.ach[:, col] = ach_dt
        self.cmd[:, col] = cmd_dt
        self._col = (col + 1) % self.n_steps
        return self.ach.sum(dim=1), self.cmd.sum(dim=1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Zero-fill the reset rows (every row for ``None``); the write column is shared and kept."""
        idx = slice(None) if env_ids is None else env_ids
        self.ach[idx] = 0.0
        self.cmd[idx] = 0.0


def yaw_window_income(sum_ach: torch.Tensor, sum_cmd: torch.Tensor, sigma: float) -> torch.Tensor:
    """``exp(-|S_ach - S_cmd| / sigma)`` in ``(0, 1]`` -- the windowed yaw-tracking income."""
    sigma = float(sigma)
    if not (math.isfinite(sigma) and sigma > 0.0):
        raise ValueError(f"yaw_window_income: sigma must be finite and positive, got {sigma}")
    return torch.exp(-torch.abs(sum_ach - sum_cmd) / sigma)


def yaw_tent_income(
    sum_ach: torch.Tensor, sum_cmd: torch.Tensor, window_t: float, w_max: float
) -> torch.Tensor:
    """The yaw-progress tent ``min(p, 2|cbar| - p)``, clamped below at ``-w_max``, 0.0 at ``cbar == 0``.

    ``cbar = S_cmd / T`` is the window's mean COMMANDED rate and ``p = sign(cbar) * S_ach / T`` the
    window's mean ACHIEVED rate toward it (rad/s), ``T = window_t`` seconds. Exactly ``+0.0`` (never
    ``-0.0``) on every row whose commanded window sums to exactly zero.
    """
    window_t, w_max = float(window_t), float(w_max)
    if not (math.isfinite(window_t) and window_t > 0.0):
        raise ValueError(f"yaw_tent_income: window_t must be finite and positive, got {window_t}")
    if not (math.isfinite(w_max) and w_max > 0.0):
        raise ValueError(f"yaw_tent_income: w_max must be finite and positive, got {w_max}")
    cbar = sum_cmd / window_t
    p = torch.sign(cbar) * sum_ach / window_t
    value = torch.clamp(torch.minimum(p, 2.0 * torch.abs(cbar) - p), min=-w_max)
    return torch.where(cbar != 0.0, value, torch.zeros_like(value))


def yaw_instant_income(wz: torch.Tensor, cmd_wz: torch.Tensor, w_max: float) -> torch.Tensor:
    """The INSTANTANEOUS turn-progress income ``min(sign(c) * wz, |c|)``, clamped below at ``-w_max``, 0.0 at ``c == 0``.

    ``KRABBY_OMNI_BODY_TENT_RATE=instant`` (default OFF; body-command campaign, 2026-10-16 c6_b stop). The windowed
    tent pays the 2-s mean rate, which the policy and critic cannot observe (their history is far shorter), so its
    credit for a rotation NOW is spread over the next 100 steps -- a non-Markov reward that GAE (lambda gamma ~ 0.94)
    mostly misses. This form pays the CURRENT yaw rate toward the command, Markov in the observation (the base
    angular velocity is observed), one-sided at the command: over-rotation earns nothing extra (the 2-s window term
    still prices accuracy), wrong-way rotation is charged down to ``-w_max``. One-sided rather than the tent's
    ``2|c| - p`` fall-off because per-step gait wobble (~0.09 rad/s std) would make a per-step tent concave: its mean
    for a perfect tracker would sit BELOW zero at slow commands.
    """
    w_max = float(w_max)
    if not (math.isfinite(w_max) and w_max > 0.0):
        raise ValueError(f"yaw_instant_income: w_max must be finite and positive, got {w_max}")
    p = torch.sign(cmd_wz) * wz
    value = torch.clamp(torch.minimum(p, torch.abs(cmd_wz)), min=-w_max)
    return torch.where(cmd_wz != 0.0, value, torch.zeros_like(value))

