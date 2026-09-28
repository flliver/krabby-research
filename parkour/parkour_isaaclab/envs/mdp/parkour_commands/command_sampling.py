# SPDX-License-Identifier: BSD-3-Clause
"""Pure velocity-slot sampler for the obstacle-exposure campaign (PLAN H B1, 2026-09-03).

No Isaac imports -- unit-tested on CPU. ``UniformParkourCommand._resample_command`` calls
:func:`sample_slot` only when ``ParkourCommandCfg.stand_frac`` is set; the unarmed path in
that class is untouched (bit-identical to before), and ``stand_frac=None`` here is a
reference re-implementation of that path used by the tests.

Today's rule (``small_commands_to_zero``): draw vx ~ U(lo, hi) and zero it when
|vx| <= lin_vel_clip. On the 0.0:0.35 training band with clip 0.2 that zeroes ~57% of the
6-s command slots, which is the mechanism behind "the robot stands most of the episode".

The armed rule replaces the *implicit* stand fraction with an explicit Bernoulli(p):
a slot stands with probability p, otherwise vx ~ U(max(clip, lo) + WALK_MARGIN, hi) --
never a sub-clip crawl and never exactly at the gait-clock stop threshold
(``CLOCK_CMD_STOP_M_S = 0.2``: the clock only advances for v > 0.2).
"""
from __future__ import annotations

import torch

WALK_MARGIN_M_S = 0.01
"""Gap above max(clip, lo) for the lowest walking command (keeps the clock running)."""


def walking_band(stand_frac: float, lo: float, hi: float, clip: float) -> tuple[float, float]:
    """The (walk_lo, walk_hi) band a walking slot is drawn from under the armed rule."""
    if not (0.0 <= stand_frac < 1.0):
        raise ValueError(f"stand_frac must be in [0, 1), got {stand_frac}")
    walk_lo = max(clip, lo) + WALK_MARGIN_M_S
    walk_hi = hi
    if walk_hi <= walk_lo:
        raise ValueError(
            f"walking band is empty: hi={hi} <= max(clip={clip}, lo={lo}) + {WALK_MARGIN_M_S}"
        )
    return walk_lo, walk_hi


def sample_slot(
    u_vel: torch.Tensor,
    u_stand: torch.Tensor,
    stand_frac: float | None,
    lo: float,
    hi: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map uniform draws to (vx, stand_mask) for one command slot per env.

    Args:
        u_vel: U[0, 1) draws, shape (n,). Positions the velocity inside its band.
        u_stand: U[0, 1) draws, shape (n,). Decides standing under the armed rule (ignored
            when ``stand_frac`` is None).
        stand_frac: None -> today's clip rule; p in [0, 1) -> Bernoulli(p) standing.
        lo, hi: the configured ``ranges.lin_vel_x`` band.
        clip: ``clips.lin_vel_clip``.

    Returns:
        (vx, stand) -- vx is 0.0 exactly on standing slots.
    """
    if u_vel.shape != u_stand.shape:
        raise ValueError("u_vel and u_stand must have the same shape")
    if stand_frac is None:
        vx = lo + u_vel * (hi - lo)
        stand = vx.abs() <= clip
        return vx * (~stand), stand
    walk_lo, walk_hi = walking_band(stand_frac, lo, hi, clip)
    vx = walk_lo + u_vel * (walk_hi - walk_lo)
    stand = u_stand < stand_frac
    return vx * (~stand), stand
