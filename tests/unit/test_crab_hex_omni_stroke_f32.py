"""The per-foot stance-stroke income under Isaac Lab's REAL float32 contact-refresh drift (stride/
posture plan, plan step 5's "40-s float32 refresh replica" bullet).

``test_crab_hex_omni_stroke_math.py`` already pins ``foot_stroke_step``'s arithmetic against an
ARTIFICIAL 1-in-5 duplicate-reading pattern (``test_stale_force_readings_move_pay_by_at_most_one_
step``). This file instead drives it with the SAME contact-sensor clock that trains
(``replay_motion_terms.sensor_elapsed_f32``, already pinned bit-for-bit against Isaac Lab's own
``SensorBase``/``ContactSensor`` update loop in ``test_crab_hex_omni_replay_motion.py``): a step
whose elapsed time comes back 0 means the sensor did not refresh that env step, so the force
reading the reward sees is last step's, repeated.

A NOTE ON THE GAIT FIXTURE. ``test_crab_hex_omni_stroke_math.py``'s own ``_alternating_gait``
teleports the swinging foot's position instantly to its next stance the moment it lifts off (fine
for that file's tests, which never disagree with the force reading about when the foot is
airborne). Combined with a REAL stale force reading -- which can keep reporting "loaded" for one
extra env step after the true liftoff -- that teleport reads as the foot instantaneously covering
a large world distance while apparently still planted, which the slip gate (correctly) recognises
as a slide and zeroes: with ``_alternating_gait`` verbatim, a 50-step (1 s) stance period is an
exact multiple of the real drift's 5-step repeat period, so EVERY stance transition in the 16-32 s
window inherits this and the window's income collapses to ~0. That is a fixture artifact, not a
training-representative failure: a real foot's position moves continuously, not by teleporting, so
this file uses ``_smooth_alternating_gait`` below (linear interpolation across the swing) instead.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour" / "parkour_tasks" / "parkour_tasks"
STROKE_MATH_PY = TASKS / "crab_hex_omni_task" / "mdp" / "crab_hex_omni_stroke_math.py"
REPLAY_MOTION_PY = TASKS / "crab_hex_omni_task" / "experiments" / "tools" / "replay_motion_terms.py"
STROKE_MATH_TEST_PY = Path(__file__).resolve().parent / "test_crab_hex_omni_stroke_math.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # replay_motion_terms.py's @dataclass needs its module registered
    spec.loader.exec_module(mod)
    return mod


sm = _load("crab_hex_omni_stroke_math__f32_test", STROKE_MATH_PY)
R = _load("replay_motion_terms__f32_test", REPLAY_MOTION_PY)
# Reuse the math test's harness (``_run_stroke``) and constants instead of forking them.
stroke_test = _load("crab_hex_omni_stroke_math_test__f32", STROKE_MATH_TEST_PY)
_run_stroke = stroke_test._run_stroke
DT = stroke_test.DT
F = stroke_test.F
TURN_R = stroke_test.TURN_R
L_TRANSLATION = stroke_test.L_TRANSLATION

V = 0.25  # m/s, steady walk
T_STEPS = 2000  # 40 s at dt 0.02
HALF = 50  # 1-s stances, two feet alternating
ONE_STEP = V * DT / L_TRANSLATION + 1e-6  # a single loaded interval's normalised sweep (sqrt units)


def _smooth_alternating_gait(T: int, v: float, half: int = HALF):
    """Two feet (0, 1) alternate ``half``-step stances at speed ``v``; the swinging foot moves in a
    straight line from its last touchdown to its next, instead of teleporting there (see module
    docstring). Returns ``((force, foot, root, heading, cmd), dict(vel=vel))`` like
    ``test_crab_hex_omni_stroke_math.py``'s ``_alternating_gait``.
    """
    lookahead = half + 2
    x = torch.arange(T + lookahead, dtype=torch.float32) * v * DT
    force = torch.zeros(T, 1, F)
    foot = torch.zeros(T, 1, F, 3)
    root = torch.zeros(T, 1, 3)
    vel = torch.zeros(T, 1, 2)
    vel[:, 0, 0] = v
    root[:, 0, 0] = x[:T]
    body = {0: torch.tensor([0.4, 0.8]), 1: torch.tensor([0.4, -0.8])}

    def td_pos(i: int, block_start_t: int) -> torch.Tensor:
        return torch.tensor([x[block_start_t].item(), 0.0]) + body[i]

    n_blocks = T // half + 2
    for i in (0, 1):
        for b in range(n_blocks):
            blk_start = b * half
            if blk_start >= T:
                break
            blk_end = min(blk_start + half, T)
            if b % 2 == i:  # foot i is planted this block
                p = td_pos(i, blk_start)
                foot[blk_start:blk_end, 0, i, :2] = p
                force[blk_start:blk_end, 0, i] = 300.0
            else:  # swinging: straight line from where it lifted off to where it lands next
                p0 = td_pos(i, max(blk_start - half, 0))
                p1 = td_pos(i, blk_start + half)
                n = blk_end - blk_start
                frac = torch.arange(n, dtype=torch.float32) / half
                foot[blk_start:blk_end, 0, i, :2] = p0 * (1 - frac[:, None]) + p1 * frac[:, None]
    cmd = torch.zeros(T, 1, 3)
    cmd[..., 0] = v
    return (force, foot, root, torch.zeros(T, 1), cmd), dict(vel=vel)


def _apply_stale(force: torch.Tensor, elapsed: np.ndarray) -> torch.Tensor:
    """The force the reward actually sees under the real sensor clock: repeat the previous step's
    reading wherever ``elapsed`` (:func:`sensor_elapsed_f32`) is 0 (no refresh this env step)."""
    stale = force.clone()
    for t in range(1, force.shape[0]):
        if elapsed[t] == 0.0:
            stale[t] = stale[t - 1]
    return stale


# ------------------------------------------------------------------------------- per-window income
def test_real_f32_refresh_pattern_pays_the_same_per_second_in_every_window():
    args, kw = _smooth_alternating_gait(T_STEPS, V)
    force, foot, root, hdg, cmd = args
    elapsed = R.sensor_elapsed_f32(T_STEPS)
    stale = _apply_stale(force, elapsed)

    # Not vacuous: the real pattern must actually create repeats, and only in the 16-32 s window
    # (matches sensor_elapsed_f32's own documented ~2 s onset / ~16 s slip-a-refresh regime).
    zero = elapsed == 0.0
    assert not zero[:800].any(), "no drift is expected before ~16 s of episode time"
    assert zero[800:1600].any(), "the real f32 pattern must contain repeats in 16-32 s, or this test is vacuous"
    assert not torch.equal(stale[800:1600], force[800:1600]), "staleness must actually change some readings"

    _, pay_fresh = _run_stroke(force, foot, root, hdg, cmd, **kw)
    _, pay_stale = _run_stroke(stale, foot, root, hdg, cmd, **kw)

    for lo, hi in ((0, 800), (800, 1600), (1600, 2000)):
        fresh_per_s = pay_fresh[lo:hi].sum().item() / ((hi - lo) * DT)
        stale_per_s = pay_stale[lo:hi].sum().item() / ((hi - lo) * DT)
        assert stale_per_s == pytest.approx(fresh_per_s, rel=0.05), (
            f"window [{lo}:{hi}] (steps): fresh {fresh_per_s:.4f}/s vs stale {stale_per_s:.4f}/s"
        )


# --------------------------------------------------------------------- dip + stale close-it-early
def test_one_step_dip_doubled_by_a_real_stale_step_closes_the_stance_early_but_barely_moves_pay():
    """A one-step load dip is, alone, fully preserved (``test_one_step_dip_neither_closes_the_
    stance_nor_loses_its_sweep``). Under the REAL sensor clock, a dip landing right before a step
    the sensor fails to refresh gets its own low reading echoed one step later -- two consecutive
    low readings -- which closes the stance ``N_OFF`` steps early instead of carrying on to the
    honest liftoff. This is the mechanism the math module's docstring calls out (``loaded_stance_
    edges``: "a stale repeated reading can shift an edge by ... at most one step"); the accumulator
    that stance loses is forfeit (an early close discards it, per the module's own "dropped if the
    stance closes" rule for the pending buffer), so the ONE affected stance itself can lose several
    steps' worth. Summed over the whole 40 s walk, though, that loss is mostly offset by the many
    OTHER stances whose closing edge merely shifts by the harmless one step the docstring promises
    (their pay moves in time, not in amount) -- so in the aggregate the net effect stays within a
    small, bounded multiple of one step's normalised sweep, not the isolated stance's own multi-step
    loss. (Measured over a spread of real repeat positions in the 16-32 s window: 1.4x-2.2x one
    step; this asserts comfortably above that observed range.)
    """
    args, kw = _smooth_alternating_gait(T_STEPS, V)
    force, foot, root, hdg, cmd = args
    elapsed = R.sensor_elapsed_f32(T_STEPS)

    # A real repeat position inside foot 0's stance at block 16 (steps 800-849); the dip sits one
    # step before it so the repeat echoes the dip's own (low) reading, not the honest reload.
    dip_t, foot_idx = 809, 0
    assert elapsed[dip_t + 1] == 0.0, "fixture assumption: this env step must be a real stale repeat"
    assert (dip_t // HALF) % 2 == foot_idx, "fixture assumption: foot 0 must be planted at dip_t"

    force_dip = force.clone()
    force_dip[dip_t, 0, foot_idx] = 5.0  # a one-step load dip (below FORCE_HOLD_N, above nothing)
    _, pay_fresh = _run_stroke(force_dip, foot, root, hdg, cmd, **kw)

    stale = _apply_stale(force_dip, elapsed)
    assert stale[dip_t + 1, 0, foot_idx].item() == pytest.approx(5.0), (
        "the real pattern must echo the dip's own low reading onto the next step"
    )
    _, pay_stale = _run_stroke(stale, foot, root, hdg, cmd, **kw)

    # The dip alone (fresh) changes nothing: it is fully preserved, same as the isolated math test.
    _, pay_no_dip = _run_stroke(force, foot, root, hdg, cmd, **kw)
    assert pay_fresh.sum().item() == pytest.approx(pay_no_dip.sum().item(), rel=1e-4)

    # It DOES close the affected stance early: the honest close (dip alone) is a single full-value
    # payment; under staleness that same stance instead closes in two pieces, earlier.
    fresh_events = [(t, pay_fresh[t, 0, foot_idx].item()) for t in range(795, 900) if pay_fresh[t, 0, foot_idx] > 0]
    stale_events = [(t, pay_stale[t, 0, foot_idx].item()) for t in range(795, 900) if pay_stale[t, 0, foot_idx] > 0]
    assert len(fresh_events) == 1, fresh_events
    assert len(stale_events) >= 2, "staleness must split the stance into an early close plus a reopened remainder"
    assert stale_events[0][0] < fresh_events[0][0], "the stale-induced close must land earlier than the honest one"

    pay_ref_total = pay_fresh.sum().item()
    pay_stale_total = pay_stale.sum().item()
    diff = abs(math.sqrt(pay_stale_total) - math.sqrt(pay_ref_total))
    assert diff <= 3.0 * ONE_STEP, (
        f"aggregate 40-s sqrt(pay) moved by {diff:.5f}, more than 3 loaded steps' worth "
        f"({3.0 * ONE_STEP:.5f}) of normalised sweep -- the dip+stale interaction is costing more "
        "than the isolated-mechanism bound; re-check loaded_stance_edges' hysteresis"
    )
