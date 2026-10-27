#!/usr/bin/env python3
"""Stand-network campaign -- the pure pieces of the random-command VIDEO (``sn_video_schedule.py`` generates the
schedule, ``sn_record_video.py`` records it). VIDEO ONLY: nothing here is read by a gate, a ledger or a reader.

Torch-free and Isaac-free at import (torch is imported inside :func:`route_step` only), so the CPU unit tests
(``tests/unit/test_sn_video.py``) load it by path and the recorder loads it before Kit starts.

THE STAND-NESS RULE THE VIDEO ROUTES ON (:data:`STAND_RULE_SCHEDULE`). A hold is a STAND iff its speed is 0 AND its
facing target equals the previous hold's (hold 0: the spawn-yaw pin, which ``spawn_yaw: first_face`` makes equal to
hold 0's own face). A speed-0 hold that changes the facing is a TURN and stays S1's for its whole length, even once
the turn has settled. The prologue (segment -1, which holds hold 0's command) takes hold 0's flag. Per step:
``stand[t] = flag[segment_id[t]]`` -- entry on the first step of a stand hold, exit on the step the hold changes (the
campaign's hard switch, PLAN D5: no debounce).

How it relates to the EVAL's rule (``enter_rule: latched_stand_slot``: the command term's ``stand_mask()`` with the
slot latch armed and the turning bit re-derived on reset). The latch re-arms the turning bit only on a FRESH facing
target (``heading_target != face``) or a reset, from the raw relay output, and keeps it until the next one; a move
at an unchanged face is NOT a fresh target. So the two rules agree on every hold the campaign's eval rows run (each
wave starts from a reset: suite settle / stand, walkstop's walk -> stand at one face, field rows), but once a hold
has changed the facing, EVERY later stand hold (not only the one right after the turn) inherits the latched TURN
until the next reset: the eval's slot would have kept S1 on all of them (:func:`eval_slot_stand_flags` predicts it
for a fall-free run). A fall-respawn re-derives the bit from the relay at the respawn yaw (the spawn pin), so after a
fall at a face other than the pin the slot reads TURN as well, and the schedule's stand hold carries a live yaw-rate
command (:func:`yaw_cmd_stats` marks it). The training sampler draws a fresh slot instead (a stand slot there is a
stand whatever came before), and the user asked for the stand network on every stand, so the video routes on the
schedule rule and the recorder records the latched slot beside it (per-step and per-hold agreement in the sanity
JSON; ``--stand-rule slot`` routes on the eval's rule instead).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

#: the rule the recorder routes on by default (also written into the sanity JSON)
STAND_RULE_SCHEDULE = ("schedule: a hold is STAND iff speed == 0 and its face equals the previous hold's face "
                       "(hold 0: the spawn-yaw pin); the prologue takes hold 0's flag; stand[t] = flag[segment_id[t]]")
STAND_RULE_SLOT = ("slot: the command term's latched stand slot (cmd_term.stand_mask() with the eval slot latch "
                   "armed, turning bit re-derived on reset) -- the campaign eval's enter_rule latched_stand_slot")
STAND_RULES = ("schedule", "slot")

#: two faces are "the same facing" within this (rad); the generator copies the previous face exactly
FACE_TOL = 1e-9
KIND_STAND, KIND_TURN, KIND_MOVE, KIND_MOVE_TURN = "stand", "turn", "move", "move_turn"
KINDS = (KIND_STAND, KIND_TURN, KIND_MOVE, KIND_MOVE_TURN)
COMMAND_KINDS = (KIND_MOVE, KIND_TURN, KIND_MOVE_TURN)

#: p2s_stand.py (phase 2): home is cam = 0; "homed" = every cam within 0.10 rad
HOMED_RAD = 0.10
#: p2s_stand.py: the G2 homed window (the stand's last 5 s) and the settle rule (homed held 0.5 s running)
HOMED_LAST_S = 5.0
SETTLE_HOLD_S = 0.5
#: the cam joints (``run_meta.joint_names`` suffix, p2s_stand.py)
CAM_SUFFIX = "_Body_CamShaft_RevoluteJoint"

LABEL_RE = re.compile(r"^\d{2}_(stand\d+\.\ds|move\d\.\d{2}|turn[+-]\d+|moveturn\d\.\d{2}[+-]\d+)$")


def wrap_to_pi(a: float) -> float:
    """``a`` wrapped into (-pi, pi] (``gait_eval.schedule.wrap_to_pi``'s convention)."""
    return float(math.remainder(float(a), 2.0 * math.pi))


def same_face(a: float, b: float) -> bool:
    return abs(wrap_to_pi(float(a) - float(b))) <= FACE_TOL


def hold_kind(speed: float, face: float, prev_face: float) -> str:
    """``stand`` / ``turn`` (speed 0, face change) / ``move`` / ``move_turn`` (speed > 0, face change)."""
    speed = float(speed)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError(f"hold_kind: speed must be finite and >= 0, got {speed}")
    turning = not same_face(face, prev_face)
    if speed == 0.0:
        return KIND_TURN if turning else KIND_STAND
    return KIND_MOVE_TURN if turning else KIND_MOVE


def _swf(h) -> tuple[float, float, float]:
    """``(speed, dir, face)`` of a hold given as a tuple, a mapping, or a ``gait_eval.schedule.Hold``."""
    if hasattr(h, "as_world"):
        return tuple(float(x) for x in h.as_world())
    if isinstance(h, dict):
        return float(h["speed"]), float(h["dir"]), float(h["face"])
    s, d, f = h
    return float(s), float(d), float(f)


def hold_kinds(holds, spawn_face: float | None = None) -> list[str]:
    """Each hold's kind relative to the PREVIOUS hold's face (hold 0: ``spawn_face``; None = hold 0's own face, i.e.
    ``spawn_yaw: first_face``)."""
    out = []
    prev = None if spawn_face is None else float(spawn_face)
    for i, h in enumerate(holds):
        speed, _d, face = _swf(h)
        if prev is None:
            prev = face
        out.append(hold_kind(speed, face, prev))
        prev = face
    return out


def hold_stand_flags(holds, spawn_face: float | None = None) -> list[bool]:
    """The schedule stand-ness rule (:data:`STAND_RULE_SCHEDULE`), one flag per hold."""
    return [k == KIND_STAND for k in hold_kinds(holds, spawn_face)]


#: the omni command term's facing dead band (``clips.ang_vel_clip`` 0.2 rad/s over the facing P-gain: 0.133 rad, the
#: ``crab_hex_omni_command.py`` cfg note); a facing change larger than this latches TURN at its fresh target
FACING_DEAD_BAND_RAD = 0.133


def eval_slot_stand_flags(holds, spawn_face: float | None = None, *,
                          dead_band_rad: float = FACING_DEAD_BAND_RAD) -> list[bool]:
    """The campaign EVAL's stand verdict per hold (``enter_rule: latched_stand_slot``), PREDICTED for a fall-free run
    in which the robot reaches each face before the next facing change.

    The command term latches its turning bit from the raw relay output when a FRESH facing target arrives (the face
    differs from the held one) or after a reset, and keeps it until the next fresh target; the relay is non-zero iff
    the facing error exceeds the dead band. Hold 0 follows the reset with the robot at the spawn pin (``spawn_face``;
    None = hold 0's own face, ``spawn_yaw: first_face``). A hold reads STAND iff its speed is 0 and the latch is
    clear. Consequence: after the first facing change larger than the dead band, every stand reads TURN until a
    reset (contrast :func:`hold_stand_flags`, the rule the video routes on)."""
    out: list[bool] = []
    prev = None if spawn_face is None else float(spawn_face)
    latched = False
    for i, h in enumerate(holds):
        speed, _d, face = _swf(h)
        if prev is None:
            prev = face
        dface = abs(wrap_to_pi(face - prev))
        if i == 0 or dface > FACE_TOL:                 # the reset's re-derivation / a fresh facing target
            latched = dface > float(dead_band_rad)
        out.append(speed == 0.0 and not latched)
        prev = face
    return out


def step_stand_mask(segment_id, flags):
    """``(T,)`` bool: ``flags[segment_id[t]]``, the prologue (``segment_id == -1``) taking hold 0's flag. ``segment_id``
    is env 0's column of ``CompiledSchedule.segment_id`` (or any 1-D int array)."""
    import numpy as np  # noqa: PLC0415

    seg = np.asarray(segment_id).astype(np.int64).reshape(-1)
    fl = np.asarray(list(flags), dtype=bool)
    if fl.size == 0:
        raise ValueError("step_stand_mask: no holds")
    if seg.size and (seg.min() < -1 or seg.max() >= fl.size):
        raise ValueError(f"step_stand_mask: segment ids {seg.min()}..{seg.max()} outside -1..{fl.size - 1}")
    return fl[np.where(seg < 0, 0, seg)]


def route_step(SN, obs, s1_policy, stand_policy, stand_estimator, stand_rows, *, p0: int, p1: int, num_prop: int,
               mask):
    """One control step of the routed pair, in the campaign launcher's order (``StandNetMode.__call__``):

    ``a_s1 = s1_policy(obs, hist_encoding=True)`` on the harness's obs (S1's own estimate already written into the
    priv-explicit slice by the caller, as the harness does); ``x = SN.stand_net_input(obs, stand_estimator, ...)`` --
    the stand ckpt's own estimate on the UNMASKED prop, a NEW tensor, masked iff ``mask``; ``a_st =
    stand_policy(x, hist_encoding=True)``; ``actions = SN.route_actions(stand_rows, a_st, a_s1)`` (the stand net on
    ``stand_rows``, S1 elsewhere). Both networks act on every row. ``obs`` is never written. Returns ``(actions, a_s1,
    a_st)``."""
    a_s1 = s1_policy(obs, hist_encoding=True)
    x = SN.stand_net_input(obs, stand_estimator, p0=p0, p1=p1, num_prop=num_prop, mask=mask)
    a_st = stand_policy(x, hist_encoding=True).to(a_s1.dtype)
    actions = SN.route_actions(stand_rows, a_st, a_s1)
    return actions, a_s1, a_st


def first_run_start(cond, need: int):
    """The first index from which ``cond`` holds for ``need`` rows running, or None (p2s_stand._first_run_start)."""
    run = 0
    for t, c in enumerate(cond):
        run = run + 1 if c else 0
        if run >= need:
            return t - need + 1
    return None


def stand_hold_stats(cam_abs_max, idx, dt: float, first_fall_step=None) -> dict:
    """Homing of ONE stand hold from the per-step worst-leg ``max |wrap(cam)|`` (pre-step, as the harness records
    ``joint_pos``): ``cam_abs_max`` (T,), ``idx`` the hold's step indices, trimmed at ``first_fall_step``.

    * ``cam_max_end_rad`` / ``homed_at_end``: the worst leg at the hold's LAST row, homed iff <= 0.10 rad;
    * ``settle_s``: seconds from the hold's first row to the first row from which every cam stays <= 0.10 for 0.5 s
      running (p2s_stand's settle rule), None if never inside the hold;
    * ``cam_max_last5s_rad`` / ``homed_last5s``: p2s_stand's G2 statistic (max over the LAST 5 s, or the whole hold
      when it is shorter -- so a 3-s stand must be homed from its first row to read homed here).
    """
    import numpy as np  # noqa: PLC0415

    cam = np.asarray(cam_abs_max, dtype=np.float64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if first_fall_step is not None:
        idx = idx[idx < int(first_fall_step)]
    out = {"rows": int(idx.size), "cam_max_end_rad": None, "homed_at_end": None, "settle_s": None,
           "settled": False, "cam_max_last5s_rad": None, "homed_last5s": None}
    if not idx.size:
        return out
    c = cam[idx]
    out["cam_max_end_rad"] = float(c[-1])
    out["homed_at_end"] = bool(c[-1] <= HOMED_RAD)
    t0 = first_run_start(c <= HOMED_RAD, max(1, int(round(SETTLE_HOLD_S / dt))))
    out["settled"] = t0 is not None
    out["settle_s"] = None if t0 is None else float(t0 * dt)
    last = c[-max(1, int(round(HOMED_LAST_S / dt))):]
    out["cam_max_last5s_rad"] = float(last.max())
    out["homed_last5s"] = bool(last.max() <= HOMED_RAD)
    return out


def yaw_cmd_stats(routed_stand, cmd_wz, idx, fall_steps=()) -> dict:
    """Stand-net steps of ONE hold that carried a LIVE yaw-rate command. ``cmd_wz`` (T,) is the body yaw-rate command
    the step's observation carried (non-zero = the relay survived the facing dead band: the command term's own
    "turning"), ``routed_stand`` (T,) the routed flag, ``idx`` the hold's steps, ``fall_steps`` the steps on which an
    episode terminated (the respawn happens inside that step, so a step ``> f`` is post-respawn).

    * ``yaw_cmd_at_entry``: the hold's first step carried one -- the robot is not at the face when the hold starts (a
      turn slower than planned, or a respawn at the spawn pin while the schedule keeps its absolute face). In the
      eval a stand entered this way reads TURN (the latch re-derives from this relay output after a reset).
    * ``stand_net_on_yaw_cmd_steps``: routed-to-stand-net steps with a yaw command. (The eval's stand net does see
      some: a latched stand that drifts past the dead band mid-hold.)
    * ``post_fall`` / ``post_fall_turn_cmd``: some step of the hold comes after a termination / some routed step after
      a termination carried a yaw command -- the stand net acting on a turn command after a respawn, which no gate
      evaluated.
    """
    import numpy as np  # noqa: PLC0415

    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = {"yaw_cmd_at_entry": None, "stand_net_on_yaw_cmd_steps": 0, "post_fall": False,
           "post_fall_turn_cmd": False}
    if not idx.size:
        return out
    routed = np.asarray(routed_stand, dtype=np.float64).reshape(-1)[idx] > 0.5
    yaw = np.asarray(cmd_wz, dtype=np.float64).reshape(-1)[idx] != 0.0
    falls = sorted(int(f) for f in fall_steps)
    after = (idx > falls[0]) if falls else np.zeros(idx.size, dtype=bool)
    on = routed & yaw
    out["yaw_cmd_at_entry"] = bool(yaw[0])
    out["stand_net_on_yaw_cmd_steps"] = int(on.sum())
    out["post_fall"] = bool(after.any())
    out["post_fall_turn_cmd"] = bool((on & after).any())
    return out


_S1_SEED_RE = re.compile(r"(?:^|_)seed(\d+)(?:_|$)")


def stand_tag_from_ckpt(sn_dagger, ckpt_path) -> str:
    """The caption name of the stand network, from the LOADED checkpoint's ``infos.sn_dagger`` (``arm`` / ``seed`` /
    ``smoke`` / ``iterations``): ``A-mask s17``; ``(model_2000)`` is appended for a checkpoint other than the run's
    last, `` SMOKE`` for a smoke run. No ``infos.sn_dagger`` (S1 itself, the identity control): ``S1 identity``."""
    if not isinstance(sn_dagger, dict):
        return "S1 identity"
    arm, seed = sn_dagger.get("arm"), sn_dagger.get("seed")
    name = f"A-{arm}" if arm in ("mask", "full") else (str(arm) if arm else Path(str(ckpt_path)).parent.name)
    tag = name + (f" s{seed}" if seed is not None else "")
    its = sn_dagger.get("iterations")
    stem = Path(str(ckpt_path)).stem
    if its is not None and stem != f"model_{its}":
        tag += f" ({stem})"
    if sn_dagger.get("smoke"):
        tag += " SMOKE"
    return tag


def s1_tag_from_ckpt(ckpt_path, *, is_stand_nets_s1: bool) -> str:
    """The caption name of the walking network: ``S1 s18`` when it is the S1 the stand net was trained beside (by
    sha256) and its run dir names a seed, ``S1 <run dir>`` without one; ``walker ...`` when it is not that S1."""
    run = Path(str(ckpt_path)).parent.name
    m = _S1_SEED_RE.search(run)
    return ("S1" if is_stand_nets_s1 else "walker") + (f" s{m.group(1)}" if m else f" {run}")
