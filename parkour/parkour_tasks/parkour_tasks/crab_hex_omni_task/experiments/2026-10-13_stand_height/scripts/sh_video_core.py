#!/usr/bin/env python3
"""Stand-height campaign (``2026-10-13_stand_height``) -- the pure pieces of the step-8 RAISE / LOWER VIDEO
(``sh_video_schedule.py`` generates the schedule, ``sh_record_video.py`` records it). VIDEO ONLY: nothing here is read
by a gate, a ledger or a reader. Adapted copy of the closed predecessor's ``2026-10-11_stand_network/scripts/
sn_video_core.py`` (never edited; its torch-free helpers -- the homing statistic, the walker caption tag -- are loaded
by path and reused, not re-stated).

Torch-free and Isaac-free at import (torch is imported inside the functions that need it), so the CPU unit tests
(``tests/unit/test_sh_video.py``) load it by path and the recorder loads it before Kit starts.

THE CONTROL PATH IS THE LAUNCHER'S, COMPOSED FROM ``sh_core`` (no control logic here). :func:`height_step` is the
``stand_net`` branch of ``sh_eval_crab_hex_gait.StandHeightMode.__call__`` (the stand_height eval launcher), call for
call, in its order:

1. ``cut = ctl.reset(episode_length_buf == 0, "walk", h_cmd=h_cmd)`` -- every eval start is a walk handover (the
   limiter starts at the command);
2. ``hs = ctl.step(slot, h_cmd, obs)`` -- :class:`sh_core.HeightController`: the H6 restore edges, completion /
   timeout, the 0.08 m/s limiter -> ``hs.h_lim``, the EFFECTIVE stand vector ``hs.stand_eff = slot | restoring``;
3. ``a_s1 = S1(obs, hist_encoding=True)`` on the harness's obs (S1's estimate already in the priv-explicit slice);
4. ``x = sh_core.stand_net_input(obs, stand_estimator, p0, p1, num_prop, mask, h_cmd_limited=hs.h_lim,
   restoring=hs.restoring)`` (column 15 = the limited command; restore rows show a stand: columns 5 / 8 / 9 / 10 = 0);
5. ``a_net = stand(x, hist_encoding=True)``; ``a_stand, q18 = sh_core.stand_action(a_net, hs.h_lim,
   deadband=deadband)`` = ``q(h_lim) + deadband(clamp(a_net, +-1))``, rods +-4.8, cams +-1 (deadband 0 = no op);
6. ``actions = sh_core.route(hs.stand_eff, a_stand, a_s1)`` -- stand rows the stand action, S1 rows ``clamp(S1, +-1)``.

The recorder raises the wrapper clip to the launcher's runtime 4.8 (``sh_core.HARNESS_CLIP``), as the launcher does.
``tests/unit/test_sh_video.py`` pins :func:`height_step` BITWISE against the launcher's own ``__call__`` (the executed
tensor it swaps in at ``env.step``) over a raise / lower / restore / reset sequence, deadband 0 and 0.25.

PER-HOLD HEIGHTS reach the controller exactly as on the launcher's ``--sh-command manifest`` path: the schedule
manifest's ``stand_height`` block, validated by ``sh_eval_common.height_row`` (one entry per hold, "stand" labels,
the frozen range, one facing, walk first and last, the gate-window hold lengths), mapped per step by
``sh_eval_common.height_table(compiled.segment_id, hold_heights)`` (walk holds 1.064, the prologue the first hold's).

CAPTION READINGS (the recorder measures them on the state the frame shows, i.e. after ``env.step``):

* mode: ``walk`` (S1 acts), ``stand`` (the latched stand slot: the stand net), ``restoring`` (an H6 restore: the stand
  net drives the command to 1.064 before handing back);
* ``h_meas`` = ``sh_core.h_meas(root z, sh_core.terrain_z_under_feet(...))`` (the harness's raycast; the trainer's
  ``KitProbe`` definition);
* feet loaded = footpad net contact >= ``sh_dagger_spec.FOOT_LOADED_N`` (20 N) AND footpad <= ``FOOT_FLOOR_M``
  (0.15 m) above the terrain under it (the trainer's ``KitProbe`` / train-time STOOD rule);
* homed = every cam within ``HOMED_RAD`` = 0.10 rad of home (p2s_stand's rule, the predecessor video's).
"""
from __future__ import annotations

import collections
import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
EXPERIMENTS = HERE.parents[1]
PRED_SCRIPTS = EXPERIMENTS / "2026-10-11_stand_network" / "scripts"
EC_PATH = HERE / "sh_eval_common.py"
SPEC_PATH = HERE / "sh_dagger_spec.py"
SNV_PATH = PRED_SCRIPTS / "sn_video_core.py"
LAUNCHER_PATH = HERE / "sh_eval_crab_hex_gait.py"


def _load_by_path(path: Path, name: str):
    """A private module instance loaded by path (not registered in ``sys.modules``; no ``__pycache__`` written)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"sh_video_core: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


#: the eval contract (schedule manifest loader, height table) and, through it, THE core (all torch-free at import)
EC = _load_by_path(EC_PATH, "_shv_eval_common")
CORE = EC.CORE
SN = CORE.SN
#: the trainer's torch-free spec: the feet-loaded rule (FOOT_LOADED_N / FOOT_FLOOR_M)
SPEC = _load_by_path(SPEC_PATH, "_shv_dagger_spec")
#: the predecessor's video core (homing statistic, walker tag, cam suffix), reused unchanged
SNV = _load_by_path(SNV_PATH, "_shv_sn_video_core")

# ===================================================================================== constants
#: the launcher's forward string (run_meta p2r_mode.stand_height.forward on a network controller)
SH_FORWARD = ("act_inference(sh_core.stand_net_input(obs, stand_estimator, mask, h_cmd_limited, restoring), "
              "hist_encoding=True)")
#: the only controller the video runs (the final head configuration: a stand-height trainer checkpoint)
CONTROLLER = "stand_net"
RESTORE = "armed"
HOMED_RAD = float(SNV.HOMED_RAD)                 # 0.10 rad, p2s_stand
CAM_SUFFIX = SNV.CAM_SUFFIX
FOOT_LOADED_N = float(SPEC.FOOT_LOADED_N)        # 20 N
FOOT_FLOOR_M = float(SPEC.FOOT_FLOOR_M)          # 0.15 m
MODE_WALK, MODE_STAND, MODE_RESTORING = "walk", "stand", "restoring"
MODES = (MODE_WALK, MODE_STAND, MODE_RESTORING)
KIND_WALK, KIND_STAND = "walk", "stand"
#: K1's window (the last 2 s of a stand hold) and the arrival settle (sh_eval_common), for the per-hold read-out
K1_WINDOW_S = float(EC.K1_WINDOW_S)
ARRIVAL_SETTLE_S = float(EC.ARRIVAL_SETTLE_S)

# caption colours (RGB, the predecessor's palette)
WHITE, YELLOW, CYAN, RED, GREEN = (255, 255, 255), (255, 230, 90), (120, 230, 255), (255, 80, 60), (120, 255, 120)
MODE_COLOR = {MODE_WALK: (170, 255, 170), MODE_STAND: (255, 120, 235), MODE_RESTORING: (255, 170, 60)}
ARROW_LEGEND = "arrows: GREEN = commanded travel   ORANGE = commanded facing   BLUE = achieved velocity"


# ===================================================================================== the controller path
def new_controller(n: int, device, *, dt: float, h_cmd0=None):
    """The launcher's ``_start_pass_state`` for ``--sh-restore armed``: ``sh_core.HeightController(n, device, dt=dt,
    engage_m=RESTORE_ENGAGE_M)`` reset as a walk handover at the pass's first command (``h_cmd0``: the height table's
    first row; None = 1.064)."""
    ctl = CORE.HeightController(n, device, dt=float(dt), engage_m=CORE.RESTORE_ENGAGE_M)
    ctl.reset(None, "walk", h_cmd=h_cmd0)
    return ctl


HeightVideoStep = collections.namedtuple(
    "HeightVideoStep", ("actions", "a_s1", "a_net", "a_stand", "q18", "hs", "cut", "x"))


def height_step(ctl, obs, *, slot, reset_rows, h_cmd, s1_policy, stand_policy, stand_estimator, p0: int, p1: int,
                num_prop: int, mask, deadband: float = 0.0) -> HeightVideoStep:
    """ONE control step of the launcher's ``stand_net`` controller (module docstring, steps 1-6), composed from
    ``sh_core`` only. ``slot``: the latched stand slot ((N,) bool, ``P2RMode.rederive_on_reset``); ``reset_rows``:
    ``episode_length_buf == 0`` ((N,) bool); ``h_cmd``: the step's raw height command ((N,) float64, the height
    table's row; None = 1.064). ``obs`` is never written. Returns the EXECUTED ``actions`` and every intermediate."""
    d = CORE.check_deadband(deadband)
    cut = ctl.reset(reset_rows, "walk", h_cmd=h_cmd)
    hs = ctl.step(slot, h_cmd, obs)
    a_s1 = s1_policy(obs, hist_encoding=True)
    x = CORE.stand_net_input(obs, stand_estimator, p0=int(p0), p1=int(p1), num_prop=int(num_prop), mask=bool(mask),
                             h_cmd_limited=hs.h_lim, restoring=hs.restoring)
    a_net = stand_policy(x, hist_encoding=True).to(a_s1.dtype)
    a_stand, q = CORE.stand_action(a_net, hs.h_lim, deadband=d)
    actions = CORE.route(hs.stand_eff, a_stand, a_s1)
    return HeightVideoStep(actions, a_s1, a_net, a_stand, q, hs, cut, x)


def mode_of(slot: bool, restoring: bool) -> str:
    """``restoring`` (an H6 restore row) > ``stand`` (the latched slot) > ``walk``."""
    if restoring:
        return MODE_RESTORING
    return MODE_STAND if slot else MODE_WALK


# ===================================================================================== readings
def feet_loaded_mask(force_n, foot_z, terrain_z):
    """(..., 6) bool: net contact >= FOOT_LOADED_N and footpad <= FOOT_FLOOR_M above the terrain under it (the
    trainer's ``KitProbe`` rule). Works on torch tensors and numpy arrays alike."""
    return (force_n >= FOOT_LOADED_N) & ((foot_z - terrain_z) <= FOOT_FLOOR_M)


def cam_abs(cam_pos):
    """|wrap(cam)| (torch or numpy, elementwise): the cam's distance from home."""
    if hasattr(cam_pos, "remainder"):
        return (cam_pos + math.pi).remainder(2.0 * math.pi).sub(math.pi).abs()
    import numpy as np  # noqa: PLC0415

    return np.abs(np.remainder(np.asarray(cam_pos) + math.pi, 2.0 * math.pi) - math.pi)


def is_homed(cam_abs_max: float) -> bool:
    return bool(float(cam_abs_max) <= HOMED_RAD)


def hold_kinds(holds) -> list[str]:
    """``walk`` (speed > 0) / ``stand`` (speed 0) per hold (a height row has one facing: no turn holds)."""
    out = []
    for h in holds:
        sp = float(h.speed if hasattr(h, "speed") else h["speed"])
        out.append(KIND_WALK if sp > 0.0 else KIND_STAND)
    return out


def stand_hold_height_stats(h_cmd, h_lim, h_meas, feet, idx, dt: float, first_fall_step=None) -> dict:
    """One STAND hold's height read-out (VIDEO ONLY; the K-readers decide nothing from it): ``h_cmd`` / ``h_lim`` /
    ``h_meas`` (T,) float, ``feet`` (T,) loaded-feet counts, ``idx`` the hold's steps (trimmed at a fall).

    * ``h_cmd``: the hold's command; ``arrival_s``: seconds from the hold's first step to the first step whose limited
      command equals it exactly (the limiter snaps), None if never; ``slack_s`` = hold length - arrival;
    * ``window``: the last ``K1_WINDOW_S`` s of the hold when it opens >= arrival + ``ARRIVAL_SETTLE_S`` (else None):
      median / max |h_meas - h_cmd| there, min loaded feet there;
    * ``h_meas_end``, ``feet_end``."""
    import numpy as np  # noqa: PLC0415

    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if first_fall_step is not None:
        idx = idx[idx < int(first_fall_step)]
    out = {"rows": int(idx.size), "h_cmd": None, "arrival_s": None, "slack_s": None, "window": None,
           "h_meas_end": None, "feet_end": None}
    if not idx.size:
        return out
    hc = np.asarray(h_cmd, dtype=np.float64)[idx]
    hl = np.asarray(h_lim, dtype=np.float64)[idx]
    hm = np.asarray(h_meas, dtype=np.float64)[idx]
    ft = np.asarray(feet)[idx]
    cmd = float(hc[0])
    out["h_cmd"] = cmd
    hit = np.nonzero(hl == cmd)[0]
    if hit.size:
        out["arrival_s"] = float(hit[0] * dt)
        out["slack_s"] = float((idx.size - hit[0]) * dt)
    w = max(1, int(round(K1_WINDOW_S / dt)))
    if hit.size and idx.size - w >= hit[0] + int(round(ARRIVAL_SETTLE_S / dt)):
        err = np.abs(hm[-w:] - cmd)
        out["window"] = {"s": K1_WINDOW_S, "median_abs_err_m": float(np.median(err)),
                         "max_abs_err_m": float(err.max()), "feet_min": int(ft[-w:].min())}
    out["h_meas_end"] = float(hm[-1])
    out["feet_end"] = int(ft[-1])
    return out


def restore_hold_stats(restoring, engaged, completed, timed_out, idx, dt: float) -> dict:
    """The H6 restore events inside one hold (a walk after a stand): counts, restore rows (s) and the completion time
    from the hold's first step."""
    import numpy as np  # noqa: PLC0415

    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = {"engaged": 0, "completed": 0, "timed_out": 0, "restoring_s": 0.0, "completed_at_s": None}
    if not idx.size:
        return out
    r = np.asarray(restoring)[idx] > 0.5
    c = np.asarray(completed)[idx] > 0.5
    out["engaged"] = int((np.asarray(engaged)[idx] > 0.5).sum())
    out["completed"] = int(c.sum())
    out["timed_out"] = int((np.asarray(timed_out)[idx] > 0.5).sum())
    out["restoring_s"] = float(r.sum() * dt)
    if c.any():
        out["completed_at_s"] = float(np.nonzero(c)[0][0] * dt)
    return out


# ===================================================================================== caption names
def stand_tag_from_ckpt(sh_dagger, ckpt_path, deadband: float) -> str:
    """The stand network's caption name from the LOADED checkpoint's ``infos.sh_dagger`` (run dir, ``seed``, the
    checkpoint stem when it is not the run's last, ``SMOKE``) and the evaluated deadband: ``SH_T18 s18 db0.25``."""
    p = Path(str(ckpt_path))
    sd = sh_dagger if isinstance(sh_dagger, dict) else {}
    tag = p.parent.name or "stand net"
    if sd.get("seed") is not None:
        tag += f" s{sd['seed']}"
    its = sd.get("iterations")
    if its is not None and p.stem != f"model_{its}":
        tag += f" ({p.stem})"
    if sd.get("smoke"):
        tag += " SMOKE"
    d = float(deadband)
    return tag + (f" db{d:g}" if d else " no-db")


def s1_tag_from_ckpt(ckpt_path, *, is_stand_nets_s1: bool) -> str:
    return SNV.s1_tag_from_ckpt(ckpt_path, is_stand_nets_s1=is_stand_nets_s1)


# ===================================================================================== caption text
def caption_lines(*, title: str, hold_i: int, n_holds: int, hold_label: str, hold_kind: str, hold_t: float,
                  hold_len: float, t_s: float, total_s: float, mode: str, stand_tag: str, s1_tag: str,
                  deadband: float, h_cmd: float, h_lim: float, h_meas: float, feet_loaded: int, homed: bool,
                  cam_abs_max: float, cmd_speed: float, cmd_dir_deg: float, cmd_face_deg: float,
                  next_text: str | None, last_event: str | None, heading_deg: float, speed_xy: float,
                  arrows: bool = False, fall_text: str | None = None) -> list[tuple[str, tuple[int, int, int]]]:
    """The per-frame overlay (the predecessor's style: title line, acting line, command line, next line, time line,
    + the height / feet / homed block). Pure; the recorder paints it."""
    if mode not in MODES:
        raise ValueError(f"caption_lines: mode {mode!r} not in {MODES}")
    head = (f"{title}  |  " + (f"hold {hold_i + 1}/{n_holds} '{hold_label}' [{hold_kind}]" if hold_i >= 0
                               else f"PROLOGUE (holds '{hold_label}')")
            + f"  {hold_t:4.1f} / {hold_len:.1f} s", WHITE)
    if mode == MODE_WALK:
        acting = f"MODE: WALK   -- {s1_tag} walking"
    elif mode == MODE_STAND:
        acting = f"MODE: STAND  -- stand net {stand_tag}: q(h) + residual" + (f" (deadband {deadband:g})"
                                                                              if deadband else "")
    else:
        acting = f"MODE: RESTORING -> {CORE.H_SETTLED:.3f} m  -- stand net {stand_tag}, then hand back to {s1_tag}"
    lines = [head, (acting, MODE_COLOR[mode])]
    lines.append((f"HEIGHT  cmd {h_cmd:.3f} m   limited {h_lim:.3f} m   measured {h_meas:.3f} m   "
                  f"(range {CORE.H_MIN:.3f}-{CORE.H_MAX:.3f}, ramp {CORE.HEIGHT_RATE_M_S:g} m/s)", YELLOW))
    lines.append((f"FEET loaded {int(feet_loaded)}/6   HOMED {'yes' if homed else 'NO '} "
                  f"(max|cam| {cam_abs_max:.2f} rad)", GREEN if homed and int(feet_loaded) >= 4 else CYAN))
    lines.append((f"CMD  speed {cmd_speed:.2f} m/s   dir {cmd_dir_deg:+.0f} deg   face {cmd_face_deg:+.0f} deg "
                  "(world frame)", YELLOW))
    if next_text:
        lines.append((next_text, CYAN))
    lines.append((f"t = {t_s:6.2f} / {total_s:.1f} s    heading {heading_deg:+6.1f} deg    |v| {speed_xy:.2f} m/s",
                  WHITE))
    if last_event:
        lines.append((last_event, RED if "TIMED OUT" in last_event else CYAN))
    if arrows:
        lines.append((ARROW_LEGEND, CYAN))
    if fall_text:
        lines.append((fall_text, RED))
    return lines


def next_hold_text(i: int, labels, kinds, heights, starts, dt: float, total_s: float) -> str:
    """``NEXT '<label>' (stand 1.300 m | walk) at t=..`` or the last-hold line."""
    nxt = i + 1
    if nxt >= len(labels):
        return f"LAST hold -- sequence ends at t={total_s:.1f} s"
    what = f"stand {heights[nxt]:.3f} m" if kinds[nxt] == KIND_STAND else "walk (S1)"
    return f"NEXT  '{labels[nxt]}' ({what}) at t={starts[nxt] * dt:.1f} s"
