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

THE TOUR EXTENSION (2026-10-15; still VIDEO ONLY). Four additions, each a pure piece here, used by the recorder:

* STAND RULE (:data:`STAND_RULES`, the predecessor video's ``--stand-rule``, its functions reused by path): ``schedule``
  (default) = a hold is STAND iff speed 0 and its face equals the previous hold's (hold 0: the spawn pin), the prologue
  takes hold 0's flag (``sn_video_core.hold_stand_flags`` / ``step_stand_mask``); every row the video accepts also
  labels exactly those holds ``"stand"`` (:func:`tour_row_reasons`), so the rule IS the explicit stand-command keying.
  ``slot`` = the eval's latched stand slot (``P2RMode.rederive_on_reset``), which keeps TURN on every stand after a
  facing change until a reset (``sn_video_core.eval_slot_stand_flags`` predicts it). :func:`routed_slot` picks the
  (N,) slot handed to :func:`height_step` (i.e. to ``HeightController.step``), so a restore engages on a
  schedule-keyed stand exit; the controller composition is unchanged.
* TERRAIN (:data:`TERRAINS`): ``flat`` = the reference stamp's env, unchanged; ``field_medium`` = the reference env +
  the env block of ``field_turn__omni`` / ``field_suite__omni`` (``2026-10-07_omni_phase2/scenarios_p2.yaml``, read
  from the file, the two blocks checked equal), written before Kit (:func:`terrain_env`). WHERE on the field: the
  generator's tiles are 10 x 10 m (rows along +x, columns along +y, a flat border around), each with its own random
  difficulty in the band; env 0 spawns at the centre of a random row of column 0. :func:`commanded_extent` /
  :func:`field_tiles_needed` size the schedule's commanded path (+ :data:`FIELD_PATH_MARGIN_M`) in tiles and
  :func:`field_spawn_row_cap` gives the ``max_init_terrain_level`` that keeps that many field rows ahead of the spawn
  (the recorder sets it on the field cfg only); :func:`field_path_reasons` re-checks the path from the realised spawn
  (fail-closed); :func:`tile_of` / :func:`tile_visits` record the tile under the robot per hold.
* TOUR ROWS (:data:`VIDEO_BLOCK`): rows with turns and a stand first (the table get-up) cannot pass the launcher's
  one-facing / walk-first rules, so they live in their own manifest block and :func:`video_height_row` validates them
  with :func:`tour_row_reasons` = ``sh_eval_common.height_row_reasons`` adapted (module function docstring); a row
  under ``stand_height.rows`` still goes through ``sh_eval_common.height_row`` (:func:`load_height_row`).
* TABLE START (``--table-start TOP[:POSE]``): :func:`parse_table_start` / :func:`table_start_reasons` (pre-Kit),
  :func:`table_start_reset` (the K3 table test's controller reset: limiter at clamp(h_meas)), :class:`FallTracker` /
  :func:`table_fall_read` (falls RECORDED, never reset: tilt > the crab_failure limit any time; hip contact > its
  threshold only once the chassis' lowest corner has risen >= 0.10 m above its spawn -- ``sh_table_core
  .hip_contact_armed``'s rise latch). Geometry, poses, the spawn event and the park position come from
  ``sh_table_core`` / ``sh_starts`` (loaded lazily, torch-free at import).
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
KIND_WALK, KIND_STAND, KIND_TURN = "walk", "stand", "turn"
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


#: the predecessor's hold kinds -> the video's (a move / move_turn is S1 walking)
_KIND_OF = {"stand": KIND_STAND, "turn": KIND_TURN, "move": KIND_WALK, "move_turn": KIND_WALK}


def hold_kinds(holds, spawn_face: float | None = None) -> list[str]:
    """``walk`` (speed > 0) / ``stand`` (speed 0 at the previous hold's face) / ``turn`` (speed 0, a facing change)
    per hold: the predecessor's ``sn_video_core.hold_kinds`` (hold 0 against ``spawn_face``; None = its own face, i.e.
    ``spawn_yaw: first_face``). ``kind == stand`` is exactly the schedule stand rule (:func:`schedule_stand_flags`). A
    one-facing row (every launcher height row) has no turn holds: walk iff speed > 0, as before."""
    return [_KIND_OF[k] for k in SNV.hold_kinds(holds, spawn_face)]


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
                  arrows: bool = False, fall_text: str | None = None,
                  extra_lines=None) -> list[tuple[str, tuple[int, int, int]]]:
    """The per-frame overlay (the predecessor's style: title line, acting line, command line, next line, time line,
    + the height / feet / homed block). ``extra_lines`` ((text, colour) pairs: the tour's setup / table lines) go
    after the time line. Pure; the recorder paints it."""
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
    for text, color in (extra_lines or ()):
        lines.append((str(text), tuple(color)))
    if last_event:
        lines.append((last_event, RED if "TIMED OUT" in last_event else CYAN))
    if arrows:
        lines.append((ARROW_LEGEND, CYAN))
    if fall_text:
        lines.append((fall_text, RED))
    return lines


def next_hold_text(i: int, labels, kinds, heights, starts, dt: float, total_s: float) -> str:
    """``NEXT '<label>' (stand 1.300 m | walk (S1) | turn (S1)) at t=..`` or the last-hold line."""
    nxt = i + 1
    if nxt >= len(labels):
        return f"LAST hold -- sequence ends at t={total_s:.1f} s"
    what = (f"stand {heights[nxt]:.3f} m" if kinds[nxt] == KIND_STAND
            else ("turn (S1)" if kinds[nxt] == KIND_TURN else "walk (S1)"))
    return f"NEXT  '{labels[nxt]}' ({what}) at t={starts[nxt] * dt:.1f} s"


# ===================================================================================== lazy campaign modules
_LAZY: dict = {}


def table_core():
    """``sh_table_core`` (the K3 table test's pure pieces: surfaces, poses, ``plan_envs``, ``event_params``, the
    ``sh_start`` reset event and its ``RUNTIME`` probe hook, ``hip_contact_armed``, the geometry constants), ONE lazily
    loaded instance (torch- / numpy-free at import)."""
    if "tc" not in _LAZY:
        _LAZY["tc"] = _load_by_path(HERE / "sh_table_core.py", "_shv_table_core")
    return _LAZY["tc"]


def starts_module():
    """``sh_starts`` (the trainer's env composition: the table park position / pose write), lazily (torch-free at
    import)."""
    if "st" not in _LAZY:
        _LAZY["st"] = _load_by_path(HERE / "sh_starts.py", "_shv_starts")
    return _LAZY["st"]


def table_test_module():
    """``sh_table_test`` (its ``crab_failure_params``: the tilt / hip-contact clause read before it is disabled),
    lazily (its module-level loads are torch-free)."""
    if "stt" not in _LAZY:
        _LAZY["stt"] = _load_by_path(HERE / "sh_table_test.py", "_shv_table_test")
    return _LAZY["stt"]


# ===================================================================================== the stand rule
#: ``--stand-rule``: the predecessor video's two rules (``sn_video_core.STAND_RULES``), default ``schedule``
STAND_RULES = tuple(SNV.STAND_RULES)
DEFAULT_STAND_RULE = "schedule"
STAND_RULE_TEXT = {
    "schedule": SNV.STAND_RULE_SCHEDULE + " -- on every row the video accepts these are exactly the 'stand'-labelled "
                                          "holds (the explicit stand command; tour_row_reasons / height_row_reasons)",
    "slot": SNV.STAND_RULE_SLOT,
}
if set(STAND_RULES) != {"schedule", "slot"}:  # pragma: no cover
    raise RuntimeError(f"sh_video_core: the predecessor's stand rules changed: {STAND_RULES}")


def schedule_stand_flags(holds, spawn_face: float | None = None) -> list[bool]:
    """The schedule rule per hold (``sn_video_core.hold_stand_flags``, the predecessor's function)."""
    return list(SNV.hold_stand_flags(holds, spawn_face))


def step_stand_mask(segment_id, flags):
    """(T,) bool per step: ``flags[segment_id[t]]``, the prologue taking hold 0's (``sn_video_core.step_stand_mask``)."""
    return SNV.step_stand_mask(segment_id, flags)


def eval_slot_stand_flags(holds, spawn_face: float | None = None) -> list[bool]:
    """The eval's latched-slot verdict per hold, predicted for a fall-free run (``sn_video_core``)."""
    return list(SNV.eval_slot_stand_flags(holds, spawn_face))


def routed_slot(rule: str, *, schedule_stand, eval_slot):
    """The (N,) bool stand slot handed to :func:`height_step` (``HeightController.step``): ``schedule`` -> the
    schedule's stand flag of this step (a bool / (N,) bool, broadcast to ``eval_slot``'s shape and device); ``slot`` ->
    ``eval_slot`` itself (the eval's latched slot, the launcher's routing)."""
    import torch  # noqa: PLC0415

    if rule == "slot":
        return eval_slot
    if rule != "schedule":
        raise ValueError(f"routed_slot: rule {rule!r} not in {STAND_RULES}")
    if torch.is_tensor(schedule_stand):
        s = schedule_stand.to(device=eval_slot.device, dtype=torch.bool)
    elif hasattr(schedule_stand, "__len__"):
        s = torch.tensor([bool(x) for x in schedule_stand], dtype=torch.bool, device=eval_slot.device)
    else:
        s = torch.tensor(bool(schedule_stand), dtype=torch.bool, device=eval_slot.device)
    return s.expand(eval_slot.shape).clone() if s.dim() == 0 else s.reshape(eval_slot.shape).clone()


# ===================================================================================== the terrain
TERRAINS = ("flat", "field_medium")
DEFAULT_TERRAIN = "flat"
FIELD_MANIFEST = EXPERIMENTS / "2026-10-07_omni_phase2" / "scenarios_p2.yaml"
#: the phase-2 rows whose (identical) env block is the medium-rough field (DIFF 0.20:0.70, the training difficulty)
FIELD_MEDIUM_ROWS = ("field_turn__omni", "field_suite__omni")


def field_env_block(manifest=FIELD_MANIFEST, rows=FIELD_MEDIUM_ROWS) -> dict:
    """``{"env": {KRABBY_*: str}, "source": {...}}``: the env block of ``rows`` READ from ``manifest`` (every row must
    carry one and they must be identical; values as the strings the harness writes)."""
    path = Path(manifest)
    data = EC.read_yaml(path)
    by_id = {r.get("id"): r for r in (data.get("scenarios") or []) if isinstance(r, dict)}
    blocks = {}
    for rid in rows:
        if rid not in by_id:
            raise ValueError(f"{path}: no row {rid!r}")
        env = by_id[rid].get("env")
        if not isinstance(env, dict) or not env:
            raise ValueError(f"{path}: row {rid!r} carries no env: block")
        blocks[rid] = {str(k): str(v) for k, v in env.items()}
    first = blocks[rows[0]]
    for rid, b in blocks.items():
        if b != first:
            raise ValueError(f"{path}: env blocks differ: {rows[0]} {first} vs {rid} {b}")
    bad = sorted(k for k in first if not k.startswith("KRABBY_"))
    if bad:
        raise ValueError(f"{path}: env block keys {bad} are not KRABBY_* knobs")
    return {"env": dict(first), "source": {"manifest": str(path.resolve()), "rows": list(rows),
                                           "manifest_sha256": EC.sha256_file(path)}}


def terrain_env(reference_env: dict, terrain: str) -> tuple[dict, dict]:
    """``(env to apply before Kit, record)``. ``flat``: the reference env itself (a copy, equal key for key: today's
    path); ``field_medium``: the reference env updated with :func:`field_env_block` -- the ONE deliberate deviation from
    the reference stamp's env, recorded as ``deviation`` ``{var: {reference, video}}``."""
    if terrain not in TERRAINS:
        raise ValueError(f"terrain {terrain!r} not in {TERRAINS}")
    ref = {str(k): str(v) for k, v in reference_env.items()}
    if terrain == "flat":
        return dict(ref), {"terrain": terrain, "env_block": None, "source": None, "deviation": {},
                           "note": "the reference stamp's env, unchanged"}
    blk = field_env_block()
    env = dict(ref)
    env.update(blk["env"])
    dev = {k: {"reference": ref.get(k), "video": v} for k, v in blk["env"].items() if ref.get(k) != v}
    return env, {"terrain": terrain, "env_block": dict(blk["env"]), "source": blk["source"], "deviation": dev,
                 "note": "the reference stamp's env + the field_turn__omni / field_suite__omni env block, written "
                         "before Kit (a deliberate deviation for the video; every other reference check unchanged)"}


# ----------------------------------------------------------------------------- where on the field (the tile grid)
#: overshoot / drift allowance added to the schedule's commanded path when sizing the field it needs (m)
FIELD_PATH_MARGIN_M = 2.0
#: env 0's spawn column: IsaacLab ``TerrainImporter._compute_env_origins_curriculum`` gives env ``i`` the column
#: ``floor(i / (num_envs / num_cols))`` = 0 for the video's one env (the grid's -y edge column; the 20-m border beyond)
FIELD_SPAWN_COL = 0


def field_grid_of(tg) -> dict:
    """The generator's tile grid (a ``TerrainGeneratorCfg`` or a dict): ``size`` (x, y) m, ``num_rows`` (along +x),
    ``num_cols`` (along +y), ``border_width`` m. Read from the env cfg at run time, never restated here."""
    get = (lambda k: tg[k]) if isinstance(tg, dict) else (lambda k: getattr(tg, k))
    sx, sy = (float(v) for v in get("size"))
    return {"size": [sx, sy], "num_rows": int(get("num_rows")), "num_cols": int(get("num_cols")),
            "border_width": float(get("border_width"))}


def commanded_extent(holds, *, prologue_s: float = 0.0) -> dict:
    """The schedule's commanded WORLD path from the spawn: per hold ``speed * (cos dir, sin dir) * hold_s`` (the
    prologue at hold 0's command, as ``compile_schedule`` fills it). An UPPER bound on a fall-free run's travel: the
    stand / restore time of a command hold counts as commanded motion too. Piecewise linear, so its extremes are at the
    hold ends: ``{"ends": [[x, y] per hold end], "x_min", "x_max", "y_min", "y_max"}`` in metres from the spawn."""
    views = [_hold_view(h) for h in holds]
    legs = ([dict(views[0], hold_s=float(prologue_s))] if views and float(prologue_s) > 0 else []) + views
    x = y = 0.0
    xs, ys, ends = [0.0], [0.0], []
    for k, v in enumerate(legs):
        x += v["speed"] * math.cos(v["dir"]) * v["hold_s"]
        y += v["speed"] * math.sin(v["dir"]) * v["hold_s"]
        xs.append(x)
        ys.append(y)
        if k >= len(legs) - len(views):
            ends.append([round(x, 6), round(y, 6)])
    return {"ends": ends, "x_min": min(xs), "x_max": max(xs), "y_min": min(ys), "y_max": max(ys)}


def field_tiles_needed(extent: dict, grid: dict, margin_m: float = FIELD_PATH_MARGIN_M) -> dict:
    """Whole tiles the commanded path (+ ``margin_m``) reaches beyond the spawn tile, from its centre (the env origin):
    ``ahead`` (+x rows), ``behind`` (-x rows), ``left`` (+y columns), ``right`` (-y columns)."""
    sx, sy = grid["size"]

    def n(reach: float, size: float) -> int:
        return max(0, int(math.ceil((float(reach) + float(margin_m) - size / 2.0) / size - 1e-9)))

    return {"ahead": n(extent["x_max"], sx), "behind": n(-extent["x_min"], sx), "left": n(extent["y_max"], sy),
            "right": n(-extent["y_min"], sy), "margin_m": float(margin_m)}


def field_spawn_row_cap(need: dict, grid: dict, spawn_col: int = FIELD_SPAWN_COL) -> int:
    """``max_init_terrain_level`` for the video: the highest spawn row that keeps ``need['ahead']`` field rows ahead of
    the spawn (+x), so the path never reaches the flat border. Raises ``ValueError`` on a path the grid cannot hold from
    env 0's spawn: rows BEHIND (-x; IsaacLab draws the spawn row from 0 up, so a lower bound cannot be pinned), columns
    beyond the grid on either side of ``spawn_col``."""
    out = []
    if need["behind"] > 0:
        out.append(f"the path reaches {need['behind']} row(s) behind (-x) the spawn tile; only rows ahead can be "
                   "pinned")
    if need["right"] > spawn_col:
        out.append(f"the path reaches {need['right']} column(s) to -y of spawn column {spawn_col} (the flat border)")
    if spawn_col + need["left"] > grid["num_cols"] - 1:
        out.append(f"the path reaches {need['left']} column(s) to +y of spawn column {spawn_col}; the grid has "
                   f"{grid['num_cols']}")
    cap = int(grid["num_rows"]) - 1 - int(need["ahead"])
    if cap < 0:
        out.append(f"the path reaches {need['ahead']} row(s) ahead (+x); the grid has {grid['num_rows']}")
    if out:
        raise ValueError("the schedule's commanded path does not fit the field grid " + str(grid) + ": "
                         + "; ".join(out))
    return cap


def field_path_box(x0: float, y0: float, extent: dict, margin_m: float = FIELD_PATH_MARGIN_M) -> list[float]:
    """The world rectangle ``[x_lo, x_hi, y_lo, y_hi]`` the commanded path (+ ``margin_m``) covers from a spawn at
    (``x0``, ``y0``)."""
    m = float(margin_m)
    return [float(x0) + extent["x_min"] - m, float(x0) + extent["x_max"] + m,
            float(y0) + extent["y_min"] - m, float(y0) + extent["y_max"] + m]


def field_path_reasons(x0: float, y0: float, extent: dict, grid: dict,
                       margin_m: float = FIELD_PATH_MARGIN_M) -> list[str]:
    """The REALISED spawn (post-Kit, fail-closed) against the path: every corner of :func:`field_path_box` must lie on a
    grid tile, not the border (the grid is a rectangle, so the corners decide). Empty = the path stays on the field."""
    x_lo, x_hi, y_lo, y_hi = field_path_box(x0, y0, extent, margin_m)
    off = [[round(x, 3), round(y, 3)] for x in (x_lo, x_hi) for y in (y_lo, y_hi) if tile_of(x, y, grid) is None]
    if not off:
        return []
    return [f"the commanded path + {margin_m} m from the spawn ({x0:.3f}, {y0:.3f}) reaches the border at {off} "
            f"(grid {grid})"]


def field_path_tiles(x0: float, y0: float, extent: dict, grid: dict,
                     margin_m: float = FIELD_PATH_MARGIN_M) -> list[tuple[int, int]]:
    """The grid tiles :func:`field_path_box` covers (row-major; border parts dropped)."""
    x_lo, x_hi, y_lo, y_hi = field_path_box(x0, y0, extent, margin_m)
    sx, sy = grid["size"]
    r0 = math.floor((x_lo + grid["num_rows"] * sx / 2.0) / sx)
    r1 = math.floor((x_hi + grid["num_rows"] * sx / 2.0) / sx)
    c0 = math.floor((y_lo + grid["num_cols"] * sy / 2.0) / sy)
    c1 = math.floor((y_hi + grid["num_cols"] * sy / 2.0) / sy)
    return [(r, c) for r in range(max(0, r0), min(grid["num_rows"] - 1, r1) + 1)
            for c in range(max(0, c0), min(grid["num_cols"] - 1, c1) + 1)]


def tile_of(x: float, y: float, grid: dict):
    """World (x, y) -> the generator tile ``(row, col)`` under it, or None on the border. IsaacLab's
    ``TerrainGenerator`` puts tile (r, c) at ``((r + 0.5) sx, (c + 0.5) sy)`` and then shifts the whole grid by
    ``-(num_rows sx / 2, num_cols sy / 2)``."""
    sx, sy = grid["size"]
    r = math.floor((float(x) + grid["num_rows"] * sx / 2.0) / sx)
    c = math.floor((float(y) + grid["num_cols"] * sy / 2.0) / sy)
    if 0 <= r < grid["num_rows"] and 0 <= c < grid["num_cols"]:
        return int(r), int(c)
    return None


def tile_key(tile) -> str:
    return "border" if tile is None else f"{tile[0]},{tile[1]}"


def tile_visits(xs, ys, grid: dict) -> dict:
    """Steps per tile over a trajectory: ``{"tiles": {"r,c": steps}, "border_steps": n, "steps": n}``."""
    counts = collections.Counter(tile_key(tile_of(x, y, grid)) for x, y in zip(xs, ys))
    border = int(counts.pop("border", 0))
    return {"tiles": dict(sorted(counts.items())), "border_steps": border, "steps": int(sum(counts.values()) + border)}


def tile_caption_text(tile, names: dict | None) -> str:
    """The setup line's live terrain piece: the tile under the robot and its sub-terrain (or the border)."""
    if tile is None:
        return "on the BORDER (off the field grid)"
    name = (names or {}).get(tile_key(tile))
    return f"tile {tile[0]},{tile[1]}" + (f" {name}" if name else "")


# ===================================================================================== tour rows (the video block)
#: the manifest block of rows the launcher's rules cannot hold (turns, a table-start first stand); VIDEO ONLY
VIDEO_BLOCK = "stand_height_video"
VIDEO_BLOCK_VERSION = 1
VIDEO_ROW_KEYS = ("hold_heights", "rules", "notes")
VIDEO_RULES = "tour"


def _hold_view(h) -> dict:
    if isinstance(h, dict):
        return {"label": str(h["label"]), "speed": float(h["speed"]), "dir": float(h.get("dir", 0.0)),
                "face": float(h["face"]), "hold_s": float(h["hold_s"])}
    sp, di, fa = (float(x) for x in h.as_world())
    return {"label": str(h.label), "speed": sp, "dir": di, "face": fa, "hold_s": float(h.hold_s)}


def tour_row_reasons(holds, hold_heights: list, *, first_level: float | None = None,
                     first_need_s: float | None = None) -> list[str]:
    """``sh_eval_common.height_row_reasons`` ADAPTED for the video's tour rows. Empty = valid. Same: one height per
    hold, a float on stand holds, None elsewhere; heights in the frozen [h_min, h_max]; ``"stand"`` labels; a stand
    entered at level ``l`` lasts >= ``|h - l| / 0.08 + 1 + 2`` s; a hold after a stand at ``h`` lasts >= ``|h - 1.064|
    / 0.08 + 1.5 + 2`` s; the last hold is not a stand (every stand handed back inside the row). Adapted:

    * TURNS: a speed-0 hold that changes the facing is a TURN (S1's; the schedule stand rule), never labelled
      ``"stand"``, without a height, and timed as a walk after a stand; a speed-0 hold at the previous face is a stand
      and MUST be labelled ``"stand"`` (so the schedule rule and the explicit stand label agree on every hold);
    * FIRST HOLD: may be a stand (the table get-up / a stand start), entered at ``first_level`` (the table start's
      spawn height; None = 1.064, a walk-handover start whose limiter starts at the command) and, when
      ``first_need_s`` is given, lasting >= it (the caller's table need: rod rise + ramp + 1 + 2 s)."""
    rate = CORE.HEIGHT_RATE_M_S
    hv = [_hold_view(h) for h in holds]
    if not hv:
        return ["empty schedule"]
    if len(hold_heights) != len(hv):
        return [f"hold_heights has {len(hold_heights)} entries for {len(hv)} holds (one per hold, in order)"]
    out = []
    for i, h in enumerate(hv):
        if not (math.isfinite(h["speed"]) and h["speed"] >= 0.0):
            out.append(f"hold {i} {h['label']!r}: speed {h['speed']} must be finite and >= 0")
    if out:
        return out
    kinds = hold_kinds(hv)
    if kinds[-1] == KIND_STAND:
        out.append(f"the last hold {hv[-1]['label']!r} is a stand (every stand is handed back inside the row)")
    level = CORE.H_SETTLED if first_level is None else float(first_level)
    prev_stand = False
    for i, (h, hh, k) in enumerate(zip(hv, hold_heights, kinds)):
        lab, hs = h["label"], h["hold_s"]
        if k == KIND_STAND:
            if lab != EC.STAND_LABEL:
                out.append(f"hold {i} {lab!r}: a speed-0 hold at the previous face is a stand and must be labelled "
                           f"{EC.STAND_LABEL!r} (the schedule rule and the stand label must agree)")
            if hh is None:
                out.append(f"hold {i} {lab!r}: a stand hold without a height")
                continue
            if not CORE.H_MIN - 1e-9 <= float(hh) <= CORE.H_MAX + 1e-9:
                out.append(f"hold {i} {lab!r}: height {hh} m is outside the frozen range [{CORE.H_MIN}, {CORE.H_MAX}]")
            need = abs(float(hh) - level) / rate + EC.ARRIVAL_SETTLE_S + EC.K1_WINDOW_S
            if hs + 1e-9 < need:
                out.append(f"hold {i} {lab!r}: hold_s {hs} < |{hh} - {level:.3f}| / {rate} + {EC.ARRIVAL_SETTLE_S} + "
                           f"{EC.K1_WINDOW_S} = {need:.2f} s (ramp, arrival + 1 s, the 2-s window)")
            if i == 0 and first_need_s is not None and hs + 1e-9 < float(first_need_s):
                out.append(f"hold 0 {lab!r}: hold_s {hs} < the start's need {float(first_need_s):.2f} s")
            level, prev_stand = float(hh), True
        else:
            what = "turn" if k == KIND_TURN else "walk"
            if lab == EC.STAND_LABEL:
                out.append(f"hold {i}: a {what} hold (speed {h['speed']}, face {h['face']}) labelled "
                           f"{EC.STAND_LABEL!r}")
            if hh is not None:
                out.append(f"hold {i} {lab!r}: a {what} hold carries a height {hh}")
            if prev_stand:
                need = abs(level - CORE.H_SETTLED) / rate + EC.K4_ONSET_SLACK_S + EC.WALK_AFTER_RESTORE_S
                if hs + 1e-9 < need:
                    out.append(f"hold {i} {lab!r}: hold_s {hs} < |{level} - 1.064| / {rate} + {EC.K4_ONSET_SLACK_S} "
                               f"+ {EC.WALK_AFTER_RESTORE_S} = {need:.2f} s (restore ramp, onset slack, 2 s of "
                               "motion)")
            level, prev_stand = CORE.H_SETTLED, False
    return out


def video_row_names(manifest) -> list[str]:
    """Row ids under the manifest's :data:`VIDEO_BLOCK` (empty when it has none)."""
    if not manifest or not Path(manifest).is_file():
        return []
    blk = EC.read_yaml(manifest).get(VIDEO_BLOCK)
    rows = blk.get("rows") if isinstance(blk, dict) else None
    return sorted(rows) if isinstance(rows, dict) else []


def video_height_row(manifest, scenario: str, *, first_level: float | None = None,
                     first_need_s: float | None = None) -> dict:
    """A tour row of :data:`VIDEO_BLOCK`, validated by :func:`tour_row_reasons` (raises ValueError with every reason);
    the same dict as ``sh_eval_common.height_row`` + ``rules`` / ``block``."""
    path = Path(manifest)
    data = EC.read_yaml(path)
    blk = data.get(VIDEO_BLOCK)
    if not isinstance(blk, dict) or blk.get("version") != VIDEO_BLOCK_VERSION:
        raise ValueError(f"{path}: no {VIDEO_BLOCK!r} block of version {VIDEO_BLOCK_VERSION}")
    rows = blk.get("rows")
    if not isinstance(rows, dict) or scenario not in rows or not isinstance(rows[scenario], dict):
        raise ValueError(f"{path}: no tour row {scenario!r} under {VIDEO_BLOCK}.rows")
    row = rows[scenario]
    extra = sorted(set(row) - set(VIDEO_ROW_KEYS))
    if extra:
        raise ValueError(f"{path}: {VIDEO_BLOCK}.rows.{scenario} carries unknown keys {extra} (only {VIDEO_ROW_KEYS})")
    if row.get("rules") != VIDEO_RULES:
        raise ValueError(f"{path}: {VIDEO_BLOCK}.rows.{scenario}.rules is {row.get('rules')!r}, want {VIDEO_RULES!r}")
    hraw = row.get("hold_heights")
    if not isinstance(hraw, list) or not hraw:
        raise ValueError(f"{path}: {VIDEO_BLOCK}.rows.{scenario}.hold_heights must be a non-empty list")
    heights = []
    for i, v in enumerate(hraw):
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v))):
            raise ValueError(f"{path}: hold_heights[{i}] = {v!r} is neither null nor a finite number")
        heights.append(None if v is None else float(v))
    scen = [s for s in (data.get("scenarios") or []) if isinstance(s, dict) and s.get("id") == scenario]
    if len(scen) != 1:
        raise ValueError(f"{path}: {len(scen)} scenarios with id {scenario!r} (want exactly 1)")
    s = scen[0]
    if str(s.get("kind", "schedule")) != "schedule" or s.get("rotate_holds_by_env"):
        raise ValueError(f"{path}: {scenario} must be a plain schedule row without rotate_holds_by_env")
    if s.get("env"):
        raise ValueError(f"{path}: {scenario} carries an env: block (the recorder's --terrain sets the terrain)")
    holds = []
    for i, hd in enumerate(s.get("schedule") or []):
        if not isinstance(hd, dict) or any(not isinstance(hd.get(k), (int, float)) or isinstance(hd.get(k), bool)
                                           for k in ("speed", "dir", "face", "hold_s")):
            raise ValueError(f"{path}: {scenario} hold {i} must be a world-frame hold with speed / dir / face / hold_s")
        holds.append({"label": str(hd.get("label") or f"hold{i}"), "speed": float(hd["speed"]),
                      "dir": float(hd["dir"]), "face": float(hd["face"]), "hold_s": float(hd["hold_s"])})
    reasons = tour_row_reasons(holds, heights, first_level=first_level, first_need_s=first_need_s)
    if reasons:
        raise ValueError(f"{path}: tour row {scenario} refused:\n  " + "\n  ".join(reasons))
    return {"manifest": str(path.resolve()), "manifest_sha256": EC.sha256_file(path), "scenario": scenario,
            "labels": [h["label"] for h in holds], "hold_heights": list(heights), "gates": [],
            "rules": VIDEO_RULES, "block": VIDEO_BLOCK}


def load_height_row(manifest, scenario: str, **kw) -> dict:
    """The recorder's row loader: a ``stand_height.rows`` row -> ``sh_eval_common.height_row`` (the launcher's path,
    unchanged); a :data:`VIDEO_BLOCK` row -> :func:`video_height_row` (``kw`` = its table-start arguments)."""
    if EC.has_height_row(manifest, scenario):
        return EC.height_row(manifest, scenario)
    if scenario in video_row_names(manifest):
        return video_height_row(manifest, scenario, **kw)
    raise ValueError(f"{manifest}: {scenario!r} is in neither {EC.MANIFEST_BLOCK}.rows nor {VIDEO_BLOCK}.rows")


# ===================================================================================== the table start
DEFAULT_TABLE_POSE = "knee_folded"


def parse_table_start(text) -> dict | None:
    """``--table-start TOP[:POSE]`` (None / '' = off) -> ``{top, pose, surface, cell, spawn_h_plan, cmd_min}``. TOP must
    be one of the K3 table surfaces (``sh_table_core.SURFACE_TOP``: 0.4572 / 0.635 / 0.6985 / 0.762 m), POSE a fixed
    retracted pose (``sh_table_core.FIXED_POSES``; default knee_folded). Pure arithmetic: no torch."""
    if text is None or str(text).strip() == "":
        return None
    TC = table_core()
    parts = str(text).strip().split(":")
    if len(parts) > 2:
        raise ValueError(f"--table-start {text!r}: want TOP[:POSE]")
    try:
        top = float(parts[0])
    except ValueError as exc:
        raise ValueError(f"--table-start {text!r}: TOP {parts[0]!r} is not a number") from exc
    pose = parts[1] if len(parts) == 2 and parts[1] else DEFAULT_TABLE_POSE
    if pose not in TC.FIXED_POSES:
        raise ValueError(f"--table-start {text!r}: pose {pose!r} not in {TC.FIXED_POSES} (a random fold needs a draw)")
    surface = TC.surface_id(top)
    if surface not in TC.SURFACE_TOP or TC.SURFACE_TOP[surface] is None:
        tops = sorted(t for t in TC.SURFACE_TOP.values() if t is not None)
        raise ValueError(f"--table-start {text!r}: top {top} m is not a K3 table surface {tops}")
    top = float(TC.SURFACE_TOP[surface])
    cell = f"{surface}{TC.CELL_SEP}{pose}"
    TC.parse_cell(cell)
    spawn_h = float(TC.table_spawn_root_z(top))
    return {"top": top, "pose": pose, "surface": surface, "cell": cell, "spawn_h_plan": spawn_h,
            "h_init_plan": min(max(spawn_h, CORE.H_MIN), CORE.H_MAX), "cmd_min": top + float(TC.CMD_MARGIN_EVAL_M),
            "cmd_margin_m": float(TC.CMD_MARGIN_EVAL_M)}


def table_start_reasons(table: dict, kinds, hold_heights) -> list[str]:
    """The table-start refusals on a schedule: the first hold must be a stand (the get-up) whose height >= top + 0.31
    (``sh_table_core.CMD_MARGIN_EVAL_M``, the K3 eval margin). Empty = valid."""
    out = []
    if not kinds or kinds[0] != KIND_STAND:
        out.append(f"--table-start: the first schedule hold must be a stand (the get-up off the table), got "
                   f"{kinds[0] if kinds else 'no hold'!r}")
        return out
    h0 = hold_heights[0]
    if h0 is None or float(h0) + 1e-9 < float(table["cmd_min"]):
        out.append(f"--table-start: the first stand's height {h0} m < top {table['top']} + {table['cmd_margin_m']} = "
                   f"{table['cmd_min']:.4f} m (the K3 table-command margin)")
    return out


def table_need_s(table: dict, h_cmd: float, h_init: float | None = None) -> dict:
    """The first stand's need on a table start (``sh_table_core.env_need_s``: the rods' rise from the pose + the
    limiter ramp from ``h_init`` (default the planned clamp(spawn height)), plus K2's 1 s and the 2-s K1 window: the
    scenarios_sh stand rule with the table's level and rod rise). Loads ``sh_kin`` (torch): CPU tests / post-Kit."""
    TC = table_core()
    p = TC.fixed_pose(table["pose"])
    hi = float(table["h_init_plan"] if h_init is None else min(max(float(h_init), CORE.H_MIN), CORE.H_MAX))
    n = TC.env_need_s(float(p["hip_rad"]), float(p["knee_left_rad"]), hi, float(h_cmd))
    return {**n, "h_init": hi, "rule_s": n["reach_s"] + EC.ARRIVAL_SETTLE_S + EC.K1_WINDOW_S}


def table_start_reset(ctl, reset_rows, *, h_cmd, h_meas):
    """The K3 table test's controller start: ``ctl.reset(rows, "table", h_cmd=h_cmd, h_meas=h_meas)`` (the limiter at
    clamp(h_meas); the restore cleared) on ``reset_rows`` -- every episode start of a table-start video (the env
    respawns on the table). Returns ``(rows left for height_step's walk reset, cut)``: no rows (the table reset replaces
    the walk handover), and the restores the reset cut."""
    import torch  # noqa: PLC0415

    rows = torch.as_tensor(reset_rows, dtype=torch.bool).reshape(-1)
    cut = ctl.reset(rows, "table", h_cmd=h_cmd, h_meas=h_meas)
    return torch.zeros_like(rows), cut


def table_stood_now(root_rise: float, feet_loaded: int, h_meas: float, h_cmd: float) -> bool:
    """K3's three STOOD clauses on ONE step (the caption's 'stood' instant; K3 itself reads a 3-s window): root risen
    >= 0.10 m above its spawn, >= 4 feet loaded, |h_meas - h_cmd| <= 0.04 m."""
    TC = table_core()
    return bool(float(root_rise) >= TC.STOOD_RISE_M - 1e-9 and int(feet_loaded) >= TC.STOOD_MIN_FEET
                and abs(float(h_meas) - float(h_cmd)) <= TC.STOOD_H_TOL_M + 1e-9)


FALL_RULE = ("falls RECORDED, never reset (crab_failure disabled): tilt = arccos(cos tilt) > the env's crab_failure "
             "limit_angle on any step; hip contact (max over the crab_failure hip bodies of the contact-history max "
             "norm) > its contact_force_threshold on a step once ARMED, armed from the first step whose chassis lowest "
             "corner has risen >= 0.10 m above its spawn (sh_table_core.hip_contact_armed's rise latch); a fall event = "
             "the first step of a run of fall steps")


def _fall_steps(cos_tilt, hip_force, armed, tilt_max: float, contact_n: float):
    import numpy as np  # noqa: PLC0415

    tilt = np.arccos(np.clip(np.asarray(cos_tilt, dtype=np.float64).reshape(-1), -1.0, 1.0))
    hit = np.asarray(hip_force, dtype=np.float64).reshape(-1) > float(contact_n)
    bad_t = tilt > float(tilt_max)
    bad_h = np.asarray(armed, dtype=bool).reshape(-1) & hit
    return tilt, hit, bad_t, bad_h


def _cause(bad_t: bool, bad_h: bool) -> str:
    return "+".join(c for c, b in (("tilt", bad_t), ("hip_contact", bad_h)) if b)


def table_fall_read(cos_tilt, hip_force, rise, dt: float, *, tilt_max: float, contact_n: float,
                    episode_start=None) -> dict:
    """The video's fall read (:data:`FALL_RULE`) over (T,) post-step series: ``cos_tilt``, ``hip_force`` (N), ``rise``
    (the chassis lowest corner above its spawn, m); ``episode_start`` (T,) bool marks the first step of an episode
    after an env reset (the arming latch restarts there). Returns the events and counts."""
    import numpy as np  # noqa: PLC0415

    TC = table_core()
    r = np.asarray(rise, dtype=np.float64).reshape(-1)
    T = int(r.size)
    starts = np.zeros(T, dtype=bool) if episode_start is None else np.asarray(episode_start, dtype=bool).reshape(-1)
    armed = np.zeros(T, dtype=bool)
    edges = [0] + [int(i) for i in np.nonzero(starts)[0] if i > 0] + [T]
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            armed[a:b] = TC.hip_contact_armed(r[a:b], TC.START_GROUND)
    tilt, hit, bad_t, bad_h = _fall_steps(cos_tilt, hip_force, armed, tilt_max, contact_n)
    fall = bad_t | bad_h
    prev = np.concatenate([[False], fall[:-1]]) & ~starts
    onset = fall & ~prev
    events = [{"step": int(i), "t_s": round((int(i) + 1) * float(dt), 6), "cause": _cause(bool(bad_t[i]),
                                                                                           bool(bad_h[i]))}
              for i in np.nonzero(onset)[0]]
    arm_idx = np.nonzero(armed)[0]
    hf = np.asarray(hip_force, dtype=np.float64).reshape(-1)
    return {"rule": FALL_RULE, "tilt_max_rad": float(tilt_max), "contact_n": float(contact_n),
            "arm_rise_m": float(TC.ARM_RISE_M), "falls": len(events), "events": events,
            "fall_steps": int(fall.sum()), "tilt_fall_steps": int(bad_t.sum()), "hip_fall_steps": int(bad_h.sum()),
            "max_tilt_rad": float(tilt.max()) if T else 0.0,
            "armed_first_s": None if arm_idx.size == 0 else round((int(arm_idx[0]) + 1) * float(dt), 6),
            "hip_contact_pre_arm_steps": int((~armed & hit).sum()),
            "hip_force_pre_arm_max_n": float(hf[~armed].max()) if (~armed).any() else None,
            "hip_force_armed_max_n": float(hf[armed].max()) if armed.any() else None}


def setup_caption_text(terrain_rec: dict, rule: str, tile_text: str | None = None) -> str:
    """The setup line: the terrain (with the field's difficulty / edges; on the field, ``tile_text`` = the tile under
    the robot, :func:`tile_caption_text`) and the stand rule routed on."""
    blk = terrain_rec.get("env_block") or {}
    if terrain_rec.get("terrain") == "flat":
        ter = "flat"
    else:
        ter = (f"{terrain_rec.get('terrain')} (field DIFF {blk.get('KRABBY_FLAT_TERRAIN_DIFF', '?')}, "
               f"{blk.get('KRABBY_FIELD_EDGES', '?')} edges)" + (f" {tile_text}" if tile_text else ""))
    return f"TERRAIN {ter}   |   stand rule: {rule}"


def table_caption_lines(table: dict, *, arrival_s, stood_s, falls: int, last_fall: dict | None,
                        parked_s=None, park: bool = False) -> list[tuple[str, tuple[int, int, int]]]:
    """The table-start caption (one line; a second, red, once a fall is recorded): top / pose, the limiter's arrival
    and the K3 'stood' instant of the get-up, the table's state, the recorded falls (never reset)."""
    arr = "--" if arrival_s is None else f"{arrival_s:.1f} s"
    st = "--" if stood_s is None else f"{stood_s:.1f} s"
    where = f"parked t={parked_s:.1f} s" if parked_s is not None else ("parks after the get-up" if park
                                                                       else "stays")
    lines = [(f"TABLE {table['top']:.3f} m {table['pose']}: arrived {arr}, stood {st}, table {where}; "
              f"falls {int(falls)} (recorded, not reset)", RED if falls else CYAN)]
    if falls:
        lf = last_fall or {}
        lines.append((f"FALL at t={lf.get('t_s', float('nan')):.2f} s: {lf.get('cause', '?')} (K3 clauses; no reset)",
                      RED))
    return lines


class FallTracker:
    """The per-step form of :func:`table_fall_read` (the caption's live count): ``update(...)`` after every env step
    returns the fall event that started on this step (or None). ``events`` must equal the read's (the recorder checks
    it)."""

    def __init__(self, dt: float, *, tilt_max: float, contact_n: float):
        self.dt, self.tilt_max, self.contact_n = float(dt), float(tilt_max), float(contact_n)
        self.armed = False
        self.prev = False
        self.events: list[dict] = []
        self.armed_first_s = None

    def update(self, step: int, cos_tilt: float, hip_force: float, rise: float, episode_start: bool = False):
        TC = table_core()
        if episode_start:
            self.armed, self.prev = False, False
        self.armed = self.armed or float(rise) >= TC.ARM_RISE_M - 1e-9
        if self.armed and self.armed_first_s is None:
            self.armed_first_s = round((int(step) + 1) * self.dt, 6)
        bad_t = math.acos(max(-1.0, min(1.0, float(cos_tilt)))) > self.tilt_max
        bad_h = self.armed and float(hip_force) > self.contact_n
        fall = bad_t or bad_h
        ev = None
        if fall and not self.prev:
            ev = {"step": int(step), "t_s": round((int(step) + 1) * self.dt, 6), "cause": _cause(bad_t, bad_h)}
            self.events.append(ev)
        self.prev = fall
        return ev
