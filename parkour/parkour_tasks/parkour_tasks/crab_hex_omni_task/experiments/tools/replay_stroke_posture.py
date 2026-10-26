#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Offline replay, readout and gate for the stroke/posture plan (plan step 6). CPU ONLY.

Plan: the campaign record ``experiments/2026-09-26_stroke_posture/`` (CHARTER.md, REPLAY_GATE.md, CLOSE.md);
the user's plan draft (steps 6-8, 14) is ``~/.claude/plans/let-s-examine-the-stride-gleaming-dream.md``
(outside the repo). Never imports
``isaaclab``, never starts Kit, never touches a GPU, ``candidates.yaml`` or git.

WHAT IT PRICES. The step-2 math, ``crab_hex_omni_task/mdp/crab_hex_omni_stroke_math.py``, is loaded BY
PATH from the checkout this file lives in (:data:`STROKE_MATH_PY`), so the replay prices exactly the
arithmetic that trains: :func:`foot_stroke_step` (the per-foot loaded stance-stroke income that
replaces ``reward_stride_length``) and :func:`body_centring_step` (the posture arm's bonus). Every
episode of a stamp is replayed at once as one batch of "envs" (float32 torch, like training).

SUBCOMMANDS

``readout <run_dir> [--vs <run_dir>] [--seed-a P --seed-b P] [--validate] [--json P]``
    Per category and per hand, the median over episodes of each episode's own-hold value (the MEAN
    for ``stride_along``), over the episode's steady own-hold window. With ``--vs``: an
    env-clustered PAIRED bootstrap (B = 2000, paired on ``env_index``, clustered on the within-wave
    env as ``resolution.cluster_draw`` does) of run minus vs-run, with its 95 % CI and MDE (the
    centred 95 % half-width, ``resolution.half_width``); also SA0 decision 7's UNISON flags
    (:func:`unison_flags`) on ARM ``run_dir`` vs BASELINE ``--vs``, over :data:`UNISON_CATEGORIES` --
    tripod coherence and roll-RMS collapse, neither of which the stroke income can see. ``--seed-a``
    / ``--seed-b`` (a same-stage seed pair, e.g. B's two seeds) add the seed band. ``--validate``
    checks the step-6 reproduction targets (:data:`VALIDATION_OMNI` on an omni stamp,
    :data:`VALIDATION_SIDE` on the sideways head) and exits 1 naming every hard target that missed.
``income <run_dir> --w-stroke W --w-centring W [--stroke-cat-w SPEC] [--params k=v ...] [--econ-env-yaml P]``
    Per category: term income/s at w = 1, tracking income/s, the DELIVERED (post-clip) pay in the
    manager's units, the absorbed share, the clip-bind fraction shift, and the stances each gate
    rejects or trims. ``--stroke-cat-w`` (the per-category stroke plan, 2026-09-27) is
    ``KRABBY_OMNI_STROKE_CAT_W``'s own spec, parsed by the knob's own parser
    (:data:`SK` ``.parse_stroke_cat_w``): nominal, delivered, absorbed, clip-bind and ``stroke_dose``
    are then priced at the effective weight ``w_stroke * m_c`` of each step's category
    (:func:`scale_rows`); ``S/s@1`` (``stroke_per_s``) stays at w = 1, unscaled.
``gate --config <json> --out <dir>``
    Plan step 7's R1-R9 and P1-P4, written to ``replay_gate.json`` and ``REPLAY_GATE.md``. Every
    criterion is printed PASS / FAIL / N/A with its numbers; nothing is hidden (:func:`run_gate`).
    Optional config keys (per-category stroke plan): ``stroke_cat_w`` (a knob spec: the table every
    weighted criterion prices) and ``dose_targets`` (``{category: delivered share}``) with ``tie``
    (default ``[["fwd", "back"]]``) -- R4 then SOLVES the table per category (:func:`_dose_per_cat`;
    ``dose_hand_min``, REQUIRED with ``dose_targets``: ``true`` binds side / turn / move_turn by their
    lower hand -- what reproduces the plan's 2026-09-27 W and X08:t3 tables -- ``false`` pools the
    hands). Neither key: the single-weight path, bit for bit as before.
``parity <dump.npz> [--tol 1e-5]``
    Recompute both terms from a per-step dump of the Isaac wrapper's own inputs and outputs and
    assert agreement (:func:`parity_check`; the dump format is below). A dump carrying ``cat`` and
    ``cat_scale`` is checked as ``reward == cat_scale[cat] * sum(pay) / dt``.
``pooled <arm_s3> <base_s3> <arm_s2> <base_s2> [--bar cat.key=value ...] [--json P]``
    A POOLED two-seed readout (:func:`pooled_readout`): per category/hand row and key, the MEAN of the
    two seed-matched paired deltas (``arm_s3`` vs ``base_s3``, ``arm_s2`` vs ``base_s2``), a POOLED mde
    (:func:`pooled_delta` -- the half-width of the two paired bootstraps' own draws averaged
    element-wise), the baseline seed delta ``|base_s2 - base_s3|``, the band
    ``max(pooled mde, |base seed delta|)``, and a PASS/TIE :func:`bar_verdict` against a bar (default
    :data:`DEFAULT_BARS`: side body_offset -0.05, side stance_stroke +0.010, side stride_along +0.02).
    JSON to stdout (and ``--json`` if given).

REPLAY CONVENTIONS (pinned by ``tests/unit/test_crab_hex_omni_replay_stroke_posture.py``)

* **Heading.** ``psi = atan2(2(wz + xy), 1 - 2(y^2 + z^2))`` of ``root_quat_w`` ``(w, x, y, z)`` --
  what ``asset.data.heading_w`` reports. Never the npz ``travel_yaw`` (the body-frame bearing of the
  command). On an omni suite stamp :func:`load_stamp` asserts
  ``max |wrap(psi - (hold_faces[segment] - heading_err))| < 1e-5`` over the steady own-hold steps.
* **Timing.** ``eval_crab_hex_gait.py`` records row ``t`` BEFORE ``env.step(actions[t])``; the reward
  of env step ``s`` reads the post-physics state, which is row ``s + 1``, under the command of row
  ``s``. So env step ``s`` is fed ``(force, foot_pos_w, root_pos_w, psi, root_lin_vel_b[:2],
  root_ang_vel_b[2])[s + 1]`` with ``(cmd_applied, stand, turning, lat_share, resampled)[s]``. A
  ``done`` at row ``k`` ends the episode: steps ``0 .. k-1`` are replayed (their post-states are
  rows ``1 .. k``), nothing after.
* **Slot from the hold's CATEGORY, never from ``||cmd||``.** stand -> STAND; fwd/side/back/diag ->
  MOVE; turn -> TURN; move_turn -> MOVE_TURN; the settle hold and the prologue (segment -1) -> STAND.
  ``turning`` = TURN / MOVE_TURN. ``lat_share = |sin(wrap(hold_dir - hold_face))|`` on MOVE holds,
  0 elsewhere (the wrapper's ``|sin(travel_dir_w - heading_target)|``). ``resampled`` = the segment id
  changed. ``cat_step`` (the per-category weight's key) = the hold's category index into
  :data:`CATEGORIES` (= the command term's ``TELEMETRY_CATEGORIES``), settle / prologue -> stand.
* **The sideways-task head** (``side_phases/.../slow__side``, ``step__side``): ``run_meta.travel_yaw``
  must be ~pi/2; ``cmd[:, :2]`` is rotated by ``+travel_yaw`` (``wz`` kept), every episode is
  category ``side`` (hand ``l``, ``cat_step`` side on every step), slot MOVE, ``lat_share`` 1; its
  own hold is every moving hold (``creep`` and ``low``); the rotated command's direction must agree
  with the achieved velocity (cos > 0.9 pooled). The math computes ``v_clk`` itself. Its episodes are
  31.5 s long.
* **Inputs.** contact force = ``foot_force_norm``; foot positions = ``foot_pos_w``, both in FOOT_ORDER
  FL, FR, ML, MR, RL, RR (``run_meta.foot_body_names`` is checked); ``turn_radius`` =
  ``omni_clock_turn_radius`` from the head's ``params/env.yaml`` when present (else 1.2).
* **Refusals.** A stamp without ``complete.json`` (omni) or with fewer raws than ``num_envs``
  (sideways), and any ``--sample-actions`` rollout.

READOUT METRICS (per episode, over its steady own-hold window; the category value is the median of
the per-episode values unless stated)

* ``body_offset`` -- :func:`body_lead` of the math's own centroid EMA (``update_centroid_ema``, tau
  2 s, re-seeded on a resample), on steps with ``||cmd_xy|| > 0.12``; per-episode median.
* ``lead_y`` / ``trail_y`` / ``lead_trail_asym`` -- median heading-frame ``|y|`` of the feet whose
  ``r . v_hat > 0`` (leading) / ``<= 0`` (trailing), rows by that sign per step and foot, never by leg
  name; ``asym = trail - lead``. Same steps as ``body_offset``.
* ``stride_along`` -- the MEAN of per-episode ``stride.pooled_along_command.mean`` from the stamp's
  ``metrics/episode_*.json``.
* ``stance_stroke`` -- median own sweep ``B`` (m) of the SCORED stances (closed, ``n >= 5`` credited
  steps, not reset-tainted) that close in the window, from ``foot_stroke_step(info=...)``.
* ``stroke_share`` -- sum of the scored stances' own sweep ``B`` / sum of their anchor sweep ``Ba``.
  ``Ba`` adds the foot's own world motion back, so it is the BODY progress along each foot's
  command direction at the foot, over the same credited steps. (The investigation's 0.81/0.40 used
  a different stance definition; the validation only requires fwd > 1.5 x side.)
* ``slip_ratio`` -- median ``W / max(B, 0.01)`` of the scored stances (the math's rho).
* ``lead_scuff_frac`` -- the fraction of leading-row loaded foot-steps (F >= 20 N) with world planar
  foot speed > 0.10 m/s.
* ``knee_cap_lead`` / ``knee_cap_trail`` -- per row, the fraction of loaded foot-steps with
  ``|femur_tibia applied torque| >= 395 N m`` (joint indices from ``run_meta.joint_names``).
* ``stroke_per_s`` / ``centring_per_s`` -- term income per second at w = 1 over the GATED moving
  own-hold steps (clock speed > 0.2 and not STAND): ``sum(pay) / (n dt)`` and ``mean(b)``.
* ``track_per_s`` -- kernel tracking income per second on the same steps, in the economy's weights:
  translation ``track_lin_vel_xy_exp``; turn ``track_ang_vel_z_exp + reward_yaw_progress``;
  move_turn all three.

The per-category stroke plan's mechanism keys (2026-09-27; every one a MEAN key -- the category value
is the mean of the per-episode values, :data:`MEAN_KEYS`, like ``stride_along``):

* ``achieved_speed`` -- mean over the ``body_offset`` steps (own hold, ``||cmd_xy|| > 0.12``) of the
  body-frame velocity projected on the commanded direction, ``root_lin_vel_b[:2] . cmd/|cmd|``, state
  rows post-shifted (row ``s + 1`` for env step ``s``). ``nan`` where nothing is commanded (stand,
  turn).
* ``stride_per_speed`` -- the episode's ``stride_along`` / its ``achieved_speed`` (s: metres per
  metre-per-second). An episode whose achieved speed is <= :data:`SPS_MIN_SPEED_M_S` (0.03 m/s) is
  EXCLUDED (``nan``) and counted in the row's ``n_sps_excluded``: a near-stalled walker's ratio is
  unbounded and would dominate a mean.
* ``touchdown_hz`` -- per-foot touchdown rate (Hz per foot): ``gait_eval``'s STEP count
  (:func:`gait_eval.metrics.step_tap_runs`, loaded = force > ``MOTION_TOUCH_REF_N`` 50 N, a step an
  unloaded run of >= ``RIGIDITY_STEP_MIN_STEPS`` steps begun at a liftoff inside the window) over the
  six feet, / (6 x window seconds) -- the eval's own ``step_rate_hz / 6``.
* ``taps_per_s`` -- ``step_tap_runs``' TAPS (unloaded runs of 1 .. ``RIGIDITY_STEP_MIN_STEPS`` - 1
  steps), summed over the six feet, per window second.

The two contact keys run over the steady own-hold window (``Stamp.own``, one contiguous block per
hold, each block its own ``step_tap_runs`` window) at the raw-row indexing ``gait_eval/report.py``
uses -- as the UNISON keys do.

THE ECONOMY (``income`` / ``gate``). :func:`load_economy` reads the dump's own ``reward_clock_schedule``
/ ``reward_clock_swing_apex`` weights and calls ``replay_clock_economy.economy_from_env_yaml`` with
whichever of ``REC.CLOCK_FREE`` (0.001 / 0.001, e.g. W) or ``REC.CLOCK_HELD`` (1.0 / 1.0, e.g. a t3
from-scratch stage) matches -- and the returned clock weights are then ASSERTED equal to the dump's own
(the helper's ``clock=`` argument sets ``Economy.clock`` directly; it does not read it off the dump
itself). A dump whose clock weights are neither is refused. ``penalty_tibia_deviation_in_stance`` (the
replay helper's gap) is priced
here from ``joint_pos`` and the knee defaults of the dump's ``init_state`` (:func:`tibia_deviation`).
``reward_stride_length`` is the term being swapped, so it is not in the context. The per-step
context is ``pre_clip_total`` (weight x value, NO dt) of row ``s + 1``. The manager sums
``func * weight * dt`` and clips at 0, so a term's DELIVERED pay at step ``s`` is
``min(x, max(0, x + ctx))`` with ``x`` its own ``w * pay`` (stroke; ``pay = R * dt``) or
``w * b * dt`` (centring) and ``ctx`` = ``dt * pre_clip_total`` plus the other new term.

PARITY DUMP FORMAT (``parity``; to be written by a small Isaac-side hook, one row per env step):
an ``.npz`` with, for ``S`` steps and ``N`` envs,

=====================  =================  ==============================================
key                    shape / dtype      meaning (as the wrapper sees it, this step)
=====================  =================  ==============================================
``force``              [S, N, 6] f32      ``net_forces_w[:, feet].norm(-1)``
``foot_pos_w``         [S, N, 6, 3] f32   ``body_pos_w[:, feet]``
``root_pos_w``         [S, N, 3] f32
``heading``            [S, N] f32         ``asset.data.heading_w``
``cmd``                [S, N, 3] f32      ``command_manager.get_command(...)``
``stand``              [S, N] bool        ``cmd_term.stand_mask()``
``root_lin_vel_b_xy``  [S, N, 2] f32
``root_ang_vel_b_z``   [S, N] f32
``turning``            [S, N] bool        ``slot_id >= SLOT_TURN``
``lat_share``          [S, N] f32         the centring wrapper's ``lat_share``
``resampled``          [S, N] bool        the centring wrapper's ``resampled``
``reward``             [S, N] f32         ``RewardFootStroke.__call__``'s return (``R``)
``b``                  [S, N] f32         ``RewardBodyCentring.__call__``'s return
``reset``              [S, N] bool        optional: the env was reset before this call
``pay`` / ``o``        [S,N,6] / [S,N]    optional: the wrappers' ``self.pay`` / ``self.o``
``cat``                [S, N] int8        optional, ``KRABBY_OMNI_STROKE_CAT_W`` armed only: the
                                          step's ``telemetry_category()`` index
``cat_scale``          [7] f64            optional, with ``cat``: the resolved multiplier table
``dt``, ``turn_radius`` scalar            ``env.step_dt``, ``env.cfg.omni_clock_turn_radius``
=====================  =================  ==============================================

plus any :class:`StrokeParams` field as a scalar when the run overrode it. The check compares the
per-step stance income in the manager's units (``reward * dt`` = sum of per-foot pay) and ``b``;
``R`` itself (= pay / dt) is also reported, but it magnifies float32 rounding by 1 / dt. With ``cat``
/ ``cat_scale`` the recomputed ``R`` is multiplied by ``cat_scale[cat]`` (float32, as the wrapper
does) before the comparison; ``pay`` is compared UNSCALED (the wrapper keeps ``self.pay`` at w = 1).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

HERE = Path(__file__).resolve()
TOOLS = HERE.parent
OMNI = HERE.parents[2]                                      # .../crab_hex_omni_task
TASKS = OMNI.parent                                         # .../parkour_tasks/parkour_tasks
GAIT_SCRIPTS = TASKS / "crab_hex_forward_task" / "scripts"
STROKE_MATH_PY = OMNI / "mdp" / "crab_hex_omni_stroke_math.py"
RMT_PY = TOOLS / "replay_motion_terms.py"
REC_PY = OMNI / "experiments" / "2026-09-23_clock_deprecation" / "replay_clock_economy.py"
EC_PY = OMNI / "eval_categories.py"
RES_PY = TOOLS / "resolution.py"
#: ``KRABBY_OMNI_STROKE_CAT_W``'s own parser (pure at module scope): ONE grammar for the knob and
#: the replay, so a table the replay prices is exactly a table training would accept.
STROKE_KNOBS_PY = OMNI / "config" / "crab_hex" / "stroke_knobs.py"
#: the candidate-ladder driver's own regex read of a training dump's ``params/env.yaml``
#: (``_yaml_block`` / ``_dict_entries`` / ``_grep_scalar`` / ``_func_swapped_ok`` / ``_num``): reused
#: by :func:`train_log_crosscheck`'s weight guard rather than re-deriving the same indent-sensitive
#: parse a second time (module docstring, "PARITY DUMP FORMAT" neighbour: one dump shape, one reader).
RUN_CANDIDATES_PY = TOOLS / "run_candidates.py"

#: W, the omni head of record (2026-09-25): its training dump is the ONE economy.
W_ENV_YAML = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
                  "2026-09-25_14-53-27_seed3_X03_c6x_A15pB/params/env.yaml")


def _load(path: Path, name: str):
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SM = _load(STROKE_MATH_PY, "stroke_posture_replay__stroke_math")   # a PRIVATE instance (R9 patches it)
RMT = _load(RMT_PY, "stroke_posture_replay__rmt")
REC = _load(REC_PY, "stroke_posture_replay__rec")
EC = _load(EC_PY, "stroke_posture_replay__eval_categories")
SK = _load(STROKE_KNOBS_PY, "stroke_posture_replay__stroke_knobs")
if set(SK.STROKE_CAT_KEYS) | {"stand"} != set(EC.CATEGORIES):
    raise ImportError(f"stroke_knobs.STROKE_CAT_KEYS {SK.STROKE_CAT_KEYS} + stand != eval_categories.CATEGORIES "
                      f"{EC.CATEGORIES}: the knob's categories and the replay's no longer name the same rows")
_RES = None
_GM = None
_RC = None


def res():
    """``resolution.py`` (``cluster_draw``, ``half_width``), loaded on first use."""
    global _RES
    if _RES is None:
        _RES = _load(RES_PY, "stroke_posture_replay__resolution")
    return _RES


def gait_metrics():
    """``gait_eval.metrics`` (the eval's own functions: ``stride_metrics``, ``step_tap_runs``)."""
    global _GM
    if _GM is None:
        _GM = RMT.gait_metrics()
    return _GM


def run_candidates():
    """``run_candidates.py`` (its ``params/env.yaml`` regex readers), loaded on first use. Its own
    module-level work is path-loading ``run_phases.py`` / ``ledger_cli.py`` (stdlib only, no
    ``isaaclab``) -- nothing here launches a subprocess or touches the GPU. NOT named ``rc`` --
    ``train_log_crosscheck`` below already binds that name locally (``rs, rc = row.get(...)``), and a
    same-named module function would silently become unreachable (``UnboundLocalError``) anywhere in
    that function's scope."""
    global _RC
    if _RC is None:
        _RC = _load(RUN_CANDIDATES_PY, "stroke_posture_replay__run_candidates")
    return _RC


CATEGORIES = tuple(EC.CATEGORIES)
TRANSLATION = tuple(EC.TRANSLATION_CATEGORIES)
ROTATIONAL = ("turn", "move_turn")
MOVING = tuple(c for c in CATEGORIES if c != "stand")
PAIRED = tuple(EC.PAIRED_CATEGORIES)
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
STAND_LABELS = ("settle",)

LOADED_N = SM.FORCE_HOLD_N                  # a loaded foot-step for the scuff / knee readouts
SCUFF_SPEED_M_S = 0.10
KNEE_CAP_NM = 395.0
OFFSET_MIN_CMD = SM.CENTRING_MIN_CMD_M_S    # 0.12, the readout's moving-step gate
HEADING_GUARD_TOL = 1e-5
SIDE_TRAVEL_YAW = math.pi / 2
SIDE_COS_MIN = 0.9
BOOT_B = 2000
BOOT_SEED = 20260925

#: Step-6 reproduction targets. ``tol`` is relative (10 %); ``hard=False`` rows are reported only.
VALIDATION_OMNI = (
    # (key, category, target, tol, hard, note)
    ("body_offset", "side", (0.23, 0.25), 0.10, True, "E1 side body lead ~+0.23-0.25 m"),
    ("lead_y", "side", 0.73, 0.10, True, "leading row median |y|"),
    ("trail_y", "side", 1.23, 0.10, True, "trailing row median |y|"),
    ("stride_along", "side", 0.075, 0.10, True, "json pooled_along_command, mean of episode means"),
    ("stride_along", "fwd", 0.196, 0.10, True, "json pooled_along_command, mean of episode means"),
    ("stroke_share", "side", 0.40, None, False, "investigation value (different stance definition)"),
    ("stroke_share", "fwd", 0.81, None, False, "investigation value (different stance definition)"),
)
VALIDATION_SIDE = (
    ("lead_y", "side", 0.93, 0.10, True, "sideways head leading row median |y|"),
    ("trail_y", "side", 0.96, 0.10, True, "sideways head trailing row median |y|"),
)
STROKE_SHARE_FWD_OVER_SIDE = 1.5

#: Step-8 premise (on W): side body_offset >= 0.15 m and side stride_along <= 0.6 x fwd.
PREMISE_OFFSET_MIN = 0.15
PREMISE_STRIDE_RATIO_MAX = 0.6

#: The per-category stroke plan's mechanism keys (module docstring): what a stride change is made of.
MECHANISM_KEYS = ("achieved_speed", "stride_per_speed", "touchdown_hz", "taps_per_s")
READOUT_KEYS = ("body_offset", "lead_y", "trail_y", "lead_trail_asym", "stride_along", "stance_stroke",
                "stroke_share", "slip_ratio", "lead_scuff_frac", "knee_cap_lead", "knee_cap_trail",
                "stroke_per_s", "centring_per_s", "track_per_s") + MECHANISM_KEYS
MEAN_KEYS = ("stride_along",) + MECHANISM_KEYS
#: ``stride_per_speed`` leaves out an episode whose achieved along-command speed is at or below this
#: (m/s) -- counted in the row's ``n_sps_excluded`` (plan step 2 / audit note 5).
SPS_MIN_SPEED_M_S = 0.03


# ============================================================================ parameters
@dataclass
class StrokeParams:
    """Every parameter the replay can vary. The first block are ``foot_stroke_step`` arguments; the
    second are module constants of the math (patched on this tool's PRIVATE module instance, never
    the training import -- :func:`patched_math`); the last block are ``body_centring_step``'s."""

    rho0: float = SM.RHO0
    drho: float = SM.DRHO
    power: float = SM.POWER
    c_max: float = SM.C_MAX
    prog_tau: float = SM.PROG_TAU_S
    prog_lo: float = SM.PROG_LO
    prog_width: float = SM.PROG_WIDTH
    force_open: float = SM.FORCE_OPEN_N
    force_hold: float = SM.FORCE_HOLD_N
    n_off: int = SM.N_OFF
    min_loaded: int = SM.MIN_LOADED_STEPS
    sigma: float = SM.CENTRING_SIGMA_M
    tau: float = SM.CENTRING_TAU_S
    min_cmd: float = SM.CENTRING_MIN_CMD_M_S
    phi_lo: float = SM.CENTRING_PHI_LO
    phi_sat: float = SM.CENTRING_PHI_SAT

    def stroke_kwargs(self) -> dict:
        return dict(rho0=self.rho0, drho=self.drho, power=self.power, c_max=self.c_max,
                    prog_tau=self.prog_tau, prog_lo=self.prog_lo, prog_width=self.prog_width)

    def centring_kwargs(self) -> dict:
        return dict(sigma=self.sigma, tau=self.tau, min_cmd=self.min_cmd, phi_lo=self.phi_lo,
                    phi_sat=self.phi_sat)

    def with_(self, **kw) -> "StrokeParams":
        return dataclasses.replace(self, **kw)


_CONST_OF = {"force_open": "FORCE_OPEN_N", "force_hold": "FORCE_HOLD_N", "n_off": "N_OFF",
             "min_loaded": "MIN_LOADED_STEPS"}


@contextlib.contextmanager
def patched_math(params: StrokeParams):
    """Set the math's stance-edge constants to ``params`` for the duration (this tool's private
    module instance only), restoring them afterwards."""
    saved = {c: getattr(SM, c) for c in _CONST_OF.values()}
    try:
        for f, c in _CONST_OF.items():
            setattr(SM, c, type(saved[c])(getattr(params, f)))
        yield
    finally:
        for c, v in saved.items():
            setattr(SM, c, v)


def parse_params(pairs: Sequence[str] | None, base: StrokeParams | None = None) -> StrokeParams:
    p = base or StrokeParams()
    names = {f.name: f for f in dataclasses.fields(StrokeParams)}
    kw = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise SystemExit(f"--params expects k=v, got {pair!r}")
        k, v = pair.split("=", 1)
        if k not in names:
            raise SystemExit(f"unknown parameter {k!r}; known: {sorted(names)}")
        kw[k] = int(v) if names[k].type in ("int", int) else float(v)
    return p.with_(**kw)


# ============================================================================ angles / geometry (numpy)
def heading_from_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    """``asset.data.heading_w`` from a ``(w, x, y, z)`` root quaternion."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi


def rotate_xy(xy: np.ndarray, angle) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.array(xy, dtype=np.float64, copy=True)
    out[..., 0] = c * xy[..., 0] - s * xy[..., 1]
    out[..., 1] = s * xy[..., 0] + c * xy[..., 1]
    return out


# ============================================================================ loading
_NPZ_KEYS = ("foot_force_norm", "foot_pos_w", "foot_lin_vel_w", "root_pos_w", "root_quat_w", "root_lin_vel_b",
             "root_ang_vel_b", "cmd_applied", "segment_id", "steady_mask", "done", "applied_torque", "joint_pos")
_ECON_KEYS = ("actions", "heading_err", "clock_phase")


@dataclass
class Stamp:
    """One eval stamp, every episode batched on axis 0 (padded to the longest with its last row)."""

    run_dir: str
    kind: str                       # "omni" | "side"
    meta: dict
    dt: float
    turn_radius: float
    params_source: str
    episodes: list[dict]            # {env_index, env, wave, category, hand, label, hold_ids, T, S}
    arr: dict[str, np.ndarray]      # [N, T, ...] float32 (cmd already rotated on the sideways head)
    psi: np.ndarray                 # [N, T] heading (float64)
    S: np.ndarray                   # [N] replayed env steps per episode
    own: np.ndarray                 # [N, T] bool -- steady own-hold rows (index = env step s)
    stand: np.ndarray               # [N, T] bool
    turning: np.ndarray             # [N, T] bool
    lat_share: np.ndarray           # [N, T] float
    resampled: np.ndarray           # [N, T] bool
    knee_idx: list[int]             # joint index of each foot's femur-tibia joint, FOOT_ORDER
    json_eps: list[dict | None]     # metrics/episode_*.json (None when absent)
    raw_paths: list[Path]
    action_clip: float | None = None
    notes: list[str] = field(default_factory=list)
    guard_max: float | None = None
    #: [N, T] int -- env step ``s``'s category, an index into :data:`CATEGORIES` (module docstring);
    #: ``None`` only on a Stamp built by hand (e.g. a calibration stand-in), which then cannot be
    #: priced with a per-category table (:func:`scale_rows` refuses).
    cat_step: np.ndarray | None = None

    @property
    def N(self) -> int:
        return len(self.episodes)


def _pad_stack(seqs: list[np.ndarray], T: int) -> np.ndarray:
    out = []
    for a in seqs:
        a = np.asarray(a)
        if a.shape[0] < T:
            a = np.concatenate([a, np.repeat(a[-1:], T - a.shape[0], axis=0)], axis=0)
        out.append(a)
    return np.stack(out, axis=0)


def _hold_slot(category: str | None) -> tuple[bool, bool]:
    """``(stand, turning)`` of a hold's category (``None`` = settle / prologue -> STAND)."""
    if category in (None, "stand"):
        return True, False
    return False, category in ROTATIONAL


def load_stamp(run_dir: Path | str, *, limit: int | None = None, heading_guard: bool = True) -> Stamp:
    """Load one eval stamp (an omni ``suite__omni`` stamp or a sideways-head stamp) for the replay."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"{run_dir}: no run_meta.json")
    meta = json.loads(meta_path.read_text())
    if meta.get("sample_actions"):
        raise SystemExit(f"{run_dir}: a --sample-actions (stochastic) rollout feeds no gate; refusing")
    names = [str(n) for n in meta.get("foot_body_names") or []]
    if names and [n.split("_", 1)[0] for n in names] != list(FOOT_ORDER):
        raise SystemExit(f"{run_dir}: foot_body_names {names} are not in FOOT_ORDER {FOOT_ORDER}")
    kind = "omni" if meta.get("suite") else "side"
    raw = run_dir / "raw"
    if kind == "omni":
        if not (run_dir / "complete.json").is_file():
            raise SystemExit(f"{run_dir}: no complete.json -- the stamp is partial; refusing")
        emap = list(meta.get("env_index_map") or [])
        labels = list(meta["hold_labels"])
        dirs = [float(x) for x in meta["hold_dirs"]]
        faces = [float(x) for x in meta["hold_faces"]]
        ents = []
        for ent in emap:
            label = ent.get("label")
            cat = EC.category_of_hold(label)
            if cat != ent.get("category"):
                raise SystemExit(f"{run_dir}: env {ent.get('env_index')} label {label!r} is category {cat!r} but "
                                 f"env_index_map says {ent.get('category')!r}")
            ents.append({"env_index": int(ent["env_index"]), "env": int(ent.get("env", ent["env_index"])),
                         "wave": int(ent.get("wave", 0)), "category": cat, "hand": ent.get("hand"),
                         "label": label, "hold_ids": (labels.index(label),)})
    else:
        ty = meta.get("travel_yaw")
        if ty is None or abs(float(ty) - SIDE_TRAVEL_YAW) > 1e-3:
            raise SystemExit(f"{run_dir}: not a suite stamp and run_meta.travel_yaw={ty!r} is not ~pi/2: "
                             "only the sideways-task head is read as a non-suite stamp")
        n_envs = int(meta.get("num_envs") or 0)
        have = len(list(raw.glob("episode_*.npz")))
        if n_envs <= 0 or have < n_envs or not (run_dir / "scenario_metrics.json").is_file():
            raise SystemExit(f"{run_dir}: partial sideways stamp ({have} raws of num_envs={n_envs}, "
                             "scenario_metrics.json required); refusing")
        labels = list(meta["hold_labels"])
        moving = tuple(i for i, lab in enumerate(labels) if lab != "stand")
        ents = [{"env_index": i, "env": i, "wave": 0, "category": "side", "hand": "l", "label": "side",
                 "hold_ids": moving} for i in range(n_envs)]
        dirs = faces = None
    if limit:
        ents = ents[:limit]
    env_d, _agent_d, src = RMT._train_dumps_for(meta, None, None)
    tp = RMT.train_params_from_dumps(env_d, None, src)
    dt = float(meta.get("dt") or tp.dt)
    if abs(dt - tp.dt) > 1e-9:
        raise SystemExit(f"{run_dir}: eval dt {dt} != training dt {tp.dt}")
    turn_radius = tp.clock_turn_r
    notes = []
    if env_d is None:
        notes.append(f"no training env.yaml beside the checkpoint; turn_radius defaults to {turn_radius}")

    seqs: dict[str, list] = {k: [] for k in _NPZ_KEYS}
    raw_paths, json_eps = [], []
    for e in ents:
        path = raw / f"episode_{e['env_index']:02d}.npz"
        raw_paths.append(path)
        with np.load(path) as z:
            for k in _NPZ_KEYS:
                seqs[k].append(z[k])
        jp = run_dir / "metrics" / f"episode_{e['env_index']:02d}.json"
        json_eps.append(json.loads(jp.read_text()) if jp.is_file() else None)
    T = max(len(s) for s in seqs["segment_id"])
    for e, seg, done in zip(ents, seqs["segment_id"], seqs["done"]):
        e["T"] = int(len(seg))
        d = np.asarray(done, dtype=bool)
        e["S"] = int(np.argmax(d)) if d.any() else int(len(seg)) - 1
    arr = {k: _pad_stack(v, T) for k, v in seqs.items()}
    N = len(ents)
    S = np.array([e["S"] for e in ents], dtype=np.int64)
    step_idx = np.arange(T)[None, :]
    alive = step_idx < S[:, None]
    psi = heading_from_quat(arr["root_quat_w"])
    seg = arr["segment_id"].astype(np.int64)
    steady = arr["steady_mask"].astype(bool)
    own = np.zeros((N, T), dtype=bool)
    for i, e in enumerate(ents):
        own[i] = np.isin(seg[i], e["hold_ids"])
    own &= steady & alive

    stand = np.zeros((N, T), dtype=bool)
    turning = np.zeros((N, T), dtype=bool)
    lat = np.zeros((N, T), dtype=np.float64)
    cat_step = np.full((N, T), CATEGORIES.index("stand"), dtype=np.int64)
    if kind == "omni":
        seg_cat = {-1: None}
        for i, lab in enumerate(labels):
            seg_cat[i] = None if lab in STAND_LABELS else EC.category_of_hold(lab)
        for sid, cat in seg_cat.items():
            m = seg == sid
            st, tu = _hold_slot(cat)
            stand[m], turning[m] = st, tu
            cat_step[m] = CATEGORIES.index(cat or "stand")
            if cat in TRANSLATION:
                lat[m] = abs(math.sin(float(wrap(dirs[sid] - faces[sid]))))
        unknown = ~np.isin(seg, list(seg_cat))
        if unknown.any():
            raise SystemExit(f"{run_dir}: segment ids {sorted(set(seg[unknown].tolist()))} are not in hold_labels")
    else:
        lat[:] = 1.0
        cat_step[:] = CATEGORIES.index("side")
        cmd = arr["cmd_applied"].astype(np.float64)
        cmd[..., :2] = rotate_xy(cmd[..., :2], float(meta["travel_yaw"]))
        arr["cmd_applied"] = cmd.astype(np.float32)
        # the rotated command must point where the robot actually went (pooled over moving steps)
        v = arr["root_lin_vel_b"][..., :2].astype(np.float64)
        mv = own & (np.linalg.norm(cmd[..., :2], axis=-1) > OFFSET_MIN_CMD)
        vs = np.zeros_like(v)
        vs[:, :-1] = v[:, 1:]
        if mv.any():
            c_sum = (cmd[..., :2][mv]).sum(axis=0)
            v_sum = (vs[mv]).sum(axis=0)
            cos = float(c_sum @ v_sum / max(np.linalg.norm(c_sum) * np.linalg.norm(v_sum), 1e-12))
            if cos <= SIDE_COS_MIN:
                raise SystemExit(f"{run_dir}: rotated sideways command vs achieved velocity cos {cos:.3f} "
                                 f"<= {SIDE_COS_MIN}: the travel_yaw rotation is wrong")
            notes.append(f"sideways head: cmd rotated by travel_yaw {float(meta['travel_yaw']):.4f}; "
                         f"cos(cmd, achieved velocity) = {cos:.3f}")
    resampled = np.zeros((N, T), dtype=bool)
    resampled[:, 1:] = seg[:, 1:] != seg[:, :-1]

    guard_max = None
    if kind == "omni" and heading_guard:
        with np.load(raw_paths[0]) as z0:
            has_herr = "heading_err" in z0.files
        if has_herr:
            worst = 0.0
            for i, path in enumerate(raw_paths):
                with np.load(path) as z:
                    herr = np.asarray(z["heading_err"], dtype=np.float64)
                n = herr.shape[0]
                m = own[i, :n]
                sid = seg[i, :n]
                face = np.asarray(faces)[np.clip(sid, 0, None)]
                d = np.abs(wrap(psi[i, :n] - (face - herr)))[m]
                if d.size:
                    worst = max(worst, float(d.max()))
            guard_max = worst
            if worst >= HEADING_GUARD_TOL:
                raise SystemExit(f"{run_dir}: heading guard failed: max |wrap(psi - (hold_face - heading_err))| = "
                                 f"{worst:.3g} >= {HEADING_GUARD_TOL} on steady own-hold steps")
        else:
            notes.append("no heading_err in the npz: heading guard skipped")

    joint_names = list(meta.get("joint_names") or [])
    knee_idx = []
    for leg in FOOT_ORDER:
        name = f"{leg}_Femur_Tibia_RevoluteJoint"
        if name not in joint_names:
            raise SystemExit(f"{run_dir}: joint {name} not in run_meta.joint_names")
        knee_idx.append(joint_names.index(name))

    return Stamp(run_dir=str(run_dir), kind=kind, meta=meta, dt=dt, turn_radius=turn_radius,
                 params_source=src, episodes=ents, arr=arr, psi=psi, S=S, own=own, stand=stand,
                 turning=turning, lat_share=lat, resampled=resampled, knee_idx=knee_idx,
                 json_eps=json_eps, raw_paths=raw_paths, notes=notes, guard_max=guard_max, cat_step=cat_step)


# ============================================================================ per-category stroke weights
def parse_cat_w(spec) -> dict[str, float] | None:
    """A per-category multiplier table (``KRABBY_OMNI_STROKE_CAT_W``'s grammar, per-category stroke
    plan 2026-09-27) -> ``{fwd, back, side, diag, turn, move_turn: m}`` with every key filled, or
    ``None`` for ``None`` / blank. ``spec`` is the knob's own string, or a ``{category: m}`` dict,
    which is written back as a spec (:func:`format_cat_w`) and parsed the same way -- so every
    refusal the knob makes (unknown / duplicate key, ``stand``, ``fwd != back``, a negative or
    non-finite value) the replay makes too, by the knob's own code (:data:`SK`)."""
    if spec is None:
        return None
    text = format_cat_w(spec) if isinstance(spec, dict) else spec
    try:
        table = SK.parse_stroke_cat_w(text)
    except ValueError as exc:
        raise SystemExit(f"stroke_cat_w: {exc}") from None
    return None if table is None else {k: table[k] for k in SK.STROKE_CAT_KEYS}


def format_cat_w(table: dict) -> str:
    """``{category: m}`` -> the knob's spec string, in :data:`SK` ``.STROKE_CAT_KEYS`` order first
    (``repr`` floats: the spec round-trips exactly). Unknown names are written through, so
    :func:`parse_cat_w` refuses them by name."""
    order = [k for k in SK.STROKE_CAT_KEYS if k in table] + [k for k in table if k not in SK.STROKE_CAT_KEYS]
    return ",".join(f"{k}:{float(table[k])!r}" for k in order)


def cat_scale_vector(table: dict[str, float] | None) -> np.ndarray:
    """``[len(CATEGORIES)]`` float64 multipliers in :data:`CATEGORIES` order (stand 1.0: it pays 0
    whatever it is scaled by -- the wrapper's ``_resolve_cat_scale`` default); all ones for ``None``."""
    vec = np.ones(len(CATEGORIES), dtype=np.float64)
    for name, m in (table or {}).items():
        vec[CATEGORIES.index(name)] = float(m)
    return vec


def scale_rows(st: Stamp, table: dict[str, float] | None) -> np.ndarray:
    """[N, T] float64: each env step's multiplier ``m_c`` by its category (``st.cat_step``); all ones
    when ``table`` is None. A table on a Stamp without ``cat_step`` is refused, never guessed."""
    T = st.arr["segment_id"].shape[1]
    if table is None:
        return np.ones((st.N, T), dtype=np.float64)
    if st.cat_step is None:
        raise SystemExit(f"{st.run_dir}: this stamp carries no per-step category (cat_step): a per-category "
                         "stroke table cannot be priced on it")
    return cat_scale_vector(table)[st.cat_step]


# ============================================================================ the replay
@dataclass
class Replay:
    """Per-step outputs of one replay (index = env step ``s``) and the closed-stance records."""

    pay: np.ndarray          # [N, T, 6] per-foot stance pay on its close step
    R: np.ndarray            # [N, T] the stroke term's return (sum pay / dt)
    b: np.ndarray            # [N, T] the centring bonus
    o: np.ndarray            # [N, T] body lead over the EMA'd centroid (m)
    gate: np.ndarray         # [N, T] clock speed > 0.2 and not STAND
    r: np.ndarray            # [N, T, 6, 2] heading-frame feet (post-state of step s)
    stances: dict[str, np.ndarray]   # one entry per CLOSING stance (see :func:`replay`)
    params: StrokeParams


def dup_pattern(T: int, every: int = 5, phase: int = 4) -> np.ndarray:
    """Steps whose contact reading repeats the previous one: ``s % every == phase``."""
    return (np.arange(T) % every) == phase


def f32_stale_pattern(T: int, *, physics_dt: float, decimation: int, update_period: float) -> np.ndarray:
    """Steps at which float32 Isaac Lab skips the contact-sensor refresh (``elapsed == 0`` in
    ``replay_motion_terms.sensor_elapsed_f32``): the sensor repeats its last reading."""
    el = RMT.sensor_elapsed_f32(T + 1, physics_dt=physics_dt, decimation=decimation, update_period=update_period)
    # env step s ends at row s + 1 of the episode timeline; the sensor clock counts from the reset
    return el[1:T + 1] == 0.0


def replay(st: Stamp, params: StrokeParams | None = None, *, dup: np.ndarray | None = None) -> Replay:
    """Run :func:`foot_stroke_step` and :func:`body_centring_step` over every episode of ``st``.

    ``dup`` ([T] or [N, T] bool): on those env steps the contact force repeats the previous step's
    READING (the sensor's stale refresh); positions stay fresh. Returns :class:`Replay`.
    """
    params = params or StrokeParams()
    N, T = st.N, st.arr["segment_id"].shape[1]
    Smax = int(st.S.max()) if N else 0
    tf = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)  # noqa: E731
    F = tf(st.arr["foot_force_norm"])
    FP = tf(st.arr["foot_pos_w"])
    RP = tf(st.arr["root_pos_w"])
    V = tf(st.arr["root_lin_vel_b"])
    Wz = tf(st.arr["root_ang_vel_b"][..., 2])
    PSI = tf(st.psi)
    CMD = tf(st.arr["cmd_applied"])
    STAND = torch.as_tensor(st.stand)
    TURN = torch.as_tensor(st.turning)
    LAT = tf(st.lat_share)
    RES_ = torch.as_tensor(st.resampled)
    if dup is not None:
        dup = np.broadcast_to(np.asarray(dup, dtype=bool)[..., :T], (N, T)) if np.ndim(dup) == 1 else dup
        DUP = torch.as_tensor(np.ascontiguousarray(dup))
    stt = SM.init_stroke_state(N, 6)
    cst = SM.init_centring_state(N)
    pay = np.zeros((N, T, 6), dtype=np.float32)
    Rr = np.zeros((N, T), dtype=np.float32)
    bb = np.zeros((N, T), dtype=np.float32)
    oo = np.zeros((N, T), dtype=np.float32)
    gate = np.zeros((N, T), dtype=bool)
    rr = np.zeros((N, T, 6, 2), dtype=np.float32)
    recs: dict[str, list] = {k: [] for k in ("env", "foot", "step", "B", "Ba", "W", "n", "mean_l", "slip_gate",
                                             "g_prog", "payable", "pay")}
    alive_all = torch.as_tensor(np.arange(T)[None, :] < st.S[:, None])
    last_force = F[:, 1] if T > 1 else F[:, 0]
    kw = params.stroke_kwargs()
    ckw = params.centring_kwargs()
    with patched_math(params):
        for s in range(Smax):
            force = F[:, s + 1]
            if dup is not None:
                force = torch.where(DUP[:, s, None], last_force, force)
            last_force = force
            info: dict = {}
            R, stt, p = SM.foot_stroke_step(
                force, FP[:, s + 1], RP[:, s + 1], PSI[:, s + 1], CMD[:, s], STAND[:, s], V[:, s + 1, :2],
                Wz[:, s + 1], stt, st.dt, st.turn_radius, TURN[:, s], RES_[:, s], info=info, **kw)
            b, cst, o = SM.body_centring_step(
                FP[:, s + 1], RP[:, s + 1], PSI[:, s + 1], CMD[:, s], STAND[:, s], V[:, s + 1, :2], cst, st.dt,
                LAT[:, s], RES_[:, s], **ckw)
            alive = alive_all[:, s]
            pay[:, s] = (p * alive[:, None]).numpy()
            Rr[:, s] = (R * alive).numpy()
            bb[:, s] = (b * alive).numpy()
            oo[:, s] = o.numpy()
            v_clk = SM.clock_speed(SM.stroke_command(CMD[:, s], TURN[:, s]), st.turn_radius)
            gate[:, s] = ((v_clk > SM.CLOCK_CMD_STOP_M_S) & ~STAND[:, s] & alive).numpy()
            rr[:, s] = SM.feet_heading_frame(FP[:, s + 1], RP[:, s + 1], PSI[:, s + 1]).numpy()
            closing = info["closing"] & alive[:, None]
            if bool(closing.any()):
                ei, fi = torch.nonzero(closing, as_tuple=True)
                recs["env"].append(ei.numpy())
                recs["foot"].append(fi.numpy())
                recs["step"].append(np.full(ei.shape[0], s))
                for k in ("B", "Ba", "W", "n", "mean_l", "slip_gate", "g_prog", "payable"):
                    recs[k].append(info[k][ei, fi].numpy())
                recs["pay"].append(p[ei, fi].numpy())
    stances = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in recs.items()}
    for k in ("env", "foot", "step", "n"):
        stances[k] = stances[k].astype(np.int64)
    stances["payable"] = stances["payable"].astype(bool)
    _stance_derived(stances, params)
    return Replay(pay=pay, R=Rr, b=bb, o=oo, gate=gate, r=rr, stances=stances, params=params)


def _stance_derived(sd: dict, params: StrokeParams) -> None:
    """Per-stance gate attribution: ``ungated`` credit (own sweep, no gate), then the pay after each
    gate in the order the math applies them (anchor cap, slip gate, progress guard)."""
    L = np.maximum(sd["mean_l"], SM.MIN_STROKE_NORM_M)
    clampp = lambda c: np.clip(c, 0.0, params.c_max) ** params.power  # noqa: E731
    sd["ungated"] = clampp(sd["B"] / L)
    sd["after_anchor"] = clampp(np.minimum(sd["B"], sd["Ba"]) / L)
    sd["after_slip"] = sd["after_anchor"] * sd["slip_gate"]
    sd["rho"] = sd["W"] / np.maximum(sd["B"], SM.MIN_SWEEP_M)
    sd["short_n"] = sd["n"] < params.min_loaded
    sd["tainted"] = (~sd["short_n"]) & (~sd["payable"])


# ============================================================================ per-episode readout
def _nanmedian(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def tracking_rate(st: Stamp, econ) -> np.ndarray:
    """[N, T] weighted kernel tracking income per env step (step ``s``: command row ``s``, state row
    ``s + 1``), by the episode's category (module docstring)."""
    N, T = st.N, st.arr["segment_id"].shape[1]
    cmd = st.arr["cmd_applied"].astype(np.float64)
    v = np.zeros_like(st.arr["root_lin_vel_b"], dtype=np.float64)
    w = np.zeros_like(v)
    v[:, :-1] = st.arr["root_lin_vel_b"][:, 1:]
    w[:, :-1] = st.arr["root_ang_vel_b"][:, 1:]
    out = np.zeros((N, T))
    for i, e in enumerate(st.episodes):
        lin = econ.track_lin_w * RMT.track_lin_vel_xy_exp(cmd[i], v[i], econ.track_lin_std)
        ang = econ.track_ang_w * RMT.track_ang_vel_z_exp(cmd[i], w[i], econ.track_ang_std)
        yp = econ.yaw_progress_w * RMT.yaw_progress(cmd[i], w[i], 0.3)
        c = e["category"]
        out[i] = lin if c in TRANSLATION else (ang + yp if c == "turn" else lin + ang + yp if c == "move_turn"
                                               else 0.0)
    return out


def episode_values(st: Stamp, rp: Replay, econ=None) -> dict[str, np.ndarray]:
    """Every readout metric per episode ([N] arrays, NaN where undefined)."""
    N = st.N
    dt = st.dt
    T = st.arr["segment_id"].shape[1]
    cmd = st.arr["cmd_applied"].astype(np.float64)
    speed = np.linalg.norm(cmd[..., :2], axis=-1)
    vhat = cmd[..., :2] / np.maximum(speed, 1e-9)[..., None]
    moving = st.own & (speed > OFFSET_MIN_CMD)
    gated = st.own & rp.gate

    # state-row arrays shifted to env-step index (row s + 1 -> s)
    def post(a):
        out = np.zeros_like(a)
        out[:, :-1] = a[:, 1:]
        return out

    force = post(st.arr["foot_force_norm"]).astype(np.float64)
    fvel = np.linalg.norm(post(st.arr["foot_lin_vel_w"])[..., :2].astype(np.float64), axis=-1)
    knee = np.abs(post(st.arr["applied_torque"])[..., st.knee_idx].astype(np.float64))
    r = rp.r.astype(np.float64)
    along = np.einsum("ntfk,ntk->ntf", r, vhat)
    lead = along > 0.0
    absy = np.abs(r[..., 1])
    loaded = force >= LOADED_N
    track = tracking_rate(st, econ) if econ is not None else None
    along_speed = np.einsum("ntk,ntk->nt", post(st.arr["root_lin_vel_b"])[..., :2].astype(np.float64), vhat)

    vals = {k: np.full(N, np.nan) for k in READOUT_KEYS}
    vals["n_gated"] = np.zeros(N)
    vals["sps_excluded"] = np.zeros(N)
    sd = rp.stances
    sc = sd["payable"] & st.own[sd["env"], sd["step"]] if sd["env"].size else np.zeros(0, dtype=bool)
    for i in range(N):
        m = moving[i]
        if m.any():
            vals["body_offset"][i] = _nanmedian(rp.o[i][m])
            ly = absy[i][m][lead[i][m]]
            ty = absy[i][m][~lead[i][m]]
            vals["lead_y"][i] = _nanmedian(ly)
            vals["trail_y"][i] = _nanmedian(ty)
            vals["lead_trail_asym"][i] = vals["trail_y"][i] - vals["lead_y"][i]
            ld = loaded[i][m]
            lf = ld & lead[i][m]
            if lf.any():
                vals["lead_scuff_frac"][i] = float((fvel[i][m][lf] > SCUFF_SPEED_M_S).mean())
                vals["knee_cap_lead"][i] = float((knee[i][m][lf] >= KNEE_CAP_NM).mean())
            tf_ = ld & ~lead[i][m]
            if tf_.any():
                vals["knee_cap_trail"][i] = float((knee[i][m][tf_] >= KNEE_CAP_NM).mean())
        g = gated[i]
        vals["n_gated"][i] = g.sum()
        if g.any():
            vals["stroke_per_s"][i] = float(rp.pay[i][g].sum() / (g.sum() * dt))
            vals["centring_per_s"][i] = float(rp.b[i][g].mean())
            if track is not None:
                vals["track_per_s"][i] = float(track[i][g].mean())
        elif st.episodes[i]["category"] == "stand" and st.own[i].any():
            vals["stroke_per_s"][i] = float(rp.pay[i][st.own[i]].sum() / (st.own[i].sum() * dt))
            vals["centring_per_s"][i] = float(rp.b[i][st.own[i]].mean())
        k = sc & (sd["env"] == i)
        if k.any():
            vals["stance_stroke"][i] = _nanmedian(sd["B"][k])
            den = sd["Ba"][k].sum()
            vals["stroke_share"][i] = float(sd["B"][k].sum() / den) if den > 1e-9 else np.nan
            vals["slip_ratio"][i] = _nanmedian(sd["rho"][k])
        j = st.json_eps[i]
        pa = ((j or {}).get("stride") or {}).get("pooled_along_command") or {}
        if pa.get("mean") is not None:
            vals["stride_along"][i] = float(pa["mean"])
        # ---- mechanism keys (per-category stroke plan): speed x stride-per-speed x cadence
        if m.any():
            vals["achieved_speed"][i] = float(along_speed[i][m].mean())
        sa, spd = vals["stride_along"][i], vals["achieved_speed"][i]
        if math.isfinite(sa) and math.isfinite(spd):
            if spd > SPS_MIN_SPEED_M_S:
                vals["stride_per_speed"][i] = sa / spd
            else:
                vals["sps_excluded"][i] = 1.0
        td, taps = episode_step_rates(st, i)
        vals["touchdown_hz"][i], vals["taps_per_s"][i] = td, taps
    return vals


def episode_step_rates(st: Stamp, i: int) -> tuple[float, float]:
    """Episode ``i``'s ``(touchdown_hz, taps_per_s)`` (module docstring): ``gait_eval``'s own
    :func:`step_tap_runs` over each contiguous block of the steady own-hold window, raw rows, loaded =
    force > ``MOTION_TOUCH_REF_N``. ``touchdown_hz`` = steps / (6 x seconds) (Hz per foot); ``taps_per_s``
    = taps / seconds (six feet summed). ``nan`` on an empty window."""
    m = st.own[i]
    if not m.any():
        return float("nan"), float("nan")
    gm = gait_metrics()
    loaded = st.arr["foot_force_norm"][i] > float(gm.MOTION_TOUCH_REF_N)
    n_steps = n_taps = n_rows = 0
    for a, b in gm.contiguous_runs(m):
        steps, taps, _censored = gm.step_tap_runs(loaded[a:b], min_step_steps=int(gm.RIGIDITY_STEP_MIN_STEPS))
        n_steps += len(steps)
        n_taps += len(taps)
        n_rows += b - a
    secs = n_rows * st.dt
    return float(n_steps / (len(FOOT_ORDER) * secs)), float(n_taps / secs)


def _row_masks(st: Stamp) -> list[tuple[str, np.ndarray]]:
    """``(row name, [N] episode mask)`` for every category and every hand of a paired one."""
    cats = np.array([e["category"] for e in st.episodes])
    hands = np.array([str(e["hand"]) for e in st.episodes])
    rows = []
    for c in CATEGORIES:
        m = cats == c
        if not m.any():
            continue
        rows.append((c, m))
        if c in PAIRED:
            for h in EC.HANDS:
                mh = m & (hands == h)
                if mh.any() and mh.sum() != m.sum():
                    rows.append((f"{c}/{h}", mh))
    return rows


def aggregate(vals: dict[str, np.ndarray], mask: np.ndarray, key: str) -> float:
    return (_nanmean if key in MEAN_KEYS else _nanmedian)(vals[key][mask])


def readout_table(st: Stamp, vals: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for name, m in _row_masks(st):
        out[name] = {"n": int(m.sum()), **{k: aggregate(vals, m, k) for k in READOUT_KEYS}}
        if "sps_excluded" in vals:
            out[name]["n_sps_excluded"] = int(vals["sps_excluded"][m].sum())
    return out


# ============================================================================ unison/phase flag (SA0 decision 7)
# R5 as written FAILs on a cross-plant clause a per-foot stroke income cannot see: the term-alone
# credit on the forward-era ``lunge`` degenerate (a tripod-locked rock, both tripods planted and paid
# through the anchor-capped sweep) is 3.0x healthy at p = 1.5 (5.1x at p = 2). SA0's adopted mitigation
# is not a gate -- it is a FLAG, read at every readout park on arms S and C against their baseline B:
# per UNISON_CATEGORIES, (a) tripod coherence falling beyond its band, or (b) roll_rms collapsing
# beyond UNISON_ROLL_COLLAPSE_FRAC of the baseline's -- the v3 tip-rock/unison signature (roll_rms
# 0.038 -> 0.014). Neither metric is priced by the stroke term; both are recomputed here straight
# from the raw npz, in gait_eval.metrics's own terms, so the flag cannot inherit a stroke bug.
UNISON_CATEGORIES = ("fwd", "side", "back", "diag")
UNISON_KEYS = ("tripod_coherence", "roll_rms")
UNISON_ROLL_COLLAPSE_FRAC = 0.5      # SA0 decision 7: "roll_rms collapsing 0.038 -> 0.014"


def _unison_schedule_defaults() -> dict:
    """``gait_eval.schedule.DEFAULTS`` -- the eval's own window/contact constants (module docstring:
    "gait_eval metrics' definitions"), not re-guessed here."""
    return RMT.gait_schedule_defaults()


def episode_tripod_coherence(st: Stamp, i: int) -> float:
    """Episode ``i``'s ``tripod_score`` (:func:`gait_eval.metrics.tripod_window_metrics`), recomputed
    from raw contact (``foot_force_norm > contact_force_threshold``) over its steady own-hold window
    -- the SAME row indexing ``gait_eval/report.py`` uses (unshifted; ``st.own`` is already built from
    the raw ``steady_mask``/``segment_id`` at that indexing, module docstring). ``nan`` (never 0.0) on
    a window too short to score, or one ``gait_eval`` itself calls ``low_confidence`` -- exactly the
    exclusion ``tripod_score_by_hold`` applies before aggregating."""
    m = st.own[i]
    if not m.any():
        return float("nan")
    gm = gait_metrics()
    d = _unison_schedule_defaults()
    contact = st.arr["foot_force_norm"][i, m] > float(d["contact_force_threshold"])
    min_window_steps = max(1, int(round(float(d["min_window_s"]) / st.dt)))
    tri = gm.tripod_window_metrics(contact, dt=st.dt, min_window_steps=min_window_steps,
                                   min_cycles=int(d["min_cycles"]))
    if not tri["valid"] or tri["low_confidence"]:
        return float("nan")
    score = tri["tripod_score"]
    return float(score) if score is not None else float("nan")


def episode_roll_rms(st: Stamp, i: int) -> float:
    """Episode ``i``'s roll RMS (rad, :func:`gait_eval.metrics.orientation_metrics`) over its steady
    own-hold window, recomputed from ``root_quat_w`` at the same unshifted indexing as
    :func:`episode_tripod_coherence`. Roll is unaffected by a yaw rotation, so no travel-frame
    correction applies (``travel_yaw`` defaults to 0.0). ``nan`` on an empty window."""
    m = st.own[i]
    if not m.any():
        return float("nan")
    gm = gait_metrics()
    out = gm.orientation_metrics(st.arr["root_quat_w"][i, m])
    rms = out["roll_rms"]
    return float(rms) if rms is not None else float("nan")


def unison_values(st: Stamp) -> dict[str, np.ndarray]:
    """Every :data:`UNISON_KEYS` metric, per episode ([N] arrays, NaN where unmeasured)."""
    out = {k: np.full(st.N, np.nan) for k in UNISON_KEYS}
    for i in range(st.N):
        out["tripod_coherence"][i] = episode_tripod_coherence(st, i)
        out["roll_rms"][i] = episode_roll_rms(st, i)
    return out


def _unison_row_masks(st: Stamp) -> list[tuple[str, np.ndarray]]:
    """One row per :data:`UNISON_CATEGORIES` member present (no hand split -- the flag is a
    category-level watch, not a paired-hand comparison)."""
    cats = np.array([e["category"] for e in st.episodes])
    return [(c, cats == c) for c in UNISON_CATEGORIES if (cats == c).any()]


def unison_table(st: Stamp, vals: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for name, m in _unison_row_masks(st):
        out[name] = {"n": int(m.sum()), **{k: aggregate(vals, m, k) for k in UNISON_KEYS}}
    return out


#: Smallest tripod-coherence drop that can raise the unison flag, whatever the band (SA0 follow-up, 2026-09-26).
UNISON_MIN_DROP = 0.10


def unison_flags(st_arm: Stamp, va: dict, st_base: Stamp, vb: dict, *,
                 seed_a: tuple[Stamp, dict] | None = None, seed_b: tuple[Stamp, dict] | None = None,
                 roll_collapse_frac: float = UNISON_ROLL_COLLAPSE_FRAC) -> list[dict]:
    """SA0 decision 7's flag, per :data:`UNISON_CATEGORIES` (module docstring above).

    ``seed_a``/``seed_b`` are an optional same-stage seed pair (e.g. B's two seeds) whose OWN paired
    tripod-coherence delta gives the seed band: the band a single-seed arm-vs-baseline delta must
    clear is ``max(bootstrap MDE, |seed-pair delta|)``, so ordinary seed noise never flags on its own.
    Without a seed pair the band is the bootstrap MDE alone. This never rejects an arm -- it is
    reported, per the plan's step-7 mitigation and decision 7.
    """
    table_a, table_b = unison_table(st_arm, va), unison_table(st_base, vb)
    boot = paired_bootstrap(st_arm, va, st_base, vb, keys=UNISON_KEYS, rows_fn=_unison_row_masks)
    seed_boot = (paired_bootstrap(seed_a[0], seed_a[1], seed_b[0], seed_b[1], keys=UNISON_KEYS,
                                  rows_fn=_unison_row_masks) if seed_a and seed_b else None)
    out: list[dict] = []
    for cat in UNISON_CATEGORIES:
        ta, tb = table_a.get(cat), table_b.get(cat)
        if ta is None or tb is None:
            continue
        rb = boot.get(cat)
        rs = seed_boot.get(cat) if seed_boot else None
        d_tri = rb["tripod_coherence"]["delta"] if rb else float("nan")
        mde_tri = rb["tripod_coherence"]["mde"] if rb else None
        seed_d_tri = rs["tripod_coherence"]["delta"] if rs else None
        seed_d_abs = abs(seed_d_tri) if seed_d_tri is not None and math.isfinite(seed_d_tri) else None
        band_parts = [v for v in (mde_tri, seed_d_abs) if v is not None]
        # a unison or phase-locked gait drives tripod coherence toward 0; healthy heads differ by a few
        # hundredths (W vs X0x side: -0.016), so a flag also needs the drop to reach UNISON_MIN_DROP
        band_tri = max(band_parts + [UNISON_MIN_DROP]) if band_parts else None
        flag_tri = band_tri is not None and math.isfinite(d_tri) and d_tri < -band_tri
        reason = ""
        if flag_tri:
            reason = f"tripod coherence fell {d_tri:+.3f} beyond its band -{band_tri:.3f}"
        elif band_tri is None and math.isfinite(d_tri):
            reason = "no band: bootstrap MDE needs >= 20 finite pairs and no seed pair was given"
        out.append({"category": cat, "metric": "tripod_coherence", "arm": ta["tripod_coherence"],
                    "baseline": tb["tripod_coherence"], "delta": d_tri, "mde": mde_tri,
                    "seed_delta": seed_d_tri, "band": band_tri, "flagged": bool(flag_tri),
                    "reason": reason})
        ra_roll, rb_roll = ta["roll_rms"], tb["roll_rms"]
        ratio = (ra_roll / rb_roll if math.isfinite(ra_roll) and math.isfinite(rb_roll) and rb_roll > 0
                else float("nan"))
        flag_roll = math.isfinite(ratio) and ratio < roll_collapse_frac
        out.append({"category": cat, "metric": "roll_rms", "arm": ra_roll, "baseline": rb_roll,
                    "ratio": ratio, "threshold": roll_collapse_frac, "flagged": bool(flag_roll),
                    "reason": (f"roll_rms collapsed to {ratio:.2f}x baseline (< {roll_collapse_frac:g})"
                              if flag_roll else "")})
    return out


# ============================================================================ paired bootstrap
def paired_bootstrap(st_a: Stamp, va: dict, st_b: Stamp, vb: dict, *, B: int = BOOT_B,
                     seed: int = BOOT_SEED, keys: Sequence[str] = READOUT_KEYS,
                     rows_fn: Callable[[Stamp], list[tuple[str, np.ndarray]]] = _row_masks,
                     return_raw: bool = False) -> dict[str, dict]:
    """Env-clustered paired bootstrap of ``a - b`` per row and metric (module docstring).

    ``keys``/``rows_fn`` default to the readout's own (:data:`READOUT_KEYS`, :func:`_row_masks`);
    :func:`unison_flags` reuses this exact pairing, clustering and bootstrap for :data:`UNISON_KEYS`
    over :func:`_unison_row_masks` instead. ``return_raw=True`` also keeps each key's own length-``B``
    bootstrap draw array (``"boot"``, NaN where a draw's resample was all-NaN) -- :func:`pooled_delta`
    combines two independent calls' own draws into a POOLED mde.
    """
    R = res()
    ia = {e["env_index"]: i for i, e in enumerate(st_a.episodes)}
    ib = {e["env_index"]: i for i, e in enumerate(st_b.episodes)}
    shared = sorted(set(ia) & set(ib))
    if not shared:
        raise SystemExit("--vs: the two stamps share no env_index")
    pa = np.array([ia[k] for k in shared])
    pb = np.array([ib[k] for k in shared])
    for k in shared:
        ea, eb = st_a.episodes[ia[k]], st_b.episodes[ib[k]]
        if (ea["category"], ea["hand"], ea["label"]) != (eb["category"], eb["hand"], eb["label"]):
            raise SystemExit(f"--vs: env_index {k} holds differ ({ea['label']} vs {eb['label']}); not a paired eval")
    sub = dataclasses.replace(st_a, episodes=[st_a.episodes[i] for i in pa])
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    clusters = np.array([st_a.episodes[i]["env"] for i in pa])
    for name, m in rows_fn(sub):
        idx = np.flatnonzero(m)
        cl = clusters[idx]
        uniq = np.unique(cl)
        members = [idx[cl == c] for c in uniq]
        kmax = max(len(x) for x in members)
        tab = np.full((uniq.size, kmax), -1)
        for j, x in enumerate(members):
            tab[j, :len(x)] = x
        draw = rng.integers(0, uniq.size, size=(B, uniq.size))
        pos = tab[draw].reshape(B, -1)
        row = {}
        for key in keys:
            xa = va[key][pa].astype(np.float64)
            xb = vb[key][pb].astype(np.float64)
            xa_p = np.where(pos >= 0, xa[np.clip(pos, 0, None)], np.nan)
            xb_p = np.where(pos >= 0, xb[np.clip(pos, 0, None)], np.nan)
            with np.errstate(all="ignore"), _quiet():
                red = np.nanmean if key in MEAN_KEYS else np.nanmedian
                d = red(xa_p, axis=1) - red(xb_p, axis=1)
            point = aggregate({key: xa}, m, key) - aggregate({key: xb}, m, key)
            fin = d[np.isfinite(d)]
            row[key] = {"delta": point,
                        "ci": ([float(np.quantile(fin, 0.025)), float(np.quantile(fin, 0.975))]
                               if fin.size >= 20 else None),
                        "mde": R.half_width(fin, 0.975) if fin.size >= 20 else None}
            if return_raw:
                row[key]["boot"] = d
        out[name] = {"n_pairs": int(m.sum()), "n_clusters": int(uniq.size), **row}
    return out


@contextlib.contextmanager
def _quiet():
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


# ============================================================================ pooled two-seed readout
#: Formation gate default bars (t3-first plan step): a side key must clear this signed shift, in this
#: direction, to PASS. Only ``side`` (and its ``side/l``, ``side/r`` hand rows) ship a default; any
#: other category/key is reported with no bar and no verdict unless ``--bar`` adds one.
DEFAULT_BARS: dict[str, dict[str, float]] = {
    "side": {"body_offset": -0.05, "stance_stroke": 0.010, "stride_along": 0.02},
}


def parse_bars(pairs: Sequence[str] | None, base: dict[str, dict[str, float]] | None = None) \
        -> dict[str, dict[str, float]]:
    """``--bar cat.key=value ...``, layered onto ``base`` (default :data:`DEFAULT_BARS`)."""
    bars = {cat: dict(keys) for cat, keys in (base if base is not None else DEFAULT_BARS).items()}
    for pair in pairs or ():
        lhs, _, v = pair.partition("=")
        cat, dot, key = lhs.partition(".")
        if not (dot and v):
            raise SystemExit(f"--bar expects cat.key=value, got {pair!r}")
        bars.setdefault(cat, {})[key] = float(v)
    return bars


def bar_verdict(mean_delta: float, band: float, bar: float) -> str:
    """PASS iff ``mean_delta`` clears ``bar`` in ``bar``'s own sign AND is resolvable (``|mean_delta|
    > band``); TIE otherwise (wrong direction, short of the bar, or not resolvable from noise) --
    never a hard FAIL (BACKGROUND: "continue only arms that move their own target")."""
    if not (math.isfinite(mean_delta) and math.isfinite(band)):
        return "TIE"
    cleared = mean_delta <= bar if bar < 0 else mean_delta >= bar
    return "PASS" if cleared and abs(mean_delta) > band else "TIE"


def pooled_delta(st_arm_a: Stamp, va_a: dict, st_base_a: Stamp, vb_a: dict,
                 st_arm_b: Stamp, va_b: dict, st_base_b: Stamp, vb_b: dict, *,
                 keys: Sequence[str] = READOUT_KEYS, B: int = BOOT_B, seed: int = BOOT_SEED) -> dict[str, dict]:
    """Two independent seed-matched paired bootstraps (``a`` e.g. seed 3, ``b`` e.g. seed 2), combined
    per row/key into the MEAN of the two paired deltas and a POOLED mde: the half-width of the two
    bootstraps' own draws averaged element-wise (same ``B``/``seed`` in both calls, so draw ``i`` of
    ``a`` and draw ``i`` of ``b`` are a matched pair to average, per the plan's "half-width of the mean
    of the two independent estimates")."""
    boot_a = paired_bootstrap(st_arm_a, va_a, st_base_a, vb_a, B=B, seed=seed, keys=keys, return_raw=True)
    boot_b = paired_bootstrap(st_arm_b, va_b, st_base_b, vb_b, B=B, seed=seed, keys=keys, return_raw=True)
    R = res()
    out: dict[str, dict] = {}
    for name in [n for n in boot_a if n in boot_b]:
        ra, rb = boot_a[name], boot_b[name]
        row: dict[str, Any] = {"n_pairs_a": ra["n_pairs"], "n_pairs_b": rb["n_pairs"]}
        for key in keys:
            da, db = ra[key], rb[key]
            mean_delta = (da["delta"] + db["delta"]) / 2.0
            comb = (da["boot"] + db["boot"]) / 2.0
            fin = comb[np.isfinite(comb)]
            row[key] = {"delta_a": da["delta"], "delta_b": db["delta"], "mean_delta": mean_delta,
                       "pooled_mde": R.half_width(fin, 0.975) if fin.size >= 20 else None}
        out[name] = row
    return out


def _sps_excluded(st: Stamp, vals: dict | None) -> dict[str, int]:
    """Per readout row, the episodes ``stride_per_speed`` left out (achieved speed <=
    :data:`SPS_MIN_SPEED_M_S`); empty when ``vals`` carries no such count."""
    if not isinstance(vals, dict) or "sps_excluded" not in vals:
        return {}
    return {name: int(vals["sps_excluded"][m].sum()) for name, m in _row_masks(st)}


def pooled_readout(arm_s3: str, base_s3: str, arm_s2: str, base_s2: str, *,
                   keys: Sequence[str] = READOUT_KEYS, bars: dict[str, dict[str, float]] | None = None,
                   limit: int | None = None) -> dict:
    """The pooled two-seed readout: per category/hand row and key, the mean of the two seed-matched
    paired deltas (arm vs its own-seed baseline, seeds 3 and 2), a pooled mde (:func:`pooled_delta`),
    the baseline seed delta ``|base_s2 - base_s3|`` (the two baselines' own paired point delta -- the
    band a real move must clear even where the bootstrap mde is small), the band
    ``max(pooled mde, |base seed delta|)``, and, where ``bars`` (default :data:`DEFAULT_BARS`) names
    one for that category, a PASS/TIE :func:`bar_verdict`. A category's bar also applies to its hand
    rows (``side`` -> ``side/l``, ``side/r``)."""
    bars = bars if bars is not None else DEFAULT_BARS

    def _vals(run_dir):
        st = load_stamp(run_dir, limit=limit)
        return st, episode_values(st, replay(st))

    sa3, va3 = _vals(arm_s3)
    sb3, vb3 = _vals(base_s3)
    sa2, va2 = _vals(arm_s2)
    sb2, vb2 = _vals(base_s2)
    pooled = pooled_delta(sa3, va3, sb3, vb3, sa2, va2, sb2, vb2, keys=keys)
    seed_delta = paired_bootstrap(sb2, vb2, sb3, vb3, keys=keys)   # the baselines' own seed-pair delta
    # stride_per_speed's excluded episodes (v_cmd <= SPS_MIN_SPEED_M_S), per run: reported beside the key
    excl = ({lab: _sps_excluded(s, v) for lab, s, v in (("arm_s3", sa3, va3), ("base_s3", sb3, vb3),
                                                         ("arm_s2", sa2, va2), ("base_s2", sb2, vb2))}
            if "stride_per_speed" in keys else None)
    rows: dict[str, dict] = {}
    for name, prow in pooled.items():
        cat = name.split("/", 1)[0]
        row_bars = bars.get(cat) or {}
        seed_row = seed_delta.get(name)
        orow: dict[str, Any] = {"n_pairs_s3": prow["n_pairs_a"], "n_pairs_s2": prow["n_pairs_b"]}
        if excl is not None:
            orow["n_sps_excluded"] = {lab: e.get(name) for lab, e in excl.items()}
        for key in keys:
            v = prow[key]
            base_seed = abs(seed_row[key]["delta"]) if seed_row else float("nan")
            cands = [x for x in (v["pooled_mde"], base_seed) if x is not None and math.isfinite(x)]
            band = max(cands) if cands else float("nan")
            entry = {"delta_s3": v["delta_a"], "delta_s2": v["delta_b"], "mean_delta": v["mean_delta"],
                     "pooled_mde": v["pooled_mde"], "base_seed_delta": base_seed, "band": band}
            bar = row_bars.get(key)
            if bar is not None:
                entry["bar"] = bar
                entry["verdict"] = bar_verdict(v["mean_delta"], band, bar)
            orow[key] = entry
        rows[name] = orow
    return {"arm_s3": arm_s3, "base_s3": base_s3, "arm_s2": arm_s2, "base_s2": base_s2, "bars": bars, "rows": rows}


# ============================================================================ validation / premise
def _within(v: float, target, tol: float) -> bool:
    if not math.isfinite(v):
        return False
    lo, hi = (target, target) if not isinstance(target, tuple) else target
    return lo * (1 - tol) <= v <= hi * (1 + tol)


def validate(st: Stamp, table: dict) -> tuple[bool, list[str]]:
    """The step-6 reproduction targets; returns ``(ok, lines)``."""
    lines = []
    ok = True
    targets = VALIDATION_OMNI if st.kind == "omni" else VALIDATION_SIDE
    for key, cat, target, tol, hard, note in targets:
        v = (table.get(cat) or {}).get(key, float("nan"))
        if hard:
            passed = _within(v, target, tol)
            ok &= passed
            lines.append(f"{'PASS' if passed else 'FAIL'} {cat}.{key} = {v:.4f} target {target} +-{tol:.0%} ({note})")
        else:
            lines.append(f"INFO {cat}.{key} = {v:.4f} vs {target} ({note})")
    if st.kind == "omni":
        f = (table.get("fwd") or {}).get("stroke_share", float("nan"))
        s = (table.get("side") or {}).get("stroke_share", float("nan"))
        passed = math.isfinite(f) and math.isfinite(s) and f > STROKE_SHARE_FWD_OVER_SIDE * s
        ok &= passed
        lines.append(f"{'PASS' if passed else 'FAIL'} stroke_share fwd {f:.3f} > {STROKE_SHARE_FWD_OVER_SIDE} x side "
                     f"{s:.3f}")
    return ok, lines


def premise(table: dict) -> list[str]:
    """Plan step 8's premise lines (informational here; step 8 STOPs on them)."""
    side, fwd = table.get("side") or {}, table.get("fwd") or {}
    bo = side.get("body_offset", float("nan"))
    sa, fa = side.get("stride_along", float("nan")), fwd.get("stride_along", float("nan"))
    ratio = sa / fa if fa and math.isfinite(fa) else float("nan")
    return [f"{'PASS' if bo >= PREMISE_OFFSET_MIN else 'FAIL'} premise side body_offset {bo:+.3f} >= "
            f"{PREMISE_OFFSET_MIN}",
            f"{'PASS' if ratio <= PREMISE_STRIDE_RATIO_MAX else 'FAIL'} premise side stride_along {sa:.3f} <= "
            f"{PREMISE_STRIDE_RATIO_MAX} x fwd {fa:.3f} (ratio {ratio:.2f})"]


# ============================================================================ the economy
@dataclass(frozen=True)
class ExtEconomy(REC.Economy):
    """``replay_clock_economy.Economy`` plus extra per-row series priced as ``weight * p[key]``:
    ``tibia`` (``penalty_tibia_deviation_in_stance``), ``stroke`` (R) and ``centring`` (b)."""

    extra: tuple = ()


@contextlib.contextmanager
def extra_terms():
    """Make ``replay_clock_economy.term_values`` price :class:`ExtEconomy.extra` (for
    ``pre_clip_total`` and ``rg1_ordering``), restoring it afterwards."""
    orig = REC.term_values

    def term_values(p, econ):
        vals = orig(p, econ)
        for key, w in getattr(econ, "extra", ()):
            vals[key] = w * np.asarray(p[key], dtype=np.float64) if w else np.zeros(p["T"])
        return vals

    REC.term_values = term_values
    try:
        yield
    finally:
        REC.term_values = orig


def load_economy(env_yaml: Path | str) -> tuple[ExtEconomy, dict]:
    """The dump's own economy (module docstring), clock-free OR clock-held depending on the dump's own
    ``reward_clock_schedule`` / ``reward_clock_swing_apex`` weights, with the tibia penalty priced.
    Asserts that the returned clock weights equal the dump's own. Behaviour on a clock-free dump (e.g.
    W) is unchanged; a clock-held dump (e.g. a t3 from-scratch stage) now builds the economy with
    ``REC.CLOCK_HELD`` instead of refusing."""
    env_yaml = Path(env_yaml)
    if not env_yaml.is_file():
        raise SystemExit(f"{env_yaml}: the economy's env.yaml is not on disk")
    d = RMT._yaml_load_tolerant(env_yaml)
    rw = d.get("rewards") or {}
    dump_clock = (float((rw.get("reward_clock_schedule") or {}).get("weight") or 0.0),
                  float((rw.get("reward_clock_swing_apex") or {}).get("weight") or 0.0))
    if dump_clock == tuple(REC.CLOCK_FREE):
        clock = REC.CLOCK_FREE
    elif dump_clock == tuple(REC.CLOCK_HELD):
        clock = REC.CLOCK_HELD
    else:
        raise SystemExit(f"{env_yaml}: dump clock weights {dump_clock} are neither CLOCK_FREE "
                         f"{REC.CLOCK_FREE} nor CLOCK_HELD {REC.CLOCK_HELD}: not a recognised economy")
    econ = REC.economy_from_env_yaml(env_yaml, clock=clock)
    if tuple(econ.clock) != dump_clock:
        raise SystemExit(f"economy clock weights {econ.clock} != the dump's {dump_clock}: {env_yaml} "
                         f"is not a {'clock-free' if clock == REC.CLOCK_FREE else 'clock-held'} economy")
    tib = rw.get("penalty_tibia_deviation_in_stance") or {}
    tib_w = float(tib.get("weight") or 0.0)
    prm = tib.get("params") or {}
    info = {"source": str(env_yaml), "clock": list(econ.clock), "dump_clock": list(dump_clock),
            "tibia_weight": tib_w,
            "tibia_force_threshold": float(prm.get("contact_force_threshold", 0.1)),
            "tibia_joints": list(((prm.get("asset_cfg") or {}).get("joint_names")) or []),
            "default_joint_pos": dict((((d.get("scene") or {}).get("robot") or {}).get("init_state") or {})
                                      .get("joint_pos") or {}),
            "notes": [n for n in econ.notes if "penalty_tibia_deviation_in_stance" not in n]
            + ["penalty_tibia_deviation_in_stance priced here from joint_pos and the dump's knee defaults"
               if tib_w else "penalty_tibia_deviation_in_stance: weight 0 in the dump"]}
    fields = {f.name: getattr(econ, f.name) for f in dataclasses.fields(REC.Economy)}
    return ExtEconomy(**fields, extra=(("tibia", tib_w),)), info


def stamp_clock(st: Stamp) -> tuple[tuple[float, float], Path] | None:
    """``((schedule, apex) clock weights, env.yaml)`` of the head a stamp evaluated -- its checkpoint's
    own ``params/env.yaml`` -- or None when the stamp names no checkpoint or that file is not on disk.

    NOTE(stroke formation 2026-09-26, verification): ``income``'s ``--econ-env-yaml`` defaults to W's
    CLOCK-FREE dump, so a clock-HELD stamp (a t3 head) priced without the flag was silently read under
    W's economy. ``income`` compares this against the economy's own clock and refuses a mismatch."""
    ck = (st.meta or {}).get("checkpoint")
    if not ck:
        return None
    p = Path(ck).parent / "params" / "env.yaml"
    if not p.is_file():
        return None
    rw = RMT._yaml_load_tolerant(p).get("rewards") or {}
    return ((float((rw.get("reward_clock_schedule") or {}).get("weight") or 0.0),
             float((rw.get("reward_clock_swing_apex") or {}).get("weight") or 0.0)), p)


def tibia_deviation(z: dict, joint_names: Sequence[str], info: dict) -> np.ndarray:
    """``penalty_joint_deviation_when_in_contact`` per row: ``sum_knees (q - q0)^2 * [F > thr]``,
    knees paired with feet by leg prefix (the cfg lists both in FOOT_ORDER)."""
    q = np.asarray(z["joint_pos"], dtype=np.float64)
    q0 = RMT.resolve_default_pos(info["default_joint_pos"], joint_names)
    force = np.asarray(z["foot_force_norm"], dtype=np.float64)
    out = np.zeros(q.shape[0])
    for f, leg in enumerate(FOOT_ORDER):
        j = list(joint_names).index(f"{leg}_Femur_Tibia_RevoluteJoint")
        out += (q[:, j] - q0[j]) ** 2 * (force[:, f] > info["tibia_force_threshold"])
    return out


def econ_primitives(st: Stamp, econ: ExtEconomy, info: dict) -> list[dict] | None:
    """Per-episode ``replay_clock_economy.episode_primitives`` (+ ``tibia``), or ``None`` for the
    sideways head (its schedule is not an omni suite: the omni economy does not apply)."""
    if st.kind != "omni":
        return None
    if st.action_clip is None:
        st.action_clip = REC.action_clip_from_agent_yaml(REC.head_agent_yaml(st.meta))
    labels = list(st.meta["hold_labels"])
    joint_names = list(st.meta["joint_names"])
    out = []
    for path in st.raw_paths:
        with np.load(path) as npz:
            z = {k: npz[k] for k in npz.files if k in _NPZ_KEYS + _ECON_KEYS}
        p = REC.episode_primitives(z, action_clip=st.action_clip, hold_labels=labels)
        p["tibia"] = tibia_deviation(z, joint_names, info)
        out.append(p)
    return out


def context_rows(st: Stamp, prims: list[dict] | None, econ: ExtEconomy) -> np.ndarray | None:
    """[N, T] ``pre_clip_total`` (weight x value, no dt) of W's economy at the reward of env step
    ``s`` (state row ``s + 1``); ``None`` without primitives."""
    if prims is None:
        return None
    T = st.arr["segment_id"].shape[1]
    ctx = np.zeros((st.N, T))
    with extra_terms():
        for i, p in enumerate(prims):
            pc = REC.pre_clip_total(p, econ)
            n = pc.shape[0]
            ctx[i, :n - 1] = pc[1:]
    return ctx


ABSORBED_MIN_PAY_PER_S = 1e-6


def delivered(x: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    """A term's post-clip pay: ``min(x, max(0, x + ctx))`` (manager units; ``ctx`` = every other
    term's ``weight * value * dt`` at that step)."""
    return np.minimum(x, np.maximum(0.0, x + ctx))


def income_table(st: Stamp, rp: Replay, *, w_stroke: float, w_centring: float, econ=None,
                 ctx: np.ndarray | None = None, cat_w: dict[str, float] | None = None) -> dict[str, dict]:
    """``income`` subcommand's per-category rows (over gated moving own-hold steps).

    ``cat_w`` (a :func:`parse_cat_w` table, per-category stroke plan): every stroke number priced in
    the manager's units -- nominal, delivered, absorbed, clip-bind, ``stroke_dose``, and the
    centring term's context -- is at ``w_stroke * m_c`` by each step's category (:func:`scale_rows`),
    and each row gains ``w_stroke_eff``. ``stroke_per_s`` stays at w = 1, unscaled. ``None``: every
    number and key exactly as before the table existed."""
    dt = st.dt
    track = tracking_rate(st, econ) if econ is not None else None
    pay_sum = rp.pay.sum(axis=-1).astype(np.float64)
    if cat_w is not None:
        pay_sum = pay_sum * scale_rows(st, cat_w)
    stroke_x = w_stroke * pay_sum       # manager units per step
    cent_x = w_centring * rp.b.astype(np.float64) * dt
    rows = {}
    sd = rp.stances
    for name, m_ep in _row_masks(st):
        cat = name.split("/")[0]
        own = st.own & m_ep[:, None]
        g = own & (rp.gate if cat != "stand" else True)
        n = int(g.sum())
        row: dict[str, Any] = {"n_ep": int(m_ep.sum()), "n_steps": n}
        if cat_w is not None:
            row["w_stroke_eff"] = w_stroke * float(cat_scale_vector(cat_w)[CATEGORIES.index(cat)])
        if n == 0:
            rows[name] = row
            continue
        secs = n * dt
        row["stroke_per_s"] = float(rp.pay.sum(axis=-1)[g].sum() / secs)
        row["centring_per_s"] = float(rp.b[g].sum() * dt / secs)
        row["track_per_s"] = float(track[g].sum() * dt / secs) if track is not None else None
        if ctx is not None:
            base = dt * ctx
            d_s = delivered(stroke_x, base + cent_x)
            d_c = delivered(cent_x, base + stroke_x)
            row["stroke_nominal_per_s"] = float(stroke_x[g].sum() / secs)
            row["stroke_delivered_per_s"] = float(d_s[g].sum() / secs)
            row["centring_nominal_per_s"] = float(cent_x[g].sum() / secs)
            row["centring_delivered_per_s"] = float(d_c[g].sum() / secs)
            # an absorbed share is reported only where the term pays more than float noise (back's
            # lat_share is |sin(pi)| ~ 1e-7 in float32)
            floor = ABSORBED_MIN_PAY_PER_S * secs
            row["stroke_absorbed"] = (1.0 - d_s[g].sum() / stroke_x[g].sum()) if stroke_x[g].sum() > floor else None
            row["centring_absorbed"] = (1.0 - d_c[g].sum() / cent_x[g].sum()) if cent_x[g].sum() > floor else None
            # the manager's own clip_bind_frac: pre-clip total < 0 (strict)
            row["clip_bind_base"] = float((base[g] < 0.0).mean())
            row["clip_bind_with"] = float((base[g] + stroke_x[g] + cent_x[g] < 0.0).mean())
            row["clip_shift_pp"] = 100.0 * (row["clip_bind_with"] - row["clip_bind_base"])
            if track is not None and row["track_per_s"]:
                row["stroke_dose"] = row["stroke_delivered_per_s"] / row["track_per_s"]
                row["centring_dose"] = row["centring_delivered_per_s"] / row["track_per_s"]
        else:
            row["stroke_nominal_per_s"] = float(stroke_x[g].sum() / secs)
            row["centring_nominal_per_s"] = float(cent_x[g].sum() / secs)
            if track is not None and row["track_per_s"]:
                row["stroke_dose"] = row["stroke_nominal_per_s"] / row["track_per_s"]
                row["centring_dose"] = row["centring_nominal_per_s"] / row["track_per_s"]
        k = g[sd["env"], sd["step"]] if sd["env"].size else np.zeros(0, dtype=bool)
        pk = k & sd["payable"]
        U = sd["ungated"][k].sum()
        row["stances"] = {
            "closed": int(k.sum()), "scored": int(pk.sum()),
            "rej_n": int((k & sd["short_n"]).sum()), "rej_taint": int((k & sd["tainted"]).sum()),
            "anchor_binds": int((pk & (sd["Ba"] < sd["B"])).sum()),
            "slip_binds": int((pk & (sd["slip_gate"] < 0.99)).sum()),
            "prog_binds": int((pk & (sd["g_prog"] < 0.99)).sum()),
            "prog_zero": int((pk & (sd["g_prog"] <= 0.0)).sum()),
            "ungated_credit": float(U),
            "lost_n_taint": float(sd["ungated"][k & ~sd["payable"]].sum()),
            "lost_anchor": float((sd["ungated"] - sd["after_anchor"])[pk].sum()),
            "lost_slip": float((sd["after_anchor"] - sd["after_slip"])[pk].sum()),
            "lost_prog": float((sd["after_slip"] - sd["pay"])[pk].sum()),
            "kept": float(sd["pay"][pk].sum() / U) if U > 0 else None,
        }
        rows[name] = row
    return rows


# ============================================================================ training-log cross-check
EVENT_TERM = "reward_stride_length"          # the stroke term trains under the swapped name
CENTRING_TERM = "reward_body_centring"


def _log_last_means(log_dir: Path, tags: Iterable[str], n: int = 200) -> dict[str, float | None]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    out = {}
    for t in tags:
        v = [e.value for e in ea.Scalars(t)][-n:] if t in have else []
        v = [x for x in v if math.isfinite(x)]
        out[t] = float(np.mean(v)) if v else None
    return out


def _train_log_stroke_weight_check(log_dir: Path, *, w_stroke: float, cat_w: dict[str, float] | None) -> None:
    """Refuse a ``train_log_crosscheck`` whose ``--w-stroke`` / ``--stroke-cat-w`` do not match the
    weight the run ACTUALLY trained with (hardening, 2026-09-28).

    ``rows`` (the replay's own nominal-per-second numbers) is priced at ``w_stroke * m_c`` by
    :func:`income_table`, using the CLI's ``--w-stroke`` / ``--stroke-cat-w``. The ``r = log / replay``
    the caller prints only means what it says when the run's own ``params/env.yaml`` shows the SAME
    price: ``reward_stride_length``'s ``weight`` times its (optional) ``params.cat_scale[c]``. Passing
    a stale or wrong ``--w-stroke`` would otherwise silently rescale every ``r`` and read as a training
    drift that is really a CLI mismatch.

    Reads the dump with :func:`run_candidates`'s own ``_yaml_block`` / ``_dict_entries`` /
    ``_grep_scalar`` -- the SAME indent-sensitive regex read the candidate-ladder audit uses on this
    exact dump shape (term at indent 2, ``params`` block at 4, a param key such as ``cat_scale`` at 6,
    that dict's own entries at 8) -- rather than a second, independently-drifting parser.

    Two cases keep TODAY's behaviour (no check, just a one-line note): no ``params/env.yaml`` under
    ``log_dir`` at all, and a ``reward_stride_length`` term that never swapped to ``RewardFootStroke``
    (a stock run, or one whose stroke knob was off) -- this cross-check does not apply to either.
    Anything else either matches to 1e-9 relative or raises ``SystemExit``, including the dump
    carrying a ``cat_scale`` table while the CLI was given no ``--stroke-cat-w`` at all: an unpriced
    table is not "close enough", it means this call is pricing every category as if uniform while the
    run itself scaled them.
    """
    env_yaml = Path(log_dir) / "params" / "env.yaml"
    if not env_yaml.is_file():
        print(f"[income] {env_yaml}: no params/env.yaml; train-log weight cross-check skipped")
        return
    txt = env_yaml.read_text()
    R = run_candidates()
    if not R._func_swapped_ok(txt, EVENT_TERM, "RewardFootStroke"):
        print(f"[income] {env_yaml}: {EVENT_TERM}.func is not RewardFootStroke; train-log weight cross-check skipped")
        return
    term_block = R._yaml_block(txt, EVENT_TERM)
    weight = R._num(R._grep_scalar(term_block, "weight", indent=r"\s{4}"))
    if weight is None:
        raise SystemExit(f"{env_yaml}: {EVENT_TERM}.weight is missing or unreadable; cannot cross-check "
                         "--w-stroke against the run that produced this log")
    cat_entries = R._dict_entries(term_block, "cat_scale", indent=6)
    if cat_entries is not None and cat_w is None:
        raise SystemExit(f"{env_yaml}: {EVENT_TERM} carries a params.cat_scale table but this call was given "
                         "no --stroke-cat-w -- pass the run's own table (or --stroke-cat-w matching it) so "
                         "the log and the replay are priced at the same per-category weight")
    mismatches = []
    for c in SK.STROKE_CAT_KEYS:
        m_logged = 1.0
        if cat_entries is not None and c in cat_entries:
            m_logged = R._num(cat_entries[c])
            if m_logged is None:
                raise SystemExit(f"{env_yaml}: {EVENT_TERM}.params.cat_scale.{c} is unreadable "
                                 f"({cat_entries[c]!r})")
        logged = weight * m_logged
        eff = w_stroke * (float(cat_w[c]) if cat_w is not None else 1.0)
        if abs(eff - logged) > 1e-9 * max(abs(eff), abs(logged), 1e-30):
            mismatches.append(f"{c}: tool {eff!r} != log {logged!r}")
    if mismatches:
        raise SystemExit(f"{env_yaml}: {EVENT_TERM} weight x cat_scale != this call's --w-stroke x "
                         "--stroke-cat-w for " + "; ".join(mismatches))


def train_log_crosscheck(st: Stamp, rp: Replay, rows: dict, econ_yaml: Path, log_dir: Path, *,
                         w_stroke: float, cat_w: dict[str, float] | None = None) -> dict | None:
    """The dose smoke's ``r`` (plan step 12(ii)): training telemetry vs this replay.

    * Stock tracking rows through ``replay_motion_terms.train_log_crosscheck`` (its keys this tool
      does not replay are NaN).
    * Per category: ``Episode_RewardCat/<c>/reward_stride_length`` and ``/reward_body_centring`` (per
      second, weight included, over the category's dwell) against this replay's nominal per-second
      pay, their ratio ``r`` (``rows`` priced with the run's own ``--stroke-cat-w`` table when it had
      one -- :func:`income_table` -- so the logged ``w_S * m_c`` income is divided by a replay nominal
      at the same ``w_S * m_c``, like for like), and ``/clip_absorbed_event`` / ``/clip_bind_frac`` against the replayed
      absorbed share / clip-bind fraction. Training draws slots from its pool and samples actions,
      so this is the ratio the SA0 rule reads, not an equality.
    * ``w_stroke`` / ``cat_w`` -- the SAME ``--w-stroke`` / ``--stroke-cat-w`` ``rows`` was priced at
      (hardening, 2026-09-28) -- are cross-checked against ``log_dir``'s own ``params/env.yaml``
      before any of the above is computed: see :func:`_train_log_stroke_weight_check`.
    """
    if log_dir is None or not Path(log_dir).is_dir():
        return None
    _train_log_stroke_weight_check(log_dir, w_stroke=w_stroke, cat_w=cat_w)
    env_d = RMT._yaml_load_tolerant(Path(econ_yaml))
    tp = RMT.train_params_from_dumps(env_d, None, f"env: {econ_yaml}")
    herr = None
    if st.kind == "omni":
        herr = np.zeros(st.S.shape + (st.arr["segment_id"].shape[1],))
        for i, path in enumerate(st.raw_paths):
            with np.load(path) as z:
                h = np.asarray(z["heading_err"], dtype=np.float64) if "heading_err" in z.files else None
            if h is not None:
                herr[i, :h.shape[0] - 1] = h[1:]
    cmd = st.arr["cmd_applied"].astype(np.float64)
    v = np.zeros_like(st.arr["root_lin_vel_b"], dtype=np.float64)
    w = np.zeros_like(v)
    v[:, :-1] = st.arr["root_lin_vel_b"][:, 1:]
    w[:, :-1] = st.arr["root_ang_vel_b"][:, 1:]
    eps = []
    nan = float("nan")
    for i in range(st.N):
        m = st.own[i]
        if not m.any():
            continue
        sums = {k: nan for k in ("action_rate", "airtime", "airtime_exact", "foot_idle")}
        sums["inc_lin"] = float(RMT.track_lin_vel_xy_exp(cmd[i], v[i], tp.track_lin_std)[m].sum() * st.dt)
        sums["inc_ang"] = float(RMT.track_ang_vel_z_exp(cmd[i], w[i], tp.track_ang_std)[m].sum() * st.dt)
        sums["inc_facing"] = float(RMT.facing_exp(herr[i])[m].sum() * st.dt) if herr is not None else nan
        sums["inc_yaw_progress"] = float(RMT.yaw_progress(cmd[i], w[i], tp.yaw_progress_cap)[m].sum() * st.dt)
        eps.append({"sums": sums, "duration_s": float(m.sum() * st.dt)})
    stock = RMT.train_log_crosscheck({"params": tp, "episodes": eps}, Path(log_dir))
    tags = []
    for c in CATEGORIES:
        tags += [f"Episode_RewardCat/{c}/{EVENT_TERM}", f"Episode_RewardCat/{c}/{CENTRING_TERM}",
                 f"Episode_RewardCat/{c}/clip_absorbed_event", f"Episode_RewardCat/{c}/clip_bind_frac"]
    logged = _log_last_means(Path(log_dir), tags)
    per_cat = {}
    for c in CATEGORIES:
        row = rows.get(c) or {}
        ls, lc = logged[f"Episode_RewardCat/{c}/{EVENT_TERM}"], logged[f"Episode_RewardCat/{c}/{CENTRING_TERM}"]
        rs, rc = row.get("stroke_nominal_per_s"), row.get("centring_nominal_per_s")
        per_cat[c] = {"stroke_log": ls, "stroke_replay": rs, "r_stroke": (ls / rs) if ls is not None and rs else None,
                      "centring_log": lc, "centring_replay": rc,
                      "r_centring": (lc / rc) if lc is not None and rc else None,
                      "absorbed_log": logged[f"Episode_RewardCat/{c}/clip_absorbed_event"],
                      "absorbed_replay": row.get("stroke_absorbed"),
                      "clip_bind_log": logged[f"Episode_RewardCat/{c}/clip_bind_frac"],
                      "clip_bind_replay": row.get("clip_bind_with")}
    return {"log_dir": str(log_dir), "stock": stock, "per_category": per_cat}


# ============================================================================ printing
def _fmt(x, f="{:.3f}", na="--"):
    if x is None:
        return na
    try:
        if not math.isfinite(float(x)):
            return na
    except (TypeError, ValueError):
        return str(x)
    return f.format(x)


def print_table(rows: list[list[str]], head: list[str], out=None) -> str:
    widths = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    lines = ["  ".join(str(h).rjust(w) for h, w in zip(head, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(str(c).rjust(w) for c, w in zip(r, widths)) for r in rows]
    text = "\n".join(lines)
    if out is not None:
        print(text, file=out)
    return text


_READOUT_COLS = (("body_offset", "{:+.3f}"), ("lead_y", "{:.3f}"), ("trail_y", "{:.3f}"),
                 ("lead_trail_asym", "{:+.3f}"), ("stride_along", "{:.3f}"), ("stance_stroke", "{:.4f}"),
                 ("stroke_share", "{:.3f}"), ("slip_ratio", "{:.2f}"), ("lead_scuff_frac", "{:.3f}"),
                 ("knee_cap_lead", "{:.3f}"), ("knee_cap_trail", "{:.3f}"), ("stroke_per_s", "{:.4f}"),
                 ("centring_per_s", "{:.3f}"), ("track_per_s", "{:.3f}"), ("achieved_speed", "{:.3f}"),
                 ("stride_per_speed", "{:.3f}"), ("touchdown_hz", "{:.3f}"), ("taps_per_s", "{:.3f}"))
_SHORT = {"body_offset": "offset", "lead_y": "lead|y|", "trail_y": "trail|y|", "lead_trail_asym": "asym",
          "stride_along": "stride", "stance_stroke": "stroke_B", "stroke_share": "share", "slip_ratio": "rho",
          "lead_scuff_frac": "scuff", "knee_cap_lead": "kneeL", "knee_cap_trail": "kneeT",
          "stroke_per_s": "S/s", "centring_per_s": "P/s", "track_per_s": "trk/s", "achieved_speed": "v_cmd",
          "stride_per_speed": "stride/v", "touchdown_hz": "td_hz/ft", "taps_per_s": "taps/s"}


def render_readout(name: str, st: Stamp, table: dict, boot: dict | None = None) -> str:
    head = ["row", "n"] + [_SHORT[k] for k, _ in _READOUT_COLS]
    rows = [[r, str(v["n"])] + [_fmt(v[k], f) for k, f in _READOUT_COLS] for r, v in table.items()]
    parts = [f"== readout {name} ({st.kind}, {st.N} episodes)", print_table(rows, head)]
    excl = {r: v["n_sps_excluded"] for r, v in table.items() if v.get("n_sps_excluded")}
    parts.append(f"stride/v leaves out episodes with v_cmd <= {SPS_MIN_SPEED_M_S:g} m/s: "
                 + (", ".join(f"{r} {n}" for r, n in excl.items()) if excl else "none"))
    if boot:
        head = ["row", "pairs"] + [_SHORT[k] for k, _ in _READOUT_COLS]
        rows = []
        for r, v in boot.items():
            rows.append([r, str(v["n_pairs"])] + [
                (f"{v[k]['delta']:+.3f}" + (f"±{v[k]['mde']:.3f}" if v[k]["mde"] is not None else ""))
                if v[k]["delta"] is not None and math.isfinite(v[k]["delta"]) else "--" for k, _ in _READOUT_COLS])
        parts += ["== delta (run - vs) ± MDE (env-clustered paired bootstrap, B=2000, 95 % half-width)",
                  print_table(rows, head)]
    return "\n".join(parts)


def render_unison(flags: list[dict]) -> str:
    """SA0 decision 7's readout section: one row per :data:`UNISON_CATEGORIES` x :data:`UNISON_KEYS`
    (module docstring above ``unison_flags``); FLAG never fails the run, it only marks the row."""
    head = ["category", "metric", "arm", "baseline", "delta/ratio", "band/thr", "seed Δ", "flag"]
    rows = []
    for f in flags:
        if f["metric"] == "tripod_coherence":
            deltacol = _fmt(f["delta"], "{:+.3f}")
            bandcol = _fmt(f["band"], "{:.3f}") if f["band"] is not None else "--"
            seedcol = _fmt(f["seed_delta"], "{:+.3f}")
        else:
            deltacol = _fmt(f["ratio"], "{:.2f}x")
            bandcol = f"<{f['threshold']:g}"
            seedcol = "--"
        rows.append([f["category"], f["metric"], _fmt(f["arm"], "{:.4f}"), _fmt(f["baseline"], "{:.4f}"),
                    deltacol, bandcol, seedcol, "FLAG" if f["flagged"] else "ok"])
    parts = ["== UNISON flags (SA0 decision 7: a per-foot stroke income cannot see unison or phase; "
            "reported, never gates)", print_table(rows, head)]
    reasons = [f"{f['category']}/{f['metric']}: {f['reason']}" for f in flags if f["reason"]]
    if reasons:
        parts.append("\n".join(reasons))
    return "\n".join(parts)


def render_income(name: str, rows: dict, *, w_stroke: float, w_centring: float,
                  cat_w: dict[str, float] | None = None) -> str:
    tabled = cat_w is not None
    head = ["row", "n_ep"] + (["w_S*m"] if tabled else []) + [
        "S/s@1", "P/s@1", "trk/s", "S_del/s", "S_dose", "S_abs%", "P_del/s", "P_dose",
        "P_abs%", "clip0%", "clip1%", "dpp"]
    t1 = []
    for r, v in rows.items():
        lead = [r, str(v["n_ep"])] + ([_fmt(v.get("w_stroke_eff"), "{:.4g}")] if tabled else [])
        if v.get("n_steps", 0) == 0:
            t1.append(lead + ["--"] * (len(head) - len(lead)))
            continue
        t1.append(lead + [_fmt(v["stroke_per_s"], "{:.4f}"), _fmt(v["centring_per_s"]),
                   _fmt(v.get("track_per_s")), _fmt(v.get("stroke_delivered_per_s", v.get("stroke_nominal_per_s")),
                                                    "{:.4f}"),
                   _fmt(v.get("stroke_dose"), "{:.3f}"), _fmt(_pc(v.get("stroke_absorbed")), "{:.1f}"),
                   _fmt(v.get("centring_delivered_per_s", v.get("centring_nominal_per_s")), "{:.4f}"),
                   _fmt(v.get("centring_dose"), "{:.3f}"), _fmt(_pc(v.get("centring_absorbed")), "{:.1f}"),
                   _fmt(_pc(v.get("clip_bind_base")), "{:.1f}"), _fmt(_pc(v.get("clip_bind_with")), "{:.1f}"),
                   _fmt(v.get("clip_shift_pp"), "{:+.1f}")])
    head2 = ["row", "closed", "scored", "rej_n", "rej_taint", "anchor", "slip", "prog", "prog0", "lost_anc%",
             "lost_slip%", "lost_prog%", "kept"]
    t2 = []
    for r, v in rows.items():
        s = v.get("stances")
        if not s:
            continue
        U = s["ungated_credit"] or float("nan")
        t2.append([r, s["closed"], s["scored"], s["rej_n"], s["rej_taint"], s["anchor_binds"], s["slip_binds"],
                   s["prog_binds"], s["prog_zero"], _fmt(100 * s["lost_anchor"] / U, "{:.1f}"),
                   _fmt(100 * s["lost_slip"] / U, "{:.1f}"), _fmt(100 * s["lost_prog"] / U, "{:.1f}"),
                   _fmt(s["kept"])])
    table_line = ([f"== per-category stroke weights (KRABBY_OMNI_STROKE_CAT_W {format_cat_w(cat_w)}): effective "
                   "w_S x m_c = " + ", ".join(f"{c} {w_stroke * m:.4g} (x{m:g})" for c, m in cat_w.items())]
                  if tabled else [])
    return "\n".join([f"== income {name} (w_stroke {w_stroke:g}, w_centring {w_centring:g}; per second over gated "
                      "moving own-hold steps; S = stroke, P = centring)", *table_line, print_table(t1, head),
                      "== stances per gate (closed in the window; binds = the gate trimmed a scored stance; lost "
                      "% of ungated credit)", print_table([[str(c) for c in r] for r in t2], head2)])


def _pc(x):
    return None if x is None else 100.0 * x


# ============================================================================ parity
PARITY_KEYS = ("force", "foot_pos_w", "root_pos_w", "heading", "cmd", "stand", "root_lin_vel_b_xy",
               "root_ang_vel_b_z", "turning", "lat_share", "resampled", "reward", "b")

#: bearing-bucket edges (``crab_hex_omni_command_math.telemetry_category``'s ``_FWD_MAX_DEG`` /
#: ``_SIDE_LO_DEG``, 22.5 / 67.5 deg), restated in ``sin(|bearing|)`` terms -- the SAME quantity the
#: dump's own ``lat_share`` already is (module docstring: ``lat_share`` [S, N] f32, "the centring
#: wrapper's lat_share" = ``|sin(travel_dir_w - heading_target)|`` on a MOVE slot, 0 elsewhere) -- so
#: :func:`_cat_column_mismatches` buckets it directly, with no per-row inverse trig. The ``1e-4`` slack
#: matches ``telemetry_category``'s own ``_BEARING_TOL_DEG``-derived boundary tolerance.
_SIDE_LAT_MIN = math.sin(math.radians(67.5)) - 1e-4
_FWD_BACK_LAT_MAX = math.sin(math.radians(22.5)) + 1e-4


def _cat_column_mismatches(cats: np.ndarray, stand: np.ndarray, turning: np.ndarray,
                           lat_share: np.ndarray) -> int:
    """Cross-check a parity dump's ``cat`` (the wrapper's own ``telemetry_category()`` index) against
    its OTHER, independently-recorded columns, instead of trusting ``cat`` on its own: the numeric
    recompute in :func:`parity_check` re-derives the reward FROM ``cat`` (``scale_vec[CAT[s]]``), so a
    ``cat`` that was mislabelled at the SOURCE (the wrapper's own ``telemetry_category()`` call, or a
    hand-edited dump) would still recompute to the same wrong number and pass silently.

    * ``stand``: must equal ``cat == CATEGORIES.index("stand")`` (``crab_hex_omni_stroke_rewards.py``:
      both ``stand`` and ``cat``'s STAND bucket come from the same ``cmd_term.stand_mask()`` /
      ``slot_id``, so they can never legitimately disagree).
    * ``turning``: must equal ``cat in {turn, move_turn}`` (``turning = slot_id >= SLOT_TURN``, which
      is exactly TURN or MOVE_TURN).
    * on the remaining rows (neither ``stand`` nor ``turning`` -- i.e. a MOVE slot), ``lat_share``
      buckets the SAME way ``telemetry_category`` buckets bearing: >= ``sin(67.5 deg)`` must be
      ``side``; <= ``sin(22.5 deg)`` must be ``fwd`` OR ``back`` (``lat_share`` cannot tell the two
      apart -- both bearings sit near 0 deg / 180 deg, where ``sin`` is ~0, so this only checks "one of
      the pair", never which one); anything in between must be ``diag``.

    Returns the mismatch count (0 = every row's ``cat`` is consistent with its own dump's other
    columns).
    """
    stand_i = CATEGORIES.index("stand")
    turn_i, move_turn_i = CATEGORIES.index("turn"), CATEGORIES.index("move_turn")
    side_i, fwd_i, back_i, diag_i = (CATEGORIES.index(c) for c in ("side", "fwd", "back", "diag"))
    n = int((stand != (cats == stand_i)).sum())
    n += int((turning != np.isin(cats, (turn_i, move_turn_i))).sum())
    move = ~stand & ~turning
    side_ok = lat_share >= _SIDE_LAT_MIN
    fwdback_ok = lat_share <= _FWD_BACK_LAT_MAX
    diag_ok = ~side_ok & ~fwdback_ok
    move_mismatch = (side_ok & (cats != side_i)) | (fwdback_ok & ~np.isin(cats, (fwd_i, back_i))) \
        | (diag_ok & (cats != diag_i))
    return n + int((move & move_mismatch).sum())


def parity_check(dump: dict, *, tol: float = 1e-5, params: StrokeParams | None = None) -> dict:
    """Recompute both terms from a wrapper dump (module docstring) and compare."""
    missing = [k for k in PARITY_KEYS if k not in dump]
    if missing:
        raise SystemExit(f"parity dump lacks {missing}")
    names = {f.name for f in dataclasses.fields(StrokeParams)}
    over = {k: (int if k in ("n_off", "min_loaded") else float)(np.asarray(dump[k]).item())
            for k in dump if k in names}
    params = (params or StrokeParams()).with_(**over)
    dt = float(np.asarray(dump["dt"]).item())
    turn_radius = float(np.asarray(dump["turn_radius"]).item())
    S, N = np.asarray(dump["reward"]).shape
    t = lambda k: torch.as_tensor(np.asarray(dump[k]))  # noqa: E731
    stt = SM.init_stroke_state(N, np.asarray(dump["force"]).shape[-1])
    cst = SM.init_centring_state(N)
    reset = np.asarray(dump["reset"], dtype=bool) if "reset" in dump else np.zeros((S, N), dtype=bool)
    worst = {"pay_sum": 0.0, "b": 0.0, "R": 0.0, "pay": 0.0, "o": 0.0}
    # per-category stroke plan: an armed wrapper dumps BOTH cat and cat_scale; one alone is a broken dump
    if ("cat" in dump) != ("cat_scale" in dump):
        raise SystemExit(f"parity dump carries {'cat' if 'cat' in dump else 'cat_scale'} without "
                         f"{'cat_scale' if 'cat' in dump else 'cat'}: an armed KRABBY_OMNI_STROKE_CAT_W dump has both")
    scale_vec = None
    cat_mismatches = None
    if "cat" in dump:
        scale_vec = torch.as_tensor(np.asarray(dump["cat_scale"], dtype=np.float64).astype(np.float32))
        cats = np.asarray(dump["cat"]).astype(np.int64)
        if scale_vec.shape != (len(CATEGORIES),):
            raise SystemExit(f"parity dump cat_scale has shape {tuple(scale_vec.shape)}, not ({len(CATEGORIES)},) "
                             f"-- one multiplier per {CATEGORIES}")
        if cats.shape != (S, N) or cats.min() < 0 or cats.max() >= len(CATEGORIES):
            raise SystemExit(f"parity dump cat must be [S, N] = {(S, N)} indices into {CATEGORIES}")
        CAT = torch.as_tensor(cats)
        # cat vs the dump's OWN independent stand/turning/lat_share columns (module docstring:
        # PARITY_KEYS already requires all three unconditionally, so a dump with cat but missing any
        # of them was already refused above, by the same message every other missing column gets).
        cat_mismatches = _cat_column_mismatches(
            cats, np.asarray(dump["stand"], dtype=bool), np.asarray(dump["turning"], dtype=bool),
            np.asarray(dump["lat_share"], dtype=np.float64))
    with patched_math(params):
        for s in range(S):
            if reset[s].any():
                ids = torch.as_tensor(np.flatnonzero(reset[s]))
                SM.reset_stroke_state(stt, ids)
                SM.reset_centring_state(cst, ids)
            R, stt, pay = SM.foot_stroke_step(
                t("force")[s], t("foot_pos_w")[s], t("root_pos_w")[s], t("heading")[s], t("cmd")[s],
                t("stand")[s].bool(), t("root_lin_vel_b_xy")[s], t("root_ang_vel_b_z")[s], stt, dt, turn_radius,
                t("turning")[s].bool(), t("resampled")[s].bool(), **params.stroke_kwargs())
            b, cst, o = SM.body_centring_step(
                t("foot_pos_w")[s], t("root_pos_w")[s], t("heading")[s], t("cmd")[s], t("stand")[s].bool(),
                t("root_lin_vel_b_xy")[s], cst, dt, t("lat_share")[s], t("resampled")[s].bool(),
                **params.centring_kwargs())
            if scale_vec is not None:
                R = R * scale_vec[CAT[s]]          # the wrapper's own float32 op; pay stays unscaled
            ref_R = np.asarray(dump["reward"][s], dtype=np.float64)
            worst["R"] = max(worst["R"], float(np.max(np.abs(R.double().numpy() - ref_R))))
            worst["pay_sum"] = max(worst["pay_sum"], float(np.max(np.abs(R.double().numpy() - ref_R))) * dt)
            worst["b"] = max(worst["b"], float(np.max(np.abs(b.double().numpy() - np.asarray(dump["b"][s])))))
            if "pay" in dump:
                worst["pay"] = max(worst["pay"], float(np.max(np.abs(pay.double().numpy() - np.asarray(dump["pay"][s])))))
            if "o" in dump:
                worst["o"] = max(worst["o"], float(np.max(np.abs(o.double().numpy() - np.asarray(dump["o"][s])))))
    checked = ["pay_sum", "b"] + [k for k in ("pay", "o") if k in dump]
    ok = all(worst[k] < tol for k in checked) and (cat_mismatches in (None, 0))
    out = {"ok": ok, "tol": tol, "steps": S, "envs": N, "max_abs": worst, "checked": checked}
    if scale_vec is not None:
        out["cat_scale"] = dict(zip(CATEGORIES, scale_vec.double().tolist()))
        out["cat_steps"] = {c: int((cats == k).sum()) for k, c in enumerate(CATEGORIES)}
        out["cat_mismatches"] = cat_mismatches
    return out


# ============================================================================ the gate (plan step 7)
DEFAULT_GRID = {"force_hold": [15.0, 25.0], "n_off": [1, 3], "rho0": [-0.1, 0.1], "drho": [-0.1, 0.1],
                "power": [1.5, 2.0],
                # the plan's R9 says "neighbours of EVERY parameter": the stroke parameters the design
                # panel's grid predates (the math revision's progress guard) or left out
                "c_max": [-0.25, 0.25], "prog_lo": [-0.05, 0.05], "prog_width": [-0.05, 0.05],
                "prog_tau": [3.0, 5.0], "force_open": [40.0, 60.0], "min_loaded": [4, 6]}
"""R9's neighbour grid. The design panel's: F_off 15/25 N, N_off 1-3, rho0 +-0.1, drho +-0.1, p 1.5/2;
plus (plan: "every parameter") c_max +-0.25, prog_lo +-0.05, prog_width +-0.05, prog_tau 3/5 s,
F_on 40/60 N, min loaded steps 4/6. A value for a :data:`_OFFSET_PARAMS` entry is an OFFSET from the
base; every other value is absolute."""
_OFFSET_PARAMS = ("rho0", "drho", "c_max", "prog_lo", "prog_width")

R1_MIN_KEEP = 0.80
R2_SIDE_MAX_KEEP = 0.40
R2_SLIDE_RHO = 0.8
R2_SLIDE_MAX_KEEP = 0.25
R2_SIDE_VS_SIDEWAYS = 0.2
R3_SHORT_B_M = 0.02
R3_SHORT_MAX_SHARE = 0.02
R3_MAX_RHO = 0.2
R4_MAX_SHARE = 0.10
R4_MIN_FWD_SHARE = 0.05
R4_CATEGORIES = ("fwd", "back", "turn", "move_turn")
R5_TERM_ALONE_MAX = 0.5
R6_MAX_INCOME_SHIFT = 0.05
R6_MAX_LIFTOFF_SHIFT = 0.02
R7_MAX_SHIFT_PP = 5.0
R7_MAX_ABSORBED = 0.15
R8_MAX_HAND_GAP = 0.10
P1_SIDEWAYS_MIN = 0.85
P1_FORK_BAND = (0.15, 0.45)
P1_ZERO_TOL = 1e-4
P2_BAND = (0.8, 1.25)
P2_HEALTHY_CAP = 0.15             # design panel's line: REPORTED only (the plan gives P no healthy cap)
P2_DIAG_CAP = 0.10
P3_STATUE_RATIO = 1.10
P3_CREEPER_MAX = 0.10


def _verdict(ok: bool | None) -> str:
    return "N/A" if ok is None else ("PASS" if ok else "FAIL")


@dataclass
class GateRun:
    name: str
    role: str                  # healthy | sideways | posture | degenerate | v4
    st: Stamp
    rp: Replay | None = None
    vals: dict | None = None
    prims: list | None = None
    ctx: np.ndarray | None = None


def _load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    for k in ("healthy", "degenerate"):
        if not cfg.get(k):
            raise SystemExit(f"gate config needs a non-empty {k!r} map")
    cfg.setdefault("primary", next(iter(cfg["healthy"])))
    if cfg["primary"] not in cfg["healthy"]:
        raise SystemExit(f"primary {cfg['primary']!r} is not a healthy run")
    _check_cat_config(cfg)
    return cfg


def _spec_keys(spec) -> list[str]:
    """The category names a ``stroke_cat_w`` spec (string or dict) lists EXPLICITLY."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        return [str(k) for k in spec]
    return [e.partition(":")[0].strip() for e in str(spec).split(",") if e.strip()]


def _check_cat_config(cfg: dict) -> None:
    """The per-category stroke plan's gate keys (module docstring): ``stroke_cat_w`` parses with the
    knob's own parser; ``dose_targets`` names knob categories with shares in (0, 1]; each ``tie``
    group names knob categories, a category in at most one group; and no category is BOTH solved
    (targeted, or tied to a targeted one) and listed in ``stroke_cat_w`` -- a silent override either
    way. Neither key: nothing to check (and nothing in the config changes)."""
    parse_cat_w(cfg.get("stroke_cat_w"))
    for key in ("tie", "dose_hand_min"):
        if key in cfg and "dose_targets" not in cfg:
            raise SystemExit(f"gate config: {key!r} only applies with 'dose_targets'")
    targets = cfg.get("dose_targets")
    if targets is None:
        return
    if not isinstance(targets, dict) or not targets:
        raise SystemExit(f"gate config: dose_targets must be a non-empty {{category: share}} map, got {targets!r}")
    # pooled vs lower-hand moves turn's dose 1-9 % (W / t3): the config must say which, never a default
    if not isinstance(cfg.get("dose_hand_min"), bool):
        raise SystemExit("gate config: dose_targets needs an explicit dose_hand_min (true: side/turn/move_turn "
                         "bound by their lower hand -- the plan's 2026-09-27 tables; false: the hands pooled)")
    for c, v in targets.items():
        if c not in SK.STROKE_CAT_KEYS:
            raise SystemExit(f"gate config: dose_targets names {c!r}, not one of {SK.STROKE_CAT_KEYS}")
        if not (isinstance(v, (int, float)) and 0.0 < float(v) <= 1.0):
            raise SystemExit(f"gate config: dose_targets.{c} must be a share in (0, 1], got {v!r}")
    tie = cfg.get("tie", [list(g) for g in DEFAULT_DOSE_TIE])
    seen: set = set()
    for group in tie:
        if not isinstance(group, (list, tuple)) or len(group) < 2:
            raise SystemExit(f"gate config: each tie group lists >= 2 categories, got {group!r}")
        for c in group:
            if c not in SK.STROKE_CAT_KEYS or c in seen:
                raise SystemExit(f"gate config: tie member {c!r} is unknown or in two groups ({tie!r})")
            seen.add(c)
    solved = set(targets) | {c for g in tie if set(g) & set(targets) for c in g}
    clash = sorted(solved & set(_spec_keys(cfg.get("stroke_cat_w"))))
    if clash:
        raise SystemExit(f"gate config: {clash} are solved by dose_targets (or their tie) and also listed in "
                         "stroke_cat_w -- list only the categories the solver leaves alone")


def _stance_sel(gr: GateRun, cats: Iterable[str], rp: Replay | None = None) -> np.ndarray:
    rp = rp or gr.rp
    sd = rp.stances
    if not sd["env"].size:
        return np.zeros(0, dtype=bool)
    catv = np.array([gr.st.episodes[i]["category"] for i in range(gr.st.N)])
    sel = gr.st.own[sd["env"], sd["step"]] & rp.gate[sd["env"], sd["step"]]
    return sel & np.isin(catv[sd["env"]], list(cats))


def kept_fraction(gr: GateRun, cats: Iterable[str], rp: Replay | None = None, extra=None) -> float | None:
    rp = rp or gr.rp
    sd = rp.stances
    k = _stance_sel(gr, cats, rp)
    if extra is not None:
        k = k & extra(sd)
    U = sd["ungated"][k].sum()
    return float(sd["pay"][k & sd["payable"]].sum() / U) if U > 0 else None


def cat_income(gr: GateRun, cat: str, rp: Replay | None = None, hand: str | None = None,
               term: str = "stroke") -> float | None:
    """Pooled term income/s at w = 1 over the gated moving own-hold steps of ``cat``."""
    rp = rp or gr.rp
    m_ep = np.array([e["category"] == cat and (hand is None or e["hand"] == hand) for e in gr.st.episodes])
    g = gr.st.own & m_ep[:, None] & (rp.gate if cat != "stand" else True)
    n = g.sum()
    if n == 0:
        return None
    x = rp.pay.sum(axis=-1) if term == "stroke" else rp.b * gr.st.dt
    return float(x[g].sum() / (n * gr.st.dt))


def _r123(runs: dict[str, GateRun], replays: dict[str, Replay], params: StrokeParams) -> dict:
    """R1-R3 on the healthy (omni + sideways) runs with ``replays`` (one per run)."""
    out: dict[str, Any] = {}
    healthy = {n: g for n, g in runs.items() if g.role == "healthy"}
    sideways = {n: g for n, g in runs.items() if g.role == "sideways"}
    # ---- R1
    r1 = {}
    for n, g in healthy.items():
        for c in ("fwd", "back", "turn"):
            r1[f"{n}.{c}"] = kept_fraction(g, (c,), replays[n])
    for n, g in sideways.items():
        r1[f"{n}.side"] = kept_fraction(g, ("side",), replays[n])
    ok1 = all(v is not None and v >= R1_MIN_KEEP for v in r1.values()) if r1 else None
    out["R1"] = {"pass": ok1, "kept": r1, "rule": f"kept >= {R1_MIN_KEEP}"}
    # ---- R2
    side_keep = {n: kept_fraction(g, ("side",), replays[n]) for n, g in healthy.items()}
    slide_num = slide_den = 0.0
    for n, g in {**healthy, **sideways}.items():
        rp = replays[n]
        sd = rp.stances
        k = _stance_sel(g, MOVING, rp) & (sd["rho"] > R2_SLIDE_RHO) if sd["env"].size else np.zeros(0, bool)
        slide_den += sd["ungated"][k].sum() if k.size else 0.0
        slide_num += sd["pay"][k & sd["payable"]].sum() if k.size else 0.0
    slide_keep = float(slide_num / slide_den) if slide_den > 0 else None
    side_inc = {n: cat_income(g, "side", replays[n]) for n, g in healthy.items()}
    sw_inc = {n: cat_income(g, "side", replays[n]) for n, g in sideways.items()}
    ratios = {f"{a}/{b}": (side_inc[a] / sw_inc[b]) if side_inc[a] is not None and sw_inc[b] else None
              for a in side_inc for b in sw_inc}
    ok2a = all(v is not None and v <= R2_SIDE_MAX_KEEP for v in side_keep.values())
    ok2b = slide_keep is None or slide_keep <= R2_SLIDE_MAX_KEEP
    ok2c = all(v is not None and v <= R2_SIDE_VS_SIDEWAYS for v in ratios.values()) if ratios else None
    out["R2"] = {"pass": bool(ok2a and ok2b and (ok2c is not False)), "side_kept": side_keep,
                 "slide_kept": slide_keep, "side_income": side_inc, "sideways_income": sw_inc,
                 "side_vs_sideways": ratios,
                 "clauses": {"side_kept<=0.40": ok2a, "slide_kept<=0.25": ok2b, "side<=0.2xsideways": ok2c}}
    # ---- R3
    short_share = {}
    rho_sw = {}
    for n, g in {**healthy, **sideways}.items():
        rp = replays[n]
        sd = rp.stances
        for c in MOVING:
            k = _stance_sel(g, (c,), rp)
            if not k.any():
                continue
            tot = sd["pay"][k].sum()
            short = sd["pay"][k & ((sd["B"] < R3_SHORT_B_M) | sd["short_n"])].sum()
            short_share[f"{n}.{c}"] = float(short / tot) if tot > 0 else None
        if g.role != "healthy":
            continue
        vals = g.vals if rp is g.rp else episode_values(g.st, rp)
        for c in MOVING:
            idx = [i for i, e in enumerate(g.st.episodes) if e["category"] == c]
            inc, sw = [], []
            for i in idx:
                j = g.st.json_eps[i] or {}
                q = ((j.get("holds") or {}).get(g.st.episodes[i]["label"]) or {}).get("quality") or {}
                inc.append(vals["stroke_per_s"][i])
                sw.append(q.get("contact_switch_hz"))
            rho, nn = RMT.spearman([float(x) for x in inc], [None if x is None else float(x) for x in sw])
            rho_sw[f"{n}.{c}"] = {"rho": rho, "n": nn}
    ok3a = all(v is None or v <= R3_SHORT_MAX_SHARE for v in short_share.values())
    ok3b = all(v["rho"] is None or v["rho"] <= R3_MAX_RHO for v in rho_sw.values())
    v4 = {n: g for n, g in runs.items() if g.role == "v4"}
    ok3c = None
    v4_rows = {}
    if v4:
        ok3c = True
        for n, g in v4.items():
            for c in MOVING:
                vi = cat_income(g, c, replays.get(n, g.rp))
                hi = [cat_income(h, c, replays[hn]) for hn, h in healthy.items()]
                hi = [x for x in hi if x is not None]
                if vi is None or not hi:
                    continue
                v4_rows[f"{n}.{c}"] = {"v4": vi, "healthy_min": min(hi)}
                ok3c &= vi <= min(hi)
    out["R3"] = {"pass": bool(ok3a and ok3b and (ok3c is not False)), "short_share": short_share,
                 "spearman_switch": rho_sw, "v4": v4_rows,
                 "clauses": {"short<=2%": ok3a, "spearman<=+0.2": ok3b, "v4<=healthy": ok3c}}
    return out


def _dose(gr: GateRun, rp: Replay, econ, cats=R4_CATEGORIES, w_hi: float = 10.0,
          scale: np.ndarray | None = None) -> dict:
    """R4: the largest w with every ``cats`` share of delivered pay / tracking <= 10 %. ``scale``
    ([N, T], :func:`scale_rows`): the per-category table the swept ``w`` (= ``w_S``) multiplies."""
    st = gr.st
    track = tracking_rate(st, econ)
    x1 = rp.pay.sum(axis=-1).astype(np.float64)
    if scale is not None:
        x1 = x1 * scale
    base = st.dt * gr.ctx

    def shares(w):
        out = {}
        for c in CATEGORIES:
            m_ep = np.array([e["category"] == c for e in st.episodes])
            g = st.own & m_ep[:, None] & (rp.gate if c != "stand" else True)
            if not g.any():
                continue
            den = track[g].sum() * st.dt
            num = delivered(w * x1, base)[g].sum()
            out[c] = float(num / den) if den > 0 else None
        return out

    def worst(w):
        s = shares(w)
        return max((s.get(c) or 0.0) for c in cats)

    lo, hi = 0.0, w_hi
    if worst(hi) <= R4_MAX_SHARE:
        w = hi
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if worst(mid) <= R4_MAX_SHARE else (lo, mid)
        w = lo
    s = shares(w)
    binding = max(cats, key=lambda c: s.get(c) or 0.0)
    return {"w_S": w, "shares": s, "binding": binding}


#: The per-category solver's mirror tie (``KRABBY_OMNI_STROKE_CAT_W`` refuses ``fwd != back``).
DEFAULT_DOSE_TIE = (("fwd", "back"),)


def _dose_per_cat(gr: GateRun, rp: Replay, econ, targets: dict[str, float], w_hi: float = 10.0,
                  hand_min: bool = False) -> dict:
    """R4, per category (per-category stroke plan, 2026-09-27): for each ``targets`` category ``c``,
    the largest ``w_c`` whose DELIVERED stroke pay over ``c``'s gated own-hold steps is <=
    ``targets[c]`` x ``c``'s tracking income. A category's steps are its own (an own hold is one
    category) and :func:`delivered` is elementwise, so each category is bisected alone and the answer
    is exact -- no cross-category coupling. ``reached`` is False when even ``w_hi`` stays under the
    target (``w`` is then ``w_hi``); ``w`` is None when the stamp has no gated step of ``c``.

    ``hand_min`` (config ``dose_hand_min``): a PAIRED category (side, turn, move_turn) is instead
    bound by its LOWER hand -- each hand bisected on its own share, ``w_c`` the smaller -- so neither
    hand exceeds the target (the weight cannot tell the hands apart). Default: the hands pooled."""
    st = gr.st
    track = tracking_rate(st, econ)
    x1 = rp.pay.sum(axis=-1).astype(np.float64)
    base = st.dt * gr.ctx

    def solve(g, target):
        den = float(track[g].sum() * st.dt) if g.any() else 0.0
        if den <= 0:
            return {"w": None, "share": None, "target": target, "reached": False, "n_steps": int(g.sum())}
        xs, bs = x1[g], base[g]

        def share(w):
            return float(delivered(w * xs, bs).sum() / den)

        lo, hi = 0.0, w_hi
        if share(hi) <= target:
            w, reached = hi, False
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if share(mid) <= target else (lo, mid)
            w, reached = lo, True
        return {"w": w, "share": share(w), "target": target, "reached": reached, "n_steps": int(g.sum())}

    out: dict[str, dict] = {}
    for c, target in targets.items():
        m_ep = np.array([e["category"] == c for e in st.episodes])
        row = solve(st.own & m_ep[:, None] & rp.gate, target)
        if hand_min and c in PAIRED:
            hands = {h: solve(st.own & (m_ep & np.array([str(e["hand"]) == h for e in st.episodes]))[:, None]
                              & rp.gate, target) for h in EC.HANDS}
            got = {h: r for h, r in hands.items() if r["w"] is not None}
            if got:
                low = min(got, key=lambda h: got[h]["w"])
                row = {**got[low], "pooled": row, "hands": hands, "bound_by_hand": low}
        out[c] = row
    return out


def _solve_cat_table(per_run: dict[str, dict], targets: dict[str, float], tie, base_table: dict | None) -> dict:
    """The per-category table from :func:`_dose_per_cat` on every dose run: each category's ``w_c`` is
    its minimum over the runs; each ``tie`` group is then bound by its LOWER-weight targeted member
    (every member, targeted or not, takes that weight -- the mirror tie ``fwd == back``); ``w_S`` =
    ``w_fwd`` (``KRABBY_OMNI_STROKE_W`` carries it), ``m_c = w_c / w_S`` for every solved category,
    ``base_table``'s multiplier (``stroke_cat_w``, default 1) for every other. The table is re-parsed by
    the knob's own parser, so a table training would refuse is refused here."""
    w: dict[str, float] = {}
    for c in targets:
        vals = [r[c]["w"] for r in per_run.values()]
        if any(v is None for v in vals):
            missing = [n for n, r in per_run.items() if r[c]["w"] is None]
            raise SystemExit(f"dose_targets.{c}: no gated {c} step on dose run(s) {missing} -- nothing to dose")
        w[c] = min(vals)
    bound = {}
    for group in tie:
        solved = [c for c in group if c in w]
        if not solved:
            continue
        lowest = min(solved, key=lambda c: w[c])
        for c in group:
            w[c] = w[lowest]
        bound[",".join(group)] = lowest
    if "fwd" not in w:
        raise SystemExit("dose_targets must fix fwd (directly or through its tie): KRABBY_OMNI_STROKE_W carries w_fwd")
    w_S = w["fwd"]
    if not w_S > 0.0:
        raise SystemExit(f"dose_targets: the solved w_fwd is {w_S!r}; the multipliers m_c = w_c / w_fwd are undefined")
    base = cat_scale_vector(base_table)
    table = {c: (w[c] / w_S if c in w else float(base[CATEGORIES.index(c)])) for c in SK.STROKE_CAT_KEYS}
    table = parse_cat_w(table)
    return {"w_S": w_S, "w_c": w, "table": table, "tie_bound_by": bound,
            "w_eff": {c: w_S * m for c, m in table.items()}, "spec": format_cat_w(table)}


def _hand_gap(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or (a + b) <= 0:
        return None
    return abs(a - b) / (0.5 * (a + b))


def run_gate(cfg: dict, out_dir: Path, *, log=print) -> dict:
    """Plan step 7's R1-R9 / P1-P4 (module docstring); writes ``replay_gate.json`` and ``REPLAY_GATE.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = parse_params([f"{k}={v}" for k, v in (cfg.get("params") or {}).items()])
    limit = cfg.get("limit")
    econ_yaml = cfg.get("economy_env_yaml")
    runs: dict[str, GateRun] = {}
    roles = (("healthy", "healthy"), ("sideways", "sideways"), ("posture", "posture"),
             ("degenerate", "degenerate"), ("v4", "v4"))
    for key, role in roles:
        for name, path in (cfg.get(key) or {}).items():
            log(f"[gate] loading {role} {name}: {path}")
            st = load_stamp(path, limit=limit)
            if role in ("healthy", "posture", "degenerate") and st.kind != "omni":
                raise SystemExit(f"{name}: a {role} run must be an omni suite stamp")
            runs[name] = GateRun(name=name, role=role, st=st)
    primary = runs[cfg["primary"]]
    if econ_yaml is None:
        ck = primary.st.meta.get("checkpoint")
        econ_yaml = Path(ck).parent / "params" / "env.yaml" if ck else W_ENV_YAML
    econ, econ_info = load_economy(econ_yaml)
    for name, g in runs.items():
        log(f"[gate] replaying {name}")
        g.rp = replay(g.st, base)
        g.vals = episode_values(g.st, g.rp, econ)
        if g.st.kind == "omni":
            clip = cfg.get("action_clip", {}).get(name) if isinstance(cfg.get("action_clip"), dict) else None
            if clip is not None:
                g.st.action_clip = float(clip)
            elif not g.st.meta.get("checkpoint"):
                g.st.action_clip = REC.action_clip_from_agent_yaml(REC.head_agent_yaml(primary.st.meta))
                g.st.notes.append("no checkpoint: the primary head's clip_actions is used for the economy")
            g.prims = econ_primitives(g.st, econ, econ_info)
            g.ctx = context_rows(g.st, g.prims, econ)
    rep: dict[str, Any] = {"config": cfg, "params": dataclasses.asdict(base), "economy": econ_info,
                           "runs": {n: {"role": g.role, "run_dir": g.st.run_dir, "kind": g.st.kind, "N": g.st.N,
                                        "notes": g.st.notes, "heading_guard_max": g.st.guard_max}
                                    for n, g in runs.items()},
                           "dropped": cfg.get("dropped") or {}}
    healthy = {n: g for n, g in runs.items() if g.role == "healthy"}
    sideways = {n: g for n, g in runs.items() if g.role == "sideways"}
    degenerate = {n: g for n, g in runs.items() if g.role == "degenerate"}
    replays = {n: g.rp for n, g in runs.items()}

    crit: dict[str, Any] = {}
    crit.update(_r123(runs, replays, base))

    # ---- R4 dose (primary, own-hold over gated moving steps)
    # Per-category stroke plan (2026-09-27): `dose_targets` SOLVES the table per category
    # (_dose_per_cat, tie groups bound by their lower-weight member); `stroke_cat_w` alone is a FIXED
    # table the single swept w_S multiplies. Neither key: `table` is None and every line below runs
    # exactly as before the table existed.
    dose_runs = cfg.get("dose_runs") or [cfg["primary"]]
    table = parse_cat_w(cfg.get("stroke_cat_w"))
    cat_solve = None
    if cfg.get("dose_targets") is not None:
        targets = {c: float(v) for c, v in cfg["dose_targets"].items()}
        tie = [tuple(g) for g in cfg.get("tie", DEFAULT_DOSE_TIE)]
        hand_min = cfg["dose_hand_min"]
        per_cat = {n: _dose_per_cat(runs[n], runs[n].rp, econ, targets, hand_min=hand_min) for n in dose_runs}
        cat_solve = _solve_cat_table(per_cat, targets, tie, table)
        cat_solve.update(targets=targets, tie=[list(g) for g in tie], per_run=per_cat, hand_min=hand_min)
        table = cat_solve["table"]
        w_S = cat_solve["w_S"]
        doses = {n: {c: d["w"] for c, d in pc.items()} for n, pc in per_cat.items()}
    else:
        doses = {n: _dose(runs[n], runs[n].rp, econ,
                          scale=(scale_rows(runs[n].st, table) if table is not None else None))
                 for n in dose_runs}
        w_S = min(d["w_S"] for d in doses.values())
    scales = {n: (scale_rows(g.st, table) if table is not None else None) for n, g in runs.items()}
    stand_pay = {n: float(g.rp.pay.sum(axis=-1)[g.st.own & np.array(
        [e["category"] == "stand" for e in g.st.episodes])[:, None]].sum()) for n, g in healthy.items()}
    at_w = {n: _dose_shares_at(runs[n], econ, w_S, scale=scales[n]) for n in healthy}
    fwd_ok = all((at_w[n].get("fwd") or 0.0) >= R4_MIN_FWD_SHARE for n in dose_runs)
    turn_ok = all((cat_income(g, "turn") or 0.0) > 0 for g in healthy.values())
    mt_ok = all((cat_income(g, "move_turn") or 0.0) > 0 for g in healthy.values())
    stand_ok = all(v == 0.0 for v in stand_pay.values())
    sw_share = {}
    for n, g in sideways.items():
        track = tracking_rate(g.st, econ)
        m = g.st.own & g.rp.gate
        den = track[m].sum() * g.st.dt
        # float32 like the unscaled sum (and the wrapper's own float32 multiply): an all-ones table
        # then reproduces the untabled number bit for bit
        x = g.rp.pay.sum(axis=-1) if scales[n] is None else g.rp.pay.sum(axis=-1) * scales[n].astype(np.float32)
        sw_share[n] = float(w_S * x[m].sum() / den) if den > 0 else None
    crit["R4"] = {"pass": bool(fwd_ok and turn_ok and mt_ok and stand_ok), "w_S": w_S, "per_run": doses,
                  "shares_at_w_S": at_w, "stand_pay": stand_pay, "sideways_side_share_nominal": sw_share,
                  "clauses": {"fwd>=5%": fwd_ok, "stand==0": stand_ok, "turn>0": turn_ok, "move_turn>0": mt_ok},
                  "note": "diag and side are reported, not gated (R4 names fwd, back, turn, move_turn); "
                          "the sideways head is not priced in W's omni economy (nominal share)."}
    if table is not None:
        crit["R4"]["stroke_cat_w"] = {"table": table, "spec": format_cat_w(table),
                                      "w_eff": {c: w_S * m for c, m in table.items()}}
    if cat_solve is not None:
        # every targeted category must have reached its share below the bisection cap; the untied
        # member of a tie group sits BELOW its target by construction (reported, not a failure)
        reached = {c: all(pc[c]["reached"] for pc in cat_solve["per_run"].values()) for c in cat_solve["targets"]}
        crit["R4"]["clauses"]["dose targets reached"] = all(reached.values())
        crit["R4"]["pass"] = crit["R4"]["pass"] and all(reached.values())
        crit["R4"]["per_category_solve"] = {**cat_solve, "reached": reached}
        crit["R4"]["note"] += (" Per-category solve: each dose_targets category bisected on its own delivered "
                               "share; tie groups bound by their lower-weight member; shares_at_w_S are at "
                               "the solved table.")

    # ---- R5 ordering (base vs base + w_S stroke) + term-alone
    for n, g in runs.items():
        if g.prims is not None:
            _attach_rows(g, scale=scales[n])
    econ_base = econ
    econ_s = dataclasses.replace(econ, extra=econ.extra + (("stroke", w_S),))
    heads_h = {n: _scored(g) for n, g in runs.items() if g.role in ("healthy", "posture")}
    heads_d = {n: _scored(g) for n, g in degenerate.items()}
    with extra_terms():
        rg = REC.rg1_ordering(heads_h, heads_d, econ_base, econ_s)
    alone = {}
    ok5b = True
    for dn, dg in degenerate.items():
        for hn, hg in healthy.items():
            for c in MOVING:
                di, hi = cat_income(dg, c), cat_income(hg, c)
                ratio = (di / hi) if (di is not None and hi) else None
                alone[f"{dn}/{hn}.{c}"] = {"degenerate": di, "healthy": hi, "ratio": ratio,
                                          "gated": c in TRANSLATION}
                if c in TRANSLATION and ratio is not None and ratio > R5_TERM_ALONE_MAX:
                    ok5b = False
    crit["R5"] = {"pass": bool((rg["pass"] is not False) and ok5b), "rg1": _rg_summary(rg), "term_alone": alone,
                  "clauses": {"no_flip": rg["pass"], "term_alone<=0.5x (translation)": ok5b}}

    # ---- R6 stale data
    r6 = {}
    ok6 = True
    for n, g in {**healthy, **sideways}.items():
        T = g.st.arr["segment_id"].shape[1]
        rpd = replay(g.st, base, dup=dup_pattern(T))
        for c in MOVING:
            a, b_ = cat_income(g, c), cat_income(g, c, rpd)
            if a is None or b_ is None:
                continue
            ka = _stance_sel(g, (c,)).sum()
            kb = _stance_sel(g, (c,), rpd).sum()
            di = abs(b_ / a - 1.0) if a > 0 else None
            dl = abs(kb / ka - 1.0) if ka > 0 else None
            r6[f"{n}.{c}"] = {"income": a, "income_dup": b_, "income_shift": di, "liftoffs": int(ka),
                              "liftoffs_dup": int(kb), "liftoff_shift": dl}
            ok6 &= (di is None or di <= R6_MAX_INCOME_SHIFT) and (dl is None or dl <= R6_MAX_LIFTOFF_SHIFT)
        if g.role == "sideways":
            tp = RMT.TrainParams()
            f32 = f32_stale_pattern(T, physics_dt=tp.physics_dt, decimation=tp.decimation,
                                    update_period=tp.sensor_update_period)
            rpf = replay(g.st, base, dup=f32)
            a, b_ = cat_income(g, "side"), cat_income(g, "side", rpf)
            r6[f"{n}.side.f32_replica"] = {"income": a, "income_dup": b_, "stale_steps": int(f32.sum()),
                                          "income_shift": (abs(b_ / a - 1.0) if a else None), "gated": False}
    crit["R6"] = {"pass": bool(ok6), "rows": r6, "pattern": "force repeats the previous reading on s % 5 == 4"}

    # ---- R7 clip (stroke at w_S)
    r7 = {}
    ok7 = True
    for n, g in healthy.items():
        rows = income_table(g.st, g.rp, w_stroke=w_S, w_centring=0.0, econ=econ, ctx=g.ctx, cat_w=table)
        for c in MOVING:
            v = rows.get(c) or {}
            if not v.get("n_steps"):
                continue
            r7[f"{n}.{c}"] = {"clip_base": v["clip_bind_base"], "clip_with": v["clip_bind_with"],
                              "shift_pp": v["clip_shift_pp"], "absorbed": v["stroke_absorbed"]}
            ok7 &= abs(v["clip_shift_pp"]) <= R7_MAX_SHIFT_PP and (v["stroke_absorbed"] is None
                                                                    or v["stroke_absorbed"] <= R7_MAX_ABSORBED)
    crit["R7"] = {"pass": bool(ok7), "rows": r7}

    # ---- R8 hands
    r8 = {}
    ok8 = True
    for n, g in healthy.items():
        for c in PAIRED:
            a, b_ = cat_income(g, c, hand="l"), cat_income(g, c, hand="r")
            gap = _hand_gap(a, b_)
            r8[f"{n}.{c}"] = {"l": a, "r": b_, "gap": gap}
            ok8 &= gap is None or gap <= R8_MAX_HAND_GAP
    crit["R8"] = {"pass": bool(ok8), "rows": r8}

    # ---- R9 neighbours
    grid = cfg.get("grid") or DEFAULT_GRID
    r9 = {}
    ok9 = True
    for pname, values in grid.items():
        for v in values:
            val = getattr(base, pname) + v if pname in _OFFSET_PARAMS else v
            if val == getattr(base, pname):
                continue
            prm = base.with_(**{pname: (int(val) if pname in ("n_off", "min_loaded") else float(val))})
            log(f"[gate] R9 neighbour {pname}={val}")
            nrp = {n: replay(g.st, prm) for n, g in runs.items() if g.role in ("healthy", "sideways", "v4")}
            sub = {n: g for n, g in runs.items() if n in nrp}
            res_ = _r123(sub, nrp, prm)
            r9[f"{pname}={val:g}"] = {k: {"pass": v_["pass"], **({"clauses": v_["clauses"]} if "clauses" in v_ else {}),
                                          **({"kept": v_["kept"]} if k == "R1" else {}),
                                          **({"side_kept": v_["side_kept"], "slide_kept": v_["slide_kept"]}
                                             if k == "R2" else {})}
                                      for k, v_ in res_.items()}
            ok9 &= all(v_["pass"] for v_ in res_.values())
    crit["R9"] = {"pass": bool(ok9), "grid": grid, "rows": r9}

    # ---- P1-P4 (centring)
    crit.update(_p_criteria(cfg, runs, econ, w_S))
    for k, v in crit.items():
        v["headline"] = _headline(k, v)
    rep["criteria"] = crit
    rep["w_S"] = w_S
    rep["w_P"] = crit["P2"].get("w_P")
    if table is not None:
        # the knob values a training run arms: KRABBY_OMNI_STROKE_W = w_S, KRABBY_OMNI_STROKE_CAT_W = spec
        rep["stroke_cat_w"] = crit["R4"]["stroke_cat_w"]
    (out_dir / "replay_gate.json").write_text(json.dumps(_jsonable(rep), indent=2, sort_keys=True))
    md = render_gate_md(rep)
    (out_dir / "REPLAY_GATE.md").write_text(md)
    log(md)
    return rep


def _dose_shares_at(gr: GateRun, econ, w: float, scale: np.ndarray | None = None) -> dict:
    """Every category's delivered stroke share of its tracking income at ``w`` (x ``scale``, the
    per-category table's [N, T] multipliers, when given)."""
    st, rp = gr.st, gr.rp
    track = tracking_rate(st, econ)
    x1 = rp.pay.sum(axis=-1).astype(np.float64)
    if scale is not None:
        x1 = x1 * scale
    x = w * x1
    d = delivered(x, st.dt * gr.ctx) if gr.ctx is not None else x
    out = {}
    for c in CATEGORIES:
        m_ep = np.array([e["category"] == c for e in st.episodes])
        g = st.own & m_ep[:, None] & (rp.gate if c != "stand" else True)
        den = track[g].sum() * st.dt
        if g.any() and den > 0:
            out[c] = float(d[g].sum() / den)
    return out


def _attach_rows(g: GateRun, scale: np.ndarray | None = None) -> None:
    """Put the stroke R and the centring b of env step ``s`` on primitive row ``s + 1``. ``scale``
    (the per-category table's [N, T] multipliers): the stroke row is ``R * scale``, so R5's
    ``("stroke", w_S)`` prices the WEIGHTED term, ``w_S * m_c * R``."""
    stroke = g.rp.R if scale is None else g.rp.R * scale
    for i, p in enumerate(g.prims):
        n = p["T"]
        for key, src in (("stroke", stroke), ("centring", g.rp.b)):
            a = np.zeros(n)
            a[1:] = src[i, :n - 1]
            p[key] = a


def _scored(g: GateRun) -> list[dict]:
    return [{"env_index": e["env_index"], "category": e["category"], "hold": e["label"],
             "hold_idx": e["hold_ids"][0], "p": p} for e, p in zip(g.st.episodes, g.prims)]


def _rg_summary(rg: dict) -> dict:
    rows = []
    for r in rg["rows"] + rg["held_reversed"]:
        rows.append({"healthy": r["healthy"], "degenerate": r["degenerate"], "category": r["category"],
                     "base": r["held"], "with": r["free"], "ratio_base": r.get("ratio_held"),
                     "ratio_with": r.get("ratio_free"), "floor": r.get("ratio_floor"), "pass": r["pass"],
                     "note": r.get("note")})
    return {"pass": rg["pass"], "rows": rows}


def _p_criteria(cfg: dict, runs: dict[str, GateRun], econ, w_S: float) -> dict:
    healthy = {n: g for n, g in runs.items() if g.role == "healthy"}
    sideways = {n: g for n, g in runs.items() if g.role == "sideways"}
    degenerate = {n: g for n, g in runs.items() if g.role == "degenerate"}
    primary = runs[cfg["primary"]]
    out: dict[str, Any] = {}

    def cmean(g, c, hand=None):
        v = cat_income(g, c, hand=hand, term="centring")
        return v

    # P1
    sw = {n: cmean(g, "side") for n, g in sideways.items()}
    fork = {n: cmean(g, "side") for n, g in healthy.items()}
    zeros = {f"{n}.{c}": cmean(g, c) for n, g in healthy.items() for c in ("fwd", "back", "stand", "turn")}
    ok_sw = all(v is not None and v >= P1_SIDEWAYS_MIN for v in sw.values()) if sw else None
    pf = fork[cfg["primary"]]
    ok_fork = pf is not None and P1_FORK_BAND[0] <= pf <= P1_FORK_BAND[1]
    ok_zero = all(v is None or abs(v) <= P1_ZERO_TOL for v in zeros.values())
    out["P1"] = {"pass": bool((ok_sw is not False) and ok_fork and ok_zero), "sideways_side": sw,
                 "healthy_side": fork, "zero_rows": zeros,
                 "clauses": {"sideways>=0.85": ok_sw, f"{cfg['primary']} side in [0.15,0.45]": ok_fork,
                             "fwd/back/stand/turn==0": ok_zero}}
    # P2 dose by prize parity with S
    s_sw = [cat_income(g, "side") for g in sideways.values()]
    p_sw = [cmean(g, "side") for g in sideways.values()]
    s_om, p_om = cat_income(primary, "side"), cmean(primary, "side")
    w_P = None
    row: dict[str, Any] = {}
    if s_sw and p_sw and None not in s_sw + p_sw and s_om is not None and p_om is not None:
        dS = float(np.mean(s_sw)) - s_om
        dP = float(np.mean(p_sw)) - p_om
        row.update({"stroke_prize_per_w": dS, "centring_prize_per_w": dP})
        if dS > 0 and dP > 0:
            w_P = w_S * dS / dP
            row["w_P_band"] = [P2_BAND[0] * w_P, P2_BAND[1] * w_P]
    ok2 = w_P is not None
    shares = {}
    if w_P is not None:
        for n, g in healthy.items():
            rows = income_table(g.st, g.rp, w_stroke=0.0, w_centring=w_P, econ=econ, ctx=g.ctx)
            for c in ("side", "diag", "move_turn"):
                v = rows.get(c) or {}
                shares[f"{n}.{c}"] = v.get("centring_dose")
        # The plan gives P NO healthy cap ("P has no healthy cap: show diag/move_turn shares at
        # SA0"); the design panel's 15 % line is REPORTED, never gated. diag <= 10 % is the SA0
        # dose rule's own clause (w_P / r_P must keep diag <= 10 % of its tracking income).
        diag_ok = all(v is None or v <= P2_DIAG_CAP for k, v in shares.items() if k.endswith(".diag"))
        ok2 = diag_ok
        row["clauses"] = {"diag<=10%": diag_ok}
        row["reported"] = {"healthy share>15% (design-panel line; not gated)":
                           [k for k, v in shares.items() if v is not None and v > P2_HEALTHY_CAP]}
    out["P2"] = {"pass": ok2, "w_P": w_P, "shares_at_w_P": shares, **row,
                 "rule": "w_P = w_S * (stroke prize) / (centring prize); prize = sideways-head side minus the "
                         "primary omni side, per second at w = 1; the parity band is [0.8, 1.25] x"}
    # P3
    if w_P is not None:
        econ_p = dataclasses.replace(econ, extra=econ.extra + (("centring", w_P),))
        heads_h = {n: _scored(g) for n, g in runs.items() if g.role in ("healthy", "posture")}
        heads_d = {n: _scored(g) for n, g in degenerate.items()}
        with extra_terms():
            rg = REC.rg1_ordering(heads_h, heads_d, econ, econ_p)
        statue = cfg.get("statue")
        x08 = cfg.get("x08")
        st_row = None
        if statue in degenerate and x08 in runs:
            with extra_terms():
                inc_h, _ = REC._category_income(_scored(runs[x08]), "side", econ_p)
                inc_d, _ = REC._category_income(_scored(degenerate[statue]), "side", econ_p)
            th, td = REC.totals(inc_h)[2], REC.totals(inc_d)[2]
            st_row = {"x08": th, "statue": td, "ratio": (th / td) if td > 0 else None}
        creeper = cfg.get("creeper")
        cr_row = None
        if creeper in degenerate and sideways:
            c_inc = cmean(degenerate[creeper], "side")
            s_inc = float(np.mean([v for v in sw.values() if v is not None])) if sw else None
            cr_row = {"creeper": c_inc, "sideways": s_inc,
                      "ratio": (c_inc / s_inc) if (c_inc is not None and s_inc) else None}
        ok_st = None if st_row is None else (st_row["ratio"] is None and st_row["x08"] > st_row["statue"]) or \
            (st_row["ratio"] is not None and st_row["ratio"] >= P3_STATUE_RATIO)
        ok_cr = None if cr_row is None else (cr_row["ratio"] is not None and cr_row["ratio"] <= P3_CREEPER_MAX)
        out["P3"] = {"pass": bool((rg["pass"] is not False) and ok_st is not False and ok_cr is not False),
                     "rg1": _rg_summary(rg), "statue": st_row, "creeper": cr_row,
                     "clauses": {"no_flip": rg["pass"], "x08 side vs statue>=1.10": ok_st,
                                 "creeper side<=0.10x sideways": ok_cr}}
        # P4
        r4 = {}
        ok4 = True
        for n, g in healthy.items():
            rows = income_table(g.st, g.rp, w_stroke=0.0, w_centring=w_P, econ=econ, ctx=g.ctx)
            for c in MOVING:
                v = rows.get(c) or {}
                if v.get("n_steps"):
                    r4[f"{n}.{c}"] = {"shift_pp": v["clip_shift_pp"], "absorbed": v["centring_absorbed"]}
                    ok4 &= abs(v["clip_shift_pp"]) <= R7_MAX_SHIFT_PP
            gap = _hand_gap(cmean(g, "side", "l"), cmean(g, "side", "r"))
            r4[f"{n}.side.hands"] = {"gap": gap}
            ok4 &= gap is None or gap <= R8_MAX_HAND_GAP
        out["P4"] = {"pass": bool(ok4), "rows": r4}
    else:
        out["P3"] = {"pass": None, "note": "no w_P (P2 had no positive prize)"}
        out["P4"] = {"pass": None, "note": "no w_P (P2 had no positive prize)"}
    return out


def _extreme(d: dict, key=None, *, largest=True, absval=False) -> tuple[str, float] | None:
    """``(row, value)`` of the largest (smallest) finite value of ``d`` (``d[row][key]`` when given)."""
    items = []
    for r, v in d.items():
        x = v.get(key) if key is not None and isinstance(v, dict) else v
        if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            continue
        items.append((r, float(x)))
    if not items:
        return None
    f = (lambda t: abs(t[1])) if absval else (lambda t: t[1])
    return (max if largest else min)(items, key=f)


def _hl(label: str, e, f="{:.3f}") -> str:
    return f"{label} --" if e is None else f"{label} {e[0]} {f.format(e[1])}"


def _headline(k: str, v: dict) -> str:
    """One line of the numbers that decide criterion ``k`` (the worst rows)."""
    try:
        if k == "R1":
            return _hl("min kept", _extreme(v["kept"], largest=False)) + f" (>= {R1_MIN_KEEP})"
        if k == "R2":
            return (_hl("max side kept", _extreme(v["side_kept"])) + f"; slide kept {_fmt(v['slide_kept'])}; "
                    + _hl("max side/sideways", _extreme(v["side_vs_sideways"])))
        if k == "R3":
            return (_hl("max short share", _extreme(v["short_share"]), "{:.4f}") + "; "
                    + _hl("max Spearman", _extreme(v["spearman_switch"], "rho")))
        if k == "R4" and "per_category_solve" in v:
            s = v["shares_at_w_S"]
            pcs = v["per_category_solve"]
            return (f"w_S {v['w_S']:.4g}; table {pcs['spec']}; tie bound by {pcs['tie_bound_by']}; shares "
                    + "; ".join(f"{n} " + ", ".join(f"{c} {_fmt(x)}" for c, x in s[n].items() if c != "stand")
                                for n in s))
        if k == "R4":
            s = v["shares_at_w_S"]
            return (f"w_S {v['w_S']:.4g}; binding " + ", ".join(f"{n}:{d['binding']}" for n, d in v["per_run"].items())
                    + "; fwd share " + ", ".join(f"{n} {_fmt(s[n].get('fwd'))}" for n in s)
                    + "; sideways side share " + ", ".join(f"{n} {_fmt(x)}" for n, x in
                                                           v["sideways_side_share_nominal"].items()))
        if k == "R5":
            judged = [r for r in v["rg1"]["rows"] if r["pass"] is not None]
            fails = [f"{r['healthy']}>{r['degenerate']} {r['category']}" for r in judged if r["pass"] is False]
            ta = {r: d for r, d in v["term_alone"].items() if d["gated"]}
            rot = {r: d for r, d in v["term_alone"].items() if not d["gated"]}
            return (f"no-flip fails: {fails or 'none'}; " + _hl("max term-alone (translation)", _extreme(ta, "ratio"))
                    + "; " + _hl("max term-alone (rotational, reported)", _extreme(rot, "ratio")))
        if k == "R6":
            rows = {r: d for r, d in v["rows"].items() if d.get("gated", True)}
            return (_hl("max income shift", _extreme(rows, "income_shift"), "{:.4f}") + "; "
                    + _hl("max liftoff shift", _extreme(rows, "liftoff_shift"), "{:.4f}"))
        if k == "R7":
            return (_hl("max |clip shift| pp", _extreme(v["rows"], "shift_pp", absval=True), "{:+.2f}") + "; "
                    + _hl("max absorbed", _extreme(v["rows"], "absorbed")))
        if k == "R8":
            return _hl("max hand gap", _extreme(v["rows"], "gap"))
        if k == "R9":
            bad = [n for n, d in v["rows"].items() if not all(x["pass"] for x in d.values())]
            return f"failing neighbours: {bad or 'none'} of {list(v['rows'])}"
        if k == "P1":
            return (f"sideways side {v['sideways_side']}; healthy side "
                    + ", ".join(f"{n} {_fmt(x)}" for n, x in v["healthy_side"].items())
                    + "; " + _hl("max |zero row|", _extreme(v["zero_rows"], absval=True), "{:.2e}"))
        if k == "P2":
            return (f"w_P {_fmt(v.get('w_P'), '{:.4g}')} band {v.get('w_P_band')}; "
                    + _hl("max healthy share", _extreme(v.get("shares_at_w_P") or {})))
        if k == "P3":
            st = v.get("statue") or {}
            cr = v.get("creeper") or {}
            return (f"no-flip {_verdict((v.get('rg1') or {}).get('pass'))}; x08/statue side totals "
                    f"{_fmt(st.get('x08'))}/{_fmt(st.get('statue'))} ratio {_fmt(st.get('ratio'))}; creeper/sideways "
                    f"{_fmt(cr.get('ratio'))}")
        if k == "P4":
            rows = v.get("rows") or {}
            return (_hl("max |clip shift| pp", _extreme(rows, "shift_pp", absval=True), "{:+.2f}") + "; "
                    + _hl("max hand gap", _extreme(rows, "gap")))
    except (KeyError, TypeError) as exc:  # a headline never hides the verdict
        return f"(headline unavailable: {exc!r})"
    return ""


def render_gate_md(rep: dict) -> str:
    c = rep["criteria"]
    L = ["# REPLAY_GATE -- stroke/posture plan step 7", "",
         f"Economy: `{rep['economy']['source']}` (clock {rep['economy']['clock']}, tibia weight "
         f"{rep['economy']['tibia_weight']}). Parameters: `{json.dumps(rep['params'])}`.", "",
         f"**w_S = {rep['w_S']:.4g}**, **w_P = {_fmt(rep['w_P'], '{:.4g}')}**", ""]
    if rep.get("stroke_cat_w"):
        sc = rep["stroke_cat_w"]
        L += [f"Per-category stroke table: `KRABBY_OMNI_STROKE_W={rep['w_S']!r}` "
              f"`KRABBY_OMNI_STROKE_CAT_W={sc['spec']}` -- effective w_S x m_c: "
              + ", ".join(f"{c} {w:.4g}" for c, w in sc["w_eff"].items())
              + ". P1-P4 still price the stroke prize at w_S (unscaled).", ""]
    L += ["| Id | Verdict | Clauses | Deciding numbers |", "|---|---|---|---|"]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "P1", "P2", "P3", "P4"):
        v = c.get(k) or {}
        cl = "; ".join(f"{a}: {_verdict(b)}" for a, b in (v.get("clauses") or {}).items())
        L.append(f"| {k} | **{_verdict(v.get('pass'))}** | {cl} | {v.get('headline', '')} |")
    if rep.get("dropped"):
        L += ["", "Dropped inputs (recorded, not silently skipped):"]
        L += [f"- {k}: {v}" for k, v in rep["dropped"].items()]
    L += ["", "## Runs", "", "| name | role | kind | N | notes |", "|---|---|---|---|---|"]
    for n, r in rep["runs"].items():
        L.append(f"| {n} | {r['role']} | {r['kind']} | {r['N']} | {'; '.join(r['notes'])} |")
    L += ["", "## Numbers", ""]
    for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "P1", "P2", "P3", "P4"):
        L += [f"### {k}: {_verdict((c.get(k) or {}).get('pass'))}", "", "```",
              json.dumps(_jsonable({a: b for a, b in (c.get(k) or {}).items() if a != "pass"}), indent=1,
                         sort_keys=True)[:20000], "```", ""]
    return "\n".join(L)


def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


# ============================================================================ CLI
def _readout(run_dir: str, params: StrokeParams, econ, limit=None):
    st = load_stamp(run_dir, limit=limit)
    rp = replay(st, params)
    vals = episode_values(st, rp, econ)
    return st, rp, vals, readout_table(st, vals)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("readout")
    a.add_argument("run_dir")
    a.add_argument("--vs", default=None)
    a.add_argument("--validate", action="store_true")
    a.add_argument("--params", nargs="*", default=[])
    a.add_argument("--econ-env-yaml", default=str(W_ENV_YAML))
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--json", default=None)
    a.add_argument("--seed-a", default=None,
                   help="SA0 decision 7's UNISON seed band: a same-stage seed pair's first run dir "
                        "(with --seed-b; only used with --vs)")
    a.add_argument("--seed-b", default=None, help="the seed pair's second run dir")
    b = sub.add_parser("income")
    b.add_argument("run_dir")
    b.add_argument("--w-stroke", type=float, required=True)
    b.add_argument("--w-centring", type=float, required=True)
    b.add_argument("--stroke-cat-w", default=None, metavar="SPEC",
                   help="KRABBY_OMNI_STROKE_CAT_W's spec (fwd:m,back:m,side:m,...; the knob's own parser): "
                        "price the stroke at w_stroke x m_c per category; stroke_per_s stays at w = 1")
    b.add_argument("--params", nargs="*", default=[])
    b.add_argument("--econ-env-yaml", default=str(W_ENV_YAML))
    b.add_argument("--action-clip", type=float, default=None)
    b.add_argument("--allow-econ-mismatch", action="store_true",
                   help="price a stamp whose head's own clock weights differ from --econ-env-yaml's "
                        "(refused by default: the default economy is W's clock-free dump)")
    b.add_argument("--train-log-dir", default=None,
                   help="a knob-on training run dir (tensorboard events): per-category r = log / replay")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--json", default=None)
    g = sub.add_parser("gate")
    g.add_argument("--config", required=True)
    g.add_argument("--out", required=True)
    p = sub.add_parser("parity")
    p.add_argument("dump")
    p.add_argument("--tol", type=float, default=1e-5)
    q = sub.add_parser("pooled", help="a pooled two-seed readout: does an arm move its own target "
                                      "past a bar on BOTH seeds, pooled (module docstring)")
    q.add_argument("arm_s3")
    q.add_argument("base_s3")
    q.add_argument("arm_s2")
    q.add_argument("base_s2")
    q.add_argument("--bar", nargs="*", default=[],
                   help="cat.key=value, layered onto the default bars (side.body_offset=-0.05, "
                        "side.stance_stroke=0.010, side.stride_along=0.02)")
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "pooled":
        rep = pooled_readout(args.arm_s3, args.base_s3, args.arm_s2, args.base_s2,
                             bars=parse_bars(args.bar), limit=args.limit)
        text = json.dumps(_jsonable(rep), indent=2, sort_keys=True)
        print(text)
        if args.json:
            Path(args.json).write_text(text)
        return 0
    if args.cmd == "parity":
        with np.load(args.dump) as z:
            dump = {k: z[k] for k in z.files}
        r = parity_check(dump, tol=args.tol)
        print(f"[parity] {r['steps']} steps x {r['envs']} envs; max |diff| " +
              ", ".join(f"{k} {v:.3g}" for k, v in r["max_abs"].items()) + f"; checked {r['checked']} < {r['tol']}")
        if "cat_scale" in r:
            print("[parity] per-category: reward checked as cat_scale[cat] * sum(pay) / dt; cat_scale "
                  + ", ".join(f"{c} {m:g}" for c, m in r["cat_scale"].items()) + "; steps per cat "
                  + ", ".join(f"{c} {n}" for c, n in r["cat_steps"].items()))
            print(f"[parity] cat vs stand/turning/lat_share: {r['cat_mismatches']} mismatch"
                  + ("" if r["cat_mismatches"] == 1 else "es"))
        print("[parity] PASS" if r["ok"] else "[parity] FAIL")
        return 0 if r["ok"] else 1
    if args.cmd == "gate":
        rep = run_gate(_load_config(Path(args.config)), Path(args.out))
        fails = [k for k, v in rep["criteria"].items() if v.get("pass") is False]
        print(f"[gate] wrote {args.out}/replay_gate.json and REPLAY_GATE.md; FAIL: {fails or 'none'}")
        return 0

    params = parse_params(args.params)
    econ = None
    econ_info = None
    try:
        econ, econ_info = load_economy(args.econ_env_yaml)
    except SystemExit as exc:
        if args.cmd == "income":
            raise
        print(f"[readout] tracking income not priced: {exc}")
    if args.cmd == "readout":
        st, rp, vals, table = _readout(args.run_dir, params, econ, args.limit)
        for n in st.notes:
            print(f"[readout] note: {n}")
        if st.guard_max is not None:
            print(f"[readout] heading guard: max |wrap(psi - (face - heading_err))| = {st.guard_max:.2e}")
        boot = None
        unison = None
        if args.vs:
            if bool(args.seed_a) != bool(args.seed_b):
                raise SystemExit("--seed-a and --seed-b must be given together")
            st2, _rp2, vals2, _t2 = _readout(args.vs, params, econ, args.limit)
            boot = paired_bootstrap(st, vals, st2, vals2)
            seed_a = seed_b = None
            if args.seed_a and args.seed_b:
                st_sa = load_stamp(args.seed_a, limit=args.limit)
                st_sb = load_stamp(args.seed_b, limit=args.limit)
                seed_a, seed_b = (st_sa, unison_values(st_sa)), (st_sb, unison_values(st_sb))
            unison = unison_flags(st, unison_values(st), st2, unison_values(st2), seed_a=seed_a, seed_b=seed_b)
        print(render_readout(Path(args.run_dir).name, st, table, boot))
        if unison is not None:
            print(render_unison(unison))
        if st.kind == "omni":
            for line in premise(table):
                print(f"[readout] {line}")
        rc = 0
        if args.validate:
            ok, lines = validate(st, table)
            for line in lines:
                print(f"[validate] {line}")
            if not ok:
                print("[validate] FAILED: " + "; ".join(x for x in lines if x.startswith("FAIL")))
                rc = 1
            else:
                print("[validate] all hard targets reproduced")
        if args.json:
            Path(args.json).write_text(json.dumps(_jsonable({"table": table, "bootstrap": boot,
                                                             "unison": unison, "params": params,
                                                             "notes": st.notes}), indent=2))
        return rc
    # income
    st = load_stamp(args.run_dir, limit=args.limit)
    own = stamp_clock(st)
    if own is not None and tuple(own[0]) != tuple(econ_info["dump_clock"]) and not args.allow_econ_mismatch:
        raise SystemExit(f"[income] the stamp's head trained with clock weights {list(own[0])} ({own[1]}) but "
                         f"the economy {econ_info['source']} has {econ_info['dump_clock']}: pass "
                         f"--econ-env-yaml {own[1]} (or --allow-econ-mismatch to price it under the other economy)")
    if args.action_clip is not None:
        st.action_clip = float(args.action_clip)
    cat_w = parse_cat_w(args.stroke_cat_w)
    rp = replay(st, params)
    prims = econ_primitives(st, econ, econ_info)
    ctx = context_rows(st, prims, econ)
    rows = income_table(st, rp, w_stroke=args.w_stroke, w_centring=args.w_centring, econ=econ, ctx=ctx,
                        cat_w=cat_w)
    print(f"[income] economy {econ_info['source']} clock {econ_info['clock']} (dump {econ_info['dump_clock']})")
    for n in econ_info["notes"]:
        print(f"[income] economy note: {n}")
    if ctx is None:
        print("[income] the sideways head is not priced in the omni economy: delivered = nominal, no clip rows")
    print(render_income(Path(args.run_dir).name, rows, w_stroke=args.w_stroke, w_centring=args.w_centring,
                        cat_w=cat_w))
    xlog = None
    if args.train_log_dir:
        xlog = train_log_crosscheck(st, rp, rows, Path(econ_info["source"]), Path(args.train_log_dir),
                                    w_stroke=args.w_stroke, cat_w=cat_w)
        if xlog is None:
            print(f"[income] --train-log-dir {args.train_log_dir}: not a directory; no cross-check")
        else:
            t = [[c, _fmt(d["stroke_log"], "{:.4f}"), _fmt(d["stroke_replay"], "{:.4f}"), _fmt(d["r_stroke"], "{:.2f}"),
                  _fmt(d["centring_log"], "{:.4f}"), _fmt(d["centring_replay"], "{:.4f}"),
                  _fmt(d["r_centring"], "{:.2f}"), _fmt(d["absorbed_log"]), _fmt(d["absorbed_replay"]),
                  _fmt(d["clip_bind_log"]), _fmt(d["clip_bind_replay"])]
                 for c, d in xlog["per_category"].items()]
            print("== training telemetry vs replay (Episode_RewardCat, last 200 iterations; r = log / replay)")
            print_table(t, ["cat", "S_log/s", "S_rep/s", "r_S", "P_log/s", "P_rep/s", "r_P", "abs_log", "abs_rep",
                            "clip_log", "clip_rep"], out=sys.stdout)
            for name, d in ((xlog.get("stock") or {}).get("rows") or {}).items():
                print(f"[income] stock {name}: log {_fmt(d['log_per_s'])}/s replay {_fmt(d['replay_per_s'])}/s")
    if args.json:
        Path(args.json).write_text(json.dumps(_jsonable({"rows": rows, "economy": econ_info, "params": params,
                                                         "train_log": xlog,
                                                         **({"stroke_cat_w": cat_w} if cat_w is not None else {})}),
                                              indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
