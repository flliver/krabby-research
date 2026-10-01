"""Offline reward-economy replay for the omni 1c reward fix (pre-GPU gate, 2026-09-18).

THE STANDING GATE. "Replay new reward terms on saved healthy + degenerate raws before any
training screen" (repo rule; precedents
``$PKG/experiments/2026-08-15_1530_task1_velocity/offline_replay/replay_track_l1_gate.py``
and ``$PKG/experiments/2026-08-31_1414_gait_income_phaseout/floor_weight_replay.py``).

THE QUESTION. The verified 1c diagnosis (``ANALYSIS_1c_failure_2026-09-18.md``) claims the
1c reward economy makes STEPPING IN PLACE earn ~99 % of what WALKING earns, so the policy
drifted from a 0.21 m/s walker (1a) to a 0.04 m/s creeper (1c) while mean reward rose
61 -> 93. The proposed fix (user decision D, 2026-09-18) is four numbers:

    KRABBY_TRACK_SIGMA2   0.1  -> 0.04     (track_lin_vel_xy_exp std sqrt(0.1) -> 0.2)
    KRABBY_TRACK_L1_W    -1.0  -> -2.0     (penalty_tracking_error_l1 weight)
    track_ang_vel_z_exp std 0.15 -> 0.5    (needs a new omni knob)
    reward_clock_schedule combine sum -> product   (needs a knob/seam)

This script replays the ACTUAL shipped formulas on the three saved canary raws and reports,
per moving hold, what each head earns under OLD vs NEW terms and a sweep around NEW.

HEADS (all from the CORRECTED driver re-eval, ``outputs/omni_plan/reeval_corrected``; the
campaign's own canaries are void -- they ran without KRABBY_OMNI_FACING_OBS_CLIP):
  walker  = 1a head model_999   (fwd speed ratio 0.76)
  mid     = 1b head model_1998  (fwd 0.16, lat_l 0.35)
  creeper = 1c head model_6997  (fwd 0.16, lat_l 0.03, diag 0.15, move_turn -0.04; tripod 0.85)

FORMULAS. Every term below is transcribed from the shipped code, not from memory:
  * ``$LIB/envs/mdp/rewards.py``          -- penalty_tracking_error_l1, forward_progress,
    reward_action_rate, reward_orientation, reward_lin_vel_z, air-time, foot-idle,
    excess-feet, delta-torques, RewardClockContactSchedule / RewardClockSwingApex wrappers,
    _clock_cmd_speed / _cmd_gate.
  * ``$PKG/mdp/crab_hex_clock_reward.py`` -- clock_command_speed, clock_schedule_income
    (sum AND product), clock_swing_apex_income.
  * ``$OMNI/mdp/crab_hex_omni_rewards.py``-- penalty_cross_command_vel_l2 (the omni
    ``penalty_lin_vel_y``), reward_facing_exp (the omni ``reward_tracking_yaw``),
    reward_yaw_progress.
  * ``$PKG/mdp/parkour_actions.py``       -- the gait-clock advance (cadence law).
Weights/params are the 1c window's own, read off the training log's reward table and
``params/env.yaml`` of run 2026-09-17_00-13-40 (see WEIGHTS below).

INCOME CONVENTION. Isaac Lab's reward manager accumulates ``weight * value * dt``, so
``Episode_Reward/<name> / episode_seconds == mean_t(weight * value)``. Every number printed
here is therefore directly comparable to the training log's per-second income table.

THE CLOCK-PHASE APPROXIMATION (the one real reconstruction gap). The raw npz does NOT carry
``clock_phase``. The cadence law is exactly reconstructible from ``cmd_applied``
(``freq = clip(v_clock/0.35, max=1) * 0.5 rev/s`` while ``v_clock > 0.2``, frozen otherwise,
with ``v_clock = max(||cmd_xy||, 1.2 * |cmd_wz|)``), so the phase is known up to ONE unknown
constant per episode (reset draws it uniformly; the eval ran KRABBY_RSI_FRAC=0 and each npz
is a single uninterrupted rollout). We therefore fit that constant per episode by maximising
the episode's OLD-economy clock income (schedule[sum] + apex) over a grid of offsets -- the
alignment the trained, clock-observing policy actually had -- and hold the SAME offset for
every variant. ``--offset-report`` prints the best/mean/worst spread so the reader can see
how much of the clock number is fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, replace

import numpy as np

# --------------------------------------------------------------------------- shipped constants
DT = 0.02
# crab_hex_dimensions
CLOCK_F_MAX_REV_S = 0.5
CLOCK_V_MAX_M_S = 0.35
CLOCK_CMD_STOP_M_S = 0.2
# crab_hex_clock_reward
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
TRIPOD_A = ("FL", "MR", "RL")
FOOT_OFFSETS = np.array([0.0 if leg in TRIPOD_A else math.pi for leg in FOOT_ORDER])
WINDOW_SHARPNESS = 0.15
FORCE_REF_N = 100.0
VEL_REF_M_S = 0.10
TOUCH_REF_N = 50.0
APEX_MID_THRESH = 0.7
APEX_TARGET_M = 0.08
APEX_SIGMA_M = 0.03
GROUND_OFFSET_FROM_ROOT_M = -1.05
MIN_UPRIGHT_GZ = 0.9
# omni env cfg (params/env.yaml of the 1c run)
OMNI_CLOCK_TURN_R = 1.2
ANG_VEL_MAX = 0.3          # reward_yaw_progress clip == command cfg ang_vel_max
DIR_EPS = 1e-6             # omni _command_dir_and_speed clamp_min
PROGRESS_EPS = 1e-8        # forward_progress divisor
PROGRESS_CAP = 1.05        # max_speed_scale
MIN_CMD_NORM = 0.12
AIR_TIME_THRESHOLD = 0.38
AIR_TIME_CMD_GATE = 0.1
CONTACT_FORCE_THRESHOLD = 0.1
MAX_IDLE_STEPS = 90
MAX_FEET_ON_GROUND = 4
# The ACTION TERM's own per-joint clip, per column block (actions = 6 CamShaft, 6 Hip_Femur,
# 6 Femur_Tibia; ``_clip_raw_actions``). It is the SECOND clamp: before the action term sees the
# policy output, the runner's ``ParkourRslRlVecEnvWrapper`` clamps every column to
# ``+-clip_actions`` (the head's ``params/agent.yaml``; 1.0 on every omni head). The eval npz
# stores the PRE-clip mean (median |mu| 5-8 on the saturated heads), so pricing ``action_rate`` on
# this clip alone overstated it ~3x (fixed 2026-10-01, ppo_construction plan step 2 item 5):
# :func:`main` now reads each head's ``clip_actions`` (:func:`head_action_clip`, default
# ``DEFAULT_RUNNER_CLIP`` when the dump is absent) and :func:`episode_primitives` applies both
# clamps in order. ``episode_primitives(z, n)`` with no ``action_clip`` keeps the old term-only
# model byte-for-byte (``replay_clock_economy.py`` and its tests call it that way on purpose).
ACTION_CLIP = np.array([1.0] * 6 + [4.8] * 6 + [4.8] * 6)
#: ``clip_actions`` assumed for a head whose ``params/agent.yaml`` (or its key) cannot be found.
DEFAULT_RUNNER_CLIP = 1.0

# 1c window weights (training log reward table, run 2026-09-17_00-13-40)
WEIGHTS = {
    "track_lin_vel_xy_exp": 1.25,
    "track_ang_vel_z_exp": 1.0,
    "penalty_tracking_error_l1": -1.0,       # overridden by the variant's l1_w
    "reward_clock_schedule": 1.0,
    "reward_clock_swing_apex": 1.0,
    "reward_forward_progress_along_command": 0.6,
    "reward_yaw_progress": 1.0,
    "reward_tracking_yaw": 0.2,              # omni reward_facing_exp
    "reward_feet_air_time_positive": 0.8,
    "penalty_lin_vel_y": -3.0,               # omni penalty_cross_command_vel_l2
    "reward_action_rate": -0.3,
    "reward_orientation": -0.7,
    "reward_lin_vel_z": -0.15,
    "penalty_foot_idle_when_forward": -0.12,
    "penalty_excess_feet_contact_forward": -0.2,
    "reward_delta_torques": -1e-6,
}

NOT_REPLAYED = (
    "reward_stride_length (w 0.5; logged income 0.000/s in every 1c window -- the term is "
    "identically 0 for these heads), penalty_tibia_deviation_in_stance (w -0.28; logged "
    "-0.042/s -- needs default_joint_pos, not in the raw), and the 18 terms at weight 0."
)

HOLD_LABELS = ["stand", "fwd", "lat_l", "diag", "back", "turn_l90", "turn_r90", "move_turn"]
MOVING_HOLDS = ("fwd", "lat_l", "diag", "back", "move_turn")
#: ``slow__omni``'s turn-type holds -- the dedicated in-place turns plus the simultaneous
#: move+turn. These are the holds whose command was DRAWN as a turn, i.e. the eval analogue of
#: a training row with ``slot_id >= SLOT_TURN``: the gate the shipped
#: :func:`reward_facing_exp_turn_slot` uses (see the transcription note below ``OLD``/``NEW``).
#: Part A5 (step 4): ``--facing-turn-w`` pays on exactly this set and exactly 0 elsewhere.
#: SCOPE: these are the labels of ``slow__omni`` (the only schedule ``HOLD_LABELS`` and
#: ``DEFAULT_RAWS`` describe). The Part A6 ``curriculum__omni`` row adds a second move+turn hold
#: ``move_turn_r``; pointing this replay at curriculum raws means extending BOTH tuples.
TURN_HOLDS = ("turn_l90", "turn_r90", "move_turn")
_TURN_HOLD_IDS = [HOLD_LABELS.index(h) for h in TURN_HOLDS]

REPO = "/home/nickmagus/krabby/krabby-research"
DEFAULT_RAWS = {
    "walker(1a)": f"{REPO}/parkour/outputs/omni_plan/reeval_corrected/1a/slow__omni/seed001/*/raw",
    "mid(1b)": f"{REPO}/parkour/outputs/omni_plan/reeval_corrected/1b/slow__omni/seed001/*/raw",
    "creeper(1c)": f"{REPO}/parkour/outputs/omni_plan/reeval_corrected/1c/slow__omni/seed001/*/raw",
}


# --------------------------------------------------------------------------- variant
@dataclass(frozen=True)
class Terms:
    """The four numbers the fix moves, plus two NOT-in-the-decision probes.

    ``ang_turn_only`` and ``progress_w`` are not part of the 2026-09-18 decision; they are
    the two cheapest additional levers the diagnosis already listed as fix candidates
    ("widen OR GATE track_ang_vel_z_exp ... turning rows only", "raise the progress weight
    well above 0.6"). They are replayed so the report can say what the decided four buy and
    what a minimal addition would buy on top.

    Part A5 (step 4, 2026-09-18): four more fields for the ranked-candidates plan's cheap
    reward-shaping levers (C2/C4/C6/C9). Every default below reproduces the value this module
    already hardcoded before this pass -- ``yaw_w``/``yaw_progress_w`` from ``WEIGHTS``,
    ``facing_turn_w``/``coupled_sigma`` at the shipped terms' own OFF values (the knobs are unset
    in every preset) -- so constructing ``OLD``/``NEW`` with no override (as this file already
    does) is bit-identical to before this pass.
    """

    sigma2: float = 0.1        # track_lin_vel_xy_exp: exp(-|err|^2 / sigma2); std = sqrt(sigma2)
    l1_w: float = -1.0         # penalty_tracking_error_l1 weight
    ang_std: float = 0.15      # track_ang_vel_z_exp std
    combine: str = "sum"       # reward_clock_schedule combine
    ang_turn_only: bool = False  # PROBE: zero track_ang_vel_z_exp on rows with cmd_wz == 0
    progress_w: float = 0.6      # PROBE: reward_forward_progress_along_command weight
    # --- Part A5 additions (candidates C2/C4/C6/C9; none shipped yet -- see term_incomes) ---
    yaw_w: float = WEIGHTS["reward_tracking_yaw"]            # reward_facing_exp weight (KRABBY_YAW_W)
    yaw_progress_w: float = WEIGHTS["reward_yaw_progress"]   # reward_yaw_progress weight (C2)
    facing_turn_w: float = 0.0                               # reward_facing_turn weight (C4); off
    coupled_sigma: float | None = None                       # facing-coupled tracking sigma (C6); off

    @property
    def tag(self) -> str:
        s = f"s2={self.sigma2:g} L1={self.l1_w:g} astd={self.ang_std:g} {self.combine}"
        if self.ang_turn_only:
            s += " ang@turn"
        if self.progress_w != 0.6:
            s += f" prog={self.progress_w:g}"
        if self.yaw_w != WEIGHTS["reward_tracking_yaw"]:
            s += f" yaw_w={self.yaw_w:g}"
        if self.yaw_progress_w != WEIGHTS["reward_yaw_progress"]:
            s += f" yaw_prog_w={self.yaw_progress_w:g}"
        if self.facing_turn_w != 0.0:
            s += f" facing_turn_w={self.facing_turn_w:g}"
        if self.coupled_sigma is not None:
            s += f" coupled_sigma={self.coupled_sigma:g}"
        return s


OLD = Terms(0.1, -1.0, 0.15, "sum")
NEW = Terms(0.04, -2.0, 0.5, "product")

# --- Part A5 (step 4): the two A2 candidate terms, TRANSCRIBED from the shipped code -----------
# Both terms are now landed in ``$OMNI/mdp/crab_hex_omni_rewards.py`` (step 2, 2026-09-18), so --
# like every other formula in this file -- they are transcribed, not reconstructed:
#   C4 ``reward_facing_turn`` = ``crab_hex_omni_rewards.reward_facing_exp_turn_slot``
#       = ``reward_facing_exp(...) * _slot_is_turning(...)``
#       = ``exp(-|wrap(heading_target - heading_w)|)`` on the rows whose command SLOT asks for a
#         turn, and exactly 0.0 on every other row.
#       Here ``exp(-|heading_err|)`` is already the primitive ``p["facing"]`` (the same value the
#       ungated ``reward_tracking_yaw`` is paid on), so C4 is that primitive times the turn gate
#       and nothing else -- no separate kernel, no ramp. The plan's own arithmetic pins this
#       reading: 0.2 * (exp(-0.1) - exp(-1.2)) = 0.121 and 1.0 * (...) = 0.604, i.e. the
#       candidates table's "0.12 -> 0.60/s over the settled-0.1-rad vs 1.2-rad span".
#       THE GATE: ``slot_id >= SLOT_TURN``. On a TRAINING row the slot is drawn and is constant
#       for the whole command, so the term pays a settled turn too -- which is the entire point
#       (``_slot_is_turning``'s own docstring: a ``cmd_wz != 0`` gate "would stop paying at the
#       instant it started being satisfied, which is an incentive to overshoot"). The eval
#       harness's
#       ``schedule_hook`` re-derives ``slot_id`` from ``vel_command_b[:, 2] != 0`` instead, so a
#       LITERAL eval-time evaluation of the shipped term would follow the facing P-control's dead
#       band and would also fire on translation holds whose facing merely drifted. That is an
#       artefact of the eval harness, not the income the candidate buys in training, so the
#       replay gates on the HOLD (``p["turn_hold"]``, i.e. ``seg in TURN_HOLDS``) -- the eval
#       analogue of the drawn slot, constant across the hold and exactly 0 on translation rows,
#       as the candidates table states.
#   C6 ``crab_hex_omni_rewards.track_lin_vel_xy_exp_facing_coupled``
#       = ``stock(env, std, ...) * exp(-(wrap(heading_target - heading_w) / facing_sigma) ** 2)``
#       -- applied on EVERY row, not gated, since it REPLACES the stock ``track_lin_vel_xy_exp``
#       func rather than adding a term.

#: which reconstruction of penalty_foot_idle_when_forward to score (see episode_primitives)
IDLE_KEY = "idle"

#: the terms a policy can only earn by actually moving (the diagnosis's "motion-requiring")
MOTION_REQUIRING = ("progress", "yaw_progress", "air_time")


# --------------------------------------------------------------------------- primitives
def _gravity_b(quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``projected_gravity_b`` components (world gravity (0,0,-1) rotated into the body)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    gx = -(2.0 * (x * z - w * y))
    gy = -(2.0 * (y * z + w * x))
    gz = -(1.0 - 2.0 * (x * x + y * y))
    return gx, gy, gz


def _clock_income(
    phi: np.ndarray, force: np.ndarray, speed_xy: np.ndarray, running: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``clock_schedule_income`` for BOTH combine modes (T,) -- pre-upright-gate."""
    i_swing = 1.0 / (1.0 + np.exp(-np.sin(phi) / WINDOW_SHARPNESS))
    i_swing = np.where(running[:, None], i_swing, 0.0)
    i_stance = 1.0 - i_swing
    swing_ok = np.exp(-((force / FORCE_REF_N) ** 2))
    loaded = 1.0 - np.exp(-((force / TOUCH_REF_N) ** 2))
    stance_ok = loaded * np.exp(-((speed_xy / VEL_REF_M_S) ** 2))
    inc_sum = (i_swing * swing_ok + i_stance * stance_ok).mean(axis=1)
    w_swing = np.maximum(i_swing.sum(axis=1), 1e-6)
    w_stance = np.maximum(i_stance.sum(axis=1), 1e-6)
    swing_q = (i_swing * swing_ok).sum(axis=1) / w_swing
    stance_q = (i_stance * stance_ok).sum(axis=1) / w_stance
    inc_prod = np.where(running, swing_q * stance_q, stance_q)
    return inc_sum, inc_prod


def _apex_income(phi: np.ndarray, height: np.ndarray, running: np.ndarray) -> np.ndarray:
    s = np.sin(phi)
    i_mid = np.clip((s - APEX_MID_THRESH) / (1.0 - APEX_MID_THRESH), 0.0, 1.0)
    i_mid = np.where(running[:, None], i_mid, 0.0)
    apex_ok = np.exp(-(((height - APEX_TARGET_M) / APEX_SIGMA_M) ** 2))
    w = np.maximum(i_mid.sum(axis=1), 1e-6)
    return (i_mid * apex_ok).sum(axis=1) / w


def episode_primitives(z, n_offsets: int, action_clip: float | None = None) -> dict:
    """Every variant-independent per-step quantity for one episode, plus the fitted phase.

    ``action_clip`` is the runner's ``+-clip_actions`` clamp (the head's ``params/agent.yaml``),
    applied BEFORE the action term's per-joint ``ACTION_CLIP``; ``None`` = no runner clamp, the
    pre-2026-10-01 term-only model (byte-identical; :func:`main` always passes the head's clip)."""
    cmd = z["cmd_applied"].astype(np.float64)
    v_b = z["root_lin_vel_b"].astype(np.float64)
    w_b = z["root_ang_vel_b"].astype(np.float64)
    quat = z["root_quat_w"].astype(np.float64)
    force = z["foot_force_norm"].astype(np.float64)
    foot_v = z["foot_lin_vel_w"].astype(np.float64)
    foot_p = z["foot_pos_w"].astype(np.float64)
    root_p = z["root_pos_w"].astype(np.float64)
    actions = z["actions"].astype(np.float64)
    torque = z["applied_torque"].astype(np.float64)
    herr = z["heading_err"].astype(np.float64)
    seg = z["segment_id"].astype(np.int32)
    # These raws must be ``slow__omni``'s: ``HOLD_LABELS`` is that schedule's label list read
    # POSITIONALLY, so ``hold_mask``, ``MOVING_HOLDS`` and ``_TURN_HOLD_IDS`` all name holds by
    # segment index. A raw from any other manifest row -- e.g. Part A6's 10-hold
    # ``curriculum__omni``, where index 3 is ``lat_r`` and not ``diag`` -- would be silently
    # re-labelled (and would pay ``facing_turn`` on whatever landed at 5/6/7). Refuse it loudly.
    if seg.size and int(seg.max()) >= len(HOLD_LABELS):
        raise SystemExit(
            f"segment_id {int(seg.max())} is outside this script's {len(HOLD_LABELS)}-hold "
            f"schedule {list(HOLD_LABELS)}: these raws are not slow__omni's. Extend HOLD_LABELS, "
            "MOVING_HOLDS and TURN_HOLDS together (and re-check the weights) before replaying "
            "another schedule's raws."
        )
    steady = z["steady_mask"].astype(bool)
    done = z["done"].astype(bool)
    T = cmd.shape[0]

    # ---- command-derived
    cmd_xy = cmd[:, :2]
    cmd_wz = cmd[:, 2]
    speed_cmd = np.linalg.norm(cmd_xy, axis=1)
    v_clock = np.maximum(speed_cmd, OMNI_CLOCK_TURN_R * np.abs(cmd_wz))   # clock_command_speed
    running = v_clock > CLOCK_CMD_STOP_M_S
    freq = np.where(running, np.minimum(v_clock / CLOCK_V_MAX_M_S, 1.0) * CLOCK_F_MAX_REV_S, 0.0)
    phase_base = np.cumsum(2.0 * math.pi * freq * DT)                      # advanced before the step
    move_gate = v_clock > MIN_CMD_NORM                                     # _cmd_gate under omni

    # ---- tracking errors
    err_xy = cmd_xy - v_b[:, :2]
    err_xy_sq = np.sum(err_xy**2, axis=1)
    err_l1 = np.linalg.norm(err_xy, axis=1)
    err_wz = cmd_wz - w_b[:, 2]

    # ---- direction-derived
    dir_prog = cmd_xy / (speed_cmd[:, None] + PROGRESS_EPS)
    dir_b = cmd_xy / np.maximum(speed_cmd, DIR_EPS)[:, None]
    progress = np.clip(np.sum(v_b[:, :2] * dir_prog, axis=1), 0.0, PROGRESS_CAP) * (
        speed_cmd > MIN_CMD_NORM
    )
    cross = v_b[:, 0] * (-dir_b[:, 1]) + v_b[:, 1] * dir_b[:, 0]
    cross_pen = cross**2 * (speed_cmd > MIN_CMD_NORM)
    yaw_prog = np.clip(w_b[:, 2] * np.sign(cmd_wz), -ANG_VEL_MAX, ANG_VEL_MAX) * (cmd_wz != 0.0)
    herr_abs = np.abs(herr)
    facing = np.exp(-herr_abs)

    # ---- attitude
    gx, gy, gz = _gravity_b(quat)
    upright = (-gz > MIN_UPRIGHT_GZ).astype(np.float64)
    orient = gx**2 + gy**2
    vz2 = v_b[:, 2] ** 2

    # ---- clock phase fit (see module docstring)
    speed_xy = np.linalg.norm(foot_v[:, :, :2], axis=2)
    ground_z = root_p[:, 2] + GROUND_OFFSET_FROM_ROOT_M
    height = foot_p[:, :, 2] - ground_z[:, None]
    score_mask = steady.copy()
    if not score_mask.any():
        score_mask = np.ones(T, dtype=bool)
    offsets = np.arange(n_offsets) * (2.0 * math.pi / n_offsets)
    scores = np.empty(n_offsets)
    cache = []
    for i, off in enumerate(offsets):
        phi = phase_base[:, None] + FOOT_OFFSETS[None, :] + off
        s_sum, s_prod = _clock_income(phi, force, speed_xy, running)
        apex = _apex_income(phi, height, running)
        cache.append((s_sum * upright, s_prod * upright, apex * upright))
        scores[i] = float(((s_sum + apex) * upright)[score_mask].mean())
    best = int(np.argmax(scores))
    clock_sum, clock_prod, apex_inc = cache[best]

    # ---- contact-derived counters (sequential, as the shipped terms are)
    contact = force > CONTACT_FORCE_THRESHOLD
    air_time = np.zeros(6)
    idle = np.zeros(6)
    air_rew = np.zeros(T)
    idle_pen = np.zeros(T)
    prev_contact = np.zeros(6, dtype=bool)
    for t in range(T):
        c = contact[t]
        first_contact = c & ~prev_contact
        air_rew[t] = float(np.sum(np.maximum(air_time - AIR_TIME_THRESHOLD, 0.0) * first_contact))
        air_time = np.where(c, 0.0, air_time + DT)
        idle = np.where(c, 0.0, idle + 1.0)
        idle_pen[t] = float(np.sum(np.maximum(idle - MAX_IDLE_STEPS, 0.0)))
        prev_contact = c
    air_rew = air_rew * (speed_cmd > AIR_TIME_CMD_GATE)
    idle_pen_carry = idle_pen * move_gate
    # Sensitivity: the shipped idle counter runs across the whole episode, so a foot parked
    # in the air during the (unpenalised) stand hold arrives at the fwd hold already past
    # the 90-step threshold. In TRAINING a 40-s episode mixes slots the same way, but the
    # canary's fixed 8-s stand -> fwd order maximises the carry-over. This variant restarts
    # the counter at each hold boundary; the two bracket the true training value.
    idle_reset = np.zeros(T)
    idle2 = np.zeros(6)
    for t in range(T):
        if t > 0 and seg[t] != seg[t - 1]:
            idle2[:] = 0.0
        idle2 = np.where(contact[t], 0.0, idle2 + 1.0)
        idle_reset[t] = float(np.sum(np.maximum(idle2 - MAX_IDLE_STEPS, 0.0)))
    idle_pen_reset = idle_reset * move_gate
    excess = np.maximum(contact.sum(axis=1) - MAX_FEET_ON_GROUND, 0.0) * move_gate

    # ---- action / torque streams
    a = actions if action_clip is None else np.clip(actions, -float(action_clip), float(action_clip))
    a = np.clip(a, -ACTION_CLIP, ACTION_CLIP)
    arate = np.zeros(T)
    arate[1:] = np.linalg.norm(a[1:] - a[:-1], axis=1)
    dtau = np.zeros(T)
    dtau[1:] = np.sum((torque[1:] - torque[:-1]) ** 2, axis=1)

    valid = np.ones(T, dtype=bool)
    if done.any():
        valid[np.argmax(done) :] = False   # drop the termination step and anything after it

    return {
        "T": T,
        "seg": seg,
        "steady": steady,
        "valid": valid,
        "upright_ok": upright > 0.5,
        "err_xy_sq": err_xy_sq,
        "err_l1": err_l1,
        "err_wz_sq": err_wz**2,
        "clock_sum": clock_sum,
        "clock_prod": clock_prod,
        "apex": apex_inc,
        "progress": progress,
        "yaw_prog": yaw_prog,
        "facing": facing,
        "herr_abs": herr_abs,
        "cross": cross_pen,
        "arate": arate,
        "orient": orient,
        "vz2": vz2,
        "air": air_rew,
        "idle": idle_pen_carry,
        "idle_reset": idle_pen_reset,
        "turning": (cmd_wz != 0.0).astype(np.float64),
        # The eval analogue of ``slot_id >= SLOT_TURN``: the hold's command was DRAWN as a turn.
        # Constant across the hold (unlike ``turning``, which follows the facing P-control's dead
        # band) -- see the C4 transcription note above ``OLD``/``NEW``.
        "turn_hold": np.isin(seg, _TURN_HOLD_IDS).astype(np.float64),
        "excess": excess,
        "dtau": dtau,
        "v_along": np.sum(v_b[:, :2] * dir_prog, axis=1),
        "speed_cmd": speed_cmd,
        "offset_scores": scores,
        "offset_best": best,
    }


def term_incomes(p: dict, mask: np.ndarray, t: Terms) -> dict[str, float]:
    """Per-second income (mean of weight*value) of every replayed term, over ``mask``."""
    if not mask.any():
        return {}
    m = mask
    clock = p["clock_sum"] if t.combine == "sum" else p["clock_prod"]
    ang = np.exp(-p["err_wz_sq"] / t.ang_std**2)
    if t.ang_turn_only:
        ang = ang * p["turning"]
    # C6: track_lin_vel_xy_exp_facing_coupled == the stock kernel times a Gaussian of the facing
    # error (see the transcription note above `OLD`/`NEW`); unset (the default) is the stock
    # kernel, bit-identical to before this pass.
    lin = np.exp(-p["err_xy_sq"] / t.sigma2)
    if t.coupled_sigma is not None:
        lin = lin * np.exp(-(p["herr_abs"] ** 2) / t.coupled_sigma**2)
    # C4: reward_facing_exp_turn_slot == the SAME exp(-|heading_err|) kernel the ungated
    # reward_tracking_yaw is paid on (p["facing"]), gated to the turn slots -- on an eval
    # schedule, the turn HOLDS.
    facing_turn_bonus = p["facing"] * p["turn_hold"]
    out = {
        "track_lin_exp": WEIGHTS["track_lin_vel_xy_exp"] * lin[m].mean(),
        "track_ang_exp": WEIGHTS["track_ang_vel_z_exp"] * ang[m].mean(),
        "clock_sched": WEIGHTS["reward_clock_schedule"] * clock[m].mean(),
        "clock_apex": WEIGHTS["reward_clock_swing_apex"] * p["apex"][m].mean(),
        "progress": t.progress_w * p["progress"][m].mean(),
        "yaw_progress": t.yaw_progress_w * p["yaw_prog"][m].mean(),
        "facing": t.yaw_w * p["facing"][m].mean(),
        "facing_turn": t.facing_turn_w * facing_turn_bonus[m].mean(),
        "air_time": WEIGHTS["reward_feet_air_time_positive"] * p["air"][m].mean(),
        "L1": t.l1_w * p["err_l1"][m].mean(),
        "cross_vel": WEIGHTS["penalty_lin_vel_y"] * p["cross"][m].mean(),
        "action_rate": WEIGHTS["reward_action_rate"] * p["arate"][m].mean(),
        "orientation": WEIGHTS["reward_orientation"] * p["orient"][m].mean(),
        "lin_vel_z": WEIGHTS["reward_lin_vel_z"] * p["vz2"][m].mean(),
        "foot_idle": WEIGHTS["penalty_foot_idle_when_forward"] * p[IDLE_KEY][m].mean(),
        "excess_feet": WEIGHTS["penalty_excess_feet_contact_forward"] * p["excess"][m].mean(),
        "delta_torque": WEIGHTS["reward_delta_torques"] * p["dtau"][m].mean(),
    }
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------- aggregation
def head_action_clip(raw_dir: str) -> tuple[float, str]:
    """``(clip_actions, source)`` of the head an eval raw dir was rolled out from: the stamp's
    ``run_meta.json`` (the raw dir's parent) names the checkpoint, whose ``params/agent.yaml``
    carries ``clip_actions``. Anything missing -> ``DEFAULT_RUNNER_CLIP`` with the reason."""
    meta_path = os.path.join(os.path.dirname(os.path.normpath(raw_dir)), "run_meta.json")
    try:
        with open(meta_path) as fh:
            ckpt = json.load(fh).get("checkpoint")
    except (OSError, ValueError):
        return DEFAULT_RUNNER_CLIP, f"default {DEFAULT_RUNNER_CLIP} (no readable {meta_path})"
    if not ckpt:
        return DEFAULT_RUNNER_CLIP, f"default {DEFAULT_RUNNER_CLIP} (run_meta names no checkpoint)"
    agent = os.path.join(os.path.dirname(ckpt), "params", "agent.yaml")
    if not os.path.isfile(agent):
        return DEFAULT_RUNNER_CLIP, f"default {DEFAULT_RUNNER_CLIP} (no {agent})"
    with open(agent) as fh:
        for line in fh:
            # top-level key only (column 0): the dump is a flat ``clip_actions: 1.0`` line
            if line.startswith("clip_actions:"):
                val = line.split(":", 1)[1].strip()
                if val in ("null", "~", ""):
                    return DEFAULT_RUNNER_CLIP, f"default {DEFAULT_RUNNER_CLIP} ({agent}: clip_actions null)"
                return float(val), agent
    return DEFAULT_RUNNER_CLIP, f"default {DEFAULT_RUNNER_CLIP} ({agent} has no clip_actions)"


def load_head(raw_glob: str, n_offsets: int, limit: int | None,
              action_clip: float | None = None) -> list[dict]:
    """Every episode of the newest raw dir matching ``raw_glob``. ``action_clip`` is passed to
    :func:`episode_primitives` (``None`` = the legacy term-only clip; :func:`main` passes the
    head's own ``clip_actions`` from :func:`head_action_clip`)."""
    dirs = sorted(glob.glob(raw_glob))
    if not dirs:
        raise SystemExit(f"no raw dir matched {raw_glob}")
    files = sorted(glob.glob(os.path.join(dirs[-1], "episode_*.npz")))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no episode_*.npz under {dirs[-1]}")
    return [episode_primitives(np.load(f), n_offsets, action_clip) for f in files]


def hold_mask(p: dict, hold: str, upright_only: bool) -> np.ndarray:
    idx = HOLD_LABELS.index(hold)
    m = (p["seg"] == idx) & p["steady"] & p["valid"]
    if upright_only:
        m = m & p["upright_ok"]
    return m


def head_hold_income(eps: list[dict], hold: str, t: Terms, upright_only: bool):
    """Step-weighted mean income per term across every episode, for one hold."""
    acc: dict[str, float] = {}
    n = 0
    for p in eps:
        m = hold_mask(p, hold, upright_only)
        k = int(m.sum())
        if k == 0:
            continue
        inc = term_incomes(p, m, t)
        for key, v in inc.items():
            acc[key] = acc.get(key, 0.0) + v * k
        n += k
    if n == 0:
        return {}, 0, 0.0
    v_along = 0.0
    for p in eps:
        m = hold_mask(p, hold, upright_only)
        if m.any():
            v_along += float(p["v_along"][m].sum())
    return {k: v / n for k, v in acc.items()}, n, v_along / n


def totals(inc: dict[str, float]) -> tuple[float, float, float]:
    pos = sum(v for v in inc.values() if v > 0)
    neg = sum(v for v in inc.values() if v < 0)
    return pos, neg, pos + neg


# Term groupings. The fix moves TRACKING directly and CLOCK/ANG indirectly; REGULARIZERS are
# invariant to the fix (they cancel in any OLD-vs-NEW comparison) but NOT invariant between
# two different gaits, so they are subtotalled separately -- the same split the
# gait-income-phaseout precedent made between its BEHAVIORAL and ACTION_STREAM columns.
TRACKING = ("track_lin_exp", "L1", "progress")
COMMAND = TRACKING + ("clock_sched", "clock_apex", "track_ang_exp", "facing", "yaw_progress",
                      "facing_turn", "air_time", "cross_vel")
REGULARIZERS = ("action_rate", "foot_idle", "excess_feet", "orientation", "lin_vel_z",
                "delta_torque")


def subtotal(inc: dict[str, float], keys) -> float:
    return sum(inc.get(k, 0.0) for k in keys)


# --------------------------------------------------------------------------- synthetic policies
def tracking_income(t: Terms, cmd_speed: float, v: float) -> float:
    """The command-tracking sub-economy at achieved along-command speed ``v``.

    track_lin_vel_xy_exp + penalty_tracking_error_l1 + forward_progress -- the three terms
    whose value is a function of the achieved planar velocity alone. On a pure-translation
    row ``cmd_wz == 0``, so track_ang_vel_z_exp and reward_yaw_progress are constants in
    ``v`` and drop out of any comparison along this axis. Perfect-direction travel assumed
    (the error is then the scalar |cmd - v|), which is the best case for the mover.

    BLIND TO TWO OF THE A5 LEVERS (read G3 accordingly). This analytic model -- which is all
    that G3, the zero-velocity sweep and the final candidate table see -- hardcodes
    ``WEIGHTS["reward_forward_progress_along_command"]`` rather than ``t.progress_w``, and it
    carries no heading term at all. So neither ``--progress-w``/``--new-progress-w`` (C9) nor
    ``--coupled-sigma`` (C6) can move G3: pre-existing behaviour for C9, newly reachable for C6.
    An unchanged G3 under those two levers is therefore NOT evidence that the lever is inert --
    G1/G2, the per-hold tables and the turn-hold table are where they show up.
    """
    err = abs(cmd_speed - v)
    return (
        WEIGHTS["track_lin_vel_xy_exp"] * math.exp(-(err**2) / t.sigma2)
        + t.l1_w * err
        + WEIGHTS["reward_forward_progress_along_command"] * min(max(v, 0.0), PROGRESS_CAP)
    )


def synthetic_rows(t: Terms, cmd_speed: float) -> dict[str, float]:
    return {
        "v=0 (stand/step-in-place)": tracking_income(t, cmd_speed, 0.0),
        f"v={cmd_speed:g} (perfect track)": tracking_income(t, cmd_speed, cmd_speed),
    }


# --------------------------------------------------------------------------- reporting
def fmt(x: float, w: int = 8, p: int = 3) -> str:
    return f"{x:>{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--walker", default=DEFAULT_RAWS["walker(1a)"])
    ap.add_argument("--mid", default=DEFAULT_RAWS["mid(1b)"])
    ap.add_argument("--creeper", default=DEFAULT_RAWS["creeper(1c)"])
    ap.add_argument("--sigma2", type=float, nargs="*", default=[0.04, 0.06],
                    help="sweep values for track_lin_vel_xy_exp sigma^2")
    ap.add_argument("--l1w", type=float, nargs="*", default=[-1.5, -2.0, -3.0],
                    help="sweep values for penalty_tracking_error_l1 weight")
    ap.add_argument("--ang-std", type=float, default=0.5, help="NEW track_ang_vel_z_exp std")
    ap.add_argument("--new-ang-turn-only", action="store_true",
                    help="probe: make the NEW variant also gate track_ang_vel_z_exp on cmd_wz != 0")
    ap.add_argument("--new-progress-w", "--progress-w", dest="new_progress_w", type=float,
                    default=None,
                    help="forward_progress weight inside the NEW variant (--progress-w is an "
                         "alias of this same flag, not a second lever, so the two spellings are "
                         "byte-identical; supplying EITHER arms the turn-hold income table; "
                         f"default {NEW.progress_w:g} = no change; C9 candidate value 1.5)")
    # The five Part A5 levers parse to None when NOT supplied, so "armed" below follows the FLAG
    # and not the value: `--facing-turn-w 0` (a deliberate price-this-at-zero run) still prints
    # the turn-hold table. Each is resolved to its "no change" default right after parsing.
    ap.add_argument("--yaw-w", type=float, default=None,
                    help="override reward_facing_exp's weight inside the NEW variant "
                         f"(KRABBY_YAW_W twin; default {WEIGHTS['reward_tracking_yaw']:g} = no change)")
    ap.add_argument("--yaw-progress-w", type=float, default=None,
                    help="override reward_yaw_progress's weight inside the NEW variant "
                         f"(KRABBY_YAW_PROGRESS_W twin; default {WEIGHTS['reward_yaw_progress']:g} "
                         "= no change; C2 candidate value 3.0)")
    ap.add_argument("--facing-turn-w", type=float, default=None,
                    help="price the shipped turn-slot settle bonus reward_facing_turn "
                         "(reward_facing_exp_turn_slot) inside the NEW variant "
                         "(KRABBY_OMNI_FACING_TURN_W twin; exp(-|heading_err|) on the turn holds, "
                         "0 on translation rows; must be >= 0, as parse_facing_turn_w requires; "
                         "default 0.0 = no change, matching the knob's off-by-default shipped "
                         "semantics; C4 candidate value 0.8)")
    ap.add_argument("--coupled-sigma", type=float, default=None,
                    help="price the shipped track_lin_vel_xy_exp_facing_coupled kernel inside "
                         "the NEW variant (KRABBY_OMNI_TRACK_FACING_SIGMA twin; the stock "
                         "tracking kernel times a Gaussian of the facing error; must be > 0, as "
                         "parse_track_facing_sigma requires); unset (default) = the stock "
                         "kernel, no change; C6 candidate value 1.0")
    ap.add_argument("--combine", nargs="*", default=["sum", "product"])
    ap.add_argument("--n-offsets", type=int, default=36, help="clock-phase fit grid size")
    ap.add_argument("--limit", type=int, default=0, help="episodes per head (0 = all)")
    ap.add_argument("--upright-only", action="store_true",
                    help="sensitivity: drop steps where the clock terms' upright gate is closed")
    ap.add_argument("--offset-report", action="store_true")
    ap.add_argument("--idle", choices=["carry", "reset"], default="carry",
                    help="foot-idle counter: shipped episode-long carry-over, or restarted "
                         "at each hold boundary (sensitivity; see episode_primitives)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "replay_reward_economy.md"))
    args = ap.parse_args()

    # Part A5 levers: which were SUPPLIED (arms the turn-hold table), then their resolved values.
    # The domains are the shipped knob parsers' own (omni_knobs.parse_facing_turn_w requires
    # >= 0, parse_track_facing_sigma requires > 0), so the replay cannot price a recipe the
    # trainer would refuse at config build.
    # ``--new-progress-w``/``--progress-w`` (C9) arms the table on the SAME supplied-ness rule: it
    # is ONE flag with two spellings (a true argparse alias, one dest) parsing to None when unset,
    # so the two spellings stay byte-identical to each other, a run with NO new flag stays
    # byte-identical to before this pass (the weight resolves to the value the flag defaulted to),
    # and C9's own declared replay (``--progress-w 1.5``, nothing else) gets the turn-hold table
    # instead of a screen report missing it. C9 is priced in the per-hold, GATE and sweep tables
    # exactly as ``--new-progress-w`` always was, and pays the turn-hold table's ``progress``
    # column.
    new_levers_armed = any(v is not None for v in (args.yaw_w, args.yaw_progress_w,
                                                   args.facing_turn_w, args.coupled_sigma,
                                                   args.new_progress_w))
    if args.facing_turn_w is not None and not args.facing_turn_w >= 0.0:
        ap.error(f"--facing-turn-w must be >= 0 (KRABBY_OMNI_FACING_TURN_W's own domain), "
                 f"got {args.facing_turn_w!r}")
    if args.coupled_sigma is not None and not args.coupled_sigma > 0.0:
        ap.error(f"--coupled-sigma must be > 0 (KRABBY_OMNI_TRACK_FACING_SIGMA's own domain), "
                 f"got {args.coupled_sigma!r}")
    yaw_w = WEIGHTS["reward_tracking_yaw"] if args.yaw_w is None else args.yaw_w
    yaw_progress_w = (WEIGHTS["reward_yaw_progress"] if args.yaw_progress_w is None
                      else args.yaw_progress_w)
    facing_turn_w = 0.0 if args.facing_turn_w is None else args.facing_turn_w
    new_progress_w = NEW.progress_w if args.new_progress_w is None else args.new_progress_w

    global IDLE_KEY
    IDLE_KEY = "idle" if args.idle == "carry" else "idle_reset"

    heads = {}
    clips: dict[str, tuple[float, str]] = {}
    for name, g in (("walker(1a)", args.walker), ("mid(1b)", args.mid), ("creeper(1c)", args.creeper)):
        dirs = sorted(glob.glob(g))
        if not dirs:
            raise SystemExit(f"no raw dir matched {g}")
        clips[name] = head_action_clip(dirs[-1])
        heads[name] = load_head(g, args.n_offsets, args.limit or None, clips[name][0])

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Offline reward-economy replay -- omni 1c fix (2026-09-18)")
    emit()
    emit(f"episodes: " + ", ".join(f"{k} n={len(v)}" for k, v in heads.items()))
    emit("action clip (runner clip_actions, then the action term's per-joint clip): "
         + "; ".join(f"{k} {c:g} [{src}]" for k, (c, src) in clips.items()))
    emit(f"clock-phase fit: {args.n_offsets} offsets/episode, argmax of OLD clock income (schedule[sum]+apex)")
    emit(f"step mask: steady_mask & segment & pre-termination"
         + (" & upright gate open" if args.upright_only else ""))
    emit(f"weights: 1c training-log reward table; income/s = mean_t(weight * value) "
         f"(Isaac Lab accumulates weight*value*dt)")
    emit(f"not replayed: {NOT_REPLAYED}")
    emit()

    if args.offset_report:
        emit("## clock-phase fit sensitivity (income/s of schedule[sum]+apex over the offset grid)")
        emit()
        emit(f"| head | best | mean | worst | spread |")
        emit("|---|---|---|---|---|")
        for name, eps in heads.items():
            b = np.mean([p["offset_scores"].max() for p in eps])
            mn = np.mean([p["offset_scores"].mean() for p in eps])
            wo = np.mean([p["offset_scores"].min() for p in eps])
            emit(f"| {name} | {b:.3f} | {mn:.3f} | {wo:.3f} | {b - wo:.3f} |")
        emit()

    # ---------------- fidelity: replay vs the 1c training log's own income table
    emit("## Reconstruction fidelity -- 1c head, OLD terms, every steady step vs the "
         "1c training log")
    emit()
    emit("Training column = ``Episode_Reward/<name>`` at iteration 6997 / 40 s (the "
         "log's per-second income). The two populations differ (100 canary episodes on a "
         "fixed 8-hold schedule vs 256 training envs on the 1c sampler), so agreement to "
         "~15 % is the expected ceiling; it is the check that the formulas, weights, "
         "gates and the fitted clock phase are right.")
    emit()
    acc: dict[str, float] = {}
    nn = 0
    for p in heads["creeper(1c)"]:
        m = p["steady"] & p["valid"]
        k = int(m.sum())
        if not k:
            continue
        for a, b in term_incomes(p, m, OLD).items():
            acc[a] = acc.get(a, 0.0) + b * k
        nn += k
    rep = {a: b / nn for a, b in acc.items()}
    log_1c = {"track_lin_exp": 0.826, "track_ang_exp": 0.759, "clock_sched": 0.708,
              "clock_apex": 0.224, "facing": 0.180, "progress": 0.027, "yaw_progress": 0.007,
              "air_time": 0.001, "L1": -0.177, "action_rate": -0.220, "foot_idle": -0.060,
              "excess_feet": -0.015, "cross_vel": -0.017}
    emit("| term | replay (canary) | training log 1c @6997 |")
    emit("|---|---|---|")
    for k, v in log_1c.items():
        emit(f"| {k} | {rep.get(k, float('nan')):.3f} | {v:.3f} |")
    emit()

    # ---------------- per-hold OLD vs NEW
    variants = [
        ("OLD", OLD),
        ("NEW", replace(NEW, ang_std=args.ang_std, ang_turn_only=args.new_ang_turn_only,
                        progress_w=new_progress_w, yaw_w=yaw_w,
                        yaw_progress_w=yaw_progress_w, facing_turn_w=facing_turn_w,
                        coupled_sigma=args.coupled_sigma)),
    ]
    # ``facing_turn`` is a COMMAND member, so it must be a DISPLAYED column whenever it can be
    # non-zero -- otherwise the row's COMMAND/POS/NET would not add up from its own columns. It
    # is non-zero iff the C4 weight is armed, so the column is shown exactly then (which also
    # keeps a no-new-flags run's table byte-identical to before this pass).
    facing_turn_cols = ["facing_turn"] if facing_turn_w != 0.0 else []
    emit("## Per-hold income (per second) on the MOVING holds")
    emit()
    for vname, t in variants:
        emit(f"### {vname}  ({t.tag})")
        emit()
        cols = ["track_lin_exp", "L1", "progress", "clock_sched", "clock_apex", "track_ang_exp",
                "facing"] + facing_turn_cols + [
                "yaw_progress", "air_time", "cross_vel", "action_rate", "foot_idle",
                "excess_feet", "orientation", "lin_vel_z", "delta_torque"]
        emit("| hold | head | v_along | " + " | ".join(cols)
             + " | TRACK | COMMAND | REG | POS | NEG | NET |")
        emit("|" + "---|" * (len(cols) + 9))
        for hold in MOVING_HOLDS:
            for hname, eps in heads.items():
                inc, n, v = head_hold_income(eps, hold, t, args.upright_only)
                if not inc:
                    emit(f"| {hold} | {hname} | - | " + " | ".join(["-"] * (len(cols) + 6)) + " |")
                    continue
                pos, neg, net = totals(inc)
                emit(f"| {hold} | {hname} | {v:.3f} | "
                     + " | ".join(f"{inc[c]:.3f}" for c in cols)
                     + f" | {subtotal(inc, TRACKING):.3f} | {subtotal(inc, COMMAND):.3f}"
                     + f" | {subtotal(inc, REGULARIZERS):.3f}"
                     + f" | {pos:.3f} | {neg:.3f} | {net:.3f} |")
        emit()

    # ---------------- Part A5 (step 4): turn-hold income ordering, only when a new lever is armed
    # (keeps a no-new-flags run's output byte-identical to before this pass -- see the dry run in
    # the step report). ``new_levers_armed`` follows the FLAGS, not the values (set above).
    #
    # ADVISORY ONLY, AND ISOLATED. Everything below this block -- the GATE, the sweeps, the
    # VERDICT (G1/G2/G3) and the `--out` file itself -- is the decided output of this screen, so
    # a failure while pricing a candidate must never take it down with it. The whole block is
    # therefore guarded: an exception is reported as one line and the report continues.
    if new_levers_armed:
        try:
            emit("## Turn-hold income ordering -- the terms the A5 levers move, on "
                 "turn_l90/turn_r90/move_turn")
            emit()
            emit("Candidates C2 (--yaw-progress-w), C4 (--facing-turn-w), C6 "
                 "(--coupled-sigma) and C9 (--progress-w / --new-progress-w, one flag under two "
                 "spellings) are priced here on the holds where a turn is commanded, OLD vs the "
                 "armed NEW variant, per head. Any one of them SUPPLIED arms this table (the "
                 "flag, not the value), and C9's weight is what pays the `progress` column. "
                 "YAW = the four facing/turn-rate terms (what C2 and C4 move); TRACK = "
                 "the translation sub-economy (what C6 rescales through track_lin_exp and what "
                 "the NEW variant's progress weight pays through progress); the two clock "
                 "columns are context, in neither subtotal. COMMAND is the row's full command "
                 "subtotal -- the same one the GATE scores -- so it also contains air_time and "
                 "cross_vel, which are NOT columns here: a row's COMMAND is the sum of its "
                 "printed columns only on the two in-place turn holds, where both of those terms "
                 "are identically 0; on move_turn they are not. 'order' ranks the four YAW terms "
                 "by income, descending -- which term dominates the hold's turning economy.")
            emit()
            yaw_cols = ("track_ang_exp", "yaw_progress", "facing", "facing_turn")
            track_cols = TRACKING                     # track_lin_exp, L1, progress
            clock_cols = ("clock_sched", "clock_apex")
            turn_cols = yaw_cols + track_cols + clock_cols
            emit("| hold | head | variant | " + " | ".join(yaw_cols) + " | YAW | "
                 + " | ".join(track_cols) + " | TRACK | " + " | ".join(clock_cols)
                 + " | COMMAND | order (desc, YAW) |")
            emit("|" + "---|" * (len(turn_cols) + 7))
            for hold in TURN_HOLDS:
                for hname, eps in heads.items():
                    for vname, t in variants:
                        inc, n, _ = head_hold_income(eps, hold, t, args.upright_only)
                        if not inc:
                            continue
                        order = ", ".join(
                            f"{c}={inc[c]:+.3f}" for c in sorted(yaw_cols, key=lambda c: -inc[c])
                        )
                        emit(f"| {hold} | {hname} | {vname} | "
                             + " | ".join(f"{inc[c]:.3f}" for c in yaw_cols)
                             + f" | {subtotal(inc, yaw_cols):.3f} | "
                             + " | ".join(f"{inc[c]:.3f}" for c in track_cols)
                             + f" | {subtotal(inc, track_cols):.3f} | "
                             + " | ".join(f"{inc[c]:.3f}" for c in clock_cols)
                             + f" | {subtotal(inc, COMMAND):.3f} | {order} |")
            emit()
        except Exception as exc:  # advisory table only -- never break the decided screen
            emit(f"turn-hold income ordering: error ({type(exc).__name__}: {exc})")
            emit()

    # ---------------- the gate: walker vs creeper margin
    emit("## GATE -- walker vs creeper margin per moving hold")
    emit()
    emit("margin = (walker - creeper) / walker POSITIVE income, on three nested subtotals:")
    emit("  NET     = every replayed term")
    emit("  COMMAND = every term that prices the command "
         "(tracking + clock + ang + facing + yaw + air-time + cross-vel"
         + (" + facing-turn" if facing_turn_w != 0.0 else "")
         + "); excludes the gait/action regularizers, which the fix does not touch")
    emit("  TRACK   = track_lin_exp + L1 + progress, the sub-economy the fix moves directly")
    emit("target: >= +25 % on every moving hold")
    emit()
    emit("| hold | economy | walker v_along | creeper v_along | "
         "TRACK margin | COMMAND margin | NET margin | creeper/walker NET |")
    emit("|---|---|---|---|---|---|---|---|")
    for hold in MOVING_HOLDS:
        for vname, t in variants:
            w_inc, _, wv = head_hold_income(heads["walker(1a)"], hold, t, args.upright_only)
            c_inc, _, cv = head_hold_income(heads["creeper(1c)"], hold, t, args.upright_only)
            if not w_inc or not c_inc:
                continue
            wpos, _, wnet = totals(w_inc)
            _, _, cnet = totals(c_inc)
            mt = (subtotal(w_inc, TRACKING) - subtotal(c_inc, TRACKING)) / wpos
            mc = (subtotal(w_inc, COMMAND) - subtotal(c_inc, COMMAND)) / wpos
            mn = (wnet - cnet) / wpos
            ratio = cnet / wnet if wnet != 0 else float("nan")
            emit(f"| {hold} | {vname} | {wv:.3f} | {cv:.3f} | {mt * 100:+.1f} % | "
                 f"{mc * 100:+.1f} % | {mn * 100:+.1f} % | {ratio * 100:.0f} % |")
    emit()

    # ---------------- sweep
    for label, keys in (("COMMAND", COMMAND), ("NET", None)):
        emit(f"## Sweep -- {label} margin (% of walker positive income) per moving hold")
        emit()
        emit("| sigma^2 | L1 | combine | ang_std | " + " | ".join(MOVING_HOLDS)
             + " | min | fwd>=25% |")
        emit("|" + "---|" * (len(MOVING_HOLDS) + 6))
        sweep_best = None
        for s2 in args.sigma2:
            for l1 in args.l1w:
                for comb in args.combine:
                    t = Terms(s2, l1, args.ang_std, comb)
                    ms = []
                    for hold in MOVING_HOLDS:
                        w_inc, _, _ = head_hold_income(heads["walker(1a)"], hold, t, args.upright_only)
                        c_inc, _, _ = head_hold_income(heads["creeper(1c)"], hold, t, args.upright_only)
                        if not w_inc or not c_inc:
                            ms.append(float("nan"))
                            continue
                        wpos, _, wnet = totals(w_inc)
                        _, _, cnet = totals(c_inc)
                        if keys is None:
                            ms.append((wnet - cnet) / wpos)
                        else:
                            ms.append((subtotal(w_inc, keys) - subtotal(c_inc, keys)) / wpos)
                    mn = float(np.nanmin(ms))
                    if sweep_best is None or mn > sweep_best[0]:
                        sweep_best = (mn, t)
                    emit(f"| {s2:g} | {l1:g} | {comb} | {args.ang_std:g} | "
                         + " | ".join(f"{m * 100:+.1f}" for m in ms)
                         + f" | {mn * 100:+.1f} | {'yes' if ms[0] >= 0.25 else 'no'} |")
        emit()
        if sweep_best:
            emit(f"best swept point by worst-hold {label} margin: {sweep_best[1].tag} "
                 f"-> {sweep_best[0] * 100:+.1f} %")
            emit()

    # ---------------- how much of the income still requires moving at all
    emit("## Motion-requiring income (progress + yaw_progress + air_time) as a share of "
         "POSITIVE income")
    emit()
    emit("The diagnosis's headline number: under 1c's terms only 1.2 % of the creeper's "
         "income required moving. If the fix does not move this, standing stays affordable.")
    emit()
    emit("| hold | head | OLD | NEW |")
    emit("|---|---|---|---|")
    for hold in MOVING_HOLDS:
        for hname, eps in heads.items():
            cells = []
            for _, t in variants:
                inc, _, _ = head_hold_income(eps, hold, t, args.upright_only)
                pos, _, _ = totals(inc)
                cells.append(subtotal(inc, MOTION_REQUIRING) / pos * 100 if pos else float("nan"))
            emit(f"| {hold} | {hname} | {cells[0]:.1f} % | {cells[1]:.1f} % |")
    emit()

    # ---------------- two extra levers, not part of the decision
    emit("## PROBE -- two levers beyond the decided four (fwd hold, COMMAND margin)")
    emit()
    emit("Not part of the 2026-09-18 decision; both are fix candidates the diagnosis already "
         "listed. `ang@turn` zeroes track_ang_vel_z_exp on rows with cmd_wz == 0 (it pays a "
         "motionless robot ~1.0/s on the 88 % of rows that command no turn); `prog=` raises "
         "reward_forward_progress_along_command's weight from 0.6.")
    emit()
    emit("| terms | walker COMMAND | creeper COMMAND | margin | creeper motion-req. share |")
    emit("|---|---|---|---|---|")
    new_t = replace(NEW, ang_std=args.ang_std)
    probes = [
        ("NEW (decided four)", new_t),
        ("NEW + ang@turn", replace(new_t, ang_turn_only=True)),
        ("NEW + prog 1.5", replace(new_t, progress_w=1.5)),
        ("NEW + prog 2.5", replace(new_t, progress_w=2.5)),
        ("NEW + ang@turn + prog 1.5", replace(new_t, ang_turn_only=True, progress_w=1.5)),
        ("NEW + ang@turn + prog 2.5", replace(new_t, ang_turn_only=True, progress_w=2.5)),
    ]
    for label, t in probes:
        w_inc, _, _ = head_hold_income(heads["walker(1a)"], "fwd", t, args.upright_only)
        c_inc, _, _ = head_hold_income(heads["creeper(1c)"], "fwd", t, args.upright_only)
        wpos, _, _ = totals(w_inc)
        cpos, _, _ = totals(c_inc)
        wc, cc = subtotal(w_inc, COMMAND), subtotal(c_inc, COMMAND)
        share = subtotal(c_inc, MOTION_REQUIRING) / cpos * 100 if cpos else float("nan")
        emit(f"| {label} | {wc:.3f} | {cc:.3f} | {(wc - cc) / wpos * 100:+.1f} % | {share:.1f} % |")
    emit()

    # ---------------- ang-std isolation
    emit("## Isolating the four knobs (fwd hold, COMMAND margin walker - creeper, % of walker POS)")
    emit()
    emit("| step | terms | walker COMMAND | creeper COMMAND | margin |")
    emit("|---|---|---|---|---|")
    ladder = [
        ("OLD", OLD),
        ("+ sigma^2 0.04", replace(OLD, sigma2=0.04)),
        ("+ L1 -2.0", replace(OLD, sigma2=0.04, l1_w=-2.0)),
        ("+ ang_std 0.5", replace(OLD, sigma2=0.04, l1_w=-2.0, ang_std=args.ang_std)),
        ("+ combine product (= NEW)", Terms(0.04, -2.0, args.ang_std, "product")),
    ]
    for step, t in ladder:
        w_inc, _, _ = head_hold_income(heads["walker(1a)"], "fwd", t, args.upright_only)
        c_inc, _, _ = head_hold_income(heads["creeper(1c)"], "fwd", t, args.upright_only)
        wpos, _, _ = totals(w_inc)
        wc, cc = subtotal(w_inc, COMMAND), subtotal(c_inc, COMMAND)
        emit(f"| {step} | {t.tag} | {wc:.3f} | {cc:.3f} | {(wc - cc) / wpos * 100:+.1f} % |")
    emit()

    # ---------------- income vs achieved speed (the gradient that has to exist)
    emit("## Tracking sub-economy vs achieved along-command speed (analytic, cmd 0.25 m/s)")
    emit()
    emit("income/s of track_lin_exp + L1 + progress at perfect direction; "
         "the last column is the slope at v = 0, i.e. the gradient PPO sees from a standstill.")
    emit()
    speeds = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    emit("| economy | " + " | ".join(f"v={s:g}" for s in speeds) + " | d/dv at 0 |")
    emit("|" + "---|" * (len(speeds) + 2))
    for vname, t in variants:
        vals = [tracking_income(t, 0.25, s) for s in speeds]
        grad = (tracking_income(t, 0.25, 0.01) - tracking_income(t, 0.25, 0.0)) / 0.01
        emit(f"| {vname} ({t.tag}) | " + " | ".join(f"{v:+.3f}" for v in vals) + f" | {grad:+.2f} |")
    emit()
    if sweep_best:
        emit(f"best swept point by worst-hold margin: {sweep_best[1].tag} -> {sweep_best[0] * 100:+.1f} %")
        emit()

    # ---------------- zero-velocity sign check
    emit("## Zero-velocity check -- the command-tracking sub-economy under a move command")
    emit()
    emit("terms: track_lin_vel_xy_exp + penalty_tracking_error_l1 + forward_progress "
         "(the three terms whose value depends on the achieved planar velocity; on a "
         "pure-translation row cmd_wz = 0 so track_ang / yaw_progress are constants).")
    emit()
    emit("| cmd | economy | v=0 net | v=cmd net | v=0 sign |")
    emit("|---|---|---|---|---|")
    for cmd_speed in (0.22, 0.25):
        for vname, t in variants:
            rows = synthetic_rows(t, cmd_speed)
            z = rows["v=0 (stand/step-in-place)"]
            f = rows[f"v={cmd_speed:g} (perfect track)"]
            emit(f"| {cmd_speed:g} m/s | {vname} | {z:+.3f} | {f:+.3f} | "
                 f"{'POSITIVE' if z > 0 else 'NEGATIVE'} |")
    emit()
    emit("sweep of the same check (v=0 net; must be negative):")
    emit()
    emit("| sigma^2 | L1 | cmd 0.22 | cmd 0.25 |")
    emit("|---|---|---|---|")
    for s2 in args.sigma2 + [0.1]:
        for l1 in args.l1w + [-1.0]:
            t = Terms(s2, l1, args.ang_std, "sum")
            v22 = synthetic_rows(t, 0.22)["v=0 (stand/step-in-place)"]
            v25 = synthetic_rows(t, 0.25)["v=0 (stand/step-in-place)"]
            emit(f"| {s2:g} | {l1:g} | {v22:+.3f} | {v25:+.3f} |")
    emit()

    # ---------------- stand hold, for the "walking must beat standing" reading
    emit("## Reference: the stand hold (commanded stop) under both economies")
    emit()
    emit("| head | economy | POS | NEG | NET |")
    emit("|---|---|---|---|---|")
    for vname, t in variants:
        for hname, eps in heads.items():
            inc, n, _ = head_hold_income(eps, "stand", t, args.upright_only)
            if not inc:
                continue
            pos, neg, net = totals(inc)
            emit(f"| {hname} | {vname} | {pos:.3f} | {neg:.3f} | {net:.3f} |")
    emit()

    # ---------------- verdict
    emit("## VERDICT")
    emit()
    t_new = variants[1][1]
    cmd_of = {"fwd": 0.25, "lat_l": 0.22, "diag": 0.25, "back": 0.22, "move_turn": 0.22}
    walkable, margins = [], {}
    for hold in MOVING_HOLDS:
        w_inc, _, wv = head_hold_income(heads["walker(1a)"], hold, t_new, args.upright_only)
        c_inc, _, cv = head_hold_income(heads["creeper(1c)"], hold, t_new, args.upright_only)
        wpos, _, _ = totals(w_inc)
        margins[hold] = (subtotal(w_inc, COMMAND) - subtotal(c_inc, COMMAND)) / wpos
        # A hold only tests WALKING vs CREEPING if the 1a head actually walks there AND
        # clearly out-travels the 1c head; otherwise the row compares two creepers.
        if wv / cmd_of[hold] >= 0.3 and wv >= 2.0 * max(cv, 1e-6):
            walkable.append(hold)
    emit("holds that actually test walking-vs-creeping (1a speed ratio >= 0.3 AND 1a "
         f"out-travels 1c by >= 2x): {walkable or ['NONE']} -- on every other moving hold "
         "the '1a walker' creeps too, so the margin there compares two creepers and the "
         "sign carries no information about whether walking pays.")
    g1 = all(margins[h] >= 0.25 for h in walkable) if walkable else False
    g2 = all(m >= 0.25 for m in margins.values())
    z22 = tracking_income(t_new, 0.22, 0.0)
    z25 = tracking_income(t_new, 0.25, 0.0)
    g3 = z22 < 0 and z25 < 0
    emit()
    emit(f"- G1 COMMAND margin >= 25 % on every hold with a genuine walker: "
         f"{'PASS' if g1 else 'FAIL'} ("
         + ", ".join(f"{h} {margins[h] * 100:+.1f} %" for h in walkable) + ")")
    emit(f"- G2 COMMAND margin >= 25 % on EVERY moving hold: {'PASS' if g2 else 'FAIL'} ("
         + ", ".join(f"{h} {m * 100:+.1f} %" for h, m in margins.items()) + ")")
    emit(f"- G3 zero-velocity tracking income negative at both commanded speeds: "
         f"{'PASS' if g3 else 'FAIL'} (0.22 -> {z22:+.3f}, 0.25 -> {z25:+.3f})")
    emit()
    emit(f"GATE: {'PASS' if (g1 and g3) else 'FAIL'} "
         f"(on the G1 + G3 reading; G2 is not decidable from these raws -- see the "
         f"walkable-hold line above)")
    emit()

    # ---------------- candidate sets, side by side
    emit("### Candidate term sets on the decidable hold (fwd)")
    emit()
    emit("| set | terms | COMMAND margin | G1 | v=0 @0.22 | v=0 @0.25 | G3 |")
    emit("|---|---|---|---|---|---|---|")
    base = replace(NEW, ang_std=args.ang_std)
    cands = [
        ("OLD (1c as run)", OLD),
        ("D: decided four", base),
        ("D + ang@turn", replace(base, ang_turn_only=True)),
        ("D + ang@turn + prog 1.5", replace(base, ang_turn_only=True, progress_w=1.5)),
        ("D + ang@turn + prog 2.5", replace(base, ang_turn_only=True, progress_w=2.5)),
        ("D with L1 -1.5", replace(base, l1_w=-1.5)),
        ("D with L1 -3.0", replace(base, l1_w=-3.0)),
        ("D with sigma^2 0.06", replace(base, sigma2=0.06)),
        ("D with combine sum", replace(base, combine="sum")),
    ]
    for label, t in cands:
        w_inc, _, _ = head_hold_income(heads["walker(1a)"], "fwd", t, args.upright_only)
        c_inc, _, _ = head_hold_income(heads["creeper(1c)"], "fwd", t, args.upright_only)
        wpos, _, _ = totals(w_inc)
        m = (subtotal(w_inc, COMMAND) - subtotal(c_inc, COMMAND)) / wpos
        a, b = tracking_income(t, 0.22, 0.0), tracking_income(t, 0.25, 0.0)
        emit(f"| {label} | {t.tag} | {m * 100:+.1f} % | {'PASS' if m >= 0.25 else 'FAIL'} | "
             f"{a:+.3f} | {b:+.3f} | {'PASS' if (a < 0 and b < 0) else 'FAIL'} |")
    emit()

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
