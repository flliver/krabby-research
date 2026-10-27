#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Offline reward-replay gate for the gait-motion-quality campaign (plan step 3). CPU ONLY.

USAGE
    <isaac_venv>/bin/python replay_motion_terms.py --run-dir <suite__omni stamp dir> --out <dir>
        [--train-env-yaml P] [--train-agent-yaml P] [--train-log-dir D]
        [--noise-sigmas 0.5,2.0] [--noise-samples 4] [--per-episode] [--limit N]
        [--noise-tax] [--mu-bound B] [--action-rate-weights -0.3,0] [--checkpoint P]

Reads one ``suite__omni`` eval stamp (``run_meta.json``, ``raw/episode_*.npz``,
``metrics/episode_*.json``; ``complete.json`` required), replays every reward term the campaign's
screens ADD or REWEIGHT on each episode's OWN hold, and writes ``REPLAY_GATE.md`` +
``replay_gate.json`` into ``--out``. It never touches Isaac, a GPU, ``candidates.yaml`` or git.

THE TWO GATES (plan step 3; ``2026-09-22_gait_motion_quality/CHARTER.md``)

* **Gate 1 -- the term tracks the defect.** Within each category where the term is active, the
  per-episode cost, normalised the way block Q normalises (per metre on fwd/side/back/diag, per
  radian on turn/move_turn, per second on stand), must rank episodes like the step-1 motion
  metrics: Spearman rho >= 0.5 against ``action_jitter`` OR ``contact_switch_hz`` (computed from
  the npz by ``gait_eval.metrics.motion_quality_metrics`` -- the eval's own function, imported).
  For ``KRABBY_AIRTIME_W`` (a DE-weight of an income) the test is the other way round: the income
  must correlate positively with the swings / contact switches it is suspected of paying for.
  A FAIL is reported, never softened: the term is dropped from its screen unless the user overrules.
* **Gate 2 -- the term does not swamp the task.** On the best-tracking quartile of each category
  (ranked by the replayed task-tracking income per second), the term's cost is < 15 % of the
  task-tracking income. Each ``TBD_replay_gate`` weight is proposed as the magnitude that makes the
  cost ~= 10 % of that income on the quartile (median over episodes), bound by the category where
  that magnitude is smallest, rounded to 2 significant figures.

HOW EACH TERM IS REPLAYED (the training code, transcribed; reward manager: ``value * weight * dt``)

Timing. ``eval_crab_hex_gait.py`` records row ``t`` BEFORE ``env.step(actions[t])``: ``actions[t]``
and ``cmd_applied[t]`` belong to env step ``t``, the post-physics state of step ``t`` (what the
reward of step ``t`` reads) is row ``t + 1``. Every per-step series below is indexed by env step
``s = 0 .. T-2``: actions/command from row ``s``, state from row ``s + 1``. The eval stores the
policy's PRE-clip mean; the runner's ``clip_actions`` (1.0) is applied first, as
``RslRlVecEnvWrapper`` does before the action term sees it.

``penalty_mechanical_power`` (``KRABBY_POWER_W``)   ``sum |tau * qdot|`` over EVERY joint (24:
    rods, camshafts and the ``*_Body_Hip_`` linkage emulator -- ``SceneEntityCfg("robot")``).
``reward_action_rate`` (stock -0.3; ``KRABBY_ACTION_RATE_SQ=1`` squares it)  ``||r_s - r_{s-1}||``
    of the action term's ``raw_actions``, which under ``use_delay`` / ``action_delay_steps [1, 1]``
    is the clipped action DELAYED one step (``r_s = a_{s-1}``); the term's own buffer and the
    action history are zeroed at reset.
``penalty_action_smooth2`` (``KRABBY_ACTION_SMOOTH2_W``)  ``||a_s - 2 a_{s-1} + a_{s-2}||^2`` of the
    UNDELAYED clipped history (``action_history_buf[:, -1..-3]``), 0 while ``episode_length_buf < 3``.
``penalty_joint_vel`` (``KRABBY_JOINT_VEL_W``)  ``sum qdot^2`` over the 12 rod joints.
``penalty_foot_slip`` (``KRABBY_FOOT_SLIP_W``)  ``sum ||v_foot_xy||`` over feet with force > 50 N.
``penalty_stand_pose`` (``KRABBY_STAND_POSE_W``)  ``sum |q - q_default|`` over the rod joints on the
    STAND slot; the eval analogue of ``slot_id == SLOT_STAND`` is the stand category's own hold.
``reward_feet_air_time_positive`` (``KRABBY_AIRTIME_W``, +0.8)  ``sum relu(last_air_time - 0.38) *
    first_contact`` gated on ``||cmd_xy|| > 0.1``, with the contact sensor's air/contact clocks
    re-run step by step (contact = force > the sensor's ``force_threshold``, 1 N). Replayed TWICE:
    ``airtime`` emulates the sensor's float32 ``_timestamp`` bookkeeping exactly as it runs in
    training (see :func:`sensor_elapsed_f32` -- after ~2 s of episode time the float32 elapsed step
    exceeds ``compute_first_contact``'s ``dt + 1e-8`` and first contacts stop registering), and
    ``airtime_exact`` uses exact timing (what the term was meant to pay).
``penalty_foot_idle_when_forward`` (stock -0.12, not a knob here; replayed because the plan names
    it)  ``sum relu(idle_steps - 90)`` with contact at 0.1 N, gated on the omni clock speed
    ``max(||cmd_xy||, 1.2 |cmd_wz|) > 0.12`` (``_cmd_gate``).
Task-tracking income (Gate 2's denominator): ``track_lin_vel_xy_exp`` (1.25, std 0.2 = sigma^2
    0.04), ``track_ang_vel_z_exp`` (1.0, std 0.5), ``reward_tracking_yaw`` = the omni
    ``reward_facing_exp`` (0.2, ``exp(-|heading_err|)``) and ``reward_yaw_progress`` (1.0,
    ``clip(wz_b sign(cmd_wz), +-0.3)`` where ``cmd_wz != 0``). The weights/params above are the
    built-in X08:c6 values; they are re-read from the training ``params/env.yaml`` /
    ``params/agent.yaml`` next to the eval's checkpoint when present (the report lists the source).

KNOWN, UNAVOIDABLE MISMATCHES (stated in the report)

* The eval rolls out the DETERMINISTIC mean; training samples at the policy's sigma (2.0 on X08:c6).
  State-driven terms (power, joint speed, slip, pose) cannot be re-simulated under noise. The three
  action-only terms are additionally Monte-Carlo'd at ``--noise-sigmas`` (``clip(mu + sigma eps)``)
  so the report shows the exploration-noise floor they would carry in training.
* The eval's holds are fixed 12-s commands; training draws slots from the C1d c6 pool.

THE PPO-CONSTRUCTION NOISE TAX (``2026-10-01_ppo_construction``, plan step 2 item 5; all opt-in,
the defaults above are unchanged)

* ``--noise-sigmas`` also takes ``grid`` (= :data:`NOISE_GRID`, incl. 0.8 -- arm A's init sigma --
  and 1.65 -- the side heads' training sigma) and ``ckpt`` (the PER-DIM sigma of the eval's own
  checkpoint, :func:`checkpoint_sigma`: ``std`` or ``exp(log_std)`` clamped to [0.05,
  ``action_std_max``] as ``CrabHexActorCriticRMA`` does, or the soft parameterisation's
  ``std_raw`` mapped through ``log_std = lo + (hi - lo)(tanh(x) + 1)/2``, lo = ln 0.05, hi = ln 2).
* ``--mu-bound B`` models a bound loss: every noise sample is ALSO drawn around ``clip(mu, +-B)``
  (``clip(clip(mu, +-B) + sigma eps, +-clip_actions)``, the SAME eps -- common random numbers), so
  a bounded mean and the recorded saturated mean are priced side by side. Implies ``--noise-tax``.
* ``--noise-tax`` adds :func:`noise_tax`: per category, the ``reward_action_rate`` value per
  second (all scored episodes pooled) for the deterministic mean and every sigma, saturated and
  bounded, priced at each ``--action-rate-weights`` weight (default -0.3 and 0, side by side) as a
  cost and as the net task income, against the fast-vs-slow income margin
  :data:`FAST_SLOW_MARGIN` (+0.29 to +0.32 /s).
* ``side22_25__omni`` / ``side22_25lr__omni`` stamps are category-suite stamps (stand + side), so
  they replay like ``suite__omni`` (only the categories they carry are scored).
* The stand gate is the eval's stand hold, not ``stand_mask()``.
* ``reward_tracking_yaw`` reads ``heading_err`` of row ``s + 1`` (whose ``heading_target`` equals step
  ``s``'s except on a hold boundary).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve()
OMNI = HERE.parents[2]                                   # .../crab_hex_omni_task
TASKS = OMNI.parent                                      # .../parkour_tasks/parkour_tasks
GAIT_SCRIPTS = TASKS / "crab_hex_forward_task" / "scripts"   # `from gait_eval import metrics`
CAMPAIGN_DIR = OMNI / "experiments" / "2026-09-22_gait_motion_quality"

CATEGORIES = ("stand", "fwd", "side", "back", "diag", "turn", "move_turn")
TRANSLATE = ("fwd", "side", "back", "diag")
TURNING = ("turn", "move_turn")
#: the unit block Q normalises each category by (and Gate 1 therefore normalises the cost by)
NORM_UNIT = {"stand": "s", **{c: "m" for c in TRANSLATE}, **{c: "rad" for c in TURNING}}

#: Windows shorter than this (a fall early in the own hold) are not scored -- the eval's own
#: ``min_window_s`` (1.0 s at dt 0.02).
MIN_WINDOW_STEPS = 50
#: Fewer paired finite values than this and a rank correlation is not reported.
MIN_RHO_N = 8

RHO_MIN = 0.5
TARGET_SHARE = 0.10
GATE2_MAX_SHARE = 0.15
QUARTILE = 0.25
MOTION_TOUCH_REF_N = 50.0                   # == gait_eval.metrics.MOTION_TOUCH_REF_N (pinned by test)

#: ``--noise-sigmas grid`` (PPO-construction plan step 2 item 5): 0.8 = arm A's init sigma, 1.65 =
#: the side heads' training sigma, 2.0 = the stock clamp ceiling.
NOISE_GRID = (0.5, 0.8, 1.0, 1.65, 2.0)
#: ``--noise-sigmas ckpt``: the label of the checkpoint's own per-dim sigma.
CKPT_SIGMA = "ckpt"
#: ``CrabHexActorCriticRMA``'s clamp (``ACTION_STD_MIN`` / stock ``ACTION_STD_MAX``).
STD_CLAMP_MIN = 0.05
STD_CLAMP_MAX = 2.0
#: ``KRABBY_STD_PARAM=soft``: log_std = lo + (hi - lo) * (tanh(x) + 1) / 2 over [ln 0.05, ln 2].
SOFT_LOG_STD_LO = math.log(0.05)
SOFT_LOG_STD_HI = math.log(2.0)
#: The fast-vs-slow income margin the noise tax is read against (RL review section 4.3; side-bisect
#: HANDOFF_PPO section 7.E: the fast gait out-earns the two-direction nets by +0.29 to +0.32 /s).
FAST_SLOW_MARGIN = (0.29, 0.32)
#: ``--action-rate-weights`` default: the stock weight and arm B's ``KRABBY_ACTION_RATE_W=0``.
ACTION_RATE_WEIGHTS = (-0.3, 0.0)


# ============================================================================ training parameters
@dataclass
class TrainParams:
    """Every number the replay needs from the TRAINING config. Defaults = X08:c6's c6 stage
    (``crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/params/{env,agent}.yaml``),
    which is also the fork's config (every screen trains on the same c6 overlay)."""

    source: str = "built-in defaults (X08:c6 params/env.yaml + agent.yaml, 2026-09-22)"
    dt: float = 0.02                     # decimation 4 x sim.dt 0.005
    physics_dt: float = 0.005
    decimation: int = 4
    episode_length_s: float = 40.0
    action_clip: float = 1.0             # agent.yaml clip_actions (RslRlVecEnvWrapper)
    action_delay: int = 1                # actions.joint_pos.action_delay_steps [1, 1], use_delay
    track_lin_w: float = 1.25
    track_lin_std: float = 0.2           # KRABBY_TRACK_SIGMA2 0.04
    track_ang_w: float = 1.0
    track_ang_std: float = 0.5           # KRABBY_OMNI_ANG_VEL_STD 0.5
    facing_w: float = 0.2                # reward_tracking_yaw (omni reward_facing_exp), KRABBY_YAW_W
    yaw_progress_w: float = 1.0
    yaw_progress_cap: float = 0.3        # commands.base_velocity.ang_vel_max
    action_rate_w: float = -0.3
    action_rate_squared: bool = False
    airtime_w: float = 0.8
    airtime_threshold: float = 0.38
    airtime_cmd_gate: float = 0.1        # literal in reward_feet_air_time_positive
    sensor_force_threshold: float = 1.0  # scene.contact_forces.force_threshold
    sensor_update_period: float = 0.02
    foot_idle_w: float = -0.12
    foot_idle_max_steps: int = 90
    foot_idle_force: float = 0.1
    foot_idle_min_cmd: float = 0.12
    clock_turn_r: float = 1.2            # omni_clock_turn_radius
    power_w: float = 0.0
    foot_slip_force: float = 50.0        # motion_knobs.FOOT_SLIP_FORCE_N
    #: robot init_state.joint_pos (= default_joint_pos), name or regex -> rad
    default_joint_pos: dict = field(default_factory=lambda: {
        ".*_Hip_Femur_RevoluteJoint": 0.027169998105984527,
        "FL_Femur_Tibia_RevoluteJoint": 0.31020612922871676,
        "ML_Femur_Tibia_RevoluteJoint": 0.31020612922871676,
        "RL_Femur_Tibia_RevoluteJoint": 0.31020612922871676,
        "FR_Femur_Tibia_RevoluteJoint": -0.31020612922871676,
        "MR_Femur_Tibia_RevoluteJoint": -0.31020612922871676,
        "RR_Femur_Tibia_RevoluteJoint": -0.31020612922871676,
    })
    notes: list = field(default_factory=list)


def _yaml_load_tolerant(path: Path) -> dict:
    """Load an Isaac Lab cfg dump without executing anything: ``!!python/tuple`` -> tuple, every
    other python tag -> None (``class_type`` strings stay strings; slices/objects are not needed)."""
    import yaml  # noqa: PLC0415

    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor("tag:yaml.org,2002:python/tuple",
                            lambda loader, node: tuple(loader.construct_sequence(node)))
    _Loader.add_multi_constructor("", lambda loader, suffix, node: None)
    with open(path) as fh:
        return yaml.load(fh, Loader=_Loader)  # noqa: S506 -- SafeLoader subclass


def resolve_default_pos(table: dict, joint_names: Sequence[str]) -> np.ndarray:
    """``default_joint_pos`` per articulation joint from an ``init_state.joint_pos`` table (exact
    name first, else the first regex that fullmatches, else 0 -- Isaac Lab's resolution)."""
    out = np.zeros(len(joint_names))
    for i, name in enumerate(joint_names):
        if name in table:
            out[i] = float(table[name])
            continue
        for key, val in table.items():
            if re.fullmatch(str(key), name):
                out[i] = float(val)
                break
    return out


def train_params_from_dumps(env: dict | None, agent: dict | None, source: str) -> TrainParams:
    """:class:`TrainParams` from a training run's ``params/env.yaml`` / ``agent.yaml`` dicts.

    Every field the dump does not carry keeps its X08:c6 default; a reward term whose ``func`` is
    not the one this replay transcribes is noted (the replay would then not match training)."""
    p = TrainParams(source=source)
    if env:
        sim = env.get("sim") or {}
        if env.get("decimation") and sim.get("dt"):
            p.decimation = int(env["decimation"])
            p.physics_dt = float(sim["dt"])
            p.dt = p.decimation * p.physics_dt
        if env.get("episode_length_s"):
            p.episode_length_s = float(env["episode_length_s"])
        if env.get("omni_clock_turn_radius") is not None:
            p.clock_turn_r = float(env["omni_clock_turn_radius"])
        rw = env.get("rewards") or {}

        def term(name):
            t = rw.get(name) or {}
            return float(t.get("weight") or 0.0), (t.get("params") or {}), str(t.get("func") or "")

        w, prm, fn = term("track_lin_vel_xy_exp")
        if fn and not fn.endswith(":track_lin_vel_xy_exp"):
            p.notes.append(f"track_lin_vel_xy_exp func is {fn}; the replay uses the stock kernel")
        p.track_lin_w, p.track_lin_std = w, float(prm.get("std", p.track_lin_std))
        w, prm, fn = term("track_ang_vel_z_exp")
        if fn and not fn.endswith(":track_ang_vel_z_exp"):
            p.notes.append(f"track_ang_vel_z_exp func is {fn}; the replay uses the stock kernel")
        p.track_ang_w, p.track_ang_std = w, float(prm.get("std", p.track_ang_std))
        w, prm, fn = term("reward_tracking_yaw")
        if fn and not fn.endswith(":reward_facing_exp"):
            p.notes.append(f"reward_tracking_yaw func is {fn}; the replay uses exp(-|heading_err|)")
        p.facing_w = w
        w, prm, fn = term("reward_yaw_progress")
        p.yaw_progress_w = w
        cap = prm.get("clip")
        if cap is None:
            cap = ((env.get("commands") or {}).get("base_velocity") or {}).get("ang_vel_max")
        if cap is not None:
            p.yaw_progress_cap = float(cap)
        w, prm, fn = term("reward_action_rate")
        p.action_rate_w, p.action_rate_squared = w, bool(prm.get("squared", False))
        w, prm, fn = term("reward_feet_air_time_positive")
        p.airtime_w, p.airtime_threshold = w, float(prm.get("threshold", p.airtime_threshold))
        w, prm, fn = term("penalty_foot_idle_when_forward")
        p.foot_idle_w = w
        p.foot_idle_max_steps = int(prm.get("max_idle_steps", p.foot_idle_max_steps))
        p.foot_idle_force = float(prm.get("contact_force_threshold", p.foot_idle_force))
        p.foot_idle_min_cmd = float(prm.get("min_forward_speed_cmd", p.foot_idle_min_cmd))
        p.power_w, _, _ = term("penalty_mechanical_power")
        for added in ("penalty_action_smooth2", "penalty_joint_vel", "penalty_foot_slip", "penalty_stand_pose"):
            if rw.get(added):
                p.notes.append(f"the training cfg ALREADY carries {added} (weight {term(added)[0]})")
        sensor = (env.get("scene") or {}).get("contact_forces") or {}
        if sensor.get("force_threshold") is not None:
            p.sensor_force_threshold = float(sensor["force_threshold"])
        if sensor.get("update_period") is not None:
            p.sensor_update_period = float(sensor["update_period"])
        act = (env.get("actions") or {}).get("joint_pos") or {}
        if act:
            steps = list(act.get("action_delay_steps") or [])
            p.action_delay = int(steps[0]) if (act.get("use_delay", True) and steps) else 0
        init = (((env.get("scene") or {}).get("robot") or {}).get("init_state") or {}).get("joint_pos")
        if isinstance(init, dict) and init:
            p.default_joint_pos = {str(k): float(v) for k, v in init.items()}
    if agent and agent.get("clip_actions") is not None:
        p.action_clip = float(agent["clip_actions"])
    return p


# ============================================================================ pure replay math
def clip_actions(actions: np.ndarray, clip: float | None) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float64)
    return a if clip is None else np.clip(a, -float(clip), float(clip))


def soft_std(x) -> np.ndarray:
    """``KRABBY_STD_PARAM=soft``: sigma from the raw parameter ``x`` (tanh-bounded log-std)."""
    x = np.asarray(x, dtype=np.float64)
    return np.exp(SOFT_LOG_STD_LO + 0.5 * (SOFT_LOG_STD_HI - SOFT_LOG_STD_LO) * (np.tanh(x) + 1.0))


def sigma_from_state_dict(sd: dict, *, std_max: float = STD_CLAMP_MAX) -> tuple[np.ndarray, str]:
    """Per-dim sigma from a policy ``model_state_dict``: the soft ``std_raw`` (bounded by
    construction), else ``log_std`` / ``std`` clamped to ``[STD_CLAMP_MIN, std_max]`` exactly as
    ``CrabHexActorCriticRMA.update_distribution`` does. Keys are matched on their last dotted
    component; exactly one std parameter must exist."""
    found = {}
    for k, v in sd.items():
        last = str(k).rsplit(".", 1)[-1]
        if last in ("std_raw", "log_std", "std"):
            found.setdefault(last, []).append((k, np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach")
                                                              else v, dtype=np.float64).reshape(-1)))
    for kind in ("std_raw", "log_std", "std"):
        if kind in found:
            if len(found[kind]) != 1:
                raise SystemExit(f"checkpoint has {len(found[kind])} '{kind}' parameters: {[k for k, _ in found[kind]]}")
            key, x = found[kind][0]
            if kind == "std_raw":
                return soft_std(x), f"{key} (soft: tanh-bounded log-std)"
            sig = np.exp(x) if kind == "log_std" else x
            return np.clip(sig, STD_CLAMP_MIN, std_max), f"{key} (clamped to [{STD_CLAMP_MIN}, {std_max:g}])"
    raise SystemExit(f"no std / log_std / std_raw parameter in the checkpoint ({sorted(sd)[:8]} ...)")


def checkpoint_sigma(ckpt: Path, agent: dict | None = None) -> tuple[np.ndarray, str]:
    """``(sigma per action dim, source)`` of a checkpoint (CPU load; ``action_std_max`` from the
    training ``agent.yaml``'s ``policy`` block when present)."""
    import torch  # noqa: PLC0415

    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    std_max = ((agent or {}).get("policy") or {}).get("action_std_max")
    sig, src = sigma_from_state_dict(sd, std_max=STD_CLAMP_MAX if std_max is None else float(std_max))
    return sig, f"{ckpt}: {src}"


def parse_noise_sigmas(spec: str) -> list:
    """``--noise-sigmas``: comma list of floats, ``grid`` (:data:`NOISE_GRID`) and ``ckpt``
    (:data:`CKPT_SIGMA`, resolved by :func:`load_run`). Order kept, duplicates dropped."""
    out: list = []
    for tok in (t.strip() for t in str(spec).split(",")):
        if not tok:
            continue
        items = list(NOISE_GRID) if tok == "grid" else ([CKPT_SIGMA] if tok == CKPT_SIGMA else [float(tok)])
        for it in items:
            if it not in out:
                out.append(it)
    return out


def sigma_label(sigma) -> str:
    """The key a noise spec is stored under (floats exactly as before: ``f"{sigma:g}"``)."""
    return sigma if isinstance(sigma, str) else f"{sigma:g}"


def _sig_sort_key(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        return math.inf


def delayed_actions(a: np.ndarray, delay: int) -> np.ndarray:
    """The action term's ``raw_actions``: ``action_history_buf[:, -1 - delay]`` (zeros before the
    episode's first action -- the history is zero-filled at reset)."""
    if delay <= 0:
        return a.copy()
    r = np.zeros_like(a)
    r[delay:] = a[:-delay]
    return r


def action_rate(a: np.ndarray, *, delay: int = 1, squared: bool = False) -> np.ndarray:
    """``reward_action_rate`` per env step: ``||r_s - r_{s-1}||`` (``squared``: its square), with
    the term's ``previous_actions`` zeroed at reset (``r_{-1} = 0``)."""
    r = delayed_actions(a, delay)
    prev = np.vstack([np.zeros((1, r.shape[1])), r[:-1]])
    sq = np.sum(np.square(r - prev), axis=1)
    return sq if squared else np.sqrt(sq)


def action_second_diff(a: np.ndarray) -> np.ndarray:
    """``penalty_action_second_diff`` per env step: ``||a_s - 2 a_{s-1} + a_{s-2}||^2``, masked
    (0) for ``s < 2`` (``episode_length_buf >= 3`` is ``s >= 2``)."""
    out = np.zeros(a.shape[0])
    if a.shape[0] >= 3:
        d2 = a[2:] - 2.0 * a[1:-1] + a[:-2]
        out[2:] = np.sum(np.square(d2), axis=1)
    return out


def joint_vel_l2(qdot: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """``penalty_joint_vel_l2``: ``sum(qdot[:, idx]^2)``."""
    return np.sum(np.square(np.asarray(qdot, dtype=np.float64)[:, list(idx)]), axis=1)


def mechanical_power(tau: np.ndarray, qdot: np.ndarray, idx: Sequence[int] | None = None) -> np.ndarray:
    """``penalty_mechanical_power``: ``sum |tau * qdot|`` (over ``idx``; None = every joint)."""
    p = np.abs(np.asarray(tau, dtype=np.float64) * np.asarray(qdot, dtype=np.float64))
    return np.sum(p if idx is None else p[:, list(idx)], axis=1)


def loaded_foot_slip(force: np.ndarray, foot_vel_w: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """``penalty_loaded_foot_slip``: summed planar speed of the feet whose force > ``threshold``."""
    speed = np.linalg.norm(np.asarray(foot_vel_w, dtype=np.float64)[..., :2], axis=-1)
    return np.sum(speed * (np.asarray(force) > threshold), axis=1)


def stand_pose_l1(q: np.ndarray, q0: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """``penalty_stand_pose_l1`` before its stand gate: ``sum |q - q_default|`` over ``idx``."""
    ids = list(idx)
    return np.sum(np.abs(np.asarray(q, dtype=np.float64)[:, ids] - np.asarray(q0)[ids]), axis=1)


def clock_command_speed(cmd: np.ndarray, turn_r: float) -> np.ndarray:
    """The omni ``clock_command_speed``: ``max(||cmd_xy||, turn_r * |cmd_wz|)``."""
    cmd = np.asarray(cmd, dtype=np.float64)
    return np.maximum(np.linalg.norm(cmd[:, :2], axis=1), turn_r * np.abs(cmd[:, 2]))


def sensor_elapsed_f32(n_steps: int, *, physics_dt: float = 0.005, decimation: int = 4,
                       update_period: float = 0.02) -> np.ndarray:
    """The contact sensor's per-env-step ``elapsed_time`` exactly as float32 Isaac Lab computes it.

    ``SensorBase.update`` runs every physics substep: ``_timestamp += physics_dt`` (float32, reset
    to 0 with the episode) and ``_is_outdated |= t - t_last + 1e-6 >= update_period``; the contact
    sensor (``history_length > 0``) then refreshes at once with ``elapsed = t - t_last`` and sets
    ``t_last = t``. The refresh the observations force right after a reset is taken to consume
    ``t = 0``. As ``t`` grows, float32 rounding moves ``elapsed`` off 0.02 by up to ~1e-6 --
    and ``ContactSensor.compute_first_contact(dt)`` tests ``current_contact_time < dt + 1e-8``.
    Past ~16 s the rounding also skips a refresh now and then (``elapsed`` 0 for that env step,
    0.025 for the next). Measured: ``elapsed < dt + 1e-8`` on ~95 % of the steps before 2 s of
    episode time and on none after.
    """
    f = np.float32
    t, last = f(0.0), f(0.0)
    pdt, tol, per = f(physics_dt), f(1e-6), f(update_period)
    out = np.zeros(n_steps, dtype=np.float32)
    outdated = False
    for s in range(n_steps):
        el = f(0.0)
        for _ in range(decimation):
            t = f(t + pdt)
            if f(f(t - last) + tol) >= per:
                outdated = True
            if outdated:
                el = f(t - last)
                last = t
                outdated = False
        out[s] = el
    return out


def feet_air_time_positive(force: np.ndarray, cmd: np.ndarray, *, dt: float, sensor_threshold: float,
                           threshold: float, cmd_gate: float = 0.1,
                           elapsed: np.ndarray | None = None) -> np.ndarray:
    """``reward_feet_air_time_positive`` per env step, the contact sensor's clocks re-run.

    ``force[s]`` = foot force after env step ``s``; contact = force > ``sensor_threshold`` (the
    sensor's ``force_threshold``). Per refresh (``ContactSensor._update_buffers_impl``):
    ``first = (air > 0) & contact``; ``last_air = where(first, air + el, last_air)``;
    ``air = where(~contact, air + el, 0)``; ``con = where(contact, con + el, 0)``; the reward reads
    ``first_contact = (con > 0) & (con < dt + 1e-8)`` and pays ``sum relu(last_air - threshold) *
    first_contact``, times ``||cmd_xy|| > cmd_gate``. ``elapsed`` None = exact float64 timing
    (``el = dt``); an array (:func:`sensor_elapsed_f32`) = the float32 clocks training runs.
    """
    force = np.asarray(force)
    S, F = force.shape
    if elapsed is None:
        dtype = np.float64
        el = np.full(S, dt, dtype=np.float64)
    else:
        dtype = np.float32
        el = np.asarray(elapsed, dtype=np.float32)[:S]
    lim = dtype(dt + 1e-8)
    thr = dtype(threshold)
    zero = dtype(0.0)
    contact = force > sensor_threshold
    air = np.zeros(F, dtype=dtype)
    con = np.zeros(F, dtype=dtype)
    last_air = np.zeros(F, dtype=dtype)
    out = np.zeros(S)
    for s in range(S):
        c = contact[s]
        e = el[s]
        if e > zero:
            first = (air > zero) & c
            last_air = np.where(first, air + e, last_air).astype(dtype)
            air = np.where(c, zero, air + e).astype(dtype)
            con = np.where(c, con + e, zero).astype(dtype)
        # e == 0: no sensor refresh inside this env step (float32 rounding past ~16 s of episode
        # time) -- the clocks are stale and the reward reads them unchanged
        fc = (con > zero) & (con < lim)
        out[s] = float(np.sum(np.maximum(last_air - thr, zero) * fc))
    gate = np.linalg.norm(np.asarray(cmd, dtype=np.float64)[:S, :2], axis=1) > cmd_gate
    return out * gate


def foot_idle(force: np.ndarray, cmd: np.ndarray, *, max_idle_steps: int, force_threshold: float,
              min_cmd: float, turn_r: float) -> np.ndarray:
    """``PenaltyFootIdleWhenForward`` per env step: ``idle = 0 if contact else idle + 1`` (from 0 at
    reset), ``sum relu(idle - max_idle_steps)``, gated on the omni ``_cmd_gate`` (clock speed >
    ``min_cmd``)."""
    contact = np.asarray(force) > force_threshold
    S = contact.shape[0]
    steps = np.arange(S)[:, None]
    last = np.maximum.accumulate(np.where(contact, steps, -1), axis=0)
    idle = np.where(contact, 0, steps - last).astype(np.float64)
    pen = np.sum(np.maximum(idle - float(max_idle_steps), 0.0), axis=1)
    return pen * (clock_command_speed(cmd, turn_r)[:S] > min_cmd)


def track_lin_vel_xy_exp(cmd: np.ndarray, v_b: np.ndarray, std: float) -> np.ndarray:
    err = np.sum(np.square(np.asarray(cmd)[:, :2] - np.asarray(v_b)[:, :2]), axis=1)
    return np.exp(-err / std**2)


def track_ang_vel_z_exp(cmd: np.ndarray, w_b: np.ndarray, std: float) -> np.ndarray:
    return np.exp(-np.square(np.asarray(cmd)[:, 2] - np.asarray(w_b)[:, 2]) / std**2)


def facing_exp(heading_err: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(np.asarray(heading_err, dtype=np.float64)))


def yaw_progress(cmd: np.ndarray, w_b: np.ndarray, cap: float) -> np.ndarray:
    wz_cmd = np.asarray(cmd, dtype=np.float64)[:, 2]
    prog = np.clip(np.asarray(w_b, dtype=np.float64)[:, 2] * np.sign(wz_cmd), -cap, cap)
    return prog * (wz_cmd != 0.0)


def step_series(z: dict, p: TrainParams, *, rod_idx: Sequence[int], q0: np.ndarray,
                elapsed_f32: np.ndarray | None = None, action_override: np.ndarray | None = None) -> dict:
    """Every replayed term's per-env-step RAW value (unweighted), length ``T - 1``.

    ``action_override`` (already clipped) replaces the recorded actions for the action-only terms
    (the exploration-noise Monte Carlo); the state-driven terms are then skipped."""
    f64 = np.float64
    T = int(np.asarray(z["actions"]).shape[0])
    S = T - 1
    if S < 1:
        return {}
    a = action_override if action_override is not None else clip_actions(z["actions"], p.action_clip)
    a = np.asarray(a, dtype=f64)[:S]
    out = {
        "action_rate": action_rate(a, delay=p.action_delay, squared=False),
        "action_rate_sq": action_rate(a, delay=p.action_delay, squared=True),
        "smooth2": action_second_diff(a),
    }
    if action_override is not None:
        return out
    cmd = np.asarray(z["cmd_applied"], dtype=f64)[:S]

    def post(key):
        return np.asarray(z[key], dtype=f64)[1:T]

    force, qd, tau = post("foot_force_norm"), post("joint_vel"), post("applied_torque")
    wb = post("root_ang_vel_b")
    out.update({
        "power": mechanical_power(tau, qd),
        "joint_vel": joint_vel_l2(qd, rod_idx),
        "foot_slip": loaded_foot_slip(force, post("foot_lin_vel_w"), p.foot_slip_force),
        "stand_pose": stand_pose_l1(post("joint_pos"), q0, rod_idx),
        "airtime": feet_air_time_positive(
            force, cmd, dt=p.dt, sensor_threshold=p.sensor_force_threshold,
            threshold=p.airtime_threshold, cmd_gate=p.airtime_cmd_gate,
            elapsed=(elapsed_f32 if elapsed_f32 is not None else
                     sensor_elapsed_f32(S, physics_dt=p.physics_dt, decimation=p.decimation,
                                        update_period=p.sensor_update_period))),
        "airtime_exact": feet_air_time_positive(
            force, cmd, dt=p.dt, sensor_threshold=p.sensor_force_threshold,
            threshold=p.airtime_threshold, cmd_gate=p.airtime_cmd_gate, elapsed=None),
        "foot_idle": foot_idle(force, cmd, max_idle_steps=p.foot_idle_max_steps,
                               force_threshold=p.foot_idle_force, min_cmd=p.foot_idle_min_cmd,
                               turn_r=p.clock_turn_r),
        "inc_lin": track_lin_vel_xy_exp(cmd, post("root_lin_vel_b"), p.track_lin_std),
        "inc_ang": track_ang_vel_z_exp(cmd, wb, p.track_ang_std),
        "inc_facing": facing_exp(post("heading_err")) if "heading_err" in z else np.zeros(S),
        "inc_yaw_progress": yaw_progress(cmd, wb, p.yaw_progress_cap),
    })
    return out


def task_income(series_sums: dict, p: TrainParams, *, kernels_only: bool = False) -> float:
    """Weighted task-tracking income from window integrals (``sum value * dt``)."""
    inc = p.track_lin_w * series_sums["inc_lin"] + p.track_ang_w * series_sums["inc_ang"]
    if not kernels_only:
        inc += p.facing_w * series_sums["inc_facing"] + p.yaw_progress_w * series_sums["inc_yaw_progress"]
    return float(inc)


# ============================================================================ statistics
def rankdata(x: Sequence[float]) -> np.ndarray:
    """Average ranks (1-based; ties share the mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float | None], y: Sequence[float | None], min_n: int = MIN_RHO_N) -> tuple[float | None, int]:
    """Spearman rank correlation over the pairs where both are finite; ``(None, n)`` below
    ``min_n`` pairs or when either side is constant."""
    xs, ys = [], []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        xs.append(a)
        ys.append(b)
    n = len(xs)
    if n < min_n:
        return None, n
    rx, ry = rankdata(xs), rankdata(ys)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return None, n
    r = float(np.corrcoef(rx, ry)[0, 1])
    return (r if math.isfinite(r) else None), n


def sig2(x: float) -> float:
    """Round to 2 significant figures."""
    if x == 0.0 or not math.isfinite(x):
        return x
    return float(f"{x:.2g}")


def best_quartile(values: Sequence[float], frac: float = QUARTILE) -> np.ndarray:
    """Indices of the top ``frac`` of ``values`` (at least one; ties broken by order)."""
    v = np.asarray(values, dtype=np.float64)
    k = max(1, int(math.ceil(frac * v.size)))
    return np.argsort(-v, kind="mergesort")[:k]


# ============================================================================ term registry
@dataclass(frozen=True)
class TermSpec:
    key: str                 # step_series key
    knob: str                # the candidates.yaml knob
    screen: str              # M2 / M3 / ...
    training_name: str       # rewards.<name>
    kind: str                # "penalty" | "income"
    tbd: bool                # a TBD_replay_gate weight
    fixed_weight: float | None = None     # the weight it trains at when not TBD
    categories: tuple = CATEGORIES        # where it can be active
    tautological: tuple = ()              # Gate-1 metrics that are (near) the same quantity
    info_only: bool = False


TERMS: tuple[TermSpec, ...] = (
    TermSpec("power", "KRABBY_POWER_W", "M2", "penalty_mechanical_power", "penalty", True),
    TermSpec("smooth2", "KRABBY_ACTION_SMOOTH2_W", "M3", "penalty_action_smooth2", "penalty", True,
             tautological=("action_jitter",)),
    TermSpec("joint_vel", "KRABBY_JOINT_VEL_W", "M3", "penalty_joint_vel", "penalty", True),
    TermSpec("action_rate_sq", "KRABBY_ACTION_RATE_SQ=1 (weight KRABBY_ACTION_RATE_W)", "M3",
             "reward_action_rate (squared)", "penalty", False, fixed_weight=-0.3,
             tautological=("action_jitter",)),
    TermSpec("foot_slip", "KRABBY_FOOT_SLIP_W", "M4", "penalty_foot_slip", "penalty", True),
    TermSpec("airtime", "KRABBY_AIRTIME_W", "M4", "reward_feet_air_time_positive", "income", True,
             fixed_weight=0.8, categories=("fwd", "side", "back", "diag", "move_turn")),
    TermSpec("stand_pose", "KRABBY_STAND_POSE_W", "M6", "penalty_stand_pose", "penalty", True,
             categories=("stand",)),
    # context rows (no weight to set)
    TermSpec("action_rate", "KRABBY_ACTION_RATE_W (stock, unsquared)", "stock", "reward_action_rate",
             "penalty", False, fixed_weight=-0.3, tautological=("action_jitter",), info_only=True),
    TermSpec("airtime_exact", "KRABBY_AIRTIME_W (counterfactual: exact sensor timing)", "M4",
             "reward_feet_air_time_positive", "income", False, fixed_weight=0.8,
             categories=("fwd", "side", "back", "diag", "move_turn"), info_only=True),
    TermSpec("foot_idle", "(stock -0.12; not a knob of this campaign)", "stock",
             "penalty_foot_idle_when_forward", "penalty", False, fixed_weight=-0.12, info_only=True),
)
TERM = {t.key: t for t in TERMS}
ACTION_TERMS = ("action_rate", "action_rate_sq", "smooth2")
INCOME_KEYS = ("inc_lin", "inc_ang", "inc_facing", "inc_yaw_progress")


# ============================================================================ run loading
def gait_metrics():
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import metrics as M  # noqa: PLC0415

    return M


def gait_schedule_defaults() -> dict:
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import schedule as S  # noqa: PLC0415

    return dict(S.DEFAULTS)


def _train_dumps_for(run_meta: dict, env_yaml: Path | None, agent_yaml: Path | None) -> tuple[dict | None, dict | None, str]:
    ckpt = run_meta.get("checkpoint")
    params_dir = Path(ckpt).parent / "params" if ckpt else None
    env_yaml = env_yaml or (params_dir / "env.yaml" if params_dir else None)
    agent_yaml = agent_yaml or (params_dir / "agent.yaml" if params_dir else None)
    env = _yaml_load_tolerant(env_yaml) if env_yaml and Path(env_yaml).is_file() else None
    agent = _yaml_load_tolerant(agent_yaml) if agent_yaml and Path(agent_yaml).is_file() else None
    parts = []
    parts.append(f"env: {env_yaml}" if env is not None else "env: built-in defaults (no env.yaml found)")
    parts.append(f"agent: {agent_yaml}" if agent is not None else "agent: built-in defaults (no agent.yaml found)")
    return env, agent, "; ".join(parts)


def replay_episode(z: dict, *, category: str, hold_idx: int, ended_in_fall: bool, hold_s: float | None,
                   p: TrainParams, M, groups: dict, q0: np.ndarray, fall_trim: int,
                   elapsed_f32: np.ndarray, noise_sigmas: Sequence = (), noise_k: int = 0,
                   rng: np.random.Generator | None = None, mu_bound: float | None = None,
                   sigma_values: dict | None = None) -> dict:
    """One episode: the own hold's window integrals of every term, the tracking income and the
    step-1 motion metrics (``motion_quality_metrics`` over the same whole-hold window the eval
    report scores -- ``report._window_mask`` with ``steady`` all True)."""
    seg = np.asarray(z["segment_id"]).astype(int)
    T = seg.shape[0]
    upper = T
    if ended_in_fall:
        upper = max(0, upper - fall_trim)
    rec = np.zeros(T, dtype=bool)
    rec[:upper] = True
    rec &= seg == hold_idx
    widx = np.flatnonzero(rec)
    rec_out: dict[str, Any] = {"n_window": int(widx.size)}
    if widx.size < MIN_WINDOW_STEPS:
        rec_out["skipped"] = f"window_too_short(<{MIN_WINDOW_STEPS} steps)"
        return rec_out

    def _opt(key):
        return np.asarray(z[key])[widx] if key in z else None

    q = M.motion_quality_metrics(
        category, dt=p.dt, actions=np.asarray(z["actions"])[widx],
        foot_force_norm=np.asarray(z["foot_force_norm"])[widx],
        foot_pos_w=np.asarray(z["foot_pos_w"])[widx], foot_lin_vel_w=np.asarray(z["foot_lin_vel_w"])[widx],
        root_pos_w=np.asarray(z["root_pos_w"])[widx], root_quat_w=np.asarray(z["root_quat_w"])[widx],
        cmd_wz=np.asarray(z["cmd_applied"])[widx, 2], joint_vel=_opt("joint_vel"),
        applied_torque=_opt("applied_torque"), groups=groups, clock_running=_opt("clock_running"),
        rod_speed=_opt("rod_speed"), action_clip=p.action_clip, hold_s_nominal=hold_s,
    )
    ser = step_series(z, p, rod_idx=groups["rod"], q0=q0, elapsed_f32=elapsed_f32)
    step_mask = rec[: T - 1]
    n_steps = int(step_mask.sum())
    sums = {k: float(v[step_mask].sum() * p.dt) for k, v in ser.items()}
    if category != "stand":
        sums["stand_pose"] = 0.0          # the stand gate (stand_mask) is the stand hold
    # the power split by joint group (where does sum |tau qdot| live?)
    qd = np.asarray(z["joint_vel"], dtype=np.float64)[1:T][step_mask]
    tau = np.asarray(z["applied_torque"], dtype=np.float64)[1:T][step_mask]
    split = {g: float(mechanical_power(tau, qd, idx).sum() * p.dt) for g, idx in (
        ("rod", groups["rod"]), ("camshaft", groups["camshaft"]), ("body_hip", groups["body_hip"]))}
    rec_out.update({
        "n_steps": n_steps,
        "duration_s": n_steps * p.dt,
        "sums": sums,
        "power_split": split,
        "income": task_income(sums, p),
        "income_kernels": task_income(sums, p, kernels_only=True),
        "quality": {k: q.get(k) for k in (
            "action_jitter", "contact_switch_hz", "swings_per_m", "swings_per_rad", "slip_per_m",
            "slip_per_rad", "work_per_m", "work_per_rad", "foot_slip", "six_down_frac", "liftoffs",
            "distance_m", "net_yaw_rad", "duration_s")},
    })
    if noise_sigmas and noise_k > 0 and rng is not None:
        mu = np.asarray(z["actions"], dtype=np.float64)
        # --mu-bound: the same eps around clip(mu, +-B) (common random numbers; the rng stream is
        # the one the default run draws, so the saturated numbers do not move)
        mu_b = None if mu_bound is None else np.clip(mu, -float(mu_bound), float(mu_bound))
        noisy: dict[str, dict[str, float]] = {}
        noisy_b: dict[str, dict[str, float]] = {}
        for sigma in noise_sigmas:
            lab = sigma_label(sigma)
            sig = (sigma_values or {}).get(lab, sigma)        # a per-dim vector for CKPT_SIGMA
            if isinstance(sig, str):
                raise SystemExit(f"noise sigma {sig!r} was not resolved to a value")
            acc = {k: 0.0 for k in ACTION_TERMS}
            acc_b = {k: 0.0 for k in ACTION_TERMS}
            for _ in range(noise_k):
                eps = rng.standard_normal(mu.shape)
                a = clip_actions(mu + sig * eps, p.action_clip)
                s2 = step_series(z, p, rod_idx=groups["rod"], q0=q0, action_override=a)
                for k in ACTION_TERMS:
                    acc[k] += float(s2[k][step_mask].sum() * p.dt) / noise_k
                if mu_b is not None:
                    ab = clip_actions(mu_b + sig * eps, p.action_clip)
                    s2b = step_series(z, p, rod_idx=groups["rod"], q0=q0, action_override=ab)
                    for k in ACTION_TERMS:
                        acc_b[k] += float(s2b[k][step_mask].sum() * p.dt) / noise_k
            noisy[lab] = acc
            if mu_b is not None:
                noisy_b[lab] = acc_b
        rec_out["noise"] = noisy
        if mu_b is not None:
            rec_out["noise_bounded"] = noisy_b
    if mu_bound is not None:
        # the deterministic bounded mean (identical to the recorded mean's when B >= clip_actions)
        ab = clip_actions(np.clip(np.asarray(z["actions"], dtype=np.float64), -float(mu_bound),
                                  float(mu_bound)), p.action_clip)
        s2b = step_series(z, p, rod_idx=groups["rod"], q0=q0, action_override=ab)
        rec_out["det_bounded"] = {k: float(s2b[k][step_mask].sum() * p.dt) for k in ACTION_TERMS}
    return rec_out


def load_run(run_dir: Path, *, p_override: tuple[Path | None, Path | None] = (None, None),
             noise_sigmas: Sequence = (0.5, 2.0), noise_k: int = 4, seed: int = 20260923,
             limit: int | None = None, fall_exclusion_s: float | None = None,
             mu_bound: float | None = None, checkpoint: Path | None = None) -> dict:
    """Replay every episode of a suite stamp; returns ``{meta, params, episodes, crosscheck}``."""
    run_dir = Path(run_dir)
    if not (run_dir / "complete.json").is_file():
        raise SystemExit(f"{run_dir}: no complete.json -- the stamp is partial (plan D10); refusing")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if meta.get("sample_actions"):
        raise SystemExit(f"{run_dir}: a --sample-actions (stochastic) rollout feeds no gate; refusing")
    if not meta.get("suite"):
        raise SystemExit(f"{run_dir}: not a category-suite stamp (run_meta has no 'suite')")
    M = gait_metrics()
    env_d, agent_d, src = _train_dumps_for(meta, *p_override)
    p = train_params_from_dumps(env_d, agent_d, src)
    if meta.get("action_clip") is not None and float(meta["action_clip"]) != p.action_clip:
        p.notes.append(f"run_meta action_clip {meta['action_clip']} != training clip_actions {p.action_clip}; "
                       "the training value is used")
    if abs(float(meta.get("dt", p.dt)) - p.dt) > 1e-12:
        raise SystemExit(f"eval dt {meta.get('dt')} != training dt {p.dt}")
    if mu_bound is not None and not float(mu_bound) > 0.0:
        raise SystemExit(f"--mu-bound must be > 0, got {mu_bound!r}")
    sigma_values: dict = {}
    sigma_sources: dict = {}
    if CKPT_SIGMA in list(noise_sigmas):
        ck = checkpoint or (Path(meta["checkpoint"]) if meta.get("checkpoint") else None)
        if ck is None or not Path(ck).is_file():
            raise SystemExit(f"--noise-sigmas {CKPT_SIGMA}: no checkpoint to read sigma from ({ck}); pass --checkpoint")
        sig, src = checkpoint_sigma(Path(ck), agent_d)
        if sig.shape[0] != int(np.asarray(meta.get("action_dim", sig.shape[0]))):
            raise SystemExit(f"checkpoint sigma has {sig.shape[0]} dims, the stamp's action_dim is {meta.get('action_dim')}")
        sigma_values[CKPT_SIGMA] = sig
        sigma_sources[CKPT_SIGMA] = {"source": src, "sigma": [float(x) for x in sig]}
    joint_names = list(meta["joint_names"])
    g = M.motion_joint_groups(joint_names)
    groups = {
        "rod": g["rod"],
        "camshaft": [i for i, n in enumerate(joint_names) if "_Body_CamShaft_" in n],
        "body_hip": [i for i, n in enumerate(joint_names) if "_Body_Hip_" in n],
        "cam": g["cam"], "actuated": g["actuated"],
    }
    if len(groups["rod"]) != 12:
        raise SystemExit(f"expected 12 rod joints, found {len(groups['rod'])} in run_meta joint_names")
    q0 = resolve_default_pos(p.default_joint_pos, joint_names)
    defaults = gait_schedule_defaults()
    fe = float(defaults["fall_exclusion_s"]) if fall_exclusion_s is None else float(fall_exclusion_s)
    fall_trim = int(round(fe / p.dt))
    hold_labels = list(meta["hold_labels"])
    hold_s = (meta["suite"] or {}).get("hold_s") or {}
    reasons = meta.get("termination_reason") or []
    emap = meta.get("env_index_map") or []
    n_total = int(meta.get("episodes_total") or len(emap))
    elapsed = sensor_elapsed_f32(max(int(v) for v in meta["n_steps_per_env"]) + 1,
                                 physics_dt=p.physics_dt, decimation=p.decimation,
                                 update_period=p.sensor_update_period)
    rng = np.random.default_rng(seed)
    episodes = []
    xcheck = {"n": 0, "max_abs_diff": {}, "category_mismatch": 0}
    for i, ent in enumerate(emap[: (limit or n_total)]):
        idx = int(ent["env_index"])
        cat, label = ent.get("category"), ent.get("label")
        if cat not in CATEGORIES or label not in hold_labels:
            continue
        ep_json_path = run_dir / "metrics" / f"episode_{idx:02d}.json"
        ep_json = json.loads(ep_json_path.read_text()) if ep_json_path.is_file() else {}
        if ep_json and ep_json.get("category") not in (None, cat):
            xcheck["category_mismatch"] += 1
        reason = ep_json.get("termination_reason") or (reasons[idx] if idx < len(reasons) else "")
        with np.load(run_dir / "raw" / f"episode_{idx:02d}.npz") as npz:
            z = {k: npz[k] for k in npz.files if k not in ("body_pos_w", "leg_contact_force", "terrain_z",
                                                           "foot_ang_vel_w")}
        rec = replay_episode(z, category=cat, hold_idx=hold_labels.index(label),
                             ended_in_fall=reason in ("fall", "hard_fall"), hold_s=hold_s.get(label),
                             p=p, M=M, groups=groups, q0=q0, fall_trim=fall_trim, elapsed_f32=elapsed,
                             noise_sigmas=noise_sigmas, noise_k=noise_k, rng=rng, mu_bound=mu_bound,
                             sigma_values=sigma_values)
        rec.update({"env_index": idx, "category": cat, "hold": label, "hand": ent.get("hand"),
                    "termination_reason": reason})
        stored = ((ep_json.get("holds") or {}).get(label) or {}).get("quality") if ep_json else None
        if stored and "quality" in rec:
            xcheck["n"] += 1
            for k in ("action_jitter", "contact_switch_hz", "distance_m", "net_yaw_rad"):
                a, b = rec["quality"].get(k), stored.get(k)
                if a is not None and b is not None:
                    d = abs(float(a) - float(b))
                    xcheck["max_abs_diff"][k] = max(xcheck["max_abs_diff"].get(k, 0.0), d)
        episodes.append(rec)
    out = {"run_dir": str(run_dir), "meta": {k: meta.get(k) for k in (
        "checkpoint", "checkpoint_sha256", "scenario_id", "episodes_total", "dt", "eval_sha", "plant",
        "started_utc", "action_clip")}, "params": p, "episodes": episodes, "crosscheck": xcheck,
            "fall_trim_steps": fall_trim, "has_fields": sorted(z.keys()) if episodes else []}
    if sigma_sources:
        out["sigma_sources"] = sigma_sources
    if mu_bound is not None:
        out["mu_bound"] = float(mu_bound)
    return out


# ============================================================================ the gates
def _norm(ep: dict, cat: str) -> float | None:
    q = ep["quality"]
    unit = NORM_UNIT[cat]
    if unit == "s":
        return q.get("duration_s") or None
    M = gait_metrics()
    if unit == "m":
        d = q.get("distance_m")
        return d if d is not None and d >= M.MIN_TRAVEL_M else None
    y = q.get("net_yaw_rad")
    return y if y is not None and y >= M.MIN_NET_YAW_RAD else None


def _swings_norm(ep: dict, cat: str) -> float | None:
    q = ep["quality"]
    if cat in TRANSLATE:
        return q.get("swings_per_m")
    if cat in TURNING:
        return q.get("swings_per_rad")
    return None


def gate1(episodes: list[dict], *, rho_min: float = RHO_MIN) -> dict:
    """Per term x category: Spearman rho of the Q-normalised cost vs action_jitter,
    contact_switch_hz and swings_per_(m|rad); verdicts."""
    by_cat = {c: [e for e in episodes if e["category"] == c and "sums" in e] for c in CATEGORIES}
    out: dict[str, dict] = {}
    for t in TERMS:
        rows = {}
        for c in CATEGORIES:
            eps = by_cat[c]
            if c not in t.categories or not eps:
                rows[c] = {"active": False, "why": "not applicable" if c not in t.categories else "no episodes"}
                continue
            xs, xs_rate, jit, sw, swings = [], [], [], [], []
            for e in eps:
                n = _norm(e, c)
                xs.append(None if n is None else e["sums"][t.key] / n)
                xs_rate.append(e["sums"][t.key] / e["duration_s"])
                jit.append(e["quality"].get("action_jitter"))
                sw.append(e["quality"].get("contact_switch_hz"))
                swings.append(_swings_norm(e, c))
            vals = [x for x in xs if x is not None]
            if not vals or max(abs(v) for v in vals) == 0.0:
                rows[c] = {"active": False, "why": "identically zero on every episode", "n": len(vals)}
                continue
            r_j, n_j = spearman(xs, jit)
            r_s, n_s = spearman(xs, sw)
            r_w, n_w = spearman(xs, swings)
            # context: the cost per SECOND against the two per-step / per-second metrics (like for
            # like -- the gate's per-m / per-rad cost shares no denominator with them)
            r_j_rate, _ = spearman(xs_rate, jit)
            r_s_rate, _ = spearman(xs_rate, sw)
            if t.kind == "income":
                cands = [r for r in (r_s, r_w) if r is not None]
                basis = "contact_switch_hz / swings (income pays the defect)"
            else:
                cands = [r for r in (r_j, r_s) if r is not None]
                basis = "action_jitter / contact_switch_hz"
            best = max(cands) if cands else None
            passed = best is not None and best >= rho_min
            nonc = None
            if t.tautological:
                nonc = r_s is not None and r_s >= rho_min
            nonzero_frac = float(np.mean([abs(v) > 0 for v in vals]))
            rows[c] = {"active": True, "n": len(vals), "nonzero_frac": nonzero_frac,
                       "rho_jitter": r_j, "rho_switch": r_s, "rho_swings": r_w,
                       "rho_jitter_per_s": r_j_rate, "rho_switch_per_s": r_s_rate,
                       "pass": passed, "pass_noncircular": nonc, "basis": basis}
        active = [c for c in CATEGORIES if rows[c].get("active")]
        fails = [c for c in active if not rows[c]["pass"]]
        nonc_fails = [c for c in active if t.tautological and not rows[c].get("pass_noncircular")]
        verdict = "INACTIVE" if not active else ("PASS" if not fails else "FAIL")
        out[t.key] = {"rows": rows, "active": active, "fails": fails, "verdict": verdict,
                      "noncircular_fails": nonc_fails}
    return out


def gate2(episodes: list[dict], p: TrainParams, *, target: float = TARGET_SHARE,
          max_share: float = GATE2_MAX_SHARE, frac: float = QUARTILE) -> dict:
    """Per term x category on the best-tracking quartile: median raw-cost / income, the 10 %
    weight, and the proposal (bound by the smallest 10 % weight over the active categories)."""
    by_cat = {c: [e for e in episodes if e["category"] == c and "sums" in e and e["income"] > 0.0]
              for c in CATEGORIES}
    quart = {}
    for c, eps in by_cat.items():
        if not eps:
            continue
        rate = [e["income"] / e["duration_s"] for e in eps]
        sel = best_quartile(rate, frac)
        quart[c] = {"episodes": [eps[i] for i in sel], "n": len(sel), "n_all": len(eps),
                    "income_per_s_q": float(np.median([rate[i] for i in sel])),
                    "income_per_s_all": float(np.median(rate))}
    out: dict[str, Any] = {"quartile": {c: {k: v for k, v in d.items() if k != "episodes"}
                                        for c, d in quart.items()}}
    terms = {}
    for t in TERMS:
        rows = {}
        for c in CATEGORIES:
            if c not in t.categories or c not in quart:
                continue
            q_eps = quart[c]["episodes"]
            r_q = [e["sums"][t.key] / e["income"] for e in q_eps]
            r_k = [e["sums"][t.key] / e["income_kernels"] for e in q_eps if e["income_kernels"] > 0]
            r_all = [e["sums"][t.key] / e["income"] for e in by_cat[c]]
            med_q = float(np.median(r_q))
            if med_q <= 0.0 and float(np.max(np.abs(r_all))) == 0.0:
                continue                                           # inactive here
            row = {"median_ratio_q": med_q, "median_ratio_q_kernels": float(np.median(r_k)) if r_k else None,
                   "median_ratio_all": float(np.median(r_all)),
                   "w10": (target / med_q) if med_q > 0 else None,
                   "w10_kernels": (target / float(np.median(r_k))) if r_k and np.median(r_k) > 0 else None}
            noise = {}
            for e in q_eps[:1]:
                for sig in (e.get("noise") or {}):
                    noise[sig] = None
            if t.key in ACTION_TERMS:
                for sig in noise:
                    rn = [e["noise"][sig][t.key] / e["income"] for e in q_eps if "noise" in e]
                    noise[sig] = float(np.median(rn)) if rn else None
                row["median_ratio_q_noise"] = noise
            rows[c] = row
        spec = {"rows": rows}
        w10s = {c: r["w10"] for c, r in rows.items() if r.get("w10")}
        if t.tbd and t.kind == "penalty" and w10s:
            bind = min(w10s, key=w10s.get)
            w_prop = -sig2(w10s[bind])
            spec.update({"proposed_weight": w_prop, "binding_category": bind,
                         "unrounded": -w10s[bind]})
            w_eval = w_prop
        else:
            w_eval = t.fixed_weight
        if w_eval is not None:
            for c, r in rows.items():
                r["share_at_weight"] = abs(w_eval) * r["median_ratio_q"]
                r["share_at_weight_kernels"] = (abs(w_eval) * r["median_ratio_q_kernels"]
                                                if r.get("median_ratio_q_kernels") is not None else None)
                r["gate2_pass"] = r["share_at_weight"] < max_share if t.kind == "penalty" else None
                if r.get("median_ratio_q_noise"):
                    r["share_at_weight_noise"] = {s: (abs(w_eval) * v if v is not None else None)
                                                  for s, v in r["median_ratio_q_noise"].items()}
            spec["weight_evaluated"] = w_eval
            if t.kind == "penalty":
                fails = [c for c, r in rows.items() if not r["gate2_pass"]]
                spec["verdict"] = "PASS" if rows and not fails else ("FAIL" if rows else "INACTIVE")
                spec["fails"] = fails
            else:
                spec["verdict"] = "n/a (income)"
        terms[t.key] = spec
    out["terms"] = terms
    return out


def airtime_proposal(g1: dict, g2: dict, p: TrainParams) -> dict:
    """The de-weight of ``KRABBY_AIRTIME_W`` (stock +0.8): halve it (the brief's reading), unless
    the training-faithful replay shows the term pays ~nothing -- then say so."""
    rows = g2["terms"]["airtime"]["rows"]
    rows_x = g2["terms"]["airtime_exact"]["rows"]
    share_now = {c: p.airtime_w * r["median_ratio_q"] for c, r in rows.items()}
    share_exact = {c: p.airtime_w * r["median_ratio_q"] for c, r in rows_x.items()}
    inert = (g1["airtime"]["verdict"] == "INACTIVE"
             or (share_now and max(share_now.values()) < 0.005))
    return {"current_weight": p.airtime_w, "proposed_weight": sig2(p.airtime_w / 2.0),
            "share_now_training_faithful": share_now, "share_exact_timing": share_exact,
            "inert_in_training": bool(inert)}


def screen_totals(episodes: list[dict], p: TrainParams, w: dict, *, frac: float = QUARTILE) -> dict:
    """Median (best-tracking quartile) of each SCREEN's combined cost / income, per category.

    ``w`` holds weight MAGNITUDES: ``smooth2``, ``joint_vel``, ``rate_sq`` (the weight the squared
    action rate trains at) and ``foot_slip``. M3 replaces the stock unsquared action rate
    (``|KRABBY_ACTION_RATE_W|``) by the squared one, so its NET added cost is
    ``smooth2 + joint_vel + rate_sq x sq - |w_rate| x plain``."""
    out = {}
    w_plain = abs(p.action_rate_w)
    for c in CATEGORIES:
        eps = [e for e in episodes if e["category"] == c and "sums" in e and e["income"] > 0.0]
        if not eps:
            continue
        rate = [e["income"] / e["duration_s"] for e in eps]
        sel = [eps[i] for i in best_quartile(rate, frac)]

        def med(fn, sel=sel):
            return float(np.median([fn(e) for e in sel]))

        def m3(e):
            s = e["sums"]
            return (w.get("smooth2", 0.0) * s["smooth2"] + w.get("joint_vel", 0.0) * s["joint_vel"]
                    + w.get("rate_sq", 0.0) * s["action_rate_sq"])

        out[c] = {
            "M3_net_added": med(lambda e: (m3(e) - w_plain * e["sums"]["action_rate"]) / e["income"]),
            "M3_all_terms": med(lambda e: m3(e) / e["income"]),
            "M4_slip": med(lambda e: w.get("foot_slip", 0.0) * e["sums"]["foot_slip"] / e["income"]),
        }
    return out


def m3_alternative(g2: dict, p: TrainParams) -> dict:
    """A screen-level M3 that keeps its NET added cost near the 10 % target: the squared action
    rate at the weight that costs what the stock unsquared term costs (cost-neutral swap, bound
    by the category where that weight is smallest) and the two new terms at half their 10 %
    proposals (5 % each)."""
    plain = g2["terms"]["action_rate"]["rows"]
    sq = g2["terms"]["action_rate_sq"]["rows"]
    w_eq = {c: abs(p.action_rate_w) * plain[c]["median_ratio_q"] / sq[c]["median_ratio_q"]
            for c in plain if c in sq and sq[c]["median_ratio_q"] > 0}
    if not w_eq:
        return {}
    bind = min(w_eq, key=w_eq.get)
    s2 = g2["terms"]["smooth2"].get("proposed_weight")
    jv = g2["terms"]["joint_vel"].get("proposed_weight")
    return {
        "KRABBY_ACTION_RATE_W": -sig2(w_eq[bind]),
        "rate_binding_category": bind,
        "KRABBY_ACTION_SMOOTH2_W": sig2(s2 / 2.0) if s2 is not None else None,
        "KRABBY_JOINT_VEL_W": sig2(jv / 2.0) if jv is not None else None,
    }


def noise_tax(episodes: list[dict], *, weights: Sequence[float] = ACTION_RATE_WEIGHTS,
              margin: tuple[float, float] = FAST_SLOW_MARGIN) -> dict:
    """The PPO-construction offline gate (plan step 2 item 5): what exploration noise costs through
    ``reward_action_rate`` for a saturated vs a bounded mean, per category, at each weight.

    Per category every scored episode is pooled (``sum`` of window integrals / ``sum`` of window
    seconds, so a long hold weighs by its length). ``rate_per_s`` is the UNWEIGHTED per-second value
    of ``||r_s - r_{s-1}||``; at weight ``w`` the cost is ``|w| rate``, the net task income is
    ``income + w rate`` and the noise tax is ``|w| (rate_sigma - rate_mean)`` -- the part the policy
    mean cannot remove. ``vs_margin`` = tax / the fast-vs-slow income margin (low, high)."""
    out: dict[str, Any] = {"weights": [float(w) for w in weights], "margin_per_s": list(margin), "categories": {}}
    for c in CATEGORIES:
        eps = [e for e in episodes if e["category"] == c and "sums" in e]
        if not eps:
            continue
        dur = float(sum(e["duration_s"] for e in eps))
        if dur <= 0.0:
            continue
        income = float(sum(e["income"] for e in eps)) / dur
        det = float(sum(e["sums"]["action_rate"] for e in eps)) / dur
        det_b = (float(sum(e["det_bounded"]["action_rate"] for e in eps)) / dur
                 if all("det_bounded" in e for e in eps) else None)
        rows = []

        def row(mode, sigma, rate, ref):
            r = {"mode": mode, "sigma": sigma, "rate_per_s": rate, "by_weight": {}}
            for w in weights:
                tax = abs(w) * (rate - ref) if ref is not None else None
                r["by_weight"][f"{w:g}"] = {
                    "cost_per_s": abs(w) * rate, "net_income_per_s": income + w * rate,
                    "noise_tax_per_s": tax,
                    "tax_vs_margin": ([tax / margin[0], tax / margin[1]] if tax is not None else None)}
            return r

        rows.append(row("saturated", "0 (mean)", det, det))
        if det_b is not None:
            rows.append(row("bounded", "0 (mean)", det_b, det_b))
        labels = []
        for e in eps[:1]:
            labels = sorted((e.get("noise") or {}), key=_sig_sort_key)
        for lab in labels:
            if all("noise" in e and lab in e["noise"] for e in eps):
                rows.append(row("saturated", lab, float(sum(e["noise"][lab]["action_rate"] for e in eps)) / dur, det))
            if all("noise_bounded" in e and lab in e["noise_bounded"] for e in eps):
                rows.append(row("bounded", lab,
                                float(sum(e["noise_bounded"][lab]["action_rate"] for e in eps)) / dur, det_b))
        out["categories"][c] = {"n": len(eps), "seconds": dur, "income_per_s": income, "rows": rows}
    return out


def render_noise_tax(nt: dict, *, mu_bound: float | None, sigma_sources: dict | None) -> list[str]:
    L: list[str] = []
    A = L.append
    ws = nt["weights"]
    A("## Action-rate noise tax: bounded vs saturated mean (PPO-construction offline gate)")
    A("")
    A("`reward_action_rate` per second for the recorded (saturated) mean and for "
      + (f"`clip(mu, +-{mu_bound:g})` (a bound loss's mean) " if mu_bound is not None else "")
      + "under `clip(mu + sigma eps, +-clip_actions)` (same eps for both means), all scored episodes "
      "pooled per category. The noise tax is the cost ABOVE the deterministic mean's at that weight; "
      f"read it against the fast-vs-slow income margin (+{nt['margin_per_s'][0]:.2f} to "
      f"+{nt['margin_per_s'][1]:.2f} /s).")
    if sigma_sources:
        for lab, d in sigma_sources.items():
            A(f"- sigma `{lab}`: {d['source']}; per dim {', '.join(f'{x:.3f}' for x in d['sigma'])}")
    A("")
    head = "| category | mean | sigma | rate /s | " + " | ".join(
        f"cost at w {w:g} | net income at w {w:g} | noise tax at w {w:g} (x margin)" for w in ws) + " |"
    A(head)
    A("|---|---|---|---|" + "---|---|---|" * len(ws))
    for c, d in nt["categories"].items():
        for r in d["rows"]:
            cells = []
            for w in ws:
                b = r["by_weight"][f"{w:g}"]
                vm = b.get("tax_vs_margin")
                cells.append(f"{_f(b['cost_per_s'])} | {_f(b['net_income_per_s'])} | {_f(b['noise_tax_per_s'])}"
                             + (f" ({vm[1]:.2f}-{vm[0]:.2f}x)" if vm else ""))
            A(f"| {c} (income {d['income_per_s']:.3f}/s, n {d['n']}) | {r['mode']} | {r['sigma']} | "
              f"{_f(r['rate_per_s'])} | " + " | ".join(cells) + " |")
    A("")
    return L


def train_log_crosscheck(run: dict, log_dir: Path | None) -> dict | None:
    """Rough sanity check: the training log's per-second incomes vs the replay pooled over every
    episode's own hold. Eval (deterministic, fixed holds) != training (sigma 2.0, slot pool): an
    order-of-magnitude check of the transcription, not an equality."""
    if log_dir is None or not Path(log_dir).is_dir():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))

    def last_mean(tag, n=200):
        if tag not in tags:
            return None
        v = [e.value for e in ea.Scalars(tag)][-n:]
        return float(np.mean(v)) if v else None

    p: TrainParams = run["params"]
    ep_len_steps = last_mean("Train/mean_episode_length")
    if not ep_len_steps:
        return None
    scale = p.episode_length_s / (ep_len_steps * p.dt)     # per max-episode-second -> per actual second
    eps = [e for e in run["episodes"] if "sums" in e]
    dur = sum(e["duration_s"] for e in eps)

    def pooled(key, w):
        return w * sum(e["sums"][key] for e in eps) / dur

    rows = {
        "track_lin_vel_xy_exp": (pooled("inc_lin", p.track_lin_w), None),
        "track_ang_vel_z_exp": (pooled("inc_ang", p.track_ang_w), None),
        "reward_tracking_yaw": (pooled("inc_facing", p.facing_w), None),
        "reward_yaw_progress": (pooled("inc_yaw_progress", p.yaw_progress_w), None),
        "reward_action_rate": (pooled("action_rate", p.action_rate_w), None),
        "reward_feet_air_time_positive": (pooled("airtime", p.airtime_w), pooled("airtime_exact", p.airtime_w)),
        "penalty_foot_idle_when_forward": (pooled("foot_idle", p.foot_idle_w), None),
    }
    noisy = {}
    for e in eps:
        for sig, d in (e.get("noise") or {}).items():
            noisy[sig] = noisy.get(sig, 0.0) + d["action_rate"]
    out = {"log_dir": str(log_dir), "mean_episode_length_s": ep_len_steps * p.dt, "rows": {}}
    for name, (rep, alt) in rows.items():
        logged = last_mean(f"Episode_Reward/{name}")
        out["rows"][name] = {"log_per_s": (logged * scale) if logged is not None else None,
                             "replay_per_s": rep, "replay_alt_per_s": alt}
    if noisy:
        out["rows"]["reward_action_rate"]["replay_noise_per_s"] = {
            s: p.action_rate_w * v / dur for s, v in noisy.items()}
    std = last_mean("Policy/mean_noise_std")
    out["policy_mean_noise_std"] = std
    return out


# ============================================================================ report
def _f(x, fmt="{:.3f}", na="--"):
    if x is None:
        return na
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        if not math.isfinite(float(x)):
            return na
    except (TypeError, ValueError):
        return str(x)
    return fmt.format(x)


def _pct(x, na="--"):
    return na if x is None else f"{100.0 * x:.1f} %"


def _g(x, na="--"):
    return na if x is None else f"{x:.2g}"


def analyse(run: dict, *, rho_min=RHO_MIN, target=TARGET_SHARE, max_share=GATE2_MAX_SHARE) -> dict:
    p: TrainParams = run["params"]
    eps = run["episodes"]
    g1 = gate1(eps, rho_min=rho_min)
    g2 = gate2(eps, p, target=target, max_share=max_share)
    air = airtime_proposal(g1, g2, p)
    props = {k: g2["terms"][k].get("proposed_weight") for k in ("power", "smooth2", "joint_vel",
                                                               "foot_slip", "stand_pose")}
    mag = {k: abs(v) for k, v in props.items() if v is not None}
    totals = screen_totals(eps, p, {**mag, "rate_sq": abs(p.action_rate_w)})
    alt = m3_alternative(g2, p)
    totals_alt = {}
    if alt:
        totals_alt = screen_totals(eps, p, {
            "smooth2": abs(alt["KRABBY_ACTION_SMOOTH2_W"] or 0.0), "joint_vel": abs(alt["KRABBY_JOINT_VEL_W"] or 0.0),
            "rate_sq": abs(alt["KRABBY_ACTION_RATE_W"]), "foot_slip": mag.get("foot_slip", 0.0)})
    # baseline context per category
    ctx = {}
    for c in CATEGORIES:
        ce = [e for e in eps if e["category"] == c and "sums" in e]
        if not ce:
            continue

        def med(fn, ce=ce):
            vals = [fn(e) for e in ce]
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else None

        def rel_iqr(key, ce=ce):
            v = np.asarray([e["quality"][key] for e in ce if e["quality"].get(key) is not None], dtype=float)
            if v.size < 4 or np.median(v) == 0:
                return None
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            return float((q3 - q1) / q2)

        tot_p = [e["sums"]["power"] for e in ce]
        ctx[c] = {
            "n": len(ce), "n_skipped": sum(1 for e in eps if e["category"] == c and "sums" not in e),
            "income_per_s": med(lambda e: e["income"] / e["duration_s"]),
            "action_jitter": med(lambda e: e["quality"]["action_jitter"]),
            "action_jitter_rel_iqr": rel_iqr("action_jitter"),
            "contact_switch_hz": med(lambda e: e["quality"]["contact_switch_hz"]),
            "contact_switch_hz_rel_iqr": rel_iqr("contact_switch_hz"),
            "swings_norm": med(lambda e: _swings_norm(e, c)),
            "power_share_body_hip": float(np.sum([e["power_split"]["body_hip"] for e in ce]) / max(1e-12, np.sum(tot_p))),
            "power_share_camshaft": float(np.sum([e["power_split"]["camshaft"] for e in ce]) / max(1e-12, np.sum(tot_p))),
            "power_share_rod": float(np.sum([e["power_split"]["rod"] for e in ce]) / max(1e-12, np.sum(tot_p))),
        }
    return {"gate1": g1, "gate2": g2, "airtime": air, "proposals": props, "screen_totals": totals,
            "m3_alternative": alt, "screen_totals_m3_alternative": totals_alt, "context": ctx}


def render(run: dict, res: dict, *, title: str, log_x: dict | None) -> str:
    p: TrainParams = run["params"]
    g1, g2, air = res["gate1"], res["gate2"], res["airtime"]
    L: list[str] = []
    A = L.append
    A(f"# {title}")
    A("")
    A("Offline reward-replay gate, plan step 3 (`/home/nickmagus/.claude/plans/there-is-a-lot-compressed-unicorn.md`).")
    A("Generated by `experiments/tools/replay_motion_terms.py`. CPU only: no Isaac, no training.")
    A("")
    A("## Input")
    A("")
    meta = run["meta"]
    A(f"- eval stamp: `{run['run_dir']}`")
    A(f"- checkpoint: `{meta.get('checkpoint')}` (sha256 `{str(meta.get('checkpoint_sha256'))[:12]}`)")
    n_sc = sum(1 for e in run["episodes"] if "sums" in e)
    A(f"- episodes: {len(run['episodes'])} own holds, {n_sc} scored (own-hold window >= {MIN_WINDOW_STEPS} "
      f"steps; a fall trims the last {run['fall_trim_steps']} steps, as the eval does)")
    A(f"- training params: {p.source}")
    A(f"- eval-side motion metrics recomputed with `gait_eval.metrics.motion_quality_metrics` "
      f"(clip {p.action_clip}); vs the stamp's own `quality` blocks: "
      + (", ".join(f"{k} max |diff| {v:.1e}" for k, v in run["crosscheck"]["max_abs_diff"].items())
         + f" over {run['crosscheck']['n']} episodes" if run["crosscheck"]["n"] else
         "the stamp has no `quality` blocks (pre-step-1 eval_sha)"))
    for note in p.notes:
        A(f"- NOTE: {note}")
    A("")
    A("Training values used: " + ", ".join([
        f"track_lin w {p.track_lin_w} std {p.track_lin_std:g}", f"track_ang w {p.track_ang_w} std {p.track_ang_std:g}",
        f"facing w {p.facing_w}", f"yaw_progress w {p.yaw_progress_w} cap {p.yaw_progress_cap}",
        f"action_rate w {p.action_rate_w} (squared={p.action_rate_squared}, delay {p.action_delay})",
        f"airtime w {p.airtime_w} thr {p.airtime_threshold} s", f"sensor force_threshold {p.sensor_force_threshold} N",
        f"foot_idle w {p.foot_idle_w} (>{p.foot_idle_max_steps} steps, {p.foot_idle_force} N)",
        f"power w {p.power_w}", f"clip_actions {p.action_clip}", f"dt {p.dt:g}"]) + ".")
    A("")

    # ---------------------------------------------------------------- headline
    A("## Headline")
    A("")
    passing = [k for k in ("power", "smooth2", "joint_vel", "foot_slip", "stand_pose")
               if g1[k]["verdict"] == "PASS"]
    failing = {k: g1[k]["fails"] for k in ("power", "smooth2", "joint_vel", "foot_slip", "stand_pose", "action_rate_sq")
               if g1[k]["verdict"] == "FAIL"}
    A(f"- **Gate 1**: {len(passing)} of the 5 new penalty weights pass in every active category"
      + (f" ({', '.join(passing)})" if passing else "") + ". "
      + ("FAIL: " + "; ".join(f"`{k}` in {len(v)}/{len(g1[k]['active'])} ({', '.join(v)})" for k, v in failing.items())
         + "." if failing else ""))
    sq = g2["terms"]["action_rate_sq"]["rows"]
    pl = g2["terms"]["action_rate"]["rows"]
    if sq:
        A(f"- **Gate 2**: every proposed weight passes by construction (10 % on its binding category). "
          f"`KRABBY_ACTION_RATE_SQ=1` at the stock -0.3 costs {_pct(min(r['share_at_weight'] for r in sq.values()))}"
          f"-{_pct(max(r['share_at_weight'] for r in sq.values()))} of the task income (Gate 2 "
          f"{g2['terms']['action_rate_sq'].get('verdict')}); the stock unsquared -0.3 already costs "
          f"{_pct(min(r['share_at_weight'] for r in pl.values()))}-{_pct(max(r['share_at_weight'] for r in pl.values()))}.")
    if air["inert_in_training"]:
        A("- **`KRABBY_AIRTIME_W` is a no-op**: `reward_feet_air_time_positive` pays ~0 in training "
          "(float32 contact-sensor clock; section below). Any de-weight leaves training unchanged.")
    A("")

    # ---------------------------------------------------------------- proposals
    A("## Proposed weights (the six `TBD_replay_gate` placeholders)")
    A("")
    A("| screen | knob | proposed | binding category | Gate 1 | Gate 2 at the proposal | note |")
    A("|---|---|---|---|---|---|---|")
    for key in ("power", "smooth2", "joint_vel", "foot_slip", "airtime", "stand_pose"):
        t = TERM[key]
        g1v = g1[key]["verdict"]
        if g1v == "FAIL":
            g1v = f"FAIL ({', '.join(g1[key]['fails'])})"
        if key == "airtime":
            val = f"{air['proposed_weight']:g}"
            bind = "--"
            g2v = "n/a (income)"
            note = ("INERT IN TRAINING: the float32 sensor clock stops first contacts registering after "
                    "~2 s of episode time, so the term pays ~0 (see below); the de-weight changes nothing"
                    if air["inert_in_training"] else "halves the air-time income")
        else:
            s2 = g2["terms"][key]
            val = _g(s2.get("proposed_weight"))
            bind = s2.get("binding_category") or "--"
            g2v = s2.get("verdict", "--")
            note = ""
            if key == "smooth2":
                nf = g1[key]["noncircular_fails"]
                note = ("jitter rho is near-tautological; judged on contact_switch_hz it "
                        + ("passes everywhere" if not nf else f"FAILS in {', '.join(nf)}"))
            if key == "power":
                c = res["context"]
                bh = [c[x]["power_share_body_hip"] for x in c]
                note = (f"{100 * min(bh):.0f}-{100 * max(bh):.0f} % of sum |tau qdot| is the Body_Hip linkage "
                        "emulator (a sim artefact) -- see the power split")
            anti = [f"{m} in {c} (rho {r[k]:+.2f})"
                    for c, r in g1[key]["rows"].items() if r.get("active")
                    for k, m in (("rho_jitter", "action_jitter"), ("rho_switch", "contact_switch_hz"))
                    if r.get(k) is not None and r[k] <= -RHO_MIN]
            if anti:
                note = (note + "; " if note else "") + "ANTI-tracks " + ", ".join(anti)
        A(f"| {t.screen} | `{t.knob}` | **{val}** | {bind} | {g1v} | {g2v} | {note} |")
    A("")
    A("M3 also sets `KRABBY_ACTION_RATE_SQ=1` at the stock `KRABBY_ACTION_RATE_W` -0.3 (no weight to "
      f"set): Gate 1 {g1['action_rate_sq']['verdict']}, Gate 2 {g2['terms']['action_rate_sq'].get('verdict')} "
      "(rows below).")
    A("")
    A("**What a FAIL means.** A Gate-1 FAIL (the term does not rank episodes like the defect in some "
      "category where it is active) or a Gate-2 FAIL (it costs >= 15 % of the task income on the "
      "best-tracking quartile) drops the term from its screen unless the user overrules. Nothing is "
      "passed silently: every per-category number is below.")
    A("")

    # ---------------------------------------------------------------- gate 1
    A("## Gate 1 -- does the term track the defect?")
    A("")
    A("Spearman rho over the category's episodes of the term's own-hold cost, normalised as block Q "
      "normalises (per m on fwd/side/back/diag, per rad on turn/move_turn, per s on stand), against "
      "the step-1 motion metrics of the same window. PASS = rho >= 0.5 vs `action_jitter` OR "
      "`contact_switch_hz` (for the air-time INCOME: vs `contact_switch_hz` OR swings per m/rad -- "
      "it must pay the defect for a de-weight to help). `swings` = `swings_per_m` / `swings_per_rad` "
      "(N/A at stand). The two `per s` columns are CONTEXT, not the gate: the same cost per second "
      "against the same two metrics (both are per-step / per-second rates, so this is like for like; "
      "the gate's per-m / per-rad cost shares no denominator with them, while `rho swings` shares its "
      "1/distance or 1/yaw with the cost -- a high `rho swings` is partly that shared denominator).")
    A("")
    A("| term | category | n | rho jitter | rho switch_hz | rho swings | verdict | rho jitter (per s) | rho switch_hz (per s) |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t in TERMS:
        for c in CATEGORIES:
            r = g1[t.key]["rows"][c]
            if not r.get("active"):
                if c in t.categories and r.get("why") != "not applicable":
                    A(f"| {t.key} | {c} | {r.get('n', '--')} | -- | -- | -- | inactive ({r['why']}) | -- | -- |")
                continue
            v = "PASS" if r["pass"] else "**FAIL**"
            if t.tautological:
                v += " (non-circular: " + ("PASS" if r["pass_noncircular"] else "FAIL") + ")"
            A(f"| {t.key} | {c} | {r['n']} | {_f(r['rho_jitter'], '{:+.2f}')} | {_f(r['rho_switch'], '{:+.2f}')} "
              f"| {_f(r['rho_swings'], '{:+.2f}')} | {v} | {_f(r.get('rho_jitter_per_s'), '{:+.2f}')} "
              f"| {_f(r.get('rho_switch_per_s'), '{:+.2f}')} |")
    A("")
    A("Overall: " + "; ".join(f"`{t.key}` {g1[t.key]['verdict']}"
                              + (f" (fails: {', '.join(g1[t.key]['fails'])})" if g1[t.key]["fails"] else "")
                              for t in TERMS) + ".")
    A("")
    reads = []
    for t in TERMS:
        lim = []
        for c in g1[t.key]["fails"]:
            r = g1[t.key]["rows"][c]
            per_s = [r.get("rho_switch_per_s")] if t.tautological else [r.get("rho_jitter_per_s"), r.get("rho_switch_per_s")]
            per_s = [v for v in per_s if v is not None]
            if per_s and max(per_s) >= RHO_MIN:
                lim.append(c)
        if lim:
            what = "`contact_switch_hz`" if t.tautological else "`action_jitter` or `contact_switch_hz`"
            reads.append(f"`{t.key}` fails the Q-normalised gate but its cost PER SECOND tracks {what} "
                         f"(rho >= 0.5) in {', '.join(lim)}")
    if reads:
        A("Reading the FAILs: " + "; ".join(reads) + ". There the per-m / per-rad normalisation "
          "(the spread of distance or yaw between episodes) decides the ranking, not the defect. The "
          "verdict stays FAIL -- the plan's rule is the Q normalisation -- but the user can weigh it.")
        A("")
    A("Circularity: `smooth2` (sum of squared second differences of the clipped actions) vs "
      "`action_jitter` (their mean absolute value) is near-tautological, and both action-rate forms "
      "share the action sequence with it; their verdict is also given against `contact_switch_hz` "
      "alone (\"non-circular\"). `foot_slip` and `contact_switch_hz` share the 50-N load mask but "
      "measure different things (speed of planted feet vs load transitions).")
    A("")

    # ---------------------------------------------------------------- gate 2
    A("## Gate 2 -- does the term swamp the task?")
    A("")
    A("Task-tracking income = `track_lin_vel_xy_exp` + `track_ang_vel_z_exp` + `reward_tracking_yaw` "
      "(facing) + `reward_yaw_progress`, weighted as in training. Best-tracking quartile = the top 25 % "
      "of a category's episodes by that income per second. `ratio` = median over the quartile of "
      "(raw term integral / income integral); the 10 % weight is 0.10 / ratio; `share` = |weight| x "
      "ratio at the proposed (or fixed) weight; PASS = share < 15 %. `kernels` = the same share "
      "against the two tracking kernels alone (a stricter denominator).")
    A("")
    A("| category | quartile n / all | income /s (quartile) | income /s (all) |")
    A("|---|---|---|---|")
    for c in CATEGORIES:
        q = g2["quartile"].get(c)
        if q:
            A(f"| {c} | {q['n']} / {q['n_all']} | {q['income_per_s_q']:.3f} | {q['income_per_s_all']:.3f} |")
    A("")
    A("| term | weight | category | ratio (quartile) | ratio (all) | 10 % weight | share | share (kernels) | verdict |")
    A("|---|---|---|---|---|---|---|---|---|")
    for t in TERMS:
        s2 = g2["terms"][t.key]
        w = s2.get("weight_evaluated")
        for c, r in s2["rows"].items():
            v = "--" if r.get("gate2_pass") is None else ("PASS" if r["gate2_pass"] else "**FAIL**")
            if c == s2.get("binding_category"):
                v += " (binding)"
            A(f"| {t.key} | {_g(w)} | {c} | {_g(r['median_ratio_q'])} | {_g(r['median_ratio_all'])} | "
              f"{_g(-r['w10'] if (r.get('w10') and t.kind == 'penalty') else r.get('w10'))} | {_pct(r.get('share_at_weight'))} | "
              f"{_pct(r.get('share_at_weight_kernels'))} | {v} |")
    A("")

    # ---------------------------------------------------------------- noise floor
    sigs = sorted({s for t in ACTION_TERMS for r in g2["terms"][t]["rows"].values()
                   for s in (r.get("median_ratio_q_noise") or {})}, key=_sig_sort_key)
    if sigs:
        A("### Exploration-noise floor of the action terms (training samples, the eval does not)")
        A("")
        A("The same share with the recorded mean actions replaced by `clip(mu + sigma * eps, +-1)` "
          f"({', '.join('sigma ' + s for s in sigs)}; Monte Carlo). X08:c6 trains at sigma 2.0; M1 pins it "
          "at 0.5. The noise part is an offset the policy mean can barely move (a constant per-step "
          "penalty mostly rewards ending the episode sooner), so a share that is fine on the mean can "
          "be large in training.")
        A("")
        A("| term | weight | category | share (mean) | " + " | ".join(f"share (sigma {s})" for s in sigs) + " |")
        A("|---|---|---|---|" + "---|" * len(sigs))
        for key in ACTION_TERMS:
            s2 = g2["terms"][key]
            for c, r in s2["rows"].items():
                sn = r.get("share_at_weight_noise") or {}
                A(f"| {key} | {_g(s2.get('weight_evaluated'))} | {c} | {_pct(r.get('share_at_weight'))} | "
                  + " | ".join(_pct(sn.get(s)) for s in sigs) + " |")
        A("")

    if res.get("noise_tax"):
        L.extend(render_noise_tax(res["noise_tax"], mu_bound=run.get("mu_bound"),
                                  sigma_sources=run.get("sigma_sources")))

    # ---------------------------------------------------------------- screens
    A("## Screen totals (median over the best-tracking quartile, share of income)")
    A("")
    A("Gate 2 is per term, but a screen arms its terms together. M3 swaps the stock unsquared action "
      "rate for the squared one AND adds two terms, so its NET added cost is smooth2 + joint_vel + "
      "|w_rate| (sq - plain).")
    A("")
    alt = res.get("m3_alternative") or {}
    ta = res.get("screen_totals_m3_alternative") or {}
    head = "| category | M3 net added (proposals, rate -0.3 squared) | M3 all terms | M4 slip |"
    sep = "|---|---|---|---|"
    if alt:
        head += " M3 net added (alternative) |"
        sep += "---|"
    A(head)
    A(sep)
    for c, r in res["screen_totals"].items():
        line = f"| {c} | {_pct(r['M3_net_added'])} | {_pct(r['M3_all_terms'])} | {_pct(r['M4_slip'])} |"
        if alt:
            line += f" {_pct((ta.get(c) or {}).get('M3_net_added'))} |"
        A(line)
    A("")
    if alt:
        A(f"**M3 alternative (screen-level ~10 %).** `KRABBY_ACTION_RATE_SQ=1` with `KRABBY_ACTION_RATE_W` "
          f"= **{alt['KRABBY_ACTION_RATE_W']:g}** (the squared term then costs what the stock unsquared "
          f"-0.3 costs; binding {alt['rate_binding_category']}), `KRABBY_ACTION_SMOOTH2_W` = "
          f"**{_g(alt['KRABBY_ACTION_SMOOTH2_W'])}** and `KRABBY_JOINT_VEL_W` = **{_g(alt['KRABBY_JOINT_VEL_W'])}** "
          "(half the 10 % proposals). This is an option for the user, not the plan's rule: the plan "
          "prices each term at 10 %, and with `ACTION_RATE_SQ` at the stock weight M3 as specified "
          "adds the net share in the first column. `KRABBY_ACTION_RATE_W` is not in M3's overlay "
          "today, so taking this option means adding it there.")
        A("")

    # ---------------------------------------------------------------- air time
    A("## `KRABBY_AIRTIME_W` (de-weight of `reward_feet_air_time_positive`, +0.8)")
    A("")
    A("| category | income share at 0.8, as trained (float32 sensor clock) | income share at 0.8, exact timing (counterfactual) |")
    A("|---|---|---|")
    for c in air["share_exact_timing"]:
        A(f"| {c} | {_pct(air['share_now_training_faithful'].get(c, 0.0))} | {_pct(air['share_exact_timing'].get(c))} |")
    A("")
    ax = g1["airtime_exact"]["rows"]
    anti = {c: ax[c].get("rho_switch_per_s") for c in ax if ax[c].get("active")}
    if anti:
        A("Even with exact timing the income does not pay the defect: its per-second rho against "
          "`contact_switch_hz` is " + ", ".join(f"{c} {_f(v, '{:+.2f}')}" for c, v in anti.items())
          + " -- it pays LONG swings (> 0.38 s), which a chattering foot does not make. Its per-m / "
          "per-rad rho against swings per m/rad (Gate 1 table) is mostly the shared 1/distance "
          "denominator.")
        A("")
    A(f"Proposal: {air['current_weight']:g} -> **{air['proposed_weight']:g}** (halves the air-time income). "
      + ("**But the term is inert in training.** `ContactSensor.compute_first_contact(dt)` tests "
         "`current_contact_time < dt + 1e-8`, and the sensor's float32 `_timestamp` (reset each episode, "
         "+= 0.005 per substep) makes the per-refresh elapsed time 0.0200001-0.0200005 s once the episode "
         "is ~2 s old, so a touchdown after that never registers as a first contact and the term pays 0. "
         "The replay reproduces this bit-for-bit (`sensor_elapsed_f32`), and the X08:c6 training log agrees "
         "(cross-check below). A de-weight therefore changes nothing: M4 is effectively `KRABBY_FOOT_SLIP_W` "
         "alone. The exact-timing column is what the term WOULD pay if the timing were fixed -- a separate "
         "decision (fixing it would switch ON a +0.8 swing income the current gait never trained with)."
         if air["inert_in_training"] else "The training-faithful replay shows the term is live.")
      )
    A("")

    # ---------------------------------------------------------------- context
    A("## Per-category notes (baseline, all scored episodes; medians)")
    A("")
    A("`(rel IQR)` = interquartile range / median across the category's episodes: how much the "
      "defect metric varies between episodes of ONE deterministic policy on ONE command -- the "
      "variation Gate 1 has to rank. A narrow spread makes every rank correlation noisy.")
    A("")
    A("| category | n (skipped) | income /s | action_jitter (rel IQR) | contact_switch_hz (rel IQR) | swings per m/rad | power: rod / camshaft / Body_Hip |")
    A("|---|---|---|---|---|---|---|")
    for c, r in res["context"].items():
        A(f"| {c} | {r['n']} ({r['n_skipped']}) | {_f(r['income_per_s'])} | {_f(r['action_jitter'])} "
          f"({_f(r.get('action_jitter_rel_iqr'), '{:.2f}')}) | {_f(r['contact_switch_hz'])} "
          f"({_f(r.get('contact_switch_hz_rel_iqr'), '{:.2f}')}) | {_f(r['swings_norm'], '{:.2f}')} | "
          f"{100 * r['power_share_rod']:.0f} / {100 * r['power_share_camshaft']:.0f} / {100 * r['power_share_body_hip']:.0f} % |")
    A("")
    A("The camshaft joint itself carries ~0 of sum |tau qdot|: the cam's mechanical load sits on the "
      "`*_Body_Hip_` joint the linkage emulator drives (the CHARTER's note: its tau qdot is the "
      "emulator's torque, a sim artefact). `KRABBY_POWER_W` prices all 24 joints, so that share of "
      "its cost is emulator power.")
    A("")

    if log_x:
        A("## Transcription cross-check against the X08:c6 training log")
        A("")
        A(f"Training `Episode_Reward/*` (last 200 iterations; per max-episode-second, rescaled to per "
          f"actual second with the logged mean episode length {log_x['mean_episode_length_s']:.1f} s; "
          f"policy std {_f(log_x.get('policy_mean_noise_std'), '{:.2f}')}) vs the replay pooled over every "
          "eval own-hold second. Different command mix and deterministic vs sampled actions, so only the "
          "order of magnitude is meaningful.")
        A("")
        A("| term | training log /s | replay /s | note |")
        A("|---|---|---|---|")
        for name, r in log_x["rows"].items():
            note = ""
            if r.get("replay_alt_per_s") is not None:
                note = f"exact-timing replay {r['replay_alt_per_s']:+.4f}"
            if r.get("replay_noise_per_s"):
                note = "; ".join(f"sigma {s}: {v:+.3f}" for s, v in sorted(r["replay_noise_per_s"].items(), key=lambda kv: _sig_sort_key(kv[0])))
            A(f"| {name} | {_f(r['log_per_s'], '{:+.4f}')} | {_f(r['replay_per_s'], '{:+.4f}')} | {note} |")
        A("")

    # ---------------------------------------------------------------- method
    A("## How each term was replayed")
    A("")
    A("Row `t` of the eval npz is recorded BEFORE `env.step(actions[t])`, so env step `s` reads actions "
      "and command from row `s` and the post-physics state from row `s + 1` (the state the reward "
      "manager reads). Integrals are `sum value * dt` over the own hold's whole window "
      "(`report._window_mask`, steady = all). Actions are the stored pre-clip policy mean clipped to "
      f"+-{p.action_clip} (`clip_actions`), as the runner's wrapper does.")
    A("")
    A("| term (knob) | training function | replayed as |")
    A("|---|---|---|")
    A("| power (`KRABBY_POWER_W`) | `penalty_mechanical_power` | sum over all 24 joints of abs(tau x qdot) |")
    A("| smooth2 (`KRABBY_ACTION_SMOOTH2_W`) | `penalty_action_second_diff` | squared norm of (a_s - 2 a_(s-1) + a_(s-2)), undelayed clipped history, 0 for s < 2 |")
    A(f"| joint_vel (`KRABBY_JOINT_VEL_W`) | `penalty_joint_vel_l2` | sum of qdot^2 over the 12 rod joints |")
    A(f"| action_rate(_sq) (`KRABBY_ACTION_RATE_W` / `_SQ`) | `reward_action_rate` | norm (squared) of r_s - r_(s-1), r_s = a_(s-{p.action_delay}) (the delayed `raw_actions`), zero at reset |")
    A(f"| foot_slip (`KRABBY_FOOT_SLIP_W`) | `penalty_loaded_foot_slip` | sum of planar foot speed over feet with force > {p.foot_slip_force:g} N |")
    A("| stand_pose (`KRABBY_STAND_POSE_W`) | `penalty_stand_pose_l1` | sum abs(q - q_default) over the rod joints, stand hold only (the eval analogue of `stand_mask`) |")
    A(f"| airtime (`KRABBY_AIRTIME_W`) | `reward_feet_air_time_positive` | contact-sensor air/contact clocks re-run per refresh (contact > {p.sensor_force_threshold:g} N), float32 timing as in training; `airtime_exact` = exact timing |")
    A(f"| foot_idle (stock) | `PenaltyFootIdleWhenForward` | idle-step counter from reset (contact > {p.foot_idle_force:g} N), relu(idle - {p.foot_idle_max_steps}), omni `_cmd_gate` |")
    A("| task income | stock kernels + omni facing / yaw progress | as trained (std / weights above); facing reads `heading_err` of row s + 1 |")
    A("")
    A("Mismatches that cannot be removed offline: the eval is the deterministic mean (training samples "
      "at sigma 2.0 -- the state-driven terms cannot be re-simulated under noise; the action terms are "
      "Monte-Carlo'd above); the eval's fixed 12-s holds are not the training slot pool; the stand gate "
      "is the eval's stand hold; the eval harness's `KRABBY_TRACK_SIGMA2 0.1` env var is an eval-env "
      "setting and is ignored -- the replay prices the TRAINING kernels.")
    A("")
    return "\n".join(L) + "\n"


def to_json(run: dict, res: dict, log_x: dict | None, *, per_episode: bool) -> dict:
    p: TrainParams = run["params"]
    out = {
        "tool": "crab_hex_omni_task/experiments/tools/replay_motion_terms.py",
        "run_dir": run["run_dir"], "meta": run["meta"], "params": asdict(p),
        "crosscheck_quality_blocks": run["crosscheck"], "fall_trim_steps": run["fall_trim_steps"],
        "thresholds": {"rho_min": RHO_MIN, "target_share": TARGET_SHARE, "gate2_max_share": GATE2_MAX_SHARE,
                       "quartile": QUARTILE, "min_window_steps": MIN_WINDOW_STEPS},
        "proposed_weights": {
            "KRABBY_POWER_W": res["gate2"]["terms"]["power"].get("proposed_weight"),
            "KRABBY_ACTION_SMOOTH2_W": res["gate2"]["terms"]["smooth2"].get("proposed_weight"),
            "KRABBY_JOINT_VEL_W": res["gate2"]["terms"]["joint_vel"].get("proposed_weight"),
            "KRABBY_FOOT_SLIP_W": res["gate2"]["terms"]["foot_slip"].get("proposed_weight"),
            "KRABBY_AIRTIME_W": res["airtime"]["proposed_weight"],
            "KRABBY_STAND_POSE_W": res["gate2"]["terms"]["stand_pose"].get("proposed_weight"),
        },
        "gate1": res["gate1"], "gate2": res["gate2"], "airtime": res["airtime"],
        "screen_totals": res["screen_totals"], "m3_alternative": res["m3_alternative"],
        "screen_totals_m3_alternative": res["screen_totals_m3_alternative"],
        "context": res["context"], "train_log_crosscheck": log_x,
    }
    if res.get("noise_tax"):
        out["noise_tax"] = res["noise_tax"]
        out["mu_bound"] = run.get("mu_bound")
        out["sigma_sources"] = run.get("sigma_sources")
    if per_episode:
        out["episodes"] = run["episodes"]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path, help="suite__omni stamp dir (has complete.json)")
    ap.add_argument("--out", required=True, type=Path, help="output dir for REPLAY_GATE.md + replay_gate.json")
    ap.add_argument("--train-env-yaml", type=Path, default=None,
                    help="training params/env.yaml (default: next to the eval's checkpoint)")
    ap.add_argument("--train-agent-yaml", type=Path, default=None)
    ap.add_argument("--train-log-dir", type=Path, default=None,
                    help="training tfevents dir for the transcription cross-check (default: the checkpoint's dir)")
    ap.add_argument("--noise-sigmas", default="0.5,2.0",
                    help="comma list of sigmas; 'grid' = %s; 'ckpt' = the checkpoint's own per-dim sigma; "
                         "'' disables the noise Monte Carlo" % ",".join(f"{x:g}" for x in NOISE_GRID))
    ap.add_argument("--noise-samples", type=int, default=4)
    ap.add_argument("--mu-bound", type=float, default=None,
                    help="also price every noise sample around clip(mu, +-B) (a bound loss's mean); implies --noise-tax")
    ap.add_argument("--noise-tax", action="store_true",
                    help="add the action-rate noise-tax table (saturated vs bounded mean, per weight)")
    ap.add_argument("--action-rate-weights", default=",".join(f"{w:g}" for w in ACTION_RATE_WEIGHTS),
                    help="--noise-tax: reward_action_rate weights priced side by side")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="--noise-sigmas ckpt: read sigma from this checkpoint (default: run_meta's)")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (development)")
    ap.add_argument("--per-episode", action="store_true", help="also write replay_episodes.json")
    ap.add_argument("--title", default="Replay gate -- gait motion quality (plan step 3)")
    ap.add_argument("--md-name", default="REPLAY_GATE.md")
    ap.add_argument("--json-name", default="replay_gate.json")
    args = ap.parse_args(argv)

    sigmas = parse_noise_sigmas(args.noise_sigmas)
    run = load_run(args.run_dir, p_override=(args.train_env_yaml, args.train_agent_yaml),
                   noise_sigmas=sigmas, noise_k=args.noise_samples, seed=args.seed, limit=args.limit,
                   mu_bound=args.mu_bound, checkpoint=args.checkpoint)
    res = analyse(run)
    if args.noise_tax or args.mu_bound is not None:
        weights = [float(w) for w in args.action_rate_weights.split(",") if w.strip()]
        res["noise_tax"] = noise_tax(run["episodes"], weights=weights)
    log_dir = args.train_log_dir
    if log_dir is None and run["meta"].get("checkpoint"):
        log_dir = Path(run["meta"]["checkpoint"]).parent
    log_x = train_log_crosscheck(run, log_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / args.md_name).write_text(render(run, res, title=args.title, log_x=log_x))
    payload = to_json(run, res, log_x, per_episode=False)
    (args.out / args.json_name).write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    if args.per_episode:
        (args.out / "replay_episodes.json").write_text(json.dumps(_jsonable(run["episodes"])))
    print(f"[replay] wrote {args.out / args.md_name} and {args.out / args.json_name}")
    for k, v in payload["proposed_weights"].items():
        print(f"[replay] {k} = {v}")
    return 0


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    sys.exit(main())
