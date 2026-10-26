#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Offline clock-economy replay -- omni clock-deprecation campaign, plan step 5 (2026-09-23). CPU
ONLY. Never touches Isaac, a GPU, ``candidates.yaml`` or git.

THE STANDING GATE. "Replay new reward terms on saved healthy + degenerate raws before any
training screen" (repo rule). Step 9's step-0 diagnostics run this tool's RG1/RG2/RG3 on the
seven ``suite__omni`` stamps E1-E7 before wave 1 launches.

THE QUESTION. Settled decision 1 (``lets-explore-options-for-groovy-spindle.md``) removes BOTH
clock rewards (``reward_clock_schedule``, ``reward_clock_swing_apex``) and the clock observation.
Before any GPU screen, this tool prices every replayed reward term under two weight settings of
just those two rewards -- CLOCK-HELD (1.0, 1.0, the shipped X08:c6 value) and CLOCK-FREE (0.001,
0.001, standing rule 13's floor weight, never 0.0) -- on the SAME saved rollouts, so healthy and
degenerate heads are compared under one fixed economy rather than each rollout's own training
recipe. Every other weight comes from X08:c6's own dump
(``parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-22_11-58-07_seed3_X08_c6_A15pB/params/
env.yaml``) regardless of which head produced the rollout -- E7's own env.yaml has
``penalty_tracking_error_l1`` at -1.0, not -2.0, and E6 (the ``--zero-actions`` statue) has no
head or env.yaml at all, so per-rollout weights would not be comparable.

REUSE (plan step 5's spec). This module is loaded by path, pure numpy/pyyaml at module scope:

* ``OMNI/experiments/2026-09-16_1805_omni_formation/replay_reward_economy.py`` (the "OF" tool,
  the omni-formation 1c-fix reward-economy replay) -- its TRANSCRIBED shipped-formula primitives:
  ``_gravity_b``, ``_clock_income``, ``_apex_income`` (called directly, byte-for-byte the same
  code) and its ``episode_primitives`` (called directly for every OTHER per-step quantity --
  tracking errors, progress, yaw progress, facing, cross-command penalty, torque stream,
  foot-idle, air time, excess-feet-contact -- none of which depend on the clock phase; the ONE
  exception is ``arate``, re-derived here with the runner's clip -- see THE ACTION CLIP). OF's own
  module-level shipped constants (``FORCE_REF_N=100``, ``VEL_REF_M_S=0.1``, ``TOUCH_REF_N=50``,
  ``APEX_MID_THRESH=0.7``, ``APEX_TARGET_M=0.08``, ``APEX_SIGMA_M=0.03``,
  ``GROUND_OFFSET_FROM_ROOT_M=-1.05``, ``MIN_UPRIGHT_GZ=0.9``, ``OMNI_CLOCK_TURN_R=1.2``,
  ``MIN_CMD_NORM=0.12``, ``PROGRESS_CAP=1.05``, ``AIR_TIME_THRESHOLD=0.38``,
  ``AIR_TIME_CMD_GATE=0.1``, ``MAX_IDLE_STEPS=90``, ``MAX_FEET_ON_GROUND=4``,
  ``CONTACT_FORCE_THRESHOLD=0.1``) are pinned equal to X08:c6's own dumped params (checked by
  hand against ``params/env.yaml`` when this file was written; the module's own tests re-check the
  ones that matter to RG1-RG3).
* ``OMNI/experiments/tools/replay_motion_terms.py`` (the "RMT" tool, the gait-motion-quality
  replay gate) -- ``_yaml_load_tolerant`` (the same untrusted-tag-safe env.yaml/agent.yaml loader)
  and ``CATEGORIES`` (the seven ``suite__omni`` category names). This module's
  :func:`load_suite_rollout` follows RMT's ``load_run``'s OWN directory contract
  (``run_meta.json`` + ``complete.json`` + ``raw/episode_*.npz``) and, like ``load_run``, resolves
  each episode's category/hold from the run's OWN ``env_index_map``/``hold_labels`` rather than a
  hardcoded label tuple -- the OF tool's ``episode_primitives`` documents exactly why a hardcoded
  tuple is unsafe (a different hold SCHEDULE reorders the segment ids and would silently
  re-label), which is why this loader is RMT's pattern, not OF's ``load_head`` glob.

THE CLOCK PHASE (plan step 5: "use the recorded clock_phase when present ... and say clearly when
it is absent"). ``eval_crab_hex_gait.py`` now records the action term's own ``clock_phase`` (plan
step 3) in every world-frame omni raw. When an episode's npz carries it, :func:`episode_primitives`
uses it DIRECTLY (one candidate, no search) -- the exact phase the trained policy actually saw,
supplanting the OF tool's one real reconstruction gap. When it is absent (every raw recorded
before step 3 landed, including the OF tool's own canary raws below), this module falls back to
OF's own offset-fit (``OF.episode_primitives``'s grid search over ``n_offsets`` candidate phase
offsets, scored by argmax clock-held income) -- the SAME code, called directly, not
reimplemented -- and every output the fallback produces is tagged ``phase_source: "fitted ..."``
so a reader always knows which one was used. A report that mixes recorded and fitted episodes
prints the count of each.

THE ACTION CLIP (fixed 2026-09-25). The eval npz stores the policy's PRE-clip mean. Before the
action term sees it, the runner's ``ParkourRslRlVecEnvWrapper`` clamps every column to
``+-clip_actions`` (the head's ``params/agent.yaml``; ``train.py`` / ``evaluation.py`` pass
``agent_cfg.clip_actions``, ``vecenv_wrapper.py`` ``step`` clamps before ``env.step``); the action
term then applies its OWN per-joint clip (``_clip_raw_actions``: CamShaft +-1, Hip_Femur and
Femur_Tibia +-4.8 -- OF's ``ACTION_CLIP``) and ``reward_action_rate`` reads the result
(``raw_actions``). OF's ``episode_primitives`` applies only the second clamp, so on the omni heads
(``clip_actions: 1.0``, bang-bang saturated -- median |mu| 5-8) it overstated ``action_rate``
2.9-3.4x on the step-0 walkers (1.4-2.1x on the creeper). :func:`episode_primitives` therefore
re-derives ``arate`` from both clamps in order (:func:`runner_clipped_actions`), and
:func:`load_suite_rollout` reads the first from each stamp's OWN head (``run_meta.checkpoint``'s
``params/agent.yaml``, RMT's ``_train_dumps_for`` convention) unless ``--action-clip`` passes it
explicitly. A missing agent.yaml is refused, never defaulted.
Step 0's ``STEP0_REPLAY.json`` predates this fix; its corrected numbers are the 2026-09-25
addendum in ``STEP0_READOUT.md``.

EMITTED QUANTITIES

* :func:`term_values` -- every replayed term's PRE-CLIP per-step ``weight * value`` series
  (length T, unmasked). "Pre-clip" names the REWARD total (no reward-side clip is reproduced), not
  the action: ``arate`` is computed on the runner-clipped actions (THE ACTION CLIP above).
* :func:`pre_clip_total` -- their per-step sum: one number per env-step, the plan's "per-step
  pre-clip totals".
* :func:`term_incomes` -- the mean of :func:`term_values` over a boolean mask (a hold), the same
  shape and semantics as OF's own ``term_incomes`` -- the function the reproduction test below
  checks this module against.
* :func:`head_hold_income` -- OF's own step-weighted multi-episode mean, generalised from a hold
  NAME (OF's own fixed schedule) to a hold INDEX (this tool's suite loader resolves the index from
  the run's own ``hold_labels``, never a hardcoded tuple).

NOT REPLAYED. :data:`NOT_REPLAYED_NOTES` is generated at load time from whichever env.yaml the
weights come from: any reward term with a nonzero weight whose name this module does not know how
to price is listed by name and weight (this is how ``reward_stride_length`` and
``penalty_tibia_deviation_in_stance`` -- the OF tool's own two gaps -- surface for X08:c6; nothing
here is a hardcoded, possibly stale, list).

THE GATES (plan step 9's own text; RG3 is report-only)

* **RG1.** For every (healthy, degenerate) pair on every moving category whose CLOCK-HELD
  ordering holds (healthy's per-second NET income > degenerate's), that ordering must survive
  under CLOCK-FREE, and the CLOCK-FREE ratio (healthy / degenerate) must be
  >= ``min(1.10, the same pair's CLOCK-HELD ratio)``. A triple already reversed under clock-held is
  reported separately (``held_reversed``: "no clock-held ordering: not an RG1 criterion"). Per
  audit item 11, a pair whose degenerate total is <= 0 has no defined ratio: the guard REPORTS
  this (``ratio: None``, a note) instead of dividing, and the pair is judged on its clock-free
  ordering alone. Every hold of a category (both hands) is pooled.
* **RG2.** Pooled over every healthy head's moving-hold steps, the fraction of
  :func:`pre_clip_total` <= 0 must rise by at most 10 percentage points from clock-held to
  clock-free, and must stay <= 0.25 under clock-free.
* **RG3 (report only).** Per category (all seven, INCLUDING stand -- the frozen clock pays
  six-leg stance on ~85% of stand steps per the plan's own fact 5), the share of clock-held
  positive income that clock removal takes away: ``(pos_held - pos_free) / pos_held`` (guarded the
  same way as RG1 when ``pos_held <= 0``). Pre-registered: a stand share >= 30% flags the
  ``stand.steps_per_s`` guard as the likely BREAK route (plan step 9).

USAGE
    <isaac_venv>/bin/python replay_clock_economy.py --healthy E1=<run_dir> --healthy E3=<run_dir> \\
        --degenerate E6=<run_dir> --degenerate E7=<run_dir> [--weights-env-yaml P] [--out DIR]
        [--n-offsets 36] [--limit N] [--action-clip C]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
CAMPAIGN_DIR = HERE.parent
OMNI = CAMPAIGN_DIR.parent.parent                                    # .../crab_hex_omni_task
OF_TOOL_PY = OMNI / "experiments" / "2026-09-16_1805_omni_formation" / "replay_reward_economy.py"
RMT_PY = OMNI / "experiments" / "tools" / "replay_motion_terms.py"

#: X08:c6's own dump -- the "one economy" every rollout is priced under (module docstring).
DEFAULT_WEIGHTS_ENV_YAML = (
    str(HERE.parents[6]) + "/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
    "2026-09-22_11-58-07_seed3_X08_c6_A15pB/params/env.yaml"
)

CLOCK_HELD = (1.0, 1.0)
CLOCK_FREE = (0.001, 0.001)          # standing rule 13: epsilon, never 0.0 (0.0 kills telemetry)
MOVING_CATEGORIES = ("fwd", "side", "back", "diag", "turn", "move_turn")
ALL_CATEGORIES = ("stand",) + MOVING_CATEGORIES
RG1_RATIO_TARGET = 1.10
RG2_MAX_RISE = 0.10
RG2_MAX_SHARE = 0.25
RG3_STAND_FLAG = 0.30


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OF = _load(OF_TOOL_PY, "omni_formation_replay_reward_economy")
RMT = _load(RMT_PY, "omni_clock_replay_motion_terms")


# ============================================================================ the economy
@dataclass(frozen=True)
class Economy:
    """Every weight/param :func:`term_values` needs. Defaults reproduce X08:c6's own
    ``params/env.yaml`` (see :func:`economy_from_env_yaml`); the two-element ``clock`` tuple is
    ``(reward_clock_schedule weight, reward_clock_swing_apex weight)``."""

    source: str = "built-in defaults (X08:c6 params/env.yaml, 2026-09-22)"
    track_lin_w: float = 1.25
    track_lin_std: float = 0.2
    track_ang_w: float = 1.0
    track_ang_std: float = 0.5
    l1_w: float = -2.0
    clock: tuple[float, float] = CLOCK_HELD          # (schedule weight, apex weight)
    combine: str = "product"
    progress_w: float = 0.6
    yaw_progress_w: float = 1.0
    facing_w: float = 0.2
    facing_turn_w: float = 0.0
    air_w: float = 0.8
    cross_w: float = -3.0
    action_rate_w: float = -0.3
    orient_w: float = -0.7
    lin_vel_z_w: float = -0.15
    foot_idle_w: float = -0.12
    excess_feet_w: float = -0.2
    delta_torque_w: float = -1e-6
    notes: tuple[str, ...] = ()

    @property
    def clock_w(self) -> float:
        return self.clock[0]

    @property
    def apex_w(self) -> float:
        return self.clock[1]

    def with_clock(self, clock: tuple[float, float]) -> "Economy":
        return replace(self, clock=clock)


#: name -> Economy field the term prices (used both to build the default Economy from a dump and
#: to detect nonzero weights this module does NOT know how to price -- see NOT_REPLAYED_NOTES).
_REPLAYED_TERMS = {
    "track_lin_vel_xy_exp": "track_lin_w",
    "track_ang_vel_z_exp": "track_ang_w",
    "penalty_tracking_error_l1": "l1_w",
    "reward_clock_schedule": None,             # -> Economy.clock[0], set separately
    "reward_clock_swing_apex": None,           # -> Economy.clock[1], set separately
    "reward_forward_progress_along_command": "progress_w",
    "reward_yaw_progress": "yaw_progress_w",
    "reward_tracking_yaw": "facing_w",
    "reward_facing_turn": "facing_turn_w",
    "reward_feet_air_time_positive": "air_w",
    "penalty_lin_vel_y": "cross_w",
    "reward_action_rate": "action_rate_w",
    "reward_orientation": "orient_w",
    "reward_lin_vel_z": "lin_vel_z_w",
    "penalty_foot_idle_when_forward": "foot_idle_w",
    "penalty_excess_feet_contact_forward": "excess_feet_w",
    "reward_delta_torques": "delta_torque_w",
}


def economy_from_env_yaml(path: Path | str, *, clock: tuple[float, float] = CLOCK_HELD) -> Economy:
    """:class:`Economy` from a training ``params/env.yaml`` dump (``RMT._yaml_load_tolerant``, the
    same untrusted-tag-safe loader RMT uses). Every replayed weight/std/turn-radius is read off
    the dump; anything the dump does not carry keeps the X08:c6 default. ``notes`` lists every
    OTHER nonzero-weight reward term in the dump (module docstring's NOT_REPLAYED)."""
    path = Path(path)
    d = RMT._yaml_load_tolerant(path)
    fields: dict[str, Any] = {"source": f"env: {path}"}
    rw = (d.get("rewards") or {})
    notes: list[str] = []
    for name, cfg in rw.items():
        w = float((cfg or {}).get("weight") or 0.0)
        prm = (cfg or {}).get("params") or {}
        if name == "track_lin_vel_xy_exp":
            fields["track_lin_std"] = float(prm.get("std", Economy.track_lin_std))
        elif name == "track_ang_vel_z_exp":
            fields["track_ang_std"] = float(prm.get("std", Economy.track_ang_std))
        elif name == "reward_clock_schedule":
            fields["combine"] = str(prm.get("combine", Economy.combine))
        elif name == "reward_yaw_progress":
            cap = prm.get("clip")
            if cap is not None:
                notes.append(f"reward_yaw_progress clip={cap!r} in the dump; this replay uses "
                             "OF's own hardcoded ANG_VEL_MAX=0.3 (the two agree for X08:c6, where "
                             "clip is None and commands.base_velocity.ang_vel_max is 0.3)")
        key = _REPLAYED_TERMS.get(name)
        if key is not None:
            fields[key] = w
        elif name not in ("reward_clock_schedule", "reward_clock_swing_apex") and w != 0.0:
            notes.append(f"{name} (weight {w:g}) has a nonzero weight in this dump but no replayed "
                         "formula -- NOT REPLAYED (see the module docstring's NOT_REPLAYED_NOTES)")
    fields["clock"] = tuple(clock)
    fields["notes"] = tuple(notes)
    return Economy(**fields)


NOT_REPLAYED_NOTES = (
    "reward_stride_length (RewardStrideLength) and penalty_tibia_deviation_in_stance "
    "(penalty_joint_deviation_when_in_contact) are the OF tool's own two gaps, inherited "
    "unchanged: the OF tool measured reward_stride_length at 0.000/s income on every walking "
    "head it checked, and penalty_tibia_deviation_in_stance needs a per-joint in-stance mask "
    "against default_joint_pos that this replay's npz does not carry. Any OTHER nonzero-weight "
    "term in the weights dump is listed by Economy.notes, generated at load time -- see "
    "economy_from_env_yaml."
)

#: Known, stated simplifications relative to the shipped code (kept 1:1 with the OF tool per the
#: plan's reuse instruction, not "fixed" here):
#: * ``action_rate`` (OF's formula) does not model X08:c6's ``action_delay_steps=[1, 1]`` the way
#:   RMT's own ``action_rate`` does; it is a REGULARIZER, outside RG1-RG3's command/clock reading.
#: * ``air_time`` uses exact per-step timing (OF's formula), not RMT's float32 contact-sensor
#:   clock emulation (RMT's ``airtime_exact`` counterfactual, not its ``airtime`` training mode).
KNOWN_SIMPLIFICATIONS = (
    "action_rate has no action_delay (OF's formula; a regularizer, not read by RG1-RG3); "
    "air_time uses exact step timing, not the float32 contact-sensor clock drift RMT emulates "
    "(RMT's own 'airtime_exact' counterfactual, not its training-faithful 'airtime')."
)


# ============================================================================ the action clip
def _check_clip(clip: float) -> float:
    clip = float(clip)
    if not (math.isfinite(clip) and clip > 0.0):
        raise ValueError(f"action clip must be finite and > 0, got {clip!r}")
    return clip


def head_agent_yaml(meta: dict) -> Path:
    """The scored head's ``params/agent.yaml``: beside ``run_meta.checkpoint`` (RMT's
    ``_train_dumps_for`` convention). A stamp with no checkpoint is refused."""
    ckpt = meta.get("checkpoint")
    if not ckpt:
        raise SystemExit("run_meta has no checkpoint, so the head's params/agent.yaml (clip_actions) "
                         "cannot be found; pass --action-clip explicitly")
    return Path(ckpt).parent / "params" / "agent.yaml"


def action_clip_from_agent_yaml(path: Path | str) -> float | None:
    """``clip_actions`` of a training ``params/agent.yaml`` (module docstring's THE ACTION CLIP).
    ``None`` only when the dump says ``clip_actions: null`` (the wrapper then does not clamp). A
    missing file or key is refused, never defaulted."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"{path}: not found, so the runner's clip_actions is unknown; pass "
                         "--action-clip explicitly")
    d = RMT._yaml_load_tolerant(path) or {}
    if "clip_actions" not in d:
        raise SystemExit(f"{path}: no clip_actions key; pass --action-clip explicitly")
    return None if d["clip_actions"] is None else _check_clip(d["clip_actions"])


def runner_clipped_actions(actions: np.ndarray, action_clip: float | None) -> np.ndarray:
    """What ``reward_action_rate`` reads (before the action delay): the runner's ``+-clip_actions``
    clamp, then the action term's own per-joint clip (``OF.ACTION_CLIP``). ``action_clip=None``
    means the runner applied no clamp, which is OF's own model -- byte-identical to OF's ``a``."""
    a = np.asarray(actions).astype(np.float64)
    if action_clip is not None:
        a = np.clip(a, -action_clip, action_clip)
    return np.clip(a, -OF.ACTION_CLIP, OF.ACTION_CLIP)


def action_rate_primitive(actions: np.ndarray, action_clip: float | None) -> np.ndarray:
    """OF's ``arate`` formula (``||a_t - a_{t-1}||``, 0 at t=0) on :func:`runner_clipped_actions`."""
    a = runner_clipped_actions(actions, action_clip)
    arate = np.zeros(a.shape[0])
    arate[1:] = np.linalg.norm(a[1:] - a[:-1], axis=1)
    return arate


# ============================================================================ per-step primitives
#: The turn-type holds (the eval analogue of a drawn turn slot; OF's ``TURN_HOLDS`` plus the
#: suite's mirrored ``move_turn_r``), named by LABEL and resolved against each stamp's own
#: ``hold_labels`` -- never by OF's positional slow__omni segment ids.
TURN_HOLD_LABELS = ("turn_l90", "turn_r90", "move_turn", "move_turn_r")


def _segment_runs_parity(seg: np.ndarray) -> np.ndarray:
    """A 0/1 stand-in for ``segment_id`` that changes value exactly where ``seg`` does. OF's
    ``episode_primitives`` reads ``segment_id`` for two things only: its slow__omni 8-hold guard
    (which must never see suite data -- the suite has 11 labels, segment ids up to 10) and the
    hold-boundary reset of ``idle_reset`` (``seg[t] != seg[t-1]``), which this preserves exactly.
    The real ids are restored on the returned dict, and ``turn_hold`` is rebuilt from the stamp's
    own labels (:func:`episode_primitives`)."""
    seg = np.asarray(seg)
    runs = np.zeros(seg.shape, dtype=np.int32)
    if seg.size > 1:
        runs[1:] = np.cumsum(seg[1:] != seg[:-1])
    return runs % 2


def episode_primitives(z: dict, *, action_clip: float | None, n_offsets: int = 36,
                       hold_labels: list[str] | tuple[str, ...] | None = None) -> dict:
    """OF's ``episode_primitives`` (transcribed shipped formulas), called directly for every
    per-step quantity that does not depend on the clock phase, with the phase itself re-sourced
    from the RECORDED ``clock_phase`` field when the npz carries it (module docstring) instead of
    OF's offset fit. Adds ``phase_source`` (``"recorded"`` or ``"fitted (clock_phase absent...)"``)
    to the returned dict.

    ``action_clip`` is the scored head's ``clip_actions`` (required, so no caller can fall back
    to OF's runner-less clip by omission); ``arate`` is re-derived from it
    (:func:`action_rate_primitive`). ``None`` = no runner clamp = OF's own ``arate``.

    ``hold_labels`` is the stamp's OWN label list (``run_meta.json``; :func:`score_rollout` always
    passes it). OF is handed a boundary-preserving 0/1 ``segment_id`` (:func:`_segment_runs_parity`)
    so its slow__omni 8-hold guard never sees suite data; the returned ``seg`` is the real one and
    ``turn_hold`` is rebuilt from ``hold_labels`` (:data:`TURN_HOLD_LABELS`). ``None`` means OF's
    own slow__omni schedule (its canary raws), and is refused for segment ids outside it."""
    seg = np.asarray(z["segment_id"]).astype(np.int32)
    if hold_labels is None:
        if seg.size and int(seg.max()) >= len(OF.HOLD_LABELS):
            raise ValueError(f"segment_id {int(seg.max())} is outside OF's slow__omni schedule "
                             f"{list(OF.HOLD_LABELS)}: pass the stamp's own hold_labels")
        hold_labels = OF.HOLD_LABELS
    p = dict(OF.episode_primitives(dict(z, segment_id=_segment_runs_parity(seg)), n_offsets))
    p["arate"] = action_rate_primitive(z["actions"], action_clip)
    p["action_clip"] = action_clip
    p["seg"] = seg
    turn_ids = [i for i, lab in enumerate(hold_labels) if lab in TURN_HOLD_LABELS]
    p["turn_hold"] = np.isin(seg, turn_ids).astype(np.float64)
    if "clock_phase" in z:
        force = np.asarray(z["foot_force_norm"], dtype=np.float64)
        foot_v = np.asarray(z["foot_lin_vel_w"], dtype=np.float64)
        foot_p = np.asarray(z["foot_pos_w"], dtype=np.float64)
        root_p = np.asarray(z["root_pos_w"], dtype=np.float64)
        root_quat = np.asarray(z["root_quat_w"], dtype=np.float64)
        cmd = np.asarray(z["cmd_applied"], dtype=np.float64)
        v_clock = np.maximum(np.linalg.norm(cmd[:, :2], axis=1),
                             OF.OMNI_CLOCK_TURN_R * np.abs(cmd[:, 2]))
        running = v_clock > OF.CLOCK_CMD_STOP_M_S
        speed_xy = np.linalg.norm(foot_v[:, :, :2], axis=2)
        ground_z = root_p[:, 2] + OF.GROUND_OFFSET_FROM_ROOT_M
        height = foot_p[:, :, 2] - ground_z[:, None]
        _, _, gz = OF._gravity_b(root_quat)
        upright = (-gz > OF.MIN_UPRIGHT_GZ).astype(np.float64)
        phi = np.asarray(z["clock_phase"], dtype=np.float64)[:, None] + OF.FOOT_OFFSETS[None, :]
        clock_sum, clock_prod = OF._clock_income(phi, force, speed_xy, running)
        apex = OF._apex_income(phi, height, running)
        p["clock_sum"] = clock_sum * upright
        p["clock_prod"] = clock_prod * upright
        p["apex"] = apex * upright
        p["phase_source"] = "recorded"
        p["offset_best"] = None
        p["offset_scores"] = None
    else:
        p["phase_source"] = (f"fitted ({n_offsets}-offset grid, OF.episode_primitives -- "
                             "clock_phase absent from this npz)")
    return p


def hold_mask(p: dict, hold_idx: int) -> np.ndarray:
    """Steady, pre-termination steps of one hold -- OF's own ``hold_mask``, by index (this tool's
    loader resolves the index from the run's own ``hold_labels``; see the module docstring)."""
    return (p["seg"] == hold_idx) & p["steady"] & p["valid"]


# ============================================================================ scoring
#: term key -> (Economy field(s) for the weight, primitives key(s) for the value). Mirrors OF's
#: ``term_incomes`` term by term; see the module docstring's REUSE section.
def term_values(p: dict, econ: Economy) -> dict[str, np.ndarray]:
    """Every replayed term's PRE-CLIP per-step ``weight * value`` series (length T, unmasked --
    callers mask with :func:`hold_mask`). Same formulas as OF's ``term_incomes``, generalised from
    its hardcoded ``WEIGHTS``/``Terms`` to an arbitrary :class:`Economy`."""
    clock = p["clock_sum"] if econ.combine == "sum" else p["clock_prod"]
    ang = np.exp(-p["err_wz_sq"] / econ.track_ang_std**2)
    lin = np.exp(-p["err_xy_sq"] / econ.track_lin_std**2)
    facing_turn_bonus = p["facing"] * p["turn_hold"]
    return {
        "track_lin_exp": econ.track_lin_w * lin,
        "track_ang_exp": econ.track_ang_w * ang,
        "clock_sched": econ.clock_w * clock,
        "clock_apex": econ.apex_w * p["apex"],
        "progress": econ.progress_w * p["progress"],
        "yaw_progress": econ.yaw_progress_w * p["yaw_prog"],
        "facing": econ.facing_w * p["facing"],
        "facing_turn": econ.facing_turn_w * facing_turn_bonus,
        "air_time": econ.air_w * p["air"],
        "L1": econ.l1_w * p["err_l1"],
        "cross_vel": econ.cross_w * p["cross"],
        "action_rate": econ.action_rate_w * p["arate"],
        "orientation": econ.orient_w * p["orient"],
        "lin_vel_z": econ.lin_vel_z_w * p["vz2"],
        "foot_idle": econ.foot_idle_w * p["idle"],
        "excess_feet": econ.excess_feet_w * p["excess"],
        "delta_torque": econ.delta_torque_w * p["dtau"],
    }


#: term keys that are also part of the clock-removal set (RG3's "clock terms").
CLOCK_TERMS = ("clock_sched", "clock_apex")


def pre_clip_total(p: dict, econ: Economy) -> np.ndarray:
    """The plan's "per-step pre-clip total": the sum of every :func:`term_values` series, one
    number per env-step (length T, unmasked)."""
    vals = term_values(p, econ)
    total = np.zeros(p["T"])
    for v in vals.values():
        total = total + v
    return total


def term_incomes(p: dict, mask: np.ndarray, econ: Economy) -> dict[str, float]:
    """Per-second income (mean of ``weight * value`` over ``mask``) of every replayed term -- the
    same shape/semantics as OF's own ``term_incomes`` (the reproduction test below checks this
    function against it directly)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {}
    vals = term_values(p, econ)
    return {k: float(v[m].mean()) for k, v in vals.items()}


def totals(inc: dict[str, float]) -> tuple[float, float, float]:
    """(positive subtotal, negative subtotal, net) -- OF's own ``totals``."""
    pos = sum(v for v in inc.values() if v > 0)
    neg = sum(v for v in inc.values() if v < 0)
    return pos, neg, pos + neg


def head_hold_income(eps: list[dict], hold_idx: int, econ: Economy) -> tuple[dict[str, float], int]:
    """Step-weighted mean income per term across every episode, for one hold index -- OF's own
    ``head_hold_income``, generalised from a hold NAME to a hold INDEX (module docstring)."""
    acc: dict[str, float] = {}
    n = 0
    for p in eps:
        m = hold_mask(p, hold_idx)
        k = int(m.sum())
        if k == 0:
            continue
        inc = term_incomes(p, m, econ)
        for key, v in inc.items():
            acc[key] = acc.get(key, 0.0) + v * k
        n += k
    if n == 0:
        return {}, 0
    return {k: v / n for k, v in acc.items()}, n


# ============================================================================ suite-npz loading
def load_suite_rollout(run_dir: Path | str, *, limit: int | None = None,
                       action_clip: float | None = None) -> dict:
    """Every episode of one ``suite__omni`` eval stamp, following RMT's ``load_run``'s directory
    contract (``run_meta.json`` + ``complete.json`` + ``raw/episode_*.npz``). Category and hold
    come from the run's OWN ``env_index_map``/``hold_labels`` (never a hardcoded label tuple --
    see the module docstring's REUSE section for why that matters). The runner's action clip is
    the head's own ``clip_actions`` (:func:`head_agent_yaml`) unless ``action_clip`` passes one
    explicitly. Returns ``{run_dir, meta, action_clip, action_clip_source,
    episodes: [{env_index, category, hold, hold_idx, z}]}``."""
    run_dir = Path(run_dir)
    if not (run_dir / "complete.json").is_file():
        raise SystemExit(f"{run_dir}: no complete.json -- the stamp is partial; refusing")
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if not meta.get("suite"):
        raise SystemExit(f"{run_dir}: not a category-suite stamp (run_meta has no 'suite')")
    if action_clip is None:
        agent_yaml = head_agent_yaml(meta)
        clip = action_clip_from_agent_yaml(agent_yaml)
        clip_source = f"agent: {agent_yaml}"
    else:
        clip = _check_clip(action_clip)
        clip_source = "explicit (--action-clip)"
    hold_labels = list(meta["hold_labels"])
    emap = meta.get("env_index_map") or []
    n_total = int(meta.get("episodes_total") or len(emap))
    episodes = []
    for ent in emap[: (limit or n_total)]:
        idx = int(ent["env_index"])
        cat, label = ent.get("category"), ent.get("label")
        if cat not in RMT.CATEGORIES or label not in hold_labels:
            continue
        with np.load(run_dir / "raw" / f"episode_{idx:02d}.npz") as npz:
            z = {k: npz[k] for k in npz.files}
        episodes.append({"env_index": idx, "category": cat, "hold": label,
                         "hold_idx": hold_labels.index(label), "z": z})
    return {"run_dir": str(run_dir), "meta": meta, "action_clip": clip,
            "action_clip_source": clip_source, "episodes": episodes}


def score_rollout(rollout: dict, *, n_offsets: int = 36) -> list[dict]:
    """Every episode's primitives (:func:`episode_primitives`, at the rollout's own
    ``action_clip``), plus its category/hold/hold_idx.
    Computed ONCE per rollout -- the phase does not depend on the economy, so both variants
    (clock-held / clock-free) reuse the same primitives (the only per-economy work is
    :func:`term_values`/:func:`pre_clip_total`, applied downstream)."""
    out = []
    hold_labels = list(rollout["meta"]["hold_labels"])
    for ep in rollout["episodes"]:
        p = episode_primitives(ep["z"], action_clip=rollout["action_clip"], n_offsets=n_offsets,
                               hold_labels=hold_labels)
        out.append({**{k: ep[k] for k in ("env_index", "category", "hold", "hold_idx")}, "p": p})
    return out


def phase_source_counts(scored: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in scored:
        src = e["p"]["phase_source"]
        key = "recorded" if src == "recorded" else "fitted"
        out[key] = out.get(key, 0) + 1
    return out


# ============================================================================ RG1 / RG2 / RG3
def rg1_ordering(healthy: dict[str, list[dict]], degenerate: dict[str, list[dict]],
                 econ_held: Economy, econ_free: Economy,
                 categories: tuple = MOVING_CATEGORIES) -> dict:
    """RG1: every (healthy, degenerate, category) triple's ordering and ratio, both economies.

    Plan step 9: "every healthy > degenerate ordering SURVIVES" -- so only a triple whose
    CLOCK-HELD ordering holds is an RG1 criterion; it passes when the clock-free ordering holds
    too and the clock-free ratio is >= ``min(1.10, clock-held ratio)``. A triple already reversed
    under clock-held is reported in ``held_reversed`` (``pass: None``, "no clock-held ordering:
    not an RG1 criterion") and never fails RG1 -- clock removal did not cause it.

    A pair whose degenerate total is <= 0 (audit item 11) is reported, never divided: ``ratio`` is
    None and ``note`` explains why; the triple is then judged on its clock-free ordering alone.
    Each category's income pools EVERY hold of the category (both hands; :func:`_category_income`).
    """
    rows = []
    for hname, heps in healthy.items():
        for dname, deps in degenerate.items():
            for cat in categories:
                inc_h_held, n_h = _category_income(heps, cat, econ_held)
                inc_d_held, n_d = _category_income(deps, cat, econ_held)
                inc_h_free, _ = _category_income(heps, cat, econ_free)
                inc_d_free, _ = _category_income(deps, cat, econ_free)
                if not inc_h_held or not inc_d_held:
                    continue
                _, _, th_held = totals(inc_h_held)
                _, _, td_held = totals(inc_d_held)
                _, _, th_free = totals(inc_h_free)
                _, _, td_free = totals(inc_d_free)
                row: dict[str, Any] = {
                    "healthy": hname, "degenerate": dname, "category": cat,
                    "n_healthy": n_h, "n_degenerate": n_d,
                    "held": {"healthy_total": th_held, "degenerate_total": td_held,
                             "ordering": th_held > td_held},
                    "free": {"healthy_total": th_free, "degenerate_total": td_free,
                             "ordering": th_free > td_free},
                }
                if td_held <= 0.0 or td_free <= 0.0:
                    row["ratio_held"] = None
                    row["ratio_free"] = None
                    row["note"] = ("RG1 ratio undefined: degenerate total <= 0 (held "
                                   f"{td_held:.4f}, free {td_free:.4f}); reported, not divided "
                                   "(audit item 11) -- judged on the clock-free ordering only")
                    row["pass"] = bool(row["free"]["ordering"])
                else:
                    ratio_held = th_held / td_held
                    ratio_free = th_free / td_free
                    floor = min(RG1_RATIO_TARGET, ratio_held)
                    row["ratio_held"] = ratio_held
                    row["ratio_free"] = ratio_free
                    row["ratio_floor"] = floor
                    row["pass"] = bool(row["free"]["ordering"] and ratio_free >= floor)
                if not row["held"]["ordering"]:
                    row["pass"] = None
                    row["note"] = "no clock-held ordering: not an RG1 criterion"
                rows.append(row)
    judged = [r for r in rows if r["held"]["ordering"]]
    return {"rows": judged,
            "held_reversed": [r for r in rows if not r["held"]["ordering"]],
            "pass": all(r["pass"] for r in judged) if judged else None}


def rg2_zero_income_fraction(healthy: dict[str, list[dict]], econ_held: Economy, econ_free: Economy,
                             categories: tuple = MOVING_CATEGORIES) -> dict:
    """RG2: pooled over every healthy head's moving-hold steps, the fraction of
    :func:`pre_clip_total` <= 0, clock-held vs clock-free."""
    held_vals: list[float] = []
    free_vals: list[float] = []
    for heps in healthy.values():
        for e in heps:
            if e["category"] not in categories:
                continue
            m = hold_mask(e["p"], e["hold_idx"])
            if not m.any():
                continue
            held_vals.append(pre_clip_total(e["p"], econ_held)[m])
            free_vals.append(pre_clip_total(e["p"], econ_free)[m])
    if not held_vals:
        return {"n": 0, "pass": None, "note": "no healthy moving-hold steps"}
    held = np.concatenate(held_vals)
    free = np.concatenate(free_vals)
    frac_held = float((held <= 0.0).mean())
    frac_free = float((free <= 0.0).mean())
    rise = frac_free - frac_held
    ok = bool(rise <= RG2_MAX_RISE + 1e-12 and frac_free <= RG2_MAX_SHARE + 1e-12)
    return {"n": int(held.size), "frac_held": frac_held, "frac_free": frac_free,
           "rise": rise, "pass": ok}


def rg3_income_share_removed(healthy: dict[str, list[dict]], econ_held: Economy, econ_free: Economy,
                             categories: tuple = ALL_CATEGORIES) -> dict:
    """RG3 (report only): per category, the share of clock-held POSITIVE income clock removal
    takes away, pooled over every healthy head. A category whose held positive income is <= 0
    (audit item 11's guard, applied the same way) is reported, not divided. Flags the stand hold
    at the pre-registered 30% threshold (plan step 9)."""
    rows = {}
    for cat in categories:
        inc_held_acc: dict[str, float] = {}
        inc_free_acc: dict[str, float] = {}
        n = 0
        for heps in healthy.values():
            inc_held, k = _category_income(heps, cat, econ_held)
            inc_free, _ = _category_income(heps, cat, econ_free)
            if not inc_held or not k:
                continue
            for key, v in inc_held.items():
                inc_held_acc[key] = inc_held_acc.get(key, 0.0) + v * k
            for key, v in inc_free.items():
                inc_free_acc[key] = inc_free_acc.get(key, 0.0) + v * k
            n += k
        if n == 0:
            rows[cat] = {"n": 0, "share": None, "note": "no healthy episodes for this category"}
            continue
        inc_held = {k: v / n for k, v in inc_held_acc.items()}
        inc_free = {k: v / n for k, v in inc_free_acc.items()}
        pos_held, _, _ = totals(inc_held)
        pos_free, _, _ = totals(inc_free)
        if pos_held <= 0.0:
            rows[cat] = {"n": n, "share": None,
                        "note": f"held positive income <= 0 ({pos_held:.4f}); reported, not divided"}
            continue
        share = (pos_held - pos_free) / pos_held
        row = {"n": n, "pos_held": pos_held, "pos_free": pos_free, "share": share}
        if cat == "stand" and share >= RG3_STAND_FLAG:
            row["flag"] = (f"stand share {share * 100:.1f}% >= {RG3_STAND_FLAG * 100:.0f}%: the "
                           "stand.steps_per_s guard is the likely BREAK route (plan step 9)")
        rows[cat] = row
    return rows


def _category_income(eps: list[dict], category: str, econ: Economy) -> tuple[dict[str, float], int]:
    """Step-weighted mean income per term over EVERY hold of ``category`` -- each episode masked
    on its OWN ``hold_idx`` (both hands: lat_l+lat_r, turn_l90+turn_r90, move_turn+move_turn_r),
    as RG2 already does."""
    acc: dict[str, float] = {}
    n = 0
    for e in eps:
        if e["category"] != category:
            continue
        inc, k = head_hold_income([e["p"]], e["hold_idx"], econ)
        for key, v in inc.items():
            acc[key] = acc.get(key, 0.0) + v * k
        n += k
    if n == 0:
        return {}, 0
    return {k: v / n for k, v in acc.items()}, n


# ============================================================================ reporting / CLI
def _head_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected NAME=RUN_DIR, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--healthy", action="append", type=_head_arg, default=[], metavar="NAME=RUN_DIR",
                    help="a healthy suite__omni stamp (repeatable; e.g. E1=<run_dir>)")
    ap.add_argument("--degenerate", action="append", type=_head_arg, default=[], metavar="NAME=RUN_DIR",
                    help="a degenerate suite__omni stamp (repeatable; e.g. E6=<run_dir>)")
    ap.add_argument("--weights-env-yaml", default=DEFAULT_WEIGHTS_ENV_YAML,
                    help="the ONE economy every rollout is priced under (default: X08:c6's own dump)")
    ap.add_argument("--n-offsets", type=int, default=36, help="clock-phase fallback fit grid size")
    ap.add_argument("--limit", type=int, default=0, help="episodes per rollout (0 = all)")
    ap.add_argument("--action-clip", type=float, default=None,
                    help="the runner's clip_actions for EVERY rollout (default: each stamp's own "
                         "head's params/agent.yaml, beside run_meta.checkpoint)")
    ap.add_argument("--out", default=str(CAMPAIGN_DIR / "replay_clock_economy.json"))
    args = ap.parse_args(argv)

    econ_held = economy_from_env_yaml(args.weights_env_yaml, clock=CLOCK_HELD)
    econ_free = replace(econ_held, clock=CLOCK_FREE)
    action_clips: dict[str, dict] = {}

    def _load_group(pairs: list[tuple[str, str]]) -> dict[str, list[dict]]:
        out = {}
        for name, path in pairs:
            rollout = load_suite_rollout(path, limit=args.limit or None, action_clip=args.action_clip)
            action_clips[name] = {"clip": rollout["action_clip"], "source": rollout["action_clip_source"]}
            out[name] = score_rollout(rollout, n_offsets=args.n_offsets)
        return out

    healthy = _load_group(args.healthy)
    degenerate = _load_group(args.degenerate)
    for name, c in action_clips.items():
        print(f"[replay] {name}: action clip {c['clip']} ({c['source']})")

    if not healthy or not degenerate:
        print("[replay] at least one --healthy and one --degenerate stamp are required for "
              "RG1/RG2/RG3; printing the economy only")

    report: dict[str, Any] = {
        "economy_source": econ_held.source,
        "economy_notes": list(econ_held.notes),
        "not_replayed": NOT_REPLAYED_NOTES,
        "known_simplifications": KNOWN_SIMPLIFICATIONS,
        "action_clips": action_clips,
        "phase_sources": {name: phase_source_counts(eps)
                          for name, eps in {**healthy, **degenerate}.items()},
    }
    if healthy and degenerate:
        report["RG1"] = rg1_ordering(healthy, degenerate, econ_held, econ_free)
        report["RG2"] = rg2_zero_income_fraction(healthy, econ_held, econ_free)
        report["RG3"] = rg3_income_share_removed(healthy, econ_held, econ_free)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    print(f"[replay] wrote {out_path}")
    if "RG1" in report:
        print(f"[replay] RG1: {'PASS' if report['RG1']['pass'] else 'FAIL'}")
        for r in report["RG1"]["held_reversed"]:
            print(f"  {r['healthy']}>{r['degenerate']} {r['category']}: no clock-held ordering: "
                  "not an RG1 criterion")
        print(f"[replay] RG2: {'PASS' if report['RG2']['pass'] else 'FAIL'} "
              f"(held {report['RG2'].get('frac_held')}, free {report['RG2'].get('frac_free')})")
        print("[replay] RG3 (report only):")
        for cat, row in report["RG3"].items():
            print(f"  {cat}: share={row.get('share')}" + (f"  ! {row['flag']}" if row.get("flag") else ""))
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
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


if __name__ == "__main__":
    sys.exit(main())
