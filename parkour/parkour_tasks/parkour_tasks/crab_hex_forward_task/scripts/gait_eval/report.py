# SPDX-License-Identifier: BSD-3-Clause
"""Scoring orchestration and report writing for the crab-hex gait eval harness.

Pure -- takes the harness's logged numpy arrays and turns them into per-episode / per-scenario
metrics, JSON files, a human-readable summary, and an optional JSONL history entry. No Isaac Sim,
so the whole scoring path is exercisable offline against a saved NPZ.

History mirrors ``parkour/scripts/curriculum_metrics.py`` (which trends *training* metrics) so a
tripod score can be trended across checkpoints with the same muscle memory, but the two are
deliberately separate: that one parses training stdout, this one measures play-time gait.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gait_eval import metrics as M
from gait_eval.schedule import CompiledSchedule, Scenario


def _window_mask(
    *,
    segment_id: np.ndarray,
    steady: np.ndarray,
    hold_idx: int,
    n_steps: int,
    fall_trim_steps: int,
    ended_in_fall: bool,
) -> np.ndarray:
    """Steps of one hold that are eligible for scoring, for a single env."""
    total = segment_id.shape[0]
    valid = np.zeros(total, dtype=bool)
    upper = min(n_steps, total)
    if ended_in_fall:
        # The last half-second before a fall is a fall, not a gait sample.
        upper = max(0, upper - fall_trim_steps)
    valid[:upper] = True
    return valid & (segment_id == hold_idx) & steady


def _travel_yaw_at(travel_yaw, index) -> Any:
    """Select ``travel_yaw`` for a step selection: a scalar passes through, a ``(T,)`` array is
    indexed (so each step is rotated by its own command direction -- omni rows)."""
    if np.ndim(travel_yaw) == 0:
        return travel_yaw
    return np.asarray(travel_yaw, dtype=np.float64)[index]


def score_episode(
    raw: dict[str, np.ndarray],
    *,
    env_idx: int,
    compiled: CompiledSchedule,
    scenario: Scenario,
    dt: float,
    term_reason: str,
    n_steps: int,
    action_groups: dict[str, list[int]] | None = None,
    joint_vel_groups: dict[str, list[int]] | None = None,
    travel_yaw: float = 0.0,
    motion_groups: dict[str, list[int]] | None = None,
    action_clip: float | None = None,
) -> dict:
    """All derived metrics for one episode, per hold plus an episode roll-up.

    ``travel_yaw`` is the yaw offset of the travel frame from the body frame (0.0 for the forward
    task, +pi/2 for the sideways task); it is threaded through to every metric that reads
    orientation or velocity so the harness serves both tasks unchanged at 0.0. On an omni
    (``command_frame == "world"``) row it is a ``(T,)`` array instead -- the direction of the body
    command at each step, since the travel frame turns with the command.

    On a world row the command is additionally rotated into the travel frame before
    ``tracking_metrics`` sees it (``(||cmd_xy||, 0, wz)``), so ``vx`` keeps meaning
    "along the commanded direction" and ``vx.ratio`` keeps meaning achieved/commanded speed; and,
    when the harness logged ``heading_err``, each hold gets a :func:`gait_eval.metrics.facing_metrics`
    entry over the WHOLE hold (a turn command is the transient -- masking the settle window out
    would discard the steps that measure it). Per hold the facing target is constant, so the hold
    entries carry ``net_yaw_rate_toward``; the episode-level entry passes ``constant_target=False``
    because an episode's target steps between holds.

    World rows also carry, per hold, ``motion`` (:func:`gait_eval.metrics.hold_motion_metrics` over
    the whole hold) and -- only when the hold's commanded speed is > 0 -- ``direction``
    (:func:`gait_eval.metrics.travel_direction_metrics` on the tracking window's travel-frame
    velocity). On a category-suite row (the compiled schedule carries ``wave_of_env``) the episode
    also carries its ``wave``, its own ``hold``, that hold's ``category`` and ``hand``. Body rows
    gain none of these keys.

    On a category-suite row the episode's OWN hold additionally carries ``quality``
    (:func:`gait_eval.metrics.motion_quality_metrics` over the whole hold, gait-motion-quality plan
    step 1): ``motion_groups`` are :func:`gait_eval.metrics.motion_joint_groups` of the
    articulation's joint names (``None`` leaves the joint/work metrics unmeasured) and
    ``action_clip`` the runner's ``clip_actions`` (the actions are clipped before the action
    metrics, as the env clips them). Every other hold and every non-suite row is unchanged.
    """
    world = scenario.command_frame == "world"
    has_facing = world and "heading_err" in raw
    threshold = float(scenario.get_default("contact_force_threshold"))
    min_window_steps = max(1, int(round(float(scenario.get_default("min_window_s")) / dt)))
    fall_trim = int(round(float(scenario.get_default("fall_exclusion_s")) / dt))
    min_cycles = int(scenario.get_default("min_cycles"))
    slip_thresh = float(scenario.get_default("slip_speed_threshold"))
    lin_vel_clip = float(scenario.get_default("lin_vel_clip"))
    ended_in_fall = term_reason in ("fall", "hard_fall")
    own_label = own_category = None
    if world and compiled.hold_of_env is not None and scenario.suite is not None:
        own_label = compiled.hold_labels[int(compiled.hold_of_env[env_idx])]
        own_category = scenario.suite.category_of_hold().get(own_label)

    contact_all = raw["foot_force_norm"][:, env_idx, :] > threshold
    segment_id = compiled.segment_id[:, env_idx]
    steady = compiled.steady_mask[:, env_idx]

    out: dict[str, Any] = {
        "env_index": env_idx,
        "termination_reason": term_reason,
        "n_steps": int(n_steps),
        "duration_s": float(n_steps * dt),
        "completed_schedule": bool(term_reason == "schedule_complete"),
        "contact_force_threshold": threshold,
        "holds": {},
    }

    scored, weights, discarded = [], [], []
    for hold_idx, label in enumerate(compiled.hold_labels):
        mask = _window_mask(
            segment_id=segment_id,
            steady=steady,
            hold_idx=hold_idx,
            n_steps=n_steps,
            fall_trim_steps=fall_trim,
            ended_in_fall=ended_in_fall,
        )
        idx = np.flatnonzero(mask)
        contact = contact_all[idx]
        tri = M.tripod_window_metrics(
            contact, dt=dt, min_window_steps=min_window_steps, min_cycles=min_cycles
        )
        entry: dict[str, Any] = {"label": label, "tripod": tri, "n_steps": int(idx.size)}

        # Tracking needs steps, not contact cycles: a policy with no measurable gait window can
        # still be creeping at 20% of command, and that is exactly the case the ratio must catch
        # (2026-08-21 creep-audit). So it is gated on window length only, never on tripod validity.
        vel_travel = None
        if idx.size >= min_window_steps:
            cmd_window = raw["cmd_applied"][idx, env_idx, :]
            if world:
                # The body command turns with the body; its travel-frame shape is
                # (speed, 0, wz) -- the same Shape B the sideways rows use, so tracking_metrics
                # and the driver's ratio gate are unchanged.
                speed = np.linalg.norm(cmd_window[:, :2], axis=-1)
                cmd_window = np.stack([speed, np.zeros_like(speed), cmd_window[:, 2]], axis=-1)
            vel_travel = M.lin_vel_to_travel_frame(
                raw["root_lin_vel_b"][idx, env_idx, :], _travel_yaw_at(travel_yaw, idx)
            )
            entry["tracking"] = M.tracking_metrics(
                cmd_window,
                vel_travel,
                raw["root_ang_vel_b"][idx, env_idx, :],
                ratio_min_cmd=lin_vel_clip,
            )

        if world:
            whole = _window_mask(
                segment_id=segment_id,
                steady=np.ones_like(steady),
                hold_idx=hold_idx,
                n_steps=n_steps,
                fall_trim_steps=fall_trim,
                ended_in_fall=ended_in_fall,
            )
            widx = np.flatnonzero(whole)
        if has_facing:
            entry["facing"] = M.facing_metrics(
                raw["heading_err"][widx, env_idx],
                raw["root_ang_vel_b"][widx, env_idx, 2],
                dt,
                wz_cmd_t=raw["cmd_applied"][widx, env_idx, 2],
            )
        if world:
            # Plan D3 (2026-09-21), additive and world-row-only like ``facing``: what the body did
            # over the whole hold, and -- on a moving hold only -- how well it held the commanded
            # direction (on the tracking window, in the travel frame tracking already reads).
            entry["motion"] = M.hold_motion_metrics(
                raw["root_pos_w"][widx, env_idx],
                raw["root_quat_w"][widx, env_idx],
                raw["root_lin_vel_b"][widx, env_idx],
                contact_all[widx],
                dt,
            )
            if scenario.schedule[hold_idx].as_world()[0] > 0.0:
                entry["direction"] = (
                    M.travel_direction_metrics(vel_travel[:, :2], cmd_window[:, 0])
                    if vel_travel is not None
                    else None
                )
            if own_category is not None and label == own_label:
                entry["quality"] = _quality_block(
                    raw, env_idx, widx, own_category, dt=dt, motion_groups=motion_groups,
                    action_clip=action_clip, hold_s_nominal=float(scenario.schedule[hold_idx].hold_s),
                )

        if tri["valid"]:
            entry["air_time"] = M.air_time_metrics(contact, dt=dt)
            entry["slip"] = M.slip_metrics(
                contact,
                raw["foot_pos_w"][idx, env_idx],
                raw["foot_lin_vel_w"][idx, env_idx],
                dt=dt,
                foot_force_norm=raw["foot_force_norm"][idx, env_idx],
                foot_ang_vel_w=raw["foot_ang_vel_w"][idx, env_idx],
                slip_speed_threshold=slip_thresh,
            )
            if not tri["low_confidence"]:
                scored.append(tri["tripod_score"])
                weights.append(int(idx.size))
        else:
            discarded.append({"label": label, "reason": tri["discard_reason"], "n_steps": int(idx.size)})
        out["holds"][label] = entry

    # Episode-level metrics use every logged step, not just steady windows -- orientation and action
    # smoothness are episode properties, and clipping them to holds would hide transients.
    ep = slice(0, max(1, n_steps))
    ep_contact = contact_all[ep]
    yaw = M.yaw_from_quat_wxyz(raw["root_quat_w"][ep, env_idx, :])
    out["air_time_episode"] = M.air_time_metrics(ep_contact, dt=dt)
    out["stride"] = M.stride_metrics(
        ep_contact,
        raw["foot_pos_w"][ep, env_idx],
        dt=dt,
        cmd_xy=raw["cmd_applied"][ep, env_idx, :2],
        # cmd_xy's x component is speed along the TRAVEL axis (Shape B), so the world-frame
        # stride must be rotated into the travel frame too, not just the body frame. On a world
        # row cmd_xy is already a true body-frame vector, so the body yaw is the right frame and
        # the projection onto it carries the direction.
        root_yaw=yaw if (world or (np.ndim(travel_yaw) == 0 and travel_yaw == 0.0)) else yaw + travel_yaw,
    )
    out["swing_clearance"] = M.swing_clearance_metrics(
        ep_contact, raw["foot_pos_w"][ep, env_idx], raw["terrain_z"][ep, env_idx]
    )
    out["slip_episode"] = M.slip_metrics(
        ep_contact,
        raw["foot_pos_w"][ep, env_idx],
        raw["foot_lin_vel_w"][ep, env_idx],
        dt=dt,
        foot_force_norm=raw["foot_force_norm"][ep, env_idx],
        foot_ang_vel_w=raw["foot_ang_vel_w"][ep, env_idx],
        slip_speed_threshold=slip_thresh,
    )
    out["orientation"] = M.orientation_metrics(
        raw["root_quat_w"][ep, env_idx, :], travel_yaw=_travel_yaw_at(travel_yaw, ep)
    )
    # PLAN G (2026-09-02): fall direction + support-polygon margins (heading frame, root as the
    # CoM proxy here; exact CoM needs run_meta body masses + the body_pos_w buffer offline).
    if "crab_failure" in raw:
        out["fall_direction"] = M.fall_direction_metrics(
            raw["root_quat_w"][ep, env_idx, :],
            raw["crab_failure"][ep, env_idx],
            dt=dt,
            root_ang_vel_b=raw["root_ang_vel_b"][ep, env_idx] if "root_ang_vel_b" in raw else None,
            travel_yaw=_travel_yaw_at(travel_yaw, ep),
        )
        # "Walking" is commanded speed above the clip: on a world row that is the norm of the
        # body command, not its x component (which is only the along-body part of it).
        cmd_speed = (
            np.linalg.norm(raw["cmd_applied"][ep, env_idx, :2], axis=-1)
            if world
            else np.abs(raw["cmd_applied"][ep, env_idx, 0])
        )
        walking = compiled.steady_mask[ep, env_idx] & (cmd_speed >= lin_vel_clip)
        out["support_polygon"] = M.support_polygon_metrics(
            raw["foot_pos_w"][ep, env_idx],
            raw["foot_force_norm"][ep, env_idx],
            raw["root_pos_w"][ep, env_idx],
            raw["root_quat_w"][ep, env_idx, :],
            dt=dt,
            crab_failure=raw["crab_failure"][ep, env_idx],
            walking_mask=walking,
            travel_yaw=_travel_yaw_at(travel_yaw, ep),
        )
    out["actions"] = M.action_metrics(
        raw["actions"][ep, env_idx],
        joint_groups=action_groups,
        joint_vel=raw["joint_vel"][ep, env_idx],
        joint_vel_groups=joint_vel_groups,
    )
    # Velocity-era spin gate: cam shafts should rotate continuously in one direction.
    if joint_vel_groups and joint_vel_groups.get("camshaft"):
        out["shaft_spin"] = M.shaft_spin_metrics(
            raw["joint_vel"][ep, env_idx],
            joint_vel_groups["camshaft"],
            dt=dt,
        )

    out["discarded_windows"] = discarded
    if scored:
        w = np.asarray(weights, dtype=np.float64)
        out["tripod_score"] = float(np.average(np.asarray(scored, dtype=np.float64), weights=w))
    else:
        # Explicitly null, never 0.0: "not measurable" and "measured and bad" need different fixes.
        out["tripod_score"] = None
    if has_facing:
        out["facing"] = M.facing_metrics(
            raw["heading_err"][ep, env_idx],
            raw["root_ang_vel_b"][ep, env_idx, 2],
            dt,
            wz_cmd_t=raw["cmd_applied"][ep, env_idx, 2],
            # An episode spans several holds, each with its own facing TARGET, so
            # ``net_yaw_rate_toward``'s constant-target derivation does not hold here: ask for it
            # only per hold (above), never episode-wide, so no one reads a meaningless number.
            constant_target=False,
        )
        out["facing_error_rms"] = out["facing"]["heading_error_rms"]
    out["tippy_tap_fraction"] = out["air_time_episode"].get("tippy_tap_fraction")
    out["slip_ratio_mean"] = out["slip_episode"]["pooled_slip_ratio"]["mean"]
    # Episode tracking ratio: mean over walking holds (ratio is None on sub-clip commands).
    ratios = [
        h["tracking"]["vx"]["ratio"]
        for h in out["holds"].values()
        if h.get("tracking") and h["tracking"]["vx"].get("ratio") is not None
    ]
    out["tracking_ratio"] = float(np.mean(ratios)) if ratios else None
    if compiled.wave_of_env is not None and compiled.hold_of_env is not None:
        # Category suite: which wave this episode ran in and its OWN hold (the settle hold is
        # never one); category / hand come from the suite so readers never re-derive them.
        own = compiled.hold_labels[int(compiled.hold_of_env[env_idx])]
        suite = scenario.suite
        out["wave"] = int(compiled.wave_of_env[env_idx])
        out["hold"] = own
        out["category"] = suite.category_of_hold().get(own) if suite is not None else None
        out["hand"] = suite.hand_of_hold().get(own) if suite is not None else None
    return out


def _quality_block(raw: dict, env_idx: int, widx: np.ndarray, category: str, *, dt: float,
                   motion_groups, action_clip, hold_s_nominal: float) -> dict | None:
    """``motion_quality_metrics`` of one episode's own hold; inputs the harness did not record (an
    older npz: no ``applied_torque`` / ``clock_running`` / ``rod_speed``) leave their metrics
    ``None``."""
    def _opt(key):
        return raw[key][widx, env_idx] if key in raw else None

    return M.motion_quality_metrics(
        category,
        dt=dt,
        actions=_opt("actions"),
        foot_force_norm=raw["foot_force_norm"][widx, env_idx],
        foot_pos_w=raw["foot_pos_w"][widx, env_idx],
        foot_lin_vel_w=raw["foot_lin_vel_w"][widx, env_idx],
        root_pos_w=raw["root_pos_w"][widx, env_idx],
        root_quat_w=raw["root_quat_w"][widx, env_idx],
        cmd_wz=raw["cmd_applied"][widx, env_idx, 2],
        joint_vel=_opt("joint_vel"),
        applied_torque=_opt("applied_torque"),
        groups=motion_groups,
        clock_running=_opt("clock_running"),
        rod_speed=_opt("rod_speed"),
        action_clip=action_clip,
        hold_s_nominal=hold_s_nominal,
    )


def score_run(
    raw: dict[str, np.ndarray],
    *,
    compiled: CompiledSchedule,
    scenario: Scenario,
    dt: float,
    term_reason: list[str],
    n_steps_env: list[int],
    action_groups: dict[str, list[int]] | None = None,
    joint_vel_groups: dict[str, list[int]] | None = None,
    travel_yaw: float = 0.0,
    motion_groups: dict[str, list[int]] | None = None,
    action_clip: float | None = None,
) -> list[dict]:
    """``travel_yaw`` is a scalar (forward/sideways), a ``(T,)`` array shared by every env, or a
    ``(T, E)`` array -- the per-step, per-env command direction the omni harness logs."""
    ndim = np.ndim(travel_yaw)
    return [
        score_episode(
            raw,
            env_idx=i,
            compiled=compiled,
            scenario=scenario,
            dt=dt,
            term_reason=term_reason[i],
            n_steps=n_steps_env[i],
            action_groups=action_groups,
            joint_vel_groups=joint_vel_groups,
            travel_yaw=(np.asarray(travel_yaw)[:, i] if ndim == 2 else travel_yaw),
            motion_groups=motion_groups,
            action_clip=action_clip,
        )
        for i in range(len(term_reason))
    ]


#: ``aggregate.suite.by_category`` keys -> where each is read on an episode's OWN hold entry
#: (plan D3). ``tripod`` is special-cased: ``tripod.tripod_score``, excluding ``low_confidence``
#: windows exactly as ``tripod_score_by_hold`` does.
CATEGORY_METRIC_SOURCES: dict[str, tuple[str, ...]] = {
    "ratio": ("tracking", "vx", "ratio"),
    "achieved_speed": ("tracking", "vx", "actual_mean"),
    "tripod": ("tripod", "tripod_score"),
    "facing_closed": ("facing", "facing_closed"),
    "yaw_progress": ("facing", "yaw_progress_rate"),
    "net_yaw": ("facing", "net_yaw_rate_toward"),
    "heading_final": ("facing", "heading_error_final_abs"),
    "settle_s": ("facing", "settle_time_s"),
    "drift_speed": ("motion", "speed_abs_mean_m_s"),
    "direction_error": ("direction", "direction_error_rad"),
    "cross_track_ratio": ("direction", "cross_track_ratio"),
    # Motion quality (gait-motion-quality plan step 1): the own hold's ``quality`` block. A key not
    # defined on the category is None there, so its by_category stats read n = 0 / median None.
    **{key: ("quality", key) for key in M.MOTION_QUALITY_KEYS},
}


def _category_value(hold: dict, key: str) -> float | None:
    """One ``by_category`` value from an episode's own hold entry, or ``None`` when unmeasured."""
    if key == "tripod":
        tri = hold.get("tripod") or {}
        return None if tri.get("low_confidence") else tri.get("tripod_score")
    value: Any = hold
    for part in CATEGORY_METRIC_SOURCES[key]:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def suite_aggregate_arg(run_meta: dict) -> dict | None:
    """``aggregate``'s ``suite=`` argument for a run: ``run_meta["suite"]`` plus the run-time
    ``waves`` / ``wave_seeds`` / ``episodes_total`` / ``trained_categories``; ``None`` on every
    non-suite row (so their aggregate is unchanged)."""
    if not run_meta.get("suite"):
        return None
    return {
        **run_meta["suite"],
        "waves": run_meta.get("waves"),
        "wave_seeds": run_meta.get("wave_seeds"),
        "episodes_total": run_meta.get("episodes_total"),
        "trained_categories": run_meta.get("trained_categories"),
    }


def _suite_block(episodes: list[dict], suite: dict, stats) -> dict:
    """``aggregate``'s ``suite`` block (plan D3): own-hold survival and the pooled ``by_category``.

    ``by_category[c][key]`` POOLS the per-episode values of the category's own-hold episodes over
    both hands (never an average of hand medians -- the pooled median is what the env-clustered
    bootstrap resamples). A hold's survival is the fraction of ITS OWN episodes with ``n_steps >=
    round((prologue_s + settle_hold_s + hold_s[label]) / dt)`` -- a fall on the wave's last step
    counts as survived (the legacy ``survival_by_hold`` convention), and ``term_reason`` is never
    read.
    """
    categories = suite.get("categories") or []
    dt = float(suite["dt"])
    prologue_s = float(suite["prologue_s"])
    settle_hold_s = float(suite["settle_hold_s"])
    hold_s = suite["hold_s"]
    hold_of_wave = [h["label"] for c in categories for h in c["holds"]]
    hold_s_map = (
        {lab: float(hold_s[lab]) for lab in hold_of_wave}
        if isinstance(hold_s, dict)
        else {lab: float(hold_s) for lab in hold_of_wave}
    )
    category_of_hold = {h["label"]: c["name"] for c in categories for h in c["holds"]}
    hand_of_hold = {h["label"]: h.get("hand") for c in categories for h in c["holds"]}

    own: dict[str, list[dict]] = {lab: [] for lab in hold_of_wave}
    for e in episodes:
        if "hold" not in e:
            raise ValueError(f"suite aggregate: episode {e.get('env_index')} carries no own 'hold'")
        if e["hold"] not in own:
            raise ValueError(f"suite aggregate: episode {e.get('env_index')} hold {e['hold']!r} is not a suite hold")
        own[e["hold"]].append(e)

    survival_by_hold: dict[str, float | None] = {}
    for lab, eps in own.items():
        end = int(round((prologue_s + settle_hold_s + hold_s_map[lab]) / dt))
        survival_by_hold[lab] = (
            float(sum(1 for e in eps if int(e["n_steps"]) >= end) / len(eps)) if eps else None
        )

    by_category: dict[str, dict] = {}
    for cat in categories:
        labels = [h["label"] for h in cat["holds"]]
        pooled = {key: [] for key in CATEGORY_METRIC_SOURCES}
        hands = sorted({hand_of_hold[lab] for lab in labels if hand_of_hold[lab] is not None})
        per_hand = {h: {key: [] for key in CATEGORY_METRIC_SOURCES} for h in hands}
        for lab in labels:
            for e in own[lab]:
                hold = e["holds"].get(lab) or {}
                for key in CATEGORY_METRIC_SOURCES:
                    value = _category_value(hold, key)
                    if value is None:
                        continue
                    pooled[key].append(value)
                    if hand_of_hold[lab] is not None:
                        per_hand[hand_of_hold[lab]][key].append(value)
        block: dict[str, Any] = {key: stats(vals) for key, vals in pooled.items()}
        block["by_hand"] = (
            {h: {key: stats(vals) for key, vals in slots.items()} for h, slots in per_hand.items()}
            if hands
            else None
        )
        by_category[cat["name"]] = block

    return {
        "waves": suite.get("waves"),
        "wave_seeds": suite.get("wave_seeds"),
        "episodes_total": suite.get("episodes_total") if suite.get("episodes_total") is not None else len(episodes),
        "dt": dt,
        "prologue_s": prologue_s,
        "settle_hold_s": settle_hold_s,
        "hold_s": hold_s,
        "hold_labels": [suite.get("settle_label") or "settle"] + hold_of_wave,
        "hold_of_wave": hold_of_wave,
        "category_of_hold": category_of_hold,
        "hand_of_hold": hand_of_hold,
        "n_by_hold": {lab: len(eps) for lab, eps in own.items()},
        "survival_by_hold": survival_by_hold,
        "trained_categories": suite.get("trained_categories"),
        "by_category": by_category,
    }


def aggregate(episodes: list[dict], *, suite: dict | None = None) -> dict:
    """Scenario-level roll-up.

    Headline is the **median** tripod score across episodes: a single fall-heavy episode should not
    move a gate. Episodes with no measurable window contribute to ``n_unscored``, not a zero.

    ``suite`` (category-suite rows only: :func:`suite_aggregate_arg` of the run's ``run_meta``)
    adds the ``suite`` block -- own-hold survival and the pooled ``by_category``; the pooled keys
    below are then means over every hold of the suite and are WATCH-only downstream. Without it
    (every other row) the output has no ``suite`` key at all.
    """
    scores = [e["tripod_score"] for e in episodes if e["tripod_score"] is not None]
    tips = [e["tippy_tap_fraction"] for e in episodes if e.get("tippy_tap_fraction") is not None]
    slips = [e["slip_ratio_mean"] for e in episodes if e.get("slip_ratio_mean") is not None]
    spin_ratios = [
        e["shaft_spin"]["one_direction_ratio_median"] for e in episodes if e.get("shaft_spin")
    ]
    spin_speeds = [
        e["shaft_spin"]["mean_abs_vel_median"] for e in episodes if e.get("shaft_spin")
    ]
    reasons: dict[str, int] = {}
    for e in episodes:
        reasons[e["termination_reason"]] = reasons.get(e["termination_reason"], 0) + 1

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"median": None, "mean": None, "p25": None, "p75": None, "n": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "n": int(arr.size),
        }

    # Per-hold breakdown: tripod quality is usually speed-dependent, and one number hides that.
    per_hold: dict[str, list[float]] = {}
    for e in episodes:
        for label, hold in e["holds"].items():
            score = hold["tripod"].get("tripod_score")
            if score is not None and not hold["tripod"].get("low_confidence"):
                per_hold.setdefault(label, []).append(score)

    # Velocity tracking (creep-audit, 2026-08-21): schedule completion is blind to a policy that
    # survives by abandoning the command, so achieved-vs-commanded is aggregated first-class.
    track_ratios = [e["tracking_ratio"] for e in episodes if e.get("tracking_ratio") is not None]
    track_by_hold: dict[str, dict[str, list[float]]] = {}
    for e in episodes:
        for label, hold in e["holds"].items():
            tr = hold.get("tracking")
            if not tr or tr["vx"]["actual_mean"] is None:
                continue
            slot = track_by_hold.setdefault(label, {"cmd": [], "achieved": [], "ratio": []})
            slot["cmd"].append(tr["vx"]["cmd_mean"])
            slot["achieved"].append(tr["vx"]["actual_mean"])
            if tr["vx"].get("ratio") is not None:
                slot["ratio"].append(tr["vx"]["ratio"])

    # PLAN G: fall-direction classes and support-polygon margins (pooled percentiles of the
    # per-episode p50s; the offline campaign scripts pool frames directly from raw npz).
    fall_classes: dict[str, int] = {}
    for e in episodes:
        fd = e.get("fall_direction")
        if fd:
            fall_classes[fd["fall_class"]] = fall_classes.get(fd["fall_class"], 0) + 1
    polygon: dict[str, dict] = {}
    for window in ("walking", "prefall"):
        for key in ("tip_angle_fwd_deg", "lead_contact_tip_deg", "fwd_ray_margin_m"):
            vals = [
                e["support_polygon"][window][key]["p50"]
                for e in episodes
                if e.get("support_polygon") and e["support_polygon"][window][key]["p50"] is not None
            ]
            polygon[f"{window}_{key}"] = _stats(vals)
        negs = [
            e["support_polygon"][window]["frac_neg_margin"]
            for e in episodes
            if e.get("support_polygon") and e["support_polygon"][window]["frac_neg_margin"] is not None
        ]
        polygon[f"{window}_frac_neg_margin"] = _stats(negs)

    orientation = {
        key: _stats([e["orientation"][key] for e in episodes
                     if e.get("orientation") and e["orientation"].get(key) is not None])
        for key in ("roll_rms", "pitch_rms", "roll_max_abs", "pitch_max_abs")
    }

    # Facing (omni rows only -- absent on body rows, whose output is unchanged). Whole-hold, so a
    # turn hold's number includes the turn itself.
    # ``yaw_progress_rate`` (the gated turning-rate quantity) and ``net_yaw_rate_toward`` are signed,
    # so their aggregates must be too: a hold where half the episodes turn the wrong way has to show
    # a median near zero, which magnitudes could never report (2026-09-17 yaw-smoke finding).
    # ``facing_closed`` / ``heading_error_initial_abs`` (2026-09-18) are additive: an aggregate
    # written before them simply has no such key, which is what the driver's proxy path keys off.
    # ``facing_closed`` is None on a window with nothing to close, so its ``n`` is the INFORMATIVE
    # episode count for that hold and ``heading_error_rms``'s ``n`` minus it is the uninformative
    # one (``metrics.facing_metrics``'s per-window ``n_uninformative``, summed).
    facing_keys = ("heading_error_rms", "heading_error_final_abs", "heading_error_initial_abs",
                   "heading_error_max_abs", "facing_closed",
                   "settle_time_s", "yaw_rate_ratio", "yaw_progress_rate", "net_yaw_rate_toward")
    facing_by_hold: dict[str, dict] = {}
    for e in episodes:
        for label, hold in e["holds"].items():
            fac = hold.get("facing")
            if not fac:
                continue
            slot = facing_by_hold.setdefault(label, {k: [] for k in facing_keys})
            for key in facing_keys:
                if fac.get(key) is not None:
                    slot[key].append(fac[key])
    facing_extra: dict[str, Any] = {}
    if facing_by_hold:
        facing_extra["facing_by_hold"] = {
            label: {key: _stats(vals) for key, vals in slots.items()}
            for label, slots in facing_by_hold.items()
        }
        facing_extra["facing_error_rms"] = _stats(
            [e["facing_error_rms"] for e in episodes if e.get("facing_error_rms") is not None]
        )

    # Motion / direction (plan D3, world rows only): ABSENT -- not empty -- when no episode carries
    # the entry, exactly as ``facing_by_hold`` is on body rows.
    motion_extra: dict[str, Any] = {}
    for block_key, out_key in (("motion", "motion_by_hold"), ("direction", "direction_by_hold")):
        by_hold: dict[str, dict[str, list[float]]] = {}
        for e in episodes:
            for label, hold in e["holds"].items():
                block = hold.get(block_key)
                if not block:
                    continue
                slot = by_hold.setdefault(label, {k: [] for k in block})
                for key, value in block.items():
                    if value is not None:
                        slot.setdefault(key, []).append(value)
        if by_hold:
            motion_extra[out_key] = {
                label: {key: _stats(vals) for key, vals in slots.items()} for label, slots in by_hold.items()
            }
    if suite is not None:
        motion_extra["suite"] = _suite_block(episodes, suite, _stats)

    n_eps = len(episodes)
    return {
        **facing_extra,
        **motion_extra,
        "n_episodes": n_eps,
        "fall_classes": fall_classes,
        "orientation": orientation,
        "support_polygon": polygon,
        "n_unscored_episodes": n_eps - len(scores),
        "tripod_score": _stats(scores),
        "tripod_score_by_hold": {k: _stats(v) for k, v in per_hold.items()},
        "tracking_ratio": _stats(track_ratios),
        "tracking_by_hold": {
            k: {
                "cmd_vx_mean": float(np.mean(v["cmd"])),
                "achieved_vx": _stats(v["achieved"]),
                "ratio": _stats(v["ratio"]),
            }
            for k, v in track_by_hold.items()
        },
        "tippy_tap_fraction": _stats(tips),
        "slip_ratio": _stats(slips),
        "shaft_one_direction_ratio": _stats(spin_ratios),
        "shaft_mean_abs_vel": _stats(spin_speeds),
        "termination_reasons": reasons,
        "schedule_completion_rate": (
            float(sum(1 for e in episodes if e["completed_schedule"]) / n_eps) if n_eps else None
        ),
    }


def _median_txt(stats: dict) -> str:
    """``stats['median']`` to 3 dp, or ``"n/a"`` -- "not measurable" must not print as 0.000."""
    value = stats.get("median")
    return "n/a" if value is None else f"{value:.3f}"


def summary_text(run_meta: dict, episodes: list[dict]) -> str:
    agg = aggregate(episodes, suite=suite_aggregate_arg(run_meta))
    lines = [
        "=== crab-hex gait eval ===",
        f"scenario   : {run_meta['scenario_id']}  ({run_meta['task']})",
        f"checkpoint : {run_meta.get('checkpoint') or '(zero actions)'}",
        f"episodes   : {agg['n_episodes']}  unscored: {agg['n_unscored_episodes']}",
        f"terrain    : {run_meta.get('terrain_source')}",
        f"cmd override max deviation: {run_meta.get('command_override_max_deviation'):.3e}",
        "",
        f"tripod_score      median={agg['tripod_score']['median']}  "
        f"p25={agg['tripod_score']['p25']}  p75={agg['tripod_score']['p75']}",
        f"tippy_tap_fraction median={agg['tippy_tap_fraction']['median']}",
        f"slip_ratio         median={agg['slip_ratio']['median']}",
        f"tracking_ratio     median={agg['tracking_ratio']['median']}  "
        f"(achieved/commanded vx, walking holds; n={agg['tracking_ratio']['n']})",
        f"schedule_completion_rate={agg['schedule_completion_rate']}",
        f"terminations={agg['termination_reasons']}",
    ]
    if agg.get("facing_error_rms"):
        lines.append(
            f"facing_error_rms   median={agg['facing_error_rms']['median']}  "
            f"(rad, whole episode; n={agg['facing_error_rms']['n']})"
        )
    if agg["tripod_score_by_hold"] or agg["tracking_by_hold"]:
        lines.append("")
        lines.append("by hold:")
        labels = dict.fromkeys(list(agg["tracking_by_hold"]) + list(agg["tripod_score_by_hold"]))
        for label in labels:
            tri = agg["tripod_score_by_hold"].get(label)
            trk = agg["tracking_by_hold"].get(label)
            tri_txt = f"tripod median={tri['median']} (n={tri['n']})" if tri else "tripod n/a"
            trk_txt = (
                f"cmd {trk['cmd_vx_mean']:.2f} -> achieved {trk['achieved_vx']['median']:.3f} m/s"
                if trk
                else "tracking n/a"
            )
            lines.append(f"  {label:>8}: {trk_txt} | {tri_txt}")
    if agg.get("facing_by_hold"):
        lines.append("")
        lines.append("facing by hold (whole hold; err rad, settle s, yaw rad/s: progress = "
                     "wz*sign(wz_cmd), net = net rotation toward the target; closed = fraction of "
                     "the initial error closed, n/a when there was none; ratio is wobble-sensitive):")
        for label, fac in agg["facing_by_hold"].items():
            lines.append(
                f"  {label:>8}: err rms={_median_txt(fac['heading_error_rms'])} "
                f"initial={_median_txt(fac.get('heading_error_initial_abs') or {})} "
                f"final={_median_txt(fac['heading_error_final_abs'])} "
                f"max={_median_txt(fac['heading_error_max_abs'])} "
                f"closed={_median_txt(fac.get('facing_closed') or {})} | "
                f"settle={_median_txt(fac['settle_time_s'])} "
                f"(n={fac['settle_time_s']['n']}/{fac['heading_error_rms']['n']}) | "
                f"progress={_median_txt(fac['yaw_progress_rate'])} "
                f"net={_median_txt(fac['net_yaw_rate_toward'])} "
                f"yaw_rate_ratio={_median_txt(fac['yaw_rate_ratio'])}"
            )
    if agg.get("suite"):
        sb = agg["suite"]
        trained = sb.get("trained_categories")
        lines.append("")
        lines.append(
            f"by category ({sb['waves']} waves, {sb['episodes_total']} episodes, one own hold each; "
            "surv = own-hold survival; medians pooled over both hands; "
            f"trained: {', '.join(trained) if trained is not None else 'n/a'}):"
        )
        for name, cat in sb["by_category"].items():
            labels = [lab for lab in sb["hold_of_wave"] if sb["category_of_hold"][lab] == name]
            n = sum(sb["n_by_hold"][lab] for lab in labels)
            surv = (
                sum(sb["survival_by_hold"][lab] * sb["n_by_hold"][lab] for lab in labels if sb["n_by_hold"][lab]) / n
                if n
                else None
            )
            surv_txt = "n/a" if surv is None else f"{surv:.3f}"
            per_hold = " ".join(
                f"{lab}={'n/a' if sb['survival_by_hold'][lab] is None else format(sb['survival_by_hold'][lab], '.2f')}"
                for lab in labels
            )
            lines.append(
                f"  {name:>9}: n={n} surv={surv_txt} ({per_hold}) | "
                f"ratio={_median_txt(cat['ratio'])} achieved={_median_txt(cat['achieved_speed'])} "
                f"tripod={_median_txt(cat['tripod'])} | closed={_median_txt(cat['facing_closed'])} "
                f"progress={_median_txt(cat['yaw_progress'])} net={_median_txt(cat['net_yaw'])} | "
                f"drift={_median_txt(cat['drift_speed'])} dir_err={_median_txt(cat['direction_error'])}"
            )
        # Motion quality: only the keys defined on each category (the rest are N/A by design).
        lines.append("")
        lines.append("motion quality by category (own hold, whole hold; medians; loaded = foot force > "
                     f"{M.MOTION_TOUCH_REF_N:g} N; per-m / per-rad over net displacement / |net yaw|):")
        for name, cat in sb["by_category"].items():
            parts = [f"{key}={_median_txt(cat[key])}" for key in M.MOTION_QUALITY_KEYS
                     if name in M.MOTION_QUALITY_APPLIES[key] and key in cat]
            lines.append(f"  {name:>9}: " + " ".join(parts))
    for key in ("command_override_warning", "schedule_state_warning", "obs_dim_warning"):
        if run_meta.get(key):
            lines.append(f"\n[WARN] {run_meta[key]}")
    return "\n".join(lines)


def write_run(
    run_dir: Path,
    *,
    run_meta: dict,
    episode_metrics: list[dict],
    raw: dict[str, np.ndarray] | None,
    scenario: Scenario,
    compiled: CompiledSchedule | None = None,
    history_root: Path | None = None,
) -> None:
    """Write a run's JSON, summary, raw npz and history line.

    On a category-suite row (``run_meta`` carries ``suite``) the aggregate gains its ``suite``
    block and ``complete.json`` (``episodes_total``, the checkpoint sha256) is written as the very
    LAST file, after every raw npz: a stamp without it is partial and no reader may use it (plan
    D10's freshness marker). Every other row writes exactly what it always did.
    """
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    suite = suite_aggregate_arg(run_meta)
    agg = aggregate(episode_metrics, suite=suite)

    (run_dir / "run_meta.json").write_text(json.dumps(M.json_safe(run_meta), indent=2, sort_keys=True))
    for ep in episode_metrics:
        path = run_dir / "metrics" / f"episode_{ep['env_index']:02d}.json"
        path.write_text(json.dumps(M.json_safe(ep), indent=2, sort_keys=True))
    (run_dir / "scenario_metrics.json").write_text(
        json.dumps(
            M.json_safe(
                {
                    "scenario_id": scenario.id,
                    "aggregate": agg,
                    "run_meta": run_meta,
                    "episodes": [
                        {
                            "env_index": e["env_index"],
                            "tripod_score": e["tripod_score"],
                            "tracking_ratio": e["tracking_ratio"],
                            "tippy_tap_fraction": e["tippy_tap_fraction"],
                            "slip_ratio_mean": e["slip_ratio_mean"],
                            "termination_reason": e["termination_reason"],
                            "n_steps": e["n_steps"],
                            # omni rows only; absent (not null) on body rows.
                            **({"facing_error_rms": e["facing_error_rms"]} if "facing_error_rms" in e else {}),
                        }
                        for e in episode_metrics
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    (run_dir / "summary.md").write_text(
        "```\n" + summary_text(run_meta, episode_metrics) + "\n```\n"
    )

    if raw is not None:
        # Raw force norms (not just booleans) are saved so the contact threshold can be re-swept
        # offline without another GPU run -- the codebase is inconsistent about it (sensor cfg 1.0,
        # reward_foot_clearance 0.1).
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # One raw file per EPISODE: W*E on a suite row, num_envs everywhere else. Never called
        # ``n_env`` -- that is the distinct-env count the tools' floor band reads.
        n_raw = len(run_meta["n_steps_per_env"])
        for env_idx in range(n_raw):
            n = int(run_meta["n_steps_per_env"][env_idx])
            payload = {k: np.asarray(v)[:n, env_idx] for k, v in raw.items()}
            if compiled is not None:
                payload["segment_id"] = compiled.segment_id[:n, env_idx]
                payload["steady_mask"] = compiled.steady_mask[:n, env_idx]
            payload["dt"] = np.asarray(run_meta["dt"], dtype=np.float64)
            np.savez_compressed(raw_dir / f"episode_{env_idx:02d}.npz", **payload)

    if history_root is not None:
        history_root = Path(history_root)
        history_root.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario.id,
            "checkpoint": run_meta.get("checkpoint"),
            "checkpoint_sha256": run_meta.get("checkpoint_sha256"),
            "run_dir": str(run_dir),
            "aggregate": agg,
        }
        with (history_root / f"gait_eval_{scenario.id}.jsonl").open("a") as fh:
            fh.write(json.dumps(M.json_safe(record)) + "\n")

    if suite is not None:
        # LAST write of the run: its presence says every file above is complete.
        (run_dir / "complete.json").write_text(
            json.dumps(
                M.json_safe(
                    {
                        "episodes_total": run_meta.get("episodes_total"),
                        "checkpoint_sha256": run_meta.get("checkpoint_sha256"),
                        "raw_written": raw is not None,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
