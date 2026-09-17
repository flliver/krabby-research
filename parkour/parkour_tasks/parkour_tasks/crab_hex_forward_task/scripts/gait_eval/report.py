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
        if idx.size >= min_window_steps:
            cmd_window = raw["cmd_applied"][idx, env_idx, :]
            if world:
                # The body command turns with the body; its travel-frame shape is
                # (speed, 0, wz) -- the same Shape B the sideways rows use, so tracking_metrics
                # and the driver's ratio gate are unchanged.
                speed = np.linalg.norm(cmd_window[:, :2], axis=-1)
                cmd_window = np.stack([speed, np.zeros_like(speed), cmd_window[:, 2]], axis=-1)
            entry["tracking"] = M.tracking_metrics(
                cmd_window,
                M.lin_vel_to_travel_frame(
                    raw["root_lin_vel_b"][idx, env_idx, :], _travel_yaw_at(travel_yaw, idx)
                ),
                raw["root_ang_vel_b"][idx, env_idx, :],
                ratio_min_cmd=lin_vel_clip,
            )

        if has_facing:
            whole = _window_mask(
                segment_id=segment_id,
                steady=np.ones_like(steady),
                hold_idx=hold_idx,
                n_steps=n_steps,
                fall_trim_steps=fall_trim,
                ended_in_fall=ended_in_fall,
            )
            widx = np.flatnonzero(whole)
            entry["facing"] = M.facing_metrics(
                raw["heading_err"][widx, env_idx],
                raw["root_ang_vel_b"][widx, env_idx, 2],
                dt,
                wz_cmd_t=raw["cmd_applied"][widx, env_idx, 2],
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
    return out


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
        )
        for i in range(len(term_reason))
    ]


def aggregate(episodes: list[dict]) -> dict:
    """Scenario-level roll-up.

    Headline is the **median** tripod score across episodes: a single fall-heavy episode should not
    move a gate. Episodes with no measurable window contribute to ``n_unscored``, not a zero.
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
    facing_keys = ("heading_error_rms", "heading_error_final_abs", "heading_error_max_abs",
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

    n_eps = len(episodes)
    return {
        **facing_extra,
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
    agg = aggregate(episodes)
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
                     "wz*sign(wz_cmd), net = net rotation toward the target; ratio is wobble-sensitive):")
        for label, fac in agg["facing_by_hold"].items():
            lines.append(
                f"  {label:>8}: err rms={_median_txt(fac['heading_error_rms'])} "
                f"final={_median_txt(fac['heading_error_final_abs'])} "
                f"max={_median_txt(fac['heading_error_max_abs'])} | "
                f"settle={_median_txt(fac['settle_time_s'])} "
                f"(n={fac['settle_time_s']['n']}/{fac['heading_error_rms']['n']}) | "
                f"progress={_median_txt(fac['yaw_progress_rate'])} "
                f"net={_median_txt(fac['net_yaw_rate_toward'])} "
                f"yaw_rate_ratio={_median_txt(fac['yaw_rate_ratio'])}"
            )
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
    run_dir = Path(run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    agg = aggregate(episode_metrics)

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
        n_env = int(run_meta["num_envs"])
        for env_idx in range(n_env):
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
