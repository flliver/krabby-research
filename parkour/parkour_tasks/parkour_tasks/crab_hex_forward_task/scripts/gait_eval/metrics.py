# SPDX-License-Identifier: BSD-3-Clause
"""Derived gait metrics for the crab-hex eval harness (Milestone 18, Task 0).

Pure ``numpy`` -- deliberately imports no ``torch``, ``isaaclab``, or ``parkour_tasks``, so the
whole scoring layer is unit-testable without booting Isaac Sim (see
``tests/unit/test_crab_hex_gait_metrics.py``). ``eval_crab_hex_gait.py`` does the simulating and
hands plain arrays to this module.

The headline number is :func:`tripod_window_metrics`'s ``tripod_score`` -- later milestone tasks
gate on it, so its edge cases (zero-variance correlation, feet that never lift, windows too short
to be meaningful) are handled explicitly rather than left to numpy's NaN semantics. A NaN that
reaches a gate is worse than a refusal to score.

Foot order is fixed by ``CRAB_HEX_FOOTPAD_BODY_NAMES`` in
``crab_hex_forward_task/mdp/crab_contact_sensors.py``; :data:`FOOT_ORDER` mirrors it here so this
module stays dependency-free, and the harness asserts the two agree at startup.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

FOOT_ORDER: tuple[str, ...] = (
    "FL_Footpad",
    "FR_Footpad",
    "ML_Footpad",
    "MR_Footpad",
    "RL_Footpad",
    "RR_Footpad",
)
"""Must equal ``CRAB_HEX_FOOTPAD_BODY_NAMES``; asserted by the harness at startup."""

TRIPOD_A: tuple[str, ...] = ("FL_Footpad", "MR_Footpad", "RL_Footpad")
TRIPOD_B: tuple[str, ...] = ("FR_Footpad", "ML_Footpad", "RR_Footpad")

TRIPOD_A_IDX: tuple[int, ...] = tuple(FOOT_ORDER.index(n) for n in TRIPOD_A)
TRIPOD_B_IDX: tuple[int, ...] = tuple(FOOT_ORDER.index(n) for n in TRIPOD_B)

# ``duty >= _DUTY_PLANTED`` means the foot effectively never leaves the ground; ``<= _DUTY_AIRBORNE``
# means it never lands. Both are reported as flags, not errors -- they are real (bad) gaits.
_DUTY_PLANTED = 0.99
_DUTY_AIRBORNE = 0.01
_VAR_EPS = 1e-12


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python and non-finite floats to ``None``.

    ``json.dump`` emits bare ``NaN``/``Infinity`` tokens, which are not valid JSON; anything
    downstream that parses a gate threshold would either explode or silently compare against a
    value that is never greater or less than anything. Convert once, here.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D boolean array as ``[start, end)`` index pairs."""
    mask = np.asarray(mask, dtype=bool).ravel()
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2])]


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when either series is constant.

    Constant input is not an error here -- it is the signature of a robot standing still (all six
    feet planted every step). Returning ``None`` rather than NaN forces callers to make an explicit
    decision; :func:`tripod_window_metrics` treats it as "no alternation" and scores 0.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.size != b.size:
        return None
    if a.var() < _VAR_EPS or b.var() < _VAR_EPS:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if math.isfinite(r) else None


def debounce(mask: np.ndarray, min_true: int = 1, min_false: int = 1) -> np.ndarray:
    """Remove runs shorter than the given minimums (``True`` runs first, then ``False``).

    Used **only** for stride/swing event extraction, where single-step contact chatter would
    otherwise manufacture thousands of ~0 m strides. Never applied to duty factor, air time, or the
    tripod score -- debouncing there would erase the very micro-stepping signature the harness
    exists to detect.
    """
    out = np.asarray(mask, dtype=bool).copy()
    if min_true > 1:
        for start, end in contiguous_runs(out):
            if end - start < min_true:
                out[start:end] = False
    if min_false > 1:
        for start, end in contiguous_runs(~out):
            if end - start < min_false:
                out[start:end] = True
    return out


def _percentiles(values: np.ndarray, keys: Sequence[int] = (10, 25, 50, 75, 90, 95)) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {f"p{k}": None for k in keys}
    return {f"p{k}": float(np.percentile(values, k)) for k in keys}


def _dist_summary(values: Sequence[float] | np.ndarray) -> dict:
    values = np.asarray(list(values), dtype=np.float64).ravel()
    out: dict[str, Any] = {"count": int(values.size)}
    if values.size == 0:
        out.update({"mean": None, "min": None, "max": None})
        out.update(_percentiles(values))
        return out
    out.update(
        {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    )
    out.update(_percentiles(values))
    return out


def yaw_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Yaw (rad) from ``(w, x, y, z)`` quaternions, shape ``(..., 4)`` -> ``(...)``.

    Isaac Lab stores root orientation as ``(w, x, y, z)`` (``root_link_quat_w``), not ``(x,y,z,w)``.
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def roll_pitch_from_quat_wxyz(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll/pitch (rad) from ``(w, x, y, z)`` quaternions; matches ``euler_xyz_from_quat`` usage."""
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    return wrap_to_pi(roll), wrap_to_pi(pitch)


# --------------------------------------------------------------------------------------
# tripod phasing -- the gateable score
# --------------------------------------------------------------------------------------
def tripod_window_metrics(
    contact: np.ndarray,
    *,
    dt: float,
    min_window_steps: int = 50,
    min_cycles: int = 2,
) -> dict:
    """Tripod-phasing metrics for one steady-state window. ``contact`` is ``(T, 6)`` bool.

    ``tripod_score = 0.5 * (coh_A + coh_B) * max(0, -corr(a, b))`` in ``[0, 1]``, where ``a``/``b``
    are the per-step stance counts of tripod sets A and B. A real tripod alternates the sets, so the
    two counts anti-correlate; a shuffle or a statue does not.

    Returns ``tripod_score = None`` (never 0.0) when the window is unusable, so callers can tell
    "not measured" apart from "measured, and bad".
    """
    contact = np.asarray(contact, dtype=bool)
    n_steps = int(contact.shape[0])
    out: dict[str, Any] = {
        "n_steps": n_steps,
        "duration_s": float(n_steps * dt),
        "valid": False,
        "discard_reason": None,
        "tripod_score": None,
        "corr_A_B": None,
        "degenerate_anti_phase": False,
        "low_confidence": False,
    }

    if n_steps < min_window_steps:
        # A 3-step window can trivially produce corr == -1. Refuse rather than flatter it.
        out["discard_reason"] = f"window_too_short(<{min_window_steps} steps)"
        return out

    duty = contact.mean(axis=0)
    out["duty_factor"] = {name: float(duty[i]) for i, name in enumerate(FOOT_ORDER)}
    out["foot_never_lifts"] = [FOOT_ORDER[i] for i in range(len(FOOT_ORDER)) if duty[i] >= _DUTY_PLANTED]
    out["foot_never_lands"] = [FOOT_ORDER[i] for i in range(len(FOOT_ORDER)) if duty[i] <= _DUTY_AIRBORNE]

    set_a = contact[:, TRIPOD_A_IDX]
    set_b = contact[:, TRIPOD_B_IDX]
    # within-set coherence == fraction of steps where all three feet of the set agree
    coh_a = float(np.mean((set_a.sum(axis=1) == 0) | (set_a.sum(axis=1) == 3)))
    coh_b = float(np.mean((set_b.sum(axis=1) == 0) | (set_b.sum(axis=1) == 3)))
    out["coherence_A"] = coh_a
    out["coherence_B"] = coh_b

    a = set_a.sum(axis=1).astype(np.float64)
    b = set_b.sum(axis=1).astype(np.float64)
    r = pearson(a, b)
    if r is None:
        # Constant stance count on at least one set: no alternation exists to measure. The statue
        # case lands here with coh_A == coh_B == 1.0, which without this branch would score 1.0*NaN.
        out["degenerate_anti_phase"] = True
        out["corr_A_B"] = None
        anti = 0.0
    else:
        out["corr_A_B"] = r
        anti = max(0.0, -r)
    out["anti_phase"] = anti

    # Rising edges of set A's stance count == gait cycles observed in this window.
    n_cycles = int(np.count_nonzero(np.diff(a) > 0))
    out["n_cycles"] = n_cycles
    if n_cycles < min_cycles and not out["degenerate_anti_phase"]:
        # Half a cycle can look like perfect anti-correlation off a single lift, so too few cycles
        # means "don't trust this estimate". A *degenerate* window is different: zero variance is a
        # confident 0.0 (the robot is not stepping at all), and marking it low-confidence would
        # exclude it from the aggregate and let a frozen policy report "unscored" instead of failing.
        out["low_confidence"] = True

    out["tripod_score"] = 0.5 * (coh_a + coh_b) * anti
    out["valid"] = True

    # Diagnostics -- reported, never gated. Make a suspicious score explicable without re-running.
    both_full = float(np.mean((a == 3) & (b == 3)))
    perfect_alt = float(np.mean(((a == 3) & (b == 0)) | ((a == 0) & (b == 3))))
    out["frac_both_sets_full_stance"] = both_full
    out["frac_perfect_alternation"] = perfect_alt
    return out


# --------------------------------------------------------------------------------------
# air time / stride / clearance / slip
# --------------------------------------------------------------------------------------
def air_time_metrics(contact: np.ndarray, *, dt: float) -> dict:
    """Air-time distribution per foot and pooled. ``contact`` is ``(T, 6)`` bool. No debouncing.

    Reported as a distribution, not a mean: the tippy-tap signature is *mass sitting at 2-3 steps*
    (~40-60 ms), which a mean happily hides. Histogram bins are pinned to integer step counts
    because air time is quantized to ``dt`` -- arbitrary bin edges invent structure that isn't there.
    """
    contact = np.asarray(contact, dtype=bool)
    n_steps = int(contact.shape[0])
    per_foot: dict[str, Any] = {}
    pooled_steps: list[int] = []
    censored = 0

    for i, name in enumerate(FOOT_ORDER):
        runs = contiguous_runs(~contact[:, i])
        lengths = []
        for start, end in runs:
            if start == 0 or end == n_steps:
                # Open at a window edge: true duration unknown, would bias the distribution short.
                censored += 1
                continue
            lengths.append(end - start)
        pooled_steps.extend(lengths)
        secs = np.asarray(lengths, dtype=np.float64) * dt
        per_foot[name] = _dist_summary(secs)
        per_foot[name]["count_steps"] = [int(v) for v in lengths]

    pooled = np.asarray(pooled_steps, dtype=np.float64)
    hist_max = int(pooled.max()) if pooled.size else 0
    # Bin edges at half-integers so bin k holds exactly the runs of length k steps.
    edges = np.arange(0.5, hist_max + 1.5, 1.0) if hist_max >= 1 else np.asarray([0.5, 1.5])
    counts, _ = np.histogram(pooled, bins=edges)

    out: dict[str, Any] = {
        "per_foot": per_foot,
        "pooled": _dist_summary(pooled * dt),
        "censored_intervals": censored,
        "histogram_step_counts": {int(k + 1): int(v) for k, v in enumerate(counts)},
        "dt": float(dt),
    }
    # 3 steps at 50 Hz == 60 ms; the flagged 0.05 s air-time reward threshold sits at 2.5 steps.
    n_short = int(np.count_nonzero(pooled <= 3)) if pooled.size else 0
    out["tippy_tap_fraction"] = float(n_short / pooled.size) if pooled.size else None
    out["n_intervals"] = int(pooled.size)
    return out


def _touchdowns(contact_foot: np.ndarray) -> np.ndarray:
    """Indices where a foot transitions air -> contact."""
    c = np.asarray(contact_foot, dtype=bool).astype(np.int8)
    return np.flatnonzero(np.diff(c) > 0) + 1


def stride_metrics(
    contact: np.ndarray,
    foot_pos_w: np.ndarray,
    *,
    dt: float,
    cmd_xy: np.ndarray | None = None,
    root_yaw: np.ndarray | None = None,
    min_contact_steps: int = 1,
    min_air_steps: int = 1,
) -> dict:
    """Touchdown-to-touchdown horizontal displacement per foot.

    ``foot_pos_w`` is ``(T, 6, 3)`` world-frame. The env-origin offset cancels in the difference,
    so no origin correction is needed. When ``cmd_xy`` and ``root_yaw`` are given, the component
    along the commanded direction is also reported -- that is the number that stays meaningful when
    Task 2 introduces lateral commands.
    """
    contact = np.asarray(contact, dtype=bool)
    foot_pos_w = np.asarray(foot_pos_w, dtype=np.float64)
    per_foot: dict[str, Any] = {}
    pooled_mag: list[float] = []
    pooled_along: list[float] = []
    raw_counts, deb_counts = 0, 0

    for i, name in enumerate(FOOT_ORDER):
        raw_counts += len(_touchdowns(contact[:, i]))
        c = debounce(contact[:, i], min_true=min_contact_steps, min_false=min_air_steps)
        tds = _touchdowns(c)
        deb_counts += len(tds)
        mags, alongs = [], []
        for k in range(len(tds) - 1):
            t0, t1 = int(tds[k]), int(tds[k + 1])
            delta = foot_pos_w[t1, i, :2] - foot_pos_w[t0, i, :2]
            mags.append(float(np.linalg.norm(delta)))
            if cmd_xy is not None and root_yaw is not None:
                cmd = np.asarray(cmd_xy, dtype=np.float64)[t0]
                if np.linalg.norm(cmd) > 1e-6:
                    yaw = float(np.asarray(root_yaw, dtype=np.float64)[t0])
                    # Rotate the world-frame step into the base frame, then project on the command.
                    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
                    dx_b = cos_y * delta[0] - sin_y * delta[1]
                    dy_b = sin_y * delta[0] + cos_y * delta[1]
                    unit = cmd / np.linalg.norm(cmd)
                    alongs.append(float(dx_b * unit[0] + dy_b * unit[1]))
        pooled_mag.extend(mags)
        pooled_along.extend(alongs)
        per_foot[name] = {
            "magnitude": _dist_summary(mags),
            "along_command": _dist_summary(alongs),
            "n_strides": len(mags),
        }

    return {
        "per_foot": per_foot,
        "pooled_magnitude": _dist_summary(pooled_mag),
        "pooled_along_command": _dist_summary(pooled_along),
        "n_touchdowns_raw": raw_counts,
        "n_touchdowns_debounced": deb_counts,
        "debounce": {"min_contact_steps": min_contact_steps, "min_air_steps": min_air_steps},
    }


def swing_clearance_metrics(
    contact: np.ndarray,
    foot_pos_w: np.ndarray,
    terrain_z: np.ndarray,
    *,
    stance_z_offset: float | None = None,
) -> dict:
    """Footpad height above terrain during each swing interval. ``terrain_z`` is ``(T, 6)``.

    ``stance_z_offset`` compensates for the footpad body origin sitting above the contact surface.
    Self-calibrated from the run's own stance samples when not supplied, so it survives changes to
    the pad's collision geometry.
    """
    contact = np.asarray(contact, dtype=bool)
    foot_z = np.asarray(foot_pos_w, dtype=np.float64)[:, :, 2]
    terrain_z = np.asarray(terrain_z, dtype=np.float64)
    height = foot_z - terrain_z

    if stance_z_offset is None:
        stance_vals = height[contact]
        stance_z_offset = float(np.median(stance_vals)) if stance_vals.size else 0.0
    clearance = height - stance_z_offset

    per_foot: dict[str, Any] = {}
    pooled_max, pooled_min_int = [], []
    for i, name in enumerate(FOOT_ORDER):
        maxes, mins, mins_int = [], [], []
        for start, end in contiguous_runs(~contact[:, i]):
            seg = clearance[start:end, i]
            if seg.size == 0:
                continue
            maxes.append(float(seg.max()))
            mins.append(float(seg.min()))
            # Endpoints are ~ground by construction (the foot just left / is about to land), so the
            # spec's min-over-swing is ~0 for any gait. The interior minimum is what detects drag.
            if seg.size > 2:
                mins_int.append(float(seg[1:-1].min()))
        pooled_max.extend(maxes)
        pooled_min_int.extend(mins_int)
        per_foot[name] = {
            "max_over_swing": _dist_summary(maxes),
            "min_over_swing": _dist_summary(mins),
            "min_over_swing_interior": _dist_summary(mins_int),
            "n_swings": len(maxes),
        }

    return {
        "per_foot": per_foot,
        "pooled_max_over_swing": _dist_summary(pooled_max),
        "pooled_min_over_swing_interior": _dist_summary(pooled_min_int),
        "stance_z_offset_m": float(stance_z_offset),
    }


def slip_metrics(
    contact: np.ndarray,
    foot_pos_w: np.ndarray,
    foot_lin_vel_w: np.ndarray,
    *,
    dt: float,
    foot_force_norm: np.ndarray | None = None,
    foot_ang_vel_w: np.ndarray | None = None,
    pad_radius_m: float = 0.03,
    slip_speed_threshold: float = 0.02,
    min_contact_steps: int = 1,
    min_air_steps: int = 1,
) -> dict:
    """Stance stability: how much a foot slides while it is nominally planted.

    Sim terrain is rigid and static, so a properly planted foot has ~zero **world-frame** horizontal
    velocity -- slip is measured directly, with no frame correction. Nothing else in the metric set
    catches skating: a foot that never lifts registers no air time and cannot break tripod
    alternation, so a shuffling policy can score well on every other number here.

    Computed on each stance interval's **interior** (touchdown/liftoff steps dropped -- the foot is
    still decelerating into, or accelerating out of, the ground there).

    ``slip_ratio`` (slip path length / the stride that follows the same touchdown) is the headline:
    dimensionless, so it compares across commanded speeds. ~0 is planted; ~1 means the foot slid as
    far as the stride advanced.
    """
    contact = np.asarray(contact, dtype=bool)
    foot_pos_w = np.asarray(foot_pos_w, dtype=np.float64)
    vel = np.asarray(foot_lin_vel_w, dtype=np.float64)[:, :, :2]
    speed = np.linalg.norm(vel, axis=-1)  # (T, 6)

    # Rotation artifact bound: body_lin_vel_w is the pad *origin*'s velocity, so a pad pivoting on a
    # planted contact point reads as slip. Bound it by |omega| * pad_radius rather than silently
    # correcting -- if the corrected and raw numbers disagree a lot, that is worth knowing.
    speed_corr = None
    if foot_ang_vel_w is not None:
        omega = np.linalg.norm(np.asarray(foot_ang_vel_w, dtype=np.float64), axis=-1)
        speed_corr = np.clip(speed - omega * pad_radius_m, 0.0, None)

    per_foot: dict[str, Any] = {}
    pooled_ratio, pooled_path, pooled_speed_mean = [], [], []
    n_intervals_skipped = 0

    for i, name in enumerate(FOOT_ORDER):
        c = debounce(contact[:, i], min_true=min_contact_steps, min_false=min_air_steps)
        stance_runs = contiguous_runs(c)
        tds = _touchdowns(c)
        # stride following touchdown k, keyed by touchdown index
        stride_by_td: dict[int, float] = {}
        for k in range(len(tds) - 1):
            t0, t1 = int(tds[k]), int(tds[k + 1])
            stride_by_td[t0] = float(np.linalg.norm(foot_pos_w[t1, i, :2] - foot_pos_w[t0, i, :2]))

        path_lens, net_disps, ratios, mean_speeds, max_speeds = [], [], [], [], []
        frac_above, work_proxy = [], []
        p95_speeds = []
        for start, end in stance_runs:
            lo, hi = start + 1, end - 1  # interior
            if hi - lo < 1:
                n_intervals_skipped += 1
                continue
            seg_xy = foot_pos_w[lo:hi, i, :2]
            step_disp = np.linalg.norm(np.diff(seg_xy, axis=0), axis=-1) if seg_xy.shape[0] > 1 else np.zeros(0)
            path = float(step_disp.sum())
            net = float(np.linalg.norm(seg_xy[-1] - seg_xy[0])) if seg_xy.shape[0] > 1 else 0.0
            seg_speed = speed[lo:hi, i]
            path_lens.append(path)
            net_disps.append(net)
            mean_speeds.append(float(seg_speed.mean()))
            max_speeds.append(float(seg_speed.max()))
            p95_speeds.append(float(np.percentile(seg_speed, 95)))
            frac_above.append(float(np.mean(seg_speed > slip_speed_threshold)))
            if foot_force_norm is not None:
                f = np.asarray(foot_force_norm, dtype=np.float64)[lo:hi, i]
                work_proxy.append(float(np.sum(seg_speed * f) * dt))
            stride = stride_by_td.get(start)
            if stride is not None and stride > 1e-6:
                ratios.append(path / stride)

        duty = float(contact[:, i].mean())
        n_steps = int(contact.shape[0])
        per_foot[name] = {
            "slip_path_len": _dist_summary(path_lens),
            "slip_net_disp": _dist_summary(net_disps),
            "slip_speed_mean": _dist_summary(mean_speeds),
            "slip_speed_p95": _dist_summary(p95_speeds),
            "slip_speed_max": _dist_summary(max_speeds),
            "slip_fraction": _dist_summary(frac_above),
            "slip_ratio": _dist_summary(ratios),
            "slip_work_proxy": _dist_summary(work_proxy) if foot_force_norm is not None else None,
            "n_stance_intervals": len(path_lens),
            # No stride is defined when the foot never lifts, so slip_ratio is null there by
            # construction; this keeps the pathology visible instead of silently dropping it.
            "slip_path_len_per_s": (
                float(np.sum(path_lens) / (n_steps * dt)) if n_steps > 0 else None
            ),
            "duty_factor": duty,
            "never_lifts": duty >= _DUTY_PLANTED,
        }
        pooled_ratio.extend(ratios)
        pooled_path.extend(path_lens)
        pooled_speed_mean.extend(mean_speeds)

    out: dict[str, Any] = {
        "per_foot": per_foot,
        "pooled_slip_ratio": _dist_summary(pooled_ratio),
        "pooled_slip_path_len": _dist_summary(pooled_path),
        "pooled_slip_speed_mean": _dist_summary(pooled_speed_mean),
        "slip_speed_threshold": float(slip_speed_threshold),
        "intervals_too_short_for_interior": n_intervals_skipped,
        "pad_radius_m": float(pad_radius_m),
    }
    if speed_corr is not None:
        stance_mask = contact
        raw_mean = float(speed[stance_mask].mean()) if stance_mask.any() else None
        corr_mean = float(speed_corr[stance_mask].mean()) if stance_mask.any() else None
        out["omega_corrected"] = {
            "stance_speed_mean_raw": raw_mean,
            "stance_speed_mean_omega_corrected": corr_mean,
            "note": "raw uses pad-origin velocity; corrected subtracts |omega|*pad_radius as an "
            "upper bound on rotation-induced apparent slip. Large disagreement means the pad is "
            "pivoting, not sliding -- investigate before gating on slip.",
        }
    return out


# --------------------------------------------------------------------------------------
# tracking / orientation / actions
# --------------------------------------------------------------------------------------
def tracking_metrics(
    cmd: np.ndarray,
    root_lin_vel_b: np.ndarray,
    root_ang_vel_b: np.ndarray,
    *,
    ratio_min_cmd: float | None = None,
) -> dict:
    """Commanded minus actual base velocity, per axis. Both in the base frame.

    Signed mean is reported alongside mean-abs and RMS: mean-abs alone hides systematic undershoot,
    which is the failure mode the speed-forcing reward terms are suspected of causing.

    When ``ratio_min_cmd`` is given, the vx entry also carries ``ratio`` = actual_mean/cmd_mean --
    but only when ``|cmd_mean| > ratio_min_cmd`` (below the env's lin-vel clip the command is a
    stop by construction, so a ratio there is noise over ~zero). The 2026-08-21 creep-audit found
    policies at 0.92 schedule completion moving at 15-30% of command; completion alone cannot see
    that, this ratio can.
    """
    cmd = np.asarray(cmd, dtype=np.float64)
    lin = np.asarray(root_lin_vel_b, dtype=np.float64)
    ang = np.asarray(root_ang_vel_b, dtype=np.float64)
    actual = np.stack([lin[:, 0], lin[:, 1], ang[:, 2]], axis=-1)
    err = cmd - actual
    axes = ("vx", "vy", "wz")
    out: dict[str, Any] = {}
    for k, axis in enumerate(axes):
        e = err[:, k]
        out[axis] = {
            "signed_mean": float(e.mean()) if e.size else None,
            "mean_abs": float(np.abs(e).mean()) if e.size else None,
            "rms": float(np.sqrt(np.mean(e**2))) if e.size else None,
            "cmd_mean": float(cmd[:, k].mean()) if e.size else None,
            "actual_mean": float(actual[:, k].mean()) if e.size else None,
        }
    if ratio_min_cmd is not None:
        vx = out["vx"]
        if vx["cmd_mean"] is not None and abs(vx["cmd_mean"]) > ratio_min_cmd:
            vx["ratio"] = vx["actual_mean"] / vx["cmd_mean"]
        else:
            vx["ratio"] = None
    return out


def orientation_metrics(root_quat_w: np.ndarray) -> dict:
    roll, pitch = roll_pitch_from_quat_wxyz(root_quat_w)
    if roll.size == 0:
        return {"roll_rms": None, "pitch_rms": None, "roll_max_abs": None, "pitch_max_abs": None}
    return {
        "roll_rms": float(np.sqrt(np.mean(roll**2))),
        "pitch_rms": float(np.sqrt(np.mean(pitch**2))),
        "roll_max_abs": float(np.abs(roll).max()),
        "pitch_max_abs": float(np.abs(pitch).max()),
    }


def _sticky_sign_reversals(series: np.ndarray, deadzone: float) -> np.ndarray:
    """Count sign reversals per column with a sticky previous direction.

    In-deadzone samples do not reset the remembered direction -- matching
    ``PenaltyMotorDirectionReversal`` in ``parkour_isaaclab/envs/mdp/rewards.py``, so the harness's
    number is directly comparable to that reward term's magnitude in a training log.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, None]
    n_cols = series.shape[1]
    prev = np.zeros(n_cols)
    counts = np.zeros(n_cols, dtype=np.int64)
    for t in range(series.shape[0]):
        cur = np.sign(series[t]) * (np.abs(series[t]) > deadzone)
        counts += ((prev * cur) < 0).astype(np.int64)
        nonzero = cur != 0
        prev = np.where(nonzero, cur, prev)
    return counts


def action_metrics(
    actions: np.ndarray,
    *,
    joint_groups: dict[str, Sequence[int]] | None = None,
    action_deadzone: float = 0.01,
    joint_vel: np.ndarray | None = None,
    joint_vel_groups: dict[str, Sequence[int]] | None = None,
    joint_vel_deadzone: float = 0.05,
) -> dict:
    """Action smoothness and direction-reversal counts, reported **per joint group**.

    The cam mechanism makes a flat 18-DOF mean actively misleading: the CamShaft DOFs are meant to
    spin continuously in one direction (the cam converts that into the leg's back-and-forth yaw), so
    a CamShaft reversal is a different pathology from a Femur_Tibia reversal. Averaging across all
    18 washes that distinction out. ``joint_groups`` maps group name -> action-column indices, which
    the harness derives from the action term's own joint names (column order is articulation order,
    not regex order, so it must not be assumed).
    """
    actions = np.asarray(actions, dtype=np.float64)
    delta = np.diff(actions, axis=0) if actions.shape[0] > 1 else np.zeros((0, actions.shape[1]))
    out: dict[str, Any] = {
        "action_rate_mean_all": float(np.abs(delta).mean()) if delta.size else None,
        "n_steps": int(actions.shape[0]),
    }

    reversals = _sticky_sign_reversals(delta, action_deadzone) if delta.size else np.zeros(actions.shape[1], dtype=np.int64)
    out["action_sign_reversals_total"] = int(reversals.sum())

    if joint_groups:
        per_group: dict[str, Any] = {}
        for group, idx in joint_groups.items():
            idx = list(idx)
            if not idx:
                continue
            d = delta[:, idx] if delta.size else np.zeros((0, len(idx)))
            per_group[group] = {
                "action_rate_mean": float(np.abs(d).mean()) if d.size else None,
                "action_sign_reversals": int(reversals[idx].sum()),
                "action_sign_reversals_per_dof": {int(j): int(reversals[j]) for j in idx},
                "n_dofs": len(idx),
            }
        out["per_group"] = per_group

    if joint_vel is not None and joint_vel_groups:
        jv = np.asarray(joint_vel, dtype=np.float64)
        jv_rev = _sticky_sign_reversals(jv, joint_vel_deadzone)
        out["joint_vel_sign_reversals"] = {
            group: int(jv_rev[list(idx)].sum()) for group, idx in joint_vel_groups.items() if list(idx)
        }
        out["joint_vel_deadzone"] = float(joint_vel_deadzone)
    return out


def shaft_spin_metrics(
    joint_vel: np.ndarray,
    shaft_ids: Sequence[int],
    *,
    dt: float,
    deadzone: float = 0.05,
) -> dict:
    """Continuous-rotation metrics for the velocity-driven cam shafts (2026-08 velocity era).

    The quick-return linkage wants the motor spinning in ONE direction; the cam converts that
    into the leg's back-and-forth yaw. The primary gate number is ``one_direction_ratio`` =
    |mean(v)| / mean(|v|): 1.0 = pure one-direction spin, ~0 = symmetric oscillation
    (the position-era habit). Reversal counting reuses the sticky rule so numbers stay
    directly comparable to ``PenaltyMotorDirectionReversal`` in training logs.
    """
    jv = np.asarray(joint_vel, dtype=np.float64)[:, list(shaft_ids)]
    n_steps = jv.shape[0]
    duration_s = n_steps * dt
    signed_mean = jv.mean(axis=0)
    abs_mean = np.abs(jv).mean(axis=0)
    ratio = np.abs(signed_mean) / np.maximum(abs_mean, 1e-9)
    reversals = _sticky_sign_reversals(jv, deadzone)
    net_revolutions = jv.sum(axis=0) * dt / (2.0 * np.pi)
    return {
        "n_shafts": int(jv.shape[1]),
        "n_steps": int(n_steps),
        "duration_s": float(duration_s),
        "deadzone": float(deadzone),
        "signed_mean_vel": [float(v) for v in signed_mean],
        "mean_abs_vel": [float(v) for v in abs_mean],
        "one_direction_ratio": [float(v) for v in ratio],
        "one_direction_ratio_median": float(np.median(ratio)),
        "mean_abs_vel_median": float(np.median(abs_mean)),
        "reversals_per_s": [float(r / max(duration_s, 1e-9)) for r in reversals],
        "reversals_per_s_median": float(np.median(reversals) / max(duration_s, 1e-9)),
        "net_revolutions": [float(v) for v in net_revolutions],
    }


# --------------------------------------------------------------------------------------
# support polygon / static stability margin and fall direction (PLAN G leg-mount
# morphology, 2026-09-02). Pure numpy; consumed by the eval report and the offline
# kinematic screen.
# --------------------------------------------------------------------------------------
def _convex_hull_xy(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain. ``points`` (N, 2) -> hull vertices CCW (M, 2), M >= 3, else (N, 2)."""
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def _signed_min_edge_margin(hull: np.ndarray, c: np.ndarray) -> float:
    """Signed distance from point ``c`` to the nearest hull edge (positive inside, CCW hull)."""
    if len(hull) < 3:
        return float("nan")
    best = float("inf")
    inside = True
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        ln = float(np.hypot(*e))
        if ln < 1e-12:
            continue
        # left-of-edge is inside for a CCW hull
        s = (e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0])) / ln
        if s < 0:
            inside = False
        # unsigned distance to the segment
        t = float(np.clip(np.dot(c - a, e) / (ln * ln), 0.0, 1.0))
        dseg = float(np.hypot(*(c - (a + t * e))))
        best = min(best, dseg)
    return best if inside else -best


def _forward_ray_margin(hull: np.ndarray, c: np.ndarray) -> float:
    """Signed +x extent of the hull along the line y = c_y, measured from c_x.

    Positive when the CoM projection has polygon ahead of it; negative when the projection
    is ahead of the polygon (pure-pitch tipping quantity). NaN when the line misses the hull
    entirely (projection laterally outside).
    """
    if len(hull) < 3:
        return float("nan")
    xs = []
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        y0, y1 = a[1] - c[1], b[1] - c[1]
        if y0 == y1:
            if y0 == 0.0:
                xs.extend([a[0], b[0]])
            continue
        if (y0 <= 0.0 < y1) or (y1 <= 0.0 < y0):
            t = y0 / (y0 - y1)
            xs.append(a[0] + t * (b[0] - a[0]))
    if not xs:
        return float("nan")
    return float(max(xs) - c[0])


def support_polygon_metrics(
    foot_pos_w: np.ndarray,
    foot_force_norm: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    *,
    dt: float,
    crab_failure: np.ndarray | None = None,
    walking_mask: np.ndarray | None = None,
    com_pos_w: np.ndarray | None = None,
    support_threshold_n: float = 50.0,
    prefall_s: float = 1.0,
    exclude_last_s: float = 0.1,
    return_series: bool = False,
) -> dict:
    """Static-stability margins of the loaded-foot support polygon, in the heading frame.

    Per frame: the loaded set ``L = {i : force >= support_threshold_n}``; feet and the CoM
    projected into the heading frame (yaw-only rotation about the root, +x' = forward);
    ``fwd_ray_margin_m`` (primary, pure-pitch tipping quantity), ``min_edge_margin_m``
    (classic static stability margin), ``tip_angle_fwd_deg = atan2(fwd_ray_margin, h)``,
    ``lead_contact_tip_deg = atan2(max x' of L - c_x', h)`` (the historical definition),
    ``fore_aft_span_m`` and ``com_offset_x_m`` (CoM ahead of the loaded-foot centroid).
    ``h`` = CoM height above the mean loaded-foot z. The CoM is ``com_pos_w`` when given,
    else the root position (a proxy; the plant's measured longitudinal CoM offset is
    ~0 mm, so this errs only in height).

    Aggregates are p10/p50/p90 over ``walking_mask`` frames (all frames if None) and over
    the ``prefall_s`` window before each ``crab_failure`` (excluding the final
    ``exclude_last_s``), plus the fraction of frames with negative forward margin and the
    fraction with fewer than three loaded feet (``frac_underdetermined``).
    """
    foot_pos_w = np.asarray(foot_pos_w, dtype=np.float64)
    force = np.asarray(foot_force_norm, dtype=np.float64)
    root = np.asarray(root_pos_w, dtype=np.float64)
    T = foot_pos_w.shape[0]
    com = root if com_pos_w is None else np.asarray(com_pos_w, dtype=np.float64)
    yaw = yaw_from_quat_wxyz(root_quat_w)
    c, s = np.cos(-yaw), np.sin(-yaw)

    def to_heading(vec_xy):  # (T, K, 2) world-relative -> heading frame
        x = c[:, None] * vec_xy[..., 0] - s[:, None] * vec_xy[..., 1]
        y = s[:, None] * vec_xy[..., 0] + c[:, None] * vec_xy[..., 1]
        return np.stack([x, y], axis=-1)

    feet_h = to_heading(foot_pos_w[..., :2] - root[:, None, :2])
    com_h = to_heading((com[:, :2] - root[:, :2])[:, None, :])[:, 0, :]

    fwd = np.full(T, np.nan)
    edge = np.full(T, np.nan)
    lead = np.full(T, np.nan)
    span = np.full(T, np.nan)
    com_off = np.full(T, np.nan)
    height = np.full(T, np.nan)
    n_loaded = np.zeros(T, dtype=np.int64)
    for t in range(T):
        L = force[t] >= support_threshold_n
        n_loaded[t] = int(L.sum())
        if n_loaded[t] == 0:
            continue
        pts = feet_h[t, L]
        height[t] = com[t, 2] - foot_pos_w[t, L, 2].mean()
        span[t] = pts[:, 0].max() - pts[:, 0].min()
        com_off[t] = com_h[t, 0] - pts[:, 0].mean()
        lead[t] = pts[:, 0].max() - com_h[t, 0]
        if n_loaded[t] >= 3:
            hull = _convex_hull_xy(pts)
            fwd[t] = _forward_ray_margin(hull, com_h[t])
            edge[t] = _signed_min_edge_margin(hull, com_h[t])
    with np.errstate(invalid="ignore"):
        tip_fwd = np.degrees(np.arctan2(fwd, height))
        tip_lead = np.degrees(np.arctan2(lead, height))

    def _agg(mask: np.ndarray) -> dict:
        def pct(a):
            v = a[mask & np.isfinite(a)]
            if v.size == 0:
                return {"p10": None, "p50": None, "p90": None, "n": 0}
            return {"p10": float(np.percentile(v, 10)), "p50": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)), "n": int(v.size)}
        n_frames = int(mask.sum())
        return {
            "tip_angle_fwd_deg": pct(tip_fwd),
            "lead_contact_tip_deg": pct(tip_lead),
            "fwd_ray_margin_m": pct(fwd),
            "min_edge_margin_m": pct(edge),
            "fore_aft_span_m": pct(span),
            "com_offset_x_m": pct(com_off),
            "frac_neg_margin": (float(np.mean(fwd[mask & np.isfinite(fwd)] < 0.0))
                                if np.any(mask & np.isfinite(fwd)) else None),
            "frac_underdetermined": (float(np.mean(n_loaded[mask] < 3)) if n_frames else None),
            "n_frames": n_frames,
        }

    walking = np.ones(T, dtype=bool) if walking_mask is None else np.asarray(walking_mask, dtype=bool)
    prefall = np.zeros(T, dtype=bool)
    if crab_failure is not None:
        fails = np.flatnonzero(np.asarray(crab_failure, dtype=bool))
        if fails.size:
            t_fail = int(fails[0])
            lo = max(0, t_fail - int(round(prefall_s / dt)))
            hi = max(lo, t_fail - int(round(exclude_last_s / dt)))
            prefall[lo:hi] = True
    out = {"walking": _agg(walking), "prefall": _agg(prefall), "support_threshold_n": support_threshold_n}
    if return_series:
        out["series"] = {"tip_angle_fwd_deg": tip_fwd, "lead_contact_tip_deg": tip_lead,
                         "fwd_ray_margin_m": fwd, "min_edge_margin_m": edge, "n_loaded": n_loaded,
                         "height_m": height, "prefall_mask": prefall}
    return out


def fall_direction_metrics(
    root_quat_w: np.ndarray,
    crab_failure: np.ndarray,
    *,
    dt: float,
    root_ang_vel_b: np.ndarray | None = None,
    class_threshold_rad: float = 0.35,
    rate_window_s: float = 0.5,
) -> dict:
    """Classify a termination by attitude at the failure step.

    ``pitch_fwd`` (nose-down, pitch >= +threshold and |pitch| >= |roll|), ``pitch_back``,
    ``roll``, or ``none`` (no crab_failure in the trace). Positive pitch = nose-down, matching
    ``penalty_base_pitch_forward_linear``. Also reports time-to-fall and the peak pitch rate
    over the last ``rate_window_s`` when ``root_ang_vel_b`` is given.
    """
    fails = np.flatnonzero(np.asarray(crab_failure, dtype=bool))
    if fails.size == 0:
        return {"fall_class": "none", "t_fail_s": None, "pitch_at_fail": None,
                "roll_at_fail": None, "max_pitch_rate": None}
    t = int(fails[0])
    roll, pitch = roll_pitch_from_quat_wxyz(np.asarray(root_quat_w)[t])
    roll, pitch = float(roll), float(pitch)
    if abs(pitch) >= class_threshold_rad and abs(pitch) >= abs(roll):
        cls = "pitch_fwd" if pitch > 0 else "pitch_back"
    elif abs(roll) >= class_threshold_rad:
        cls = "roll"
    else:
        cls = "other"
    rate = None
    if root_ang_vel_b is not None:
        w = np.asarray(root_ang_vel_b, dtype=np.float64)
        lo = max(0, t - int(round(rate_window_s / dt)))
        seg = w[lo:t + 1, 1]
        rate = float(np.abs(seg).max()) if seg.size else None
    return {"fall_class": cls, "t_fail_s": t * dt, "pitch_at_fail": pitch,
            "roll_at_fail": roll, "max_pitch_rate": rate}
