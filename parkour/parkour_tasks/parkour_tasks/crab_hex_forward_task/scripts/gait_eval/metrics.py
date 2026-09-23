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
# travel frame (sideways-task harness sharing, 2026-09-11): every helper below returns the
# SAME object/array at travel_yaw == 0.0 (or skips the rotation entirely), so the forward task
# -- which always calls these with travel_yaw 0.0 -- stays byte-identical. This module stays
# torch-free by design (see the module docstring), so these duplicate --  in numpy -- the
# rotation convention pinned in ``parkour_isaaclab/envs/mdp/travel_frame.py``.
#
# Omni task (2026-09-16): ``travel_yaw`` may also be a ``(T,)`` array -- one angle per step, the
# direction of that step's command, since an omni policy's travel frame turns during the episode.
# The scalar branches below are TODAY'S expressions, character for character, and the identity
# fast path is still keyed on a scalar 0.0, so the forward and sideways rows are bit-identical.
# --------------------------------------------------------------------------------------
def _is_identity_yaw(travel_yaw) -> bool:
    """True only for a scalar 0.0 -- the identity fast path (never for an array, even all-zero)."""
    return np.ndim(travel_yaw) == 0 and float(travel_yaw) == 0.0


def _bcast_angle(angle, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """``(cos, sin)`` of a per-step angle array, shaped to broadcast against ``ndim``-D operands."""
    ang = np.asarray(angle, dtype=np.float64)
    c, s = np.cos(ang), np.sin(ang)
    extra = ndim - 1 - ang.ndim
    if extra > 0:
        c = c.reshape(c.shape + (1,) * extra)
        s = s.reshape(s.shape + (1,) * extra)
    return c, s


def _rotate_xy(xy: np.ndarray, angle) -> np.ndarray:
    """Rotate the planar vectors in ``xy[..., :2]`` by ``angle`` radians (numpy).

    Same convention as ``parkour_isaaclab.envs.mdp.travel_frame.rotate_xy``, duplicated here
    because this module stays torch-free. ``angle`` is a scalar, or an array broadcastable against
    ``xy``'s leading axes (e.g. ``(T,)`` for ``xy`` of shape ``(T, 2)``).
    """
    if np.ndim(angle) == 0:
        c, s = math.cos(angle), math.sin(angle)
    else:
        c, s = _bcast_angle(angle, np.asarray(xy).ndim)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack((c * x - s * y, s * x + c * y), axis=-1)


def _yaw_rotate_quat_wxyz(quat: np.ndarray, angle) -> np.ndarray:
    r"""Right-multiply ``(w, x, y, z)`` quaternions by the z-rotation ``q_z(angle)``.

    Computes :math:`q \otimes q_z(\text{angle})`, i.e. the quaternion is rotated about its OWN z
    axis by ``angle`` (intrinsic rotation) -- so the rotated frame's x axis is the travel axis
    when ``angle == travel_yaw``. Vectorised over any leading shape, e.g. ``(T, 4)``; ``angle``
    may be a scalar or a per-step array broadcastable against those leading axes.
    """
    quat = np.asarray(quat, dtype=np.float64)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    if np.ndim(angle) == 0:
        cw = math.cos(angle / 2.0)
        sw = math.sin(angle / 2.0)
    else:
        cw, sw = _bcast_angle(np.asarray(angle, dtype=np.float64) / 2.0, quat.ndim)
    return np.stack(
        (w * cw - z * sw, x * cw + y * sw, y * cw - x * sw, z * cw + w * sw), axis=-1
    )


def lin_vel_to_travel_frame(lin_vel_b: np.ndarray, travel_yaw=0.0) -> np.ndarray:
    """Express a body-frame ``(..., 3)`` linear velocity in the travel frame.

    The numpy analogue of ``parkour_isaaclab.envs.mdp.travel_frame.vel_travel_xy`` (this module
    stays torch-free), extended to carry the z component through unchanged. At ``travel_yaw ==
    0.0`` this is the identity and returns ``lin_vel_b`` itself (same object). At ``travel_yaw ==
    +pi/2`` it maps ``(vx, vy, vz) -> (vy, -vx, vz)``. ``travel_yaw`` may also be a ``(T,)``
    per-step array (omni rows), in which case each step is rotated by its own angle.
    """
    if _is_identity_yaw(travel_yaw):
        return lin_vel_b
    lin_vel_b = np.asarray(lin_vel_b, dtype=np.float64)
    angle = -np.asarray(travel_yaw, dtype=np.float64) if np.ndim(travel_yaw) else -travel_yaw
    xy = _rotate_xy(lin_vel_b[..., :2], angle)
    return np.concatenate([xy, lin_vel_b[..., 2:3]], axis=-1)


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


def orientation_metrics(root_quat_w: np.ndarray, *, travel_yaw=0.0) -> dict:
    """``travel_yaw`` is a scalar or a ``(T,)`` per-step array (see the travel-frame note above)."""
    quat = root_quat_w if _is_identity_yaw(travel_yaw) else _yaw_rotate_quat_wxyz(root_quat_w, travel_yaw)
    roll, pitch = roll_pitch_from_quat_wxyz(quat)
    if roll.size == 0:
        return {"roll_rms": None, "pitch_rms": None, "roll_max_abs": None, "pitch_max_abs": None}
    return {
        "roll_rms": float(np.sqrt(np.mean(roll**2))),
        "pitch_rms": float(np.sqrt(np.mean(pitch**2))),
        "roll_max_abs": float(np.abs(roll).max()),
        "pitch_max_abs": float(np.abs(pitch).max()),
    }


# --------------------------------------------------------------------------------------
# facing (omni task, 2026-09-16): the second command channel. Travel direction is scored by
# ``tracking_metrics``; this scores "did the body reach and hold the commanded world heading".
# --------------------------------------------------------------------------------------
#: |heading error| below this counts as "on target" for ``settle_time_s`` (rad).
FACING_SETTLE_TOL_RAD = 0.15
#: Smallest INITIAL |heading error| a ``facing_closed`` fraction is computed from (rad).
#:
#: The closed fraction divides by the error the window started with, so a window that starts
#: already on target measures nothing: ``1 - final/initial`` on ``initial = 0.02`` turns a 0.01-rad
#: wobble into -0.5 "opened", and on ``initial = 0`` it is undefined. Below this floor the metric is
#: ``None`` (UNINFORMATIVE, counted in ``n_uninformative``) rather than a number nobody can compare
#: -- a stand hold's facing entry must not dilute a turn hold's. 0.15 rad is the settle tolerance:
#: an error the metric would call "already on target" cannot also be a turn worth scoring.
FACING_CLOSED_MIN_INITIAL_RAD = 0.15


def facing_metrics(
    heading_err_t: np.ndarray,
    wz_t: np.ndarray,
    dt: float,
    *,
    wz_cmd_t: np.ndarray | None = None,
    settle_tol: float = FACING_SETTLE_TOL_RAD,
    cmd_eps: float = 1e-6,
    constant_target: bool = True,
    min_initial: float = FACING_CLOSED_MIN_INITIAL_RAD,
) -> dict:
    """Facing-command tracking over one window.

    ``heading_err_t`` is the per-step wrapped error ``wrap(heading_target - heading_w)`` (rad),
    ``wz_t`` the achieved body yaw rate and ``wz_cmd_t`` (optional) the commanded one.

    * ``heading_error_rms`` / ``heading_error_max_abs`` / ``heading_error_final_abs`` /
      ``heading_error_initial_abs`` -- the window's error, its worst step, where it ended up and
      where it started. Scored over the WHOLE hold, not the steady window: a turn command *is* the
      transient, and masking the settle window out would throw away the only steps that measure it.
    * ``facing_closed`` -- the fraction of the window's INITIAL heading error that was closed by its
      end: ``clip(1 - final/max(initial, min_initial), -1, 1)``. 1.0 = the turn was completed, 0.0 =
      the error ended where it started, negative = the body opened the error (clipped at -1 so one
      spinning episode cannot dominate a median). Scale-free where ``heading_error_final_abs`` is
      not: a hold that starts at 1.57 rad and one that starts at 0.3 rad are directly comparable,
      which is what makes it a decision-rule quantity rather than a report line. ``None``
      (UNINFORMATIVE, with ``n_uninformative`` 1) when ``initial < min_initial`` -- there was no
      turn to close -- and when ``constant_target`` is False, for the same reason
      ``net_yaw_rate_toward`` is: across holds the target moves, so "the initial error" is the
      initial error of a DIFFERENT command than the final one. A hold containing a fall is scored on
      its truncated window (the caller's window mask already ends at the fall), so ``final`` is the
      error at the fall, not at a hold end the episode never reached.
    * ``n_uninformative`` -- 1 when the window carried steps and a constant target but was too
      close to its target to yield a ``facing_closed``, else 0. Summed over a HOLD's episodes (the
      ``constant_target=True`` call) it is the count of episodes the closed fraction could not be
      measured on: the same number as ``heading_error_rms``'s aggregate ``n`` minus
      ``facing_closed``'s. That identity is per hold only -- an EPISODE-level window
      (``constant_target=False``) yields ``facing_closed`` ``None`` with ``n_uninformative`` 0,
      because the refusal there is "the target moved", not "there was nothing to close".
    * ``settle_time_s`` -- time from the window start until ``|err|`` drops below ``settle_tol``
      **and stays there to the end**; ``None`` when it never does (so "slow" and "never" are
      different answers, and neither is silently 0).
    * ``yaw_progress_rate`` -- mean of ``wz_t * sign(wz_cmd_t)`` over the steps that carry a yaw
      command (rad/s **toward** the commanded turn; ``None`` when there are no such steps). SIGNED
      and direction-aware: gait wobble averages to ~0 here, a body turning the wrong way is
      negative, and a body turning with the command is positive. **This is the gate quantity**
      (``gate_1c``'s turning-rate criterion).
    * ``net_yaw_rate_toward`` -- the window's SIGNED net rotation toward its initial facing target,
      as a mean rate (rad/s; positive = turned toward the target). Derivation: with the target
      heading CONSTANT over the window, ``err_t = wrap(target - yaw_t)``, so
      ``unwrap(err)[-1] - unwrap(err)[0] = -(yaw[-1] - yaw[0])`` -- the unwrapped error carries the
      body's net rotation with the opposite sign, and the ``+-pi`` wrap that makes the raw error
      unusable is removed. Hence ``net = -(unwrap(err)[-1] - unwrap(err)[0]) / (n*dt) *
      sign(err[0])``, the last factor turning "rotation in the +z sense" into "rotation toward the
      target" for a left (``err[0] > 0``) or a right (``err[0] < 0``) turn alike. ``n*dt`` is the
      window duration, matching ``settle_time_s``'s ``(i+1)*dt`` convention. ``None`` when ``n < 2``
      or when ``constant_target=False``; an ``err[0]`` of exactly 0.0 leaves "toward" undefined and
      yields 0.0. **Only valid per hold** -- pass ``constant_target=False`` for a window that spans
      several holds (an episode), where the target steps between holds and ``unwrap`` would absorb
      those steps into the body's rotation.
    * ``yaw_rate_ratio`` -- mean achieved ``|wz|`` / mean commanded ``|wz|`` over the steps that
      carry a yaw command (``n_turn_steps``). Kept for continuity only. **WOBBLE-SENSITIVE, NOT a
      gate quantity**: both sides are magnitudes, so the per-step yaw oscillation of the gait
      ("|wz| large, net rotation ~0") scores as well as a real turn. Measured 2026-09-17 on the
      yaw-capability smoke: 0.946 on a ``turn_l90`` hold whose net rotation was -0.025 rad/s against
      a +0.30 rad/s command. Gate on ``yaw_progress_rate`` instead.
    """
    err = np.asarray(heading_err_t, dtype=np.float64).ravel()
    wz = np.asarray(wz_t, dtype=np.float64).ravel()
    out: dict[str, Any] = {
        "n_steps": int(err.size),
        "heading_error_rms": None,
        "heading_error_final_abs": None,
        "heading_error_initial_abs": None,
        "heading_error_max_abs": None,
        "facing_closed": None,
        "n_uninformative": 0,
        "settle_time_s": None,
        "yaw_rate_ratio": None,
        "yaw_progress_rate": None,
        "net_yaw_rate_toward": None,
        "n_turn_steps": 0,
    }
    if err.size:
        abs_err = np.abs(err)
        out["heading_error_rms"] = float(np.sqrt(np.mean(err**2)))
        out["heading_error_final_abs"] = float(abs_err[-1])
        out["heading_error_initial_abs"] = float(abs_err[0])
        out["heading_error_max_abs"] = float(abs_err.max())
        if constant_target:
            initial = float(abs_err[0])
            if initial < float(min_initial):
                out["n_uninformative"] = 1
            else:
                closed = 1.0 - float(abs_err[-1]) / max(initial, float(min_initial))
                out["facing_closed"] = float(min(1.0, max(-1.0, closed)))
        # First index from which every later step is inside the tolerance.
        outside = np.flatnonzero(abs_err >= settle_tol)
        if outside.size == 0:
            out["settle_time_s"] = 0.0
        elif int(outside[-1]) + 1 < abs_err.size:
            out["settle_time_s"] = float((int(outside[-1]) + 1) * dt)
    if err.size >= 2 and constant_target and dt > 0.0:
        unwrapped = np.unwrap(err)
        d_yaw = -(unwrapped[-1] - unwrapped[0])          # constant target => dyaw = -d(unwrap(err))
        out["net_yaw_rate_toward"] = float(d_yaw / (err.size * dt) * np.sign(err[0]))

    if wz_cmd_t is not None:
        cmd = np.asarray(wz_cmd_t, dtype=np.float64).ravel()
        n = min(cmd.size, wz.size)
        turning = np.abs(cmd[:n]) > cmd_eps
        out["n_turn_steps"] = int(turning.sum())
        if out["n_turn_steps"]:
            out["yaw_progress_rate"] = float((wz[:n][turning] * np.sign(cmd[:n][turning])).mean())
            denom = float(np.abs(cmd[:n][turning]).mean())
            if denom > 0.0:
                out["yaw_rate_ratio"] = float(np.abs(wz[:n][turning]).mean() / denom)
    return out


# --------------------------------------------------------------------------------------
# per-hold motion / travel direction (omni per-category eval, 2026-09-21, plan D3): world rows
# only, additive -- no body-row code path calls these.
# --------------------------------------------------------------------------------------
def travel_direction_metrics(vel_travel_xy: np.ndarray, cmd_speed, *, min_speed: float = 0.02) -> dict | None:
    """How well a moving hold held its commanded DIRECTION (plan D3 ``direction``).

    ``vel_travel_xy`` is the ``(T, 2)`` achieved velocity in the travel frame (x along the
    commanded direction, y to its left -- the frame ``tracking_metrics`` already reads), and
    ``cmd_speed`` the commanded speed (scalar or ``(T,)``; its mean is used). With ``v̄`` the mean
    travel-frame velocity over the window:

    * ``direction_error_rad`` -- ``|atan2(v̄_y, v̄_x)|``, the angle between where the body went and
      where it was sent (0 = on the commanded line, pi = backwards). ``None`` when ``|v̄| <
      min_speed``: a body that did not move has no direction, and a number there would be noise;
    * ``cross_track_ratio`` -- ``|v̄_y| / mean(cmd_speed)``, the sideways drift as a fraction of the
      commanded speed (``None`` on a zero command);
    * ``travel_speed_m_s`` -- ``|v̄|``, the net travel speed in whatever direction the body went
      (WATCH only: the along-command speed a block reads is ``tracking.vx.actual_mean``).

    ``None`` on an empty window.
    """
    v = np.asarray(vel_travel_xy, dtype=np.float64).reshape(-1, 2)
    if v.shape[0] == 0:
        return None
    mean = v.mean(axis=0)
    speed = float(math.hypot(float(mean[0]), float(mean[1])))
    cmd = float(np.mean(np.asarray(cmd_speed, dtype=np.float64)))
    return {
        "direction_error_rad": (
            float(abs(math.atan2(float(mean[1]), float(mean[0])))) if speed >= float(min_speed) else None
        ),
        "cross_track_ratio": (float(abs(float(mean[1])) / cmd) if cmd > 0.0 else None),
        "travel_speed_m_s": speed,
    }


def hold_motion_metrics(
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    root_lin_vel_b: np.ndarray,
    contact: np.ndarray,
    dt: float,
) -> dict | None:
    """What the body did over one whole hold (plan D3 ``motion``); ``None`` on an empty window.

    * ``speed_abs_mean_m_s`` -- mean planar body speed ``‖v_xy,body‖`` (a stand hold's drift);
    * ``pos_drift_m`` -- planar distance between the window's first and last root position;
    * ``path_len_m`` -- planar path length of the root over the window;
    * ``yaw_drift_rad`` -- SIGNED net yaw change (unwrapped last minus first; + = left);
    * ``foot_lift_count`` -- contact -> air transitions, summed over the six feet;
    * ``contact_churn_per_s`` -- every contact state change (lift or touchdown), summed over the
      feet, per second of window (``n_steps * dt``);
    * ``n_steps``.
    """
    pos = np.asarray(root_pos_w, dtype=np.float64)
    n = int(pos.shape[0]) if pos.ndim >= 1 else 0
    if n == 0:
        return None
    xy = pos[:, :2]
    vel = np.asarray(root_lin_vel_b, dtype=np.float64)[:, :2]
    yaw = np.unwrap(yaw_from_quat_wxyz(root_quat_w))
    c = np.asarray(contact, dtype=bool).astype(np.int8)
    changes = np.diff(c, axis=0) if n > 1 else np.zeros((0,) + c.shape[1:], dtype=np.int8)
    return {
        "speed_abs_mean_m_s": float(np.linalg.norm(vel, axis=-1).mean()),
        "pos_drift_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "path_len_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()) if n > 1 else 0.0,
        "yaw_drift_rad": float(yaw[-1] - yaw[0]),
        "foot_lift_count": int(np.count_nonzero(changes < 0)),
        "contact_churn_per_s": float(np.count_nonzero(changes) / (n * dt)) if dt > 0.0 else None,
        "n_steps": n,
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
    travel_yaw=0.0,
) -> dict:
    """Static-stability margins of the loaded-foot support polygon, in the heading frame.

    Per frame: the loaded set ``L = {i : force >= support_threshold_n}``; feet and the CoM
    projected into the heading frame (yaw-only rotation about the root, +x' = forward, i.e. the travel axis when travel_yaw is nonzero);
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
    if not _is_identity_yaw(travel_yaw):
        # Heading frame built from the travel axis rather than body +x -- see module-level
        # convention note. fwd_ray_margin_m / tip_angle_fwd_deg / fore_aft_span_m then read along
        # the travel axis instead of body +x. A ``(T,)`` travel_yaw turns the frame per step.
        yaw = yaw + travel_yaw
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
    travel_yaw=0.0,
) -> dict:
    """Classify a termination by attitude at the failure step.

    ``pitch_fwd`` (nose-down, pitch >= +threshold and |pitch| >= |roll|), ``pitch_back``,
    ``roll``, or ``none`` (no crab_failure in the trace). Positive pitch = nose-down, matching
    ``penalty_base_pitch_forward_linear``. Also reports time-to-fall and the peak pitch rate
    over the last ``rate_window_s`` when ``root_ang_vel_b`` is given.

    ``travel_yaw`` rotates the quaternion (and the ang-vel xy plane) into the travel frame before
    the roll/pitch decomposition, so the labels ``pitch_fwd``/``pitch_back`` keep meaning "tips
    along the travel axis" for a sideways-walking robot. At ``travel_yaw == 0.0`` (the forward
    task) this is exactly today's body-frame computation.
    """
    fails = np.flatnonzero(np.asarray(crab_failure, dtype=bool))
    if fails.size == 0:
        return {"fall_class": "none", "t_fail_s": None, "pitch_at_fail": None,
                "roll_at_fail": None, "max_pitch_rate": None}
    t = int(fails[0])
    quat_t = np.asarray(root_quat_w)[t]
    if not _is_identity_yaw(travel_yaw):
        # A (T,) travel_yaw is per step: the failure step's own command direction is the frame the
        # fall is classified in.
        quat_t = _yaw_rotate_quat_wxyz(quat_t, travel_yaw[t] if np.ndim(travel_yaw) else travel_yaw)
    roll, pitch = roll_pitch_from_quat_wxyz(quat_t)
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
        if _is_identity_yaw(travel_yaw):
            seg = w[lo:t + 1, 1]
        else:
            angle = np.asarray(travel_yaw, dtype=np.float64)[lo:t + 1] if np.ndim(travel_yaw) else travel_yaw
            seg = _rotate_xy(w[lo:t + 1, :2], -angle)[:, 1]
        rate = float(np.abs(seg).max()) if seg.size else None
    return {"fall_class": cls, "t_fail_s": t * dt, "pitch_at_fail": pitch,
            "roll_at_fail": roll, "max_pitch_rate": rate}


# ---------------------------------------------------------------------------------------------
# Motion quality (gait-motion-quality plan, step 1, 2026-09-22)
# ---------------------------------------------------------------------------------------------
#: Contact threshold for every motion-quality metric, newtons. Mirrors
#: ``crab_hex_forward_task/mdp/crab_hex_clock_reward.TOUCH_REF_N`` (the clock reward's "really
#: loaded" half-load point; pinned against that file's source by a unit test) -- NOT the scenario's
#: ``contact_force_threshold``, which the tripod score uses and which a lightly grazing foot clears.
MOTION_TOUCH_REF_N = 50.0
#: ``settle_six_down_s``: six-down must then HOLD this long to count as settled (a touchdown that
#: is lifted again 0.1 s later is not a settled stance).
SIX_DOWN_DWELL_S = 0.5
#: Per-metre metrics are undefined (``None``) below this net planar displacement, metres.
MIN_TRAVEL_M = 0.05
#: Per-radian metrics are undefined (``None``) below this |net yaw|, radians.
MIN_NET_YAW_RAD = 0.05
#: Spectra need at least this many samples (``action_fft_freq`` / ``action_sm``).
MIN_FFT_STEPS = 16

#: Every motion-quality key a category summary carries, in print order.
MOTION_QUALITY_KEYS: tuple[str, ...] = (
    "action_jitter", "action_fft_freq", "joint_speed_mean", "rod_speed_mean", "contact_switch_hz",
    "foot_slip", "wz_active_frac", "clock_restarts",
    "swings_per_m", "slip_per_m", "work_per_m", "cam_work_per_m",
    "swings_per_rad", "slip_per_rad", "work_per_rad", "cam_work_per_rad",
    "six_down_frac", "settle_six_down_s", "foot_disp_end", "work_per_s",
)

_ALL_CATS = frozenset({"stand", "fwd", "side", "back", "diag", "turn", "move_turn"})
_TRANSLATE = frozenset({"fwd", "side", "back", "diag"})
_TURNING = frozenset({"turn", "move_turn"})
#: key -> the categories it is DEFINED on (the plan's step-1 table). Everywhere else the key is
#: ``None`` -- N/A, never 0: a per-metre number where the only distance is drift is noise, and a
#: stand-only number on a walking hold means nothing.
MOTION_QUALITY_APPLIES: dict[str, frozenset] = {
    **{k: _ALL_CATS for k in ("action_jitter", "action_fft_freq", "joint_speed_mean", "rod_speed_mean",
                              "contact_switch_hz", "foot_slip", "wz_active_frac", "clock_restarts")},
    **{k: _TRANSLATE for k in ("swings_per_m", "slip_per_m", "work_per_m", "cam_work_per_m")},
    **{k: _TURNING for k in ("swings_per_rad", "slip_per_rad", "work_per_rad", "cam_work_per_rad")},
    **{k: frozenset({"stand"}) for k in ("six_down_frac", "settle_six_down_s", "foot_disp_end", "work_per_s")},
}


def motion_joint_groups(joint_names: Sequence[str]) -> dict[str, list[int]]:
    """Articulation joint indices for the motion-quality metrics.

    * ``rod`` -- the linear-actuator (lead-screw) joints: hip pitch ``*_Hip_Femur_RevoluteJoint`` and
      knee ``*_Femur_Tibia_RevoluteJoint`` (``parkour_actions.CrabHexDelayedJointPositionAction``'s
      pitch linkage);
    * ``cam`` -- the cam mechanism: the velocity-commanded ``*_Body_CamShaft_RevoluteJoint`` motor
      AND the ``*_Body_Hip_RevoluteJoint`` it drives through the high-gain linkage emulator (that
      actuator carries the cam's mechanical load in sim);
    * ``actuated`` -- the 18 policy-commanded joints (camshaft + rod), i.e. the action set.
    """
    groups: dict[str, list[int]] = {"rod": [], "cam": [], "actuated": []}
    for i, name in enumerate(joint_names):
        if "_Hip_Femur_" in name or "_Femur_Tibia_" in name:
            groups["rod"].append(i)
            groups["actuated"].append(i)
        elif "_Body_CamShaft_" in name:
            groups["cam"].append(i)
            groups["actuated"].append(i)
        elif "_Body_Hip_" in name:
            groups["cam"].append(i)
    return groups


def action_jitter(actions: np.ndarray) -> float | None:
    """Mean absolute SECOND difference of the actions, ``mean_t,d |a[t+1] - 2 a[t] + a[t-1]|``.

    Action units per step^2 (raw policy units; the harness clips to ``clip_actions`` first). A
    constant or a straight ramp scores 0; a +-a sawtooth at the step rate scores 4a. ``None`` under
    three steps.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < 3:
        return None
    return float(np.abs(a[2:] - 2.0 * a[1:-1] + a[:-2]).mean())


def _action_spectrum(actions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray] | None:
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    n = a.shape[0]
    if n < MIN_FFT_STEPS or not dt > 0.0:
        return None
    spec = np.fft.rfft(a - a.mean(axis=0, keepdims=True), axis=0)
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, np.abs(spec)


def action_fft_freq(actions: np.ndarray, dt: float) -> float | None:
    """POWER-weighted mean frequency of the actions, Hz: ``sum_{d,i} P_di f_i / sum_{d,i} P_di``.

    Each action dimension is de-meaned (so a held offset carries no power), ``P = |rfft|^2``, the DC
    bin is excluded and the power is pooled over dimensions (a dimension that barely moves barely
    counts). The Christmann et al. 2024 smoothness family (see :func:`action_sm` for their exact
    amplitude-weighted ``Sm``); this form reads directly as "where the action energy sits".
    ``None`` under :data:`MIN_FFT_STEPS` steps or when the actions carry no AC power at all.
    """
    got = _action_spectrum(actions, dt)
    if got is None:
        return None
    freqs, amp = got
    power = amp[1:] ** 2
    total = float(power.sum())
    if not total > _VAR_EPS:
        return None
    return float((power * freqs[1:, None]).sum() / total)


def action_sm(actions: np.ndarray, dt: float) -> float | None:
    """Christmann et al. 2024's smoothness ``Sm = 2 / (n f_s) * sum_i M_i f_i``, mean over dims.

    ``M_i = |rfft(a - mean)|_i / T`` (amplitude), ``n`` the number of rfft bins, ``f_s = 1/dt``.
    Kept beside :func:`action_fft_freq` as the literature's own number (diagnostic, not a category
    key). ``None`` under :data:`MIN_FFT_STEPS` steps.
    """
    got = _action_spectrum(actions, dt)
    if got is None:
        return None
    freqs, amp = got
    T = (freqs.size - 1) * 2 if freqs.size > 1 else 1
    m = amp / max(1, T)
    fs = 1.0 / dt
    per_dim = 2.0 / (freqs.size * fs) * (m * freqs[:, None]).sum(axis=0)
    return float(per_dim.mean())


def contact_transitions(contact: np.ndarray) -> tuple[int, int]:
    """``(liftoffs, touchdowns)`` summed over feet: True->False and False->True between steps."""
    c = np.asarray(contact, dtype=bool).astype(np.int8)
    if c.shape[0] < 2:
        return 0, 0
    d = np.diff(c, axis=0)
    return int(np.count_nonzero(d < 0)), int(np.count_nonzero(d > 0))


def settle_time_s(mask: np.ndarray, dt: float, dwell_s: float = SIX_DOWN_DWELL_S) -> float | None:
    """Seconds from the window start to the first step that begins a run of ``mask`` at least
    ``dwell_s`` long (``ceil(dwell_s/dt)`` steps); ``None`` when no such run exists (censored)."""
    m = np.asarray(mask, dtype=bool)
    need = max(1, int(math.ceil(dwell_s / dt - 1e-9)))
    for start, end in contiguous_runs(m):
        if end - start >= need:
            return float(start * dt)
    return None


def motion_quality_metrics(
    category: str,
    *,
    dt: float,
    actions: np.ndarray | None,
    foot_force_norm: np.ndarray,
    foot_pos_w: np.ndarray,
    foot_lin_vel_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
    cmd_wz: np.ndarray,
    joint_vel: np.ndarray | None = None,
    applied_torque: np.ndarray | None = None,
    groups: dict[str, Sequence[int]] | None = None,
    clock_running: np.ndarray | None = None,
    rod_speed: np.ndarray | None = None,
    action_clip: float | None = None,
    hold_s_nominal: float | None = None,
    touch_ref_n: float = MOTION_TOUCH_REF_N,
) -> dict | None:
    """The motion-quality block of one episode's OWN hold (``hold["quality"]``); ``None`` if empty.

    Every array is the hold's window only (time first). The keys of :data:`MOTION_QUALITY_KEYS`
    are always present; a key not defined on ``category`` (:data:`MOTION_QUALITY_APPLIES`) or not
    measurable on this window (missing input, too short, no distance) is ``None`` -- never 0.

    Definitions (``loaded`` = foot force > ``touch_ref_n`` = 50 N; ``W`` = the window, ``|W|`` its
    duration ``n * dt``):

    * ``action_jitter`` -- :func:`action_jitter` of the actions clipped to ``+-action_clip``;
    * ``action_fft_freq`` -- :func:`action_fft_freq`, Hz;
    * ``joint_speed_mean`` -- mean ``|qdot|`` over the ``actuated`` joints, rad/s;
    * ``rod_speed_mean`` -- mean ``|rod speed|`` over the 12 linear actuators, m/s (the action term's
      slew-limited rod-length rate, recorded by the harness as ``rod_speed``);
    * ``contact_switch_hz`` -- loaded<->unloaded transitions per foot per second (mean over feet);
    * ``foot_slip`` -- mean planar speed of the loaded feet over loaded (step, foot) samples, m/s;
    * ``wz_active_frac`` -- fraction of steps whose APPLIED yaw-rate command is nonzero (the heading
      relay engaged: the dead band zeroes it exactly);
    * ``clock_restarts`` -- gait-clock stopped->running transitions inside the window (a count;
      needs the harness's ``clock_running``);
    * per metre (translation categories), ``D`` = net planar root displacement over the window,
      ``None`` below :data:`MIN_TRAVEL_M`: ``swings_per_m`` = liftoffs (loaded->unloaded, summed over
      feet) / D; ``slip_per_m`` = (sum over loaded samples of planar foot speed * dt) / D (m/m);
      ``work_per_m`` = (sum over rod joints and steps of max(tau*qdot, 0) * dt) / D, J/m;
      ``cam_work_per_m`` = (sum over cam joints of |tau*qdot| * dt) / D, J/m;
    * per radian (turn, move_turn), the same numerators over ``|net yaw|`` (unwrapped, ``None`` below
      :data:`MIN_NET_YAW_RAD`): ``swings_per_rad``, ``slip_per_rad`` (m/rad), ``work_per_rad``,
      ``cam_work_per_rad`` (J/rad);
    * stand only: ``six_down_frac`` = fraction of steps with all six feet loaded;
      ``settle_six_down_s`` = :func:`settle_time_s` of six-down (first run >= 0.5 s), CENSORED at
      ``hold_s_nominal`` (else ``|W|``) when never reached, with ``settle_six_down_censored`` True;
      ``foot_disp_end`` = mean over feet of the planar distance between each foot's position at the
      window's last step and at its first (the footholds at the end of the settle hold), m;
      ``work_per_s`` = (rod positive work + cam |work|) / |W|, W.

    Extra (non-category) fields: ``category``, ``n_steps``, ``duration_s``, ``distance_m``,
    ``net_yaw_rad``, ``liftoffs``, ``touchdowns``, ``rod_work_pos_j``, ``cam_work_abs_j``,
    ``slip_m``, ``action_sm``, ``settle_six_down_censored``, ``touch_ref_n``.
    """
    force = np.asarray(foot_force_norm, dtype=np.float64)
    n = int(force.shape[0]) if force.ndim >= 1 else 0
    if n == 0:
        return None
    dur = n * dt
    loaded = force > touch_ref_n
    applies = {k: category in cats for k, cats in MOTION_QUALITY_APPLIES.items()}
    out: dict[str, Any] = {k: None for k in MOTION_QUALITY_KEYS}
    extra: dict[str, Any] = {"category": category, "n_steps": n, "duration_s": dur,
                             "touch_ref_n": float(touch_ref_n), "settle_six_down_censored": None}

    # ---- actions
    acts = None
    if actions is not None:
        acts = np.asarray(actions, dtype=np.float64)
        if action_clip is not None:
            acts = np.clip(acts, -float(action_clip), float(action_clip))
    out["action_jitter"] = action_jitter(acts) if acts is not None else None
    out["action_fft_freq"] = action_fft_freq(acts, dt) if acts is not None else None
    extra["action_sm"] = action_sm(acts, dt) if acts is not None else None

    # ---- joints
    g = {k: list(v) for k, v in (groups or {}).items()}
    jv = np.asarray(joint_vel, dtype=np.float64) if joint_vel is not None else None
    if jv is not None and g.get("actuated"):
        out["joint_speed_mean"] = float(np.abs(jv[:, g["actuated"]]).mean())
    if rod_speed is not None:
        rs = np.asarray(rod_speed, dtype=np.float64)
        out["rod_speed_mean"] = float(np.abs(rs).mean()) if rs.size else None
    rod_work = cam_work = None
    if jv is not None and applied_torque is not None and g.get("rod") and g.get("cam"):
        power = np.asarray(applied_torque, dtype=np.float64) * jv
        rod_work = float(np.clip(power[:, g["rod"]], 0.0, None).sum() * dt)
        cam_work = float(np.abs(power[:, g["cam"]]).sum() * dt)
    extra["rod_work_pos_j"], extra["cam_work_abs_j"] = rod_work, cam_work

    # ---- contacts, slip
    lifts, downs = contact_transitions(loaded)
    extra["liftoffs"], extra["touchdowns"] = lifts, downs
    out["contact_switch_hz"] = float((lifts + downs) / (loaded.shape[1] * dur)) if dur > 0 else None
    speed_xy = np.linalg.norm(np.asarray(foot_lin_vel_w, dtype=np.float64)[..., :2], axis=-1)
    out["foot_slip"] = float(speed_xy[loaded].mean()) if loaded.any() else None
    slip_m = float(speed_xy[loaded].sum() * dt)
    extra["slip_m"] = slip_m

    # ---- command / clock
    wz = np.asarray(cmd_wz, dtype=np.float64)
    out["wz_active_frac"] = float(np.count_nonzero(wz != 0.0) / wz.size) if wz.size else None
    if clock_running is not None:
        cr = np.asarray(clock_running, dtype=bool).astype(np.int8)
        out["clock_restarts"] = int(np.count_nonzero(np.diff(cr) > 0)) if cr.size > 1 else 0

    # ---- normalisers
    xy = np.asarray(root_pos_w, dtype=np.float64)[:, :2]
    dist = float(np.linalg.norm(xy[-1] - xy[0]))
    yaw = np.unwrap(yaw_from_quat_wxyz(np.asarray(root_quat_w, dtype=np.float64)))
    net_yaw = float(abs(yaw[-1] - yaw[0]))
    extra["distance_m"], extra["net_yaw_rad"] = dist, net_yaw

    def _per(num: float | None, den: float, floor: float) -> float | None:
        return None if num is None or den < floor else float(num / den)

    out["swings_per_m"] = _per(float(lifts), dist, MIN_TRAVEL_M)
    out["slip_per_m"] = _per(slip_m, dist, MIN_TRAVEL_M)
    out["work_per_m"] = _per(rod_work, dist, MIN_TRAVEL_M)
    out["cam_work_per_m"] = _per(cam_work, dist, MIN_TRAVEL_M)
    out["swings_per_rad"] = _per(float(lifts), net_yaw, MIN_NET_YAW_RAD)
    out["slip_per_rad"] = _per(slip_m, net_yaw, MIN_NET_YAW_RAD)
    out["work_per_rad"] = _per(rod_work, net_yaw, MIN_NET_YAW_RAD)
    out["cam_work_per_rad"] = _per(cam_work, net_yaw, MIN_NET_YAW_RAD)

    # ---- stand
    six = loaded.all(axis=1)
    out["six_down_frac"] = float(six.mean())
    settle = settle_time_s(six, dt)
    if settle is None:
        out["settle_six_down_s"] = float(hold_s_nominal) if hold_s_nominal is not None else float(dur)
        extra["settle_six_down_censored"] = True
    else:
        out["settle_six_down_s"] = settle
        extra["settle_six_down_censored"] = False
    fp = np.asarray(foot_pos_w, dtype=np.float64)
    out["foot_disp_end"] = float(np.linalg.norm(fp[-1, :, :2] - fp[0, :, :2], axis=-1).mean())
    out["work_per_s"] = (float((rod_work + cam_work) / dur)
                         if rod_work is not None and cam_work is not None and dur > 0 else None)

    for k in MOTION_QUALITY_KEYS:
        if not applies[k]:
            out[k] = None
    if category != "stand":
        extra["settle_six_down_censored"] = None
    return {**out, **extra}
