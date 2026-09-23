# SPDX-License-Identifier: BSD-3-Clause
"""Clock-referenced contact-schedule reward (gait-formation-v2 Phase 1, 2026-08-22).

The literature's standard fix for the chicken-and-egg that killed every crossing-credit
campaign (see parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/lit-review-hexapod-reward-stability.md section 2; Siekmann et al. ICRA
2021, Margolis & Agrawal CoRL 2022): the target gait exists in the reward FROM STEP 0 as a
periodic per-foot contact schedule driven by an external monotonic clock, instead of being
an income that only pays once the gait already exists.

Schedule: alternating tripod. Set A = {FL, MR, RL} swings during the first half of the
clock cycle, set B = {FR, ML, RR} during the second half (offsets 0 / pi). Per foot:

  income_i = I_swing(phi_i) * exp(-(F_i / force_ref)^2)      (unloaded when told to swing)
           + I_stance(phi_i) * exp(-(|v_i| / vel_ref)^2)     (planted-still when told to stand)

averaged over feet, gated upright and (all-stance) on stop commands. Every term is
POSITIVE income in [0, 1]: no holdable state earns full income while the clock advances
(standing violates half the schedule at all times), and there is no penalty mass for a
falling policy to escape by dying early.

Duty is fixed at 50/50 by the smooth half-cycle windows -- the duty asymmetry that was
"not economic" to fix under crossing credit is *specified* here, per the review's ranked
recommendation #1.

The pure function is Isaac-free so the offline replay gate and unit tests exercise the
exact shipped math.
"""
from __future__ import annotations

import torch

# Foot order everywhere in the crab task: FL, FR, ML, MR, RL, RR.
FOOT_ORDER = ("FL", "FR", "ML", "MR", "RL", "RR")
TRIPOD_A = ("FL", "MR", "RL")
# Clock phase offset per foot: set A swings first (phi_i = phi + offset_i; swing when
# sin(phi_i) > 0).
FOOT_OFFSETS = tuple(0.0 if leg in TRIPOD_A else torch.pi for leg in FOOT_ORDER)

WINDOW_SHARPNESS = 0.15
FORCE_REF_N = 100.0
VEL_REF_M_S = 0.10
# Stance income requires REAL load, not merely a slow foot: without this gate a foot
# hovering motionless above ground farms full stance income (unit test caught it), and on
# this statically-stable slow-actuator plant hover-holding is cheap. Half-load at ~50 N
# against the ~700 N nominal per-foot stance share.
TOUCH_REF_N = 50.0


def clock_command_speed(
    cmd: torch.Tensor,
    planar: bool = False,
    turn_radius: float = 0.0,
) -> torch.Tensor:
    """The commanded speed that drives the gait clock's cadence, ``[N]``.

    One seam, two tasks. The forward-walk task (and the sideways sibling, whose command
    also lives in ``cmd[:, 0]``) travels along ONE body axis, so the cadence is set by the
    magnitude of that single channel -- ``planar=False`` returns the literal expression the
    action term and the two clock rewards have always used, ``cmd[:, 0].abs()``, so the
    default path is bit-identical (no arithmetic added, same tensor op).

    An omnidirectional task commands a 2-D planar velocity plus a yaw rate, and a robot that
    is turning in place is walking even though ``cmd[:, 0]`` may be 0. ``planar=True``
    therefore takes the larger of the planar command norm and the tangential foot speed a
    turn of radius ``turn_radius`` implies (``r * |wz|``), so the clock never stalls under an
    active turn. ``turn_radius=0.0`` degenerates to the pure planar norm.

    Args:
        cmd: ``[N, >=3]`` base-velocity command (vx, vy, wz) in the body frame.
        planar: read the planar command (omni tasks) instead of ``cmd[:, 0]`` alone.
        turn_radius: metres; the effective turning radius mapping |wz| to a foot speed.
    """
    if not planar:
        return cmd[:, 0].abs()
    return torch.maximum(
        torch.linalg.norm(cmd[:, :2], dim=-1), turn_radius * cmd[:, 2].abs()
    )


#: Tolerance (in units of pi) under which a phase counts as sitting ON a touchdown boundary.
PARK_EPS = 1e-5


def park_clock_phase(phase: torch.Tensor, max_step: float | torch.Tensor) -> torch.Tensor:
    """Advance a PARKED gait clock toward its next touchdown and hold it there, ``[N]``.

    The stand-park knob (``KRABBY_OMNI_STAND_PARK``, omni task only; default OFF, and nothing
    on the forward/sideways path calls this). Under the alternating-tripod schedule above a
    swing window is ``sin(phase + offset_i) > 0``: set A swings on ``(0, pi)`` and touches down
    at ``pi``, set B swings on ``(pi, 2 pi)`` and touches down at ``2 pi == 0``. So the next
    TOUCHDOWN is the next multiple of ``pi`` at or after ``phase``; there every foot is at the
    boundary of its stance window, and the clock resumes from a touchdown -- not from the
    middle of a swing it froze in -- when motion is commanded again.

    ``phase`` advances by at most ``max_step`` rad and never past that multiple (it lands on it
    exactly), then is wrapped to ``[0, 2 pi)``. A phase already ON a multiple of ``pi`` stays
    where it is (a float32 phase a few ulp PAST a multiple counts as on it -- :data:`PARK_EPS`
    -- so rounding can never send a parked clock round another half cycle). Pure torch: the
    unit tests pin exactly the function the action term calls.
    """
    target = torch.ceil(phase / torch.pi - PARK_EPS) * torch.pi
    advanced = torch.minimum(phase + max_step, target)
    return torch.remainder(advanced, 2.0 * torch.pi)


def clock_schedule_income(
    phase: torch.Tensor,
    foot_force_n: torch.Tensor,
    foot_speed_xy: torch.Tensor,
    clock_running: torch.Tensor,
    *,
    force_ref: float = FORCE_REF_N,
    vel_ref: float = VEL_REF_M_S,
    sharpness: float = WINDOW_SHARPNESS,
    combine: str = "sum",
) -> torch.Tensor:
    """Per-env schedule income in [0, 1].

    Args:
        phase: ``[N]`` clock phase (rad).
        foot_force_n: ``[N, 6]`` contact force magnitude per foot, FOOT_ORDER.
        foot_speed_xy: ``[N, 6]`` planar foot speed (m/s), FOOT_ORDER.
        clock_running: ``[N]`` bool -- False on stop commands (all-stance schedule).
        combine: ``"sum"`` = additive halves (original); ``"product"`` = swing-quality x
            stance-quality. The additive form pays a creeper its entire stance half for
            free (~0.5 ceiling), leaving only ~0.1/step between creep and true walking --
            measured 2026-08-22: seed-2 creep earned 0.28-0.48 vs seed-1 walking 0.41,
            and the basin lottery persisted. The product form zeroes the creep ceiling
            (swing quality ~0 multiplies everything) while walking earns ~0.8, and the
            gradient at creep is STRONGER (each swing improvement is scaled by the
            already-good stance quality).
    """
    offsets = torch.as_tensor(FOOT_OFFSETS, device=phase.device, dtype=phase.dtype)
    phi = phase[:, None] + offsets[None, :]
    # Smooth half-cycle swing window: sigma(sin(phi)/kappa) ~ 1 in (0, pi), ~0 in (pi, 2pi).
    i_swing = torch.sigmoid(torch.sin(phi) / sharpness)
    i_swing = torch.where(clock_running[:, None], i_swing, torch.zeros_like(i_swing))
    i_stance = 1.0 - i_swing
    swing_ok = torch.exp(-((foot_force_n / force_ref) ** 2))
    loaded = 1.0 - torch.exp(-((foot_force_n / TOUCH_REF_N) ** 2))
    stance_ok = loaded * torch.exp(-((foot_speed_xy / vel_ref) ** 2))
    if combine == "sum":
        return (i_swing * swing_ok + i_stance * stance_ok).mean(dim=1)
    if combine != "product":
        raise ValueError(f"unknown combine mode '{combine}'")
    w_swing = i_swing.sum(dim=1).clamp_min(1e-6)
    w_stance = i_stance.sum(dim=1).clamp_min(1e-6)
    swing_quality = (i_swing * swing_ok).sum(dim=1) / w_swing
    stance_quality = (i_stance * stance_ok).sum(dim=1) / w_stance
    # Stop commands: no swing windows exist -- income is pure stance quality (unchanged
    # station-keeping semantics).
    product = swing_quality * stance_quality
    return torch.where(clock_running, product, stance_quality)


# Mid-swing apex window: pay only near the swing's center (sin(phi_i) above this), where
# the commanded apex should be reached.
APEX_MID_THRESH = 0.7
APEX_TARGET_M = 0.08
APEX_SIGMA_M = 0.03


def clock_swing_apex_income(
    phase: torch.Tensor,
    foot_height_m: torch.Tensor,
    clock_running: torch.Tensor,
    *,
    apex_m: float = APEX_TARGET_M,
    sigma_m: float = APEX_SIGMA_M,
    mid_thresh: float = APEX_MID_THRESH,
) -> torch.Tensor:
    """Scheduled swing-apex income in [0, 1] (gait-formation-v2 Phase 4, 2026-08-24).

    Walk-These-Ways-style SPECIFIED footswing height: at each foot's mid-swing (its clock
    window center), pay for the foot's height above ground tracking a commanded apex.
    Income-priced clearance lost to survival economics at every weight/floor tried
    (G1-G3, I1: clearance pinned 0.042-0.051); this term makes the apex part of the
    schedule instead of a price negotiation. Dense: a 4 cm swing told to reach 8 cm earns
    partial credit with the gradient pointing up.

    Args:
        phase: ``[N]`` clock phase (rad).
        foot_height_m: ``[N, 6]`` foot height above nominal ground, FOOT_ORDER.
        clock_running: ``[N]`` bool -- stops pay nothing (no swings are scheduled).
    """
    offsets = torch.as_tensor(FOOT_OFFSETS, device=phase.device, dtype=phase.dtype)
    phi = phase[:, None] + offsets[None, :]
    s = torch.sin(phi)
    # Smooth mid-swing indicator: 0 outside the window center, ~1 at dead center.
    i_mid = ((s - mid_thresh) / (1.0 - mid_thresh)).clamp(0.0, 1.0)
    i_mid = torch.where(clock_running[:, None], i_mid, torch.zeros_like(i_mid))
    apex_ok = torch.exp(-(((foot_height_m - apex_m) / sigma_m) ** 2))
    # Normalize by total mid-window weight so income is per-scheduled-apex, not per-foot.
    w = i_mid.sum(dim=1).clamp_min(1e-6)
    return (i_mid * apex_ok).sum(dim=1) / w
