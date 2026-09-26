# SPDX-License-Identifier: BSD-3-Clause
"""Per-foot loaded stance-stroke income and the body-centring bonus (stride/posture plan, 2026-09-25). Pure torch.

WHY A REPLACEMENT. ``reward_stride_length`` (``RewardStrideLength`` -> ``crab_hex_stride_reward``) is
direction-correct but does not work in any direction:

* **Dead.** Its touchdown/liftoff edges come from ``compute_first_contact`` / ``compute_first_air(step_dt)``
  with Isaac Lab's default ``abs_tol`` 1e-8. Once the float32 sensor ``_timestamp`` drifts, about 2 s
  after each reset, the elapsed time never passes that test again. The same bug kills
  ``reward_feet_air_time_positive``.
* **Negligible.** It pays each stance once, but the manager still multiplies by ``weight * dt``, so even
  alive it would be about 0.1 % of tracking income.
* **Mis-specified.**
  - It credits BODY progress to every planted leg. On omni side holds only about 40 % of what it pays is
    the foot's own stroke; the rest is slip and sway.
  - Its per-step clamp rectifies sway.
  - Its planar gate pays 0 on turn-in-place.

Evidence: ``~/.claude/plans/let-s-examine-the-stride-gleaming-dream.evidence/``.

:func:`foot_stroke_step` pays each foot's OWN loaded sweep, once per stance, at close:

* **Edges** (:func:`loaded_stance_edges`) come from the current contact force with hysteresis:
  - a stance opens at ``F > FORCE_OPEN_N``;
  - it closes after ``N_OFF`` consecutive steps below ``FORCE_HOLD_N``;
  - a step is *loaded* only if the stance is open and ``F >= FORCE_HOLD_N`` at both t-1 and t.

  No sensor timer is read, so the float32 drift cannot touch it. The contact sensor repeats its last
  reading on about 1 env step in 5 between 16 and 32 s of an episode. A repeated reading can shift an edge
  by one step: a repeated low reading can close a stance one step early, turning a one-step dip into a
  close. It cannot open a stance that never loaded.
* **Dips count.** The sweep of the steps inside an open stance that are not loaded (a one-step load dip
  and the step after it) is held in a pending buffer. It is added in full if the stance carries on, and
  dropped if the stance closes (an honest liftoff). Otherwise a stick-slip foot would be paid for its
  stick phases alone, with its forward repositions hidden in the dips.
* **Own sweep, capped at body progress.** ``d = -dr . u_i`` is the foot's heading-frame sweep along
  ``u_i``, the direction of its own commanded velocity ``v_i = v_cmd + w_cmd x r_i``. The *anchor* sweep
  ``d_a = d + (R(-psi) dp_world) . u_i`` removes the foot's own world motion. The pay uses
  ``min(own, anchor)``: a planted foot is unchanged, and a foot slipping BACKWARD under load (paddling)
  earns no more than a planted one.
* **Net, not rectified.** Sweeps are accumulated signed and clamped only on the stance total, so sway
  inside a stance nets out.
* **Slip gate per stance:** ``rho = world path of the loaded foot / own sweep``. A per-step gate was
  measured to RECTIFY side income.
* **Progress guard.** Sway ACROSS stances -- a tripod-locked rock that is paid on the forward half and
  zero on the backward half -- is caught by an EMA of each foot point's tracking ratio,
  ``(v_b + w_b x r_i) . v_i / |v_i|^2`` (without ``w_b x r_i`` on non-turning rows), clamped per step.
  It pays nothing below ``PROG_LO``. A fresh hold starts at the pass value. A rock slower than the EMA
  (half-periods >= 2 s) can still pass it, but at the R4 dose such a rock loses several times more
  tracking income than it earns here.
* **No yaw credit on non-turning rows.** There the sweep is differenced with both foot positions in the
  CURRENT heading frame, so heading wiggle or drift is never paid as stroke.
* **Normalised** to the gait clock's planned stroke: ``L = max(|v_i|, v_clk / 2) / (2 f)``, which is
  0.35 m for any translation. The floor stops a foot near a turn's centre from being over-weighted.
  Normalised as ``C = own sweep / mean L``.
* **Paid as ``pay / dt``**, so the manager's ``weight * dt`` delivers exactly ``weight * pay`` per stance:
  an event term without the dt dilution.
* **Gate: command only** -- ``clock speed > 0.2`` and not STAND. On non-turning slots the facing relay's
  yaw rate is ignored (``turning``), so the projection is the translation the slot commands.
  Accumulators keep only the current gated run: a stance that crosses into STAND pays nothing, and
  nothing carries across a stand hold.

:func:`body_centring_step` is the posture-arm lever: a dense bonus for keeping the body over its feet
along a LATERAL command. The omni heads lead their six-foot centroid by about 0.23 m on side holds; the
sideways-task head leads it by about 0.01 m and strokes long.

Both terms are bonuses (the reward manager clips the summed step reward at 0, so penalties would be
absorbed). Both are equivariant under the omni mirror group {I, M_y, M_x, R_z(pi)}: every quantity is a
dot product, a norm or ``|v_y|``, and the sum over feet does not depend on foot order.

No ``isaaclab`` import: the Isaac wrappers live in ``crab_hex_omni_stroke_rewards``, and the offline replay
tool path-loads this module, so replay prices exactly the arithmetic that trains. ``pay`` lands on the
stance's close step, so per-category telemetry attributes a stance that straddles a resample to the
category it closed in.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

# --------------------------------------------------------------------------------------- constants
#: Stance opens above this load (N). Same as ``motion_knobs.FOOT_SLIP_FORCE_N`` / the clock's
#: ``TOUCH_REF_N``: about half the ~700 N per-foot stance share is real support, not a brush.
FORCE_OPEN_N = 50.0
#: A stance stays open, and a step can count as loaded, while the load is at least this (N).
FORCE_HOLD_N = 20.0
#: Consecutive steps below ``FORCE_HOLD_N`` that close a stance.
N_OFF = 2
#: Credited loaded steps a stance needs to be paid (0.1 s at dt 0.02).
MIN_LOADED_STEPS = 5
#: A stance that opens during the first ``RESET_TAINT_STEPS`` reward calls after a reset is never paid
#: (RSI frames that start mid-stance, and the settle after a reset).
RESET_TAINT_STEPS = 3

#: The gait clock's cadence law (``crab_hex_dimensions``: CLOCK_CMD_STOP_M_S / CLOCK_V_MAX_M_S /
#: CLOCK_F_MAX_REV_S): ``f = 0.5 * min(v / 0.35, 1)`` Hz above 0.2 m/s, frozen below.
CLOCK_CMD_STOP_M_S = 0.2
CLOCK_V_MAX_M_S = 0.35
CLOCK_F_MAX_REV_S = 0.5

#: A foot whose commanded velocity is slower than this (m/s) has no defined direction -> no credit.
MIN_FOOT_CMD_M_S = 0.05
#: Floor on the stroke normaliser (m); the ``v_clk / 2`` floor is the operative one on gated rows.
MIN_STROKE_NORM_M = 0.05
#: Floor on the own sweep in the slip ratio's denominator (m).
MIN_SWEEP_M = 0.01

#: Starting values (plan step 2); the replay gate (step 7) calibrates them.
RHO0 = 0.5
DRHO = 0.3
POWER = 2.0
C_MAX = 1.5

#: Progress guard: EMA time constant (s), and the tracking ratio below which a stance pays nothing /
#: above which it pays in full is ``PROG_LO`` / ``PROG_LO + PROG_WIDTH``. Every omni head's side ratio
#: (0.39-0.52) is above the top. The EMA must span a whole gait cycle. A tripod-locked rock at the clock's
#: 0.5 Hz swings a 2-s EMA to about +-0.25, which would still pay most of a stroke; a 4-s EMA swings to
#: about +-0.12, under ``PROG_LO``.
PROG_TAU_S = 4.0
PROG_LO = 0.15
PROG_WIDTH = 0.15
#: Per-step clamp on the tracking ratio before the EMA.
PROG_Q_MIN = -1.0
PROG_Q_MAX = 1.5

#: Centring defaults: EMA time constant (s, about one gait cycle), Gaussian width (m), the command
#: speed below which there is no travel direction, and the statue guard's dead zone and saturation
#: (on the EMA'd tracking ratio).
CENTRING_TAU_S = 2.0
CENTRING_SIGMA_M = 0.2
CENTRING_MIN_CMD_M_S = 0.12
CENTRING_PHI_LO = 0.10
CENTRING_PHI_SAT = 0.25

_EPS = 1e-6


# --------------------------------------------------------------------------------------- geometry
def heading_frame_xy(vec_xy: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame planar vectors into the heading frame, ``R_z(-heading) v``. ``vec_xy`` is ``[N, ..., 2]``."""
    shape = (-1,) + (1,) * (vec_xy.dim() - 2)
    c = torch.cos(heading).reshape(shape)
    s = torch.sin(heading).reshape(shape)
    return torch.stack((c * vec_xy[..., 0] + s * vec_xy[..., 1], -s * vec_xy[..., 0] + c * vec_xy[..., 1]), dim=-1)


def feet_heading_frame(foot_pos_w: torch.Tensor, root_pos_w: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Foot positions relative to the root in the heading frame, ``[N, F, 2]``.

    ``heading`` is the body yaw ``asset.data.heading_w`` reports, i.e. ``atan2(2(wz + xy), 1 - 2(y^2 + z^2))``
    for a ``(w, x, y, z)`` root quaternion -- NOT the eval npz's ``travel_yaw`` (the body-frame bearing of
    the command).
    """
    return heading_frame_xy(foot_pos_w[..., :2] - root_pos_w[:, None, :2], heading)


def clock_cadence_hz(v_clk: torch.Tensor) -> torch.Tensor:
    """The gait clock's cadence for a commanded clock speed, ``[N]`` (0 below the stop speed)."""
    f = (v_clk / CLOCK_V_MAX_M_S).clamp(max=1.0) * CLOCK_F_MAX_REV_S
    return torch.where(v_clk > CLOCK_CMD_STOP_M_S, f, torch.zeros_like(f))


def stroke_command(cmd: torch.Tensor, turning: torch.Tensor | None) -> torch.Tensor:
    """The command the stroke term projects on: ``cmd`` with ``wz`` zeroed on non-``turning`` rows.

    On a MOVE (non-turning) slot the yaw-rate channel is only the facing relay's P-control. Keeping it
    would rotate every foot's projection toward a turn the slot does not command. ``turning=None`` keeps
    ``wz`` everywhere.
    """
    if turning is None:
        return cmd
    out = cmd.clone()
    out[:, 2] = torch.where(turning.bool(), cmd[:, 2], torch.zeros_like(cmd[:, 2]))
    return out


def clock_speed(cmd: torch.Tensor, turn_radius: float) -> torch.Tensor:
    """``max(||v_xy||, turn_radius * |wz|)`` -- the omni gait clock's commanded speed (``clock_command_speed``)."""
    return torch.maximum(torch.linalg.norm(cmd[:, :2], dim=-1), turn_radius * cmd[:, 2].abs())


def foot_command_velocity(cmd: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Each foot's commanded velocity ``v_cmd + w_cmd x r_i`` in the heading frame, ``[N, F, 2]``."""
    vx, vy, wz = cmd[:, 0:1], cmd[:, 1:2], cmd[:, 2:3]
    return torch.stack((vx - wz * r[..., 1], vy + wz * r[..., 0]), dim=-1)


# --------------------------------------------------------------------------------------- stance edges
class StanceEdges(NamedTuple):
    opening: torch.Tensor  # [N, F] bool -- a closed foot's load crossed FORCE_OPEN_N this step
    closing: torch.Tensor  # [N, F] bool -- an open stance's N_OFF-th consecutive low step
    is_open: torch.Tensor  # [N, F] bool -- open this step (including the closing step)
    loaded: torch.Tensor  # [N, F] bool -- open, and loaded now and at the previous step
    low_run: torch.Tensor  # [N, F] long -- updated consecutive-low count
    ok: torch.Tensor  # [N, F] bool -- F >= FORCE_HOLD_N now


def loaded_stance_edges(
    force: torch.Tensor, open_: torch.Tensor, low_run: torch.Tensor, prev_ok: torch.Tensor, has_prev: torch.Tensor,
) -> StanceEdges:
    """Hysteresis stance edges from the current contact-force norm; no sensor timer is read (pure, no mutation).

    ``has_prev`` is ``[N]``: whether the previous step's reading belongs to this episode. Reusable by any
    term that needs loaded-stance edges immune to the float32 ``compute_first_*`` drift (e.g. air time).
    """
    ok = force >= FORCE_HOLD_N
    opening = ~open_ & (force > FORCE_OPEN_N)
    is_open = open_ | opening
    zero = torch.zeros_like(low_run)
    low_run = torch.where(opening, zero, torch.where(open_ & ~ok, low_run + 1, torch.where(open_, zero, low_run)))
    closing = open_ & (low_run >= N_OFF)
    loaded = is_open & ok & prev_ok & has_prev[:, None]
    return StanceEdges(opening, closing, is_open, loaded, low_run, ok)


# --------------------------------------------------------------------------------------- stroke term
class StrokeState(NamedTuple):
    """Per-env, per-foot state of :func:`foot_stroke_step`. Build with :func:`init_stroke_state`."""

    open: torch.Tensor  # [N, F] bool
    low_run: torch.Tensor  # [N, F] long
    prev_ok: torch.Tensor  # [N, F] bool
    tainted: torch.Tensor  # [N, F] bool -- the open stance began during the reset taint window
    n: torch.Tensor  # [N, F] long -- credited loaded steps in the current gated run
    B: torch.Tensor  # [N, F] -- signed own sweep (m)
    Ba: torch.Tensor  # [N, F] -- signed anchor sweep (m)
    W: torch.Tensor  # [N, F] -- world path of the foot over credited steps (m)
    Lsum: torch.Tensor  # [N, F] -- sum of the stroke normaliser over credited steps (m)
    m: torch.Tensor  # [N, F] -- credited steps behind Lsum (float)
    pB: torch.Tensor  # [N, F] -- pending (dip) own sweep
    pBa: torch.Tensor  # [N, F] -- pending anchor sweep
    pW: torch.Tensor  # [N, F] -- pending world path
    pL: torch.Tensor  # [N, F] -- pending normaliser sum
    pm: torch.Tensor  # [N, F] -- pending step count (float)
    prog: torch.Tensor  # [N, F] -- EMA of the foot point's tracking ratio
    prog_has: torch.Tensor  # [N, F] bool -- prog holds a real value
    prev_r: torch.Tensor  # [N, F, 2] -- heading-frame foot position at the previous step
    prev_p: torch.Tensor  # [N, F, 2] -- world foot xy at the previous step
    prev_root: torch.Tensor  # [N, 2] -- world root xy at the previous step
    steps: torch.Tensor  # [N] long -- reward calls since the last reset (0 on the first)
    has_prev: torch.Tensor  # [N] bool -- prev_r / prev_p hold a real previous step


_RUN_FLOAT = ("B", "Ba", "W", "Lsum", "m", "pB", "pBa", "pW", "pL", "pm")
_PENDING = ("pB", "pBa", "pW", "pL", "pm")


def init_stroke_state(num_envs: int, num_feet: int, device: torch.device | str = "cpu") -> StrokeState:
    nf = (num_envs, num_feet)
    fields = {}
    for name in StrokeState._fields:
        if name in ("open", "prev_ok", "tainted", "prog_has"):
            fields[name] = torch.zeros(nf, dtype=torch.bool, device=device)
        elif name in ("low_run", "n"):
            fields[name] = torch.zeros(nf, dtype=torch.long, device=device)
        elif name in ("prev_r", "prev_p"):
            fields[name] = torch.zeros(num_envs, num_feet, 2, device=device)
        elif name == "prev_root":
            fields[name] = torch.zeros(num_envs, 2, device=device)
        elif name == "steps":
            fields[name] = torch.zeros(num_envs, dtype=torch.long, device=device)
        elif name == "has_prev":
            fields[name] = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            fields[name] = torch.zeros(nf, device=device)
    return StrokeState(**fields)


def reset_stroke_state(state: StrokeState, env_ids=None) -> StrokeState:
    """Forget everything for ``env_ids`` (all envs when ``None``). An open stance is discarded UNPAID."""
    idx = slice(None) if env_ids is None else env_ids
    for t in state:
        t[idx] = False if t.dtype == torch.bool else 0
    return state


def _clear(state: StrokeState, mask: torch.Tensor, names=_RUN_FLOAT) -> None:
    for name in names:
        getattr(state, name).masked_fill_(mask, 0.0)


def foot_stroke_step(
    force: torch.Tensor,
    foot_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    heading: torch.Tensor,
    cmd: torch.Tensor,
    stand: torch.Tensor,
    root_lin_vel_b_xy: torch.Tensor,
    root_ang_vel_b_z: torch.Tensor,
    state: StrokeState,
    dt: float,
    turn_radius: float,
    turning: torch.Tensor | None = None,
    resampled: torch.Tensor | None = None,
    rho0: float = RHO0,
    drho: float = DRHO,
    power: float = POWER,
    c_max: float = C_MAX,
    prog_tau: float = PROG_TAU_S,
    prog_lo: float = PROG_LO,
    prog_width: float = PROG_WIDTH,
    info: dict | None = None,
) -> tuple[torch.Tensor, StrokeState, torch.Tensor]:
    """One env step of the per-foot stance-stroke income.

    Args:
        force: ``[N, F]`` current contact-force norm per foot (N).
        foot_pos_w: ``[N, F, >=2]`` world foot positions.
        root_pos_w: ``[N, >=2]`` world root position.
        heading: ``[N]`` body yaw (rad), ``asset.data.heading_w``.
        cmd: ``[N, 3]`` body-frame command (vx, vy, wz).
        stand: ``[N]`` bool, the STAND slot.
        root_lin_vel_b_xy / root_ang_vel_b_z: ``[N, 2]`` / ``[N]`` body velocity, for the progress guard.
        state: from :func:`init_stroke_state`; updated in place and returned.
        dt: env step (s).
        turn_radius: the omni clock's turn radius (``omni_clock_turn_radius``).
        turning: ``[N]`` bool, the slots that command a turn (TURN / MOVE_TURN); ``None`` = every row.
        resampled: ``[N]`` bool, the command was resampled this step: the progress guard re-seeds at its
            pass value for the new hold.
        info: when a dict is passed, it is filled with this step's per-foot stance internals BEFORE the
            closed stances are cleared (``closing``, ``payable``, ``B``, ``Ba``, ``W``, ``n``, ``mean_l``,
            ``slip_gate``, ``g_prog``, ``loaded``, ``credit``), for the offline readout. Training passes
            nothing.

    Returns:
        ``(reward [N], state, pay [N, F])``. ``reward = sum_i pay_i / dt``, so the manager's
        ``weight * dt`` pays ``weight * pay_i`` once per completed stance. ``pay`` is the per-foot
        stance pay on its close step (0 elsewhere), for telemetry and replay.
    """
    r = feet_heading_frame(foot_pos_w, root_pos_w, heading)
    p = foot_pos_w[..., :2]
    cmd_s = stroke_command(cmd, turning)
    v_clk = clock_speed(cmd_s, turn_radius)
    gate = (v_clk > CLOCK_CMD_STOP_M_S) & ~stand.bool()  # [N]
    f_hz = clock_cadence_hz(v_clk)
    v_i = foot_command_velocity(cmd_s, r)
    speed_i = torch.linalg.norm(v_i, dim=-1)
    u_i = v_i / speed_i.clamp_min(_EPS).unsqueeze(-1)
    stroke_norm = (torch.maximum(speed_i, 0.5 * v_clk[:, None]) / (2.0 * f_hz.clamp_min(_EPS))[:, None]).clamp_min(
        MIN_STROKE_NORM_M)
    valid = gate[:, None] & (speed_i > MIN_FOOT_CMD_M_S) & state.has_prev[:, None]

    e = loaded_stance_edges(force, state.open, state.low_run, state.prev_ok, state.has_prev)

    # A fresh stance starts from nothing.
    _clear(state, e.opening)
    state.n.masked_fill_(e.opening, 0)
    state.tainted.copy_(torch.where(e.opening, (state.steps < RESET_TAINT_STEPS)[:, None], state.tainted))

    # This step's increments. On turning rows the heading-frame displacement -dr includes the body's
    # rotation: that rotation IS the commanded motion. On non-turning rows both positions are taken in the
    # CURRENT heading frame, so body yaw (heading wiggle, drift) is never credited as a foot's stroke.
    root = root_pos_w[:, :2]
    dp = p - state.prev_p
    d_rot = -((r - state.prev_r) * u_i).sum(dim=-1)
    d_lin = -(heading_frame_xy(dp - (root - state.prev_root)[:, None, :], heading) * u_i).sum(dim=-1)
    rot_rows = torch.ones_like(gate) if turning is None else turning.bool()
    d = torch.where(rot_rows[:, None], d_rot, d_lin)
    d_a = d + (heading_frame_xy(dp, heading) * u_i).sum(dim=-1)
    dw = torch.linalg.norm(dp, dim=-1)
    z = torch.zeros_like(d)

    # Dips: open, not loaded, not opening or closing -> pending; added on the next credited step.
    gap = e.is_open & ~e.opening & ~e.closing & ~e.loaded & valid
    for name, inc in (("pB", d), ("pBa", d_a), ("pW", dw), ("pL", stroke_norm), ("pm", torch.ones_like(d))):
        getattr(state, name).add_(torch.where(gap, inc, z))
    credit = e.loaded & valid
    for name, pname, inc in (("B", "pB", d), ("Ba", "pBa", d_a), ("W", "pW", dw), ("Lsum", "pL", stroke_norm),
                             ("m", "pm", torch.ones_like(d))):
        getattr(state, name).add_(torch.where(credit, inc + getattr(state, pname), z))
    _clear(state, credit, _PENDING)
    state.n.add_(credit.long())

    # Only the current gated run counts: an ungated step (STAND, a stop command) wipes the stance's credit.
    ungated = e.is_open & ~gate[:, None]
    _clear(state, ungated)
    state.n.masked_fill_(ungated, 0)

    # Progress guard: EMA of each foot point's tracking ratio, on steps with a defined foot command. The
    # ratio is clamped (a foot near a turn's centre has a tiny commanded speed, which would otherwise let
    # one step swing the EMA). A fresh hold -- after a reset or a resample -- starts AT the pass value: a
    # guard that started cold would tax every honest hold's first seconds (every eval hold starts from
    # rest), and replay and training would then price different statistics.
    w_b = torch.where(rot_rows, root_ang_vel_b_z, torch.zeros_like(root_ang_vel_b_z))[:, None]
    v_pt = torch.stack((root_lin_vel_b_xy[:, None, 0] - w_b * r[..., 1], root_lin_vel_b_xy[:, None, 1] + w_b * r[..., 0]),
                       dim=-1)
    q = ((v_pt * v_i).sum(dim=-1) / (speed_i * speed_i).clamp_min(_EPS)).clamp(PROG_Q_MIN, PROG_Q_MAX)
    if resampled is not None:
        state.prog_has.logical_and_(~resampled.bool()[:, None])
    upd = gate[:, None] & (speed_i > MIN_FOOT_CMD_M_S)
    seed = torch.full_like(state.prog, prog_lo + prog_width)
    base = torch.where(state.prog_has, state.prog, seed)
    state.prog.copy_(torch.where(upd, base + (dt / prog_tau) * (q - base), state.prog))
    state.prog_has.logical_or_(upd)
    g_prog = ((state.prog - prog_lo) / prog_width).clamp(0.0, 1.0)

    # Pay at close.
    mean_l = state.Lsum / state.m.clamp_min(1.0)
    stroke = torch.minimum(state.B, state.Ba) / mean_l.clamp_min(MIN_STROKE_NORM_M)
    rho = state.W / state.B.clamp_min(MIN_SWEEP_M)
    slip_gate = torch.exp(-(((rho - rho0).clamp_min(0.0) / drho) ** 2))
    payable = e.closing & (state.n >= MIN_LOADED_STEPS) & ~state.tainted
    pay = torch.where(payable, g_prog * slip_gate * stroke.clamp(0.0, c_max).pow(power), z)
    if info is not None:
        info.update(closing=e.closing.clone(), payable=payable.clone(), B=state.B.clone(), Ba=state.Ba.clone(),
                    W=state.W.clone(), n=state.n.clone(), mean_l=mean_l.clone(), slip_gate=slip_gate,
                    g_prog=g_prog.clone(), loaded=e.loaded.clone(), credit=credit.clone())

    # Clear closed stances.
    _clear(state, e.closing)
    state.n.masked_fill_(e.closing, 0)
    state.tainted.masked_fill_(e.closing, False)
    state.open.copy_(e.is_open & ~e.closing)
    state.low_run.copy_(torch.where(e.closing, torch.zeros_like(e.low_run), e.low_run))

    state.prev_ok.copy_(e.ok)
    state.prev_r.copy_(r)
    state.prev_p.copy_(p)
    state.prev_root.copy_(root)
    state.steps.add_(1)
    state.has_prev.fill_(True)

    return pay.sum(dim=-1) / dt, state, pay


# --------------------------------------------------------------------------------------- centring term
class CentringState(NamedTuple):
    """Per-env state of :func:`body_centring_step`. Build with :func:`init_centring_state`."""

    c_bar: torch.Tensor  # [N, 2] -- EMA of the six-foot centroid, heading frame, relative to the root
    v_bar: torch.Tensor  # [N, 2] -- EMA of the body planar velocity, body frame
    has_prev: torch.Tensor  # [N] bool -- the EMAs hold real values (False after a reset or a resample)


def init_centring_state(num_envs: int, device: torch.device | str = "cpu") -> CentringState:
    return CentringState(
        c_bar=torch.zeros(num_envs, 2, device=device),
        v_bar=torch.zeros(num_envs, 2, device=device),
        has_prev=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_centring_state(state: CentringState, env_ids=None) -> CentringState:
    idx = slice(None) if env_ids is None else env_ids
    state.c_bar[idx] = 0.0
    state.v_bar[idx] = 0.0
    state.has_prev[idx] = False
    return state


def update_centroid_ema(
    foot_pos_w: torch.Tensor, root_pos_w: torch.Tensor, heading: torch.Tensor, root_lin_vel_b_xy: torch.Tensor,
    state: CentringState, dt: float, tau: float = CENTRING_TAU_S, resampled: torch.Tensor | None = None,
) -> CentringState:
    """Advance the centroid and velocity EMAs (``x += (dt / tau)(x_now - x)``).

    An env that is fresh -- after a reset, or on a command resample (``resampled``) -- starts at the
    instantaneous values, so a new hold never inherits the previous hold's posture.
    """
    if resampled is not None:
        state.has_prev.logical_and_(~resampled.bool())
    c = feet_heading_frame(foot_pos_w, root_pos_w, heading).mean(dim=1)
    v = root_lin_vel_b_xy[:, :2]
    a = dt / tau
    warm = state.has_prev[:, None]
    state.c_bar.copy_(torch.where(warm, state.c_bar + a * (c - state.c_bar), c))
    state.v_bar.copy_(torch.where(warm, state.v_bar + a * (v - state.v_bar), v))
    state.has_prev.fill_(True)
    return state


def body_lead(c_bar: torch.Tensor, cmd_xy: torch.Tensor) -> torch.Tensor:
    """Body lead over the six-foot centroid along the command, ``[N]`` (m): ``-c_bar . v_hat``.

    This IS the readout's pinned ``body_offset``: positive when the body sits ahead of its feet in the
    travel direction. 0 where the command has no direction.
    """
    speed = torch.linalg.norm(cmd_xy, dim=-1)
    v_hat = cmd_xy / speed.clamp_min(_EPS).unsqueeze(-1)
    return -(c_bar * v_hat).sum(dim=-1)


def body_centring_step(
    foot_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    heading: torch.Tensor,
    cmd: torch.Tensor,
    stand: torch.Tensor,
    root_lin_vel_b_xy: torch.Tensor,
    state: CentringState,
    dt: float,
    lat_share: torch.Tensor | None = None,
    resampled: torch.Tensor | None = None,
    sigma: float = CENTRING_SIGMA_M,
    tau: float = CENTRING_TAU_S,
    min_cmd: float = CENTRING_MIN_CMD_M_S,
    phi_lo: float = CENTRING_PHI_LO,
    phi_sat: float = CENTRING_PHI_SAT,
) -> tuple[torch.Tensor, CentringState, torch.Tensor]:
    """One env step of the body-centring bonus: ``b = lambda * g_phi * exp(-(o / sigma)^2)``.

    * ``o``: the body lead over the EMA'd six-foot centroid, along the command (:func:`body_lead`).
    * ``lambda``: the command's LATERAL share. It is taken from ``lat_share`` when given: the wrapper
      passes ``|sin(travel - facing)|`` of the hold's INTENDED bearing, so a heading error cannot steer
      it. Otherwise it is ``|v_y| / ||v_xy||`` of the body-frame command.
    * ``g_phi``: a statue guard. It ramps from 0 at an EMA'd tracking ratio of ``phi_lo`` to 1 at
      ``phi_sat``, below every omni head's side ratio (0.39-0.52), so it pays no speed in the operating
      range. It is EMA'd so a statue that rocks does not pass it.

    Rows with ``||v_xy|| <= min_cmd`` or the STAND slot pay 0 (stand, turn-in-place). Returns
    ``(b [N], state, o [N])``.
    """
    state = update_centroid_ema(foot_pos_w, root_pos_w, heading, root_lin_vel_b_xy, state, dt, tau, resampled)
    cmd_xy = cmd[:, :2]
    speed = torch.linalg.norm(cmd_xy, dim=-1)
    o = body_lead(state.c_bar, cmd_xy)
    lam = lat_share if lat_share is not None else cmd_xy[:, 1].abs() / speed.clamp_min(_EPS)
    phi = (state.v_bar * cmd_xy).sum(dim=-1) / (speed * speed).clamp_min(_EPS)
    g_phi = ((phi - phi_lo) / (phi_sat - phi_lo)).clamp(0.0, 1.0)
    active = (speed > min_cmd) & ~stand.bool()
    b = torch.where(active, lam * g_phi * torch.exp(-((o / sigma) ** 2)), torch.zeros_like(speed))
    return b, state, o
