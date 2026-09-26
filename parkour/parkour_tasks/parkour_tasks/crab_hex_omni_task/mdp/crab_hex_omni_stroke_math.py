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

:func:`foot_stroke_step` pays each foot's OWN loaded sweep:

* **Edges are term-internal**, from the current contact force with hysteresis:
  - a stance opens at ``F > FORCE_OPEN_N``;
  - it closes after ``N_OFF`` consecutive steps below ``FORCE_HOLD_N``;
  - a step is *loaded* only if the stance is open and ``F >= FORCE_HOLD_N`` at both t-1 and t.

  No sensor timer is read, so the float32 drift cannot touch it. A stale (repeated) force reading, which
  the contact sensor produces on about 1 env step in 5 between 16 and 32 s of an episode, can shift an
  edge by at most one step and can never create one.
* **Projection per foot.** Each foot's sweep is projected on its own commanded velocity
  ``v_i = v_cmd + w_cmd x r_i``, so turn-in-place and move+turn are paid by the same formula as
  translation.
* **Net stroke, not rectified sway.** The sweep is accumulated signed and clamped only on the stance
  total, so sway nets out.
* **Slip gate per stance.** ``rho = world path of the loaded foot / body-relative sweep``. A per-step gate
  was measured to RECTIFY side income (x1.31); the per-stance gate is monotone.
* **Normalised** to the gait clock's planned stroke ``L_i = |v_i| / (2 f)``, which is 0.35 m for any
  translation.
* **Paid once, at close**, as ``pay / dt``, so the manager's ``weight * dt`` delivers exactly
  ``weight * pay`` per stance: an event term without the dt dilution.
* **Gate: command only** -- ``clock speed > 0.2`` and not the STAND slot. It is valid whether or not the
  policy runs a gait clock.

:func:`body_centring_step` is the posture-arm lever: a dense bonus for keeping the body over its feet
along a LATERAL command. The omni heads lead their six-foot centroid by about 0.23 m on side holds; the
sideways-task head leads it by about 0.01 m and strokes long.

Both terms are bonuses (the reward manager clips the summed step reward at 0, so penalties would be
absorbed). Both are equivariant under the omni mirror group {I, M_y, M_x, R_z(pi)}: every quantity is a
dot product, a norm or ``|v_y|``, and the sum over feet does not depend on foot order.

No ``isaaclab`` import: the Isaac wrappers live in ``crab_hex_omni_stroke_rewards``, and the offline replay
tool path-loads this module, so replay prices exactly the arithmetic that trains.
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
#: Loaded steps a stance needs to be paid (0.1 s at dt 0.02).
MIN_LOADED_STEPS = 5
#: A stance that opens within this many env steps of a reset is never paid (RSI frames that start
#: mid-stance, and the settle after a reset).
RESET_TAINT_STEPS = 2

#: The gait clock's cadence law (``crab_hex_dimensions``: CLOCK_CMD_STOP_M_S / CLOCK_V_MAX_M_S /
#: CLOCK_F_MAX_REV_S): ``f = 0.5 * min(v / 0.35, 1)`` Hz above 0.2 m/s, frozen below.
CLOCK_CMD_STOP_M_S = 0.2
CLOCK_V_MAX_M_S = 0.35
CLOCK_F_MAX_REV_S = 0.5

#: A foot whose commanded velocity is slower than this (m/s) has no defined direction -> no credit.
MIN_FOOT_CMD_M_S = 0.05
#: Floor on the stroke normaliser (m).
MIN_STROKE_NORM_M = 0.05
#: Floor on the body-relative sweep in the slip ratio's denominator (m).
MIN_SWEEP_M = 0.01

#: Starting values (plan step 2); the replay gate (step 7) calibrates them.
RHO0 = 0.5
DRHO = 0.3
POWER = 2.0
C_MAX = 1.5

#: Centring defaults: EMA time constant (s, about one gait cycle), Gaussian width (m), the command
#: speed below which there is no travel direction, and the statue guard's saturation ratio.
CENTRING_TAU_S = 2.0
CENTRING_SIGMA_M = 0.2
CENTRING_MIN_CMD_M_S = 0.12
CENTRING_PHI_SAT = 0.25

_EPS = 1e-6


# --------------------------------------------------------------------------------------- geometry
def heading_frame_xy(foot_pos_w: torch.Tensor, root_pos_w: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Foot positions relative to the root in the heading frame: ``R_z(-heading) (p_i - p_root)_xy``.

    Args:
        foot_pos_w: ``[N, F, >=2]`` world foot positions.
        root_pos_w: ``[N, >=2]`` world root position.
        heading: ``[N]`` body yaw (rad), the same yaw ``asset.data.heading_w`` reports.

    Returns:
        ``[N, F, 2]``.
    """
    d = foot_pos_w[..., :2] - root_pos_w[:, None, :2]
    c = torch.cos(heading)[:, None]
    s = torch.sin(heading)[:, None]
    return torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), dim=-1)


def clock_cadence_hz(v_clk: torch.Tensor) -> torch.Tensor:
    """The gait clock's cadence for a commanded clock speed, ``[N]`` (0 below the stop speed)."""
    f = (v_clk / CLOCK_V_MAX_M_S).clamp(max=1.0) * CLOCK_F_MAX_REV_S
    return torch.where(v_clk > CLOCK_CMD_STOP_M_S, f, torch.zeros_like(f))


def foot_command_velocity(cmd: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Each foot's commanded velocity ``v_cmd + w_cmd x r_i`` in the body/heading frame, ``[N, F, 2]``.

    ``cmd`` is ``[N, 3]`` = (vx, vy, wz), body frame; ``r`` is ``[N, F, 2]``.
    """
    vx, vy, wz = cmd[:, 0:1], cmd[:, 1:2], cmd[:, 2:3]
    return torch.stack((vx - wz * r[..., 1], vy + wz * r[..., 0]), dim=-1)


# --------------------------------------------------------------------------------------- stroke term
class StrokeState(NamedTuple):
    """Per-env, per-foot state of :func:`foot_stroke_step`. Build with :func:`init_stroke_state`."""

    open: torch.Tensor  # [N, F] bool -- a stance is open
    low_run: torch.Tensor  # [N, F] long -- consecutive steps below FORCE_HOLD_N while open
    prev_ok: torch.Tensor  # [N, F] bool -- F >= FORCE_HOLD_N at the previous step
    tainted: torch.Tensor  # [N, F] bool -- the open stance began within RESET_TAINT_STEPS of a reset
    n: torch.Tensor  # [N, F] long -- loaded steps in the open stance
    C: torch.Tensor  # [N, F] -- normalised signed sweep, sum(d / L)
    B: torch.Tensor  # [N, F] -- signed body-relative sweep (m)
    W: torch.Tensor  # [N, F] -- world path length of the loaded foot (m)
    prev_r: torch.Tensor  # [N, F, 2] -- heading-frame foot position at the previous step
    prev_p: torch.Tensor  # [N, F, 2] -- world foot xy at the previous step
    steps: torch.Tensor  # [N] long -- env steps since the last reset (0 on the first step)
    has_prev: torch.Tensor  # [N] bool -- prev_r / prev_p hold a real previous step


def init_stroke_state(num_envs: int, num_feet: int, device: torch.device | str = "cpu") -> StrokeState:
    z = torch.zeros(num_envs, num_feet, device=device)
    zb = torch.zeros(num_envs, num_feet, dtype=torch.bool, device=device)
    zl = torch.zeros(num_envs, num_feet, dtype=torch.long, device=device)
    return StrokeState(
        open=zb.clone(), low_run=zl.clone(), prev_ok=zb.clone(), tainted=zb.clone(), n=zl.clone(),
        C=z.clone(), B=z.clone(), W=z.clone(),
        prev_r=torch.zeros(num_envs, num_feet, 2, device=device),
        prev_p=torch.zeros(num_envs, num_feet, 2, device=device),
        steps=torch.zeros(num_envs, dtype=torch.long, device=device),
        has_prev=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_stroke_state(state: StrokeState, env_ids=None) -> StrokeState:
    """Forget everything for ``env_ids`` (all envs when ``None``). An open stance is discarded UNPAID."""
    idx = slice(None) if env_ids is None else env_ids
    for name, t in state._asdict().items():
        t[idx] = False if t.dtype == torch.bool else 0
    return state


def foot_stroke_step(
    force: torch.Tensor,
    foot_pos_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    heading: torch.Tensor,
    cmd: torch.Tensor,
    stand: torch.Tensor,
    v_clk: torch.Tensor,
    state: StrokeState,
    dt: float,
    rho0: float = RHO0,
    drho: float = DRHO,
    power: float = POWER,
    c_max: float = C_MAX,
) -> tuple[torch.Tensor, StrokeState, torch.Tensor]:
    """One env step of the per-foot stance-stroke income.

    Args:
        force: ``[N, F]`` current contact-force norm per foot (N).
        foot_pos_w: ``[N, F, >=2]`` world foot positions.
        root_pos_w: ``[N, >=2]`` world root position.
        heading: ``[N]`` body yaw (rad).
        cmd: ``[N, 3]`` body-frame command (vx, vy, wz).
        stand: ``[N]`` bool, the STAND slot.
        v_clk: ``[N]`` the clock's commanded speed, ``max(||v_xy||, r_turn |wz|)`` (``_clock_cmd_speed``).
        state: from :func:`init_stroke_state`; updated in place and returned.
        dt: env step (s).

    Returns:
        ``(reward [N], state, pay [N, F])``. ``reward = sum_i pay_i / dt``, so the manager's
        ``weight * dt`` pays ``weight * pay_i`` once per completed stance. ``pay`` is the per-foot
        stance pay on its close step (0 elsewhere), for telemetry and replay.
    """
    r = heading_frame_xy(foot_pos_w, root_pos_w, heading)
    p = foot_pos_w[..., :2]

    gate = (v_clk > CLOCK_CMD_STOP_M_S) & ~stand.bool()  # [N]
    f_hz = clock_cadence_hz(v_clk)  # [N]
    v_i = foot_command_velocity(cmd, r)  # [N, F, 2]
    speed_i = torch.linalg.norm(v_i, dim=-1)  # [N, F]
    u_i = v_i / speed_i.clamp_min(_EPS).unsqueeze(-1)
    stroke_norm = torch.where(
        gate[:, None],
        (speed_i / (2.0 * f_hz.clamp_min(_EPS))[:, None]).clamp_min(MIN_STROKE_NORM_M),
        torch.full_like(speed_i, MIN_STROKE_NORM_M),
    )

    ok = force >= FORCE_HOLD_N
    was_open = state.open

    # Open: a closed foot crossing the open threshold starts a fresh stance.
    opening = ~was_open & (force > FORCE_OPEN_N)
    for acc in (state.C, state.B, state.W):
        acc.masked_fill_(opening, 0.0)
    state.n.masked_fill_(opening, 0)
    state.low_run.masked_fill_(opening, 0)
    state.tainted.copy_(torch.where(opening, (state.steps <= RESET_TAINT_STEPS)[:, None], state.tainted))
    is_open = was_open | opening

    # Hold / close: an open foot below the hold threshold for N_OFF consecutive steps closes.
    low = was_open & ~ok
    state.low_run.copy_(torch.where(low, state.low_run + 1, torch.where(was_open, torch.zeros_like(state.low_run), state.low_run)))
    closing = was_open & (state.low_run >= N_OFF)

    # Accumulate on loaded steps: open, loaded now and at the previous step, with a real previous step.
    loaded = is_open & ok & state.prev_ok & state.has_prev[:, None]
    credit = loaded & gate[:, None] & (speed_i > MIN_FOOT_CMD_M_S)
    d = -((r - state.prev_r) * u_i).sum(dim=-1)  # signed body-relative sweep along the foot's command
    dw = torch.linalg.norm(p - state.prev_p, dim=-1)
    state.n.add_(loaded.long())
    state.C.add_(torch.where(credit, d / stroke_norm, torch.zeros_like(d)))
    state.B.add_(torch.where(credit, d, torch.zeros_like(d)))
    state.W.add_(torch.where(credit, dw, torch.zeros_like(dw)))

    # Pay at close.
    rho = state.W / state.B.clamp_min(MIN_SWEEP_M)
    slip_gate = torch.exp(-(((rho - rho0).clamp_min(0.0) / drho) ** 2))
    stroke = state.C.clamp(min=0.0, max=c_max)
    payable = closing & (state.n >= MIN_LOADED_STEPS) & ~state.tainted
    pay = torch.where(payable, slip_gate * stroke.pow(power), torch.zeros_like(stroke))

    # Clear closed stances.
    for acc in (state.C, state.B, state.W):
        acc.masked_fill_(closing, 0.0)
    state.n.masked_fill_(closing, 0)
    state.low_run.masked_fill_(closing, 0)
    state.tainted.masked_fill_(closing, False)
    state.open.copy_(is_open & ~closing)

    state.prev_ok.copy_(ok)
    state.prev_r.copy_(r)
    state.prev_p.copy_(p)
    state.steps.add_(1)
    state.has_prev.fill_(True)

    return pay.sum(dim=-1) / dt, state, pay


# --------------------------------------------------------------------------------------- centring term
class CentringState(NamedTuple):
    """Per-env state of :func:`body_centring_step`. Build with :func:`init_centring_state`."""

    c_bar: torch.Tensor  # [N, 2] -- EMA of the six-foot centroid, heading frame, relative to the root
    has_prev: torch.Tensor  # [N] bool -- c_bar holds a real value (False right after a reset)


def init_centring_state(num_envs: int, device: torch.device | str = "cpu") -> CentringState:
    return CentringState(
        c_bar=torch.zeros(num_envs, 2, device=device),
        has_prev=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_centring_state(state: CentringState, env_ids=None) -> CentringState:
    idx = slice(None) if env_ids is None else env_ids
    state.c_bar[idx] = 0.0
    state.has_prev[idx] = False
    return state


def update_centroid_ema(
    foot_pos_w: torch.Tensor, root_pos_w: torch.Tensor, heading: torch.Tensor,
    state: CentringState, dt: float, tau: float = CENTRING_TAU_S,
) -> CentringState:
    """Advance the centroid EMA ``c_bar += (dt / tau)(c - c_bar)``; a fresh env starts at ``c``."""
    c = heading_frame_xy(foot_pos_w, root_pos_w, heading).mean(dim=1)
    ema = state.c_bar + (dt / tau) * (c - state.c_bar)
    state.c_bar.copy_(torch.where(state.has_prev[:, None], ema, c))
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
    sigma: float = CENTRING_SIGMA_M,
    tau: float = CENTRING_TAU_S,
    min_cmd: float = CENTRING_MIN_CMD_M_S,
    phi_sat: float = CENTRING_PHI_SAT,
) -> tuple[torch.Tensor, CentringState, torch.Tensor]:
    """One env step of the body-centring bonus: ``b = lambda * g_phi * exp(-(o / sigma)^2)``.

    * ``o``: the body lead over the EMA'd six-foot centroid, along the command (:func:`body_lead`).
    * ``lambda = |v_y| / ||v_xy||``: the command's LATERAL share -- 1 on side, about 0.71 on diag, 0 on
      fwd/back.
    * ``g_phi = clamp(phi / phi_sat, 0, 1)`` with ``phi = v_b . v_cmd / ||v_cmd||^2``: a statue guard. It
      saturates below every omni head's measured side ratio (0.39-0.52), so it pays no speed in the
      operating range.

    Rows with ``||v_xy|| <= min_cmd`` or the STAND slot pay 0 (stand, turn-in-place). Returns
    ``(b [N], state, o [N])``.
    """
    state = update_centroid_ema(foot_pos_w, root_pos_w, heading, state, dt, tau)
    cmd_xy = cmd[:, :2]
    speed = torch.linalg.norm(cmd_xy, dim=-1)
    o = body_lead(state.c_bar, cmd_xy)
    lam = cmd_xy[:, 1].abs() / speed.clamp_min(_EPS)
    phi = (root_lin_vel_b_xy[:, :2] * cmd_xy).sum(dim=-1) / (speed * speed).clamp_min(_EPS)
    g_phi = (phi / phi_sat).clamp(0.0, 1.0)
    active = (speed > min_cmd) & ~stand.bool()
    b = torch.where(active, lam * g_phi * torch.exp(-((o / sigma) ** 2)), torch.zeros_like(speed))
    return b, state, o
