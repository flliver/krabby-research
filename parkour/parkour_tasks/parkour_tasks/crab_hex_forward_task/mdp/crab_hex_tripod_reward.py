"""Pure-torch tripod-alternation reward math for the crab hexapod (Milestone 18 Task 1 follow-on).

Config-only reward-weight/param sweeps (`parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/`) found
that no combination of existing terms -- a stance-count bracket at exactly 3, air-time weight/
threshold, a signed forward-pitch penalty at two doses, or angular-velocity damping -- meaningfully
moved the gait-eval harness's `tripod_score` (`gait_eval/metrics.py`) within a 1000-iteration
fine-tune from a checkpoint that already scores 0.401 on it. This term is the explicit
contact-schedule reward anticipated by Task 1 §2.3 ("reward alternating 3-foot stance sets... only
if air-time/stride do not produce regular phasing").

It is a dense, per-step proxy for the same quantity the eval metric measures over a whole steady
window: `tripod_score = 0.5 * (coh_A + coh_B) * max(0, -corr(a, b))`, where `a`/`b` are the
per-step stance counts of tripod sets A = {FL, MR, RL} and B = {FR, ML, RR} (`TRIPOD_A`/`TRIPOD_B`
in `gait_eval/metrics.py`) -- a real tripod alternates the two sets as units, so a good gait has
both high within-set coherence (both feet-in-a-set touch down/lift off together) and strong
anti-correlation between the sets (one is planted while the other swings).

Per step: each foot's raw contact state is treated as "in stable contact" only once it has
persisted continuously for ``debounce_s`` (matches the ``last_contact_time`` flicker-rejection
role in ``crab_hex_stride_reward.stride_length_reward_step``, but applied per-step rather than
only at liftoff, since this term needs a live stance count every step, not just at phase
transitions). ``a``/``b`` are the counts of stably-contacting feet in each tripod set (0-3).
Coherence is `1.0` only when a set is fully planted (3) or fully airborne (0) -- anything in
between (a leg out of sync with its own tripod) scores 0 for that set. The shape reward
`c_A * c_B * |a - b| / 3` is maximal (`1.0`) exactly when one tripod is fully planted and the
other fully airborne.

A pure state-coherence reward would let the policy farm reward by freezing in one tripod stance
forever (that state is coherent and maximally "opposed" even though nothing is alternating) -- the
eval's anti-correlation term is what actually penalizes that, since a constant `a`/`b` has zero
variance and thus zero correlation, but `max(0, -corr)` degenerately reports 0 for a frozen signal
(see `tripod_window_metrics`'s ``degenerate_anti_phase`` handling), not a reward. This module's
anti-freeze gate reproduces that requirement directly: it tracks which tripod set is currently
"dominant" (the sign of `a - b`, only once the count gap is at least 2 and that gap's sign has
itself persisted for ``min_swap_interval`` -- debouncing the *swap* separately from debouncing
individual foot contacts) and zeroes the reward once the dominant set has held for longer than
``max_hold_s`` without a confirmed swap. Genuine alternation keeps resetting the hold timer and
keeps the reward flowing; a frozen stance does not.

Reward is zero whenever the commanded planar speed is below ``min_cmd_norm`` (matches
``reward_forward_progress_along_command`` and the stride-length term's gating -- no reward for
standing still or pure in-place turning, where "tripod gait" isn't a meaningful concept).

**v2 addendum (in-band stance-count support bonus).** v1 of this term -- just the shape reward
and anti-freeze gate above -- caused a catastrophic regression when trained from scratch (see
`parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/RESULTS.md`): the policy converged on a unison
lunging gait (all legs moving together, flight phases, 0.488m strides, half the eval episodes
ending in falls). Two structural flaws enabled that: (1) unison motion keeps ``a == b`` at all
times, so ``|a - b| = 0`` and v1 scored it exactly 0 -- no gradient *away* from the exploit, only
a failure to reward it; (2) the anti-freeze gate zeroes ``r_shape`` for any never-swapping policy
within 0.6s of episode start, so even a policy drifting *toward* alternation earned nothing until
it produced a full confirmed swap -- a coordination cliff v1 never climbed.

v2 adds the Task 1 §2.3 stance-count band constraint ("penalize stance counts outside {3, 4}
during commanded motion") -- but as its reward-dual, an in-band **support bonus**, because
``ParkourRewardManager.compute`` clips the summed per-step reward at zero (legged-gym
convention): a literal penalty would be silently clipped away exactly during the exploit's
flight/all-down phases, where the other terms already sum non-positive. Under PPO advantage
normalization the bonus is the same shaping as the penalty, but it survives the clip:

    support = 1 - relu(3 - total)/3 - relu(total - 4)/2   (total = a + b, debounced)

i.e. count 0 (flight) -> 0, 1 -> 1/3, 2 -> 2/3, counts 3-4 (proper tripod / double-stance) -> 1,
5 -> 1/2, 6 (all-down) -> 0. The asymmetric normalization pins *both* halves of the unison
exploit at exactly 0, and the graded ramp at counts 1/2/5 is the mechanism that gives the
escape path a slope (a binary in-band indicator would recreate v1's flat-zero surface).
Critically, the support term is **not** gated by the anti-freeze timer -- a policy drifting
toward keeping one tripod planted earns strictly increasing reward from the first step, no
confirmed swap required. The accepted trade-off is that a parked in-band stance earns
``support_scale`` from this term indefinitely; the tracking/forward-progress/foot-idle terms
are what make parking a net loss overall.

**v3 addendum (body-stability gate on the support bonus).** v2's from-scratch test converged on
a tip-over-and-correct unison gait (see the campaign RESULTS.md): the support bonus fixed v1's
falls (100% completion) but counts planted *feet*, not body *attitude* -- a tipping robot passes
through 3-4-feet-down configurations and farms the bonus while rocking all six legs in unison
(tripod 0.0 throughout, roll_rms collapsed to 0.014 vs the healthy 0.038). v3 multiplies the
support half (only) by a stability gate. The gate signal was chosen from measured eval data, not
intuition -- pitch magnitude cannot discriminate (degenerate pitch_rms 0.2095 vs healthy 0.211,
both dominated by the healthy gait's own sustained ~12deg lean) and body angular rates are
*anti*-discriminative (healthy ``|w_xy|`` 0.72 vs degenerate 0.55 rad/s -- the rocking is
smoother than walking's leg-cycle jitter). The clean separator is EMA-smoothed world-frame
vertical speed: healthy episodes p95 <= 0.237 m/s vs v2-degenerate p05 >= 0.362 (no overlap;
cross-validated against v1's lunge). Causally sound: every exploit found so far pumps the CoM
vertically, while a tripod gait's whole point is keeping it level.

    s = EMA(|v_z_world|, tau=0.5s);   gate = clamp((hi - s) / (hi - lo), 0, 1)

with ``(lo, hi) = (0.20, 0.50)``: a clamped linear ramp, not a binary threshold (advantage-noise
cliff + v1-style flat-zero surface) and not an exponential (which would apply shaping pressure
inside the healthy band). The flat-1 shoulder above the healthy band means the healthy gait's
own vertical dynamics are completely unpenalized (measured transmission 0.996), while the ramp
keeps a nonzero slope across most of the degenerate EMA range (0.36-0.62) so an exploit that
bounces less earns measurably more -- an escape slope, not a cliff. The EMA state must reset to
0 (gate fully open) on episode reset, or freshly-reset envs would inherit a closed gate and
healthy post-reset exploration would go unrewarded.

**v4 addendum (event-based swap credit; support bonus and v_z gate REMOVED).** Four consecutive
from-scratch failures told one story (see the campaign RESULTS.md): v1 -> unison lunge, v2 ->
tip-and-correct rock, v3 -> level-bodied skate-shuffle, v3b (v3 + a feet_slide penalty) ->
near-stationary drag. Each version's gate eliminated its target behavior, and each time the
policy relocated to the cheapest remaining **state-holding** strategy that satisfied the current
gate set -- because the support bonus paid for holdable *states*, and its clip-immune income
stream was the constant across every exploit (while penalties like feet_slide were muted by the
manager's zero-floor clip exactly in those basins). Meanwhile the plain baked config reaches
tripod 0.34-0.40 emergently, earning its income from *motion* (tracking, forward progress).

v4 therefore abandons state-based income entirely. The support bonus and its v_z stability gate
are removed (nothing worth gating remains), and the term's income becomes **event-based**: a
lump credit is paid exactly on the step a dominant-set swap is *confirmed* (the same debounced
detection the anti-freeze timer already uses -- sign(a-b) with |a-b| >= 2 persisting
``min_swap_interval``), scaled by the opposition quality ``|a-b|/3`` at the confirm step. A swap
physically requires lifting one tripod set and planting the other; no static configuration --
frozen stance, all-down statue, flight, level drag -- produces confirmed swaps, so no holdable
state earns anything. And "farming" swap events faster is not an exploit: rapid genuine
alternation of the tripod sets *is* the target behavior. The per-step shape channel
(``r_shape * anti_freeze``) is kept unchanged -- it pays during sustained coherent alternation
and remains worth 0 to every degenerate family observed. ``swap_credit`` (default 15.0) sizes
the lump so that at a healthy alternation cadence (~3 swaps/s) the event income is comparable to
what the shape channel pays during perfect alternation. Unlike v1-v3, this term no longer tries
to *create* phasing from nothing; it amplifies and sharpens the alternation the base config
already produces emergently.

**v5 addendum (amplitude- and anti-correlation-qualified crossing credit; v4 machinery
REPLACED).** v4's screen produced falls by iteration 5000 -- but the decisive autopsy came from a
new mandatory gate: replaying the reward function offline over saved gait-eval npz traces
(``parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-10_0058_tripod_stability/offline_replay/``). That replay showed v4's
income structure was *inverted*: the healthy baseline earned exactly 0.000/min (its ~0.30s-period
gait has raw contact bouts of median 0.10s, so the 0.08s debounce erased its stance sets, the
swap detector never fired, and the never-fed anti-freeze timer kept ``r_shape`` at zero too),
while the only traces earning anything were the v2 tip-rock (1.2/min) and v3 skate (1.8/min),
whose slow deliberate unison transitions are precisely what a 0.1s-persistence detector can see.
A parameter grid over debounce/persistence found no robust operating point: every setting either
stays blind to the healthy gait or pays the slow degenerates.

Trace measurements that drove the v5 design: the healthy gait's alternation is *fast* (stride
period ~0.30s) and *one-sided* -- tripod set B carries the body (b >= 2 for ~62% of steady
steps) while set A is fully airborne ~75% of the time and fully planted only ~0.6%, which is
exactly the "not stably supported by its planted legs" problem this campaign targets. So the
reward must (a) operate at the stride timescale on *raw* contacts, no debounce; (b) pay the
baseline's real-but-shallow alternation something; and (c) place its maximum at deep, symmetric,
full-set alternation. v5 does this with an **event credit per zero-crossing** of the smoothed
support difference ``x = s_A - s_B`` (fast EMA, ``ema_tau``): a crossing is the moment support
duty actually transfers between the sets. Credit per crossing is

    min(prev_peak, peak) * q_anti^2        (0 if the swing peak < ``min_amp``)

where ``prev_peak``/``peak`` are the max ``|x|`` reached on either side of the crossing (both
sets must genuinely take and give up support -- deeper swings pay more, with the maximum at full
alternation), and ``q_anti = clamp(-corr(s_A, s_B), 0, 1)`` from EMA moments at ``corr_tau``
(stride-matched, NOT slower): the two sets must move in *opposition*, which is what separates
walking from the v3 skate whose sets chatter *together* (its |a-b| noise still crosses zero, but
its correlation is positive, so q_anti == 0). Credit is paid only when the time since the
previous crossing lies in ``[min_period, max_period]`` -- a band-pass on gait period that
excludes both contact chatter and the slow (~0.67s) weight-shift oscillation of the v3b drag.
Static configurations produce no crossings at all; unison motion keeps ``x ~ 0`` (fails
``min_amp`` and q_anti); tip-rock moves the sets together (q_anti == 0). Replay-gate result:
healthy refs earn 1.4-1.9/min (weighted, at weight 0.15), all four degenerate basins earn
<= 0.013/min except the semi-healthy v4-falls trace at 0.111/min, and an ideal synthetic tripod
(0.30s period, full sets, brief double-support) earns 36.6/min -- the optimum sits at the target
behavior with a smooth amplitude slope from the baseline's shallow taps toward it. Event
conditioning is also better than v4's: frequent small lumps (~0.07 credit at ~2.5 Hz on the
baseline) rather than rare 15.0 spikes.

State is packed in a single ``[N, STATE_DIM]`` tensor (see the ``S_*`` index constants); reset
must zero every column and then set the ``S_T_SINCE`` column to ``RESET_T_SINCE`` so the first
crossing after a reset is never in-band (no free credit at episode start).

See ``RewardTripodSchedule`` in ``parkour_isaaclab/envs/mdp/rewards.py`` for the stateful
``ManagerTermBase`` wrapper that drives this from real env/sensor data (raw per-foot contact
from the contact sensor -- v5 deliberately uses undebounced contact, see above); this module
stays free of any ``isaaclab`` import so it can be unit-tested without Isaac Sim, matching
``crab_hex_stride_reward.py``'s own isolation pattern.
"""

from __future__ import annotations

from typing import Sequence

import torch

#: Foot-order convention shared with ``gait_eval/metrics.py``'s ``FOOT_ORDER`` and
#: ``_CRAB_FOOT_BODY_NAMES`` in ``parkour_mdp_cfg.py`` -- (FL, FR, ML, MR, RL, RR).
TRIPOD_A_IDX: tuple[int, ...] = (0, 3, 4)
"""Index positions of tripod set A = {FL, MR, RL} within the 6-foot ``FOOT_ORDER`` convention."""
TRIPOD_B_IDX: tuple[int, ...] = (1, 2, 5)
"""Index positions of tripod set B = {FR, ML, RR} within the 6-foot ``FOOT_ORDER`` convention."""

# v5 state-tensor layout: one row per env, columns indexed by the S_* constants below.
S_SA, S_SB = 0, 1  #: fast EMAs of the two sets' support fractions (a/3, b/3)
S_MA, S_MB, S_MAA, S_MBB, S_MAB = 2, 3, 4, 5, 6  #: stride-timescale EMA moments of (s_A, s_B)
S_SIGN = 7  #: sign of x = s_A - s_B at the last step it was nonzero (-1/0/+1)
S_T_SINCE = 8  #: seconds since the last zero-crossing of x
S_PEAK = 9  #: running max |x| since the last crossing
S_PREV_PEAK = 10  #: max |x| over the swing before the last crossing
STATE_DIM = 11
RESET_T_SINCE = 1.0e6
"""Post-reset value for ``S_T_SINCE``: large, so the first crossing is never in the period band."""


def tripod_swap_crossing_reward_step(
    contact: torch.Tensor,
    command_xy: torch.Tensor,
    state: torch.Tensor,
    dt: float,
    tripod_a_idx: Sequence[int] = TRIPOD_A_IDX,
    tripod_b_idx: Sequence[int] = TRIPOD_B_IDX,
    min_cmd_norm: float = 0.12,
    ema_tau: float = 0.06,
    corr_tau: float = 0.20,
    min_period: float = 0.10,
    max_period: float = 0.60,
    min_amp: float = 0.15,
    var_min: float = 0.01,
    credit_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of the v5 crossing-credit reward (see the v5 addendum in the module docstring).

    Args:
        contact: ``[N, 6]`` bool -- raw per-foot contact this step, in ``FOOT_ORDER``. v5
            deliberately uses undebounced contact: the healthy gait's stance bouts (median
            ~0.10s) are shorter than any useful debounce window.
        command_xy: ``[N, 2]`` commanded planar velocity (body frame).
        state: ``[N, STATE_DIM]`` state in, laid out per the ``S_*`` constants. Fresh envs must
            be all-zero except ``S_T_SINCE = RESET_T_SINCE``.
        dt: physics step duration (s).
        tripod_a_idx: index positions (into the 6-foot axis) of tripod set A.
        tripod_b_idx: index positions (into the 6-foot axis) of tripod set B.
        min_cmd_norm: below this commanded planar speed there's no defined gait direction, so no
            credit (matches ``reward_forward_progress_along_command``).
        ema_tau: smoothing timescale for the per-set support fractions -- just enough to reject
            single-step contact chatter without hiding the ~0.30s stride rhythm.
        corr_tau: timescale of the EMA moments behind ``q_anti``. Stride-matched on purpose: a
            slower window (the 1.0s tried first in the offline gate) averages the fast healthy
            alternation away entirely while resonating with slow degenerate weight-shifts.
        min_period: crossings closer together than this earn nothing (contact chatter).
        max_period: crossings farther apart than this earn nothing (slow weight-shift
            oscillations, e.g. the v3b drag at ~0.67s).
        min_amp: if the swing peak ``|x|`` since the last crossing is below this, the crossing
            earns nothing (rejects unison gaits whose support difference only wiggles).
        var_min: if either set's support variance (EMA moments) is below this, ``q_anti`` is 0 --
            a degenerate constant signal has no defined correlation (mirrors the eval metric's
            ``degenerate_anti_phase`` handling).
        credit_scale: multiplier on the per-crossing credit; the natural credit is already in
            ``[0, 1]`` (``min(prev_peak, peak) * q_anti^2``).

    Returns:
        ``(reward[N], new_state[N, STATE_DIM])``. Reward is nonzero only on crossing steps.
    """
    a = contact[:, list(tripod_a_idx)].float().sum(dim=1) / 3.0
    b = contact[:, list(tripod_b_idx)].float().sum(dim=1) / 3.0

    alpha = dt / ema_tau
    alpha_c = dt / corr_tau
    s_a = state[:, S_SA] + alpha * (a - state[:, S_SA])
    s_b = state[:, S_SB] + alpha * (b - state[:, S_SB])
    m_a = state[:, S_MA] + alpha_c * (s_a - state[:, S_MA])
    m_b = state[:, S_MB] + alpha_c * (s_b - state[:, S_MB])
    m_aa = state[:, S_MAA] + alpha_c * (s_a * s_a - state[:, S_MAA])
    m_bb = state[:, S_MBB] + alpha_c * (s_b * s_b - state[:, S_MBB])
    m_ab = state[:, S_MAB] + alpha_c * (s_a * s_b - state[:, S_MAB])

    x = s_a - s_b
    t_since = state[:, S_T_SINCE] + dt
    sign_prev = state[:, S_SIGN]
    peak = state[:, S_PEAK]
    prev_peak = state[:, S_PREV_PEAK]

    s = torch.sign(x)
    crossing = (s != 0.0) & (sign_prev != 0.0) & (s != sign_prev)

    # Anti-correlation quality from the stride-timescale moments; zero when either signal is
    # (near-)constant, where correlation is undefined and a frozen stance must not earn.
    var_a = m_aa - m_a * m_a
    var_b = m_bb - m_b * m_b
    cov = m_ab - m_a * m_b
    denom = torch.sqrt(torch.clamp(var_a, min=1e-8) * torch.clamp(var_b, min=1e-8))
    q_anti = torch.clamp(-cov / denom, min=0.0, max=1.0)
    q_anti = torch.where(
        (var_a < var_min) | (var_b < var_min), torch.zeros_like(q_anti), q_anti
    )

    # Both swings around the crossing must be real: the smaller of the two peaks scales the
    # credit (deeper, more symmetric alternation pays more, maxing out at full-set exchange).
    amp_ok = peak >= min_amp
    credit = torch.minimum(prev_peak, peak) * q_anti * q_anti * credit_scale
    credit = torch.where(amp_ok, credit, torch.zeros_like(credit))

    in_band = (t_since >= min_period) & (t_since <= max_period)
    cmd_active = torch.norm(command_xy, dim=1) > min_cmd_norm
    reward = credit * (crossing & in_band & cmd_active).float()

    # Crossing bookkeeping happens regardless of whether the credit was paid (an out-of-band or
    # low-quality crossing still starts a new swing).
    new_prev_peak = torch.where(crossing, peak, prev_peak)
    new_peak = torch.where(crossing, torch.zeros_like(peak), peak)
    new_t_since = torch.where(crossing, torch.zeros_like(t_since), t_since)
    new_sign = torch.where(s != 0.0, s, sign_prev)
    new_peak = torch.maximum(new_peak, x.abs())

    new_state = torch.stack(
        [s_a, s_b, m_a, m_b, m_aa, m_bb, m_ab, new_sign, new_t_since, new_peak, new_prev_peak],
        dim=1,
    )
    return reward, new_state
