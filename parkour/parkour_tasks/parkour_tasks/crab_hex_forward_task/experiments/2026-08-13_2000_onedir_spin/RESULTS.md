<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-13_2000_onedir_spin/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_2000_onedir_spin/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.


## Round 2 (escalation, user-approved): -0.6 / -1.0 — CONCLUSIVE FAIL of the penalty route

| w | reward @3k | 1-dir ratio | completion |
|------|-------|-------|-----|
| -0.6 | 22.7 | 0.004 | 0.7 |
| -1.0 | 1.6 | 0.415 | 0.0 |

The transition is a cliff: -0.6 (~20% of income) still fully absorbed; -1.0 (~33%)
finally produces directionality (ratio 0.42 — proof the policy CAN discover it) but by
collapsing locomotion (standing still, spinning shafts to dodge the tax). No weight both
flips the basin and preserves walking. Penalty route closed on 5-point dose-response.

Remaining options, updated by this evidence:
(b) POSITIVE spin reward (shape toward the basin; -1.0's ratio 0.42 shows the behavior is
    reachable — it needs to be made attractive, not everything else made expensive).
    Candidate term: reward per-shaft signed-consistency (|mean v| / mean |v|) or net
    revolutions per window, gated on nonzero command. Replay gate first per SOP.
(d) ACCEPT oscillation: 20k reference at w=-0.3 (best locomotion; hardware can reverse,
    so transfer is safe). Spin question revisits later (e.g. after tripod consolidates).
Hybrid: (d) now for the reference + (b) as a separate follow-on campaign is also viable.

## 20k run (w=-0.3) certification — SPIN ACHIEVED, stepping un-phased
model_19999 (fromscratch_w0.3_20k/logs/rsl_rl/crab_hex_flat_walk/2026-08-14_08-41-28/):
one_direction_ratio 1.000 (10/10 episodes), shaft 5.99 rad/s continuous, completion 1.0,
reward 19.95; tripod 0.0, slip 29% median, tippy 15%. The campaign's core objective —
the quick-return mechanism driven as designed — is achieved and stable. Stepping phase
never consolidated into tripod within 20k.

Note for next lever: at 6 rad/s the support-swap half-cycle ~= 0.52 s — INSIDE the
reward_tripod_schedule 0.10-0.60 s band (unlike the old oscillation). The inert tripod
term can see this gait.

## Bake decision (user): see chat/AskUserQuestion

## Round 4 final (r4d, 10k from-scratch, all terms): spin gate FAIL, stepping best-in-class
ratio 0.015 (oscillation basin — shaping does not force basin entry), slip 10.1%, tippy
7.3%, completion 0.7 (3 falls), tripod 0, reward 22.9. Novel structure: entire left side
phase-locked (FL/ML 1.0, RL 0.89), mirror spin directions by side.

## Campaign conclusions (4 rounds of gated evidence)
1. Continuous one-direction spin IS a stable attractor of the velocity-era MDP
   (model_19999 -> model_22998: ratio 1.0, completion 1.0, slip 24.7%).
2. Basin ENTRY is early-training stochastics; no reward configuration tried (penalty
   dose-response 5 points, positive spin reward, phase-lock pull, schedule pressure)
   selects it reliably within 3-10k.
3. Fine-tuning refines execution (slip 29->24.7) but never reorganizes phase structure
   (rear lock 0.18->0.19) — consistent with the lit review's fine-tune null.
4. PenaltyCamContactSchedule reliably improves stepping in any basin (4.2-10.1% slip
   from scratch vs 29% unshaped).

## Escalation menu (future campaign)
(a) Seed-basin lottery with the spin gate as selector (repo precedent:
    2026-08-12_1550_seed_basin_search) — N seeds x ~5k iters, continue spinners.
(b) CPG/oscillator action space (lit review #5, held-in-reserve escalation): phase
    variable is architectural — basin question dissolves. Largest integration cost.
(c) Warm-start hybrid: init cam-channel weights from model_22998, rest fresh.

## Current best artifacts
- SPIN reference candidate: screen_r4c_ft19999/model_22998 (ratio 1.0, completion 1.0,
  slip 24.7%) — the mechanism working as designed, stepping mediocre.
- STEPPING benchmark: screen_r4b_gated/model_2999 (slip 4.2%, completion 1.0) —
  oscillation basin.

## BAKED (user decision, 2026-08-14)
- Velocity-era flat-walk reference: screen_r4c_ft19999/.../model_22998.pt (sha 580b6f3b…)
- penalty_motor_direction_reversal = -0.3 baked as flat-walk default
- scenarios_v1.yaml flat_walk_forward pin updated (with sha256)
- Plan of record for next campaign: seed-basin lottery (spin gate as selector)

## Seed-basin lottery (overnight 2026-08-14/15): 0/7 true winners — measured, decisive
Corrected scoring (ratio >= 0.8 AND completion >= 0.7): 5 oscillators (walking, ratio
~0.01; seed 3 slip 2.5% = best stepping of the era), 2 FALLEN SPINNERS (ratio ~1.0,
completion 0.00 — the fall-and-spin attractor captures spin-adjacent starts even with
the upright-gated phase-lock, because all shaping penalties go silent when airborne).
Full table: lottery_summary.md.

Basin probability estimate at 5k under current shaping: walking-spin ~0/7; the ONLY
observed walking-spin entry (seed-1 20k run) transitioned FROM established walking
(~18 reward through 7k) to spin by 10k. Reading: spin that emerges before walking
collapses into falling; spin that emerges after walking is stable. Suggests a STAGED
ramp (walk first, then ramp spin pressure mid-training) rather than more from-scratch
lottery — or the CPG action space, which sidesteps basins entirely.

## TRACKING REGRESSION ROOT CAUSE (2026-08-15, Task-1 speed-pressure follow-up)
Speed-feasibility analysis found something bigger: ALL velocity-era policies are
command-blind (fixed body speed across the 0.30-0.65 band; shaft speed constant), while
the position-era ref tracks to <=0.02 m/s and modulates cam activity with the command.
Root cause chain (probes A/B/C, offline):
  A. Velocity-era policies respond to the command obs 3-6x more weakly than position-era
     (action deltas across holds: cam 0.60/0.31 vs 1.80) — channel alive, never trained.
  B. track_lin_vel_xy_exp (weight 1.25, sigma^2=0.02) has capture radius ~+-0.25 m/s;
     gradient peaks at err=0.10 and is numerically ZERO at the spin gait's 0.48 deficit.
  C. Training curves: position-era tracking income hit 0.53/1.25 by iter 500 and ~0.95 at
     convergence; velocity-era 20k flatlined at ~0.25 from iter 2500 for 17,500 iters.
Mechanism: position-action exploration starts slow -> errors inside the narrow well ->
tracking learned first. Velocity-action exploration (scale 6 rad/s) lands the proto-gait
far outside the well -> zero gradient from the era's largest term -> gait built entirely
on secondary income, command ignored. sigma^2=0.02 only ever worked by exploration luck.
Implication: the week's basin economics were computed WITHOUT the dominant reward term;
kinematics say spin at 6 rad/s can deliver 1.08 m/s (0.71 single-set), so tracking via
cadence modulation is feasible for spin — the CPG-native control structure.
PROPOSED FIX: widen tracking sigma^2 0.02 -> 0.25 (legged-gym standard), optionally
annealed; rescreen 3k from scratch; gate on tracking income > 0.8/1.25 AND per-hold
deficit < 0.1 at eval.

## Sigma-fix screen (2026-08-15): NECESSARY BUT NOT SUFFICIENT — paused for review
sigma^2 0.02 -> 0.25: training tracking income 0.73 by iter 500 (vs 0.25-flatline era),
reward 28.2 @3k (era best). BUT eval: still command-blind (vx 0.21-0.23 flat across the
band, deficits to -0.44, shafts pinned 5.7 rad/s) — the wide well PAYS at large deficits
(46-78% income), so it fixed gradient reach and destroyed gradient pressure. Refined
diagnosis: opposing cost is credit assignment under velocity semantics — stride is
cam-fixed, so speed control requires sustained cadence modulation (~50-step diffuse
credit) vs position actions' within-step speed response. A constant needs no control law;
the shallow well doesn't pay enough to force learning one.
Queued candidates (NOT launched, campaign paused): (a) linear |v_err| penalty ~-0.5
(constant pressure, unfarmable), (b) sigma anneal 0.25->0.02, (c) CPG action space —
which now solves BOTH open problems at once: omega is an explicit action, making
command->speed a one-step credit assignment, and spin is structural.
NOTE for certification SOP: add a per-hold tracking-deficit gate (<0.1 m/s) — the
completion metric masked command-blindness in the model_22998 bake.

## CAMPAIGN CLOSED — UNBAKED (user decision 2026-08-15)
The model_22998 reference bake and the flat-walk reversal -0.3 default are rescinded:
all era weight evidence was scored with the dominant tracking term inert. Robot config
(limits, velocity actions, gains, inertia) is unchanged and correct. Successor:
sim_fine_tuning/2026-08-15_*_task1_velocity/ — Task-1 discipline, amended certification
gates, change queue C1 (tracking) -> C2 (air-time recal) -> C3 (stance band) -> C4+
(spin instruments re-gated) -> CPG escalation.
