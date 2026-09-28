# One-direction spin campaign (velocity-era reference)

Goal: certified velocity-era flat-walk reference with sustained one-direction shaft spin.
SOP: offline replay gate -> 3000-it weight screens -> 20k from-scratch -> certification.
Approved plan: full autonomous chain; two arms (-0.1, -0.3); teacher carry-up deferred.

## Step 1: spin metrics (DONE)
shaft_spin_metrics() in gait_eval/metrics.py + report.py wiring + aggregate stats
(shaft_one_direction_ratio, shaft_mean_abs_vel). 3 new unit tests (26 pass). Validated
offline vs velact npz: mean|v| 5.606 (ad hoc 5.61), ratio 0.0057, reversals 2.85/s.

## Step 2: reversal-term replay gate (DONE, PASS)
offline_replay/replay_reversal_gate.py. Degen oscillator: 21.5 unweighted rev*dt/min ->
-0.1 = 3.3% and -0.3 = 9.9% of locomotion income (both in the 3-15% window); synthetic
one-direction spin pays exactly 0 (sticky rule verified). Context: the position-era
reference gait would have paid 2.6x more (55.8/min) -- explains why the term hurt then.

## Step 3: weight screens (KRABBY_REVERSAL_W env override added to
CrabHexFlatWalkRewardsCfg.__post_init__; default 0.0 until bake)
Progression rule per arm: (i) one_direction_ratio median >= 0.8, (ii) completion >= 0.9,
(iii) final reward >= 0.8x w0 control (>= ~17), (iv) mean |shaft v| >= 3 rad/s.
w0 control = geometry campaign fromscratch_velact_short (reward 21.0, ratio 0.006).

## Step 3 verdict: BOTH ARMS FAIL the spin criterion — HARD STOP (per plan)

| arm | reward @3k | 1-dir ratio | reversals/s | tripod | completion | slip |
|-----|-----------|-------------|-------------|--------|------------|------|
| w=0 control | 21.0 | 0.006 | 2.85 | 0.0 | 0.9 | 7.9% |
| -0.1 | 17.7 | 0.013 | 3.49 | 0.0 | 0.9 | — |
| -0.3 | 27.1 | 0.005 | 2.87 | **0.139** | **1.0** | 4.8% |

Reading: the penalty (up to ~10% of locomotion income) does not move the policy off the
oscillation attractor at all — reversal rates are unchanged; policies absorb the cost.
Unexpectedly, -0.3 produced the best velocity-era locomotion yet (reward 27 @ 3k = the
position-era 20k reference level; first nonzero tripod 0.139; completion 1.0; slip 4.8%)
— the reward gain is real gait improvement, not penalty avoidance. Mean |shaft v| stays
~5.6 rad/s in all arms (shafts never suppressed).

Design fork (user decision): see RESULTS.md.

## Round 3 (hybrid path, user-approved): positive spin reward + 20k reference in flight
- 20k from-scratch at KRABBY_REVERSAL_W=-0.3 LAUNCHED (best-locomotion config; hardware
  can reverse so oscillation transfers; mid-run 10k gait-eval gate armed).
- RewardOneDirectionSpin added (rewards.py): EMA signed-consistency x speed scale,
  command-gated. First replay caught tau=0.5 leaking 19.4% of spinner income to the
  1.4 Hz oscillator -> tau=2.0 attenuates to 5.4%. Replay gate PASS at +0.1 (9.1% of
  locomotion income) and +0.2 (18.2%); partial-spinner traces earn intermediate 13.9/min
  (smooth gradient toward the spin basin). Registered weight 0.0, KRABBY_SPIN_REWARD_W
  override. Screens queued AFTER the 20k run (serial GPU).

## 20k mid-run gate @10k: SPIN BASIN FOUND
one_direction_ratio = 1.0 (all 10 episodes), shaft |v| 5.98 rad/s — continuous
one-direction quick-return operation, discovered at w=-0.3 alone (basin roulette: the 3k
screen with identical config+seed landed in the oscillation basin instead; GPU
nondeterminism). Reward plateau ~18 = immature stepping on top of spin: tripod 0.0,
set-corr +0.53, slip 35%, completion 0.8. Formal mid-gate condition fired (tripod 0 +
positive corr) but killing the run would be wrong — this IS the target mechanism regime;
remaining 10k tests whether stepping consolidates on top of spin. Implication for the
spin-reward screens: the +0.1/+0.2 shaping may mainly serve to make basin entry RELIABLE
rather than lucky.

## Round 4 (lit-review synthesis, user-approved): shaft-phase contact schedule + phase lock
Per docs/lit-review-hexapod-reward-stability.md: Siekmann-style swing/stance windows (#1
ranked rec) with the CAM PHASE as the clock (the review's own s6/s9 observation, now
unlocked by the achieved continuous spin), + in-set phase locking toward tripod offsets.
reward_tripod_schedule route was DROPPED: its replay gate failed on the spin gait (0.4
paid crossings/min — the sparse-gradient signature of the 4 failed campaigns). Delivery
is FROM-SCRATCH per the review's fine-tune null (s9).
- PenaltyCamContactSchedule: contact during return stroke (|g|>0.55) + slide during
  power-stroke stance. Replica: ideal pays 0.0/min; spin gait 31.8; oscillator 32.9.
  Pure penalty -> no farmable state.
- RewardCamPhaseLock: in-set phase coherence x one-direction gate. Replica: ideal 59,
  spin gait 26 (half-climbed: FL-MR locked at 0 deg, rears lock=0.18), oscillator 3.
- Screen: fromscratch 3k, KRABBY_REVERSAL_W=-0.3 CAM_SCHED_W=-0.1 PHASE_LOCK_W=0.1.
  Gates: ratio>=0.8 (spin entry should now be reliable), completion>=0.9, slip<29%,
  reward>=17; tripod>0 is the prize.

## Round 4 screen r1: FAIL — fall-and-spin reward hack (fixed, rerunning)
screen_r4_synthesis flatlined at reward ~1.0: 100% crab_failure at ~80 steps for all
3000 iters. Post-mortem: a FALLEN robot is optimal under the ungated phase-lock term —
shafts spin/lock trivially with feet off the ground and the contact-schedule penalty
goes silent (no contact). The offline replica missed it because all fixtures were
walking traces (no fallen-robot family in the fixture set — noted as a gap for future
gates). Fix: RewardCamPhaseLock now gated on command-active x upright (projected
gravity), mirroring RewardOneDirectionSpin's cmd gate. Rerunning as screen_r4b_gated.

## Round 4b screen (gated): locomotion best-of-era, basin roulette persists
reward 19.7, completion 1.0, slip 4.2% (schedule penalty works!), tippy 6.8% — but
ratio 0.007 (oscillation basin at 3k; spin did not emerge this fast) and tripod 0.
Next: FT-from-model_19999 arm — terms are ALIGNED with its spin core (earns 26/min
phase-lock, pays 31.8/min schedule on its slip), so this differs from the lit review's
hostile-shaping fine-tune null. Fallback: 10k from-scratch screen.

## Round 4c (FT arm) + 4d launch
FT-from-19999 with round-4 terms: spin preserved (ratio 1.0), slip 29->24.7%, but rear
phase lock STATIC (0.18->0.19) — confirms the lit review's fine-tune null for phase
reorganization; FT refines execution, not structure. r4d (user-approved): 10k
from-scratch, all round-4 terms — spin historically emerges by 10k unaided; phase-lock
adds basin pull from step 0. Success = reference candidate.

## Staged ramp + power term (2026-08-15, user-approved)
- Lit review s5 CLOSED with a positive: replayed sum|tau*qdot| — walking spinner 738 W
  vs oscillators 1259-1423 W (position-era ref 1286 W). The saving is in the LEG chain
  (496 vs 988-1234 W): smooth yaw spares hip/knee reversal transients. Physics-grounded
  basin selector; penalty_mechanical_power added (weight 0.0, KRABBY_POWER_W).
- Env overrides now presence-based (explicit 0 disables baked defaults — needed for
  phase A).
- Staged ramp launched: 2 seeds; phase A 5k walk-first (schedule -0.1 only), phase B
  resume +10k (reversal -0.3, lock +0.1, power -0.001). Gates: ratio>=0.8 AND
  completion>=0.9 AND slip<0.15.
