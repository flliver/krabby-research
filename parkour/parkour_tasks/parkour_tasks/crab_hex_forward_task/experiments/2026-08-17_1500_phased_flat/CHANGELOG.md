<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-17_1500_phased_flat/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-17_1500_phased_flat/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`.

# Phased flat-walk curriculum campaign

Protocol: Task-1 discipline per phase — baseline -> replay gate (new terms) -> one change
per 3k screen -> gait_eval (flat + spin metrics + clearance) -> CHANGELOG row -> commit.
Plasticity telemetry (actor/critic weight norms) recorded at every phase boundary.
L2-init is contingency-only (user decision).

## Phase A — gait stabilization: COMPLETE (no GPU)
Artifact: C2 model_2999 (tripod 0.573, tracking pass, actor norm 26.3 — plastic).

## Phase B — oscillation -> spin (queue)
| arm | config | status |
|-----|--------|--------|
| B0 baseline | resume 2999, KRABBY_REVERSAL_W=-0.3, 3k | queued behind v3 batches 5-6 |
| B1 critic reset | B0 + fresh critic at boundary | pending |
| B2 energy-only | KRABBY_REVERSAL_W=0 + KRABBY_POWER_W=-0.001 | pending |
| B3 dose/spin/lock arms | per plan | pending |
Gates: ratio >= 0.8, tracking < 0.1, completion >= 0.9, tripod >= 0.3, reward >= 0.8x B0.

## Phase C — obstacle introduction (design)
KRABBY_FLAT_TERRAIN_MODE for the flat env (~80/20, difficulty 0.05-0.2, frozen);
build during Phase B screens.

## B0 baseline — COMPLETE 2026-08-17
Resume model_2999 + KRABBY_REVERSAL_W=-0.3, 3k iters (model_5998, 74 min).
Train: reward 28.2, failure 4.4%, ep_len 982, reversal income -0.106 (absorbed).
Eval (10 ep): **one_direction_ratio 0.0055** (pure oscillation, 2.92 reversals/s),
tripod 0.601 (> base 0.573), deficits low/mid/high -0.023/+0.013/+0.092 (tracking PASS),
completion 0.9, slip 10.1%, tippy 20.4%.
Verdict: pressure alone does not convert the gait even on the plastic base — walking
improves, spin unchanged. This is the do-nothing-clever number the arms must beat.
Gates for arms: ratio >=0.8 (win) / >=0.3 (stop-rule floor), deficits <0.1,
completion >=0.9, tripod >=0.3, reward >=22.6 (0.8x B0).

## B1 critic reset — COMPLETE 2026-08-17 — NEGATIVE
Surgery: fresh critic (norm 30.7->17.3), Adam cleared; resume + reversal -0.3, 3k
(model_5998). Train: reward 24.4, failure 7.5%, reversal income -0.108 (absorbed).
Eval: ratio 0.0057 (no conversion, 3.03 rev/s), tripod 0.450 (<< B0 0.601),
completion 0.8 (FAIL), high-hold deficit +0.111 (FAIL), slip 12.5%, tippy 28.7%.
Verdict: critic reset at the B boundary does not enable conversion and costs walking
quality. Combined with the v2-base falsification (lit review §5), the primacy-bias
hypothesis is now negative in BOTH the rigid and plastic settings — drop critic reset
from the remaining Phase-B/C candidate lists except as a no-cost adjunct.

## B2 energy-only — COMPLETE 2026-08-17 — NO CONVERSION, best walker
KRABBY_REVERSAL_W=0 + KRABBY_POWER_W=-0.001, resume model_2999, 3k (model_5998).
Train: reward 18.9 (power income -0.74 dominates the gap), failure 5.1%, ep_len 960.
Eval: ratio 0.0056 (no conversion; reversals 2.62/s, mildly down), tripod **0.621**
(best yet), tippy 18.9% (best), slip 10.5%, completion 0.9, deficits -0.014/+0.010/
+0.076 (all PASS). Verdict: the physics cost gradient improves gait quality across the
board but cannot bootstrap the oscillation->spin flip on its own — the 42%-cheaper spin
basin is separated by a barrier the local gradient doesn't cross. Candidate keeper as a
quality term regardless of spin outcome.

## Ratio scoreboard after singles round 1
B0 pressure -0.3: 0.0055 | B1 critic reset: 0.0057 | B2 energy: 0.0056.
Remaining singles: B3a spin reward +0.2 (positive income on the EMA spin metric — only
untried mechanism class), then phase-lock +0.1 / reversal dose -0.6 if needed.
Stop rule floor (no arm >= 0.3) not yet triggered — singles not exhausted.

## B3a spin reward +0.2 — COMPLETE 2026-08-17 — NO CONVERSION
Baseline stack + KRABBY_SPIN_REWARD_W=0.2, resume model_2999, 3k (model_5998).
Train: spin income 0.009/0.2 (EMA ratio ~5% throughout — never bootstrapped), reward
26.9, failure 10.3%. Eval: ratio 0.0043, tripod 0.569, completion 1.0, deficits
-0.024/+0.004/+0.090 (PASS), slip 10.3%, tippy 24.3%, 2.95 rev/s.
Verdict: positive income on the spin metric cannot bootstrap from a ~0 base — the EMA
gate means near-zero gradient until spinning already exists. Fourth mechanism at
ratio ~0.005. Remaining singles: B3b phase-lock +0.1, B3c reversal dose -0.6.

## B3b phase-lock +0.1 — COMPLETE 2026-08-17 — NO CONVERSION
Baseline stack + KRABBY_PHASE_LOCK_W=0.1, resume model_2999, 3k (model_5998).
Train: phase-lock income 0.005/0.1 (same bootstrap failure as B3a — pays only for
cam-consistent contacts, which don't exist at ratio ~0.005), reward 28.2, failure 3.5%.
Eval: ratio 0.0045, tripod 0.563, **completion 0.6 (4 falls — worst arm)**, deficits
PASS, 2.95 rev/s. Verdict: null on spin, negative on robustness. Fifth mechanism null.

## B3c reversal dose -0.6 — COMPLETE 2026-08-17 — ABSORBED
KRABBY_REVERSAL_W=-0.6, resume model_2999, 3k (model_5998). Train: reversal income
-0.193 (~2x the -0.3 income = unchanged reversal rate), reward 26.7, failure 6.0%.
Eval: ratio 0.0042, tripod 0.543, completion 0.6 (4 falls), deficits PASS, 2.90 rev/s.

## PHASE B SINGLES ROUND CLOSED — STOP RULE TRIGGERED 2026-08-17
Six mechanisms, one change per screen, all from the plastic C2 base, all ratio ~0.005
(gate floor 0.3): B0 pressure -0.3 (0.0055), B1 critic reset (0.0057), B2 energy
attraction (0.0056), B3a spin income +0.2 (0.0043), B3b phase-lock +0.1 (0.0045),
B3c pressure -0.6 (0.0042). No arm moved the ratio AT ALL — the oscillation basin is
not escapable by reward shaping from this base, plastic or not. Per the plan's stop
rule: hard stop, user fork required before combos or structural changes.
Best walker artifact of the round: B2 model_5998 (tripod 0.621, tippy 18.9%, all
non-spin gates PASS) — candidate Phase-C base if the fork de-scopes spin from Phase B.

## COMBO ROUND — COMPLETE 2026-08-18 03:35 — ALL SIX NULL ON SPIN
Six overnight combo screens (run_combo_round.sh), all 3k resumes from the C2 base:

| arm | stack | ratio | tripod | completion | failure | rev/s |
|---|---|---|---|---|---|---|
| CB1 | pow+rev0.3 | 0.0072 | 0.604 | 1.0 | 3.3% | 2.54 |
| CB2 | pow+rev0.3+spin0.2 | 0.0064 | 0.605 | 0.9 | 2.3% | 2.59 |
| CB3 | pow+spin0.2 | 0.0052 | 0.593 | 1.0 | 2.7% | 2.59 |
| CB4 | pow+rev0.6 | 0.0062 | **0.635** | 1.0 | **1.9%** | 2.56 |
| CB5 | rev0.3+spin0.2+lock0.1 | 0.0052 | 0.574 | 0.8 | 2.6% | 2.82 |
| CB6 | all four | 0.0053 | **0.649** | 0.9 | 3.9% | 2.63 |

Best ratio 0.0072 (CB1) — noise-level, ~40x below the 0.3 stop-rule floor. The energy
backbone again bought gait quality (CB6 tripod 0.649 = best-ever; CB4 failure 1.9% =
best-ever) and shaved reversals to ~2.55/s, but no combo initiated conversion.
PHASE B REWARD-SHAPING IS EXHAUSTED: 12 arms (6 singles + 6 combos) spanning pressure,
attraction, income, timing, surgery, and their combinations — ratio never left
[0.004, 0.008]. Conclusion stands: the oscillation basin is structurally inescapable
by reward shaping; remaining forks are structural (unidirectional cam action clamp)
or de-scope (Phase C from the best walker). Best walker artifacts now: CB6 model_5998
(tripod 0.649) and CB4 model_5998 (failure 1.9%, tripod 0.635, completion 1.0).

## FS1 from-scratch under CB4 stack — COMPLETE 2026-08-18 — NEGATIVE (creep basin)
User-requested: pow -0.001 + rev -0.6 from scratch, seed 1, 3k (model_2999).
Train: reward 8.1 (still rising at 3k but toward the wrong basin), failure 17%,
air-time income ~0.002 flat all run, tripod income 0.0 all run.
Eval: **tripod 0.000**, ratio 0.0169 (3x resume arms, still nothing), deficits
+0.20/+0.36/+0.54 (walk never formed — commanded speeds never reached), slip 24.7%,
tippy 36%, completion 1.0 only because it never falls: it creeps.
Verdict: penalties present during formation push the policy into an energy-minimizing
creep instead of a gait — the outcome §5 of the reward-stability review predicted for
energy terms before locomotion exists. From-scratch basin selection under the CB4
stack finds neither tripod nor spin. The resume-then-shape and scratch-with-shape
directions are now BOTH closed; remaining forks are structural (unidirectional cam
clamp) or de-scope (Phase C from CB4/CB6 walker).

## FS2 energy-only from scratch — COMPLETE 2026-08-18 — NEGATIVE (no tripod)
User-requested: pow -0.001 alone (reversal off) from scratch, seed 1, 3k (model_2999).
Train: reward 11.3, failure 9.6%, ep_len 968 — healthier formation than FS1, but
tripod income 0.0 and air-time ~0 for the entire run.
Eval: **tripod 0.000** again, ratio 0.0084, deficits -0.079/+0.005/+0.128 (tracks at
low/mid — it genuinely locomotes, unlike FS1), slip 16.4%, **tippy 41.5%**,
rev/s 1.74 (lowest ever). Verdict: without reversal tax the policy reaches commanded
speed, but via a low-lift dragging/vibratory mode — energy pressure during formation
suppresses exactly the leg-lifting (air time) that tripod formation requires, so the
cheap mode wins before stepping exists. Same family as the C5a tippy-tap failure.
Scratch direction now closed in both variants: rev+pow -> creep; pow alone -> drag.
Energy terms are safe only AFTER a stepping gait exists (B2/CB4/CB6 precedent).

## FORK DECISION (user, 2026-08-18): de-scope spin — PHASE B CLOSED
Phase-B winner: **CB4 model_5998** (pow -0.001 + rev -0.6 resumed from C2 base;
failure 1.9%, completion 1.0, tripod 0.635, deficits PASS). Spin objective de-scoped
from the sim curriculum; the sim-to-real cam-direction question moves out of scope of
this campaign. Phase C begins: build KRABBY_FLAT_TERRAIN_MODE (light terrain in the
flat env, ~80/20 flat/easy-obstacles, difficulty 0.05-0.2, frozen levels), then C0
baseline = CB4 winner resumed +3k under terrain mode with the same reward env.
Gates: clearance alive >=0.05 by +3k, flat metrics within 10% of CB4, completion >=0.8.

## C0 baseline — COMPLETE 2026-08-18
CB4 winner resumed +3k under KRABBY_FLAT_TERRAIN_MODE=light (80/20, diff 0.05-0.2,
frozen levels) + CB4 reward env + clearance instrument 0.01 (model_8997).
Train: reward 19.0, failure 5.0% (vs CB4 1.9% pure-flat — modest terrain cost),
ep_len 982. Clearance income ~0.0000 (peaks 0.0003, raw ~0.03): lifting NOT learned.
Flat eval: **tripod 0.660 (new best)**, completion 1.0, deficits -0.023/+0.009/+0.081
(PASS), slip 11.1%, tippy 21.2%. Gates: flat-retention PASS, completion PASS,
clearance FAIL (lifting lags). Per plan: C1 = clearance to shaping weight (1.8,
teacher-validated) — the designated knob for this exact outcome.

## C1 clearance shaping 1.8 — COMPLETE 2026-08-18 — LIFTING STILL ABSENT
Same as C0 but KRABBY_CLEARANCE_W=1.8 (model_8997). Train: reward 18.5, failure 5.9%.
Clearance income DECLINED across the run (0.0035 -> 0.0014) — the weight didn't ignite
lifting; income drifts down as flat-majority optimization dominates. Flat eval: tripod
0.636, completion 1.0, slip 11.5%, tippy 19.7% (flat retention PASS).
Diagnosis: at difficulty 0.05-0.2 the obstacles are small enough to walk through/over
without lifting — no gradient exists. Next knob per plan: difficulty raise.
C2 = KRABBY_FLAT_TERRAIN_DIFF=0.2:0.4, one change vs C1.

## C2 difficulty 0.2-0.4 — COMPLETE 2026-08-18 — SAME NULL SIGNATURE
One change vs C1 (KRABBY_FLAT_TERRAIN_DIFF=0.2:0.4), model_8997. Train: reward 18.5,
failure 4.9% (did NOT rise with difficulty), clearance income declining 0.0030->0.0019.
Flat eval skipped (not a winner candidate; GPU saved). Two knobs, one signature:
low failure + declining clearance = the policy slows/stalls at obstacles instead of
falling, and the clearance term's gates (min_forward_speed 0.25, min_goal_progress
0.15) exclude slow crossings from income — no gradient toward lifting. Next: video
diagnostic of C2 on light terrain before any gate-param surgery (which would need the
offline replay gate).

## Video diagnostic + C3 gate relaxation — 2026-08-18
600-step video of C2 model on 100% obstacle tiles (diff 0.2-0.4): the robot CREEPS —
legs cycle, goal markers stay ahead, no falls. Confirms the mechanism: on obstacle
tiles the policy drops below the clearance term's income gates (min_forward_speed
0.25, min_goal_progress 0.15), so the slow first crossings that need reinforcing pay
zero. Fix: flat-stack copy of the term relaxes gates to 0.10/0.05 (teacher copy
untouched). Replay gate N/A: function unchanged + teacher-validated, params loosen
toward permissiveness; the too-permissive failure mode is covered by the C3 screen's
income trace + flat-retention eval. C3 = C2 config + relaxed gates.

## C3 relaxed gates — COMPLETE 2026-08-18 — level doubled, trend still down
Same as C2 + relaxed clearance gates (model_8997). Train: reward 18.8, failure 8.8%,
clearance 0.0044 -> 0.0025 (end-level ~2x C2's but still declining; small final uptick).
Diagnosis update: the CB4 reward env carries KRABBY_POWER_W=-0.001 which taxes the
marginal work of lifting — the exact interaction FS2 exposed (power suppresses
leg-lift). Phase C has been shaping clearance against an energy term that pays the
policy to keep feet low. C4 = C3 minus the power term (KRABBY_POWER_W=0, one change).

## C4 no-power — COMPLETE 2026-08-18 — decline stopped, no growth
C3 config + KRABBY_POWER_W=0 (model_8997). Train: reward 26.5, failure 9.2%,
clearance 0.0041 -> 0.0030 (holds ~0.0035-0.0043 mid-run, no monotonic decline for
the first time — the power tax WAS suppressing lifting — but no ignition either).
Remaining gap vs teacher-terrain lifting (v1 income 0.08-0.15): the teacher stack's
dense swing-height shaper reward_foot_clearance +2.0 does the mechanical teaching;
the sparse outcome bonus alone has nothing to climb. C5 = C4 + KRABBY_FOOT_CLEAR_W=2.0
(term added to flat stack as 0.0 instrument, teacher-validated params).

## C5 foot-clearance shaper — COMPLETE 2026-08-18 — pays but flat
C4 + KRABBY_FOOT_CLEAR_W=2.0 (model_8997). Train: reward 27.3, failure 7.7%.
foot_clearance income ~0.02 (raw ~0.01 — swings graze the 0.05m band), obstacle
bonus ~0.004; BOTH flat across the run. Five reward arms, one shape: income
level responds to each fix, trend never grows. Suspicion moves to EXPOSURE: at 80/20
the policy earns ~all income without lifting; teacher stages force lifting because
obstacle-majority terrain gates progress on it. C6 = C5 + KRABBY_FLAT_TERRAIN_FLAT_FRAC=0.5.

## C6 exposure 50/50 — BATCH 1 COMPLETE 2026-08-18 — FIRST GROWTH SIGNATURE
C5 + KRABBY_FLAT_TERRAIN_FLAT_FRAC=0.5 (model_8997). Train: obstacle_clearance income
GROWS monotonically 0.0131->0.0307 (2.3x, still climbing at 3k — the first growing
trend in Phase C); foot_clearance 0.063->0.073; goal_idx 1.09. EXPOSURE WAS THE
BOTTLENECK. Acquisition cost: failure 28.6%, ep_len 666, reward 16.6 (completion
~0.71 < 0.8 gate). Flat eval: tripod 0.466 (FAIL vs 10% retention: CB4 0.635),
completion 0.9, slip 16.1%, tippy 31.5%. Verdict: continue (primary metric climbing)
— batch 2 = resume model_8997, same env, +3k. Watch: failure recovery + tripod;
if lifting consolidates but flat degrades further, next knob is a fraction SCHEDULE
(50% acquisition -> 20% consolidation), not more exposure.

## C6 batch 2 — COMPLETE 2026-08-18 — lifting consolidating, flat eroding
Resume b1, +3k (model_11996). Train: clearance 0.039->0.098 (windowed 0.107 —
**through the 0.05 gate**, still climbing), failure 28.6->20.3%, ep_len 666->747,
reward 16.6->17.9 — all recovering during acquisition. Flat eval: tripod 0.412
(0.635->0.466->0.412 across lineage — still eroding), completion 0.8, tippy 28.3%.
Decision: batch 3 at 50/50 (clearance not plateaued; stopping mid-curve wastes the
unlocked skill), THEN consolidation batch at 20% exposure to restore the flat gait —
the tripod is the original deep skill and should restore cheaper than lifting re-learns.

## C6 batch 3 — COMPLETE 2026-08-19 — virtuous curve
+3k (model_14995). Train: clearance 0.086->0.164 (windowed 0.145 — 2b2-gate scale on
light terrain), failure 20.3->12.0%, ep_len 829, reward 21.1 — all recovering while
clearance climbs, no plateau. Flat eval: tripod 0.384 (erosion decelerating:
0.635/0.466/0.412/0.384) but completion 0.9, slip 10.0% (lineage best), tippy 23.1% —
robustness recovers; only strict tripod phasing relaxes. Continue batch 4 at 50/50;
20% consolidation pass queued post-plateau to restore phasing.

## C6 batch 4 — COMPLETE 2026-08-19 — acquisition phase closed
+3k (model_17994). Clearance 0.153->0.183 (windowed 0.240, late surge), growth
slowing (+12%/batch vs +35%); failure regressed 12.0->18.0% (surge riskier), ep_len
812, reward 21.6. Verdict: stop 50/50 acquisition at 4 batches/12k iters; start
CONSOLIDATION at 20% exposure (FLAT_FRAC default 0.8), same terms otherwise, resume
model_17994. Success = tripod restoration toward 0.6 + clearance retention
(exposure-adjusted: >=~0.07 income at 20% ~= batch-4 rate).

## C7 consolidation — COMPLETE 2026-08-19 — C-WINNER CANDIDATE (one gate deviation)
Resume C6-b4, 20% exposure, +3k (model_20993). Train: failure 7.6%, ep_len 941,
reward 24.4; clearance income 0.023 steady (5-6x the pre-acquisition C5 level at the
same exposure — lifting RETAINED; below pure exposure-scaling of batch 4, so some
crossing aggressiveness relaxed). Flat eval: **completion 1.0 (no falls)**, slip
11.7%, tippy 27.3%, tripod 0.409 (recovery marginal from 0.384; 0.635 bar NOT met).
Phase-boundary weight norms: C2 base 29.5/30.7 -> CB4 31.7/35.2 -> C7 37.0/57.1
(actor/critic). Gate status: clearance PASS (income alive/steady, learned while
plastic), completion PASS, flat robustness PASS, strict tripod-phasing retention FAIL
(0.41 vs within-10%-of-0.635). NOTE: the 2b2 §4.2b gates do not score tripod phasing;
they score clearance/failure/ep_len/progress/goal_idx — the profile C7 is strong in.
DECISION POINT (user): accept C7 as C winner and hand off to teacher stack, or spend
more consolidation batches chasing tripod restoration first.

## C WINNER DECLARED (user, 2026-08-19): C7 model_20993 — HAND-OFF BEGINS
Teacher-stack hand-off per plan: fresh lineage phased-flat-C7-2026-08-19 in
curriculum_state.json; bridge (100 it) -> 2b1 (100 it) -> 2b2 (500/batch, §4.2b gates).
Flat-stage KRABBY_* overrides do not apply to teacher rewards (flat-stack-only block).

## 2b2 CEILING REPORT — 2026-08-19, 6 batches / 3000 iters (C7 lineage)
Trajectory (failure / clearance / levels): 34.1/0.071/- -> 32.2/0.117 -> 28.7/0.083 ->
33.5/0.150/2.39 -> 35.8/0.154/2.38 -> 35.2/0.186/2.48.
GATES: clearance **0.186 PASS** (>0.15 — FIRST velocity-era pass, climbed every batch,
still climbing at ceiling); goal_idx 1.23 PASS; failure 0.287 best-seen / 0.352 final
vs <0.20 FAIL (plateau promotion-confounded — levels rose 2.39->2.48); ep_len 559 vs
750 FAIL; progress 0.075 vs 0.15 FAIL. vs prior best lineage (v3): failure best
0.287 vs 0.435, clearance 0.186 vs 0.121-never-passing. The lifting-first curriculum
beat every prior lineage on every 2b2 metric. Awaiting user: extend past ceiling /
frozen-levels decouple experiment / stop and evaluate.

## STOP-AND-EVALUATE (user option 3) — 2026-08-19 — LINEAGE BEATS THE POSITION-ERA TEACHER
Fixed 2b2 course (teacher_2b2_forward, frozen levels, pinned seed):

| ckpt | completion | tripod | slip | tippy | deficits low/mid/high |
|---|---|---|---|---|---|
| model_22688 (batch 3) | **0.8** | **0.237** | 8.9% | 14.8% | +0.05/+0.14/+0.30 |
| model_24185 (batch 6) | 0.7 | 0.182 | 9.5% | 14.0% | +0.01/+0.15/+0.32 |
| model_21100 (position-era baseline teacher) | 0.7 | 0.052 | - | - | - |

model_22688 EXCEEDS the position-era certified teacher on the pinned scenario:
completion 0.8 vs 0.7, tripod 0.237 vs 0.052 (4.5x). The training-side failure gate
(<20%) was promotion-inflated: on fixed terrain the policy is better than the
treadmill metric suggested. Weakness: high hold (0.85 m/s) deficit 0.30 — beyond the
flat stage's trained 0.30-0.65 envelope, the known next frontier.
ARTIFACT: model_22688 = phased-flat lineage 2b2 teacher candidate.
Campaign disposition: Phase A (existing) -> B (spin de-scoped after 13 arms) ->
C (lifting-first: the win) -> hand-off (bridge/2b1 records, 2b2 clearance gate first
pass, eval beats baseline). CAMPAIGN COMPLETE pending user's next direction
(distillation / high-speed extension / hardware).

## STUDENT DISTILLATION — COMPLETE 2026-08-20, 12k ceiling (model_34685)
Loss 3.73 -> 2.08 -> 1.85 -> 1.82 (batch 3 +1.5% < 3% bar: converged). Evals vs the
position-era student baseline model_29098:
| scenario | model_34685 | model_29098 |
|---|---|---|
| 2b2: tripod / completion | **0.161** / 0.6 (4 falls) | 0.079 / **0.9** |
| flat: tripod / completion | **0.138** / 0.5 | 0.08 / **0.8** |
Deficits low/mid/high: +0.04/+0.14/+0.33 (2b2) — the teacher's high-speed envelope gap
(flat stage trained 0.30-0.65, schedule demands up to 0.85) transfers to the student
and drives the falls. Gait quality 2x the old student; robustness below it.
Disposition: student stage recorded at ceiling; the binding constraint for both
teacher and student is now the SPEED ENVELOPE, not terrain skill.
