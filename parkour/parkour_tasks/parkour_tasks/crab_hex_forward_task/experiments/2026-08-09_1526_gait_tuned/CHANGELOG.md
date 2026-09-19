# gait_tuned: Teacher-stack carry-up study (Milestone 18 Task 1 follow-on)

**Predecessor**: `sim_fine_tuning/2026-08-09_0920_short_runs/` — a 14-run autonomous flat-walk
weight-tuning campaign that baked a winning config into `CrabHexFlatWalkRewardsCfg`
(`reward_feet_air_time_positive=0.8`, `penalty_motor_direction_reversal=0.0`,
`reward_action_rate=-0.3`, `reward_delta_torques=-1e-6`), clearing every stated Task 1 §1c/§1f
target on flat terrain (tippy 13.39%, stride 0.1975m, completion 100%).

**Open threads carried into this study**: (1) the winning combo raised training-time
`error_vel_xy` to ~0.21 on flat — harmless there, but bridge/2b1/2b2 lean harder on tracking
responsiveness; (2) zeroing the reversal penalty risked regressing camshaft reversal count, a
real hardware-wear metric `motor_reversal_on` had previously cut 48-65%.

**Method**: (1) retrain flat-walk from scratch on the now-baked config to validate it
reproduces r14's quality without fine-tuning ancestry; (2) screen three teacher-stack arms
(T0 control / T1 full mirror / T2 hybrid) through bridge→2b1→2b2-batch1, scored on training
gates + `teacher_2b2_forward` gait-eval + camshaft reversal count; (3) extend the winning arm
with more 2b2 batches; (4) propose (not apply) a config bake based on the result.

## 1. Phase 0: from-scratch flat-walk retrain (baked config, 20000 iters)

Checkpoint: `logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14/model_19999.pt`.
crab_failure=0.0000 (train). Gait-eval (`flat_walk_forward`): tippy_tap 7.97% (series-best),
slip 2.31%, **tripod_score 0.401** (vs the series' established 0.0-0.025 range), completion
100%, stride 0.163m.

**Finding**: tripod_score jumped from ~0 to 0.401 purely from training the baked config from
scratch rather than fine-tuning on a v3-converged policy — no reward-code change. Directly
answers Task 1 §1d ("tripod score improves... via an explicit contact-schedule term or
demonstrated as emergent") in favor of emergence. Trade-off: stride (0.163m) is below r14's
fine-tuned 0.1975m and the ≥0.178m target — from-scratch training lands at a different point on
the gait-quality/stride trade-off than fine-tuning did.

## 2. Teacher-stack screening: T0 (control) vs T1 (full mirror) vs T2 (hybrid)

All three arms seeded from the Phase 0 checkpoint, run through bridge(100)→2b1(100)→2b2-batch1(500),
scored via training gates, `teacher_2b2_forward` gait-eval (10 episodes), and camshaft reversal
count read from the harness's own `metrics/episode_*.json` output.

| axis | T0 (control) | T1 (full mirror) | T2 (hybrid) | v3 anchor |
|---|---|---|---|---|
| tippy_tap (gait-eval) | 27.14% | 27.13% | **21.35%** | 21.0% |
| slip_ratio | **4.40%** | 5.15% | 4.58% | 3.0% |
| tripod_score | **0.325** | 0.321 | 0.272 | n/a |
| completion | 50% | **70%** | 60% | 80% |
| stride | 0.169m | 0.160m | 0.163m | 0.162m |
| camshaft reversals/ep | 700 | 733 | **663** | — |
| crab_failure (train, 2b2 batch1) | 37.68% | **31.25%** | 34.89% | 6.8-14% |
| error_vel_xy (train, 2b2 batch1) | 0.584 | 0.655 | **0.499** | — |

No arm swept every axis. **T2 (hybrid: keep the reversal penalty at its default, add the
smoothness terms) was selected** because it resolved both open threads without regressing the
primary gait metric: lowest tracking error of the three (better than even the unmodified
control), lowest camshaft reversal count (better than the control too — the reversal penalty and
smoothness terms compound rather than conflict), and by far the best tippy_tap (matching the v3
anchor almost exactly). Its cost was the lowest tripod_score and a completion rate 10 points
below T1's. T1 (the flat-walk winner carried up unchanged) had the best completion and the most
stable training, but re-opened both concerns this study set out to check.

Full arm-by-arm breakdown: `RESULTS.md`.

## 3. Winner extension: T2 2b2 batches 2-4

Ran three more 500-iter 2b2 batches from T2's batch1 checkpoint (~1997 total 2b2 iterations),
gate-checking each against the standard sweet-spot rule. **The stage plateaued without reaching
gate convergence** — `crab_failure` stayed in a 32-35% band across all four batches (gate needs
<20%), `obstacle_clearance` never exceeded ~0.06 (gate needs >0.15), `episode_length` stayed
490-570 (gate needs ≥750). Batch3 was the best training-time snapshot and was carried to a final
gait-eval.

**The extension made held-out gait quality worse, not better**: despite batch3's marginally
better training-time `crab_failure` (31.82% vs batch1's 34.89%), its gait-eval tippy_tap nearly
doubled (21.35%→34.64%) and tripod_score dropped by a third (0.272→0.185); stride, completion,
and reversal count were essentially unchanged. This is a clear divergence between the windowed
training metric and held-out gait quality — a caution against reading `crab_failure` alone as a
gait-quality proxy at this stage. Full batch-by-batch table: `RESULTS.md`.

**Consequence**: T2's original batch1 checkpoint remains the best result found. The extension
attempt is documented for the record but does not change the recommendation below.

## Verdict

**KEPT** — T2's reward-weight combination (`reward_action_rate=-0.3`, `reward_delta_torques=-1e-6`,
`penalty_motor_direction_reversal` unchanged at its existing default) is recommended for the
teacher stack, based on T2's batch1 result (the extension did not improve on it).

**Bake proposal** (not yet applied — pending user review, per the plan's ground rules):

In `parkour/parkour_tasks/parkour_tasks/crab_hexapod_task/config/crab_hex/agents/parkour_mdp_cfg.py`,
two classes actually declare `reward_action_rate`/`reward_delta_torques` in the teacher-stack
inheritance chain (`CrabHexRewardsCfg` → `TeacherWarmupRewardsCfg` → `TeacherBridgeRewardsCfg` →
`Stage2BPhase1RewardsCfg` → `Stage2BPhase2RewardsCfg`; Warmup/Bridge/Phase2 don't re-declare
these two terms, so editing only the two declaring classes covers every stage):

- `CrabHexRewardsCfg` (base, ~lines 232/273) — covers Warmup and Bridge, currently
  `reward_action_rate=-0.1`, `reward_delta_torques=-1e-7`.
- `CrabHexStage2BPhase1RewardsCfg` (~lines 462/482) — covers 2b1 and 2b2 (Phase2 inherits from
  Phase1 without re-declaring), currently the same defaults, `-0.1`/`-1e-7`.

Change both to `reward_action_rate=-0.3`, `reward_delta_torques=-1e-6`.
`penalty_motor_direction_reversal` needs **no change** anywhere — it stays at its existing
`-0.3` default (declared once in `CrabHexRewardsCfg`, never overridden down the teacher stack),
consistent with T2's design of keeping it on.

**Open item**: this recommendation rests on one 2b2 batch at the curriculum's established rocky
point, not a converged comparison, and the extension attempt showed training-time `crab_failure`
can diverge from held-out gait quality at this stage — a full sweet-spot convergence run (proper
4+ batches from a *bake-config* start, not a Hydra-override extension) would be the natural next
validation if the bake is applied.
