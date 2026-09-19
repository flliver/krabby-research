# Changelog: `sim_fine_tuning/2026-08-09_0920_short_runs`

**Milestone 18, Task 1 item**: §1f "Tippy-tap eliminated on the Task 0 eval set" and §1c "Stride-length term ... measured stride length improves against baseline" -- the reward-*tuning* phase of Task 1, following the reward-*shaping* series (`baseline` -> `motor_reversal_on` -> `stride_length_on` (v2, reverted) -> `stride_length_v3` (kept) -> `stride_length_v4` (tried, reverted)). With the stride-length term's definition settled at v3, this campaign tunes the existing reward weights against the v3 flat-walk checkpoint.

**Predecessor**: `sim_fine_tuning/2026-08-08_1701_stride_length_v3` (flat-walk checkpoint `model_19999.pt`, unchanged reward code).

## Method

Per Task 1 §1's own guidance ("Short resumes (100-2000 iters) are enough to see gait changes; from-scratch runs are not needed here"), this campaign ran **short flat-walk-only resumes** (1000-2000 iterations) from the v3 checkpoint, sweeping reward-term **weights and existing config params only** -- no reward function code was changed. Each candidate override was applied via Hydra CLI overrides (`env.rewards.<term>.weight=X`, zero code churn) and scored with the Task 0 gait-eval harness (`--scenario flat_walk_forward`) against the v3 reference. Full run-by-run ledger: `RESULTS.md` (14 runs).

**Guardrails applied to every candidate**: gait-eval `schedule_completion_rate` must stay 100% (any fall is treated as a real regression, matching the discipline used throughout the reward-shaping series), `slip_ratio` within ~1pt of reference, training-time `crab_failure` not exploding.

## Exploration and results, in order

1. **Air-time threshold** (`reward_feet_air_time_positive.params.threshold`, Task 1 §2 item 1's headline change): swept 0.05 (default) -> 0.10 -> 0.15 at default weight. Both alternate values were flat-to-slightly-worse vs reference -- REVERTED, threshold stays at 0.05.
2. **Air-time weight**: swept 0.40 (default) -> 0.8 -> 1.2. **0.8 KEPT** (tippy 19.09%->18.02%, stride 0.178->0.181m, slip 1.71%->1.59%, crab_failure 1.17%, improves every axis). 1.2 REVERTED (guardrail violation: 1 fall/10 in gait-eval, worse on every axis than 0.8).
3. **Stride-length weight** (on top of airtime=0.8): swept 0.5 (default) -> 0.75 -> 1.0. Neither cleanly beat the airtime-only baseline -- 1.0 improved tippy (17.20%) but broke completion (1 fall); 0.75 restored completion but tippy regressed past reference (19.33%). Non-monotonic; REVERTED both, stride weight stays at 0.5.
4. **Anti-shuffle / contact-shape** (on top of airtime=0.8): `penalty_foot_idle_when_forward` -0.12->-0.25, `penalty_excess_feet_contact_forward` -0.20->-0.4, `reward_stance_support_feet_when_forward` 0.0->+0.1. None beat the airtime-only baseline -- two hit the completion guardrail, one was flat. All REVERTED.
5. **Slip guardrail** (`feet_slide`): skipped -- its own trigger condition (slip degrading in the kept config) was never met.
6. **Motor-reversal vs smoothness, 3-arm study** (§2.5; both mechanisms target the same goal of suppressing high-frequency motor reversals, tested as separate arms per explicit user direction rather than stacked blindly):
   - **Arm A** (`penalty_motor_direction_reversal` alone, swept -0.15 / -0.3 default / -0.6): neither alternate value beat the airtime-only baseline. -0.15 (weaker) made tippy *worse* (19.49%) and hit the completion guardrail; -0.6 (stronger) restored completion and gave the best stride yet (0.183m) but tippy reverted to roughly reference level, cancelling the air-time gain. Both REVERTED.
   - **Arm B** (reversal penalty OFF, `reward_action_rate` + `reward_delta_torques` swept instead): mild combo (-0.1 / -1e-7) was a rough tie with the airtime-only baseline. **Strong combo (-0.3 / -1e-6) was the best result of the entire campaign** -- tippy 16.64% (best-of-campaign), stride 0.1835m (best-of-campaign), completion 100%, slip flat. Training-time tracking error was elevated (0.212 vs ~0.17-0.18 typical) but this did not show up as a problem in gait-eval completion or crab_failure.
   - Since Arm A produced no winner, there was nothing to combine into an "Arm C" -- Arm B's strong combo stands as the study's result on its own.

## Combined-winners config

```
env.rewards.reward_feet_air_time_positive.weight=0.8
env.rewards.penalty_motor_direction_reversal.weight=0.0
env.rewards.reward_action_rate.weight=-0.3
env.rewards.reward_delta_torques.weight=-1e-6
```

Turning the motor-reversal penalty **off** and replacing it with general action-smoothness terms outperformed every attempt to retune the reversal penalty itself -- a genuinely interesting finding: the two mechanisms both suppress high-frequency reversals, but the general-smoothness route reached a better gait-quality/stability trade-off than the reversal-specific one did at any weight tested.

## Final validation (2000 iterations)

| | v3 reference | short_runs r14 (2000 iters) |
|---|---|---|
| tippy_tap_fraction | 19.09% | **13.39%** |
| stride (pooled mean magnitude) | 0.178 m | **0.1975 m** |
| slip_ratio | 1.71% | 1.66% |
| schedule_completion_rate | 100% | 100% |
| crab_failure (training) | -- | 2.23% |

**All three campaign targets cleared**: tippy_tap <= baseline's 13.5% (Task 1 §1f), stride >= v3's 0.178m (Task 1 §1c), completion stays 100%. r14 is the first run in the entire Task 1 series to clear the tippy-tap target while *also* holding the best stride length across every run compared (baseline, motor_reversal_on, v2, v3, v4) -- previously these two metrics traded off against each other in every prior comparison.

## Verdict

**KEPT (pending user review before baking into the persistent config).** All four weight overrides are config-only (existing reward functions, no code changes) and directly addressable via Hydra CLI, so nothing here required touching `parkour_mdp_cfg.py`. Per the campaign's own ground rules, this changelog documents the result; whether to bake the combined-winners config into `CrabHexFlatWalkRewardsCfg` as the new persistent default (and whether/how it should propagate up the teacher stack, likely as a new stage class per Task 1 §1a) is left for explicit user decision rather than applied autonomously.

**Open thread**: the strong-smoothness combo's elevated training-time tracking error (0.206-0.212 vs ~0.17-0.18 typical) did not translate into worse gait-eval outcomes here, but this is only tested on the flat-walk stage over a short (2000-iteration) budget. Worth watching if this config is carried up the teacher stack, where velocity tracking matters more (obstacle approach, goal-course navigation).
