<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-09_0920_short_runs/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_0920_short_runs/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/short_runs`.

# Short-run gait fine-tuning ledger (Milestone 18, Task 1)

Autonomous weight/param sweep campaign, per the plan approved 2026-08-09 (revised same day: air-time threshold is tuned before air-time weight). Every run is a **flat-walk-only**, short resume (1000-2000 iters) from the v3 reference checkpoint, with overrides passed to `train.py` via Hydra CLI (`env.rewards.<term>.weight=X`, `env.rewards.<term>.params.<p>=Y`) -- no code changes for weight/param exploration. Each run is gait-eval'd (`--scenario flat_walk_forward`) and scored against the reference row below. See `/home/nickmagus/.claude/plans/the-newest-version-of-cheerful-codd.md` for the full plan.

**Discipline**: one *new* change per run; previously-KEPT overrides stack forward; REVERTED overrides are dropped. Reward *function* code changes are out of scope for this ledger -- those stop the campaign, get proposed to the user, and (if approved) get their own commit before implementation.

## Reference / anchors

| run id | override(s) | iters | tippy_tap | stride (pooled mag) | slip | completion | tripod | verdict |
|---|---|---|---|---|---|---|---|---|
| `v3_reference` | none (stride_length_v3 flat checkpoint, as-is) | -- | 19.09% | 0.178 m | 1.71% | 100% | 0.0 | REFERENCE -- every run below is scored against this row |
| `baseline_anchor` | none (original baseline flat checkpoint) | -- | 13.52% | 0.120 m | 2.96% | 100% | 0.0 | ANCHOR ONLY -- Task 1 §1f target: tippy_tap ≤ this |

Reference checkpoint: `sim_fine_tuning/2026-08-08_1701_stride_length_v3/logs/rsl_rl/crab_hex_flat_walk/2026-08-08_17-01-57/model_19999.pt`

**Campaign targets** (combined-winners run must clear all three): tippy_tap ≤ 13.5% AND stride ≥ 0.178m AND completion = 100%.

**Exploration order**: air-time threshold (this section) → air-time weight (on top of kept threshold) → stride-length weight/power → anti-shuffle/contact-shape → slip guardrail → motor-reversal-vs-smoothness 3-arm study → speed pressure (only if implicated).

## Runs

| run id | override(s) | iters | tippy_tap | stride (pooled mag) | slip | completion | tripod | tracking err (vel_xy) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| r01_airtime_thresh0.10 | `reward_feet_air_time_positive.params.threshold=0.10` (weight stays 0.4, from v3 ref) | 1000 | 18.99% | 0.1785 m | 1.67% | 100% | 0.0 | 0.183 | REVERTED -- essentially flat vs reference on every axis (tippy -0.1pt, stride +0.0005m); threshold alone at default weight 0.4 doesn't move the target metric within 1000 iters |
| r02_airtime_thresh0.15 | `reward_feet_air_time_positive.params.threshold=0.15` (weight stays 0.4, from v3 ref) | 1000 | 19.20% | 0.1746 m | 1.79% | 100% | 0.0 | -- | REVERTED -- slightly worse on every axis (tippy +0.11pt, stride -0.0034m, slip +0.08pt); threshold alone at default weight is not the lever, dropping to next knob |

<!-- Threshold sub-sweep concluded: neither 0.10 nor 0.15 moved anything meaningful at the
default weight (0.4). Threshold stays at its original 0.05 default going into the weight sweep
below. -->

| r03_airtime_w0.8 | `reward_feet_air_time_positive.weight=0.8` (threshold default 0.05, from v3 ref) | 1000 | 18.02% | 0.1807 m | 1.59% | 100% | 0.0 | 0.178 | KEPT -- improves tippy/stride/slip together, crab_failure 1.17% (healthy); deterministic re-run of the earlier probe, identical numbers confirmed |
| r04_airtime_w1.2 | `reward_feet_air_time_positive.weight=1.2` (threshold default 0.05, from v3 ref) | 1000 | 18.79% | 0.1759 m | 1.76% | **90% (1 fall)** | 0.0 | -- | REVERTED -- guardrail violation (completion dropped below 100%), and worse than w=0.8 on every axis (tippy/stride/slip all regress). Weight=0.8 stands as the winner. |

<!-- Air-time sub-campaign concluded. Winner: weight=0.8, threshold stays at default 0.05.
Carrying `reward_feet_air_time_positive.weight=0.8` forward into every run below. -->

| r05_stride_w1.0 | airtime.weight=0.8, `reward_stride_length.weight=1.0` (from v3 ref) | 1000 | 17.20% | 0.1806 m | 1.74% | **90% (1 fall)** | 0.0 | -- | REVERTED -- guardrail violation (completion below 100%), same failure pattern as airtime w=1.2; BUT tippy improved to best-of-campaign (17.20%, vs r03's 18.02%) -- promising direction, retrying with a smaller step (0.75) to see if it retains stability |
| r06_stride_w0.75 | airtime.weight=0.8, `reward_stride_length.weight=0.75` (from v3 ref) | 1000 | 19.33% | 0.1813 m | 1.62% | 100% | 0.0 | -- | REVERTED -- completion recovers to 100% and stride is best-of-campaign, but tippy_tap *regresses* past both r03 (18.02%) and even the raw reference (19.09%). Non-monotonic vs w=1.0 (improved tippy, broke completion) -- neither weight value beats r03 cleanly on the primary tippy target. Stride weight stays at v3 default (0.5); moving to the next knob. |

<!-- Stride-length weight sweep concluded without a clean win in {0.75, 1.0} -- both showed real
trade-offs against r03. Stride weight stays at its v3 default 0.5 going forward (not stacked).
Carrying only `reward_feet_air_time_positive.weight=0.8` forward into runs below. -->

| r07_footidle_w-0.25 | airtime.weight=0.8, `penalty_foot_idle_when_forward.weight=-0.25` (from v3 ref) | 1000 | 18.14% | 0.1776 m | 1.58% | 100% | 0.0 | 0.175 | REVERTED -- essentially flat vs r03 (tippy +0.12pt, stride -0.003m); doubling this penalty doesn't move anything beyond the air-time win already banked |

| r08_excessfeet_w-0.4 | airtime.weight=0.8, `penalty_excess_feet_contact_forward.weight=-0.4` (from v3 ref) | 1000 | 17.92% | 0.1797 m | 1.82% | **90% (1 fall)** | 0.0 | 0.175 | REVERTED -- guardrail violation (completion below 100%, 3rd occurrence in this campaign); marginal tippy gain not worth the stability cost |

<!-- Fall pattern note: 3 of 8 evaluated checkpoints so far (airtime w=1.2, stride w=1.0, excess-feet
w=-0.4) have hit exactly 1 fall / 10 episodes on the gait-eval schedule. Each is a distinct
override, so this isn't one bad knob -- more likely the 10-episode low/mid/high schedule sits
close to a stability edge that any push away from the tuned v3 defaults can occasionally trip.
Treating every fall as a real guardrail violation per established discipline, but noting this as
context for interpreting the campaign's overall pattern. -->

| r09_stancesupport_w0.1 | airtime.weight=0.8, `reward_stance_support_feet_when_forward.weight=0.1` (from v3 ref) | 1000 | 18.08% | 0.1773 m | 1.81% | 100% | 0.0 | 0.172 | REVERTED -- essentially flat vs r03 on tippy, slightly worse on stride and slip; no meaningful gain |

<!-- Anti-shuffle/contact-shape sub-campaign concluded: none of the three knobs
(penalty_foot_idle_when_forward, penalty_excess_feet_contact_forward,
reward_stance_support_feet_when_forward) beat r03 cleanly -- two hit the completion guardrail,
one was flat. None carried forward. Only `reward_feet_air_time_positive.weight=0.8` (r03) stays
in the running config. -->

<!-- Slip guardrail (feet_slide=-0.1) skipped per its own trigger condition in the plan ("only if
slip degrades anywhere above the KEPT set") -- r03 (the only KEPT run) shows slip=1.59%, BETTER
than the v3 reference's 1.71%. No degradation to guard against. -->

## Motor-reversal vs smoothness 3-arm study

| run id | override(s) | iters | tippy_tap | stride (pooled mag) | slip | completion | tripod | tracking err (vel_xy) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| r10_reversal_w-0.15 | airtime.weight=0.8, `penalty_motor_direction_reversal.weight=-0.15` (from v3 ref, weaker than the -0.3 default) | 1000 | 19.49% | 0.1791 m | 1.48% | **90% (1 fall)** | 0.0 | 0.180 | REVERTED -- guardrail violation, AND tippy got *worse* than r03 (19.49% vs 18.02%) and even worse than the raw reference (19.09%) -- weakening the reversal penalty did not help gait quality here |

| r11_reversal_w-0.6 | airtime.weight=0.8, `penalty_motor_direction_reversal.weight=-0.6` (from v3 ref, stronger than the -0.3 default) | 1000 | 19.13% | 0.1829 m | 1.71% | 100% | 0.0 | 0.179 | REVERTED -- completion recovers to 100% and stride is best-of-campaign, but tippy reverts to roughly reference level (19.13% vs r03's 18.02%), cancelling out the air-time win. Doesn't beat r03 on the primary target. |

<!-- Arm A (reversal-only) concluded: neither -0.15 nor -0.6 beats r03. Moving to Arm B
(smoothness-only): penalty_motor_direction_reversal turned OFF (0.0), reward_action_rate and
reward_delta_torques (pre-added at 0.0 in setup) turned on instead. -->

| r12_smooth_mild | airtime.weight=0.8, `penalty_motor_direction_reversal.weight=0.0`, `reward_action_rate.weight=-0.1`, `reward_delta_torques.weight=-1e-7` | 1000 | 18.31% | 0.1780 m | 1.53% | 100% | 0.0 | 0.180 | ROUGH TIE with r03 (tippy +0.29pt, stride -0.003m, slip -0.06pt) -- not a clear win, but notable: smoothness terms alone (reversal penalty OFF) reach nearly the same gait quality as the reversal-penalty mechanism, via a completely different route. Trying the stronger combo before deciding. |

| r13_smooth_strong | airtime.weight=0.8, `penalty_motor_direction_reversal.weight=0.0`, `reward_action_rate.weight=-0.3`, `reward_delta_torques.weight=-1e-6` | 1000 | **16.64%** | **0.1835 m** | 1.61% | 100% | 0.0 | 0.212 (elevated, vs ~0.17-0.18 typical) | **KEPT -- best-of-campaign on both tippy AND stride simultaneously**, completion intact, slip flat vs r03. Tracking error is elevated (policy trades some velocity-tracking precision for smoother actions) but this hasn't shown up as a problem in completion/crab_failure. |

**Arm A/B/C conclusion**: Arm A (reversal-only, both directions) produced no winner -- REVERTED. Arm B's strong combo (r13) is the best result of the entire campaign. Since Arm A contributed nothing, there is no meaningful "Arm C" combination to test separately -- r13 stands as the study's winner. Carrying `penalty_motor_direction_reversal.weight=0.0`, `reward_action_rate.weight=-0.3`, `reward_delta_torques.weight=-1e-6` forward alongside `reward_feet_air_time_positive.weight=0.8` as the campaign's combined-winners config.

## Combined-winners config (going into final validation)

```
env.rewards.reward_feet_air_time_positive.weight=0.8
env.rewards.penalty_motor_direction_reversal.weight=0.0
env.rewards.reward_action_rate.weight=-0.3
env.rewards.reward_delta_torques.weight=-1e-6
```

vs v3 reference: tippy 19.09% -> 16.64% (-2.45pt), stride 0.178m -> 0.1835m (+0.006m), slip 1.71% -> 1.61%, completion 100% -> 100%. Campaign target (tippy <=13.5%) not yet cleared, but this is the closest approach in the series and a real, multi-axis improvement. Re-validating at 2000 iterations next.

## Final validation (2000 iterations)

| run id | override(s) | iters | tippy_tap | stride (pooled mag) | slip | completion | tripod | tracking err (vel_xy) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| r14_combined_final | airtime.weight=0.8, `penalty_motor_direction_reversal.weight=0.0`, `reward_action_rate.weight=-0.3`, `reward_delta_torques.weight=-1e-6` (from v3 ref) | 2000 | **13.39%** | **0.1975 m** | 1.66% | 100% | 0.0 | 0.206 | **ALL THREE CAMPAIGN TARGETS CLEARED**: tippy <=13.5% (13.39%), stride >=0.178m (0.1975m -- best of the entire comparison series, beats even motor_reversal_on's 0.180m), completion=100%. crab_failure=2.23% (healthy). |

**Campaign result: SUCCESS.** At 1000 iterations (r13) the combined config was already the best-of-campaign on tippy+stride simultaneously; extending to 2000 iterations let it converge further and cleared every stated target, including the Task 1 §1f tippy-tap target (<= baseline's 13.5%) that motivated this whole reward-shaping effort.

## Series-wide comparison (flat-walk stage, pooled mean stride magnitude / tippy_tap_fraction)

| run | stride | tippy_tap | completion |
|---|---|---|---|
| baseline | 0.120 m | 13.52% | 100% |
| motor_reversal_on | 0.180 m | 31.0% | 100% |
| stride_length_on (v2) | 0.183 m | 36.6% | 80% |
| stride_length_v3 (reference) | 0.178 m | 19.09% | 100% |
| stride_length_v4 (reverted) | 0.170 m | 39.2% | 100% |
| **short_runs combined-winners (r14)** | **0.1975 m** | **13.39%** | **100%** |

r14 is the first run in the entire Milestone 18 Task 1 series to clear baseline's tippy-tap level while *also* holding the best stride length of the series -- previously these two metrics traded off against each other in every comparison run.
