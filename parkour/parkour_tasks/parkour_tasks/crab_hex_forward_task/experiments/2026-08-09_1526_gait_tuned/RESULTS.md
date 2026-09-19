<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-09_1526_gait_tuned/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-09_1526_gait_tuned/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/gait_tuned`.

# gait_tuned: Teacher-stack carry-up study (T0/T1/T2)

Milestone 18 Task 1 follow-on to `2026-08-09_0920_short_runs/`. That campaign baked a winning
flat-walk config into `CrabHexFlatWalkRewardsCfg` (`reward_feet_air_time_positive=0.8`,
`penalty_motor_direction_reversal=0.0`, `reward_action_rate=-0.3`, `reward_delta_torques=-1e-6`).
This study asks whether the smoothness-heavy, reversal-penalty-off combo should carry up into the
teacher stack (bridge/2b1/2b2), where tracking responsiveness and camshaft hardware wear both
matter more than on flat. Plan: `~/.claude/plans/the-newest-version-of-cheerful-codd.md`.

Per-arm protocol, gates, and the decision rule are defined in that plan. Reversal counts and
pooled stride below are read directly from the harness's own `metrics/episode_*.json` output
(`actions.joint_vel_sign_reversals.camshaft`, `stride.pooled_magnitude`) — n_td-weighted mean
across all episodes in the run, summed reversal count across all episodes.

v3-series stage anchors (for context, not a hard target for this study): bridge crab_failure
3.91%, 2b1 1.56%, 2b2 batch-range 6.8-14%, forward_progress peak 0.1496; 2b2 gait-eval completion
80%, tippy 21.0%, stride 0.162m, slip 3.0%.

## Phase 0: from-scratch flat-walk retrain (baked config, 20000 iters)

Checkpoint: `logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14/model_19999.pt`

| metric | value |
|---|---|
| crab_failure (train, final window) | 0.0000 |
| tippy_tap_fraction (gait-eval, flat_walk_forward) | 0.0797 |
| slip_ratio | 0.0231 |
| tripod_score | 0.4011 |
| schedule_completion_rate | 1.00 |
| stride (pooled, n_td-weighted) | 0.163 m |

**Major finding**: tripod_score jumped from this series' established 0.0-0.025 range to 0.401
median, and tippy_tap hit a series-best 7.97%, purely from training the baked config from scratch
rather than fine-tuning on top of a v3-converged policy — no reward-code change. Directly relevant
to Task 1 §1d (tripod score improving "via an explicit contact-schedule term or demonstrated as
emergent"). Trade-off: stride length (0.163m) is below r14's fine-tuned 0.1975m and the ≥0.178m
target — from-scratch training reaches a materially different point on the gait-quality/stride
trade-off than fine-tuning did.

## Arm T0: control (teacher weights unchanged — reversal=-0.3, action_rate=-0.1, delta_torques=-1e-7)

Seeded from the Phase 0 checkpoint above. No Hydra overrides.

### Training gates (windowed, last 10 iteration-blocks)

| stage | checkpoint (iter) | crab_failure | current_goal_idx | forward_progress_along_command | error_vel_xy |
|---|---|---|---|---|---|
| bridge (100 iters) | model_20098.pt | 7.01% | 0.648 | 0.1705 | 0.753 |
| 2b1 (100 iters) | model_20197.pt | 7.53% | 0.971 | 0.1938 | 0.672 |
| 2b2 batch1 (500 iters) | model_20696.pt | 37.68% | 0.837 | 0.0554 | 0.584 |

2b2 batch1's 37.68% crab_failure is consistent with the series' established "rocky first 2b2
batch" pattern (every prior series run has shown an elevated first-batch crab_failure before
recovering in batch 2+) — not itself an abort trigger per the plan's abort rule (bridge>25% or
2b1-not-recovering).

### Gait-eval (`teacher_2b2_forward`, on model_20696.pt, 10 episodes)

| metric | value | v3 2b2 anchor |
|---|---|---|
| tippy_tap_fraction (median) | 0.2714 | 0.210 |
| slip_ratio (median) | 0.0440 | 0.030 |
| tripod_score (median) | 0.3254 | n/a |
| schedule_completion_rate | 0.50 | 0.80 |
| stride (pooled, n_td-weighted) | 0.169 m | 0.162 m |
| camshaft reversal count (sum / 10 episodes) | 7000 (700/ep) | — |

T0 (unchanged teacher weights, but seeded from the stronger from-scratch flat policy) is
running **worse than the v3-series 2b2 anchors** on completion (50% vs 80%) and tippy (27.1% vs
21.0%), though stride is slightly better (0.169m vs 0.162m). This is one batch, at the
established rocky point in the curriculum — not yet a verdict on T0 itself. The comparison that
actually decides this study is T0 vs T1 vs T2, not T0 vs the older v3 anchors.

## Arm T1: full mirror (flat-walk winner mirrored onto the teacher stack)

`penalty_motor_direction_reversal.weight=0.0`, `reward_action_rate.weight=-0.3`,
`reward_delta_torques.weight=-1e-6` — Hydra overrides at every stage, seeded from the same
Phase 0 checkpoint.

### Training gates (windowed, last 10 iteration-blocks)

| stage | checkpoint (iter) | crab_failure | current_goal_idx | forward_progress_along_command | error_vel_xy |
|---|---|---|---|---|---|
| bridge (100 iters) | model_20098.pt | 8.76% | 0.524 | 0.1541 | 0.665 |
| 2b1 (100 iters) | model_20197.pt | 7.83% | — | — | 0.544 |
| 2b2 batch1 (500 iters) | model_20696.pt | 31.25% | 0.964 | 0.0641 | 0.655 |

(2b1's `current_goal_idx`/`forward_progress` weren't pulled individually — bridge/2b2 bracket it
and neither shows a discontinuity.) All three stages track T0 closely on `crab_failure`
(bridge/2b1 within ~1.5 points; 2b2 batch1 is **better** than T0's 37.68%, at 31.25%) — the
mirrored smoothness/reversal overrides did not destabilize training relative to control.

### Gait-eval (`teacher_2b2_forward`, on model_20696.pt, 10 episodes)

| metric | T1 | T0 | v3 2b2 anchor |
|---|---|---|---|
| tippy_tap_fraction (median) | 0.2713 | 0.2714 | 0.210 |
| slip_ratio (median) | 0.0515 | 0.0440 | 0.030 |
| tripod_score (median) | 0.3208 | 0.3254 | n/a |
| schedule_completion_rate | **0.70** | 0.50 | 0.80 |
| stride (pooled, n_td-weighted) | 0.160 m | 0.169 m | 0.162 m |
| camshaft reversal count (sum / 10 episodes) | 7333 (733/ep) | 7000 (700/ep) | — |

T1 beats T0 on completion (70% vs 50%, +20 points — the biggest gap of any metric) and matches
it on tippy/tripod almost exactly. Slip is somewhat worse (+1.1 points) and stride is slightly
lower. Reversal count is +4.8% over T0 (733 vs 700/episode) — nowhere near the plan's "present
the trade-off" threshold (>2x). Net: T1 looks like a mild-to-moderate win over T0, driven mostly
by completion rate, at a negligible hardware-wear cost.

## Arm T2: hybrid (reversal penalty kept + smoothness added)

Reversal kept at `-0.3` (unchanged) + `reward_action_rate.weight=-0.3`, `reward_delta_torques.weight=-1e-6`
— Hydra overrides at every stage, seeded from the same Phase 0 checkpoint.

### Training gates (windowed, last 10 iteration-blocks)

| stage | checkpoint (iter) | crab_failure | current_goal_idx | forward_progress_along_command | error_vel_xy |
|---|---|---|---|---|---|
| bridge (100 iters) | model_20098.pt | 8.61% | 0.529 | 0.1650 | 0.708 |
| 2b1 (100 iters) | model_20197.pt | 9.92% | 0.890 | 0.1820 | 0.668 |
| 2b2 batch1 (500 iters) | model_20696.pt | 34.89% | 0.904 | 0.0446 | 0.499 |

### Gait-eval (`teacher_2b2_forward`, on model_20696.pt, 10 episodes)

| metric | T0 | T1 | T2 | v3 2b2 anchor |
|---|---|---|---|---|
| tippy_tap_fraction (median) | 0.2714 | 0.2713 | **0.2135** | 0.210 |
| slip_ratio (median) | **0.0440** | 0.0515 | 0.0458 | 0.030 |
| tripod_score (median) | **0.3254** | 0.3208 | 0.2718 | n/a |
| schedule_completion_rate | 0.50 | **0.70** | 0.60 | 0.80 |
| stride (pooled, n_td-weighted) | 0.169 m | 0.160 m | 0.163 m | 0.162 m |
| camshaft reversal count (sum / 10 episodes) | 7000 (700/ep) | 7333 (733/ep) | **6631 (663/ep)** |  — |
| 2b2 batch1 crab_failure (train) | 37.68% | **31.25%** | 34.89% | 6.8-14% |
| 2b2 batch1 error_vel_xy (train) | 0.584 | 0.655 | **0.499** | — |

T2 posts the best tippy_tap of the study (21.35%, essentially matching the v3 anchor of 21.0% —
the other two arms are both around 27%) and the lowest camshaft reversal count (663/ep, beating
even T0's unmodified-reversal-penalty control). It also has the lowest training-time
`error_vel_xy` (0.499) of all three arms at 2b2 batch1, directly answering this study's first
open thread (elevated tracking error) in T2's favor. Its weak points are tripod_score (lowest of
the three, 0.272) and completion (60%, between T0's 50% and T1's 70%).

## Decision: T0 vs T1 vs T2

| axis | T0 (control) | T1 (full mirror) | T2 (hybrid) | best |
|---|---|---|---|---|
| tippy_tap (gait-eval) | 0.2714 | 0.2713 | 0.2135 | **T2** |
| slip_ratio | 0.0440 | 0.0515 | 0.0458 | T0 |
| tripod_score | 0.3254 | 0.3208 | 0.2718 | T0 |
| completion | 0.50 | 0.70 | 0.60 | **T1** |
| stride vs v3 anchor (0.162m) | 0.169 | 0.160 | 0.163 | T2 (closest) |
| camshaft reversals/ep | 700 | 733 | 663 | **T2** |
| 2b2batch1 crab_failure (train) | 37.68% | 31.25% | 34.89% | **T1** |
| 2b2batch1 error_vel_xy (train) | 0.584 | 0.655 | 0.499 | **T2** |

No arm sweeps every axis — this is a genuine three-way trade-off, not a clean win, so per the
plan's decision rule this gets presented rather than auto-decided:

- **T2 (hybrid)** best answers both open threads this study was designed around: it has the
  *lowest* tracking error (`error_vel_xy` 0.499, better than even T0) and the *lowest* camshaft
  reversal count (663/ep, better than T0's own 700/ep) — i.e. keeping the reversal penalty **and**
  adding smoothness compounds rather than conflicts. It also has by far the best tippy_tap
  (21.35%, matching the v3 anchor almost exactly, a ~6-point improvement over T0/T1's ~27%).
  Its cost: worst tripod_score (0.272 vs T0's 0.325) and middling completion (60%, below T1's
  70%).
- **T1 (full mirror)** — the direct carry-up of the flat-walk winner — has the best completion
  (70%) and best training-time `crab_failure` at 2b2 batch1 (31.25%, clearly the most stable of
  the three during training), but is worst on reversal count and tracking error, i.e. it re-opens
  both concerns this study set out to check.
- **T0 (control)** wins slip and tripod narrowly but is worst on completion and tied-worst on
  tippy — seeded-from-a-better-flat-policy alone does not carry the flat-walk gains through the
  teacher stack on its own.

All differences are one 2b2 batch at the curriculum's established rocky point (batch 1 of what
was a 4-batch process in the v3 series) — not a converged comparison. Recommendation: **T2 looks
like the strongest candidate** given it resolves the two concerns that motivated this study
without regressing the primary gait metric, but the completion/tripod trade-off against T1 is
real and worth a second opinion before committing further compute to extending one arm.

## Winner extension: T2 2b2 batches 2-4

User directed extending T2 (per the plan's "Winner extension" step). Ran three more 500-iter
2b2 batches from T2's batch1 checkpoint, gate-checking each per the crab-hex-train SKILL.md
sweet-spot rule (`crab_failure<20%`, `obstacle_clearance>0.15`, `episode_length>=750-800`,
`forward_progress>0.15-0.20`, `current_goal_idx>0.7-0.9`).

### Batch-by-batch training gates (windowed, last 10 iteration-blocks)

| batch | checkpoint (iter) | crab_failure | obstacle_clearance | forward_progress | current_goal_idx | episode_length |
|---|---|---|---|---|---|---|
| 1 | model_20696.pt | 34.89% | 0.033 | 0.045 | 0.904 | 492 |
| 2 | model_21195.pt | 35.00% | 0.038 | 0.055 | 1.040 | 533 |
| 3 | model_21694.pt | 31.82% | 0.056 | 0.070 | 1.222 | 569 |
| 4 | model_22193.pt | 33.64% | 0.060 | 0.071 | 1.088 | 515 |

**The stage plateaued without reaching gate convergence.** Across all 4 batches (~1997 total 2b2
iterations, iter 20197→22193), `crab_failure` stayed in a 32-35% band (gate needs <20%),
`obstacle_clearance` never exceeded ~0.06 on a windowed basis (gate needs >0.15), and
`episode_length` stayed 490-570 (gate needs ≥750). Only `current_goal_idx` cleared its gate
throughout. Batch4 did not improve on batch3 (crab_failure 33.64% vs 31.82%, episode_length 515
vs 569) — judged as plateau noise rather than a clear trend, so batch3's checkpoint
(model_21694.pt, the best training-time snapshot of the four) was carried forward for final
gait-eval, per the "don't just use the last batch" rule.

### Final gait-eval on the extension's best checkpoint (model_21694.pt, batch3)

| metric | T2 batch1 (used for the decision above) | T2 extension (batch3) | v3 2b2 anchor |
|---|---|---|---|
| tippy_tap_fraction (median) | 0.2135 | **0.3464** (worse) | 0.210 |
| slip_ratio (median) | 0.0458 | 0.0560 (worse) | 0.030 |
| tripod_score (median) | 0.2718 | **0.1845** (worse) | n/a |
| schedule_completion_rate | 0.60 | 0.60 (unchanged) | 0.80 |
| stride (pooled, n_td-weighted) | 0.163 m | 0.164 m (unchanged) | 0.162 m |
| camshaft reversal count (sum/10 episodes) | 663/ep | 669/ep (unchanged) | — |

**The extension made gait quality worse, not better.** Despite batch3 showing marginally better
*training-time* `crab_failure` than batch1 (31.82% vs 34.89%), its held-out gait-eval is clearly
worse on both headline metrics: tippy_tap nearly doubled (21.35%→34.64%) and tripod_score dropped
by a third (0.272→0.185). Stride, completion, and reversal count are essentially unchanged. This
is a real divergence between the windowed training metric and held-out gait quality, not just
eval noise on two of the biggest deltas in this entire study — a caution against reading
training-time `crab_failure` alone as a proxy for gait quality at this stage.

**Consequence for the recommendation**: T2's *original batch1 checkpoint* remains the best T2
result found, not the extension. The extension attempt is documented for the record but the bake
proposal below is based on T2 batch1's numbers, unchanged from the Decision section above.
