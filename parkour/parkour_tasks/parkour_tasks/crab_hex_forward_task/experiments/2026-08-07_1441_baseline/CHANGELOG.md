<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-07_1441_baseline/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-07_1441_baseline/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/baseline`.

# Changelog: `sim_fine_tuning/2026-08-07_1441_baseline`

**Milestone 18 context**: this run is the Task 1 (Reward Shaping) starting point, not itself a
reward-shaping change. It carries forward the CAD-derived Whitworth cam-mapping correction
(`crab_hex_cam_mapping.py` — `K` measured directly from `KrabV3-Legs.svg`, not calibrated from an
assumed swing limit) from earlier work in this session, which is a mechanism/kinematics fix, not a
reward change, and predates Task 1.

**Note on Task 1 §1's "resumed from 2b2 `6300`" instruction**: `model_6300.pt` predates the
cam-mechanism migration entirely — it drove `Body_Hip` directly, whereas the current asset drives
`Body_CamShaft` with `Body_Hip` passive and kinematically slaved to it. The joints mean different
things now, so resuming from `6300` would not be meaningful. This run's own 2b2 checkpoint
(`model_21500.pt`, see below) is the valid Task 1 starting point instead, and everything in this
changelog series resumes from it (or its descendants) rather than from `6300`.

## Change

None — this is the reference run. `penalty_motor_direction_reversal` weight is deliberately
zeroed (`0.0`, normally `-0.3`) specifically to isolate the CAD-mapping correction's own effect on
gait from that term's effect, at the user's request. All later runs in this series restore it.

## Stages / checkpoints

| stage | checkpoint | crab_failure | notes |
|---|---|---|---|
| flat (20000 iter, from scratch) | `logs/rsl_rl/crab_hex_flat_walk/2026-08-07_15-31-12/model_19999.pt` | 0.78% | mean_reward=46.18 |
| bridge (100 iter) | `.../crab_hex_teacher/2026-08-07_21-38-48/model_20098.pt` | 6.33% | fixed stage |
| 2b1 (100 iter) | `.../crab_hex_teacher/2026-08-07_21-47-42/model_20197.pt` | 6.30% | fixed stage |
| 2b2 (4 batches, 2000 iter) | `.../crab_hex_teacher/2026-08-07_22-06-09/model_21500.pt` | 18.75% | 4/5 README §4.2b gates cleared (obstacle_clearance=0.211, episode_length=812, current_goal_idx=1.61 pass; forward_progress=0.121 misses, target >0.15-0.20) |

## Gait-eval metrics (Task 0 harness)

| | flat | 2b2 |
|---|---|---|
| schedule_completion_rate | 100% | 90% (1 fall / 10) |
| tripod_score (median) | 0.0 | 0.0035 |
| slip_ratio | 3.0% | 7.3% |
| tippy_tap_fraction | 13.5% | 14.7% |
| stride length (pooled mean magnitude) | 0.120 m | 0.139 m |
| touchdowns / 10-episode set | 5406 | 4291 |

Both flat and 2b2 checkpoints reproduce the same gait pattern: middle legs (ML/MR) carry
substantially higher duty factor (~0.55-0.62) than corner legs, a real, mechanism-consistent
strategy rather than an artifact (confirmed identical across every run in this series). Tripod
score near-zero (Task 1 §1d) is expected and not itself a problem here — see Task 0's own finding
that the policy does not use a left/right tripod gait at all, only a front/middle/rear duty split.
This series never pursues §1d's explicit contact-schedule term (§2 item 3), since air-time/stride
changes were the priority per Task 1's stated ordering and none of them moved tripod score either
direction across any run.

## Verdict

**Reference.** This run's flat and 2b2 checkpoints are the baseline every later change in this
series is measured against.
