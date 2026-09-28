<!-- paths-note -->
> **Paths note (2026-09-09):** this campaign moved from `sim_fine_tuning/2026-08-07_2303_motor_reversal_on/` to `parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-07_2303_motor_reversal_on/`. Absolute paths below (`/home/.../sim_fine_tuning/...`, `parkour/logs/rsl_rl/...`) name raw artifacts that stay on disk untracked; the tracked checkpoint of record is `head/` (see `bundle.yaml`) and the eval summaries are in place / under `evals/`. Older records may call this directory `sim_fine_tuning/motor_reversal_on`.

# Changelog: `sim_fine_tuning/2026-08-07_2303_motor_reversal_on`

**Milestone 18, Task 1 item**: adjacent to §2.5 "Energy/smoothness" (reversal suppression), though
via a pre-existing term rather than the doc's suggested `reward_action_rate`/`reward_delta_torques`
sweep. `penalty_motor_direction_reversal` already existed from the earlier cam-mechanism migration
and was simply zeroed for `baseline`'s isolation; this run restores it to its original weight.

**Predecessor**: `sim_fine_tuning/2026-08-07_1441_baseline`.

## Change

`parkour_mdp_cfg.py`, both `CrabHexRewardsCfg` (teacher) and `CrabHexFlatWalkRewardsCfg`:

```
penalty_motor_direction_reversal.weight:  0.0  ->  -0.3
```

Penalizes the CamShaft motor reversing rotational direction; encourages sustained one-directional
spin so the cam geometry (not motor reversal) produces the leg's back-and-forth yaw motion. Full
run: flat (20000 iter, from scratch) -> bridge -> 2b1 -> 2b2 (4 batches, 2000 iter, matching
baseline's budget).

## Metric delta vs baseline

**Direct mechanism check** (camshaft direction-reversal count, 975-step episode):

| | baseline | this run | delta |
|---|---|---|---|
| flat | 1070 | 561 | **-48%** |
| 2b2 | 1113 | 385 | **-65%** |

**Training**:

| | baseline | this run |
|---|---|---|
| flat crab_failure | 0.78% | 0.78% (unchanged) |
| flat mean_reward | 46.18 | 40.85 (lower -- expected, penalty subtracts from raw reward) |
| bridge crab_failure | 6.3% | 27.6% (rockier transition) |
| 2b1 crab_failure | 6.3% | 14.3% (recovering) |
| 2b2 crab_failure (final, `model_21400.pt`) | 18.75% | **7.8%** |
| 2b2 gates cleared | 4/5 | 4/5 (same 5, but stronger margins) |

**Gait-eval** (Task 0 harness):

| | flat: baseline / this run | 2b2: baseline / this run |
|---|---|---|
| schedule_completion_rate | 100% / 100% | 90% (1 fall) / **100% (0 falls)** |
| tripod_score (median) | 0.0 / 0.0 | 0.0035 / 0.0 |
| slip_ratio | 3.0% / 2.5% | 7.3% / 5.9% |
| stride length (pooled mean magnitude) | 0.120 m / **0.180 m (+50%)** | 0.139 m / **0.171 m (+23%)** |
| touchdowns / 10-episode set | 5406 / 3998 (-26%) | 4291 / 3315 (-23%) |
| tippy_tap_fraction | 13.5% / **31.0%** | 14.7% / **22.2%** |

Notably, this term alone -- without any explicit stride-length reward -- produced a larger
measured stride-length increase (+50% flat, +23% 2b2) and fewer, longer touchdowns. Fewer
motor reversals evidently correlates with committing to longer strides rather than chattering.

**Task 1 §1d (tripod score)**: near-zero at every stage and speed hold, for both this run and
baseline -- this is not a regression. The policy uses a front/middle/rear duty-factor split (middle
legs carry ~0.55-0.62 duty, corner legs lower) rather than a left/right alternating tripod gait, a
finding established from `baseline`'s own duty-factor breakdown and confirmed identical across
every run in this series. §1d's threshold, if pursued, would require an explicit contact-schedule
term (§2 item 3) -- it is not emerging on its own from any of these reward changes.

**The tippy-tap increase is real and not fully understood.** Per-leg breakdown of the flat-walk
air-time distribution shows it is not uniform: front/rear-left legs (FL, RL) show zero short-blip
air-time events and long, clean swing durations (median 0.35s); the increase is concentrated on
FR/ML/MR/RR, with MR the worst (57% of its air-time events are single-physics-step, ~20ms). This
does not appear to be sensor noise -- see `stride_length_on`'s changelog for a direct measurement
showing these are genuine (if extremely fast) foot movements, not threshold artifacts.

## Verdict

**Kept.** Directly achieves its stated goal (reversal count down 48-65%, a real hardware-longevity
win) and produced a *more* stable, *longer-striding* 2b2 policy at the same training budget as the
baseline (lower crab_failure, zero eval falls, better obstacle clearance and episode length). The
tippy-tap regression is a genuine cost, not an unambiguous win overall, but the primary target
metric (reversal count) moved decisively in the intended direction and the secondary training
metrics improved rather than regressed, so this passes Task 1's "revert changes that do not move
their target metric" bar. Flagged for follow-up, not reverted.
